"""
the (beta, Delta) tables, read once into host memory; for the GPU in
page-locked memory, copied to the device in groups.

redshift store   bd_*.npz, one file per signal power node s = u y^2, members
                 T (tau, phi+, beta, Delta) float32, beta, delta,
                 tau_idx, alpha (amplitude axis), s_node; the distance
                 is a parameter, its prior is applied by the analysis

The tables hold lnS at the points of a grid and are read by linear
interpolation.  On a grid of spacing h a function of curvature T'' is then
overestimated by h^2 T'' f (1 - f) / 2 at the fraction f of a cell, h^2 T'' / 12
on average, and lnS is convex in beta and Delta (a log-sum-exp of the
orders): under noise the likelihood ratio read from the tables has mean
1 + 7e-4 s instead of 1.  unbias() subtracts that average, the second
differences of the table over 12, and leaves an error of zero mean over a cell.
"""

import glob
import os
import struct
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

NPHI = 96                 # phase bins of the full circle; the tables hold NPHI/2 + 1


def read_npz_member(path, name):
    """one uncompressed .npy member of an npz, read sequentially"""
    info = zipfile.ZipFile(path).getinfo(name)
    if info.compress_type != 0:
        raise ValueError("member %s of %s is compressed" % (name, path))
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        nlen, elen = struct.unpack("<HH", fh.read(30)[26:30])
        fh.seek(info.header_offset + 30 + nlen + elen)
        ver = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(fh)
        arr = np.fromfile(fh, dtype=dtype, count=int(np.prod(shape)))
    return arr.reshape(shape, order="F" if fortran else "C")


def map_npz_member(path, name):
    """read-only memory map of one uncompressed .npy member of an npz (nothing is read until used)"""
    info = zipfile.ZipFile(path).getinfo(name)
    if info.compress_type != 0:
        raise ValueError("member %s of %s is compressed" % (name, path))
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        nlen, elen = struct.unpack("<HH", fh.read(30)[26:30])
        fh.seek(info.header_offset + 30 + nlen + elen)
        ver = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(fh)
        offset = fh.tell()
    return np.asarray(np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C"))


def unbias(T):
    """T minus its second differences along phi, beta and Delta over 12 (float32, a new array): the table whose
    linear interpolation has no mean error.  phi runs over [0, pi] and the table is even about both ends; along beta
    and Delta the end points are kept."""
    out = np.empty(T.shape, np.float32)
    for i in range(T.shape[0]):
        t = np.asarray(T[i], np.float64)
        c = np.zeros_like(t)
        c[1:-1] += t[2:] - 2.0 * t[1:-1] + t[:-2]
        c[0] += 2.0 * (t[1] - t[0])
        c[-1] += 2.0 * (t[-2] - t[-1])
        c[:, 1:-1] += t[:, 2:] - 2.0 * t[:, 1:-1] + t[:, :-2]
        c[:, :, 1:-1] += t[:, :, 2:] - 2.0 * t[:, :, 1:-1] + t[:, :, :-2]
        out[i] = t - c / 12.0
    return out


def _read_table(path, pinned, unbiased=False):
    """the T member of one store file, in page-locked memory when pinned (pageable if that fails)"""
    T = read_npz_member(path, "T.npy")
    if unbiased:
        T = unbias(T)
    if not pinned:
        return T
    import cupy as cp
    try:
        buf = cp.cuda.alloc_pinned_memory(T.nbytes)
    except Exception:
        return T
    P = np.frombuffer(buf, T.dtype, T.size).reshape(T.shape)
    P[...] = T
    return P


def _read_tables(files, pinned, readers, mmap=False, unbiased=False):
    if mmap:
        if unbiased:
            raise ValueError("unbiased tables are computed in memory: use mmap=False")
        return [map_npz_member(f, "T.npy") for f in files]
    with ThreadPoolExecutor(max_workers=readers) as ex:
        return list(ex.map(lambda f: _read_table(f, pinned, unbiased), files))


@dataclass
class RedshiftStore:
    """signal power nodes s (sorted), their amplitude axis limits, grids, host tables and the truncation order"""
    s: np.ndarray
    alpha_lo: np.ndarray
    alpha_hi: np.ndarray
    beta: list
    delta: list
    tau_idx: np.ndarray
    tables: list
    n_max: int


def load_redshift_store(store_dir, pinned=False, readers=6, mmap=False, unbiased=False):
    """every node of the redshift store in host memory (about 0.75 GB per node);
    pinned=True puts them in page-locked memory for fast copies to the GPU, mmap=True
    maps them from disk instead of reading them (CPU), unbiased=True removes the mean
    error of the linear interpolation (unbias)"""
    files = glob.glob(os.path.join(store_dir, "bd_*.npz"))
    if not files:
        raise FileNotFoundError("no bd_*.npz in %s" % store_dir)
    head = []
    for f in files:
        with np.load(f) as h:
            head.append((float(h["s_node"]), float(h["alpha"][0]), float(h["alpha"][-1]),
                         h["beta"], h["delta"], h["tau_idx"], f))
    head.sort(key=lambda r: r[0])
    s, alo, ahi, beta, delta, tau, files = zip(*head)
    return RedshiftStore(s=np.array(s), alpha_lo=np.array(alo), alpha_hi=np.array(ahi),
                         beta=list(beta), delta=list(delta), tau_idx=tau[0],
                         tables=_read_tables(files, pinned, readers, mmap, unbiased),
                         n_max=int(os.path.basename(files[0]).split("_n")[1].split("_")[0]))


def upload(tables, tau_idx, beta, delta, stream=None):
    """host tables -> one contiguous device stack, in the layout kernel.gather reads.
    with a stream the copies from page-locked tables run asynchronously on it"""
    import cupy as cp
    ntau, nphif, nb, nd = tables[0].shape
    tstride = int(tables[0].size)
    T = cp.empty(tstride * len(tables), np.float32)
    for i, Ti in enumerate(tables):
        T[i * tstride:(i + 1) * tstride].set(np.ascontiguousarray(Ti, np.float32).ravel(), stream=stream)
    return dict(T=T, tau=cp.asarray(np.ascontiguousarray(tau_idx, np.int64)), tstride=tstride,
                shape=(ntau, NPHI, nb, nd), beta=beta, delta=delta)


def free():
    """return the device and page-locked staging pools to the system"""
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
