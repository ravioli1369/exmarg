"""
the (beta, Delta) tables, read once into host memory; for the GPU in
page-locked memory, copied to the device in groups.

redshift store   bd_*.npz, one file per signal power node s = u y^2, members
                 T (tau, phi+, beta, Delta) float32, beta, delta,
                 tau_idx, alpha (amplitude axis), s_node, and asym (absent:
                 0, the published store); the distance is a parameter, its
                 prior is applied by the analysis

A store of asymmetry w_j holds at node s the tables of the damping exp(-s [(1 + w_j) |R_H|^2 + (1 - w_j) |R_L|^2]):
s is the mean power sbar = (s_H + s_L) / 2, s_H = s (1 + w_j), s_L = s (1 - w_j) (docs/2026-11-15_unequal_noise_design.md,
Sec. 2).

unbiased cache   the unbiased tables of a store written once (write_unbiased_cache): T_<k>.npy float32 per node k (in
                 the order of s), header.npz, cache.json (the source files).  load_redshift_store(cache, mmap=True)
                 maps them, so that the jobs of a machine share one copy in the page cache instead of each holding a
                 private page-locked one.

The tables hold lnS at the points of a grid and are read by linear
interpolation.  On a grid of spacing h a function of curvature T'' is then
overestimated by h^2 T'' f (1 - f) / 2 at the fraction f of a cell, h^2 T'' / 12
on average, and lnS is convex in beta and Delta (a log-sum-exp of the
orders): under noise the likelihood ratio read from the tables has mean
1 + 7e-4 s instead of 1.  unbias() subtracts that average, the second
differences of the table over 12, and leaves an error of zero mean over a cell.
"""

import glob
import json
import os
import struct
import time
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


def _pinned(T):
    """a copy of T in page-locked memory of its own size (T itself if that fails).  Not from cupy's pinned pool, which
    rounds every block up to a power of two: 1 GiB for a 0.70 GiB table"""
    import cupy as cp
    try:
        buf = cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(T.nbytes), 0)
    except Exception:
        return T
    P = np.frombuffer(buf, T.dtype, T.size).reshape(T.shape)
    P[...] = T
    return P


def _read_table(path, pinned, unbiased=False):
    """the table of one store file (member T) or cache file (T_<k>.npy), in page-locked memory when pinned (pageable
    if that fails)"""
    T = np.load(path) if path.endswith(".npy") else read_npz_member(path, "T.npy")
    if unbiased:
        T = unbias(T)
    return _pinned(T) if pinned else T


def _read_tables(files, pinned, readers, mmap=False, unbiased=False):
    if mmap:
        if unbiased:
            raise ValueError("unbiased tables are computed in memory: use mmap=False, or map an unbiased cache "
                             "(write_unbiased_cache)")
        return [np.asarray(np.load(f, mmap_mode="r")) if f.endswith(".npy") else map_npz_member(f, "T.npy")
                for f in files]
    with ThreadPoolExecutor(max_workers=readers) as ex:
        return list(ex.map(lambda f: _read_table(f, pinned, unbiased), files))


@dataclass
class RedshiftStore:
    """signal power nodes s (sorted), their amplitude axis limits, grids, host tables, the truncation order and the
    asymmetry w_j of the damping (0: the published store; s is then the mean power of the two detectors)"""
    s: np.ndarray
    alpha_lo: np.ndarray
    alpha_hi: np.ndarray
    beta: list
    delta: list
    tau_idx: np.ndarray
    tables: list
    n_max: int
    asym: float = 0.0


CACHE = "cache.json"           # the record of an unbiased cache, written last


def is_cache(store_dir):
    return os.path.exists(os.path.join(store_dir, CACHE))


def _sources(store_dir):
    """[(name, size)] of the store files bd_*.npz of a folder, sorted"""
    return sorted((os.path.basename(f), os.path.getsize(f)) for f in glob.glob(os.path.join(store_dir, "bd_*.npz")))


def read_header(store_dir):
    """the small arrays of a store folder or unbiased cache, nodes sorted by s: dict s, alpha_lo, alpha_hi, beta,
    delta (lists), tau_idx, n_max, asym, files (the table of each node).  The files of a store must agree on asym
    (absent: 0); a cache whose source folder still exists must have been written from its present files"""
    if is_cache(store_dir):
        with open(os.path.join(store_dir, CACHE)) as fh:
            rec = json.load(fh)
        if os.path.isdir(rec["source"]) and [list(x) for x in _sources(rec["source"])] != sorted(rec["files"]):
            raise ValueError("the cache %s was written from other files than those now in %s" % (store_dir, rec["source"]))
        with np.load(os.path.join(store_dir, "header.npz")) as h:
            n = h["s"].size
            return dict(s=h["s"], alpha_lo=h["alpha_lo"], alpha_hi=h["alpha_hi"], tau_idx=h["tau_idx"],
                        beta=[h["beta_%d" % k] for k in range(n)], delta=[h["delta_%d" % k] for k in range(n)],
                        n_max=int(h["n_max"]), asym=float(h["asym"]),
                        files=[os.path.join(store_dir, "T_%03d.npy" % k) for k in range(n)])
    files = glob.glob(os.path.join(store_dir, "bd_*.npz"))
    if not files:
        raise FileNotFoundError("no bd_*.npz in %s" % store_dir)
    head = []
    for f in files:
        with np.load(f) as h:
            head.append((float(h["s_node"]), float(h["alpha"][0]), float(h["alpha"][-1]),
                         h["beta"], h["delta"], h["tau_idx"], float(h["asym"]) if "asym" in h.files else 0.0, f))
    head.sort(key=lambda r: r[0])
    s, alo, ahi, beta, delta, tau, asym, files = zip(*head)
    if len(set(asym)) > 1:
        raise ValueError("the files of %s have different asym: %s" % (store_dir, sorted(set(asym))))
    return dict(s=np.array(s), alpha_lo=np.array(alo), alpha_hi=np.array(ahi), beta=list(beta), delta=list(delta),
                tau_idx=tau[0], n_max=int(os.path.basename(files[0]).split("_n")[1].split("_")[0]), asym=asym[0],
                files=list(files))


def load_redshift_store(store_dir, pinned=False, readers=6, mmap=False, unbiased=False, asym=None, nodes=None):
    """the nodes of the redshift store in host memory (about 0.75 GB per node);
    pinned=True puts them in page-locked memory for fast copies to the GPU, mmap=True
    maps them from disk instead of reading them, unbiased=True removes the mean
    error of the linear interpolation (unbias).  store_dir may be an unbiased cache
    (write_unbiased_cache): its tables are already unbiased (unbiased=True is then
    required), and with mmap=True the jobs of a machine share them.  asym replaces
    the asymmetry of the files (tests, timing); nodes loads only those node indices
    (in the order of s of the whole store)"""
    h = read_header(store_dir)
    cache = is_cache(store_dir)
    if cache and not unbiased:
        raise ValueError("%s is an unbiased cache: load it with unbiased=True" % store_dir)
    keep = np.arange(h["s"].size) if nodes is None else np.unique(np.asarray(nodes, int))
    if keep.size == 0 or keep[0] < 0 or keep[-1] >= h["s"].size:
        raise ValueError("nodes must be indices of the %d nodes of %s" % (h["s"].size, store_dir))
    return RedshiftStore(s=h["s"][keep], alpha_lo=h["alpha_lo"][keep], alpha_hi=h["alpha_hi"][keep],
                         beta=[h["beta"][k] for k in keep], delta=[h["delta"][k] for k in keep], tau_idx=h["tau_idx"],
                         tables=_read_tables([h["files"][k] for k in keep], pinned, readers, mmap,
                                             unbiased and not cache),
                         n_max=h["n_max"], asym=h["asym"] if asym is None else float(asym))


def write_unbiased_cache(store_dir, cache_dir, readers=4):
    """the unbiased tables of the store folder store_dir written once to cache_dir: T_<k>.npy (float32, node k in the
    order of s), header.npz (s, alpha_lo, alpha_hi, tau_idx, n_max, asym, beta_<k>, delta_<k>) and, last, cache.json
    (the source folder and its files with their sizes).  Nothing is done when cache_dir already holds a complete cache
    of the same files.  readers tables are unbiased at once (about 4 GB of memory each)"""
    h = read_header(store_dir)
    if is_cache(store_dir):
        raise ValueError("%s is already a cache" % store_dir)
    record = dict(source=os.path.abspath(store_dir), files=[list(x) for x in _sources(store_dir)], unbiased=True,
                  asym=h["asym"], n_node=int(h["s"].size))
    if is_cache(cache_dir):
        with open(os.path.join(cache_dir, CACHE)) as fh:
            old = json.load(fh)
        if {k: old.get(k) for k in record} == record:
            return cache_dir
        raise ValueError("%s holds the cache of other files" % cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    def save(path, write):
        tmp = path + ".%d.tmp" % os.getpid()
        with open(tmp, "wb") as fh:
            write(fh)
        os.replace(tmp, path)

    def one(k):
        T = unbias(read_npz_member(h["files"][k], "T.npy"))
        save(os.path.join(cache_dir, "T_%03d.npy" % k), lambda fh: np.save(fh, T))
    with ThreadPoolExecutor(max_workers=readers) as ex:
        list(ex.map(one, range(h["s"].size)))
    grids = {}
    for k in range(h["s"].size):
        grids["beta_%d" % k], grids["delta_%d" % k] = h["beta"][k], h["delta"][k]
    save(os.path.join(cache_dir, "header.npz"), lambda fh: np.savez(fh, s=h["s"], alpha_lo=h["alpha_lo"],
                                                                       alpha_hi=h["alpha_hi"], tau_idx=h["tau_idx"],
                                                                       n_max=h["n_max"], asym=h["asym"], **grids))
    save(os.path.join(cache_dir, CACHE), lambda fh: fh.write(json.dumps(dict(record, written=time.strftime(
        "%Y-%m-%d %H:%M:%S"))).encode()))
    return cache_dir


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
