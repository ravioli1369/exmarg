"""gather_gpu against gather_cpu with and without time weights (gather.py), and the GPU time with and without
them at a realistic size.  Needs a GPU and cupy:

    python scripts/validate/gather_gpu_weights.py                      the real geometry (87 lags, 200 x 219)
    python scripts/validate/gather_gpu_weights.py --store <store_dir>  shapes, grids and lags of the store's first node
    python scripts/validate/gather_gpu_weights.py --ntab 1             one table on the device instead of six

The tables are synthetic (smooth random content) of the given shape; the trajectories are random.  Checked, for
the four combinations of {every arrival time, select_times} x {no weights, weights}: a small case (n_time 256,
one partial chunk of the shared-memory sort) and the realistic shape on a few templates (n_time 32768, two
chunks).  Every ln Lambda of the GPU must be within --tol of the CPU's (float32 sums against float64).  The
selected times are drawn on the host and copied to the device, so both visit the same times.

Exit status 1 if a check fails.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from exmarg.search import gather, store

NPHI = 96


def grids(args):
    """(ntau, nbeta, ndelta), beta, delta, tau of the store's first node, or from the arguments"""
    if args.store:
        f = sorted(glob.glob(os.path.join(args.store, 'bd_*.npz')))[0]
        shape = store.map_npz_member(f, 'T.npy').shape
        with np.load(f) as h:
            beta, delta, tau = h['beta'], h['delta'], h['tau_idx']
        return (shape[0], shape[2], shape[3]), np.asarray(beta, float), np.asarray(delta, float), np.asarray(tau)
    tau = np.arange(args.ntau) - args.ntau // 2
    beta = args.beta0 + args.dbeta * np.arange(args.nbeta)
    delta = -args.ddelta * (args.ndelta // 2) + args.ddelta * np.arange(args.ndelta)
    return (args.ntau, args.nbeta, args.ndelta), beta, delta, tau


def table(rng, ntau, nb, nd):
    """smooth random float32 table (ntau, NPHI/2 + 1, nb, nd), built one lag at a time"""
    c = rng.normal(size=6)
    ph = np.cos(2 * np.pi * np.arange(NPHI // 2 + 1) / NPHI).astype(np.float32)[:, None, None]
    xb = np.linspace(0, 1, nb, dtype=np.float32)[None, :, None]
    xd = np.linspace(-1, 1, nd, dtype=np.float32)[None, None, :]
    T = np.empty((ntau, NPHI // 2 + 1, nb, nd), np.float32)
    for r, xt in enumerate(np.linspace(-1, 1, ntau)):
        T[r] = (c[0] + c[1] * ph * (1 + xb) + 3 * xb + c[2] * xb * xb - 2 * xd * xd + c[3] * xd * xb
                + 0.3 * ph * xd + 0.2 * c[4] * xt + 0.1 * c[5] * xt * ph)
    return T


def inputs(rng, n_template, n_time, beta, delta, ns, ntab):
    """random trajectories (a loud stretch in each), ln a inside the beta grid, slot n reading table n % ntab,
    log time weights"""
    lqH = rng.normal(0.0, 0.8, (n_template, n_time))
    lqL = rng.normal(0.0, 0.8, (n_template, n_time))
    lqH[:, 40:46] += 2.0
    lqL[:, 38:44] += 2.0
    traj = dict(lqH=lqH.astype(np.float32), lqL=lqL.astype(np.float32),
                phH=rng.uniform(-np.pi, np.pi, (n_template, n_time)).astype(np.float32),
                phL=rng.uniform(-np.pi, np.pi, (n_template, n_time)).astype(np.float32))
    mid = 0.5 * (beta[0] + beta[-1])
    lna = rng.uniform(mid - 0.25 * (beta[-1] - beta[0]), mid + 0.25 * (beta[-1] - beta[0]), (n_template, ns))
    tab_ids = np.tile(np.arange(ns) % ntab, (n_template, 1))
    lk = [np.log(rng.uniform(0.0, 1.0, n_time)).astype(np.float32) for _ in range(2)]
    lk[0][::7] = -np.inf                                # zero weights, clipped at -80
    return traj, lna, tab_ids, tuple(lk)


def compare(label, traj, host_tables, dev_tables, ids, lna, tab_ids, lk, tol, block, n_loud, n_random):
    import cupy as cp
    dtraj = {k: cp.asarray(v) for k, v in traj.items()}
    times = gather.select_times(traj, host_tables['tau'], n_loud, n_random, seed=1)
    dlk = tuple(cp.asarray(w) for w in lk)
    ok = True
    for tm in (None, times):
        dtm = None if tm is None else tuple(cp.asarray(a) for a in tm)
        for w, dw in ((None, None), (lk, dlk)):
            c = gather.gather_cpu(traj, host_tables, ids, lna[ids], tab_ids[ids], tm, time_weights=w)
            g = gather.gather_gpu(dtraj, dev_tables, ids, lna[ids], tab_ids[ids], block, dtm, time_weights=dw)
            if g.shape != c.shape:
                print('%-26s shapes differ: gpu %s cpu %s  FAIL' % (label, g.shape, c.shape))
                ok = False
                continue
            d = np.abs(g - c).reshape(-1, *c.shape[-2:])
            d = np.where(np.isfinite(d), d, np.inf)
            good = bool(np.all(d <= tol))
            ok &= good
            print('%-26s %-9s %-10s max |gpu - cpu| per variant: %s  %s'
                  % (label, 'all times' if tm is None else 'selected', 'weights' if w is not None else 'no weights',
                     ' '.join('%.2e' % x for x in d.reshape(d.shape[0], -1).max(axis=1)), 'pass' if good else 'FAIL'))
    return ok


def timing(traj, dev_tables, lna, tab_ids, lk, block, repeat, n_loud, n_random, host_tau):
    import cupy as cp
    dtraj = {k: cp.asarray(v) for k, v in traj.items()}
    ids = np.arange(traj['lqH'].shape[0])
    sel, lw = gather.select_times(traj, host_tau, n_loud, n_random, seed=1)
    dlk = tuple(cp.asarray(w) for w in lk)
    rows = []
    for tm in (None, (cp.asarray(sel), cp.asarray(lw))):
        res = {}
        for w in (None, dlk):
            f = lambda: gather.gather_gpu(dtraj, dev_tables, ids, lna, tab_ids, block, tm, time_weights=w)
            f()                                         # compile
            cp.cuda.Device().synchronize()
            ts = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                f()                                     # returns on the host: synchronous
                ts.append(time.perf_counter() - t0)
            res[w is not None] = float(np.median(ts))
        rows.append(('all times' if tm is None else 'selected %d' % (n_loud + n_random), res[False], res[True]))
    for name, t0, t1 in rows:
        print('  %-14s no weights %8.1f ms   weights %8.1f ms   ratio %.2f' % (name, 1e3 * t0, 1e3 * t1, t1 / t0))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--store', help='redshift store directory: shapes, grids and lags from its first bd_*.npz')
    p.add_argument('--ntau', type=int, default=87)
    p.add_argument('--nbeta', type=int, default=200)
    p.add_argument('--ndelta', type=int, default=219)
    p.add_argument('--beta0', type=float, default=-7.824046)       # build_bd.py: ln 0.16 + 2 ln 0.05
    p.add_argument('--dbeta', type=float, default=0.1394661)       # 2 ln(2000) / 109
    p.add_argument('--ddelta', type=float, default=0.06973305)     # ln(2000) / 109, delta symmetric about 0
    p.add_argument('--ns', type=int, default=6, help='slots (nodes per launch)')
    p.add_argument('--ntab', type=int, default=6, help='distinct tables on the device (slot n reads n %% ntab)')
    p.add_argument('--n-template', type=int, default=120)
    p.add_argument('--n-time', type=int, default=32768)
    p.add_argument('--ncheck', type=int, default=4, help='templates compared with the CPU at the realistic shape')
    p.add_argument('--n-loud', type=int, default=1024)
    p.add_argument('--n-random', type=int, default=1024)
    p.add_argument('--block', type=int, default=256)
    p.add_argument('--block2', type=int, default=512, help='second block size for the small case (0: none)')
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--tol', type=float, default=2e-4, help='largest |gpu - cpu| in ln Lambda')
    p.add_argument('--no-timing', action='store_true')
    args = p.parse_args()

    import cupy as cp
    dev = cp.cuda.Device()
    print('device %d, MaxSharedMemoryPerBlockOptin %d B' % (dev.id, dev.attributes['MaxSharedMemoryPerBlockOptin']))
    for nv in (1, 3):
        st = gather._static(args.ns, nv)
        ch = gather._chunk(NPHI, st)
        print('  NV=%d: chunk %d, dynamic shared %d B + static %d B' % (nv, ch, 7 * ch + 4 * NPHI + 256, st))
    rng = np.random.default_rng(0)
    ok = True

    # small: n_time 256, 7 lags, 12 x 10 tables, 3 tables
    beta_s, delta_s, tau_s = -1.0 + 0.5 * np.arange(12), -2.0 + 0.4 * np.arange(10), np.arange(-3, 4)
    Ts = [table(rng, tau_s.size, 12, 10) for _ in range(3)]
    traj, lna, tab_ids, lk = inputs(rng, 4, 256, beta_s, delta_s, args.ns, 3)
    lna = rng.uniform(-0.5, 2.0, lna.shape)
    host = dict(T=Ts, tau=tau_s, beta=beta_s, delta=delta_s)
    dev_tables = store.upload(Ts, tau_s, beta_s, delta_s)
    ok &= compare('small (n_time 256)', traj, host, dev_tables, np.array([2, 0, 3]), lna, tab_ids, lk, args.tol,
                  args.block, 24, 40)
    if args.block2:
        # more warps in the block reduction (wm, ws filled up to nwarp NS NV); may not launch for want of registers
        try:
            ok &= compare('small, block %d' % args.block2, traj, host, dev_tables, np.array([2, 0, 3]), lna, tab_ids,
                          lk, args.tol, args.block2, 24, 40)
        except cp.cuda.driver.CUDADriverError as e:
            print('small, block %d: not launched (%s), skipped' % (args.block2, e))
    del dev_tables

    # realistic shape
    (ntau, nb, nd), beta, delta, tau = grids(args)
    print('realistic: %d lags, tables (%d, %d, %d, %d) float32 = %.2f GB each, %d on the device'
          % (ntau, ntau, NPHI // 2 + 1, nb, nd, ntau * (NPHI // 2 + 1) * nb * nd * 4 / 1e9, args.ntab))
    T = table(rng, ntau, nb, nd)
    host = dict(T=[T] * args.ntab, tau=tau, beta=beta, delta=delta)
    dev_tables = store.upload(host['T'], tau, beta, delta)
    traj, lna, tab_ids, lk = inputs(rng, args.n_template, args.n_time, beta, delta, args.ns, args.ntab)
    ids = rng.choice(args.n_template, min(args.ncheck, args.n_template), replace=False)
    ok &= compare('realistic (n_time %d)' % args.n_time, traj, host, dev_tables, ids, lna, tab_ids, lk, args.tol,
                  args.block, args.n_loud, args.n_random)

    # the compiled kernels: registers, spills, and the static shared memory that _chunk assumes
    for (nb_, nd_, _, nt_, ns_, ch, blk, ks, nv), (fn, shm) in sorted(gather._CACHE.items()):
        at = fn.attributes
        good = at['shared_size_bytes'] <= gather._static(ns_, nv)
        ok &= good
        print('kernel n_time %d NS=%d NV=%d SEL=%d block %d chunk %d: %d registers, %d B local (spills), static '
              'shared %d B (assumed %d)  %s' % (nt_, ns_, nv, ks > 0, blk, ch, at['num_regs'], at['local_size_bytes'],
                                                 at['shared_size_bytes'], gather._static(ns_, nv),
                                                 'pass' if good else 'FAIL'))

    if not args.no_timing:
        print('gather_gpu, %d templates x %d lags x %d slots, n_time %d (median of %d):'
              % (args.n_template, ntau, args.ns, args.n_time, args.repeat))
        timing(traj, dev_tables, lna, tab_ids, lk, args.block, args.repeat, args.n_loud, args.n_random, tau)

    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
