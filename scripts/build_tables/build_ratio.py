"""The redshift store of asymmetry w: the tables for unequal noise in H1 and L1 (notes/2026-11-15_unequal_noise_design.md,
Sec. 2).  At a node sbar of data/tables/s_grid.npz, with s_H = sbar (1 + w), s_L = sbar (1 - w),

    T(tau, phi, beta, Delta) = ln sum_{n <= 120} alpha^n K_n,
    K_n = < e^{-(s_H |R_H|^2 + s_L |R_L|^2)} |conj(R_H) z_1 + conj(R_L) z_2|^(2n) >_{sky, psi, chi in the lag tau} / n!^2,

beta = ln alpha + 2 ln q2, Delta = ln q1 - ln q2, q_i = |z_i|, phi = arg z_1 - arg z_2 (detector 1 = H), on the grids of
the published store (w = 0: build_w.py, build_k.py, build_bd.py), and in the same files.

Stage W, one GPU sweep of the sky for a group of nodes (the integrand is w_gpu._chunk_vc_s_gpu):
    V^(n)_ab(tau) = < e^{-(s_H |R_H|^2 + s_L |R_L|^2)} (|R_H|/sqrt2)^(a+b) (|R_L|/sqrt2)^(2n-a-b) cos[(a-b)(arg R_H - arg R_L)] >
The published W is V times a storage scale; here ONE reference, sbar, serves both detectors:
    W^(n)_ab = e^{l_n} V^(n)_ab,   l_n = n ln sbar - ln n! + n ln 2      (W = sbar^n / n! < e^{-D} |R_H|^(a+b) ... >)
and build_k.py removes o_n = ln_offset(n, sbar, sbar) = ln n! + n ln sbar.

Stage T, directly from V on the 219 points of Delta (no (q1, q2) grid), in float64 until the final float32:
    G_n(tau, phi, Delta) = sum_{a,b <= n} C(n,a) C(n,b) V^(n)_ab(tau) e^{(a+b) Delta - 2n max(Delta, 0)} cos[(a-b) phi]
    g_n = ln G_n + l_n + 2n max(Delta, 0) - o_n,        T = logsumexp_n [n beta + g_n],
so that e^{2n ln q2 + g_n} = K_n exactly (l_n - o_n = n ln 2 - 2 ln n!: the reference cancels).  Every factor
e^{(a+b) Delta - 2n max(Delta, 0)} is <= 1.  The sum over (a, b) is signed and taken in float64: where it comes out
<= 0 its true value (>= 0) is below its rounding, and the order is left out; in the lags without sky (|tau idx| = 42,
43, V = 0) T = ln 1e-300, the smallest value of the published tables.
Differences from the published chain, which stored W and lnK as float32 and floored every sum at 1e-300 on the scale
of W (ln 1e-300 + 2n max(Delta, 0) - o_n, which exceeds the true order by up to e^700 at small s and in the lags
without sky): here V and the scale are kept apart, nothing underflows and nothing is floored.
Common to both: where T lies far below T_abs (the same series with every term made positive, an upper bound of T at
any phi), at large Bessel arguments and phases near 0, where H and L respond with opposite signs, the signed sum is
dominated by rounding: error ~ (0.5 to 2.5) 1e-16 e^(T_abs - T), measured against the quadrature.
Stage R (repair).  The search reads the table for data below its loudness cuts (single-detector |z_I|^2 / u_I < zcut,
that is w_L = e^beta / s_L < zcut and w_H = e^(beta + 2 Delta) / s_H < zcut; network W = w_H + w_L <= wnet), bilinearly
in (beta, Delta) (search/gather.py) and after the removal of the interpolation bias, which takes second differences
along beta and Delta (search/store.py, unbias).  A read therefore depends on raw entries up to two grid steps outside
the cuts (W up to 1.5 wnet); need_mask is that set, and stage R answers for every entry of it.  An entry keeps its
series value when both errors of the series are under control: the cancellation (T_abs - T <= gap, 18: error < 2e-8),
and the truncation, which is either measured, for W <= W_SERIES = 84 (below 1e-4 at every node), or proven below tol
(1e-6) by the bound below with the entry's own T.  Every other entry is replaced by the quadrature of the
formula itself over the points of the lag, a sum of positive terms with no series (function repair; its own i0e on
the GPU, since cupy's is wrong above 88 in float32 and above 713 in float64).  The search uses zcut 72 and wnet 84,
and the files record the cuts they were repaired for (repair_zcut, repair_wnet, repair_gap, repair_tol).
Truncation at n = 120: at every sky point a|X|^2 <= x W (Cauchy-Schwarz), x = s_H |R_H|^2 + s_L |R_L|^2 <= s_H + s_L,
so the lag sum loses at most omega_lag sup_{x <= s_H + s_L} e^{-x} sum_{n > 120} (x W)^n / (n!)^2 (absolute; in ln,
divided by e^T; omega_lag the sky weight of the lag), and each term is largest at x = min(n, s_H + s_L) (ln_tail).
For s_H + s_L >= 121 this is e^W P(Poisson(W) > 120) / sqrt(2 pi 121).  Measured against the quadrature: below 1e-4 for
W <= 84 at every node, nothing at sbar <= 34 for any W <= 144.

    python scripts/build_tables/build_ratio.py <out_dir> --asym W [--nodes i j k ...] [--keep-w DIR]
           [--per-sweep 53] [--chunk 8] [--part k K] [--slices K] [--repair-zcut 72] [--repair-wnet 84]
           [--repair-gap 18] [--repair-tol 1e-6] [--repair-dry]
    -> <out_dir>/bd_n120_s<sbar %09.3f>.npz: the members of the published files plus asym, s_H, s_L and the cuts of
       stage R (repair_zcut, repair_wnet, repair_gap, repair_tol; repair_zcut 0: built without stage R)
       --keep-w DIR keeps V on the data lags (float64, rows (n, a, b) with b <= a) and l_n per node, and resumes
       from them.  --part k K (needs --keep-w) sweeps only the k-th of K slices of the sky and writes its V; a run
       without --part sums the K slices (the deposit keeps the global 1/npix, so they add up to the whole sky) and
       writes the tables.  A sweep costs the same for 1 or 53 nodes (the gather of the monomials dominates), so the
       sky, not the nodes, is what to split between GPUs.  --slices K (needs --keep-w) sweeps the K slices one after
       the other in one run and writes the running sum after each: a killed run resumes after the last finished slice.
       --repair-dry builds no table: per node, the number of entries stage R would replace and why.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import healpy as hp
from scipy.special import comb, gammaln

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src'))
import build_k
from build_bd import ALPHA
from exmarg.detectors import c, x, basis, Fplus_Fcross
from exmarg.numerics import ln_offset
from exmarg.tables import _quad, _tau_grid, _deposit, patterns, rebin

N_MAX, NSIDE, N_TAU = 120, 32, 257             # the settings of build_w.py
LNRMAX2 = np.log(2.0)                           # ln max|R|^2 of w_gpu.py
LN_FLOOR = np.log(1e-300)                       # T in the lags without sky (the minimum of the published tables)
W_SERIES = 84.0                                 # w_H + w_L up to which the truncation at N_MAX was measured below 1e-4


def grids(tau_f):
    """the grids of the published store: data lags (tau_idx, the rebin matrix from the delays tau_f of W), lnq,
    Delta, beta, phi.  As build_k.py (which read tau_f as float32) and build_bd.py."""
    tau_f = np.asarray(tau_f).astype(np.float32)
    keep = np.abs(build_k.tau_all) <= 1.05 * np.abs(tau_f).max()
    tau_lag = build_k.tau_all[keep]
    tau_idx = np.rint(tau_lag / build_k.t_step).astype(int)
    eye = np.eye(tau_f.size)
    rows = rebin(tau_f, {j: eye[j] for j in range(tau_f.size)}, tau_lag)
    rb = np.stack([rows[j] for j in range(tau_f.size)], axis=1)            # (n_lag, n_tau_f), bin sums
    lnq = np.log(np.geomspace(build_k.QMAX / build_k.QSPAN, build_k.QMAX, build_k.NQ))
    last = build_k.NQ - 1
    delta = np.concatenate([lnq - lnq[last], (lnq[last] - lnq)[::-1][1:]])
    dbeta = 2.0 * float(lnq[1] - lnq[0])
    b_lo = np.log(ALPHA[0]) + 2.0 * lnq[0]
    b_hi = np.log(ALPHA[-1]) + 2.0 * lnq[-1]
    beta = b_lo + dbeta * np.arange(int(np.ceil((b_hi - b_lo) / dbeta)) + 1)
    nphif = build_k.NPHI // 2 + 1
    phi = np.arange(nphif) * (2 * np.pi / build_k.NPHI)
    return dict(tau_idx=tau_idx, rb=rb, lnq=lnq, delta=delta, beta=beta, phi=phi, nphif=nphif)


def ln_scale(n, s_ref):
    """l_n = n ln s_ref - ln n! + n ln 2, the storage scale of W at the reference s_ref"""
    n = np.asarray(n, float)
    return n * np.log(s_ref) - gammaln(n + 1.0) + n * LNRMAX2


def w_sweep(s_H, s_L, n_max=N_MAX, nside=NSIDE, n_tau=N_TAU, chunk=8, pix=None, verbose=False):
    """stage W for a group of nodes in one sky sweep (the loop of w_gpu.w_table_s_gpu without its scale): the fine
    delays tau, the keys (n, a, b), b <= a, and V (n_nodes, n_lag, n_keys) float64 on the host, the normalised
    averages with damping e^{-(s_H |R_H|^2 + s_L |R_L|^2)} node by node, on the data lags of grids(tau).  Each pixel
    is deposited on the fine delays and the fine delays summed into the data lags (build_k's rebin) before the
    accumulation: the same linear map as rebinning the finished fine table, in a third of the device memory.
    pix = (lo, hi) sweeps only those pixels, with the weight 1/npix of the whole sky"""
    import w_gpu
    cp = w_gpu.cp
    s_H = np.atleast_1d(np.asarray(s_H, float))
    s_L = np.atleast_1d(np.asarray(s_L, float))
    psi, cg, wq = _quad(n_max, None, None)
    _, tau, order, ts, dtau = _tau_grid(None, n_tau)
    rb = cp.asarray(grids(tau)['rb'])
    npix = hp.nside2npix(nside)
    keys = patterns(n_max)
    kidx = {k: j for j, k in enumerate(keys)}
    plan = w_gpu._blocks(n_max, keys, kidx)
    V = cp.zeros((s_H.size, rb.shape[0], len(keys)), cp.float64)
    p_lo, p_hi = (0, npix) if pix is None else pix
    t0 = time.time()
    for lo in range(p_lo, p_hi, chunk):
        hi = min(lo + chunk, p_hi)
        th, ph = hp.pix2ang(nside, np.arange(lo, hi))
        Vc = w_gpu._chunk_vc_s_gpu(th, ph, s_H, s_L, n_max, psi, cg, wq, plan, len(keys))
        t_del = (basis(th, ph)[0] @ (x['H'] - x['L'])) / c
        Pg = rb @ cp.asarray(_deposit(tau, order, ts, dtau, t_del, npix).toarray(), cp.float64)
        for j in range(s_H.size):
            V[j] += Pg @ Vc[:, j, :]
        del Vc
        if verbose and ((lo - p_lo) // chunk) % max(1, (p_hi - p_lo) // chunk // 20) == 0:
            print('  W sweep: pixels %d..%d of %d..%d, %.0f s' % (lo, hi, p_lo, p_hi, time.time() - t0), flush=True)
    if not bool(cp.isfinite(V).all()):
        raise FloatingPointError('V non-finite')
    return tau, keys, cp.asnumpy(V)


def order_plan(n_max):
    """per order n: the columns of its keys in patterns(n_max) and their m = a - b, p = a + b and coefficient
    C(n,a) C(n,b), doubled for a != b (the mirrored key (n, b, a))"""
    plan, k0 = [], 0
    for n in range(n_max + 1):
        ab = np.array([(a, b) for a in range(n + 1) for b in range(a + 1)])
        a, b = ab[:, 0], ab[:, 1]
        coef = comb(n, a) * comb(n, b) * np.where(a > b, 2.0, 1.0)
        plan.append((k0, k0 + a.size, a - b, a + b, coef))
        k0 += a.size
    return plan


def t_from_v(V, s_ref, delta, beta, phi, n_max=N_MAX, xp=np, out_dtype=np.float32, nonpos=None):
    """stage T from V on the data lags (n_lag, n_keys): T (n_lag, n_phi, n_beta, n_delta), formula of the module doc.
    nonpos, an (n_lag,) array, receives the number of (n, phi, Delta) with G_n <= 0 (orders left out) per lag"""
    V = xp.asarray(V, xp.float64)
    d = xp.asarray(delta, xp.float64)
    dpos = xp.maximum(d, 0.0)
    ph = xp.asarray(phi, xp.float64)
    b = xp.asarray(beta, xp.float64)
    n_lag = V.shape[0]
    g = xp.empty((n_max + 1, n_lag, ph.size, d.size))
    count = xp.zeros(n_lag, xp.int64)
    for n, (k0, k1, m, p, coef) in enumerate(order_plan(n_max)):
        M = xp.zeros((n_lag, n + 1, 2 * n + 1))
        M[:, xp.asarray(m), xp.asarray(p)] = V[:, k0:k1] * xp.asarray(coef)
        E = xp.exp(xp.arange(2 * n + 1, dtype=xp.float64)[:, None] * d[None, :] - 2.0 * n * dpos[None, :])
        G = xp.cos(xp.outer(ph, xp.arange(n + 1, dtype=xp.float64))) @ (M @ E)      # (n_lag, n_phi, n_delta)
        o_n = float(ln_offset(n, s_ref, s_ref))
        pos = G > 0
        lnG = xp.where(pos, xp.log(xp.where(pos, G, 1.0)) + float(ln_scale(n, s_ref)), -np.inf)
        g[n] = lnG + 2.0 * n * dpos[None, None, :] - o_n
        count += (~pos).sum(axis=(1, 2))
    if nonpos is not None:
        nonpos[...] = count.get() if hasattr(count, 'get') else count
    # the sum over the orders as build_bd.py: the maximum first, then the sum of the exponentials
    mx = xp.full((n_lag, ph.size, b.size, d.size), -np.inf)
    for n in range(n_max + 1):
        mx = xp.maximum(mx, n * b[None, None, :, None] + g[n][:, :, None, :])
    sky = xp.isfinite(mx)
    mx = xp.where(sky, mx, 0.0)
    tot = xp.zeros_like(mx)
    for n in range(n_max + 1):
        tot += xp.exp(n * b[None, None, :, None] + g[n][:, :, None, :] - mx)
    return xp.where(sky, mx + xp.log(xp.where(sky, tot, 1.0)), LN_FLOOR).astype(out_dtype)


def t_abs(V, s_ref, beta, delta, n_max=N_MAX, xp=np):
    """T_abs(lag, beta, Delta): the series of t_from_v with every term made positive,
        ln sum_n e^{n beta} sum_{a,b} C(n,a) C(n,b) |W^(n)_ab| e^{(a+b) Delta} / (s^n n!)   (q2 = 1),
    an upper bound of T at every phi; -inf in the lags without sky"""
    V = xp.abs(xp.asarray(V, xp.float64))
    d = xp.asarray(delta, xp.float64)
    dpos = xp.maximum(d, 0.0)
    b = xp.asarray(beta, xp.float64)
    g = []
    for n, (k0, k1, m, p, coef) in enumerate(order_plan(n_max)):
        onehot = np.zeros((k1 - k0, 2 * n + 1))
        onehot[np.arange(k1 - k0), p] = coef
        A = (V[:, k0:k1] @ xp.asarray(onehot)) @ xp.exp(xp.arange(2 * n + 1, dtype=xp.float64)[:, None] * d[None, :]
                                                         - 2.0 * n * dpos[None, :])
        pos = A > 0
        g.append(xp.where(pos, xp.log(xp.where(pos, A, 1.0)), -np.inf) + float(ln_scale(n, s_ref))
                 + 2.0 * n * dpos[None, :] - float(ln_offset(n, s_ref, s_ref)))
    mx = xp.full((V.shape[0], b.size, d.size), -np.inf)
    for n in range(n_max + 1):
        mx = xp.maximum(mx, n * b[None, :, None] + g[n][:, None, :])
    ok = xp.isfinite(mx)
    m0 = xp.where(ok, mx, 0.0)
    tot = xp.zeros_like(m0)
    for n in range(n_max + 1):
        tot += xp.exp(n * b[None, :, None] + g[n][:, None, :] - m0)
    return xp.where(ok, m0 + xp.log(xp.where(ok, tot, 1.0)), -np.inf)


def lag_weights(nside=NSIDE, n_tau=N_TAU):
    """theta, phi of the sky pixels and their weight in each data lag (n_lag, npix): the linear deposit of stage W on
    the fine delays (tables._deposit, 1/npix per pixel) summed into the data lags (the rebin matrix of grids)"""
    _, tau, order, ts, dtau = _tau_grid(None, n_tau)
    npix = hp.nside2npix(nside)
    th, ph = hp.pix2ang(nside, np.arange(npix))
    t_del = (basis(th, ph)[0] @ (x['H'] - x['L'])) / c
    return th, ph, grids(tau)['rb'] @ _deposit(tau, order, ts, dtau, t_del, npix).toarray()


# exponentially scaled I0 on the device: the library function below 30, above it the asymptotic series
# i0e(x) = (2 pi x)^-1/2 sum_k ((2k - 1)!!)^2 / (k! (8x)^k), 8 terms (relative error 1e-11 in float64, float32 rounding
# in float32, measured against scipy for 30 <= x <= 1e5).  cupyx.scipy.special.i0e is exp(-x) I0(x): 0 or nan above 88
# in float32 and inf or nan above 713 in float64.
I0E_SERIES = (1.0, 1.0, 4.5, 37.5, 459.375, 7441.875, 150077.8125, 3623307.1875)
ROWS = 1024                                     # parallel rows of the reductions of the repair
_KERNELS = {}


def _repair_kernels():
    """two fused reductions over the points of a lag: top = max_p (base_p + c y_p) and
    sum_p exp(float32(base_p + c y_p - top)) i0e(float32(c y_p)) accumulated in float64"""
    if not _KERNELS:
        import cupy as cp
        horner = ''.join('s = s * r + %.10ff; ' % ck for ck in I0E_SERIES[::-1])
        pre = ('__device__ float i0e_f(float x) { if (x < 30.f) return expf(-x) * cyl_bessel_i0f(x); '
               'float r = 1.f / (8.f * x); float s = 0.f; ' + horner +
               'return s * rsqrtf(6.283185307179586f * x); }')
        _KERNELS['max'] = cp.ReductionKernel('float64 base, float64 y, float64 c', 'float64 out', 'base + c * y',
                                             'max(a, b)', 'out = a', '-CUDART_INF', 'repair_max')
        _KERNELS['sum'] = cp.ReductionKernel(
            'float64 base, float64 y, float64 c, float64 top', 'float64 out',
            '(double)(expf((float)(base + c * y - top)) * i0e_f((float)(c * y)))', 'a + b', 'out = a', '0',
            'repair_sum', preamble='#include <cupy/math_constants.h>\n' + pre)
    return _KERNELS


def need_mask(beta, delta, s_H, s_L, zcut, wnet=np.inf):
    """the (beta, Delta) entries of the raw table on which a read of data below the loudness cuts depends, and
    W = w_L + w_H of every entry.  Data: w_L = e^beta / s_L < zcut, w_H = e^(beta + 2 Delta) / s_H < zcut, W <= wnet, a
    set closed under lowering beta or Delta.  The search reads the unbiased table bilinearly (gather.py): entry (k, m)
    is read when a datum lies in the open square (k - 1, k + 1) x (m - 1, m + 1) of grid steps, that is when the grid
    point (k - 1, m - 1) is strictly below the cuts.  The unbiased entry is the raw one minus its second differences
    along beta and Delta (store.unbias): the raw entries (k +- 1, m) and (k, m +- 1) enter it.  Together: the raw
    entry (k, m) is needed when (k - 2, m - 1) or (k - 1, m - 2) is strictly below the cuts (indices below 0 taken as
    0: data below the table are read at its edge).  beta and delta increasing"""
    beta, delta = np.asarray(beta, float), np.asarray(delta, float)
    wL = np.exp(beta)[:, None] / s_L + np.zeros(delta.size)
    wH = np.exp(beta[:, None] + 2 * delta[None, :]) / s_H
    W = wL + wH
    below = (wL < zcut) & (wH < zcut) & (W < wnet)
    k, m = np.arange(beta.size), np.arange(delta.size)

    def from_corner(i, j):
        return below[np.ix_(np.maximum(k - i, 0), np.maximum(m - j, 0))]
    return from_corner(2, 1) | from_corner(1, 2), W


def ln_tail(W, x_max, n_max=N_MAX, w_big=3000.0):
    """ln of a bound of the orders left out, per unit sky weight of the lag:
        sup_{x <= x_max} e^{-x} sum_{n > n_max} (x W)^n / (n!)^2  <=  sum_{n > n_max} e^{-x_n} (x_n W)^n / (n!)^2,
    x_n = min(n, x_max), each term at its own largest value (e^{-x} x^n grows up to x = n).  The terms fall once
    n > W and n^2 > x_max W; the sum runs until they are e^-60 of the largest, and the rest is smaller still.
    Above w_big no bound is attempted (+inf: the entry goes to the quadrature)"""
    W = np.atleast_1d(np.asarray(W, float))
    out = np.full(W.shape, np.inf)
    use = np.flatnonzero(W.ravel() <= w_big)
    if not use.size:
        return out
    top = float(W.ravel()[use].max())
    n = np.arange(n_max + 1.0, max(n_max + 200.0, np.ceil(6.0 * max(top, np.sqrt(x_max * top)))) + 1.0)
    xn = np.minimum(n, x_max)
    for lo in range(0, use.size, 4096):
        idx = use[lo:lo + 4096]
        lt = n[:, None] * (np.log(xn)[:, None] + np.log(W.ravel()[idx])[None, :]) - xn[:, None] \
            - 2.0 * gammaln(n + 1.0)[:, None]
        mx = lt.max(axis=0)
        if not np.all(lt[-1] < mx - 60.0) or not np.all(lt[-1] < lt[-2] - np.log(2.0)):
            raise ArithmeticError('ln_tail: the orders up to %d do not exhaust the sum' % n[-1])
        out.flat[idx] = mx + np.log(np.exp(lt - mx).sum(axis=0))
    return out


def flag_entries(T, V, s_H, s_L, s_ref, beta, delta, zcut, gap_max, n_max=N_MAX, nside=NSIDE, n_tau=N_TAU, xp=np,
                 wnet=np.inf, tol=1e-6, wseries=W_SERIES):
    """the entries of the series table T (float64, n_lag x n_phi x n_beta x n_delta) that stage R replaces: those of
    need_mask (what a read of data below the cuts zcut, wnet depends on), in the lags with sky, where the series is not
    under control: the cancellation, T_abs - T > gap_max; or the truncation, for the entries with W > wseries (up to
    which it was measured: W_SERIES at the order of the tables), unless the bound proves it below tol:
        ln omega_lag + ln_tail(W, s_H + s_L) - T <= ln tol
    (T of the series is below the true one, the orders left out being positive).  Returns the flags and the counts
    need (entries answered for), cancellation, truncation (flagged for it alone), flagged"""
    beta, delta = np.asarray(beta, float), np.asarray(delta, float)
    Ta = t_abs(V, s_ref, beta, delta, n_max, xp)
    Ta = Ta.get() if hasattr(Ta, 'get') else Ta
    omega = lag_weights(nside, n_tau)[2].sum(axis=1)
    need, W = need_mask(beta, delta, s_H, s_L, zcut, wnet)
    sky = (omega > 0)[:, None, None, None]
    cancel = Ta[:, None] - T > gap_max
    outside = need & (W > wseries)
    trunc = np.zeros(T.shape, bool)
    if outside.any():
        lt = np.full(W.shape, -np.inf)
        lt[outside] = ln_tail(W[outside], s_H + s_L, n_max)
        with np.errstate(divide='ignore'):
            trunc = np.log(omega)[:, None, None, None] + lt[None, None] - T > np.log(tol)
    flags = sky & need[None, None] & (cancel | trunc)
    counts = dict(need=int(need.sum()) * int((omega > 0).sum()) * T.shape[1],
                  cancellation=int((sky & need[None, None] & cancel).sum()),
                  truncation=int((flags & ~cancel).sum()), flagged=int(flags.sum()))
    return flags, counts


def repair(T, flags, s_H, s_L, beta, delta, phi, n_max=N_MAX, nside=NSIDE, n_tau=N_TAU, xp=np):
    """T (float64, n_lag x n_phi x n_beta x n_delta) with the flagged entries (flag_entries) replaced by the quadrature
    with no series
        T = ln sum_p w_p e^{-D_p} I0(2 e^{beta/2} |conj(R_H,p) e^{Delta + i phi} + conj(R_L,p)|),
        D_p = s_H |R_H,p|^2 + s_L |R_L,p|^2,
    over the points p (pixel, psi, chi) of the lag with the weights of stage W (pixel weight in the lag times the
    psi/chi weight; complex128 responses).  All its terms are positive: nothing cancels, nothing is truncated.  With
    ub_p = ln(w_p e^{-D_p}) + arg_p (>= ln of each term, since i0e <= 1) and top = max_p ub_p, T = top + ln sum_p
    e^{ub_p - top} i0e(arg_p).  numpy: float64 with scipy's i0e.  cupy: two fused reductions, the exponential and i0e
    in float32 (the largest terms lie within ~5 of top: ~3e-7 in T), the sum in float64.  Returns T and the number of
    entries replaced"""
    beta, delta, phi = (np.asarray(v, float) for v in (beta, delta, phi))
    th, ph, wl = lag_weights(nside, n_tau)
    psi, cg, wq = _quad(n_max, None, None)
    done = 0
    c2p, s2p = xp.asarray(np.cos(2 * psi)), xp.asarray(np.sin(2 * psi))
    Ap, cgx = xp.asarray(0.5 * (1 + cg**2)), xp.asarray(cg)
    ker = None if xp is np else _repair_kernels()
    if xp is np:
        from scipy.special import i0e
    for l in np.nonzero(flags.any(axis=(1, 2, 3)))[0]:
        pix = np.nonzero(wl[l] > 0)[0]
        R = []
        for I in 'HL':                       # tables._responses, the psi rotation and chi dressing on the device
            Fp0, Fx0 = (xp.asarray(v) for v in Fplus_Fcross(I, th[pix], ph[pix], np.zeros(pix.size)))
            Fp = Fp0[:, None] * c2p[None, :] - Fx0[:, None] * s2p[None, :]
            Fx = Fp0[:, None] * s2p[None, :] + Fx0[:, None] * c2p[None, :]
            R.append(Ap[None, None, :] * Fp[:, :, None] + 1j * cgx[None, None, :] * Fx[:, :, None])
        base = (xp.asarray(np.log(wl[l, pix]))[:, None, None] + xp.asarray(np.log(wq))[None, None, :]
                - (s_H * xp.abs(R[0])**2 + s_L * xp.abs(R[1])**2)).ravel()
        c1, c2 = xp.conj(R[0]).ravel(), xp.conj(R[1]).ravel()
        del R
        if ker is not None:                  # rows of a reduction run in parallel: 1024 rows, padded with base = -inf
            pad = (-base.size) % ROWS
            base = xp.concatenate([base, xp.full(pad, -np.inf)]).reshape(ROWS, -1)
            c1 = xp.concatenate([c1, xp.zeros(pad, c1.dtype)])
            c2 = xp.concatenate([c2, xp.zeros(pad, c2.dtype)])
        sel = np.argwhere(flags[l])
        for k, d in {(int(k), int(d)) for k, b, d in sel}:
            y = xp.abs(c1 * np.exp(delta[d] + 1j * phi[k]) + c2)
            bs = np.nonzero(flags[l, k, :, d])[0]
            if ker is None:
                for b in bs:
                    cb = 2.0 * np.exp(beta[b] / 2)
                    ub = base + cb * y
                    top = ub.max()
                    T[l, k, b, d] = top + np.log((np.exp(ub - top) * i0e(cb * y)).sum())
            else:
                y = y.reshape(ROWS, -1)
                vals = []
                for b in bs:
                    cb = 2.0 * np.exp(beta[b] / 2)
                    top = ker['max'](base, y, cb, axis=1).max()
                    vals.append(top + xp.log(ker['sum'](base, y, cb, top, axis=1).sum()))
                T[l, k, bs, d] = xp.stack(vals).get()
            done += bs.size
    return T, done


def noise_mean(V_sum, keys, s_ref, w):
    """the noise mean of the truncated series from the diagonal of V summed over the delays.  For data rescaled to the
    model, E|z_H|^2 = ubar (1 + w), E|z_L|^2 = ubar (1 - w), and alpha = sbar / ubar, E|X|^2n = n! (ubar A)^n with
    A = (1 + w)|R_H|^2 + (1 - w)|R_L|^2, so with W = e^{l_n} V at s_ref = sbar
        E sum_{n <= N} alpha^n K_n = sum_{n <= N} sum_a C(n, a) (1 + w)^a (1 - w)^(n - a) sum_tau W^(n)_aa
                                   = < sum_{n <= N} e^{-sbar A} (sbar A)^n / n! >  (1 for the full series)"""
    terms = []
    for j, (n, a, b) in enumerate(keys):
        if a == b:
            terms.append(gammaln(n + 1.0) - gammaln(a + 1.0) - gammaln(n - a + 1.0) + a * np.log1p(w)
                         + (n - a) * np.log1p(-w) + ln_scale(n, s_ref) + np.log(V_sum[j]))
    terms = np.array(terms)
    top = terms.max()
    return float(np.exp(top) * np.exp(terms - top).sum())


def out_name(sbar):
    return 'bd_n%03d_s%09.3f.npz' % (N_MAX, sbar)


def save_v(stem, V, keys, tau, sbar, w, pix):
    """V of one node on the data lags, rows (n, a, b), b <= a, float64, and its l_n per row (W = e^{l_n} V), from
    the sky pixels pix = (lo, hi); written under a temporary name and renamed"""
    n_key = np.array([k[0] for k in keys], float)
    np.save(str(stem) + '.tmp.npy', np.ascontiguousarray(V.T))
    np.savez(str(stem) + '.meta.npz', tau=tau, keys=np.array(keys, np.int16), ln_scale=ln_scale(n_key, sbar),
             s_node=float(sbar), asym=float(w), s_H=float(sbar * (1 + w)), s_L=float(sbar * (1 - w)), n_max=N_MAX,
             nside=NSIDE, pix=np.array(pix))
    os.replace(str(stem) + '.tmp.npy', str(stem) + '.V.npy')


def load_v(stem, sbar, w):
    m = np.load(str(stem) + '.meta.npz')
    if abs(float(m['s_node']) - sbar) > 1e-12 * sbar or abs(float(m['asym']) - w) > 1e-12:
        raise ValueError('%s was built for s %g, w %g' % (stem, float(m['s_node']), float(m['asym'])))
    return m['tau'], [tuple(int(v) for v in k) for k in m['keys']], np.load(str(stem) + '.V.npy').T, tuple(m['pix'])


def part_range(k, K, npix):
    return (k * npix) // K, ((k + 1) * npix) // K


def load_group(stems, sbar, w):
    """V of a group from kept files: whole-sky files, or the K slices of every node (summed in slice order);
    None if neither is complete"""
    npix = hp.nside2npix(NSIDE)
    if all(Path(str(st) + '.V.npy').exists() for st in stems):
        got = [load_v(st, s, w) for st, s in zip(stems, sbar)]
        if all(g[3] == (0, npix) for g in got):
            return got[0][0], got[0][1], np.stack([g[2] for g in got])
    parts = sorted(Path(stems[0]).parent.glob(Path(stems[0]).name + '.part*of*.V.npy'))
    for K in sorted({int(str(f).split('of')[-1].split('.')[0]) for f in parts}):
        names = [[Path('%s.part%dof%d' % (st, k, K)) for k in range(K)] for st in stems]
        if all(Path(str(f) + '.V.npy').exists() for row in names for f in row):
            V, tau, keys = [], None, None
            for row, s in zip(names, sbar):
                acc = None
                for k, f in enumerate(row):
                    tau, keys, Vk, pix = load_v(f, s, w)
                    if pix != part_range(k, K, npix):
                        raise ValueError('%s holds pixels %s, not slice %d of %d' % (f, pix, k, K))
                    acc = Vk.copy() if acc is None else acc + Vk
                V.append(acc)
            return tau, keys, np.stack(V)
    return None


def sweep_in_slices(sbar, w, kdir, K, chunk):
    """the whole sky in K slices, one after the other, for a group of nodes; after each slice the running sum of V of
    all nodes is written to one checkpoint file in kdir, under a new name, and a small meta file (renamed into place)
    names it and the number of slices done; a run that finds the meta file resumes after that slice, so a kill loses
    at most one slice"""
    npix = hp.nside2npix(NSIDE)
    base = kdir / ('ratio_n%03d_w%+.4f_ck%d_s%09.3f-%09.3f' % (N_MAX, w, len(sbar), sbar[0], sbar[-1]))
    meta = Path(str(base) + '.meta.npz')
    V, done, tau, keys = None, 0, None, None
    if meta.exists():
        m = np.load(meta)
        if not (np.array_equal(m['sbar'], sbar) and float(m['asym']) == w and int(m['K']) == K):
            raise ValueError('%s belongs to another sweep' % meta)
        done = int(m['done'])
        V = np.load(kdir / str(m['ckfile']))
        tau, keys = m['tau'], [tuple(int(v) for v in k) for k in m['keys']]
        print('resuming after slice %d of %d from %s' % (done, K, m['ckfile']), flush=True)
    for k in range(done, K):
        t0 = time.time()
        pix = part_range(k, K, npix)
        tau, keys, Vk = w_sweep(sbar * (1 + w), sbar * (1 - w), n_max=N_MAX, nside=NSIDE, chunk=chunk, pix=pix,
                                verbose=True)
        V = Vk if V is None else V + Vk
        del Vk
        name = '%s.ck%02d.npy' % (base.name, k + 1)
        np.save(kdir / (name[:-4] + '.tmp.npy'), V)
        os.replace(kdir / (name[:-4] + '.tmp.npy'), kdir / name)
        np.savez(str(base) + '.meta.tmp.npz', sbar=sbar, asym=w, K=K, done=k + 1, ckfile=name, tau=tau,
                 keys=np.array(keys, np.int16))
        os.replace(str(base) + '.meta.tmp.npz', meta)
        old = kdir / ('%s.ck%02d.npy' % (base.name, k))
        if k > 0 and old.exists():
            old.unlink()
        print('slice %d of %d (pixels %d..%d) %.0f s, running sum written to %s' % (k + 1, K, *pix, time.time() - t0,
                                                                                   name), flush=True)
    return tau, keys, V, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out_dir')
    ap.add_argument('--asym', type=float, required=True)
    ap.add_argument('--nodes', type=int, nargs='*')
    ap.add_argument('--keep-w', default=None)
    ap.add_argument('--per-sweep', type=int, default=53, help='nodes per sky sweep (GPU memory: 0.21 GB each)')
    ap.add_argument('--chunk', type=int, default=8, help='sky pixels per GPU batch')
    ap.add_argument('--part', type=int, nargs=2, metavar=('k', 'K'), help='sweep slice k of K of the sky only')
    ap.add_argument('--slices', type=int, default=0, help='sweep the sky in this many slices, checkpointed (--keep-w)')
    ap.add_argument('--repair-zcut', type=float, default=72.0,
                    help='repair the entries reachable below this single-detector |z|^2/u (0: no repair)')
    ap.add_argument('--repair-wnet', type=float, default=84.0,
                    help='the network loudness cut of the search on w_H + w_L (inf: none)')
    ap.add_argument('--repair-gap', type=float, default=18.0,
                    help='repair where T_abs - T exceeds this (the cancellation error is ~1e-16 e^gap)')
    ap.add_argument('--repair-tol', type=float, default=1e-6,
                    help='for w_H + w_L above %g, keep the series only where the truncation is proven below this'
                         % W_SERIES)
    ap.add_argument('--repair-dry', action='store_true', help='count the entries stage R would replace; write nothing')
    args = ap.parse_args()
    import cupy as cp
    w = args.asym
    if not abs(w) < 1:
        raise SystemExit('|w| must be < 1')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kdir = Path(args.keep_w) if args.keep_w else None
    if kdir:
        kdir.mkdir(parents=True, exist_ok=True)
    if (args.part or args.slices) and not kdir:
        raise SystemExit('--part and --slices write V: give --keep-w')
    s_all = np.load(ROOT / 'data' / 'tables' / 's_grid.npz')['s_nodes']
    pick = args.nodes if args.nodes else list(range(s_all.size))
    todo = [i for i in pick if not (out / out_name(s_all[i])).exists()]
    print('w = %+.4f: %d nodes asked, %d to build, %d per sweep, chunk %d'
          % (w, len(pick), len(todo), args.per_sweep, args.chunk), flush=True)
    for g0 in range(0, len(todo), args.per_sweep):
        group = todo[g0:g0 + args.per_sweep]
        sbar = s_all[group]
        stems = [kdir / ('ratio_n%03d_w%+.4f_s%09.3f' % (N_MAX, w, s)) for s in sbar] if kdir else None
        t0 = time.time()
        if args.part:
            k, K = args.part
            pix = part_range(k, K, hp.nside2npix(NSIDE))
            names = [Path('%s.part%dof%d' % (st, k, K)) for st in stems]
            if all(Path(str(f) + '.V.npy').exists() for f in names):
                print('nodes %s: slice %d of %d exists' % (group, k, K), flush=True)
                continue
            tau, keys, V = w_sweep(sbar * (1 + w), sbar * (1 - w), n_max=N_MAX, nside=NSIDE, chunk=args.chunk, pix=pix,
                                   verbose=True)
            for f, s, Vj in zip(names, sbar, V):
                save_v(f, Vj, keys, tau, s, w, pix)
            print('nodes %s: W sweep of pixels %d..%d %.0f s, V written' % (group, *pix, time.time() - t0), flush=True)
            continue
        got = load_group(stems, sbar, w) if stems else None
        if got is not None:
            tau, keys, V = got
            print('nodes %s: V read from %s' % (group, kdir), flush=True)
        else:
            if args.repair_dry:
                raise SystemExit('--repair-dry reads kept sweeps (--keep-w) and starts none: nodes %s have none' % group)
            ck = None
            if args.slices:
                tau, keys, V, ck = sweep_in_slices(sbar, w, kdir, args.slices, args.chunk)
            else:
                tau, keys, V = w_sweep(sbar * (1 + w), sbar * (1 - w), n_max=N_MAX, nside=NSIDE, chunk=args.chunk,
                                       verbose=True)
            print('nodes %s: W sweep %.0f s' % (group, time.time() - t0), flush=True)
            if stems:
                for st, s, Vj in zip(stems, sbar, V):
                    save_v(st, Vj, keys, tau, s, w, (0, hp.nside2npix(NSIDE)))
            if ck is not None:                      # the per-node files now hold the whole sky
                m = np.load(ck)
                (kdir / str(m['ckfile'])).unlink()
                ck.unlink()
        cp.get_default_memory_pool().free_all_blocks()
        gr = grids(tau)
        for i, s, Vj in zip(group, sbar, V):
            t1 = time.time()
            E = noise_mean(Vj.sum(axis=0), keys, s, w)
            nonpos = np.zeros(gr['tau_idx'].size, np.int64)
            T = cp.asnumpy(t_from_v(Vj, s, gr['delta'], gr['beta'], gr['phi'], n_max=N_MAX, xp=cp, nonpos=nonpos,
                                    out_dtype=np.float64))
            t2, n_rep, counts = time.time(), 0, None
            if args.repair_zcut > 0:
                flags, counts = flag_entries(T, Vj, s * (1 + w), s * (1 - w), s, gr['beta'], gr['delta'],
                                             args.repair_zcut, args.repair_gap, n_max=N_MAX, nside=NSIDE, xp=cp,
                                             wnet=args.repair_wnet, tol=args.repair_tol)
                if args.repair_dry:
                    print('node %2d s %10.5f: stage R answers for %d entries; cancellation %d, truncation alone %d, '
                          'to replace %d' % (i, s, counts['need'], counts['cancellation'], counts['truncation'],
                                             counts['flagged']), flush=True)
                    continue
                T, n_rep = repair(T, flags, s * (1 + w), s * (1 - w), gr['beta'], gr['delta'], gr['phi'], n_max=N_MAX,
                                  nside=NSIDE, xp=cp)
                del flags
            elif args.repair_dry:
                raise SystemExit('--repair-dry needs --repair-zcut > 0')
            T = T.astype(np.float32)
            if not np.isfinite(T).all():
                raise FloatingPointError('node %d: T non-finite' % i)
            path = out / out_name(s)
            tmp = str(path)[:-4] + '.tmp.npz'
            done = args.repair_zcut > 0
            np.savez(tmp, T=T, beta=gr['beta'], delta=gr['delta'], tau_idx=gr['tau_idx'], lnq=gr['lnq'], alpha=ALPHA,
                     s_node=float(s), u_node=float(s), n_max=N_MAX, nphif=gr['nphif'], norm='taylor_off1',
                     asym=float(w), s_H=float(s * (1 + w)), s_L=float(s * (1 - w)),
                     repair_zcut=float(args.repair_zcut), repair_wnet=float(args.repair_wnet) if done else 0.0,
                     repair_gap=float(args.repair_gap) if done else np.inf,
                     repair_tol=float(args.repair_tol) if done else np.inf)
            os.replace(tmp, path)
            del T
            cp.get_default_memory_pool().free_all_blocks()
            full = nonpos == (N_MAX + 1) * gr['phi'].size * gr['delta'].size
            print('node %2d s %10.5f: T %.0f s (repair of %d entries %.0f s%s), noise mean of the truncated series '
                  '%.15f (deficit %.3e), G_n <= 0 everywhere in lags %s, at %d other points -> %s'
                  % (i, s, time.time() - t1, n_rep, time.time() - t2,
                     '' if counts is None else '; of %d answered for, cancellation %d, truncation alone %d'
                     % (counts['need'], counts['cancellation'], counts['truncation']), E, 1.0 - E,
                     [int(v) for v in gr['tau_idx'][full]], int(nonpos[~full].sum()), path.name), flush=True)
        del V
    print('done', flush=True)


if __name__ == '__main__':
    main()
