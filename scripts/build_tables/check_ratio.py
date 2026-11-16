"""Checks of a store built by build_ratio.py, at the settings of the published tables (n_max 120, nside 32), with code
that shares nothing with the builder but the definition of the table (exmarg.detectors, the psi/chi rule of
exmarg.tables._quad and the grids of the store files):

  files     every bd_*.npz of the store: members and shapes as the published file of the same name (plus asym, s_H,
            s_L and the cuts of stage R: repair_zcut, repair_wnet, repair_gap, repair_tol, which must be those given
            with --zcut and --wnet), beta, delta, tau_idx, lnq equal to it, alpha to 1e-15, s_H = s_node (1 + asym),
            s_L = s_node (1 - asym), T finite;
  bessel    at random table points (tau, phi, beta, Delta) of the chosen nodes, T against the quadrature with no
            series, ln < e^{-s_H |R_H|^2 - s_L |R_L|^2} I0(2 e^{beta/2} |conj(R_H) e^{Delta + i phi} + conj(R_L)|) > in the
            lag (float64 responses from Fplus_Fcross at each psi; each pixel's delay shared linearly between the two
            nearest of the 257 fine delays, each fine delay in its nearest data lag).  Without --reach, only points
            where the series to n = 120 has converged: v = e^{beta/2} (e^Delta + 1) <= vmax bounds the Bessel argument
            by 2 v (|R_I| <= 1), and the tail after n = 120 is below e^-50 of I0(2v) for v <= 60.  Half the points have
            v in [20, vmax] (when there are such points).  --reach draws from the entries that the search depends on
            (reach_mask: every raw entry that enters a read of data below the loudness cuts |z_I|^2 / u_I < zcut and
            w_H + w_L <= wnet, through the bilinear read and the removal of the interpolation bias), at any v: a
            store that went through stage R must agree with the quadrature at all of them;
  noise     < P(Poisson(D) <= 120) > over the sky, psi, chi, D = s_H |R_H|^2 + s_L |R_L|^2: the noise mean of the
            truncated series, to compare with the value build_ratio.py prints from the diagonal of W.

    python scripts/build_tables/check_ratio.py <store_dir> --published DIR [--nodes i j ...] [--points 256]
           [--vmax 60] [--seed 0] [--workers 8] [--compare DIR ...] [--reach] [--zcut 72] [--wnet 84] [--noise]
           [--dump PREFIX]
--compare evaluates other stores (e.g. the published one at w = 0) at the same points.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import healpy as hp
from scipy.special import gammaincc, i0e

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from exmarg.detectors import Fplus_Fcross, c, x
from exmarg.tables import _quad

N_MAX, NSIDE, N_TAU, T_STEP = 120, 32, 257, 1.0 / 4096
MEMBERS = ('T', 'beta', 'delta', 'tau_idx', 'lnq', 'alpha', 's_node', 'u_node', 'n_max', 'nphif', 'norm')
REPAIR = ('repair_zcut', 'repair_wnet', 'repair_gap', 'repair_tol')


def check_files(store, published, zcut, wnet):
    """members, shapes and grids of every file against the published file of the same name; T finite; the cuts of
    stage R recorded in the file equal to zcut, wnet"""
    files = sorted(glob.glob(os.path.join(store, 'bd_*.npz')))
    bad = []
    for f in files:
        h, p = np.load(f), np.load(os.path.join(published, os.path.basename(f)))
        if set(h.files) != set(MEMBERS) | {'asym', 's_H', 's_L'} | set(REPAIR):
            bad.append((f, 'members %s' % sorted(h.files)))
            continue
        if float(h['repair_zcut']) != zcut or float(h['repair_wnet']) != wnet:
            bad.append((f, 'repaired for zcut %g, wnet %g' % (float(h['repair_zcut']), float(h['repair_wnet']))))
        for k in MEMBERS:
            if h[k].shape != p[k].shape or h[k].dtype != p[k].dtype:
                bad.append((f, '%s: %s %s, published %s %s' % (k, h[k].shape, h[k].dtype, p[k].shape, p[k].dtype)))
        for k in ('beta', 'delta', 'tau_idx', 'lnq', 'n_max', 'nphif', 'norm'):
            if not np.array_equal(h[k], p[k]):
                bad.append((f, '%s differs from the published file' % k))
        # alpha: the published files themselves differ by 1 ulp in two entries (np.geomspace of two numpy versions)
        if not np.allclose(h['alpha'], p['alpha'], rtol=1e-15, atol=0):
            bad.append((f, 'alpha differs from the published file by more than 1e-15'))
        s, w = float(h['s_node']), float(h['asym'])
        if not (abs(s - float(p['s_node'])) <= 1e-12 * s and float(h['s_H']) == s * (1 + w)
                and float(h['s_L']) == s * (1 - w) and float(h['u_node']) == s):
            bad.append((f, 'nodes s %r, u %r, s_H %r, s_L %r, w %r' % (s, float(h['u_node']), float(h['s_H']),
                                                                       float(h['s_L']), w)))
        if not np.isfinite(h['T']).all():
            bad.append((f, 'T has %d non-finite entries' % int((~np.isfinite(h['T'])).sum())))
    return files, bad


def lag_pixels(tau_idx):
    """per data lag: the pixels with a share in it and their weights (1/npix per pixel in total)"""
    npix = hp.nside2npix(NSIDE)
    th, ph = hp.pix2ang(NSIDE, np.arange(npix))
    dx = x['H'] - x['L']
    t = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) @ dx / c
    tf = np.linspace(-1.0, 1.0, N_TAU) * np.linalg.norm(dx) / c
    g = (t - tf[0]) / (tf[1] - tf[0])
    j0 = np.clip(np.floor(g).astype(int), 0, N_TAU - 2)
    lag_of_fine = np.argmin(np.abs(tau_idx[:, None] * T_STEP - tf[None, :]), axis=0)
    wt = np.zeros((tau_idx.size, npix))
    np.add.at(wt, (lag_of_fine[j0], np.arange(npix)), (1 - (g - j0)) / npix)
    np.add.at(wt, (lag_of_fine[j0 + 1], np.arange(npix)), (g - j0) / npix)
    return th, ph, wt


def responses(th, ph):
    """R_I = (1 + chi^2)/2 F+_I(psi) + i chi Fx_I(psi) on the quadrature of the tables, complex128, (pix, psi, chi)"""
    psi, cg, wq = _quad(N_MAX, None, None)
    out = []
    for I in 'HL':
        Fp, Fx = Fplus_Fcross(I, th[:, None], ph[:, None], psi[None, :])
        out.append(0.5 * (1 + cg**2)[None, None, :] * Fp[:, :, None] + 1j * cg[None, None, :] * Fx[:, :, None])
    return out[0], out[1], wq


def _lag_job(args):
    """ln of the quadrature at the points of one lag: (s_H, s_L, th, ph, weights, [(phi, beta, Delta), ...])"""
    s_H, s_L, th, ph, wpix, pts = args
    R1, R2, wq = responses(th, ph)
    base = np.log(wpix)[:, None, None] + np.log(wq)[None, None, :] - (s_H * np.abs(R1)**2 + s_L * np.abs(R2)**2)
    out = []
    for phi, beta, dl in pts:
        arg = 2 * np.exp(beta / 2) * np.abs(np.conj(R1) * np.exp(dl + 1j * phi) + np.conj(R2))
        e = base + arg + np.log(i0e(arg))
        top = e.max()
        out.append(top + np.log(np.exp(e - top).sum()))
    return out


def reach_mask(beta, delta, s_H, s_L, zcut, wnet):
    """the raw entries (beta, Delta) that enter a read of data below the loudness cuts, w_L = e^beta / s_L < zcut,
    w_H = e^(beta + 2 Delta) / s_H < zcut, w_L + w_H <= wnet.  A datum is read from the four corners of its grid cell
    (search/gather.py), so a cell is read when its lower corner is below the cuts (the cuts only tighten upwards);
    the table read is the unbiased one, each entry of which is the raw entry and its two neighbours along beta and
    along Delta (search/store.py, unbias)"""
    wL = np.exp(beta)[:, None] / s_L
    wH = np.exp(beta[:, None] + 2 * delta[None, :]) / s_H
    cell = (wL < zcut) & (wH < zcut) & (wL + wH < wnet)
    read = cell.copy()
    read[1:] |= cell[:-1]
    read[:, 1:] |= cell[:, :-1]
    read[1:, 1:] |= cell[:-1, :-1]
    raw = read.copy()
    raw[1:] |= read[:-1]
    raw[:-1] |= read[1:]
    raw[:, 1:] |= read[:, :-1]
    raw[:, :-1] |= read[:, 1:]
    return raw


def draw_points(h, wt, n, vmax, rng, reach=None):
    """random grid indices (lag, phi, beta, Delta): lags with sky, v = e^{beta/2} (e^Delta + 1) <= vmax, inside the
    mask reach (beta, Delta) if given; half of them with v >= 20 when there are such points"""
    v = np.exp(h['beta'][:, None] / 2) * (np.exp(h['delta'][None, :]) + 1)
    allowed = (v <= vmax) if reach is None else (v <= vmax) & reach
    ok_all, ok_loud = np.argwhere(allowed), np.argwhere(allowed & (v >= 20.0))
    lags = np.nonzero(wt.sum(axis=1) > 0)[0]
    pts = []
    for k in range(n):
        ok = ok_loud if k % 2 and len(ok_loud) else ok_all
        b, d = ok[rng.integers(ok.shape[0])]
        pts.append((rng.choice(lags), rng.integers(h['T'].shape[1]), b, d))
    return pts


def check_bessel(path, n, vmax, seed, workers, compare=(), zcut=None, wnet=np.inf):
    h = np.load(path)
    s_H, s_L = (float(h['s_H']), float(h['s_L'])) if 's_H' in h.files else (float(h['s_node']),) * 2   # published: w = 0
    th, ph, wt = lag_pixels(h['tau_idx'])
    phi_g = np.arange(h['T'].shape[1]) * (2 * np.pi / 96)
    reach = None if zcut is None else reach_mask(h['beta'], h['delta'], s_H, s_L, zcut, wnet)
    pts = draw_points(h, wt, n, vmax if zcut is None else np.inf, np.random.default_rng(seed), reach)
    by_lag = {}
    for i, (l, k, b, d) in enumerate(pts):
        by_lag.setdefault(l, []).append(i)
    jobs = [(s_H, s_L, th[wt[l] > 0], ph[wt[l] > 0], wt[l][wt[l] > 0],
             [(phi_g[pts[i][1]], h['beta'][pts[i][2]], h['delta'][pts[i][3]]) for i in idx]) for l, idx in by_lag.items()]
    ref = np.empty(n)
    with Pool(workers) as pool:
        for idx, vals in zip(by_lag.values(), pool.map(_lag_job, jobs)):
            ref[idx] = vals
    T = h['T']
    tab = np.array([T[l, k, b, d] for (l, k, b, d) in pts], float)
    others = [np.array([np.load(os.path.join(cdir, os.path.basename(path)))['T'][l, k, b, d] for (l, k, b, d) in pts],
                       float) for cdir in compare]
    v = np.array([np.exp(h['beta'][b] / 2) * (np.exp(h['delta'][d]) + 1) for (l, k, b, d) in pts])
    return pts, v, tab, ref, others


def _noise_job(args):
    """sum over the pixels lo..hi of the quadrature of Q(121, s_H |R_H|^2 + s_L |R_L|^2)"""
    s_H, s_L, lo, hi = args
    th, ph = hp.pix2ang(NSIDE, np.arange(lo, hi))
    R1, R2, wq = responses(th, ph)
    return float((gammaincc(N_MAX + 1, s_H * np.abs(R1)**2 + s_L * np.abs(R2)**2) * wq[None, None, :]).sum())


def noise_quadrature(s_H, s_L, workers, chunk=64):
    """< P(Poisson(D) <= 120) > = < Q(121, D) > over the whole sky, psi and chi"""
    npix = hp.nside2npix(NSIDE)
    with Pool(workers) as pool:
        parts = pool.map(_noise_job, [(s_H, s_L, lo, min(lo + chunk, npix)) for lo in range(0, npix, chunk)])
    return float(np.sum(parts)) / npix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('store')
    ap.add_argument('--published', required=True)
    ap.add_argument('--nodes', type=int, nargs='*', default=[])
    ap.add_argument('--points', type=int, default=256)
    ap.add_argument('--vmax', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--compare', nargs='*', default=[])
    ap.add_argument('--noise', action='store_true', help='also the noise mean by quadrature of the Poisson probability')
    ap.add_argument('--dump', default=None, help='write every point to <dump>_node<i>.npz')
    ap.add_argument('--reach', action='store_true', help='only entries that a read of data below the loudness cuts '
                                                         'depends on, at any v')
    ap.add_argument('--zcut', type=float, default=72.0, help='the single-detector loudness cut on |z|^2 / u')
    ap.add_argument('--wnet', type=float, default=84.0, help='the network loudness cut on w_H + w_L')
    args = ap.parse_args()
    if not os.path.samefile(args.store, args.published):
        files, bad = check_files(args.store, args.published, args.zcut, args.wnet)
        print('%s: %d files, %d problems' % (args.store, len(files), len(bad)), flush=True)
        for f, why in bad:
            print('  %s: %s' % (os.path.basename(f), why))
    s_all = np.load(ROOT / 'data' / 'tables' / 's_grid.npz')['s_nodes']
    for i in args.nodes:
        path = os.path.join(args.store, 'bd_n%03d_s%09.3f.npz' % (N_MAX, s_all[i]))
        pts, v, tab, ref, others = check_bessel(path, args.points, args.vmax, args.seed + i, args.workers, args.compare,
                                                args.zcut if args.reach else None, args.wnet)
        if args.dump:
            np.savez('%s_node%02d.npz' % (args.dump, i), pts=np.array(pts), v=v, tab=tab, ref=ref,
                     others=np.array(others, float).reshape(len(others), len(pts)), compare=np.array(args.compare, dtype=str))
        dif = tab - ref
        j = int(np.argmax(np.abs(dif)))
        hi = v >= 20
        print('node %2d s %9.4f: %d points, |T - quadrature| max %.3e (rms %.3e; v < 20: max %.3e, v >= 20: max %.3e), '
              'largest at (lag %d, phi %d, beta %d, Delta %d), v %.1f, T %.6f'
              % (i, s_all[i], len(pts), np.abs(dif).max(), np.sqrt(np.mean(dif**2)), np.abs(dif[~hi]).max(),
                 np.abs(dif[hi]).max() if hi.any() else np.nan, *pts[j], v[j], tab[j]), flush=True)
        for cdir, o in zip(args.compare, others):
            do = o - ref
            jo = int(np.argmax(np.abs(do)))
            print('    %s at the same points: max %.3e (rms %.3e; v < 20: max %.3e, v >= 20: max %.3e), largest at '
                  '(lag %d, phi %d, beta %d, Delta %d), v %.1f' % (cdir, np.abs(do).max(), np.sqrt(np.mean(do**2)),
                                                                   np.abs(do[~hi]).max(), np.abs(do[hi]).max() if hi.any()
                                                                   else np.nan, *pts[jo], v[jo]), flush=True)
        if args.noise:
            h = np.load(path)
            E = noise_quadrature(*((float(h['s_H']), float(h['s_L'])) if 's_H' in h.files else (float(h['s_node']),) * 2),
                                 args.workers)
            print('    noise mean by quadrature of P(Poisson(D) <= %d): %.15f (deficit %.3e)' % (N_MAX, E, 1 - E),
                  flush=True)


if __name__ == '__main__':
    main()
