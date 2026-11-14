"""Control 2 of docs/glitch_likelihood.md (Sec. 7), the marginal identity

    E_{d_H ~ noise}[ Lambda_S(d_H, d_L) ] = Lambda_S^L(d_L)

on one simulated segment.  Needs the redshift tables; meant for a GPU.

    python scripts/validate/marginal_identity.py tables/redshift gpu 4000 1 [--pairs 6] [--snr-L 4] [--mass 60]
                                                 [--smax 1] [--chunk 8]

A merger of detector-frame total mass --mass (equal masses, no spin) is simulated with exmarg.pp.simulate at the
distance where its optimal SNR in L is --snr-L (weak by default).  d_L is kept fixed and the data of H are replaced
by n independent draws of Gaussian noise (noise only, the noise curve of the simulation).  Every realization goes
through pe._node_values with the store, on three templates around the injected mass, summing over every arrival time
(n_loud = 0), with the single-detector values of single=True.  For --pairs (node, template) pairs with
s <= --smax, spread over s, the mean over the realizations of exp(lamN) must equal exp(lamN_L) within three Monte
Carlo standard errors.  Both are normalized sums over the arrival times (the coherent one also over the lags of the
table), each divided by n_time; the mean over the realizations is divided by n.  lamN_L is the same in every
realization (d_L is fixed), up to the order of the floating-point sums on the GPU, and its spread is printed.

Exit status 1 when a pair fails or fewer pairs than requested were checked.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import exmarg.search as search
from exmarg.pp import simulate
from exmarg.search import pe

T, FMIN, FMAX = 8.0, 20.0, 512.0
YMIN = 0.1267


def injection(mass, snr_L, seed):
    """the simulated segment with its merger at the distance where the optimal SNR in L is snr_L, and that y"""
    draw = lambda rng: (mass, 1.0, 0.0)
    ref, _, _ = simulate(1, seed, draw, lambda rng: 1.0, T, FMIN, FMAX)        # the same draws for every y:
    zero, _, _ = simulate(1, seed, draw, lambda rng: 0.0, T, FMIN, FMAX)       # y enters only the signal
    h_L = ref['d_L'][0] - zero['d_L'][0]
    y = snr_L / np.sqrt(2.0 * np.sum(np.abs(h_L) ** 2 / ref['sig2']))
    seg, truth, snr = simulate(1, seed, draw, lambda rng: y, T, FMIN, FMAX)
    return seg, y, snr[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('store')
    ap.add_argument('device', choices=['cpu', 'gpu'])
    ap.add_argument('n', type=int, help='realizations of the noise of H')
    ap.add_argument('seed', type=int)
    ap.add_argument('--pairs', type=int, default=6)
    ap.add_argument('--snr-L', type=float, default=4.0)
    ap.add_argument('--mass', type=float, default=60.0)
    ap.add_argument('--smax', type=float, default=1.0)
    ap.add_argument('--chunk', type=int, default=8)
    a = ap.parse_args()

    seg1, y, snr_net = injection(a.mass, a.snr_L, a.seed)
    rng = np.random.default_rng(a.seed + 7919)
    sig2 = seg1['sig2']
    d_H = np.sqrt(sig2 / 2.0) * (rng.standard_normal((a.n, sig2.size)) + 1j * rng.standard_normal((a.n, sig2.size)))
    segments = dict(seg1, d_H=d_H, d_L=np.repeat(seg1['d_L'], a.n, axis=0))
    seg = search.read_segments(segments)
    store = search.load_redshift_store(a.store, pinned=a.device == 'gpu')
    bank = search.mass_bank(a.mass * np.array([0.8, 1.0, 1.25]), segments, a.device)
    print('injection: M_det %.1f, y = %.4f, optimal SNR %.2f in L and %.2f in H + L (H replaced by noise); '
          'templates M %s, u %s; %d realizations, n_time %d'
          % (a.mass, y, a.snr_L, snr_net, bank.M, np.array2string(bank.u, precision=3), a.n, bank.n_time), flush=True)

    geometry = pe._node_geometry(bank, store, YMIN)
    low = np.nonzero(store.s <= a.smax)[0]
    groups = [list(low[i:i + 6]) for i in range(0, low.size, 6)]
    sd = pe._single_setup(bank, seg, store, geometry)
    nn = store.s.size
    lam = np.empty((a.n, nn + 1, bank.M.size))
    lam_L = np.empty_like(lam)
    chunk = pe._fit_chunk(bank, store, a.chunk, 6, False)
    for ev in pe._chunks(seg, chunk):
        lamN, one = pe._node_values(bank, seg, ev, store, geometry, groups, 256, False, 0, 0, a.seed, sd)
        lam[ev.start:ev.stop], lam_L[ev.start:ev.stop] = lamN, one['lamN_L']
        print('  %d / %d realizations' % (ev.stop, a.n), flush=True)

    fin = np.isfinite(lam_L).all(axis=0)
    spread = np.abs(lam_L[:, fin] - lam_L[:1, fin]).max() if fin.any() else 0.0
    valid = geometry[0]
    cand = [(k, m) for k in low for m in range(bank.M.size) if valid[k, m] and np.isfinite(lam[:, k, m]).all()]
    cand.sort(key=lambda km: store.s[km[0]])
    pick = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, a.pairs).round().astype(int))] if cand else []
    print('\nlamN_L across the realizations: largest spread %.1e' % spread)
    print('denominators: both sums over the arrival times / n_time = %d (coherent: also over %d lags), mean over '
          'the realizations / n = %d' % (bank.n_time, np.asarray(store.tau_idx).size, a.n))
    print('%5s %10s %7s %14s %14s %12s %7s %9s  %s' % ('node', 's', 'M', 'mean exp(lamN)', 'exp(lamN_L)', 'std error',
                                                       'z', 'ratio', ''))
    failed = 0
    for k, m in pick:
        x = np.exp(lam[:, k, m])
        mean, se, target = x.mean(), x.std(ddof=1) / np.sqrt(a.n), np.exp(lam_L[0, k, m])
        z = (mean - target) / se
        ok = abs(z) <= 3.0
        failed += not ok
        print('%5d %10.4g %7.1f %14.8g %14.8g %12.4g %7.2f %9.6f  %s'
              % (k, store.s[k], bank.M[m], mean, target, se, z, mean / target, 'pass' if ok else 'FAIL'))
    print('\n%d pairs checked of %d requested (%d candidates with s <= %g), %d failed'
          % (len(pick), a.pairs, len(cand), a.smax, failed))
    sys.exit(1 if failed or len(pick) < a.pairs else 0)


if __name__ == '__main__':
    main()
