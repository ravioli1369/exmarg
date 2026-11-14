"""Test 3 of docs/real_data.md: on coloured Gaussian noise conditioned by exmarg.strain, the likelihood ratio averaged
over the population has mean 1 with the correction factor of the spectrum estimate, and not without it (the test of
notebooks/07_estimated_psd.ipynb, through the full path from strain).  Needs the redshift tables; meant for a GPU.

    python scripts/validate/conditioned_noise.py tables/redshift gpu [--epochs 64] [--N 32] [--seed 1]
                                                 [--stretch 8] [--no-resident] [--out lambda_bar.npz]

The noise of tests/strain_sim.py (O4 curve, wall below 15 Hz, lines, L better than H by 20 percent in amplitude
except near 200 Hz) is simulated in stretches of --stretch epochs (--epochs is rounded up to a multiple) and
conditioned six ways on the same noise: with the true spectra (the reference), with the mean and the median of the
periodograms, each with and without its correction factor (handed over on the curve of the neighbouring epochs), and
with the mean handed over on the epoch's own curve (reference='epoch').  Every segment is analysed by pe_redshift on 120
equal-mass templates from 20 to 400 Msun (detector frame) and Lambda is averaged over the population of notebook 7.
For each variant the table gives the mean of Lambda_bar - 1, its standard error from the scatter of the epoch means
(the segments of an epoch share their spectrum estimate), the number of segments, pass if |mean - 1| < 3 sigma, and
the paired difference to the reference (same noise), which is far more precise.  Without the factor the first-order
shift of the noise power is E[1/g] - 1 = 1/(N-1) for the mean, about 2/N for the median.  Exit status 1 when the
reference or a corrected variant fails.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import exmarg.search as search
from exmarg import strain
import strain_sim as sim

T, FS = 8.0, sim.FS
MASSES = np.geomspace(20.0, 400.0, 120)
VARIANTS = {'true spectrum': dict(psd=(sim.psd('H'), sim.psd('L'))),
            'mean': dict(method='mean'), 'mean, c = 1': dict(method='mean', correct=False),
            'median': dict(method='median'), 'median, c = 1': dict(method='median', correct=False),
            'mean, own epoch': dict(method='mean', reference='epoch')}
MUST_PASS = ('true spectrum', 'mean', 'median', 'mean, own epoch')


def population_weights():
    """ln of the population weights on the (m1, z) grid of pe_redshift, as in notebook 7"""
    pop = np.load(os.path.join(ROOT, 'src', 'exmarg', 'data', 'demo_population.npz'))
    m1 = np.geomspace(4.0, 200.0, 150)
    z = np.linspace(0.02, 10.0, 150)
    p_m1 = np.interp(m1, pop['m1'], pop['p_m1'], left=0.0, right=0.0)
    p_z = np.interp(z, pop['z'], pop['p_z'], left=0.0, right=0.0)
    M_det = 2.0 * m1[:, None] * (1.0 + z[None, :])
    w = (p_m1[:, None] * p_z[None, :] * np.gradient(m1)[:, None] * np.gradient(z)[None, :]
         * ((M_det >= 20.0) & (M_det <= 400.0)))
    return np.log(np.maximum(w / w.sum(), 1e-300))


def lambda_bar(seg, store, device, ln_w, resident):
    """the population-averaged likelihood ratio of every segment of one epoch"""
    bank = search.mass_bank(MASSES, seg, device)
    r = search.pe_redshift(seg, bank, store, m1_range=(4.0, 200.0), nm1=150, zmax=10.0, nz=150, resident=resident,
                           mdet_range=(20.0, 400.0))
    L = np.where(np.isfinite(r['ln_lambda']), r['ln_lambda'], -np.inf)
    return np.exp(logsumexp(L + ln_w[None], axis=(1, 2)))


def summary(res, N):
    """prints the table; returns the names of the variants that fail"""
    ref = np.array([e.mean() for e in res['true spectrum']])
    failed = []
    print('\nN = %d, %d epochs, %d segments per variant' % (N, ref.size, sum(e.size for e in res['true spectrum'])))
    print('%-15s %10s %9s %7s %6s %10s   %-22s %s' % ('variant', 'mean - 1', 'sigma', 'n_seg', 'pass', 'factor c',
                                                    'paired minus true', 'expected'))
    for name, epochs in res.items():
        allv = np.concatenate(epochs)
        means = np.array([e.mean() for e in epochs])
        se = means.std(ddof=1) / np.sqrt(means.size)
        ok = abs(allv.mean() - 1.0) < 3.0 * se
        d = means - ref
        method = VARIANTS[name].get('method')
        c = strain.correction(N, method) if (method and VARIANTS[name].get('correct', True)) else 1.0
        expect = 'pass' if name in MUST_PASS else 'fail (noise power %+.3f)' % (strain.correction(N, method) - 1.0)
        print('%-15s %+10.4f %9.4f %7d %6s %10.4f   %+.4f +- %.4f        %s'
              % (name, allv.mean() - 1.0, se, allv.size, 'yes' if ok else 'NO', c, d.mean(),
                 d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0.0, expect))
        if not ok:
            failed.append(name)
    return failed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('store', help='folder of the redshift tables')
    p.add_argument('device', choices=('gpu', 'cpu'))
    p.add_argument('--epochs', type=int, default=64)
    p.add_argument('--N', type=int, default=32, help='an epoch is N + 1 segments')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--stretch', type=int, default=8, help='epochs per simulated stretch (at least 2)')
    p.add_argument('--no-resident', action='store_true', help='stream the tables (GPUs with less than ~48 GB)')
    p.add_argument('--out', help='save Lambda_bar of every variant and segment to this .npz')
    a = p.parse_args(argv)
    per = max(2, a.stretch)
    n_stretch = -(-a.epochs // per)
    resident = a.device == 'gpu' and not a.no_resident
    store = search.load_redshift_store(a.store, pinned=(a.device == 'gpu'), mmap=(a.device == 'cpu'))
    ln_w = population_weights()
    res = {name: [] for name in VARIANTS}
    start = time.time()
    for s in range(n_stretch):
        n = int(round((per * (a.N + 1) * T + 2 * T) * FS))
        x = {I: sim.noise(I, n, [a.seed, s, j]) for j, I in enumerate('HL')}
        for name, opts in VARIANTS.items():
            eps = strain.condition(x['H'], x['L'], FS, 1.0e9 + 1.0e5 * s, T=T, N=a.N, notches=sim.NOTCHES,
                                   seed=a.seed, **opts)
            res[name] += [lambda_bar(ep, store, a.device, ln_w, resident) for ep in eps]
        print('stretch %d of %d done (%.0f s)' % (s + 1, n_stretch, time.time() - start), flush=True)
    failed = summary(res, a.N)
    if a.out:
        np.savez(a.out, **{name.replace(' ', '').replace(',', '_').replace('=', ''): np.stack(v)
                           for name, v in res.items()})
    bad = [f for f in failed if f in MUST_PASS]
    print('\n%s' % ('FAIL: ' + ', '.join(bad) if bad else 'reference and corrected variants pass'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
