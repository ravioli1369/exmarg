"""The noise mean of the likelihood ratio read from the tables (Sec. 1 of docs/glitch_likelihood.md).  Needs the
redshift tables and a GPU.

    python scripts/validate/table_noise_mean.py <n segments> <seed> <smax> [unbiased]      (from the repository root)

On pure Gaussian noise in both detectors, for every node with s <= smax and three templates: the mean over the
segments of exp(lamN) (coherent, from the tables) and of the one-detector exp(lamN_H), exp(lamN_L), each minus 1 and
divided by s, with its standard error.  A likelihood ratio has unit mean under noise, so each must vanish.  The last
column is the mean of exp(lamN) - exp(lamN_H) - exp(lamN_L) + 1: the one-detector terms carry most of the scatter of
the coherent one and have unit mean by construction (tests/test_single.py), so this difference measures the error of
the tables far more precisely.  Measured on 1200 segments: +7.7e-4 s (40 sigma) with the tables as published,
+7e-6 s (0.4 sigma) with load_redshift_store(unbiased=True).  Above s of a few the sample mean is dominated by rare
segments and underestimates the mean."""
import sys
import numpy as np
sys.path.insert(0, 'src')
import exmarg.search as search
from exmarg.pp import simulate
from exmarg.search import pe

n, seed, smax = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
unbiased = len(sys.argv) > 4 and sys.argv[4] == 'unbiased'
T, FMIN, FMAX, YMIN = 8.0, 20.0, 512.0, 0.1267
segments, _, _ = simulate(n, seed, lambda rng: (60.0, 1.0, 0.0), lambda rng: 0.0, T, FMIN, FMAX)
seg = search.read_segments(segments)
store = search.load_redshift_store('tables/redshift', pinned=True, unbiased=unbiased)
print('unbiased tables' if unbiased else 'tables as published')
bank = search.mass_bank(np.array([30.0, 60.0, 120.0]), segments, 'gpu')
geometry = pe._node_geometry(bank, store, YMIN)
low = np.nonzero(store.s <= smax)[0]
groups = [list(low[i:i + 6]) for i in range(0, low.size, 6)]
sd = pe._single_setup(bank, seg, store, geometry)
nn = store.s.size
out = {k: np.full((n, nn + 1, bank.M.size), np.nan) for k in ('lamN', 'lamN_H', 'lamN_L')}
for ev in pe._chunks(seg, 16):
    lamN, one = pe._node_values(bank, seg, ev, store, geometry, groups, 256, False, 0, 0, seed, sd)
    out['lamN'][ev.start:ev.stop], out['lamN_H'][ev.start:ev.stop], out['lamN_L'][ev.start:ev.stop] = lamN, one['lamN_H'], one['lamN_L']
valid = geometry[0]
print('%d noise-only segments, templates M %s u %s' % (n, bank.M, np.round(bank.u, 3)))
print('%4s %9s %5s | %-34s | %-34s | %-34s | %s' % ('node', 's', 'M', 'coherent (mean-1)/s +- se', 'H only', 'L only', 'coherent - H - L + 1, /s'))
for k in low:
    for m in range(bank.M.size):
        if not valid[k, m]:
            continue
        cols = []
        for key in ('lamN', 'lamN_H', 'lamN_L'):
            x = np.exp(out[key][:, k, m])
            cols.append('%+.3e +- %.1e (%+5.1f sig)' % ((x.mean() - 1) / store.s[k], x.std(ddof=1) / np.sqrt(n) / store.s[k],
                                                        (x.mean() - 1) / (x.std(ddof=1) / np.sqrt(n))))
        x = np.exp(out['lamN'][:, k, m]) - np.exp(out['lamN_H'][:, k, m]) - np.exp(out['lamN_L'][:, k, m]) + 1.0
        cols.append('%+.3e +- %.1e (%+5.1f sig)' % (x.mean() / store.s[k], x.std(ddof=1) / np.sqrt(n) / store.s[k], x.mean() / (x.std(ddof=1) / np.sqrt(n))))
        print('%4d %9.4g %5.0f | %s | %s | %s | %s' % (k, store.s[k], bank.M[m], *cols))
