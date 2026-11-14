"""The fits of the analysed week (scripts/real_data/analyse.py) in the order fixed by docs/real_data.md, "Order of the
analysis and its criteria": the phase-randomized week, then the time-reversed and the time-slid weeks, then the
injections into the slid week; the week as recorded last, only with --recorded and only once the controls, checked
again here, pass.

    python scripts/real_data/summarize.py runs/week [--rho-min 5 4.5 6] [--veto-w 72] [--w-inf-max 5e-4]
    python scripts/real_data/summarize.py runs/week --recorded

One row per variant (plant_fit.report on <out>/<variant>): segments kept and set aside by the loudness veto, nbar and
its error with the Gaussian likelihood and with the mixture (joint power-law fit, the first --rho-min), the fitted
eps_H, eps_L, gamma_H, gamma_L, the end and maximum of the evidence curve of each likelihood, and the verdict on the
frozen criteria: on the controls nbar within 2.5 standard errors of 0 or at the lower edge and the evidence below
ln 100, for the mixture and, on the phase-randomized week only, for the Gaussian likelihood too (on the slid and
reversed weeks the Gaussian likelihood is expected to fail: a control of the test); on the injections the mixture nbar
within 2.5 sigma of the injected rate.  Without --recorded the directory of the recorded week is not opened.  Exit
status 1 when a control is missing or fails.
"""
import argparse
import glob
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
_spec = importlib.util.spec_from_file_location('plant_fit', os.path.join(ROOT, 'scripts', 'validate', 'plant_fit.py'))
plant_fit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plant_fit)

ORDER = ('pr', 'rev', 'slide+0.5', 'slide-1.75', 'inj')
RECORDED = 'zero'
HEAD = ('variant', 'kept', 'aside', 'Gaussian nbar', 'mixture nbar', 'eps_H', 'eps_L', 'gamma_H', 'gamma_L',
        'Gaussian ev. end/max', 'mixture ev. end/max', 'verdict')
WIDTH = (11, 7, 6, 20, 20, 15, 15, 8, 8, 21, 21, 0)


def _pm(r, p, fmt):
    v, e = r[p], r['err'].get(p, np.nan)
    return '%s+-%s' % (plant_fit._f(v, fmt), plant_fit._f(e, '%.2g'))


def verdict(r):
    """PASS or FAIL on the counted criteria, with the failing ones; the controls of the test that behave as expected"""
    if r['kind'] == 'recorded':
        failed = [name for name, ok, _, counted in r['criteria'] if counted and not ok]
        return 'recorded (no criterion on nbar)' + ('; FAIL: ' + ', '.join(failed) if failed else '')
    failed = [name for name, ok, _, counted in r['criteria'] if counted and not ok]
    test = [name for name, ok, _, counted in r['criteria'] if not counted and name.startswith('Gaussian')]
    note = ''
    if test:
        fails = all(not ok for name, ok, _, counted in r['criteria'] if name in test)
        note = '; Gaussian %s (control of the test)' % ('fails as expected' if fails else 'passes, expected to fail')
    return ('PASS' if r['ok'] else 'FAIL: ' + ', '.join(failed)) + note


def row(variant, r):
    """the cells of one variant's row"""
    if r is None:
        return (variant, '', '', '', '', '', '', '', '', '', '', 'MISSING: no files')
    F, J, G = r['fits'], r['fits']['joint'], r['fits']['gauss']
    cg, cm = r['curves']['gauss'], r['curves']['mixture']
    return (variant, str(int(r['keep'].sum())), str(int((~r['keep']).sum())), _pm(G, 'nbar', '%.3g'),
            _pm(J, 'nbar', '%.3g'), _pm(J, 'eps_H', '%.3g'), _pm(J, 'eps_L', '%.3g'),
            plant_fit._f(J['gamma_H'], '%.3g'), plant_fit._f(J['gamma_L'], '%.3g'),
            '%+.2f/%+.2f' % (cg[-1], cg.max()), '%+.2f/%+.2f' % (cm[-1], cm.max()), verdict(r))


def table(rows):
    fmt = lambda cells: ' '.join(('%-*s' % (w, c)) if w else c for w, c in zip(WIDTH, cells))
    return [fmt(HEAD)] + [fmt(c) for c in rows]


def fit(out, variant, a):
    path = os.path.join(out, variant)
    if not glob.glob(os.path.join(path, 'plant_*.npz')):
        return None
    return plant_fit.report(path, veto_w=a.veto_w, w_inf_max=a.w_inf_max, rho_mins=a.rho_min)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', help='the output folder of analyse.py (<out>/<variant>/plant_*.npz)')
    p.add_argument('--recorded', action='store_true', help='after the controls pass, also the week as recorded')
    p.add_argument('--rho-min', type=float, nargs='+', default=None, help='as plant_fit.py (default 5 4.5 6)')
    p.add_argument('--veto-w', type=float, default=plant_fit.VETO_W)
    p.add_argument('--w-inf-max', type=float, default=plant_fit.W_INF_MAX)
    a = p.parse_args(argv)
    results = {}
    for v in ORDER:
        results[v] = fit(a.out, v, a)
        print('fitted %s' % v if results[v] is not None else 'no files in %s' % os.path.join(a.out, v), flush=True)
    rho = next((r['rho_min'] for r in results.values() if r is not None), None)
    print('\nthe week, in the order fixed before the first run; loudness veto w > %g, glitch law from rho_min %s; '
          'evidence: ln of the evidence ratio against no mergers, prior uniform in ln nbar on [%g, %g]\n'
          % (a.veto_w, plant_fit._f(rho, '%g'), plant_fit.PRIOR[0], plant_fit.PRIOR[1]))
    print('\n'.join(table([row(v, results[v]) for v in ORDER])), flush=True)
    passed = all(results[v] is not None and results[v]['ok'] for v in ORDER)
    if not a.recorded:
        print('\ncontrols and injections: %s.  The week as recorded (%s) is not opened; run with --recorded once every '
              'row above passes.' % ('all pass' if passed else 'NOT all pass', RECORDED))
        return 0 if passed else 1
    if not passed:
        print('\n--recorded: controls and injections checked again, NOT all pass: the week as recorded stays closed.')
        return 1
    print('\n--recorded: controls and injections checked again, all pass: opening the week as recorded.', flush=True)
    r = fit(a.out, RECORDED, a)
    print('\n'.join(table([row(RECORDED, r)])[1:]))
    if r is not None:
        print('\n' + '\n'.join(r['lines']))
    return 0 if r is not None and r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
