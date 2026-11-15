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
status 1 when a control fails.

Variants analysed with the window model (analyse.py --glitch-model; their files carry ln_B_bar) are fitted with it in
place of the mixture (plant_fit.py): the row then shows nbar of the window model, the glitches per segment of each
detector's model (n_window sum_s f_s q_s, from <out>/<variant>/glitch_model.json when it is the model of the files) and
the start of the model's SHA-256.  Every variant must then carry the window model (glitch_model and n_window are
campaign keys); the SHA-256 belongs to the variant, each being analysed with the model of its own data.

Before any fit, <out>/selection.txt is read (analyse.py writes it) and every variant that is used must cover the
segments k = 0 .. n_selected - 1 exactly once (no gap, no segment in two files, none missing at the end), and the
controls must carry the same campaign (every key of the files' meta that the variant does not define: manifest hash,
seed, segment folder, ymin, zmin, ...).  The recorded week is checked from the names of its files alone.  A violation is
an error that names the variant and what is missing, and nothing is fitted.
"""
import argparse
import glob
import importlib.util
import json
import os
import re
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
HEAD_W = ('variant', 'kept', 'aside', 'Gaussian nbar', 'window nbar', 'glitches/seg H', 'glitches/seg L', 'model',
          'Gaussian ev. end/max', 'window ev. end/max', 'verdict')
WIDTH_W = (11, 7, 6, 20, 20, 15, 15, 13, 21, 21, 0)


# the meta entries that belong to a variant: the others must be the same in every variant of the week
VARIANT_KEYS = ('variant', 'control', 'nbar', 'shift', 'merger_truth', 'truth_note', 'glitch_model_sha256')
FILE = re.compile(r'^plant_(\d+)_(\d+)\.npz$')


def selected_total(out):
    """the number of selected segments of selection.txt (epoch, GPS, segments, selected, first k per line)"""
    path = os.path.join(out, 'selection.txt')
    if not os.path.exists(path):
        raise SystemExit('%s: no selection.txt: the number of segments the week should have is not known' % out)
    with open(path) as fh:
        rows = [line.split() for line in fh if line.strip()]
    return sum(int(r[3]) for r in rows)


def _ranges(ks, n):
    """'k a to b, ...' for the runs of the sorted unique indices ks, within 0 .. n - 1 as given"""
    ks = np.asarray(sorted(ks), int)
    if not ks.size:
        return 'none'
    cut = np.flatnonzero(np.diff(ks) > 1) + 1
    runs = [(g[0], g[-1]) for g in np.split(ks, cut)]
    return ', '.join('k %d to %d' % r if r[0] != r[1] else 'k %d' % r[0] for r in runs)


def coverage(out, variant, n, opened):
    """the problems (list of text) of the files of out/variant against k = 0 .. n - 1: a segment missing, in more than
    one file, or beyond n; the name of a file that is not of the form plant_<first>_<end>.npz; with opened, the k of
    each file against its name.  Returns (problems, meta of the files or None when not opened)"""
    path = os.path.join(out, variant)
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(path, 'plant_*.npz')))
    if not names:
        return ['%s: no files in %s; all %d segments (k 0 to %d) missing' % (variant, path, n, n - 1)], None
    count, problems, metas = np.zeros(max(n, 1), int), [], []
    over = []
    for name in names:
        m = FILE.match(name)
        if not m:
            problems.append('%s: file name %s is not plant_<first k>_<end k>.npz' % (variant, name))
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if opened:
            with np.load(os.path.join(path, name)) as z:
                k = z['k']
                metas.append((name, json.loads(str(z['meta']))))
            if not np.array_equal(k, np.arange(a, b)):
                problems.append('%s: %s holds segments other than k %d to %d' % (variant, name, a, b - 1))
        count[a:min(b, n)] += 1
        over += list(range(max(a, n), b))
    miss, twice = np.flatnonzero(count[:n] == 0), np.flatnonzero(count[:n] > 1)
    if miss.size:
        problems.append('%s: %d of %d segments missing: %s (files present cover %s)'
                        % (variant, miss.size, n, _ranges(miss, n), _ranges(np.flatnonzero(count[:n] > 0), n)))
    if twice.size:
        problems.append('%s: segments in more than one file: %s' % (variant, _ranges(twice, n)))
    if over:
        problems.append('%s: segments beyond the %d selected: %s' % (variant, n, _ranges(over, n)))
    return problems, metas if opened else None


def campaign_problems(metas):
    """{variant: [(file, meta)]} -> the problems (list of text): a key that the variant does not define differs from
    the first variant's, or the files of a variant differ among themselves, or meta['variant'] is not the folder"""
    problems, ref, ref_name = [], None, None
    for v, files in metas.items():
        own = None
        for name, meta in files:
            meta = {k: x for k, x in meta.items() if k != 'run'}
            if meta.get('variant') != v:
                problems.append('%s: %s says it is variant %r' % (v, name, meta.get('variant')))
            mine = {k: x for k, x in meta.items() if k in VARIANT_KEYS}
            common = {k: x for k, x in meta.items() if k not in VARIANT_KEYS}
            if own is None:
                own = mine
            elif mine != own:
                problems.append('%s: %s differs from the other files of the variant in %s'
                                % (v, name, ', '.join(sorted(k for k in set(mine) | set(own) if mine.get(k) != own.get(k)))))
            if ref is None:
                ref, ref_name = common, '%s/%s' % (v, name)
            elif common != ref:
                keys = sorted(k for k in set(common) | set(ref) if common.get(k) != ref.get(k))
                problems.append('%s: %s belongs to another campaign than %s (differs in %s)'
                                % (v, name, ref_name, ', '.join(keys)))
    return problems, ref


def check_week(out, variants, recorded_by_name=()):
    """the guard: SystemExit with every problem when a variant of `variants` does not cover k = 0 .. n_selected - 1 once
    each, or the files of the variants disagree on the campaign; the variants of recorded_by_name are checked from the
    file names alone (their files are not opened).  returns (n_selected, the common campaign keys)"""
    n = selected_total(out)
    problems, metas = [], {}
    for v in variants:
        p, m = coverage(out, v, n, True)
        problems += p
        if m:
            metas[v] = m
    for v in recorded_by_name:
        problems += coverage(out, v, n, False)[0]
    p, ref = campaign_problems(metas)
    problems += p
    if problems:
        raise SystemExit('%s: the week is not complete or not one campaign, nothing is fitted:\n  %s'
                         % (out, '\n  '.join(problems)))
    return n, ref


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
    """the cells of one variant's row (HEAD, or HEAD_W for the window model)"""
    if r is None:
        return (variant, '', '', '', '', '', '', '', '', '', '', 'MISSING: no files')
    if r.get('model') == 'windows':
        G, W, m = r['fits']['gauss'], r['fits']['windows'], r.get('window_model')
        cg, cw = r['curves']['gauss'], r['curves']['windows']
        rate = lambda I: plant_fit._f(m['n_window'] * float(np.dot(m[I]['f'], m[I]['q'])), '%.4g') if m else 'n/a'
        return (variant, str(int(r['keep'].sum())), str(int((~r['keep']).sum())), _pm(G, 'nbar', '%.3g'),
                _pm(W, 'nbar', '%.3g'), rate('H'), rate('L'), str(r['cfg'].get('glitch_model_sha256'))[:12],
                '%+.2f/%+.2f' % (cg[-1], cg.max()), '%+.2f/%+.2f' % (cw[-1], cw.max()), verdict(r))
    F, J, G = r['fits'], r['fits']['joint'], r['fits']['gauss']
    cg, cm = r['curves']['gauss'], r['curves']['mixture']
    return (variant, str(int(r['keep'].sum())), str(int((~r['keep']).sum())), _pm(G, 'nbar', '%.3g'),
            _pm(J, 'nbar', '%.3g'), _pm(J, 'eps_H', '%.3g'), _pm(J, 'eps_L', '%.3g'),
            plant_fit._f(J['gamma_H'], '%.3g'), plant_fit._f(J['gamma_L'], '%.3g'),
            '%+.2f/%+.2f' % (cg[-1], cg.max()), '%+.2f/%+.2f' % (cm[-1], cm.max()), verdict(r))


def table(rows, windows=False):
    width = WIDTH_W if windows else WIDTH
    fmt = lambda cells: ' '.join(('%-*s' % (w, c)) if w else c for w, c in zip(width, cells))
    return [fmt(HEAD_W if windows else HEAD)] + [fmt(c) for c in rows]


def fit(out, variant, a, expected=None):
    path = os.path.join(out, variant)
    if not glob.glob(os.path.join(path, 'plant_*.npz')):
        return None
    return plant_fit.report(path, veto_w=a.veto_w, w_inf_max=a.w_inf_max, rho_mins=a.rho_min, monitor_cut=a.monitor_cut,
                            expected=expected)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', help='the output folder of analyse.py (<out>/<variant>/plant_*.npz)')
    p.add_argument('--recorded', action='store_true', help='after the controls pass, also the week as recorded')
    p.add_argument('--rho-min', type=float, nargs='+', default=None, help='as plant_fit.py (default 5 4.5 6)')
    p.add_argument('--veto-w', type=float, default=plant_fit.VETO_W)
    p.add_argument('--w-inf-max', type=float, default=plant_fit.W_INF_MAX)
    p.add_argument('--monitor-cut', type=float, default=None, help='as plant_fit.py: set aside segments with a band '
                   'power this many widths above the median')
    a = p.parse_args(argv)
    n_selected, campaign = check_week(a.out, ORDER, (RECORDED,) if a.recorded else ())
    results = {}
    for v in ORDER:
        results[v] = fit(a.out, v, a, n_selected)
        print('fitted %s' % v if results[v] is not None else 'no files in %s' % os.path.join(a.out, v), flush=True)
    rho = next((r['rho_min'] for r in results.values() if r is not None), None)
    windows = any(r is not None and r.get('model') == 'windows' for r in results.values())
    print('\nthe week, in the order fixed before the first run; loudness veto w > %g%s, %s; '
          'evidence: ln of the evidence ratio against no mergers, prior uniform in ln nbar on [%g, %g]\n'
          % (a.veto_w, '' if a.monitor_cut is None else ', band-power cut at %g widths' % a.monitor_cut,
             'the window model of each variant\'s own data (glitch parameters of stage 1)' if windows else
             'glitch law from rho_min %s' % plant_fit._f(rho, '%g'), plant_fit.PRIOR[0], plant_fit.PRIOR[1]))
    print('\n'.join(table([row(v, results[v]) for v in ORDER], windows)), flush=True)
    passed = all(results[v] is not None and results[v]['ok'] for v in ORDER)
    if not a.recorded:
        print('\ncontrols and injections: %s.  The week as recorded (%s) is not opened; run with --recorded once every '
              'row above passes.' % ('all pass' if passed else 'NOT all pass', RECORDED))
        return 0 if passed else 1
    if not passed:
        print('\n--recorded: controls and injections checked again, NOT all pass: the week as recorded stays closed.')
        return 1
    print('\n--recorded: controls and injections checked again, all pass: opening the week as recorded.', flush=True)
    r = fit(a.out, RECORDED, a, n_selected)
    other = sorted(k for k in set(r['cfg']) - set(VARIANT_KEYS) | set(campaign) - set(VARIANT_KEYS)
                   if r['cfg'].get(k) != campaign.get(k))
    if other:
        print('\n--recorded: the files of %s belong to another campaign than the controls (differ in %s): not shown.'
              % (RECORDED, ', '.join(other)))
        return 1
    print('\n'.join(table([row(RECORDED, r)], windows)[1:]))
    if r is not None:
        print('\n' + '\n'.join(r['lines']))
    return 0 if r is not None and r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
