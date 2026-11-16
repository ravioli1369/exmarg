"""Stage 1 of the model with many glitches per segment (docs/glitch_likelihood.md, Sec. 4a): the glitch model of each
detector, fitted to the window evidences of one campaign of plant.py or one variant of analyse.py (--stage windows), and
written to <folder>/glitch_model.json for the merger stage (--glitch-model).  On the CPU.  Exit status 1, and no file,
when a fit of the model written does not converge (the fits with other numbers of states are a record).

    python scripts/validate/windows_fit.py runs/states [--n-state 2] [--basis bins|power] [--rho-min 4] [--rho-max 12.2]
                                           [--veto-w 72] [--zone 1]
    python scripts/validate/windows_fit.py runs/week_w/slide+0.5        # a variant of the week: the whole selection
    python scripts/validate/windows_fit.py runs/states --nbar 0.0184 --model runs/states_pass2/glitch_model.json

The files.  Every win_<first>_<end>.npz of the folder, of one campaign (the meta of every file the same, stage
'windows'), each holding the segments first .. end - 1, together a contiguous range of k once each; for a variant of the
week (meta source 'strain') the whole selection, k = 0 .. n_selected - 1 of <folder>/../selection.txt.

The fit.  A segment whose wmax_H or wmax_L exceeds --veto-w, or whose wnet exceeds --w-net (84; inf: off; files
without wnet: said, and the single-detector veto alone), is set aside in both detectors (as plant_fit.py sets it
aside).  For each detector mixture.fit_windows on ln E_w[k] of the stored amplitude bins inside [--rho-min, --rho-max]
(the law is zero outside), with --n-state states and --basis bins (a free weight per bin) or power (the law of a state a
mixture of the laws rho^-g, g in GAMMAS, each normalized on those bins); 1, 2 and 3 states are fitted too, for the
record (parameters (K - 1) + K n_basis).

glitch_model.json: model 'windows', n_window (of the files), zone (--zone), pi_m null (uniform over the templates),
rho_edges (single.RHO_EDGES); per detector H, L: f (K,), q (K,), pi_k (K rows over every bin of rho_edges, zero outside
the fitted bins), a (K, n_basis) the weights of the basis, err_f, err_q, err_a (fit_windows, from its Hessian; null
where not finite), lnl, n_iter, n_parameters; fits {H, L: [{n_state, lnl, n_parameters, converged, n_iter}] for 1, 2,
3 states}; settings {veto_w, rho_min, rho_max, n_state, basis, gammas (power) or null, fitted_bins [first, end) of
rho_edges, fitted_edges, and w_net when the network cut acted}; segments {files, first, end, total, used, set_aside,
set_aside_H, set_aside_L, and set_aside_net when it acted}; campaign
(plant.campaign_key of the files: the same for both stages of a campaign); sha256 of the rest (plant.model_sha256, the
JSON with sorted keys and no spaces).  f, q and pi_k are the arguments of single.window_posterior.

--nbar N > 0 (docs/2026-11-15_mergers_in_stage1.md): every fit with the mergers of the population at the rate N of stage
2, mixture.fit_windows with the merger evidence ln_merger_I of the files (the windows stage run with the store) and the
zone --zone; refused for a control of the data (a time slide, time reversal, phase randomization, segment slide: control
not none, zero or inj).  The report adds the law, f and n_window q of the fit without mergers next to those with them,
and the model records nbar and the merger_population of the files (a model at N = 0 has neither entry, and is the file
written before these options, byte for byte).  --model: where to write the model (default <folder>/glitch_model.json).

Run as a script the fit uses one BLAS thread whatever the environment says (set before numpy is loaded), so that the
model and its sha256 do not depend on the number of cores of the machine.
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import socket
import sys

if __name__ == '__main__':
    # as a script: one BLAS thread, set before numpy loads the library.  The fit runs its parts in parallel itself;
    # with a threaded BLAS inside them the sums depend on the number of threads (on 32 cores the model differs from
    # that of one thread by 1e-15 to 1e-12 and its sha256 changes; one thread repeats bit for bit, and is faster:
    # 17 s against 46 s), so the file written would depend on the cores the machine gives the fit
    for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[_v] = '1'

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import plant
from exmarg.search import mixture
from exmarg.search.single import RHO_EDGES

_spec = importlib.util.spec_from_file_location('summarize', os.path.join(ROOT, 'scripts', 'real_data', 'summarize.py'))
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

VETO_W = 72.0
W_NET = 84.0                    # the network cut of plant_fit.py: wnet above it is set aside too (inf: off)
RHO = (4.0, 12.2)
N_STATE = 2
RECORD = (1, 2, 3)
GAMMAS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
ZONE = 1
MODEL = 'glitch_model.json'
FILE = re.compile(r'^win_(\d+)_(\d+)\.npz$')
SHARED = ('window_rho_edges', 'window_bin0')
LARGE = ('ln_window_H', 'ln_window_L')
MERGERS = ('ln_merger_H', 'ln_merger_L')
WITH_MERGERS = ('none', 'zero', 'inj')     # controls of the data whose model may be fitted with mergers (nbar > 0)


def load(folder):
    """(campaign settings, the per-segment arrays of the win files but ln_window_I, concatenated in order of k, the
    shared arrays, the paths in order of k): SystemExit, naming what is wrong, unless the files are of one campaign of
    the windows stage and cover a contiguous range of k once each, the whole selection for a variant of the week"""
    paths = glob.glob(os.path.join(folder, 'win_*.npz'))
    if not paths:
        raise SystemExit('no win_*.npz in %s' % folder)
    ranges, problems = [], []
    for path in paths:
        m = FILE.match(os.path.basename(path))
        if not m:
            raise SystemExit('%s: file name is not win_<first k>_<end k>.npz' % path)
        ranges.append((int(m.group(1)), int(m.group(2)), path))
    ranges.sort()
    cfg, shared, parts = None, None, []
    for a, b, path in ranges:
        with np.load(path) as f:
            meta = json.loads(str(f['meta']))
            if meta.get('stage') != 'windows' or not set(SHARED + LARGE) <= set(f.files):
                raise SystemExit('%s: not a file of the windows stage' % path)
            same = {key: np.asarray(f[key]) for key in SHARED}
            arrays = {key: f[key] for key in f.files if key not in ('meta',) + SHARED + LARGE + MERGERS}
        meta.pop('run', None)
        if cfg is None:
            cfg, shared = meta, same
        elif meta != cfg or not all(np.array_equal(same[key], shared[key]) for key in SHARED):
            diff = sorted(key for key in set(meta) | set(cfg) if meta.get(key) != cfg.get(key))
            diff += [key for key in SHARED if not np.array_equal(same[key], shared[key])]
            raise SystemExit('%s belongs to another campaign (differs in %s)' % (path, ', '.join(diff)))
        if not np.array_equal(arrays['k'], np.arange(a, b)):
            problems.append('%s holds segments other than k %d to %d' % (os.path.basename(path), a, b - 1))
        parts.append(arrays)
    first, end = ranges[0][0], max(b for _, b, _ in ranges)
    if cfg.get('source') == 'strain':
        n_sel = summarize.selected_total(os.path.dirname(os.path.normpath(folder)))
        if first != 0 or end != n_sel:
            problems.append('the files cover k %d to %d of the %d selected segments' % (first, end - 1, n_sel))
        first, end = 0, max(end, n_sel)
    count = np.zeros(end - first, int)
    for a, b, _ in ranges:
        count[a - first:b - first] += 1
    miss, twice = np.flatnonzero(count == 0) + first, np.flatnonzero(count > 1) + first
    if miss.size:
        problems.append('%d segments missing: %s' % (miss.size, summarize._ranges(miss, end)))
    if twice.size:
        problems.append('segments in more than one file: %s' % summarize._ranges(twice, end))
    if problems:
        raise SystemExit('%s: the files do not cover one range of segments once each, nothing is fitted:\n  %s'
                         % (folder, '\n  '.join(problems)))
    data = {key: np.concatenate([p[key] for p in parts]) for key in set.intersection(*(set(p) for p in parts))}
    return cfg, data, shared, [p for _, _, p in ranges]


def evidences(paths, I, keep, cols):
    """ln E_w[k] (n_kept, n_window, n_col) float64 of detector I: the columns cols of the stored bins, kept segments"""
    parts = []
    for path in paths:
        with np.load(path) as f:
            parts.append(np.asarray(f['ln_window_' + I])[:, :, cols])
    return np.concatenate(parts)[keep].astype(float)


def merger_evidences(paths, I, keep):
    """ln M_w (n_kept, n_window) float64 of detector I, kept segments; SystemExit when a file has none"""
    parts = []
    for path in paths:
        with np.load(path) as f:
            if 'ln_merger_' + I not in f.files:
                raise SystemExit('%s holds no merger evidence (ln_merger_%s): run the windows stage with the store '
                                 'folder in place of none' % (path, I))
            parts.append(np.asarray(f['ln_merger_' + I], float))
    return np.concatenate(parts)[keep]


def fit_columns(edges, rho_min, rho_max):
    """(n_bin,) bool: the stored bins of edges inside [rho_min, rho_max]; SystemExit when the files do not hold that
    range or no bin is inside"""
    tol = 1e-9 * max(1.0, rho_max)
    if rho_min < edges[0] - tol or rho_max > edges[-1] + tol:
        raise SystemExit('the files hold the amplitude bins from rho %g to %g only, not [%g, %g]'
                         % (edges[0], edges[-1], rho_min, rho_max))
    cols = (edges[:-1] >= rho_min - tol) & (edges[1:] <= rho_max + tol)
    if not cols.any():
        raise SystemExit('no stored amplitude bin inside [%g, %g]' % (rho_min, rho_max))
    return cols


def power_basis(edges, gammas=GAMMAS):
    """(n_gamma, n_bin): the laws dN/drho ~ rho^-g on the bins of edges, each normalized over them"""
    a, b = np.asarray(edges[:-1], float), np.asarray(edges[1:], float)
    rows = np.array([np.log(b / a) if g == 1.0 else (a ** (1.0 - g) - b ** (1.0 - g)) / (g - 1.0) for g in gammas])
    return rows / rows.sum(axis=1, keepdims=True)


def n_parameters(K, n_basis):
    """free parameters of K states: f on the simplex, and per state q and the basis weights on their simplex"""
    return (K - 1) + K * n_basis


def planted(cfg, I, fe):
    """the planted window model of detector I, states by increasing q: (f, q, the fraction of the planted law dN/drho ~
    rho^-gamma on [rho_min, rho_max] inside the fitted range fe[0] .. fe[-1], the planted law on the fitted bins fe,
    normalized), or None when not planted"""
    st = cfg.get('glitch_states')
    if not st or not cfg.get('glitch_truth', True):
        return None
    fq = np.asarray(st[I], float)
    fq = fq[np.argsort(fq[:, 1], kind='stable')]
    g, lo, hi = cfg['gamma'], cfg['rho_min'], cfg['rho_max']
    mass = lambda a, b: 0.0 if b <= a else (np.log(b / a) if g == 1.0 else (a ** (1 - g) - b ** (1 - g)) / (g - 1))
    total = mass(lo, hi)
    law = np.array([mass(max(a, lo), min(b, hi)) for a, b in zip(fe[:-1], fe[1:])])
    inside = law.sum() / total
    return fq[:, 0], fq[:, 1], inside, law / law.sum() if law.sum() > 0 else law


def _list(x):
    """a JSON list of floats, None where not finite"""
    return [None if not np.isfinite(v) else float(v) for v in np.ravel(x)] if np.ndim(x) <= 1 else [_list(r) for r in x]


def describe(cfg):
    """the data of the campaign in words"""
    if cfg.get('source') == 'strain':
        return 'recorded strain %s (manifest sha256 %s), variant %s, seed %d' % (
            cfg['segments'], cfg['manifest_sha256'][:12], cfg['variant'], cfg['seed'])
    text = 'campaign: seed %d, control %s, nbar %g, eps_H %g, eps_L %g, glitches %s, dN/drho ~ rho^-%g on [%g, %g]' % (
        cfg['seed'], cfg['control'], cfg['nbar'], cfg['eps_H'], cfg['eps_L'], cfg['glitch_kind'], cfg['gamma'],
        cfg['rho_min'], cfg['rho_max'])
    st = cfg.get('glitch_states')
    if st:
        text += '; window model, %d windows, state (f, q) H %s, L %s' % (
            st['n_window'], ', '.join('(%g, %g)' % tuple(p) for p in st['H']),
            ', '.join('(%g, %g)' % tuple(p) for p in st['L']))
    return text


def fit(folder, n_state=N_STATE, basis='bins', rho_min=RHO[0], rho_max=RHO[1], veto_w=VETO_W, zone=ZONE,
        record=RECORD, nbar=0.0, w_net=W_NET):
    """the fits and the model of the folder: dict lines (the report), model (the content of glitch_model.json without
    its sha256), ok (the fits of the model written converged), fits {I: {K: fit_windows result}}, keep (n,), cols, data.
    nbar > 0: every fit with the mergers of the population at that rate (fit_windows with the merger evidence of the
    files, docs/2026-11-15_mergers_in_stage1.md), refused (SystemExit) for a control of the data (a time slide, time
    reversal, phase randomization: their model is the one without mergers, Sec. 3) and for files without the merger
    evidence; the model then records nbar and the merger_population of the files"""
    if basis not in ('bins', 'power'):
        raise ValueError("basis must be 'bins' or 'power'")
    if n_state < 1 or zone < 0:
        raise ValueError('need n_state >= 1 and zone >= 0')
    if not 0.0 <= nbar <= 1.0:
        raise ValueError('need 0 <= nbar <= 1')
    cfg, d, shared, paths = load(folder)
    if nbar > 0 and cfg.get('control') not in WITH_MERGERS:
        raise SystemExit('%s: the data are a control (%s): no merger has a partner in the other detector, the glitch '
                         'model is fitted without mergers (nbar 0)' % (folder, cfg.get('control')))
    if nbar > 0 and 'merger_population' not in cfg:
        raise SystemExit('%s: the files hold no merger evidence (no merger_population in their campaign): run the '
                         'windows stage with the store folder in place of none' % folder)
    edges, b0 = np.asarray(shared['window_rho_edges'], float), int(shared['window_bin0'])
    if not np.array_equal(edges, RHO_EDGES[b0:b0 + edges.size]):
        raise SystemExit('%s: window_rho_edges are not bins of single.RHO_EDGES' % folder)
    nw = cfg['n_window']
    if 2 * zone + 1 > nw:
        raise SystemExit('a zone of %d windows on each side does not fit in %d windows' % (zone, nw))
    cols = fit_columns(edges, rho_min, rho_max)
    c = np.flatnonzero(cols)
    fe = edges[c[0]:c[-1] + 2]
    i0, i1 = b0 + c[0], b0 + c[-1] + 1
    if basis == 'power' and fe[0] <= 0:
        raise SystemExit('the power basis needs rho_min > 0')
    B = None if basis == 'bins' else power_basis(fe)
    nb = cols.sum() if B is None else B.shape[0]
    n = d['k'].size
    v = summarize.plant_fit.veto(d, veto_w, w_net)
    if not v['applied'] and w_net is not None and w_net != np.inf:
        lack = summarize.plant_fit._lacking(folder, 'wnet', 'win_*.npz')
        v['note'] += ' (%d of %d files lack it)' % (lack, len(paths))
    loud, keep = {I: v['loud_' + I] for I in 'HL'}, v['keep']
    Ks = sorted(set(record) | {n_state})
    R = {}
    for I in 'HL':
        E = evidences(paths, I, keep, cols)
        if nbar > 0:
            lnM = merger_evidences(paths, I, keep)
            R[I] = {K: mixture.fit_windows(E, n_state=K, basis=B, ln_M=lnM, nbar=nbar, zone=zone) for K in Ks}
        else:
            R[I] = {K: mixture.fit_windows(E, n_state=K, basis=B) for K in Ks}
        del E
    ok = all(R[I][n_state]['converged'] for I in 'HL')            # of the model written; the others are a record

    L = ['glitch model of the window evidences of %s (docs/glitch_likelihood.md, Sec. 4a)' % folder, describe(cfg),
         '%d segments in %d files, k %d to %d; %d windows%s; evidences stored on rho %.4g to %.4g'
         % (n, len(paths), d['k'][0], d['k'][-1], nw, ' of %.4g s' % (cfg['T'] / nw) if 'T' in cfg else '',
            edges[0], edges[-1]),
         'loudness veto: set aside %d segments whose wmax_H or wmax_L exceeds %g (H %d, L %d)%s; fitted on %d'
         % ((~keep).sum(), veto_w, loud['H'].sum(), loud['L'].sum(), ' or wnet exceeds %g (%d, %d of them by this '
            'alone)' % (w_net, v['net'].sum(), (v['net'] & ~loud['H'] & ~loud['L']).sum()) if v['applied'] else '',
            keep.sum()),
         'network cut %s' % ('on: wnet > %g' % w_net if v['applied'] else v['note']),
         'fitted bins: %d, rho %.4g to %.4g (the law is zero outside); amplitude basis: %s' % (
             cols.sum(), fe[0], fe[-1], 'a free weight per bin' if B is None else
             'the laws rho^-g, g in %s, each normalized on these bins' % ', '.join('%g' % g for g in GAMMAS))]
    if nbar > 0:
        mp = cfg['merger_population']
        L.append('with mergers at nbar %.6g (fixed; docs/2026-11-15_mergers_in_stage1.md): the merger evidence of the '
                 'files, population z >= %g, M_det %g to %g, ladder of %d nodes s %.4g to %.4g, ymin %.6g; zone %d'
                 % (nbar, mp['zmin'], mp['mdet_range'][0], mp['mdet_range'][1], len(mp['s']), mp['s'][0], mp['s'][-1],
                    mp['ymin'], zone))
    det, record = {}, {}
    for I in 'HL':
        L.append('\n%s: fits with %s states' % (I, ', '.join(map(str, Ks))))
        L.append('  %6s %11s %16s %8s %10s' % ('states', 'parameters', 'ln L', 'updates', 'converged'))
        for K in Ks:
            r = R[I][K]
            L.append('  %6d %11d %16.3f %8d %10s' % (K, n_parameters(K, nb), r['lnl'], r['n_iter'],
                                                     'yes' if r['converged'] else 'NO'))
        L.append('  gain in ln L: %s' % '; '.join(
            '%d over %d states %+.3f for %+d parameters' % (K, K - 1, R[I][K]['lnl'] - R[I][K - 1]['lnl'],
                                                            n_parameters(K, nb) - n_parameters(K - 1, nb))
            for K in Ks if K - 1 in Ks))
        r = R[I][n_state]
        tr = planted(cfg, I, fe)
        L.append('  the model, %d state%s (states by increasing q)%s:' % (
            n_state, 's' if n_state > 1 else '', '; planted: fraction, glitches per segment n_window q, and those in '
            'the fitted bins' if tr is not None else ''))
        L.append('    %5s %24s %36s%s' % ('state', 'fraction of segments', 'glitches per segment (n_window q)',
                                         '   planted' if tr is not None else ''))
        for s in range(n_state):
            L.append('    %5d %15.4g +- %-6.2g %24.4g +- %-9.2g%s' % (
                s + 1, r['f'][s], r['err_f'][s], nw * r['q'][s], nw * r['err_q'][s],
                '   %.4g, %.4g, %.4g' % (tr[0][s], nw * tr[1][s], nw * tr[1][s] * tr[2])
                if tr is not None and s < tr[0].size else ''))
        if tr is not None and 'glitch_state_' + I in d:
            st, ng = d['glitch_state_' + I][keep], d['n_glitches_' + I][keep]
            order = np.argsort(np.asarray(cfg['glitch_states'][I], float)[:, 1], kind='stable')
            L.append('    realized among the kept segments: %s' % ', '.join(
                'state %d fraction %.4g, %.4g glitches per segment' % (j + 1, (st == o).mean(),
                                                                       ng[st == o].mean() if (st == o).any() else 0)
                for j, o in enumerate(order)))
        L.append('  amplitude law of each state, probability of each bin%s:' % (
            ' +- error' if B is None else ' (the basis weights below)'))
        L.append('    %-16s' % 'rho bin' + ''.join('%22s' % ('state %d' % (s + 1)) for s in range(n_state))
                 + ('%12s' % 'planted' if tr is not None else ''))
        for j in range(cols.sum()):
            cells = ''.join('%22s' % ('%.4f +- %.4f' % (r['pi'][s, j], r['err_a'][s, j]) if B is None else
                                      '%.4f' % r['pi'][s, j]) for s in range(n_state))
            L.append('    %-16s' % ('%.2f - %.2f' % (fe[j], fe[j + 1])) + cells
                     + ('%12.4f' % tr[3][j] if tr is not None else ''))
        if B is not None:
            for s in range(n_state):
                L.append('    state %d, weights of rho^-g: %s' % (s + 1, ', '.join(
                    'g %g %.4f +- %.4f' % (g, r['a'][s, b], r['err_a'][s, b]) for b, g in enumerate(GAMMAS))))
        if nbar > 0:
            r0 = r['fit0']
            L.append('  without mergers (nbar 0, the start of the fit): ln L %.3f; with them %.3f' % (r0['lnl'], r['lnl']))
            L.append('    %-16s' % 'rho bin' + ''.join('%34s' % ('state %d, nbar 0 -> %g' % (s + 1, nbar))
                                                       for s in range(n_state)))
            for j in range(nb):
                L.append('    %-16s' % (('%.2f - %.2f' % (fe[j], fe[j + 1])) if B is None else 'basis %d' % j) + ''.join(
                    '%34s' % ('%.4f -> %.4f' % (r0['a'][s, j], r['a'][s, j])) for s in range(n_state)))
            L.append('    %-16s' % 'f' + ''.join('%34s' % ('%.5g -> %.5g' % (r0['f'][s], r['f'][s]))
                                                 for s in range(n_state)))
            L.append('    %-16s' % 'n_window q' + ''.join('%34s' % ('%.5g -> %.5g' % (nw * r0['q'][s], nw * r['q'][s]))
                                                          for s in range(n_state)))
        pi = np.zeros((n_state, RHO_EDGES.size - 1))
        pi[:, i0:i1] = r['pi']
        det[I] = dict(f=_list(r['f']), q=_list(r['q']), pi_k=_list(pi), a=_list(r['a']), err_f=_list(r['err_f']),
                      err_q=_list(r['err_q']), err_a=_list(r['err_a']), lnl=float(r['lnl']), n_iter=int(r['n_iter']),
                      n_parameters=int(n_parameters(n_state, nb)))
        record[I] = [dict(n_state=int(K), lnl=float(R[I][K]['lnl']), n_parameters=int(n_parameters(K, nb)),
                          converged=bool(R[I][K]['converged']), n_iter=int(R[I][K]['n_iter'])) for K in Ks]
    model = dict(model='windows', n_window=int(nw), zone=int(zone), pi_m=None, rho_edges=_list(RHO_EDGES), H=det['H'],
                 L=det['L'], fits=record,
                 settings=dict(veto_w=float(veto_w), rho_min=float(rho_min), rho_max=float(rho_max),
                               n_state=int(n_state), basis=basis, gammas=list(GAMMAS) if B is not None else None,
                               fitted_bins=[int(i0), int(i1)], fitted_edges=_list(fe)),
                 segments=dict(files=len(paths), first=int(d['k'][0]), end=int(d['k'][-1]) + 1, total=int(n),
                               used=int(keep.sum()), set_aside=int((~keep).sum()), set_aside_H=int(loud['H'].sum()),
                               set_aside_L=int(loud['L'].sum())),
                 campaign=plant.campaign_key(cfg))
    if v['applied']:                # recorded only when it acted: without it the model is that of the single veto
        model['settings']['w_net'] = float(w_net)
        model['segments']['set_aside_net'] = int(v['net'].sum())
    if nbar > 0:
        model.update(nbar=float(nbar), merger_population=cfg['merger_population'])
    model = json.loads(json.dumps(model))
    slow = ['%s with %d states' % (I, K) for I in 'HL' for K in Ks if not R[I][K]['converged']]
    if not ok:
        L.append('\nNOT CONVERGED: %s; no model written' % ', '.join(slow))
    elif slow:
        L.append('\nnot converged, among the fits kept for the record only: %s (their ln L is a lower bound)'
                 % ', '.join(slow))
    return dict(lines=L, model=model, ok=ok, fits=R, keep=keep, cols=cols, data=d, cfg=cfg)


def write_model(path, model):
    """the model with its sha256 (plant.model_sha256), written under a temporary name and renamed; returns the
    SHA-256"""
    model = dict(model, sha256=plant.model_sha256(model))
    tmp = '%s.%s.%d.tmp' % (path, socket.gethostname(), os.getpid())
    with open(tmp, 'w') as fh:
        json.dump(model, fh, indent=1, sort_keys=True)
        fh.write('\n')
    os.replace(tmp, path)
    return model['sha256']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', help='the folder of the win_*.npz files of one campaign or variant')
    p.add_argument('--veto-w', type=float, default=VETO_W,
                   help='set aside a segment whose wmax_H or wmax_L exceeds this')
    p.add_argument('--w-net', type=float, default=W_NET,
                   help='network cut: also set aside a segment whose wnet exceeds this (default %g; inf: off; files '
                        'without wnet: said, and not applied)' % W_NET)
    p.add_argument('--rho-min', type=float, default=RHO[0], help='the fitted bins: those inside [rho-min, rho-max]')
    p.add_argument('--rho-max', type=float, default=RHO[1])
    p.add_argument('--n-state', type=int, default=N_STATE, help='states of the model written (1, 2, 3 are fitted too)')
    p.add_argument('--basis', choices=('bins', 'power'), default='bins',
                   help='the amplitude law: a free weight per bin, or a mixture of power laws rho^-g')
    p.add_argument('--zone', type=int, default=ZONE, help='windows on each side of a window in its zone')
    p.add_argument('--record', default=','.join(map(str, RECORD)),
                   help='numbers of states fitted for the record besides --n-state (comma-separated; a fit of 3 '
                        'states of a week takes several minutes)')
    p.add_argument('--nbar', type=float, default=0.0,
                   help='the merger rate of stage 2 at which the glitch model is fitted with the mergers of the '
                        'population (default 0: without; not for the controls of the data)')
    p.add_argument('--model', default=None, metavar='JSON',
                   help='where to write the model (default <folder>/%s)' % MODEL)
    a = p.parse_args(argv)
    if not (0 <= a.rho_min < a.rho_max) or a.n_state < 1 or a.zone < 0 or not 0 <= a.nbar <= 1:
        p.error('need 0 <= rho-min < rho-max, n-state >= 1, zone >= 0, 0 <= nbar <= 1')
    r = fit(a.folder, a.n_state, a.basis, a.rho_min, a.rho_max, a.veto_w, a.zone,
            tuple(int(x) for x in a.record.split(',') if x), a.nbar, a.w_net)
    print('\n'.join(r['lines']), flush=True)
    if not r['ok']:
        return 1
    path = os.path.join(a.folder, MODEL) if a.model is None else a.model
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    old = None
    if os.path.exists(path):
        with open(path) as fh:
            try:
                old = json.load(fh).get('sha256')
            except (ValueError, AttributeError):
                old = 'unreadable'
    sha = write_model(path, r['model'])
    print('\nwritten %s, SHA-256 %s%s' % (path, sha, '' if old in (None, sha) else
                                           ' (replaces the model of SHA-256 %s: runs of the merger stage with that '
                                           'model are of another campaign now)' % old), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
