"""The fits of a planted campaign of plant.py (controls 4 to 9 of docs/glitch_likelihood.md, Sec. 7), on the CPU, with
the truth alongside.  Exit status 1 when a criterion fails.

    python scripts/validate/plant_fit.py runs/template [--rho-min-fit] [--expected N]

For all the segments of the directory together it prints
  - the planted (expected) and realized numbers: segments, mergers, glitches per detector, mergers sharing a segment
    with a glitch;
  - nbar from the Gaussian likelihood (eps = 0), Smith and Thrane's form (sg=False, noise model free as in the joint
    fit), the full mixture with the joint power-law fit (nbar, eps_I, gamma_I free) and the two-stage fit (stage 1
    mixture.fit_noise per detector, stage 2 nbar with its binned laws and eps fixed, then with eps free), each with its
    pull against the truth; eps_I and gamma_I against the truth.  rho_min is held at the planted value (glitches below
    rho ~ 4 are nearly degenerate with Gaussian noise), free with --rho-min-fit.  A detector whose eps ends at 0 has
    no law to measure: its gamma is then held and the fit repeated.  Fits start at nbar 0.01, eps 0.01, gamma 3.  The
    lambda terms are those of pe_redshift at its fiducial law, held fixed (Sec. 4);
  - the unit mean of Sec. 1: the mean of Lambda_bar - 1 (Gaussian) and of B - 1 (mixture, at the planted noise model
    and at the joint fit) over the segments with no merger signal in either detector, with standard errors, for all of
    them and split into those with a glitch and those without;
  - the time curve: ln of the evidence ratio of mergers against none over the segments in order of k, the prior
    uniform in ln nbar on [1e-4, 0.5]; end value and maximum, Gaussian and mixture (noise model of the joint fit);
  - the selection: a segment is set aside when wmax_H or wmax_L exceeds --veto-w (72) or wnet exceeds --w-net (84;
    inf: off), what each condition sets aside and, where known, how many of those hold a merger signal or a glitch;
    files written before wnet was computed are said to lack it and get the single-detector veto alone;
  - the detected fraction of that selection, f = 1 - P(kept | merger) / P(kept | no merger), measured on a plant's
    truth with its binomial error, and nbar of the window model (or of the mixture with the noise model of the joint
    fit held) for f = 0, for --f-det and for the measured f (nbar_fit, Eq. 35 of the paper);
  - pass or fail: the mixture nbar (joint fit) within 2.5 sigma of the truth; on control data (control not none, or
    nbar = 0) nbar consistent with 0 (below 2.5 sigma) and the mixture time curve below ln 100 everywhere (Ville's
    bound at 1 percent); every segment's population weight on cells without ln Lambda (w_inf) below 1e-4.
The truth of nbar is the --nbar of the campaign without a control and 0 with one.

Files of the merger stage with the window model (plant.py, analyse.py --glitch-model; they carry ln_B_bar and no
ln_lambda_g*) are fitted with it in place of the mixture (docs/glitch_likelihood.md, Sec. 4a): nbar maximizes
sum_i ln(1 + nbar (Bbar_i - 1)) on the kept segments, the glitch parameters fixed in stage 1 (nbar_fit; its error from
the curvature), the time curve is that of Bbar, and the criteria are those above with the window model in place of the
mixture; the Gaussian likelihood is fitted as above.  The unit mean adds the number of segments that carry half of
sum |Bbar - 1| (heavy tails), and the model is described from <directory>/glitch_model.json when that is the file whose
SHA-256 the campaign records.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exmarg.search import mixture

PRIOR = (1e-4, 0.5)
NBAR_GRID = np.geomspace(PRIOR[0], PRIOR[1], 600)
VILLE = np.log(100.0)
PULL = 2.5
W_INF_MAX = 5e-4
VETO_W = 72.0
W_NET = 84.0        # the network cut: a segment whose wnet (the largest w_H(t + tau) + w_L(t) of any table read)
                    # exceeds this is set aside too: the tables hold 1e-3 in ln for W = w_H + w_L <= 84 only
VETO_OFFSET, VETO_WIDTH = 0.1, 1.0      # sqrt(2 wmax) - rho of planted glitches: +0.10 +- 0.95 (H), +0.20 +- 1.01 (L)
START = dict(nbar=0.01, eps_H=0.01, eps_L=0.01, gamma_H=3.0, gamma_L=3.0)
SHARED = ('rho_edges', 'glitch_pi_k', 'glitch_window')
PR_LABELS = ('pr', 'phase-randomize')       # controls where the Gaussian likelihood must pass too


def _equal(a, b):
    return (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b, equal_nan=True))


def load(directory):
    """(campaign settings, rho_edges, the per-segment arrays of every plant_*.npz of directory concatenated in order of
    k (the keys every file has), the 'run' entries of the files, the (m1, z) grid indices (n_cell, 2) of the cells
    without ln Lambda over all files or None when a file lacks them); the settings, rho_edges and the fiducial glitch
    law and window must agree across the files"""
    files = sorted(glob.glob(os.path.join(directory, 'plant_*.npz')))
    if not files:
        raise SystemExit('no plant_*.npz in %s' % directory)
    cfg, fixed, parts, runs, cells = None, None, [], [], []
    for path in files:
        with np.load(path) as f:
            meta = json.loads(str(f['meta']))
            same = [np.asarray(f[key], float) if key in f.files else None for key in SHARED]
            arrays = {key: f[key] for key in f.files if key not in ('meta', 'w_inf_cells') + SHARED}
            cells.append(f['w_inf_cells'] if 'w_inf_cells' in f.files else None)
        run = meta.pop('run')
        if cfg is None:
            cfg, fixed = meta, same
        elif meta != cfg or not all(_equal(a, b) for a, b in zip(same, fixed)):
            diff = sorted(key for key in set(meta) | set(cfg) if meta.get(key) != cfg.get(key))
            diff += [key for key, a, b in zip(SHARED, same, fixed) if not _equal(a, b)]
            raise SystemExit('%s belongs to another campaign (differs in %s)' % (path, ', '.join(diff)))
        if not np.array_equal(arrays['k'], np.arange(run['first'], run['end'])):
            raise SystemExit('%s: segment indices do not match its range' % path)
        parts.append(arrays)
        runs.append(run)
    keys = set.intersection(*(set(p) for p in parts))
    data = {key: np.concatenate([p[key] for p in parts]) for key in sorted(keys)}
    order = np.argsort(data['k'], kind='stable')
    data = {key: v[order] for key, v in data.items()}
    twice = data['k'][1:][np.diff(data['k']) == 0]
    if twice.size:
        raise SystemExit('segments in more than one file: %s' % twice[:10])
    cells = None if any(c is None for c in cells) else np.unique(np.concatenate(cells).reshape(-1, 2), axis=0)
    return cfg, fixed[0], data, runs, cells


def truth(cfg):
    """(the true parameters, None where not known, and the kind of the data): 'plant' (mergers planted or injected at a
    known nbar), 'control' (no coherent merger: nbar 0) or 'recorded' (nbar unknown, the meta's nbar null).  Glitch
    parameters are known only for planted glitches (meta glitch_truth, true by default)"""
    known = cfg.get('glitch_truth', True)
    g = lambda key: cfg[key] if known else None
    tr = dict(eps_H=g('eps_H'), eps_L=g('eps_L'), gamma_H=g('gamma'), gamma_L=g('gamma'), rho_min_H=g('rho_min'),
              rho_min_L=g('rho_min'))
    if cfg['nbar'] is None:
        return dict(tr, nbar=None), 'recorded'
    if cfg['control'] in ('none', 'inj') and cfg['nbar'] > 0:
        return dict(tr, nbar=cfg['nbar']), 'plant'
    return dict(tr, nbar=0.0), 'control'


def fit(d, edges, free, start, sg=True, pi_k=(None, None), cut=None):
    """mixture.fit on the stored logarithms, the parameters free and the others at start; cut: the end of the law,
    dict(rho_max=...) (hard) or dict(veto=(cut, offset, width)) (soft), see mixture.amplitude_weights"""
    return mixture.fit(d['ln_lambda_bar'], d['ln_lambda_gH_bar'], d['ln_lambda_gL_bar'], d['ln_glitch_H'],
                       d['ln_glitch_L'], edges, start=start, log=True,
                       fixed=tuple(p for p in mixture.PARAMETERS if p not in free), sg=sg,
                       lam_gHL=d['ln_lambda_gHL_bar'], pi_k_H=pi_k[0], pi_k_L=pi_k[1], **(cut or {}))


def law_fit(d, edges, start, rho_min_fit=False, sg=True, cut=None):
    """nbar, eps_I and the power law of each detector free (rho_min too with rho_min_fit); the law of a detector whose
    eps ends at 0 is then held and the fit repeated.  returns the fit with 'held' the detectors whose law was held"""
    free = ['nbar', 'eps_H', 'eps_L', 'gamma_H', 'gamma_L'] + (['rho_min_H', 'rho_min_L'] if rho_min_fit else [])
    r = fit(d, edges, free, start, sg, cut=cut)
    held = [I for I in 'HL' if r['eps_' + I] <= 1e-9]
    if held:
        free = [p for p in free if not (p.split('_')[0] in ('gamma', 'rho') and p[-1] in held)]
        r = fit(d, edges, free, {p: r[p] for p in mixture.PARAMETERS}, sg, cut=cut)
    r['held'] = held
    return r


def gauss_fit(d, edges, start):
    """the Gaussian likelihood (eps = 0) by mixture.fit; the lambda terms given a glitch do not enter at eps = 0, and
    are taken as ln_lambda_bar in files that have none (window model)"""
    g = d if 'ln_lambda_gH_bar' in d else dict(d, **{key: d['ln_lambda_bar'] for key in (
        'ln_lambda_gH_bar', 'ln_lambda_gL_bar', 'ln_lambda_gHL_bar')})
    return fit(g, edges, ['nbar'], dict(start, eps_H=0.0, eps_L=0.0))


def nbar_fit(ln_B, f_det=0.0):
    """the population estimate of the window model: nbar maximizing sum_i ln(1 + nbar (B_i - 1)) on [0, 1], from ln B
    (n,), concave in nbar.  0 when the slope at 0, sum_i (B_i - 1), is not positive; 1 when the slope at 1 is not
    negative; else the root of the slope sum_i u_i, u_i = (B_i - 1) / (1 + nbar (B_i - 1)) = 1 / (nbar + 1 / (B_i - 1)),
    by brentq.  The error is 1 / sqrt(sum_i u_i^2) at the estimate (the curvature, on an edge too).  returns dict nbar,
    err {nbar}, lnl (the sum at the estimate), free ['nbar'], edge (None, 'lower' or 'upper'), f_det.

    f_det (a number < 1): the detected fraction f = 1 - P(kept | merger) / P(kept | no merger) of the selection that
    kept these n segments (detected_fraction).  The likelihood of the kept segments is then (the paper's Eq. 35)
    prod_i [1 + nbar (B_i - 1)] / (1 - nbar f)^n: its log adds -n ln(1 - nbar f), the slope + n f / (1 - nbar f) and
    the curvature - n f^2 / (1 - nbar f)^2, which may spoil concavity but not the uniqueness of the maximum: at a root
    of the slope sum_i u_i = -n g, g = f / (1 - nbar f), and Cauchy-Schwarz gives sum_i u_i^2 >= (sum_i u_i)^2 / n =
    n g^2, so every stationary point is a maximum and there is at most one; the same three cases and brentq hold.
    The error is 1 / sqrt(sum_i u_i^2 - n g^2) at the estimate (nan when not positive).  f_det = 0: the expressions
    without these terms, every number as before"""
    lnB = np.asarray(ln_B, float)
    if np.isnan(lnB).any():
        raise ValueError('ln B is nan in %d segments' % np.isnan(lnB).sum())
    f = float(f_det)
    if not (np.isfinite(f) and f < 1.0):
        raise ValueError('f_det must be a finite number below 1, not %r' % (f_det,))
    with np.errstate(over='ignore', divide='ignore'):
        inv = 1.0 / np.expm1(lnB)                       # 1 / (B - 1): 0 at B = inf, inf at B = 1, -1 at B = 0

    def u(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.isinf(inv), 0.0, 1.0 / (x + inv))

    if f == 0.0:
        slope = lambda x: float(np.sum(u(x)))
    else:
        slope = lambda x: float(np.sum(u(x))) + lnB.size * f / (1.0 - x * f)
    s0, s1 = slope(0.0), slope(1.0)
    if not s0 > 0:
        x, edge = 0.0, 'lower'
    elif s1 >= 0:
        x, edge = 1.0, 'upper'
    else:
        a = 0.0 if np.isfinite(s0) else 1e-15
        x, edge = (brentq(slope, a, 1.0, xtol=1e-15, rtol=1e-15, maxiter=500) if slope(a) > 0 else a), None
    with np.errstate(over='ignore'):
        info = float(np.sum(u(x) ** 2))
    lnl = mixture.log_likelihood(x, lnB, 0.0, 0.0, log=True)
    if f != 0.0:
        info -= lnB.size * (f / (1.0 - x * f)) ** 2
        lnl -= lnB.size * np.log1p(-x * f)
    err = 1.0 / np.sqrt(info) if info > 0 else np.nan
    return dict(nbar=x, err=dict(nbar=err), lnl=lnl, free=['nbar'], edge=edge, f_det=f)


def detected_fraction(merger, keep):
    """the detected fraction of a selection, measured on a plant's truth: f = 1 - k1 / k0, k1 = P(kept | merger) and
    k0 = P(kept | no merger) estimated by the fractions of kept segments among those with and without a planted merger
    (merger, keep: (n,) bool).  Its binomial error from the two independent fractions, to first order:
    err = (k1 / k0) sqrt((1 - k1) / (n1 k1) + (1 - k0) / (n0 k0)) (0 when no segment of either kind is set aside: then
    the 95 percent bound of the rule of three, 3 / n1, is in upper).  returns dict f, err, upper (or None), n1, kept1,
    n0, kept0; f and err nan when n1 or n0 is 0 or no merger segment is kept"""
    merger, keep = np.asarray(merger, bool), np.asarray(keep, bool)
    n1, n0 = int(merger.sum()), int((~merger).sum())
    a1, a0 = int((keep & merger).sum()), int((keep & ~merger).sum())
    out = dict(n1=n1, kept1=a1, n0=n0, kept0=a0, f=np.nan, err=np.nan, upper=None)
    if n1 == 0 or n0 == 0 or a1 == 0 or a0 == 0:
        return out
    k1, k0 = a1 / n1, a0 / n0
    r = k1 / k0
    out.update(f=1.0 - r, err=r * float(np.sqrt((1.0 - k1) / (n1 * k1) + (1.0 - k0) / (n0 * k0))),
               upper=3.0 / n1 if a1 == n1 else None)
    return out


def veto(d, veto_w=VETO_W, w_net=W_NET):
    """the segments the loudness veto and the network cut keep: a segment is set aside when wmax_H > veto_w or
    wmax_L > veto_w or wnet > w_net.  w_net None or inf: no network cut (the single-detector veto alone, as before it);
    data without wnet (files written before it was computed): the single-detector veto alone, and note says so.
    returns dict keep, loud_H, loud_L, net (n,) bool, applied (the network cut acted), note (None or why it did not)"""
    loud = {I: np.asarray(d['wmax_' + I]) > veto_w for I in 'HL'}
    n = loud['H'].size
    note = None
    if w_net is None or w_net == np.inf:
        note = 'off (w_net %s): the single-detector veto alone' % w_net
    elif 'wnet' not in d:
        note = ('requested (wnet > %g) but NOT APPLIED: these files carry no wnet (written before it was computed); '
                'the single-detector veto alone' % w_net)
    net = np.zeros(n, bool) if note else np.asarray(d['wnet']) > w_net
    return dict(keep=~(loud['H'] | loud['L'] | net), loud_H=loud['H'], loud_L=loud['L'], net=net, applied=note is None,
                note=note)


def heavy_tail(ln_x):
    """the number of segments that carry half of sum |x - 1| (x = exp(ln_x)), the largest first"""
    dev = np.sort(np.abs(_excess(ln_x)))[::-1]
    if not dev.size or not dev.sum() > 0:
        return 0
    return int(np.searchsorted(np.cumsum(dev), 0.5 * dev.sum())) + 1


def model_lines(directory, cfg):
    """the window model of the files in words: its SHA-256 and, when <directory>/glitch_model.json is that model, its
    states (fraction of segments, glitches per segment n_window q) per detector"""
    sha = cfg.get('glitch_model_sha256')
    path = os.path.join(directory, 'glitch_model.json')
    head = '\nwindow model of the files: SHA-256 %s, n_window %s' % (sha, cfg.get('n_window'))
    if not os.path.exists(path):
        return [head + '; no glitch_model.json in the folder to describe it'], None
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import plant
    with open(path) as fh:
        model = json.load(fh)
    if plant.model_sha256(model) != model.get('sha256') or model.get('sha256') != sha:
        return [head + '; glitch_model.json in the folder is another model (SHA-256 %s): not described'
                % model.get('sha256')], None
    lines = [head + ' (glitch_model.json in the folder), zone %s, fitted bins rho %g to %g'
             % (model['zone'], model['settings']['fitted_edges'][0], model['settings']['fitted_edges'][-1])]
    for I in 'HL':
        f, q = np.asarray(model[I]['f']), np.asarray(model[I]['q'])
        lines.append('  %s: %s; glitches per segment %.4g' % (I, ', '.join(
            'state %d fraction %.4g, %.4g glitches per segment' % (s + 1, f[s], model['n_window'] * q[s])
            for s in range(f.size)), model['n_window'] * float(f @ q)))
    return lines, model


def stage_bins(edges, rho_min, rho_cut):
    """(n_rho,) bool: the amplitude bins that overlap [rho_min, rho_cut), the only ones stage 1 is given"""
    return (edges[1:] > rho_min) & (edges[:-1] < rho_cut)


def fits(d, edges, rho_min, rho_min_fit=False, cut=None, sel=None):
    """every estimate of nbar: gauss, smith_thrane, joint (the power law from rho_min, ended by cut), stage1
    (fit_noise per detector on the bins sel), two_stage, two_stage_eps (its weights, zero outside sel)"""
    start = dict(START, rho_min_H=rho_min, rho_min_L=rho_min)
    sel = np.ones(edges.size - 1, bool) if sel is None else sel
    out = dict(gauss=gauss_fit(d, edges, start),
               smith_thrane=law_fit(d, edges, start, rho_min_fit, sg=False, cut=cut),
               joint=law_fit(d, edges, start, rho_min_fit, cut=cut))
    s1 = {I: mixture.fit_noise(d['ln_glitch_' + I][:, sel], log=True) for I in 'HL'}
    pk = []
    for I in 'HL':
        pk.append(np.zeros(edges.size - 1))
        pk[-1][sel] = s1[I]['pi_k']
    s2 = dict(start, eps_H=s1['H']['eps'], eps_L=s1['L']['eps'])
    out.update(stage1=s1, two_stage=fit(d, edges, ['nbar'], s2, pi_k=pk),
               two_stage_eps=fit(d, edges, ['nbar', 'eps_H', 'eps_L'], s2, pi_k=pk))
    return out


def ln_B(d, edges, eps_H, eps_L, pi_H, pi_L):
    """ln B (n,) of the mixture with the noise model (eps_I, amplitude weights pi_I)"""
    lnG = [mixture.ln_glitch_evidence(d['ln_glitch_' + I], p) for I, p in (('H', pi_H), ('L', pi_L))]
    return mixture.ln_mixture(d['ln_lambda_bar'], d['ln_lambda_gH_bar'], d['ln_lambda_gL_bar'], lnG[0], lnG[1], eps_H,
                              eps_L, ln_lam_gHL=d['ln_lambda_gHL_bar'])


def planted_law(edges, cfg, cut=None):
    """the amplitude weights of the planted law, dN/drho ~ rho^-gamma on [rho_min, rho_max], ended by cut as the
    fitted law is"""
    cut = cut or {}
    top = min(cfg['rho_max'], cut.get('rho_max') or np.inf)
    return mixture.amplitude_weights(edges, cfg['gamma'], cfg['rho_min'], top, cut.get('veto'))


def fitted_law(edges, r, I, cut=None):
    return mixture.amplitude_weights(edges, r['gamma_' + I], r['rho_min_' + I], **(cut or {}))


def mean_se(x):
    """(mean, standard error, number) of x"""
    x = np.asarray(x, float)
    if x.size < 2:
        return (float(x.mean()) if x.size else np.nan), np.nan, x.size
    with np.errstate(over='ignore', invalid='ignore'):
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)), x.size


def _excess(lnx):
    with np.errstate(over='ignore'):
        return np.exp(np.asarray(lnx, float)) - 1.0


def unit_table(d, rows, glitch_known=True):
    """the mean of x - 1 for each of rows [(name, ln x)] over the segments with no merger signal, split by glitch where
    known (a glitch: one or more in the segment), and the number of segments that carry half of sum |x - 1|:
    ({sample: {name: mean_se}}, {sample: {name: heavy_tail}})"""
    quiet = ~(d['merger_H'] | d['merger_L'])
    glitched = d['glitch_H'] | d['glitch_L']
    samples = (('all', quiet), ('with a glitch', quiet & glitched), ('without', quiet & ~glitched)) if glitch_known \
        else (('all', quiet),)
    return ({sample: {name: mean_se(_excess(v[sel])) for name, v in rows} for sample, sel in samples},
            {sample: {name: heavy_tail(v[sel]) for name, v in rows} for sample, sel in samples})


def unit_mean(d, lnB_true, lnB_fit, glitch_known=True):
    """the mean of Lambda_bar - 1 and B - 1 over the segments with no merger signal: {sample: {name: mean_se}}; the
    split by glitch and the planted noise model (lnB_true) only where they are known"""
    rows = [('Gaussian Lambda_bar - 1', d['ln_lambda_bar'])] + (
        [] if lnB_true is None else [('B - 1, planted noise model', lnB_true)]) + [('B - 1, joint fit', lnB_fit)]
    return unit_table(d, rows, glitch_known)[0]


def _windows_fits(L, directory, cfg, k, edges, rho0, tr, f_det=0.0):
    """the fits of the files of the window model (report): the model in words, the Gaussian likelihood and nbar_fit on
    the kept segments k (with the detected fraction f_det), appended to the lines L.  returns (fits, scan [(rho0,
    nbar_fit, time curve of Bbar)], None, ln Bbar, the model or None)"""
    lines, model = model_lines(directory, cfg)
    L += lines
    F = dict(gauss=gauss_fit(k, edges, dict(START, rho_min_H=rho0, rho_min_L=rho0)),
             windows=nbar_fit(k['ln_B_bar'], f_det))
    L.append('\nfits on the kept segments; nbar truth %s; the glitch parameters of the window model fixed (stage 1):'
             % ('not known' if tr['nbar'] is None else '%g' % tr['nbar']))
    for key, name in (('gauss', 'Gaussian likelihood (eps = 0)'), ('windows', 'window model, ln(1 + nbar (Bbar - 1))'
                                                                   + ('' if f_det == 0 else ', f %g' % f_det))):
        r = F[key]
        e = r['err'].get('nbar', np.nan)
        pull = '' if tr['nbar'] is None else '   pull %s' % _f(_pull(r['nbar'], e, tr['nbar']), '%+.2f')
        edge = '   (at the %s edge)' % r['edge'] if r.get('edge') else ''
        L.append('  %-44s %s +- %s%s   ln L %.3f%s' % (name, _f(r['nbar']), _f(e), pull, r['lnl'], edge))
    curve = mixture.time_curve(k['ln_B_bar'], NBAR_GRID, lambda g: 1.0 / g, log=True)
    return F, [(rho0, F['windows'], curve)], None, k['ln_B_bar'], model


def _f(x, fmt='%.5g'):
    return 'nan' if x is None or not np.isfinite(x) else fmt % x


def _pull(est, err, tr):
    return (est - tr) / err if err and np.isfinite(err) and err > 0 else np.nan


def describe(d, i, glitch_known=True, merger_known=True):
    """the truth of segment i in words (what is known of it)"""
    m = ('merger (optimal SNR %.2f, z %.2f)' % (d['snr'][i], d['z'][i])) if d['merger'][i] else 'no merger' \
        if merger_known else 'merger not known'
    parts = ['k %d%s: %s' % (d['k'][i], ' (GPS %.0f)' % d['gps'][i] if 'gps' in d else '', m)]
    for I in 'HL':
        g = 'glitch not known' if not glitch_known else ('%s glitch rho %.2f' % (d['kind_' + I][i], d['rho_' + I][i])
                                                         if d['glitch_' + I][i] else 'no glitch')
        parts.append('%s %s, sqrt(2 wmax) %.2f' % (I, g, np.sqrt(2.0 * d['wmax_' + I][i])))
    return '; '.join(parts)


def cells_text(cfg, cells):
    """the (m1, z) cells without ln Lambda recorded by plant.py, in words"""
    if cells is None:
        return 'cells not recorded in these files'
    if not cells.size:
        return 'no cell'
    g = cfg['grid']
    m1 = np.geomspace(g['m1_range'][0], g['m1_range'][1], g['nm1'])[cells[:, 0]]
    z = np.linspace(0.02, g['zmax'], g['nz'])[cells[:, 1]]
    Md = 2.0 * m1 * (1.0 + z)
    return '%d cells, z %.3f to %.3f, m1 %.3f to %.3f, M_det %.1f to %.1f' % (
        cells.shape[0], z.min(), z.max(), m1.min(), m1.max(), Md.min(), Md.max())


def _consistent_with_zero(r):
    """nbar within PULL of its standard errors of 0, or at the lower edge"""
    e = r['err'].get('nbar', np.nan)
    return bool(r['nbar'] <= 1e-9 or r['nbar'] < PULL * e), e


def missing_text(k, expected=None):
    """the missing segments of the sorted indices k: '<n> missing in between', and with the expected total (k from 0 to
    expected - 1) '<n> missing in between, <a> at the start and <b> at the end of the <expected> expected'"""
    between = int(k[-1] - k[0] + 1 - k.size)
    if expected is None:
        return '%d missing in between' % between
    return '%d missing in between, %d at the start and %d at the end of the %d expected' % (
        between, k[0], expected - 1 - k[-1], expected)


def _lacking(directory, key, pattern='plant_*.npz'):
    """the number of files pattern of directory without the member key"""
    n = 0
    for path in glob.glob(os.path.join(directory, pattern)):
        with np.load(path) as f:
            n += key not in f.files
    return n


def net_lines(v, veto_w, w_net, merger=None, glitch=None):
    """the network cut in words (veto(): v), and what each condition sets aside, with the segments that hold a merger
    signal (merger, (n,) bool) or a glitch in H, in L (glitch, a pair) where those are known"""
    if not v['applied']:
        return ['  network cut %s' % v['note']]
    alone = v['net'] & ~v['loud_H'] & ~v['loud_L']
    L = ['  network cut: also set aside when wnet, the largest w_H(t + tau) + w_L(t) over the templates, every '
         'arrival time and every lag of the tables, exceeds %g (the tables hold 1e-3 in ln up to W = 84): %d segments, '
         '%d of them by this cut alone' % (w_net, v['net'].sum(), alone.sum())]
    cols = ['set aside'] + (['merger'] if merger is not None else []) + (['glitch H', 'glitch L'] if glitch else [])
    L.append('  %-46s' % 'by condition (a segment can meet several)' + ''.join('%11s' % c for c in cols))
    for name, sel in (('wmax_H > %g' % veto_w, v['loud_H']), ('wmax_L > %g' % veto_w, v['loud_L']),
                      ('wnet > %g' % w_net, v['net']),
                      ('wnet > %g alone (both wmax at most %g)' % (w_net, veto_w), alone),
                      ('any of the three (set aside)', ~v['keep'])):
        nums = [sel.sum()] + ([(sel & merger).sum()] if merger is not None else []) + (
            [(sel & glitch[0]).sum(), (sel & glitch[1]).sum()] if glitch else [])
        L.append('    %-44s' % name + ''.join('%11d' % x for x in nums))
    return L


def fraction_lines(det, stat, f_det=0.0, truth=None):
    """the detected fraction of the selection in words and nbar_fit of the statistic stat = (ln B (n_kept,), its name)
    for f = 0, for f_det when given and for the fraction measured on the plant's truth (det['measured'], of
    detected_fraction, or None) when known, with the pull against truth (None: not known); the fits go into
    det['fits'] {label: nbar_fit}"""
    lnB, name = stat
    m = det['measured']
    L = ['\ndetected fraction of the selection (the paper\'s Eq. 35): f = 1 - P(kept | merger) / P(kept | no merger); '
         'the kept segments have the likelihood prod_i [1 + nbar (B_i - 1)] / (1 - nbar f)^n_kept, and the mean of B - 1 '
         'over kept segments without a merger is -f']
    if m is None:
        L.append('  not measured on these data (no planted merger): for real data f comes from a plant with the same '
                 'cut (--f-det)')
    elif not np.isfinite(m['f']):
        L.append('  measured: not defined (kept: mergers %d of %d, merger-free %d of %d)'
                 % (m['kept1'], m['n1'], m['kept0'], m['n0']))
    else:
        L.append('  measured on the truth: f = %s +- %s (binomial); kept: mergers %d of %d, merger-free %d of %d%s'
                 % (_f(m['f'], '%.4g'), _f(m['err'], '%.2g'), m['kept1'], m['n1'], m['kept0'], m['n0'],
                    '; no merger set aside: f below %.3g at 95 percent (rule of three)' % m['upper']
                    if m['upper'] is not None else ''))
    rows = [('f = 0', 0.0)] + ([('f = %g, given' % f_det, f_det)] if f_det != 0 else []) + (
        [('f = %.4g, measured' % m['f'], m['f'])] if m is not None and np.isfinite(m['f']) else [])
    det['fits'] = {}
    for label, f in rows:
        r = det['fits'][label] = nbar_fit(lnB, f)
        e = r['err']['nbar']
        pull = '' if truth is None else '   pull %s' % _f(_pull(r['nbar'], e, truth), '%+.2f')
        L.append('  %-60s %s +- %s%s%s' % ('%s, %s' % (name, label), _f(r['nbar']), _f(e), pull,
                                           '   (at the %s edge)' % r['edge'] if r['edge'] else ''))
    return L


def report(directory, rho_min_fit=False, veto_w=VETO_W, w_inf_max=W_INF_MAX, rho_mins=None, veto_law='hard',
           veto_offset=VETO_OFFSET, veto_width=VETO_WIDTH, monitor_cut=None, expected=None, w_net=W_NET, f_det=0.0):
    """everything plant_fit prints: dict with lines (the text), kind, counts, keep, rho_cut, fits, scan, unit, curves,
    criteria [(name, ok, text, counted)], ok (every counted criterion passes).  rho_mins: lower ends of the fitted
    glitch law, the first for every fit and criterion, all of them for the joint fit side by side (default: the
    planted rho_min, or 5, 4.5, 6 for data whose glitches are not known).  veto_law 'hard': the fitted laws stop at
    rho_cut = sqrt(2 veto_w); 'soft': they are multiplied by the probability of passing the veto, Phi((rho_cut - rho -
    veto_offset) / veto_width) (mixture.amplitude_weights).  With either, the fitted slopes of the glitch law are biased
    (gamma +0.27, 6 sigma, at 300000 toy segments, none when the veto acts on the true SNR) while nbar and eps are not:
    the per-bin evidences assume the amplitude uniform inside a bin, and the veto on the noisy measured SNR changes that
    next to the cut (the conditioning of docs/glitch_likelihood.md, Sec. 8).  expected: the number of segments the
    campaign should have (k = 0 .. expected - 1; real data, where the files do not carry their own range): the
    segments missing at the start and at the end are reported too, not only those in between.  w_net: the network cut,
    a segment is also set aside when wnet > w_net (veto; None or inf: off, today's selection; files without wnet: said,
    and not applied).  f_det: the detected fraction of the selection in the estimate of the window model (nbar_fit;
    default 0, the estimate without it); on a plant the fraction measured on its truth (detected_fraction) is shown
    beside it, with nbar for f = 0 and for the measured f"""
    cfg, edges, d, runs, cells = load(directory)
    planted, kind = truth(cfg)
    glitch_known = cfg.get('glitch_truth', True)
    merger_known = cfg.get('merger_truth', 'planted') != 'none'
    if rho_mins is None:
        rho_mins = [cfg['rho_min']] if glitch_known else [5.0, 4.5, 6.0]
    rho0 = float(rho_mins[0])
    n = d['k'].size
    L = []
    gaps = missing_text(d['k'], expected)
    if cfg.get('source') == 'strain':
        what = {'recorded': 'THE WEEK AS RECORDED: no truth, no pull and no criterion on nbar',
                'control': 'a control, no coherent merger: nbar truth 0, the frozen criteria of docs/real_data.md',
                'plant': 'mergers injected at nbar %s into the week slid by %+g s' % (cfg['nbar'], cfg['shift'] or 0)}
        L.append('recorded strain %s (manifest sha256 %s), variant %s, seed %d: %s'
                 % (cfg['segments'], cfg['manifest_sha256'][:12], cfg['variant'], cfg['seed'], what[kind]))
    else:
        L.append('campaign: seed %d, control %s, nbar %g, eps_H %g, eps_L %g, glitches %s, dN/drho ~ rho^-%g on '
                 '[%g, %g]' % (cfg['seed'], cfg['control'], cfg['nbar'], cfg['eps_H'], cfg['eps_L'],
                               cfg['glitch_kind'], cfg['gamma'], cfg['rho_min'], cfg['rho_max']))
        if cfg.get('glitch_states'):
            st = cfg['glitch_states']
            L.append('glitches of the window model, %d windows: state (f, q) H %s, L %s' % (
                st['n_window'], ', '.join('(%g, %g)' % tuple(p) for p in st['H']),
                ', '.join('(%g, %g)' % tuple(p) for p in st['L'])))
    windows = 'ln_B_bar' in d
    L.append('%d segments in %d files, k %d to %d (%s)' % (n, len(runs), d['k'][0], d['k'][-1], gaps))
    t = {key: [r[key] for r in runs if r.get(key) is not None] for key in ('t_sim', 't_pe', 't_plain')}
    L.append('cost per segment: simulation %s s, analysis %s s, plain analysis %s s (mean over files)'
             % tuple(_f(np.mean(v) if v else np.nan, '%.4f') for v in t.values()))

    m, gH, gL = d['merger'], d['glitch_H'], d['glitch_L']
    nb = cfg['nbar'] or 0.0
    counts = [('segments', n, n)]
    if merger_known:
        counts += [('mergers planted', n * nb, m.sum()),
                   ('merger signals in H, in L', None, (d['merger_H'].sum(), d['merger_L'].sum()))]
    states = cfg.get('glitch_states') if glitch_known else None
    if states:                                  # the window model: a segment 'has a glitch' when it has one or more
        nw = states['n_window']
        for I in 'HL':
            f, q = np.asarray(states[I], float).T
            counts += [('glitches in %s (all windows)' % I, n * nw * float(f @ q), d['n_glitches_' + I].sum())]
            counts += [('segments of %s in state %d (q %g)' % (I, s + 1, q[s]), n * f[s],
                        (d['glitch_state_' + I] == s).sum()) for s in range(f.size)]
        eps = {I: float(np.asarray(states[I], float)[:, 0] @ (1.0 - (1.0 - np.asarray(states[I], float)[:, 1]) ** nw))
               for I in 'HL'}
        names_g = ('segments with a glitch in H', 'segments with a glitch in L')
    else:
        eps = {I: cfg['eps_' + I] for I in 'HL'} if glitch_known else None
        names_g = ('glitches in H', 'glitches in L')
    if glitch_known:
        counts += [(names_g[0], n * eps['H'], gH.sum()), (names_g[1], n * eps['L'], gL.sum()),
                   ('merger signal and glitch in H', n * nb * eps['H'], (d['merger_H'] & gH).sum()),
                   ('merger signal and glitch in L', n * nb * eps['L'], (d['merger_L'] & gL).sum()),
                   ('merger signal and glitches in both', n * nb * eps['H'] * eps['L'],
                    (d['merger_H'] & d['merger_L'] & gH & gL).sum())]
        if cfg['glitch_kind'] == 'mixed':
            for g_kind in np.unique(np.concatenate([d['kind_H'][gH], d['kind_L'][gL]])):
                counts.append(('glitches %s (H, L)%s' % (g_kind, ', loudest' if states else ''), None,
                               ((d['kind_H'] == g_kind).sum(), (d['kind_L'] == g_kind).sum())))
    L.append('\n%-36s %12s %14s' % ('all segments', 'planted', 'realized'))
    for name, exp, got in counts:
        L.append('%-36s %12s %14s' % (name, '' if exp is None else '%.1f' % exp,
                                      ', '.join(map(str, got)) if isinstance(got, tuple) else str(got)))
    if not (merger_known and glitch_known):
        L.append('(not known in these data: %s)' % ' and '.join(
            x for x, k in (('mergers', merger_known), ('glitches', glitch_known)) if not k))
    if merger_known or glitch_known:
        L.append('realized fractions: %s' % ', '.join(
            ([('mergers %.5f' % m.mean())] if merger_known else []) +
            (['glitches H %.5f, L %.5f' % (gH.mean(), gL.mean())] if glitch_known else [])))

    rho_cut = float(np.sqrt(2.0 * veto_w))
    sel_ = veto(d, veto_w, w_net)
    if not sel_['applied'] and w_net is not None and w_net != np.inf:
        sel_['note'] += ' (%d of %d files lack it)' % (_lacking(directory, 'wnet'), len(runs))
    loud = {I: sel_['loud_' + I] for I in 'HL'}
    keep = sel_['keep'].copy()
    noisy = {I: np.zeros(n, bool) for I in 'HL'}
    if monitor_cut is not None:
        if 'monitor_H' not in d or 'monitor_L' not in d:
            raise SystemExit('--monitor-cut needs the band-power monitors that analyse.py stores')
        for I in 'HL':
            mon = np.asarray(d['monitor_' + I], float)
            width = 0.5 * (np.quantile(mon, 0.84, axis=0) - np.quantile(mon, 0.16, axis=0))
            noisy[I] = ((mon - np.median(mon, axis=0)) / width).max(axis=1) >= monitor_cut
        keep &= ~(noisy['H'] | noisy['L'])
    out = ~keep
    signal = d['merger_H'] | d['merger_L']
    L.append('\nloudness veto: a segment is set aside when wmax_H or wmax_L exceeds %g (single-detector matched-filter '
             'SNR sqrt(2 w) above %.4g)' % (veto_w, rho_cut))
    by_truth = ([('glitch in H %d, glitch in L %d' % ((out & gH).sum(), (out & gL).sum()))] if glitch_known else []) + \
        ([('merger %d' % (out & signal).sum())] if merger_known else []) + \
        ([('neither glitch nor merger %d' % (out & ~gH & ~gL & ~signal).sum())]
         if glitch_known and merger_known else [])
    if monitor_cut is not None:
        L.append('band-power cut: a segment is also set aside when the whitened power of one of its bands, in either '
                 'detector, is %g or more widths above the median over the segments (width: half the 16 to 84 percent '
                 'range); H %d, L %d, either %d'
                 % (monitor_cut, noisy['H'].sum(), noisy['L'].sum(), (noisy['H'] | noisy['L']).sum()))
    L.append('  set aside %d of %d segments%s' % (out.sum(), n, '; by truth (a segment can count twice): ' + ', '.join(
        by_truth) if by_truth else ''))
    L.append('  by detector: H above %g in %d%s, L in %d%s'
             % (veto_w, loud['H'].sum(), ' (glitch in H %d)' % (loud['H'] & gH).sum() if glitch_known else '',
                loud['L'].sum(), ' (glitch in L %d)' % (loud['L'] & gL).sum() if glitch_known else ''))
    L += net_lines(sel_, veto_w, w_net, signal if merger_known else None, (gH, gL) if glitch_known else None)
    law_max = rho_cut if rho_cut < edges[-1] else None
    if veto_law == 'hard':
        cut = dict(rho_max=law_max)
        how = ('the fitted laws stop at rho_cut = sqrt(2 x %g) = %.4g, where a glitch is set aside about half the time '
               '(sqrt(2 wmax) is rho plus noise of unit width, plus the gain of the maximum over templates and times)'
               % (veto_w, rho_cut))
    elif veto_law == 'soft':
        cut = dict(veto=(rho_cut, veto_offset, veto_width))
        how = ('the fitted laws are multiplied by the probability that a glitch passes the veto, Phi((%.4g - rho - %g) '
               '/ %g) (soft cut; sqrt(2 wmax) taken as rho + %g + %g N(0, 1))' % (rho_cut, veto_offset, veto_width,
                                                                                 veto_offset, veto_width))
    else:
        raise ValueError("veto_law must be 'hard' or 'soft'")
    sel = stage_bins(edges, rho0, rho_cut)
    if not windows:
        L.append('  %s; stage 1 and the two-stage fits use the %d bins that overlap [%g, %.4g), rho %.4g to %.4g'
                 % (how, sel.sum(), rho0, rho_cut, edges[:-1][sel].min(), edges[1:][sel].max()))
    if glitch_known:
        for I in 'HL':
            g, rho = d['glitch_' + I], d['rho_' + I]
            near = g & (np.abs(rho - rho_cut) < 1.0)
            off = np.sqrt(2.0 * d['wmax_' + I][g]) - rho[g]
            L.append('  %s: planted glitches within 1 of rho_cut %d (set aside %d); rho above the cut but kept %d, '
                     'below it but set aside %d; sqrt(2 wmax) - rho over the glitches %s +- %s (sd)'
                     % (I, near.sum(), (near & loud[I]).sum(), (g & (rho > rho_cut) & ~loud[I]).sum(),
                        (g & (rho <= rho_cut) & loud[I]).sum(), _f(off.mean() if off.size else np.nan, '%+.2f'),
                        _f(off.std() if off.size > 1 else np.nan, '%.2f')))

    k = {key: v[keep] for key, v in d.items()}
    nk = int(keep.sum())
    tr = dict(planted, **({'eps_H': float(k['glitch_H'].mean()), 'eps_L': float(k['glitch_L'].mean())}
                          if glitch_known else {}))
    L.append('  kept %d segments%s' % (nk, '; realized fractions among them: %s' % ', '.join(
        ([('mergers %.5f' % k['merger'].mean())] if merger_known else []) +
        (['glitches H %.5f, L %.5f' % (tr['eps_H'], tr['eps_L'])] if glitch_known else []))
        if merger_known or glitch_known else ''))

    prior = lambda g: 1.0 / g
    if windows:
        F, scan, lnB_true, lnB_fit, model = _windows_fits(L, directory, cfg, k, edges, rho0, tr, f_det)
        largest = (('ln Lambda_bar', k['ln_lambda_bar']), ('ln Bbar, window model', k['ln_B_bar']))
        unit_rows = [('Gaussian Lambda_bar - 1', k['ln_lambda_bar']), ('Bbar - 1, window model', k['ln_B_bar'])]
        robust = ('windows', 'window model', 'window model, parameters of stage 1')
    else:
        if f_det != 0:
            raise SystemExit('%s: f_det %g is applied by nbar_fit, the estimate of the window model; these files '
                             'carry the mixture of one glitch per segment (its joint fit has no detected fraction): '
                             'give f_det 0' % (directory, f_det))
        F = fits(k, edges, rho0, rho_min_fit, cut, sel)
        L.append('\nfits on the kept segments; nbar truth %s; eps truth: %s; rho_min %s, the law up to %.4g%s:' % (
            'not known' if tr['nbar'] is None else '%g' % tr['nbar'],
            'the realized fraction of kept segments with a glitch' if glitch_known else 'not known',
            'free from %g' % rho0 if rho_min_fit else 'held at %g' % rho0, rho_cut,
            ' (soft)' if veto_law == 'soft' else ''))
        names = (('gauss', 'Gaussian likelihood (eps = 0)'), ('smith_thrane', 'Smith & Thrane (sg=False)'),
                 ('joint', 'mixture, joint power-law fit'), ('two_stage', 'mixture, two-stage (bins and eps fixed)'),
                 ('two_stage_eps', 'mixture, two-stage (bins fixed, eps free)'))
        for key, name in names:
            r = F[key]
            e = r['err'].get('nbar', np.nan)
            pull = '' if tr['nbar'] is None else '   pull %s' % _f(_pull(r['nbar'], e, tr['nbar']), '%+.2f')
            held = '   (law held: %s)' % ', '.join(r['held']) if r.get('held') else ''
            L.append('  %-44s %s +- %s%s   ln L %.3f%s' % (name, _f(r['nbar']), _f(e), pull, r['lnl'], held))
        for key, name in (('joint', 'joint fit'), ('smith_thrane', 'Smith & Thrane')):
            r = F[key]
            for I in 'HL':
                parts = []
                for p in ('eps_' + I, 'gamma_' + I, 'rho_min_' + I):
                    if p in r['free']:
                        e = r['err'][p]
                        known = '' if tr[p] is None else ' (truth %.4g, pull %s)' % (tr[p], _f(_pull(r[p], e, tr[p]),
                                                                                               '%+.2f'))
                        parts.append('%s %s +- %s%s' % (p, _f(r[p], '%.4g'), _f(e, '%.2g'), known))
                    else:
                        parts.append('%s %s (held)' % (p, _f(r[p], '%.4g')))
                L.append('  %-15s %s' % (name if I == 'H' else '', '; '.join(parts)))
        s1 = F['stage1']
        L.append('  %-15s %s' % ('stage 1', '; '.join(
            'eps_%s %s +- %s%s, %d iterations%s' % (
                I, _f(s1[I]['eps'], '%.4g'), _f(s1[I]['err_eps'], '%.2g'),
                '' if tr['eps_' + I] is None else ' (truth %.4g)' % tr['eps_' + I], s1[I]['n_iter'],
                '' if s1[I]['converged'] else ' NOT CONVERGED') for I in 'HL')))
        if glitch_known and cfg['glitch_kind'] != 'template':
            L.append('  (glitches outside the model: gamma and rho_min have no true value in the fitted family)')

        J = F['joint']
        law = planted_law(edges, cfg, cut) if glitch_known and not cfg.get('glitch_states') else None
        lnB_true = None if law is None else ln_B(k, edges, tr['eps_H'], tr['eps_L'], law, law)
        lnB_fit = ln_B(k, edges, J['eps_H'], J['eps_L'], fitted_law(edges, J, 'H', cut), fitted_law(edges, J, 'L', cut))
        scan = []
        for rm in sorted(set(float(x) for x in rho_mins)):
            r = J if rm == rho0 else law_fit(k, edges, dict(START, rho_min_H=rm, rho_min_L=rm), rho_min_fit,
                                             cut=cut)
            lb = lnB_fit if rm == rho0 else ln_B(k, edges, r['eps_H'], r['eps_L'], fitted_law(edges, r, 'H', cut),
                                                  fitted_law(edges, r, 'L', cut))
            scan.append((rm, r, mixture.time_curve(lb, NBAR_GRID, prior, log=True)))
        if len(scan) > 1:
            L.append('\nthe joint fit for each lower end of the glitch law (side by side):')
            L.append('  %-22s' % 'rho_min' + ''.join('%22g' % rm for rm, _, _ in scan))
            rows = [('nbar', lambda r, c: '%s +- %s' % (_f(r['nbar'], '%.4g'), _f(r['err'].get('nbar'), '%.2g')))]
            for p in ('eps_H', 'eps_L', 'gamma_H', 'gamma_L'):
                rows.append((p, lambda r, c, p=p: '%s +- %s' % (_f(r[p], '%.4g'),
                                                                _f(r['err'].get(p, np.nan), '%.2g'))))
            rows += [('ln L', lambda r, c: '%.3f' % r['lnl']),
                     ('evidence end, max', lambda r, c: '%+.2f, %+.2f' % (c[-1], c.max()))]
            for name, fn in rows:
                L.append('  %-22s' % name + ''.join('%22s' % fn(r, c) for _, r, c in scan))

        model = None
        largest = (('ln Lambda_bar', k['ln_lambda_bar']), ('ln B, planted noise model', lnB_true),
                   ('ln B, joint fit', lnB_fit))
        robust = ('joint', 'mixture', 'mixture, noise model of the joint fit')

    det = dict(measured=detected_fraction(d['merger'], keep) if kind == 'plant' and merger_known else None)
    stat = (k['ln_B_bar'], 'window model') if windows else (lnB_fit, 'mixture, noise model of the joint fit held')
    L += fraction_lines(det, stat, f_det, tr['nbar'])
    L.append('\nlargest among the kept segments:')
    for name, v in largest:
        if v is not None:
            i = int(np.argmax(v))
            L.append('  %-26s %10.3f   %s' % (name, v[i], describe(k, i, glitch_known, merger_known)))
    if windows:
        U, carry = unit_table(k, unit_rows, glitch_known)
        L.append('\nunit mean over the kept segments with no %smerger signal (Sec. 1): mean +- standard error (n), and '
                 'the number of segments that carry half of sum |x - 1| (heavy tails)'
                 % ('' if merger_known else 'known '))
        for sample, rows_ in U.items():
            for name, (mu, se, c) in rows_.items():
                L.append('  %-14s %-28s %s +- %s  (%d, %s sigma); half of sum |x - 1| in %d' % (
                    sample, name, _f(mu, '%+.3e'), _f(se, '%.2e'), c, _f(mu / se if se else np.nan, '%+.1f'),
                    carry[sample][name]))
    else:
        U = unit_mean(k, lnB_true, lnB_fit, glitch_known)
        L.append('\nunit mean over the kept segments with no %smerger signal (Sec. 1): mean +- standard error (n)%s'
                 % ('' if merger_known else 'known ', '; the planted noise model has the realized eps among them and '
                    'the planted law up to %.4g' % min(cfg['rho_max'], rho_cut) if glitch_known else ''))
        for sample, rows_ in U.items():
            for name, (mu, se, c) in rows_.items():
                L.append('  %-14s %-28s %s +- %s  (%d, %s sigma)' % (sample, name, _f(mu, '%+.3e'), _f(se, '%.2e'), c,
                                                                  _f(mu / se if se else np.nan, '%+.1f')))
    if 'ln_lambda_bar_plain' in d:
        diff = np.abs(d['ln_lambda_bar'] - d['ln_lambda_bar_plain'])
        L.append('  ln Lambda_bar with and without the glitch terms (control 4): largest difference %.3g' % diff.max())
    L.append('  w_inf, population weight on cells without ln Lambda: largest %.3g, mean %.3g; %s'
             % (d['w_inf'].max(), d['w_inf'].mean(), cells_text(cfg, cells)))

    rkey, rname, rcurve = robust
    curves = {'gauss': mixture.time_curve(k['ln_lambda_bar'], NBAR_GRID, prior, log=True),
              ('windows' if windows else 'mixture'): [c for rm, _, c in scan if rm == rho0][0]}
    L.append('\ntime curve over the kept segments, ln evidence ratio against no mergers (prior uniform in ln nbar on '
             '[%g, %g]):' % PRIOR)
    for key, name in (('gauss', 'Gaussian likelihood'), ('windows' if windows else 'mixture', rcurve)):
        c = curves[key]
        L.append('  %-40s end %+.3f, max %+.3f at segment %d of %d' % (name, c[-1], c.max(), np.argmax(c) + 1, nk))

    crit = []
    J = F[rkey]
    if kind == 'plant':
        e = J['err'].get('nbar', np.nan)
        crit.append(('%s nbar within %g sigma of the truth' % (rname, PULL),
                     bool(abs(J['nbar'] - tr['nbar']) < PULL * e),
                     'nbar %s +- %s, truth %g, pull %s' % (_f(J['nbar']), _f(e), tr['nbar'],
                                                           _f(_pull(J['nbar'], e, tr['nbar']), '%+.2f')), True))
    elif kind == 'control':
        for lik, key, counted in ((rname, rkey, True), ('Gaussian', 'gauss', cfg['control'] in PR_LABELS)):
            ok, e = _consistent_with_zero(F[key])
            crit.append(('%s: nbar consistent with 0' % lik, ok, 'nbar %s +- %s (within %g standard errors of 0, or '
                         'at the lower edge)' % (_f(F[key]['nbar']), _f(e), PULL), counted))
            c = curves['gauss' if key == 'gauss' else ('windows' if windows else 'mixture')]
            crit.append(('%s: evidence below ln 100' % lik, bool(c.max() < VILLE),
                         'max %+.3f, ln 100 = %.3f' % (c.max(), VILLE), counted))
    crit.append(('population covered: w_inf < %g' % w_inf_max, bool(d['w_inf'].max() < w_inf_max),
                 'largest w_inf %.3g' % d['w_inf'].max(), True))
    L.append('')
    if kind == 'recorded':
        L.append('the week as recorded: no criterion on nbar (the controls decide whether it is looked at)')
    for name, ok, text, counted in crit:
        status = ('PASS' if ok else 'FAIL') if counted else (
            'expected to fail, and does (control of the test, not a failure of the run)' if not ok else
            'expected to fail, but passes (control of the test, not a failure of the run)')
        L.append('%s  %s: %s' % (status, name, text))
    ok = all(c[1] for c in crit if c[3])
    return dict(lines=L, cfg=cfg, kind=kind, truth=tr, counts=counts, keep=keep, rho_cut=rho_cut, bins=sel, fits=F,
                scan=scan, unit=U, curves=curves, criteria=crit, ok=ok, lnB_true=lnB_true, lnB_fit=lnB_fit, data=d,
                kept=k, edges=edges, rho_min=None if windows else rho0, model='windows' if windows else 'mixture',
                window_model=model, veto=sel_, detected=det, f_det=f_det)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('results', help='directory of the plant_*.npz files of one campaign')
    p.add_argument('--rho-min-fit', action='store_true', help='fit rho_min of each detector too')
    p.add_argument('--veto-w', type=float, default=VETO_W,
                   help='set aside a segment whose wmax_H or wmax_L exceeds this (inf: none)')
    p.add_argument('--w-net', type=float, default=W_NET,
                   help='network cut: also set aside a segment whose wnet, the largest w_H(t + tau) + w_L(t) of any '
                        'table read, exceeds this (default %g; inf: off, the single-detector veto alone; files without '
                        'wnet: said, and not applied)' % W_NET)
    p.add_argument('--f-det', type=float, default=0.0,
                   help='the detected fraction of the selection in the estimate of the window model (Eq. 35 of the '
                        'paper; default 0); a plant shows the one measured on its truth beside it')
    p.add_argument('--w-inf-max', type=float, default=W_INF_MAX, help='criterion on the largest w_inf')
    p.add_argument('--rho-min', type=float, nargs='+', default=None,
                   help='lower end(s) of the fitted glitch law: the first for every fit and criterion, all of them for '
                        'the joint fit side by side (default the planted value, or 5 4.5 6 for real data)')
    p.add_argument('--veto-law', choices=('hard', 'soft'), default='hard',
                   help='end of the fitted glitch law: hard cut at sqrt(2 veto_w), or the probability of passing it. '
                        'Neither removes the bias of the slopes (6 sigma at 300000 toy segments; none with a veto on '
                        'the true SNR): the evidences assume the amplitude uniform inside a bin, which the veto on the '
                        'noisy SNR changes next to the cut')
    p.add_argument('--monitor-cut', type=float, default=None,
                   help='also set aside a segment whose whitened power in one band of one detector is this many '
                        'widths (half the 16 to 84 percent range) above the median (needs the monitors of analyse.py)')
    p.add_argument('--veto-offset', type=float, default=VETO_OFFSET, help='soft: mean of sqrt(2 wmax) - rho')
    p.add_argument('--veto-width', type=float, default=VETO_WIDTH, help='soft: its standard deviation')
    p.add_argument('--expected', type=int, default=None,
                   help='real data: the number of segments of the campaign (k = 0 .. N - 1), to report those missing '
                        'at the start and at the end (planted campaigns carry their own range)')
    a = p.parse_args(argv)
    if not a.f_det < 1:
        p.error('need f-det < 1')
    r = report(a.results, a.rho_min_fit, a.veto_w, a.w_inf_max, a.rho_min, a.veto_law, a.veto_offset, a.veto_width,
               a.monitor_cut, a.expected, a.w_net, a.f_det)
    print('\n'.join(r['lines']), flush=True)
    return 0 if r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
