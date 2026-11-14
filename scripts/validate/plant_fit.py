"""The fits of a planted campaign of plant.py (controls 4 to 9 of docs/glitch_likelihood.md, Sec. 7), on the CPU, with
the truth alongside.  Exit status 1 when a criterion fails.

    python scripts/validate/plant_fit.py runs/template [--rho-min-fit]

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
  - pass or fail: the mixture nbar (joint fit) within 2.5 sigma of the truth; on control data (control not none, or
    nbar = 0) nbar consistent with 0 (below 2.5 sigma) and the mixture time curve below ln 100 everywhere (Ville's
    bound at 1 percent); every segment's population weight on cells without ln Lambda (w_inf) below 1e-4.
The truth of nbar is the --nbar of the campaign without a control and 0 with one.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exmarg.search import mixture

PRIOR = (1e-4, 0.5)
NBAR_GRID = np.geomspace(PRIOR[0], PRIOR[1], 600)
VILLE = np.log(100.0)
PULL = 2.5
W_INF_MAX = 1e-4
START = dict(nbar=0.01, eps_H=0.01, eps_L=0.01, gamma_H=3.0, gamma_L=3.0)
SHARED = ('rho_edges', 'glitch_pi_k', 'glitch_window')


def _equal(a, b):
    return (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b, equal_nan=True))


def load(directory):
    """(campaign settings, rho_edges, the per-segment arrays of every plant_*.npz of directory concatenated in order of
    k (the keys every file has), the 'run' entries of the files); the settings, rho_edges and the fiducial glitch law
    and window must agree across the files"""
    files = sorted(glob.glob(os.path.join(directory, 'plant_*.npz')))
    if not files:
        raise SystemExit('no plant_*.npz in %s' % directory)
    cfg, fixed, parts, runs = None, None, [], []
    for path in files:
        with np.load(path) as f:
            meta = json.loads(str(f['meta']))
            same = [np.asarray(f[key], float) if key in f.files else None for key in SHARED]
            arrays = {key: f[key] for key in f.files if key not in ('meta',) + SHARED}
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
    return cfg, fixed[0], data, runs


def truth(cfg):
    """the true parameters of the fit and whether the data are a control (no coherent merger)"""
    tr = dict(nbar=cfg['nbar'] if cfg['control'] == 'none' else 0.0, eps_H=cfg['eps_H'], eps_L=cfg['eps_L'],
              gamma_H=cfg['gamma'], gamma_L=cfg['gamma'], rho_min_H=cfg['rho_min'], rho_min_L=cfg['rho_min'])
    return tr, cfg['control'] != 'none' or cfg['nbar'] == 0


def fit(d, edges, free, start, sg=True, pi_k=(None, None)):
    """mixture.fit on the stored logarithms, the parameters free and the others at start"""
    return mixture.fit(d['ln_lambda_bar'], d['ln_lambda_gH_bar'], d['ln_lambda_gL_bar'], d['ln_glitch_H'],
                       d['ln_glitch_L'], edges, start=start, log=True,
                       fixed=tuple(p for p in mixture.PARAMETERS if p not in free), sg=sg,
                       lam_gHL=d['ln_lambda_gHL_bar'], pi_k_H=pi_k[0], pi_k_L=pi_k[1])


def law_fit(d, edges, start, rho_min_fit=False, sg=True):
    """nbar, eps_I and the power law of each detector free (rho_min too with rho_min_fit); the law of a detector whose
    eps ends at 0 is then held and the fit repeated.  returns the fit with 'held' the detectors whose law was held"""
    free = ['nbar', 'eps_H', 'eps_L', 'gamma_H', 'gamma_L'] + (['rho_min_H', 'rho_min_L'] if rho_min_fit else [])
    r = fit(d, edges, free, start, sg)
    held = [I for I in 'HL' if r['eps_' + I] <= 1e-9]
    if held:
        free = [p for p in free if not (p.split('_')[0] in ('gamma', 'rho') and p[-1] in held)]
        r = fit(d, edges, free, {p: r[p] for p in mixture.PARAMETERS}, sg)
    r['held'] = held
    return r


def fits(d, edges, cfg, rho_min_fit=False):
    """every estimate of nbar: gauss, smith_thrane, joint, stage1 (fit_noise per detector), two_stage, two_stage_eps"""
    start = dict(START, rho_min_H=cfg['rho_min'], rho_min_L=cfg['rho_min'])
    out = dict(gauss=fit(d, edges, ['nbar'], dict(start, eps_H=0.0, eps_L=0.0)),
               smith_thrane=law_fit(d, edges, start, rho_min_fit, sg=False),
               joint=law_fit(d, edges, start, rho_min_fit))
    s1 = {I: mixture.fit_noise(d['ln_glitch_' + I], log=True) for I in 'HL'}
    s2 = dict(start, eps_H=s1['H']['eps'], eps_L=s1['L']['eps'])
    pk = (s1['H']['pi_k'], s1['L']['pi_k'])
    out.update(stage1=s1, two_stage=fit(d, edges, ['nbar'], s2, pi_k=pk),
               two_stage_eps=fit(d, edges, ['nbar', 'eps_H', 'eps_L'], s2, pi_k=pk))
    return out


def ln_B(d, edges, eps_H, eps_L, pi_H, pi_L):
    """ln B (n,) of the mixture with the noise model (eps_I, amplitude weights pi_I)"""
    lnG = [mixture.ln_glitch_evidence(d['ln_glitch_' + I], p) for I, p in (('H', pi_H), ('L', pi_L))]
    return mixture.ln_mixture(d['ln_lambda_bar'], d['ln_lambda_gH_bar'], d['ln_lambda_gL_bar'], lnG[0], lnG[1], eps_H,
                              eps_L, ln_lam_gHL=d['ln_lambda_gHL_bar'])


def planted_law(edges, cfg):
    """the amplitude weights of the planted law, dN/drho ~ rho^-gamma on [rho_min, rho_max], on the bins"""
    return mixture.amplitude_weights(np.minimum(edges, cfg['rho_max']), cfg['gamma'], cfg['rho_min'])


def fitted_law(edges, r, I):
    return mixture.amplitude_weights(edges, r['gamma_' + I], r['rho_min_' + I])


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


def unit_mean(d, lnB_true, lnB_fit):
    """the mean of Lambda_bar - 1 and B - 1 over the segments with no merger signal: {sample: {name: mean_se}}"""
    quiet = ~(d['merger_H'] | d['merger_L'])
    glitched = d['glitch_H'] | d['glitch_L']
    out = {}
    for sample, sel in (('all', quiet), ('with a glitch', quiet & glitched), ('without', quiet & ~glitched)):
        out[sample] = {name: mean_se(_excess(v[sel])) for name, v in
                       (('Gaussian Lambda_bar - 1', d['ln_lambda_bar']), ('B - 1, planted noise model', lnB_true),
                        ('B - 1, joint fit', lnB_fit))}
    return out


def _f(x, fmt='%.5g'):
    return 'nan' if x is None or not np.isfinite(x) else fmt % x


def _pull(est, err, tr):
    return (est - tr) / err if err and np.isfinite(err) and err > 0 else np.nan


def report(directory, rho_min_fit=False):
    """everything plant_fit prints: dict with lines (the text), counts, fits, unit, curves, criteria [(name, ok, text)],
    ok"""
    cfg, edges, d, runs = load(directory)
    tr, control = truth(cfg)
    n = d['k'].size
    L = []
    gaps = int(d['k'][-1] - d['k'][0] + 1 - n)
    L.append('campaign: seed %d, control %s, nbar %g, eps_H %g, eps_L %g, glitches %s, dN/drho ~ rho^-%g on [%g, %g]'
             % (cfg['seed'], cfg['control'], cfg['nbar'], cfg['eps_H'], cfg['eps_L'], cfg['glitch_kind'],
                cfg['gamma'], cfg['rho_min'], cfg['rho_max']))
    L.append('%d segments in %d files, k %d to %d (%d missing in between)'
             % (n, len(runs), d['k'][0], d['k'][-1], gaps))
    t = {key: [r[key] for r in runs if r.get(key) is not None] for key in ('t_sim', 't_pe', 't_plain')}
    L.append('cost per segment: simulation %s s, analysis %s s, plain analysis %s s (mean over files)'
             % tuple(_f(np.mean(v) if v else np.nan, '%.4f') for v in t.values()))

    m, gH, gL = d['merger'], d['glitch_H'], d['glitch_L']
    counts = [('segments', n, n),
              ('mergers planted', n * cfg['nbar'], m.sum()),
              ('merger signals in H, in L', None, (d['merger_H'].sum(), d['merger_L'].sum())),
              ('glitches in H', n * cfg['eps_H'], gH.sum()),
              ('glitches in L', n * cfg['eps_L'], gL.sum()),
              ('merger signal and glitch in H', n * cfg['nbar'] * cfg['eps_H'], (d['merger_H'] & gH).sum()),
              ('merger signal and glitch in L', n * cfg['nbar'] * cfg['eps_L'], (d['merger_L'] & gL).sum()),
              ('merger signal and glitches in both', n * cfg['nbar'] * cfg['eps_H'] * cfg['eps_L'],
               (d['merger_H'] & d['merger_L'] & gH & gL).sum())]
    if cfg['glitch_kind'] == 'mixed':
        for kind in np.unique(np.concatenate([d['kind_H'][gH], d['kind_L'][gL]])):
            counts.append(('glitches %s (H, L)' % kind, None,
                           ((d['kind_H'] == kind).sum(), (d['kind_L'] == kind).sum())))
    L.append('\n%-36s %12s %14s' % ('', 'planted', 'realized'))
    for name, exp, got in counts:
        L.append('%-36s %12s %14s' % (name, '' if exp is None else '%.1f' % exp,
                                      ', '.join(map(str, got)) if isinstance(got, tuple) else str(got)))
    L.append('realized fractions: mergers %.5f, glitches H %.5f, L %.5f' % (m.mean(), gH.mean(), gL.mean()))

    F = fits(d, edges, cfg, rho_min_fit)
    L.append('\nnbar, truth %g (rho_min %s):' % (tr['nbar'], 'free' if rho_min_fit else 'held at %g' % cfg['rho_min']))
    names = (('gauss', 'Gaussian likelihood (eps = 0)'), ('smith_thrane', 'Smith & Thrane (sg=False)'),
             ('joint', 'mixture, joint power-law fit'), ('two_stage', 'mixture, two-stage (bins and eps fixed)'),
             ('two_stage_eps', 'mixture, two-stage (bins fixed, eps free)'))
    for key, name in names:
        r = F[key]
        e = r['err'].get('nbar', np.nan)
        L.append('  %-44s %s +- %s   pull %s   ln L %.3f%s' % (
            name, _f(r['nbar']), _f(e), _f(_pull(r['nbar'], e, tr['nbar']), '%+.2f'), r['lnl'],
            '   (law held: %s)' % ', '.join(r['held']) if r.get('held') else ''))
    for key, name in (('joint', 'joint fit'), ('smith_thrane', 'Smith & Thrane')):
        r = F[key]
        for I in 'HL':
            parts = []
            for p in ('eps_' + I, 'gamma_' + I, 'rho_min_' + I):
                if p in r['free']:
                    e = r['err'][p]
                    parts.append('%s %s +- %s (truth %g, pull %s)' % (p, _f(r[p], '%.4g'), _f(e, '%.2g'), tr[p],
                                                                     _f(_pull(r[p], e, tr[p]), '%+.2f')))
                else:
                    parts.append('%s %s (held)' % (p, _f(r[p], '%.4g')))
            L.append('  %-15s %s' % (name if I == 'H' else '', '; '.join(parts)))
    s1 = F['stage1']
    L.append('  %-15s %s' % ('stage 1', '; '.join(
        'eps_%s %s +- %s (truth %g, realized %.5f), %d iterations%s' % (
            I, _f(s1[I]['eps'], '%.4g'), _f(s1[I]['err_eps'], '%.2g'), tr['eps_' + I], d['glitch_' + I].mean(),
            s1[I]['n_iter'], '' if s1[I]['converged'] else ' NOT CONVERGED') for I in 'HL')))
    if cfg['glitch_kind'] != 'template':
        L.append('  (glitches outside the model: gamma and rho_min have no true value in the fitted family)')

    J = F['joint']
    lnB_true = ln_B(d, edges, tr['eps_H'], tr['eps_L'], planted_law(edges, cfg), planted_law(edges, cfg))
    lnB_fit = ln_B(d, edges, J['eps_H'], J['eps_L'], fitted_law(edges, J, 'H'), fitted_law(edges, J, 'L'))
    U = unit_mean(d, lnB_true, lnB_fit)
    L.append('\nunit mean over the segments with no merger signal (Sec. 1): mean +- standard error (n)')
    for sample, rows in U.items():
        for name, (mu, se, k) in rows.items():
            L.append('  %-14s %-28s %s +- %s  (%d, %s sigma)' % (sample, name, _f(mu, '%+.3e'), _f(se, '%.2e'), k,
                                                              _f(mu / se if se else np.nan, '%+.1f')))
    if 'ln_lambda_bar_plain' in d:
        diff = np.abs(d['ln_lambda_bar'] - d['ln_lambda_bar_plain'])
        L.append('  ln Lambda_bar with and without the glitch terms (control 4): largest difference %.3g' % diff.max())
    L.append('  w_inf, population weight on cells without ln Lambda: largest %.3g, mean %.3g'
             % (d['w_inf'].max(), d['w_inf'].mean()))

    prior = lambda g: 1.0 / g
    curves = dict(gauss=mixture.time_curve(d['ln_lambda_bar'], NBAR_GRID, prior, log=True),
                  mixture=mixture.time_curve(lnB_fit, NBAR_GRID, prior, log=True))
    L.append('\ntime curve, ln evidence ratio against no mergers (prior uniform in ln nbar on [%g, %g]):'
             % PRIOR)
    for key, name in (('gauss', 'Gaussian likelihood'), ('mixture', 'mixture, noise model of the joint fit')):
        c = curves[key]
        L.append('  %-40s end %+.3f, max %+.3f at segment %d of %d' % (name, c[-1], c.max(), np.argmax(c) + 1, n))

    e = J['err'].get('nbar', np.nan)
    crit = [('mixture nbar within %g sigma of the truth' % PULL, bool(abs(J['nbar'] - tr['nbar']) < PULL * e),
             'nbar %s +- %s, truth %g, pull %s' % (_f(J['nbar']), _f(e), tr['nbar'],
                                                   _f(_pull(J['nbar'], e, tr['nbar']), '%+.2f')))]
    if control:
        crit.append(('control: nbar consistent with 0', bool(J['nbar'] < PULL * e),
                     'nbar %s < %g x %s' % (_f(J['nbar']), PULL, _f(e))))
        crit.append(('control: time curve below ln 100', bool(curves['mixture'].max() < VILLE),
                     'max %+.3f, ln 100 = %.3f' % (curves['mixture'].max(), VILLE)))
    crit.append(('population covered: w_inf < %g' % W_INF_MAX, bool(d['w_inf'].max() < W_INF_MAX),
                 'largest w_inf %.3g' % d['w_inf'].max()))
    L.append('')
    for name, ok, text in crit:
        L.append('%s  %s: %s' % ('PASS' if ok else 'FAIL', name, text))
    ok = all(c[1] for c in crit)
    return dict(lines=L, cfg=cfg, truth=tr, counts=counts, fits=F, unit=U, curves=curves, criteria=crit, ok=ok,
                lnB_true=lnB_true, lnB_fit=lnB_fit, data=d, edges=edges)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('results', help='directory of the plant_*.npz files of one campaign')
    p.add_argument('--rho-min-fit', action='store_true', help='fit rho_min of each detector too')
    a = p.parse_args(argv)
    r = report(a.results, a.rho_min_fit)
    print('\n'.join(r['lines']), flush=True)
    return 0 if r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
