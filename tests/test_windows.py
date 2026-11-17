"""planted-truth tests of the model with many glitches per segment (docs/glitch_likelihood.md, Sec. 4a):
single.window_evidence, single.window_posterior and mixture.fit_windows.

The toy is that of test_mixture.py (one template, white noise z_I(t) complex Gaussian with E|z|^2 = 1, no delay),
with NT = 256 arrival times cut into NW = 32 windows of NWIN = 8.  Each detector-segment draws a state s (quiet or
noisy, probability f_s); every window then holds a glitch with probability q_s, at a uniform time in the window, of
amplitude rho uniform in a bin of EDGES drawn from the law pi^s; it adds sqrt(rho^2 / 2) e^{i phi} at that time.  A
merger (probability NBAR) adds sqrt(x) e^{i phi} at one time t to both detectors, x from (X_S, P_S); in the data
'model' it is not added to a detector that has a glitch in the zone of t (the window of t and its neighbours,
circular), as Sec. 4a states; in 'physical' it is always added.  Every likelihood ratio is exact, per time:

    S(t)   = sum_j p_j exp(-2 x_j) I0(2 sqrt(x_j) |z_H + z_L|),   S^I(t) = sum_j p_j exp(-x_j) I0(2 sqrt(x_j) |z_I|)
    E_w[k] = (1/NWIN) sum_{t in w} l_k(t),   l_k the glitch ratio of bin k (test_mixture.Toy, tabulated in |z|)
    B      = mean_t [S kH kL + S^L kL (1 - kH) + S^H kH (1 - kL) + (1 - kH)(1 - kL)]

kI = kbar_I of the window of t (window_posterior); there is no delay, so o_H = 1 - kbar_H.  B has mean 1 over
merger-free segments drawn from the model with its true parameters.  The unit mean is measured with weak mergers
(X_W), since Var_0 S = I0(4x) - 1 makes the mean over noise of the strong ones invisible to any sample; it is
measured against the control variate Lambda_S^0 (the same mean of S over the noise without the glitches, whose
mean is exactly 1), which removes the noise shared by both.  Nothing here uses mixture.py or single.py to build the
data or the evidences: those modules only fit and combine them.
"""
import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import i0e, logsumexp
from scipy.stats import chi2

from exmarg.search import mixture, single
from test_mixture import EDGES, P_S, X_S, Toy, law

NT, NW = 256, 32
NWIN = NT // NW
ZONE = 1
NK = EDGES.size - 1
TRUTH = dict(f=np.array([0.9, 0.1]), q=np.array([0.002, 0.15]), pi=np.array([law(EDGES, 3.0, 2.0), law(EDGES, 2.0, 2.0)]))
X_W, P_W = np.array([0.5, 1.0, 2.0]), np.array([0.5, 0.3, 0.2])
NBAR = 0.05
N_REC, N_UNIT, N_ONE, CHUNK = 40000, 120000, 20000, 5000
SEED = 20261115


# ----------------------------------------------------------------------------------------------- the toy


@lru_cache(maxsize=None)
def table():
    return Toy()


def merger(a, c, x_s, p_s):
    """sum_j p_j exp(-c x_j) I0(2 sqrt(x_j) a), per time"""
    out = np.zeros(a.shape)
    for x, p in zip(x_s, p_s):
        arg = 2.0 * np.sqrt(x) * a
        out += p * np.exp(arg - c * x) * i0e(arg)
    return out


def draws(n, seed):
    """the random numbers of n segments, shared by the data sets with and without mergers (seed an int or a
    SeedSequence)"""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    g = [np.random.default_rng(s) for s in ss.spawn(4)]
    noise = {I: (g[0].standard_normal((n, NT)) + 1j * g[0].standard_normal((n, NT))) / np.sqrt(2.0) for I in 'HL'}
    m = dict(u=g[1].random(n), t=g[1].integers(NT, size=n), ux=g[1].random(n), ph=np.exp(2j * np.pi * g[1].random(n)))
    gl = {I: dict(us=r.random(n), u=r.random((n, NW)), dt=r.integers(NWIN, size=(n, NW)), uk=r.random((n, NW)),
                  ur=r.random((n, NW)), ph=np.exp(2j * np.pi * r.random((n, NW)))) for I, r in zip('HL', g[2:])}
    return dict(noise=noise, m=m, g=gl)


def data(d, nbar, truth, mode='model', x_s=X_S, p_s=P_S):
    """z_H, z_L and the flags: state s, glitch per window g, merger m, and busy (a glitch in the zone of the merger)"""
    f, q, pi = (np.asarray(truth[k], float) for k in ('f', 'q', 'pi'))
    z = {I: d['noise'][I].copy() for I in 'HL'}
    fl = {}
    for I in 'HL':
        gI = d['g'][I]
        s = np.minimum(np.searchsorted(np.cumsum(f) / f.sum(), gI['us'], side='right'), f.size - 1)
        has = gI['u'] < q[s][:, None]
        cum = np.cumsum(pi, axis=1) / pi.sum(axis=1, keepdims=True)
        k = np.empty(has.shape, int)
        for j in range(f.size):
            k[s == j] = np.searchsorted(cum[j], gI['uk'][s == j], side='right')
        k = np.minimum(k, NK - 1)
        i, w = np.nonzero(has)
        rho = EDGES[k[i, w]] + gI['ur'][i, w] * (EDGES[k[i, w] + 1] - EDGES[k[i, w]])
        z[I][i, w * NWIN + gI['dt'][i, w]] += np.sqrt(0.5 * rho ** 2) * gI['ph'][i, w]
        fl['s' + I], fl['g' + I] = s, has
    m = d['m']
    fl['m'] = m['u'] < nbar
    i = np.nonzero(fl['m'])[0]
    x = x_s[np.searchsorted(np.cumsum(p_s) / p_s.sum(), m['ux'][i], side='right')]
    wm = m['t'][i] // NWIN
    for I in 'HL':
        busy = np.zeros(i.size, bool)
        for dw in range(-ZONE, ZONE + 1):
            busy |= fl['g' + I][i, (wm + dw) % NW]
        add = np.ones(i.size, bool) if mode == 'physical' else ~busy
        fl['busy' + I] = busy
        z[I][i[add], m['t'][i[add]]] += np.sqrt(x[add]) * m['ph'][i[add]]
    return z, fl


def _evidences(zH, zL, laws):
    a = {'H': np.abs(zH), 'L': np.abs(zL)}
    out = {}
    for I in 'HL':
        lk = table().ln_lk(a[I]).reshape(a[I].shape[0], NW, NWIN, NK)
        out['E' + I] = logsumexp(lk, axis=2) - np.log(NWIN)
    for name, (xs, ps) in laws.items():
        for key, v in (('S', merger(np.abs(zH + zL), 2.0, xs, ps)), ('SH', merger(a['H'], 1.0, xs, ps)),
                       ('SL', merger(a['L'], 1.0, xs, ps))):
            out[key + name] = np.log(v.reshape(-1, NW, NWIN).sum(axis=2))       # ln sum_{t in w} S
    return out


def evidences(z, laws, part=500, workers=8):
    """per segment: ln E_I (n, NW, NK) and, for every merger law of laws (name: (x, p)), ln sum_{t in w} of S, S^H,
    S^L (n, NW)"""
    n = z['H'].shape[0]
    parts = [slice(c, min(c + part, n)) for c in range(0, n, part)]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda s: _evidences(z['H'][s], z['L'][s], laws), parts))
    return {k: np.concatenate([r[k] for r in res]) for k in res[0]} if n else {}


def chunks(n, seed):
    return zip(range(0, n, CHUNK), np.random.SeedSequence(seed).spawn((n + CHUNK - 1) // CHUNK))


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


@lru_cache(maxsize=None)
def recovery_data():
    """N_REC segments with the strong mergers: 'free' without mergers, 'model' and 'physical' with mergers at NBAR
    (the same noise and glitches; only the segments with a merger are recomputed)"""
    out = {k: [] for k in ('free', 'model', 'physical', 'flags')}
    for a, ss in chunks(N_REC, SEED):
        d = draws(min(CHUNK, N_REC - a), ss)
        z, fl = data(d, 0.0, TRUTH)
        ev = evidences(z, dict(s=(X_S, P_S)))
        out['free'].append(ev)
        for mode in ('model', 'physical'):
            z, fl = data(d, NBAR, TRUTH, mode)
            i = np.nonzero(fl['m'])[0]
            e = {k: v.copy() for k, v in ev.items()}
            for k, v in evidences({I: z[I][i] for I in 'HL'}, dict(s=(X_S, P_S))).items():
                e[k][i] = v
            out[mode].append(e)
        out['flags'].append(fl)
    return {k: cat(v) for k, v in out.items()}


def post(ln_E, p, zone=ZONE):
    return single.window_posterior(ln_E, p['f'], p['q'], p['pi'], zone)


def post_prior(ln_E, p, zone=ZONE):
    """a wrong formula: kbar = sum_s f_s kappa^s, the prior of the state in place of its posterior"""
    one = [post(ln_E, dict(f=[1.0], q=[p['q'][s]], pi=[p['pi'][s]]), zone) for s in range(len(p['f']))]
    return {k: logsumexp(np.stack([np.log(p['f'][s]) + one[s][k] for s in range(len(p['f']))]), axis=0)
            for k in ('ln_kbar', 'ln_near')}


def ln_B(ev, name, pH, pL):
    """ln B per segment from the windows: the kbar of a window is that of each of its times"""
    kH, nH, kL, nL = pH['ln_kbar'], pH['ln_near'], pL['ln_kbar'], pL['ln_near']
    t = np.stack([kH + kL + ev['S' + name], nH + kL + ev['SL' + name], kH + nL + ev['SH' + name],
                  nH + nL + np.log(NWIN)])
    return logsumexp(t, axis=(0, 2)) - np.log(NT)


def nbar_fit(lnB):
    """the maximum of sum ln(1 + nbar (B - 1)) on [0, 1), and its error from the curvature there"""
    if np.sum(np.expm1(np.minimum(lnB, 700.0))) <= 0:
        x = 0.0
    else:
        x = minimize_scalar(lambda v: -mixture.log_likelihood(v, lnB, 0.0, 0.0, log=True), bounds=(0.0, 0.99),
                            method='bounded', options=dict(xatol=1e-10)).x
    with np.errstate(over='ignore', divide='ignore'):
        u = 1.0 / (x + 1.0 / np.expm1(lnB))                                     # (B - 1) / (1 + x (B - 1))
    return x, 1.0 / np.sqrt(np.sum(u * u))


def truth_lnl(ln_E):
    return math.fsum(post(ln_E, TRUTH)['ln_c'])


@lru_cache(maxsize=None)
def stage_one(mode):
    """fit_windows of each detector on recovery_data()[mode] (two states, a free weight per bin)"""
    ev = recovery_data()[mode]
    return {I: mixture.fit_windows(ev['E' + I]) for I in 'HL'}


def params(r):
    return dict(f=r['f'], q=r['q'], pi=r['pi'])


# ----------------------------------------------------------------------------------------------- a. window_evidence


def test_window_evidence():
    """window_evidence against the direct sum of the kernel (single.ln_kernel, summed in the log domain) over the
    times of each window and the templates (columns l(w; x), x up to 900, tabulated from it and read by the
    interpolation of ln_evidence), with loud times: 2e-5 (as in test_single) wherever E_w[k] is within e^20 of the
    largest column of its window, and in units of that column everywhere.  (On a window of quiet times the
    loud columns go through the branch of ln_evidence linear in T, whose relative error grows with x: 3e-4 at
    x = 200, 5e-3 at x = 900, on values ~e^-150 and e^-800 that do not count in L_w); the mean of E_w over the windows equals sum_m pi_m E[m] of ln_evidence on the
    whole segment to 1e-12, with given and with uniform template weights; n_time not a multiple raises"""
    rng = np.random.default_rng(3)
    nm, nt, nwin = 3, 96, 8
    w = rng.exponential(size=(nm, nt))
    w[0, 5], w[1, 40], w[2, 41], w[2, 95] = 30.0, 400.0, 6.0, 600.0
    ln_u = rng.normal(0.0, 1.0, nm)
    lq = (0.5 * (np.log(w) + ln_u[:, None])).astype(np.float32)
    w = np.exp(2.0 * lq.astype(float) - ln_u[:, None])                          # what the float32 trajectory holds
    xs = np.array([0.5, 4.0, 50.0, 900.0])
    lnT = single.ln_kernel(single.V[:, None] ** 2, xs[None, :])
    for pm in (np.array([0.5, 0.2, 0.3]), None):
        got = single.window_evidence(lq, ln_u, lnT, nt // nwin, pi_m=pm)
        p = np.full(nm, 1.0 / nm) if pm is None else pm
        lk = single.ln_kernel(w.reshape(nm, nt // nwin, nwin)[..., None], xs)              # (m, w, t, k)
        direct = logsumexp(lk + np.log(p)[:, None, None, None], axis=(0, 2)) - np.log(nwin)
        top = direct.max(axis=1, keepdims=True)
        err = np.abs(got - direct)[direct > top - 20.0].max()                     # where a column counts in L_w
        err_top = np.abs(np.exp(got - top) - np.exp(direct - top)).max()           # in units of the largest column
        whole = logsumexp(single.ln_evidence(lq, ln_u, lnT) + np.log(p)[:, None], axis=0)
        mean_err = np.abs(logsumexp(got, axis=0) - np.log(nt // nwin) - whole).max()
        print('window_evidence: shape %s, against the direct sum: max |d ln E_w| %.1e within e^20 of the largest column '
              'of the window, %.1e over all (%.1e in units of the largest); mean over windows against the whole segment '
              '%.1e' % (got.shape, err, np.abs(got - direct).max(), err_top, mean_err))
        assert got.shape == (nt // nwin, xs.size) and err < 2e-5 and err_top < 2e-5 and mean_err < 1e-12
    with pytest.raises(ValueError):
        single.window_evidence(lq, ln_u, lnT, 7)


# ----------------------------------------------------------------------------------------------- b. window_posterior


def brute(E, f, q, pi, zone):
    """c, r, P(zone of w free), P(zone of w not free) by enumerating every state and every window content (no glitch,
    or a glitch of column k), each a sum of positive terms (math.fsum)"""
    nw, nk = E.shape
    terms = []
    for s in range(len(f)):
        for g in itertools.product(range(nk + 1), repeat=nw):
            v = f[s]
            for j, gj in enumerate(g):
                v *= (1.0 - q[s]) if gj == 0 else q[s] * pi[s][gj - 1] * E[j, gj - 1]
            terms.append((s, g, v))
    c = math.fsum(t[2] for t in terms)
    r = np.array([math.fsum(t[2] for t in terms if t[0] == s) for s in range(len(f))]) / c
    free, busy = np.empty(nw), np.empty(nw)
    for j in range(nw):
        zj = [(j + dw) % nw for dw in range(-zone, zone + 1)]
        free[j] = math.fsum(t[2] for t in terms if all(t[1][i] == 0 for i in zj)) / c
        busy[j] = math.fsum(t[2] for t in terms if any(t[1][i] != 0 for i in zj)) / c
    return c, r, free, busy


def posterior_errors(E, f, q, pi, zone, zone_ref=None):
    """the largest error of window_posterior against brute: ln c, r, ln kbar, ln(1 - kbar), per segment of E
    (..., n_w, n_k)"""
    p = single.window_posterior(np.log(E), f, q, pi, zone)
    flat = E.reshape(-1, *E.shape[-2:])
    errs = []
    for i, Ei in enumerate(flat):
        c, r, free, busy = brute(Ei, f, q, pi, zone if zone_ref is None else zone_ref)
        idx = np.unravel_index(i, E.shape[:-2])
        with np.errstate(divide='ignore'):
            errs.append([abs(p['ln_c'][idx] - np.log(c)), np.abs(p['r'][idx] - r).max(),
                         np.abs(p['ln_kbar'][idx] - np.log(free)).max(), np.abs(p['ln_near'][idx] - np.log(busy)).max()])
    return np.max(errs, axis=0), p


CASES = dict(
    generic=dict(f=[0.7, 0.3], q=[0.05, 0.4]),
    empty_and_quiet=dict(f=[0.6, 0.4, 0.0], q=[0.0, 0.2, 0.5]),
)


def test_window_posterior_brute_force():
    """window_posterior against the enumeration of every glitch configuration (6 windows, 3 columns, 2 states and a
    case with a state without glitches and an empty state; zones 1 and 2; leading axes (2, 2)): ln c, r, ln kbar and
    ln(1 - kbar) to 1e-12.  The two limits: a window of evidence e^60 (kbar ~ e^-58 in its zone) and q = 1e-12,
    3e-12 on E = 1 (1 - kbar ~ 1e-11), each to 1e-12 in the logarithm, i.e. relative"""
    rng = np.random.default_rng(4)
    nw, nk = 6, 3
    lines, worst = [], 0.0
    for name, case in CASES.items():
        K = len(case['f'])
        pi = rng.dirichlet(np.ones(nk), K)
        for zone, lead in ((1, (2, 2)), (2, (1,))):
            E = np.exp(rng.normal(0.0, 1.5, lead + (nw, nk)))
            err, p = posterior_errors(E, case['f'], case['q'], pi, zone)
            lines.append('%s, zone %d: errors ln c %.1e, r %.1e, ln kbar %.1e, ln(1 - kbar) %.1e' % ((name, zone) + tuple(err)))
            worst = max(worst, err.max())
            assert p['ln_c'].shape == lead and p['r'].shape == lead + (K,) and p['ln_kbar'].shape == lead + (nw,)
    pi = rng.dirichlet(np.ones(nk), 2)
    E = np.ones((nw, nk))
    E[2] = np.exp(60.0)
    err, p = posterior_errors(E, [0.7, 0.3], [0.05, 0.4], pi, 1)
    lines.append('E = e^60 in window 2: ln kbar %s; errors %s' % (np.round(p['ln_kbar'], 3), np.array2string(err, precision=1)))
    worst = max(worst, err.max())
    assert p['ln_kbar'][2] < -55.0
    err, p = posterior_errors(np.ones((nw, nk)), [0.7, 0.3], [1e-12, 3e-12], pi, 1)
    lines.append('q = 1e-12, 3e-12: ln(1 - kbar) %s; errors %s' % (np.round(p['ln_near'], 4), np.array2string(err, precision=1)))
    worst = max(worst, err.max())
    print('\n' + '\n'.join(lines))
    assert worst < 1e-12, worst


def test_window_posterior_brute_force_detects_errors():
    """control of the comparison: the same enumeration against window_posterior with zone 0 in place of 1, or with
    q of one state off by 1 percent, differs by more than 1e-4"""
    rng = np.random.default_rng(5)
    pi = rng.dirichlet(np.ones(3), 2)
    E = np.exp(rng.normal(0.0, 1.5, (6, 3)))
    f, q = [0.7, 0.3], np.array([0.05, 0.4])
    zone_err = posterior_errors(E, f, q, pi, 0, zone_ref=1)[0]
    c, r, free, busy = brute(E, f, q * np.array([1.0, 1.01]), pi, 1)
    p = single.window_posterior(np.log(E), f, q, pi, 1)
    q_err = max(abs(p['ln_c'] - np.log(c)), np.abs(p['ln_kbar'] - np.log(free)).max())
    print('\nzone 0 against an enumeration with zone 1: %s; q off by 1%%: %.1e' % (np.array2string(zone_err, precision=1), q_err))
    assert zone_err[2:].max() > 1e-4 and q_err > 1e-4


# ----------------------------------------------------------------------------------------------- c. fit_windows


def test_fit_windows_recovers_truth():
    """fit_windows on the window evidences of N_REC merger-free segments, each detector (2 states, a free weight per
    bin, 27 parameters): converged; ln L equals sum ln c of window_posterior at the fitted parameters to 1e-9
    relative; the whole model, laws included: 0 <= 2 (ln L_max - ln L_truth) < chi2_27(0.999); f, q and the rate of
    glitches louder than rho = 6 in each state (q times the weight above 6) within 3.5 standard errors of the truth
    (from the Hessian).  The single weights are printed, not tested: neighbouring weak bins trade weight and several
    sit at zero, where the errors are not Gaussian"""
    ev = recovery_data()['free']
    bound = chi2.ppf(0.999, 1 + 2 * NK)
    loud = EDGES[:-1] >= 6.0
    P = NK + 1
    lines, bad = [], []
    for I in 'HL':
        r = stage_one('free')[I]
        lt = truth_lnl(ev['E' + I])
        lr = 2.0 * (r['lnl'] - lt)
        check = math.fsum(post(ev['E' + I], params(r))['ln_c'])
        pf = (r['f'] - TRUTH['f']) / r['err_f']
        pq = (r['q'] - TRUTH['q']) / r['err_q']
        rate, rate_true, err_rate = r['q'] * r['a'][:, loud].sum(axis=1), TRUTH['q'] * TRUTH['pi'][:, loud].sum(axis=1), []
        for s in range(2):
            i = 2 + s * P + 1 + np.nonzero(loud)[0]                             # these weights of Om[s] in cov
            err_rate.append(np.sqrt(r['cov'][np.ix_(i, i)].sum()))
        pr = (rate - rate_true) / np.array(err_rate)
        lines.append('%s: %d updates, converged %s, 2 dlnL against the truth %.1f (bound %.1f)\n  f %s +- %s (truth %s)\n'
                     '  q %s +- %s (truth %s)\n  rate above rho 6 %s +- %s (truth %s); weights at zero per state %s'
                     % (I, r['n_iter'], r['converged'], lr, bound, np.round(r['f'], 4), np.round(r['err_f'], 4), TRUTH['f'],
                        np.round(r['q'], 5), np.round(r['err_q'], 5), TRUTH['q'], np.round(rate, 5),
                        np.round(err_rate, 5), np.round(rate_true, 5), (r['a'] == 0).sum(axis=1)))
        for s in range(2):
            lines.append('  law %d: %s\n  truth   %s\n  error   %s' % (s, ' '.join('%.3f' % v for v in r['a'][s]),
                                                                      ' '.join('%.3f' % v for v in TRUTH['pi'][s]),
                                                                      ' '.join('%.3f' % v for v in r['err_a'][s])))
        ok = (r['converged'] and abs(check / r['lnl'] - 1) < 1e-9 and 0.0 <= lr < bound and np.all(np.abs(pf) < 3.5)
              and np.all(np.abs(pq) < 3.5) and np.all(np.abs(pr) < 3.5))
        if not ok:
            bad.append(I)
    print('\n' + '\n'.join(lines))
    assert not bad, lines


def test_fit_windows_monotone():
    """ln L never decreases along the updates of fit_windows, from its default start and from a poor one (q 0.3 and
    1e-4, laws on the loudest bin); both end at the same maximum to 1e-6 nats"""
    E = recovery_data()['free']['EH']
    r0 = stage_one('free')['H']
    a = np.full((2, NK), 1e-3)
    a[:, -1] = 1.0 - 1e-3 * (NK - 1)
    r1 = mixture.fit_windows(E, start=dict(f=[0.5, 0.5], q=[0.3, 1e-4], a=a))
    msg = []
    for tag, r in (('default start', r0), ('poor start', r1)):
        d = np.diff(r['trace'])
        msg.append('%s: %d updates, ln L from %.3f to %.6f, smallest step %.2e, converged %s'
                   % (tag, r['n_iter'], r['trace'][0], r['lnl'], d.min(), r['converged']))
    print('\n' + '\n'.join(msg))
    assert np.all(np.diff(r0['trace']) >= 0) and np.all(np.diff(r1['trace']) >= 0), msg
    assert r0['converged'] and r1['converged'] and abs(r0['lnl'] - r1['lnl']) < 1e-6, msg


@lru_cache(maxsize=None)
def one_state_fits():
    one = dict(f=np.array([1.0]), q=np.array([0.05]), pi=TRUTH['pi'][1:])
    parts = []
    for a, ss in chunks(N_ONE, SEED + 3):
        z, _ = data(draws(min(CHUNK, N_ONE - a), ss), 0.0, one)
        parts.append(evidences(z, {}))
    E = cat(parts)['EH']
    return E, mixture.fit_windows(E, n_state=1), mixture.fit_windows(E, n_state=2)


def empty_or_equal(E, r1, r2):
    """the two-state fit is the one-state fit within the noise: 0 <= dlnL < chi2_14(0.999) / 2, the glitch rate f . q
    within 3 sigma of q of one state, and each state empty (f < 0.01) or with q within 3 sigma of it; and the two
    describe the segments alike, mean |d ln c_i| < 0.05"""
    d = r2['lnl'] - r1['lnl']
    rate = r2['f'] @ r2['q']
    eq = [r2['f'][s] < 0.01 or abs(r2['q'][s] - r1['q'][0]) < 3.0 * np.hypot(r2['err_q'][s], r1['err_q'][0])
          for s in range(2)]
    dc = np.abs(post(E, params(r2))['ln_c'] - post(E, params(r1))['ln_c']).mean()
    msg = ('dlnL %.2f (bound %.1f), f %s, q %s +- %s, one state q %.4f +- %.4f, rate f.q %.4f, mean |d ln c| %.4f'
           % (d, chi2.ppf(0.999, 1 + NK) / 2, np.round(r2['f'], 3), np.round(r2['q'], 4), np.round(r2['err_q'], 4),
              r1['q'][0], r1['err_q'][0], rate, dc))
    ok = (-1e-6 <= d < chi2.ppf(0.999, 1 + NK) / 2 and abs(rate - r1['q'][0]) < 3.0 * r1['err_q'][0] and all(eq)
          and dc < 0.05)
    return ok, msg


def test_fit_windows_one_state():
    """data from one state (q 0.05, the noisy law, N_ONE segments) fitted with two states: the second state is empty or
    equal within the noise (empty_or_equal); the maximum of a split always exists, and fits the noise by a few nats.
    Control: the same criterion fails on the two-state data of the recovery tests"""
    E, r1, r2 = one_state_fits()
    ok, msg = empty_or_equal(E, r1, r2)
    E2 = recovery_data()['free']['EH']
    ok2, msg2 = empty_or_equal(E2, mixture.fit_windows(E2, n_state=1), stage_one('free')['H'])
    print('\none state: %s (converged %s %s)\ntwo states (control): %s' % (msg, r1['converged'], r2['converged'], msg2))
    assert ok and r1['converged'] and r2['converged'] and not ok2, (msg, msg2)


# ----------------------------------------------------------------------------------------------- d. unit mean


@lru_cache(maxsize=None)
def unit_data():
    """N_UNIT merger-free segments with the weak mergers: the window evidences, ln sum_w S, and Lambda_S^0, the mean
    of S over the times on the noise without the glitches"""
    parts = []
    for a, ss in chunks(N_UNIT, SEED + 1):
        d = draws(min(CHUNK, N_UNIT - a), ss)
        z, _ = data(d, 0.0, TRUTH)
        ev = evidences(z, dict(w=(X_W, P_W)))
        ev['S0'] = merger(np.abs(d['noise']['H'] + d['noise']['L']), 2.0, X_W, P_W).mean(axis=1)
        parts.append(ev)
    return cat(parts)


def unit_mean(p_H, p_L):
    """B per segment with the posteriors of each detector, and the statistics of its mean"""
    ev = unit_data()
    B = np.exp(ln_B(ev, 'w', p_H, p_L))
    D = B - ev['S0']
    n = B.size
    se, se_d = B.std() / np.sqrt(n), D.std() / np.sqrt(n)
    batch = D.reshape(20, -1).mean(axis=1).std() / np.sqrt(20)
    dev = np.sort(np.abs(B - 1.0))[::-1]
    carry = int(np.searchsorted(np.cumsum(dev), 0.5 * dev.sum())) + 1
    return dict(B=B, mean=B.mean(), se=se, cv=D.mean() + 1.0, se_cv=se_d, batch=batch, carry=carry, max=B.max())


def show_mean(tag, u):
    return ('%s: mean B %.5f +- %.5f (%+.1f se); with the control variate %.5f +- %.5f (%+.1f se, batch means %.5f); '
            'half of sum |B - 1| in %d of %d segments, largest B %.3g' % (
                tag, u['mean'], u['se'], (u['mean'] - 1) / u['se'], u['cv'], u['se_cv'], (u['cv'] - 1) / u['se_cv'],
                u['batch'], u['carry'], u['B'].size, u['max']))


def test_unit_mean():
    """the central requirement: on N_UNIT merger-free segments drawn from the model, B with the true parameters has
    mean 1 within 3 standard errors, both raw and against the control variate, whose error must be below 1e-3
    (measured 4.3e-4, which makes a bias of 2e-3 a 4.7-sigma effect)"""
    ev = unit_data()
    u = unit_mean(post(ev['EH'], TRUTH), post(ev['EL'], TRUTH))
    msg = show_mean('true parameters', u)
    print('\n' + msg)
    assert abs(u['mean'] - 1) < 3 * u['se'] and abs(u['cv'] - 1) < 3 * u['se_cv'] and u['se_cv'] < 1e-3, msg


@lru_cache(maxsize=None)
def old_model():
    """the model of Sec. 4 (one glitch per detector and segment, W = 0) on the same segments: eps and the law of each
    detector from fit_noise on the whole-segment evidences, then its B (mixture.ln_mixture) and nbar (mixture.fit,
    eps and the law held)"""
    ev = unit_data()
    fn = {I: mixture.fit_noise(logsumexp(ev['E' + I], axis=1) - np.log(NW), log=True) for I in 'HL'}
    weak = Toy(EDGES, X_W, P_W, NT)
    laws = [{I: fn[I]['pi_k'] for I in 'HL'}]
    parts = []
    for a, ss in chunks(N_UNIT, SEED + 1):
        z, _ = data(draws(min(CHUNK, N_UNIT - a), ss), 0.0, TRUTH)
        o = weak.evidences(z, laws)
        parts.append({k: o[k][0] if k in ('gH', 'gL', 'HL') else o[k] for k in ('S', 'EH', 'EL', 'gH', 'gL', 'HL')})
    o = cat(parts)
    lnG = {I: mixture.ln_glitch_evidence(o['E' + I], fn[I]['pi_k']) for I in 'HL'}
    lnB = mixture.ln_mixture(o['S'], o['gH'], o['gL'], lnG['H'], lnG['L'], fn['H']['eps'], fn['L']['eps'],
                             ln_lam_gHL=o['HL'])
    r = mixture.fit(o['S'], o['gH'], o['gL'], o['EH'], o['EL'], EDGES, log=True, lam_gHL=o['HL'], pi_k_H=fn['H']['pi_k'],
                    pi_k_L=fn['L']['pi_k'], fixed=('eps_H', 'eps_L'),
                    start=dict(nbar=0.01, eps_H=fn['H']['eps'], eps_L=fn['L']['eps']))
    return fn, lnB, r


def test_unit_mean_old_model_leaks():
    """the leak that Sec. 4a removes: on the same segments the one-glitch model of mixture.py (eps and law fitted by
    fit_noise) has nbar above 0 by more than 10 standard errors, where the new model with the true parameters has nbar
    within 2.5 standard errors of 0.  Its mean B is reported with how few segments carry it"""
    ev = unit_data()
    fn, lnB, r = old_model()
    B = np.exp(np.minimum(lnB, 700.0))
    dev = np.sort(np.abs(B - 1.0))[::-1]
    carry = int(np.searchsorted(np.cumsum(dev), 0.5 * dev.sum())) + 1
    new = nbar_fit(ln_B(ev, 'w', post(ev['EH'], TRUTH), post(ev['EL'], TRUTH)))
    msg = ('one-glitch model (eps_H %.4f, eps_L %.4f): mean B %.3g +- %.2g, half of sum |B - 1| in %d segments, %d '
           'segments with B > 100; nbar %.4f +- %.4f\nthis model, true parameters: nbar %.5f +- %.5f' % (
               fn['H']['eps'], fn['L']['eps'], B.mean(), B.std() / np.sqrt(B.size), carry, (lnB > np.log(100)).sum(),
               r['nbar'], r['err']['nbar'], new[0], new[1]))
    print('\n' + msg)
    assert r['nbar'] > 10 * r['err']['nbar'] and new[0] < 2.5 * new[1], msg


WRONG = {
    'q / 10': lambda E: post(E, dict(TRUTH, q=TRUTH['q'] / 10)),
    'laws swapped': lambda E: post(E, dict(TRUTH, pi=TRUTH['pi'][::-1])),
    'prior f for the posterior': lambda E: post_prior(E, TRUTH),
}


@pytest.mark.parametrize('name', list(WRONG))
def test_unit_mean_wrong_setting(name):
    """a wrong setting on the same data moves the mean of B (control variate) from 1 by more than 5 standard errors:
    q ten times too small, the laws of the two states swapped, or the prior f in place of the posterior r in kbar"""
    ev = unit_data()
    u = unit_mean(WRONG[name](ev['EH']), WRONG[name](ev['EL']))
    msg = show_mean(name, u)
    print('\n' + msg)
    assert abs(u['cv'] - 1) > 5 * u['se_cv'], msg


# ----------------------------------------------------------------------------------------------- e. recovery


def show_fit(tag, s1):
    return '; '.join('%s %s: f %s q %s' % (tag, I, np.round(s1[I]['f'], 4), np.round(s1[I]['q'], 4)) for I in 'HL')


def recover(mode, wrong=None):
    """stage 1 (fit_windows on the data of mode, mergers included), then nbar from ln(1 + nbar (B - 1))"""
    ev, s1 = recovery_data()[mode], stage_one(mode)
    p = wrong or post
    return nbar_fit(ln_B(ev, 's', p(ev['EH'], params(s1['H'])), p(ev['EL'], params(s1['L'])))), s1


def test_recovery_null():
    """merger-free segments, glitch parameters fitted on them: nbar within 2.5 standard errors of 0"""
    (nb, err), s1 = recover('free')
    msg = 'no mergers: nbar %.5f +- %.5f; %s' % (nb, err, show_fit('stage 1', s1))
    print('\n' + msg)
    assert nb < 2.5 * err, msg


def test_recovery_model():
    """e1, data exactly as the model states (no merger added to a detector with a glitch in its zone): nbar within
    2.5 standard errors of NBAR, the glitch parameters fitted on the same data (mergers taken for noise)"""
    (nb, err), s1 = recover('model')
    fl = recovery_data()['flags']
    msg = ('model data, %d mergers (%d with a glitch in the zone in H, %d in L): nbar %.4f +- %.4f (%+.2f sigma, '
           'truth %.2f)\n%s\ntruth: f %s q %s' % (fl['m'].sum(), fl['busyH'].sum(), fl['busyL'].sum(), nb, err,
                                                   (nb - NBAR) / err, NBAR, show_fit('stage 1', s1), TRUTH['f'], TRUTH['q']))
    print('\n' + msg)
    assert abs(nb - NBAR) < 2.5 * err, msg


def test_recovery_physical():
    """e2, the merger always added (glitch and merger superposed in the same zone, which the model does not
    describe): the bias is printed against the truth and against e1 on the same noise and glitches.  Measured over 8
    seeds of 40000 segments: +0.60 +- 0.24 sigma against the truth, and e2 - e1 = +0.0019 +- 0.0001 (+3.9 percent of
    NBAR, sd 0.0002 per seed).  Asserted loosely: 0 < e2 - e1 < 0.1 NBAR and |e2 - NBAR| < 4 sigma"""
    (nb2, err2), s2 = recover('physical')
    (nb1, err1), _ = recover('model')
    msg = ('physical data: nbar %.4f +- %.4f (%+.2f sigma, truth %.2f); model data %.4f; shift %+.5f = %+.1f%% of '
           'NBAR\n%s' % (nb2, err2, (nb2 - NBAR) / err2, NBAR, nb1, nb2 - nb1, 100 * (nb2 - nb1) / NBAR,
                         show_fit('stage 1', s2)))
    print('\n' + msg)
    assert 0.0 < nb2 - nb1 < 0.1 * NBAR and abs(nb2 - NBAR) < 4 * err2, msg


def test_recovery_wrong_weights():
    """e1 with a wrong formula, kbar = sum_s f_s kappa^s (the prior of the state for its posterior): nbar off the
    truth by more than 5 standard errors (measured +9.4 sigma over 8 seeds)"""
    (nb, err), _ = recover('model', wrong=post_prior)
    msg = 'prior f for the posterior: nbar %.4f +- %.4f (%+.1f sigma)' % (nb, err, (nb - NBAR) / err)
    print('\n' + msg)
    assert abs(nb - NBAR) > 5 * err, msg


def stall_data():
    """ln E_w[k] (600, 8, 4): quiet windows ln E ~ N(-0.8, 0.4) (a ramp over the bins); in 30 percent of the segments
    a window holds a glitch with probability 0.3, in the others 0.03, adding exp(N(2, 0.7)) in one bin"""
    rng = np.random.default_rng(3)
    n, nw, nk = 600, 8, 4
    q = np.where(rng.random(n) < 0.3, 0.3, 0.03)
    has = rng.random((n, nw)) < q[:, None]
    base = rng.normal(-0.8, 0.4, (n, nw, 1)) + np.linspace(-0.3, 0.3, nk)[None, None]
    boost = np.zeros((n, nw, nk))
    np.put_along_axis(boost, rng.integers(nk, size=(n, nw, 1)), rng.normal(2.0, 0.7, (n, nw, 1)), axis=2)
    return np.where(has[..., None], np.logaddexp(base, boost), base)


def stall_start(E, r, eps):
    """the maximum r of a fit moved off by a relative eps in q and in the weights of the states: start dict"""
    f = r['f'] * (1.0 + eps * np.array([1.0, -1.0]))
    return dict(f=f / f.sum(), q=r['q'] * (1.0 + eps), a=r['a'])


def test_fit_windows_stall_converged(monkeypatch):
    """fit_windows counts a fit as converged when neither a Newton step nor an EM step can raise ln L (here: the model
    evaluation is capped at its value at the start, so no step does) and the gain the Newton step predicts is below
    mixture.STALL_TOL.  Start a little off the maximum of a small data set, where the gain to the maximum (the real fit
    from there, the Newton prediction to first order) is 1e-7 to 1e-6 nats, between tol = 1e-9 and STALL_TOL = 1e-5: the
    capped fit stops at once with converged True, no update, ln L unchanged; with STALL_TOL set to 1e-12 the same fit is
    not converged, and so is the capped fit from a start 1e-2 nats or more from the maximum (gain above STALL_TOL); the
    uncapped fit from the near start converges by the ordinary rule, to the maximum"""
    E = stall_data()
    r = mixture.fit_windows(E, start=dict(f=[0.7, 0.3], q=[0.02, 0.2]))
    assert r['converged'] and mixture.STALL_TOL == 1e-5
    gain = lambda eps: r['lnl'] - mixture.fit_windows(E, start=stall_start(E, r, eps), max_iter=0)['lnl']
    near = next(eps for eps in np.geomspace(1e-5, 0.3, 80) if gain(eps) > 1e-7)
    far = next(eps for eps in np.geomspace(near, 0.3, 80) if gain(eps) > 1e-2)
    g_near, g_far = gain(near), gain(far)
    assert 1e-7 < g_near < 1e-6 and g_far > 1e-2, (near, g_near, far, g_far)
    real, cap = mixture._Windows.__call__, {}

    def capped(self, x, K, hess=False):
        out = real(self, x, K, hess)
        out['lnl'] = min(out['lnl'], cap.setdefault('lnl', out['lnl']))
        return out
    stuck = {}
    for tag, eps in (('near', near), ('far', far)):
        cap.clear()
        monkeypatch.setattr(mixture._Windows, '__call__', capped)
        stuck[tag] = mixture.fit_windows(E, start=stall_start(E, r, eps))
        monkeypatch.setattr(mixture._Windows, '__call__', real)
    start_lnl = r['lnl'] - g_near
    print('\nnear start: gain to the maximum %.3g nats, capped fit converged %s after %d updates; far start: gain %.3g, '
          'converged %s' % (g_near, stuck['near']['converged'], stuck['near']['n_iter'], g_far,
                            stuck['far']['converged']))
    assert stuck['near']['converged'] is True and stuck['near']['n_iter'] == 0
    assert abs(stuck['near']['lnl'] - start_lnl) < 1e-9 and len(stuck['near']['trace']) == 1
    assert stuck['far']['converged'] is False and stuck['far']['n_iter'] == 0
    cap.clear()
    monkeypatch.setattr(mixture._Windows, '__call__', capped)
    monkeypatch.setattr(mixture, 'STALL_TOL', 1e-12)
    assert mixture.fit_windows(E, start=stall_start(E, r, near))['converged'] is False
    monkeypatch.setattr(mixture, 'STALL_TOL', 1e-5)
    monkeypatch.setattr(mixture._Windows, '__call__', real)
    free = mixture.fit_windows(E, start=stall_start(E, r, near))
    assert free['converged'] and free['n_iter'] > 0 and abs(free['lnl'] - r['lnl']) < 1e-6


def boundary_data(glitches=True, seed=3):
    """ln E_w[k] (600, 8, 4) as stall_data, with the glitches (probability 0.3 per window in 30 percent of the
    segments, none in the others) in the first bin only; glitches False: none at all"""
    rng = np.random.default_rng(seed)
    n, nw, nk = 600, 8, 4
    q = np.where(rng.random(n) < 0.3, 0.3, 0.0)
    has = (rng.random((n, nw)) < q[:, None]) & glitches
    base = rng.normal(-0.8, 0.4, (n, nw, 1)) + np.linspace(-0.3, 0.3, nk)[None, None]
    boost = np.full((n, nw, nk), -np.inf)
    boost[:, :, 0] = rng.normal(2.0, 0.7, (n, nw))
    return np.where(has[..., None], np.logaddexp(base, boost), base)


def test_fit_windows_at_the_boundary(monkeypatch):
    """a maximum on the boundary is a converged fit (the phase-randomized week of real data, which has almost no
    glitch, ended 'not converged' in two ways).  (1) No glitch at all: every q goes to zero, ln L = 0 no longer depends
    on anything left free (the fractions of two equal states), and the fit has converged; before, the Newton step in
    that flat space was 0 / 0.  (2) Weights that the EM update is taking to zero: at the maximum of data whose
    glitches are all in one bin the other weights and the q of the quiet state are exactly zero.  From that maximum
    with 1e-14 in their place, and with the model evaluation capped at its starting value and STALL_TOL = 0 (so that
    no step raises ln L and the stall rule alone never accepts), the fit puts them at zero, their limit, and
    converges to the same maximum; with 3e-9 in their place (small entries that together could still give more than
    tol) and with 1e-6 (not small) it does not, and neither does the fit that leaves the tiny weights where they are
    (the rule taken out)"""
    E0 = boundary_data(glitches=False)
    r0 = mixture.fit_windows(E0)
    assert r0['converged'] and np.all(r0['q'] == 0) and abs(r0['lnl']) < 1e-9, (r0['converged'], r0['q'], r0['lnl'])
    E = boundary_data()
    r = mixture.fit_windows(E)
    quiet, noisy = np.argsort(r['q'])
    assert r['converged'] and r['q'][quiet] == 0 and np.all(r['a'][noisy, 1:] == 0) and r['lnl'] > 50

    def start(tiny):
        a = np.where(r['a'] > 0, r['a'], tiny)
        return dict(f=r['f'], q=np.where(r['q'] > 0, r['q'], tiny), a=a / a.sum(axis=1, keepdims=True))
    real, cap = mixture._Windows.__call__, {}

    def capped(self, x, K, hess=False):
        out = real(self, x, K, hess)
        out['lnl'] = min(out['lnl'], cap.setdefault('lnl', out['lnl']))
        return out
    monkeypatch.setattr(mixture, 'STALL_TOL', 0.0)
    monkeypatch.setattr(mixture._Windows, '__call__', capped)
    res = {}
    for tiny in (1e-14, 3e-9, 1e-6):
        cap.clear()
        res[tiny] = mixture.fit_windows(E, start=start(tiny))
    monkeypatch.setattr(mixture._Windows, '__call__', real)
    s = res[1e-14]
    print('\nfrom 1e-14: converged %s, q %s, weights of the noisy state %s; from 1e-6: converged %s'
          % (s['converged'], s['q'], s['a'][np.argmax(s['q'])], res[1e-6]['converged']))
    assert s['converged'] is True and s['q'].min() == 0 and np.all(s['a'][np.argmax(s['q']), 1:] == 0)
    assert abs(s['q'].max() - r['q'].max()) < 1e-9 and np.abs(np.sort(s['f']) - np.sort(r['f'])).max() < 1e-9
    assert res[3e-9]['converged'] is False and res[1e-6]['converged'] is False
