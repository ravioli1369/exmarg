"""planted-truth tests of the glitch-robust mixture likelihood (docs/glitch_likelihood.md, Sec. 4, controls 4 and 5).

The toy: one template, NT arrival times, white noise z_I(t) complex Gaussian with E|z|^2 = 1 (u = 1, w = |z|^2, no
delay, window W = 0).  A merger adds sqrt(x) e^{i phi} at one time to both detectors, x from the discrete (X_S, P_S); a
glitch adds sqrt(rho^2 / 2) e^{i phi} at one time to one detector, rho uniform in an amplitude bin drawn with the
weights of the power law (the law of the model).  Every likelihood ratio is then exact, per time t:

    S(t)     = sum_j p_j exp(-2 x_j) I0(2 sqrt(x_j) |z_H + z_L|)       Lambda_S   = mean_t S
    S^I(t)   = sum_j p_j exp(-x_j) I0(2 sqrt(x_j) |z_I|)               Lambda_S^I = mean_t S^I
    l_k(t)   = int d rho b_k(rho) exp(-rho^2/2) I0(sqrt(2) rho |z_I|)   E_I[k]     = mean_t l_k
    p_I(t)   = l_I(t) / (NT G_I),  l_I = sum_k pi_k l_k,  q_I = 1 - p_I
    lambda_H = mean_t [S q_H + S^L p_H]                                (H <-> L for lambda_L)
    lambda_HL = mean_t [S q_H q_L + S^L p_H q_L + S^H q_H p_L + p_H p_L]
    Y_H[k]   = NT^-2 sum_t l_k^H(t) [sum_{t' != t} S(t') + S^L(t)]     G_H lambda_H = sum_k pi_k Y_H[k]

up to a merger and a glitch in the same time sample (1/NT of the segments that hold both).  l_k is tabulated in
v = |z| by Gauss-Legendre in rho and interpolated linearly, 1.4e-7 of its unit mean.  Nothing here uses mixture.py
to build the data or the evidences; mixture.py only combines them.  The joint fit (power law) and the two-stage
route (fit_noise per detector, then nbar with the binned law fixed) are both tested.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import i0e, logsumexp
from scipy.stats import chi2

from exmarg.search import mixture

NT = 256
X_S, P_S = np.array([2.0, 4.0, 8.0, 16.0]), np.array([0.4, 0.3, 0.2, 0.1])
EDGES = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0])
NSEG, NBAR, EPS = 40000, 0.05, dict(H=0.10, L=0.03)
LAWS = dict(loud=(3.0, 6.0), weak=(6.0, 2.0))            # (gamma, rho_min): mostly rho >= 6; steep from rho = 2
KAPPA = (1 - EPS['H']) * (1 - EPS['L'])
EPS_TWO = 0.2                                              # a two-population law: steep from rho = 2, flat in 10-20
SEED = 20261114
DV, VMAX = 1e-3, 32.0


def law(edges, gamma, rho_min):
    """bin probabilities of dN/drho ~ rho^-gamma above rho_min, by quadrature"""
    p = np.array([quad(lambda r: r ** -gamma, max(a, rho_min), b)[0] if b > rho_min else 0.0
                  for a, b in zip(edges[:-1], edges[1:])])
    return p / p.sum()


def _one_minus_p(u):
    """1 - u / sum(u) and u / sum(u) over the last axis, the largest term without cancellation"""
    tot = u.sum(axis=-1, keepdims=True)
    star = u == u.max(axis=-1, keepdims=True)
    rest = np.where(star, np.sum(np.where(star, 0.0, u), axis=-1, keepdims=True), tot - u)
    return rest / tot, u / tot


class Toy:
    def __init__(self, edges=EDGES, x_s=X_S, p_s=P_S, nt=NT):
        self.edges, self.x, self.p, self.nt = edges, x_s, p_s, nt
        self.v = np.arange(0.0, VMAX + 2 * DV, DV)
        g, w = leggauss(16)
        self.T = np.empty((self.v.size, edges.size - 1))            # ln l_k(v)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            s = np.linspace(a, b, int(np.ceil(b - a)) + 1)
            rho = (0.5 * np.diff(s)[:, None] * g[None, :] + 0.5 * (s[1:] + s[:-1])[:, None]).ravel()
            wt = (0.5 * np.diff(s)[:, None] * w[None, :]).ravel() / (b - a)
            arg = np.sqrt(2.0) * rho[None, :] * self.v[:, None]
            self.T[:, k] = logsumexp(-0.5 * rho ** 2 + arg + np.log(i0e(arg)), axis=1, b=wt[None, :])

    def ln_lk(self, a):
        if np.any(a > VMAX):
            raise ValueError('|z| above the table')
        u = a / DV
        j = u.astype(np.int64)
        f = (u - j)[..., None]
        return self.T[j] * (1.0 - f) + self.T[j + 1] * f

    def merger(self, a, c):
        """sum_j p_j exp(-c x_j) I0(2 sqrt(x_j) a), per time"""
        out = np.zeros(a.shape)
        for x, p in zip(self.x, self.p):
            arg = 2.0 * np.sqrt(x) * a
            out += p * np.exp(arg - c * x) * i0e(arg)
        return out

    def draws(self, n, seed):
        """the random numbers of n segments, so that data sets with other rates and laws share them"""
        g = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
        noise = {I: (g[0].standard_normal((n, self.nt)) + 1j * g[0].standard_normal((n, self.nt))) / np.sqrt(2.0)
                 for I in 'HL'}
        m = dict(u=g[1].random(n), t=g[1].integers(self.nt, size=n), ux=g[1].random(n),
                 ph=np.exp(2j * np.pi * g[1].random(n)))
        gl = {I: dict(u=r.random(n), t=r.integers(self.nt, size=n), uk=r.random(n), ur=r.random(n),
                      ph=np.exp(2j * np.pi * r.random(n))) for I, r in zip('HL', g[2:])}
        return dict(noise=noise, m=m, g=gl)

    def data(self, d, nbar, eps, pis):
        """z_H, z_L and the truth: a merger with probability nbar, a glitch in I with probability eps[I]"""
        z = {I: d['noise'][I].copy() for I in 'HL'}
        m = d['m']
        flags = dict(m=m['u'] < nbar)
        i = np.nonzero(flags['m'])[0]
        x = self.x[np.searchsorted(np.cumsum(self.p) / self.p.sum(), m['ux'][i], side='right')]
        for I in 'HL':
            z[I][i, m['t'][i]] += np.sqrt(x) * m['ph'][i]
        for I in 'HL':
            gI = d['g'][I]
            flags['g' + I] = gI['u'] < eps[I]
            i = np.nonzero(flags['g' + I])[0]
            k = np.searchsorted(np.cumsum(pis[I]) / pis[I].sum(), gI['uk'][i], side='right')
            rho = self.edges[k] + gI['ur'][i] * (self.edges[k + 1] - self.edges[k])
            z[I][i, gI['t'][i]] += np.sqrt(0.5 * rho ** 2) * gI['ph'][i]
        return z, flags

    def _evidences(self, zH, zL, laws):
        n, lnt = zH.shape[0], np.log(self.nt)
        a = {'H': np.abs(zH), 'L': np.abs(zL)}
        S = self.merger(np.abs(zH + zL), 2.0)
        SI = {I: self.merger(a[I], 1.0) for I in 'HL'}
        assert np.isfinite(S).all() and np.isfinite(SI['H']).all() and np.isfinite(SI['L']).all()
        out = dict(S=np.log(S.mean(axis=1)), SH=np.log(SI['H'].mean(axis=1)), SL=np.log(SI['L'].mean(axis=1)))
        rest = _one_minus_p(S)[0] * S.sum(axis=1, keepdims=True)              # sum_{t' != t} S(t')
        U, sh = {}, {}
        for I, J in (('H', 'L'), ('L', 'H')):
            lk = self.ln_lk(a[I])                                              # (n, t, k)
            sh[I] = lk.max(axis=1)
            U[I] = np.exp(lk - sh[I][:, None, :])
            out['E' + I] = sh[I] + np.log(U[I].mean(axis=1))
            out['Y' + I] = sh[I] + np.log(np.einsum('ntk,nt->nk', U[I], rest + SI[J])) - 2 * lnt
        for key in ('GH', 'GL', 'gH', 'gL', 'HL'):
            out[key] = np.empty((len(laws), n))
        for j, pis in enumerate(laws):
            q, p = {}, {}
            for I in 'HL':
                with np.errstate(divide='ignore'):
                    s = sh[I] + np.log(pis[I])[None, :]
                smax = s.max(axis=1, keepdims=True)
                l = np.einsum('ntk,nk->nt', U[I], np.exp(s - smax))           # l_I(t) e^-smax
                q[I], p[I] = _one_minus_p(l)
                out['G' + I][j] = smax[:, 0] + np.log(l.mean(axis=1))
            out['gH'][j] = np.log(np.mean(S * q['H'] + SI['L'] * p['H'], axis=1))
            out['gL'][j] = np.log(np.mean(S * q['L'] + SI['H'] * p['L'], axis=1))
            out['HL'][j] = np.log(np.mean(S * q['H'] * q['L'] + SI['L'] * p['H'] * q['L'] + SI['H'] * q['H'] * p['L']
                                          + p['H'] * p['L'], axis=1))
        return out

    def evidences(self, z, laws, chunk=500, workers=8):
        """ln of Lambda_S (S), Lambda_S^I (SI), E_I (EI), Y_I (YI) per segment, and of G_I (GI), lambda_I (gI),
        lambda_HL (HL) for each law of laws (a list of dicts I: pi_k), index 0"""
        n = z['H'].shape[0]
        parts = [slice(c, min(c + chunk, n)) for c in range(0, n, chunk)]
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda s: self._evidences(z['H'][s], z['L'][s], laws), parts))
        return {k: np.concatenate([r[k] for r in res], axis=1 if k in ('GH', 'GL', 'gH', 'gL', 'HL') else 0)
                for k in res[0]}


@lru_cache(maxsize=None)
def toy():
    return Toy()


@lru_cache(maxsize=None)
def draws():
    return toy().draws(NSEG, SEED)


@lru_cache(maxsize=None)
def pis(name):
    if name == 'two':
        flat = ((EDGES[:-1] >= 10) & (EDGES[1:] <= 20)) / 3.0
        return 0.5 * law(EDGES, *LAWS['weak']) + 0.5 * flat
    return law(EDGES, *LAWS[name])


@lru_cache(maxsize=None)
def plant(name):
    """'gauss': mergers in Gaussian noise (lambda for the weak and the loud law); 'loud', 'weak': mergers and glitches
    at EPS with that law; 'glitch': the glitches of 'loud' without the mergers; 'two': glitches in H at EPS_TWO from
    the two-population law, Gaussian L, no mergers"""
    if name == 'two':
        z, fl = toy().data(draws(), 0.0, dict(H=EPS_TWO, L=0.0), dict(H=pis('two'), L=pis('two')))
        return toy().evidences(z, [dict(H=pis('two'), L=pis('two'))]), fl
    if name == 'gauss':
        z, fl = toy().data(draws(), NBAR, dict(H=0.0, L=0.0), dict(H=pis('weak'), L=pis('weak')))
        laws = [dict(H=pis(w), L=pis(w)) for w in ('weak', 'loud')]
    else:
        w = 'loud' if name == 'glitch' else name
        z, fl = toy().data(draws(), 0.0 if name == 'glitch' else NBAR, EPS, dict(H=pis(w), L=pis(w)))
        laws = [dict(H=pis(w), L=pis(w))]
    return toy().evidences(z, laws), fl


def fit(ev, j=0, per_bin=False, double='exact', **kw):
    """mixture.fit on toy evidences; double: 'exact' (lambda_HL) or 'one' (lambda_HL = 1)"""
    gH, gL = (ev['YH'], ev['YL']) if per_bin else (ev['gH'][j], ev['gL'][j])
    return mixture.fit(ev['S'], gH, gL, ev['EH'], ev['EL'], EDGES, log=True,
                       lam_gHL=ev['HL'][j] if double == 'exact' else None, **kw)


@lru_cache(maxsize=None)
def full_fit(name, per_bin=False, double='exact'):
    """the fit of nbar, eps_H, eps_L, gamma_H, gamma_L on a glitch plant, started at the truth"""
    return fit(plant(name)[0], per_bin=per_bin, double=double, start=start(name))


def start(name, **kw):
    g, r0 = LAWS[name]
    return dict(dict(nbar=NBAR, eps_H=EPS['H'], eps_L=EPS['L'], gamma_H=g, gamma_L=g, rho_min_H=r0, rho_min_L=r0), **kw)


def truth(name):
    g = LAWS[name][0]
    return dict(nbar=NBAR, eps_H=EPS['H'], eps_L=EPS['L'], gamma_H=g, gamma_L=g)


ALL = ('nbar', 'eps_H', 'eps_L', 'gamma_H', 'gamma_L', 'rho_min_H', 'rho_min_L')
ONLY_NBAR = tuple(p for p in ALL if p != 'nbar')


def show(tag, r, tr):
    return '%s: ' % tag + ', '.join('%s %.4g +- %.2g (truth %.4g, %+.1f sigma)'
                                    % (p, r[p], r['err'][p], tr[p], (r[p] - tr[p]) / r['err'][p]) for p in r['free'])


def within(r, tr, k, names):
    return all(abs(r[p] - tr[p]) <= k * r['err'][p] for p in names)


# ----------------------------------------------------------------------------------------------- the functions


def test_amplitude_weights_match_quadrature():
    """pi_k against quadrature of rho^-gamma (gamma across 1, negative, rho_min inside a bin): agree to 1e-12"""
    for g, r0 in ((3.0, 6.0), (6.0, 2.0), (1.0, 3.3), (1.0 + 1e-10, 2.2), (0.0, 2.0), (-2.0, 4.5), (2.5, 29.0)):
        pk = mixture.amplitude_weights(EDGES, g, r0)
        assert np.max(np.abs(pk - law(EDGES, g, r0))) < 1e-12, (g, r0, pk)
        assert abs(pk.sum() - 1) < 1e-13 and np.all(pk[EDGES[1:] <= r0] == 0)
    with pytest.raises(ValueError):
        mixture.amplitude_weights(EDGES, 3.0, 31.0)


def law_cut(edges, gamma, rho_min, rho_max):
    """bin probabilities of dN/drho ~ rho^-gamma on [rho_min, rho_max], by quadrature"""
    lo, hi = np.maximum(edges[:-1], rho_min), np.minimum(edges[1:], rho_max)
    p = np.array([quad(lambda r: r ** -gamma, a, b)[0] if b > a else 0.0 for a, b in zip(lo, hi)])
    return p / p.sum()


def test_amplitude_weights_rho_max():
    """pi_k with an upper end rho_max against quadrature of rho^-gamma on [rho_min, rho_max] (rho_max inside a bin, on
    an edge, inside the bin of rho_min, at and beyond the last edge; gamma across 1 and negative): agree to 1e-12, no
    weight outside [rho_min, rho_max], the ln_ form is the log, and rho_max at or beyond the last edge is the law
    without it; rho_max not above rho_min, or not above the first edge, raises"""
    for g, r0, r1 in ((3.0, 6.0, 11.0), (3.0, 6.0, 12.0), (6.0, 2.0, 2.3), (1.0, 3.3, 17.5), (1.0 + 1e-10, 2.2, 7.7),
                      (0.0, 2.0, 9.0), (-2.0, 4.5, 25.0), (2.5, 6.0, 30.0), (2.5, 6.0, 1e3)):
        pk = mixture.amplitude_weights(EDGES, g, r0, r1)
        assert np.max(np.abs(pk - law_cut(EDGES, g, r0, r1))) < 1e-12, (g, r0, r1, pk)
        assert abs(pk.sum() - 1) < 1e-13 and np.all(pk[EDGES[:-1] >= r1] == 0) and np.all(pk[EDGES[1:] <= r0] == 0)
        assert np.allclose(np.exp(mixture.ln_amplitude_weights(EDGES, g, r0, r1)), pk, rtol=1e-13, atol=0)
        if r1 >= EDGES[-1]:
            assert np.array_equal(pk, mixture.amplitude_weights(EDGES, g, r0))
    for r0, r1 in ((6.0, 6.0), (6.0, 5.0), (1.0, 2.0)):
        with pytest.raises(ValueError):
            mixture.amplitude_weights(EDGES, 3.0, r0, r1)


def law_soft(edges, gamma, rho_min, veto, rho_max=np.inf):
    """bin probabilities of rho^-gamma Phi((cut - rho - offset) / width) on [rho_min, rho_max], by quadrature"""
    from scipy.special import ndtr
    cut, off, width = veto
    lo, hi = np.maximum(edges[:-1], rho_min), np.minimum(edges[1:], rho_max)
    p = np.array([quad(lambda r: r ** -gamma * ndtr((cut - r - off) / width), a, b, epsabs=0, epsrel=1e-13)[0]
                  if b > a else 0.0 for a, b in zip(lo, hi)])
    return p / p.sum()


def test_amplitude_weights_soft_veto():
    """pi_k of the glitches that pass a soft veto, rho^-gamma Phi((cut - rho - offset) / width), against quadrature
    (cut inside a bin and on an edge, gamma across 1 and negative, narrow and wide widths, with rho_max): agree to
    1e-10; a cut far above the last edge is the law without a veto, and a narrow width tends to the hard cut at
    cut - offset"""
    for g, r0, veto, r1 in ((3.0, 6.0, (12.0, 0.1, 1.0), np.inf), (3.0, 6.0, (11.0, 0.0, 0.3), np.inf),
                            (1.0, 3.3, (12.0, 0.2, 1.0), np.inf), (-2.0, 4.5, (20.0, -0.5, 2.0), np.inf),
                            (6.0, 2.0, (8.0, 0.1, 1.0), 25.0), (2.5, 6.0, (12.0, 0.1, 1.0), 13.0)):
        pk = mixture.amplitude_weights(EDGES, g, r0, None if r1 == np.inf else r1, veto)
        ref = law_soft(EDGES, g, r0, veto, r1)
        assert np.max(np.abs(pk - ref)) < 1e-10, (g, r0, veto, r1, np.max(np.abs(pk - ref)))
        assert abs(pk.sum() - 1) < 1e-13 and np.all(pk[EDGES[1:] <= r0] == 0)
    far = mixture.amplitude_weights(EDGES, 3.0, 6.0, veto=(1e4, 0.0, 1.0))
    assert np.allclose(far, mixture.amplitude_weights(EDGES, 3.0, 6.0), rtol=1e-12, atol=0)
    narrow = mixture.amplitude_weights(EDGES, 3.0, 6.0, veto=(12.1, 0.1, 1e-3))
    assert np.max(np.abs(narrow - mixture.amplitude_weights(EDGES, 3.0, 6.0, 12.0))) < 2e-3


def test_log_and_linear_forms_agree():
    """ln_ variants equal the logs of the linear ones to 1e-12, a trailing grid broadcasts, eps = 0 gives Lambda_S,
    and time_curve equals the integral over the prior done segment by segment"""
    rng = np.random.default_rng(1)
    n, nr = 50, EDGES.size - 1
    E = np.exp(rng.normal(0, 2, (n, 3, nr)))
    pk, pm = mixture.amplitude_weights(EDGES, 3.0, 4.0), np.array([0.2, 0.5, 0.3])
    G = mixture.glitch_evidence(E, pk, pm)
    assert np.allclose(G, np.einsum('imk,m,k->i', E, pm, pk), rtol=1e-13)
    assert np.allclose(np.log(G), mixture.ln_glitch_evidence(np.log(E), pk, pm), rtol=0, atol=1e-12)
    lam, lH, lL = (np.exp(rng.normal(0, 3, (n, 4, 5))) for _ in range(3))
    GH, GL = np.exp(rng.normal(0, 2, n)), np.exp(rng.normal(0, 2, n))
    for sg, both in ((True, None), (True, lam * 0.7), (False, None)):
        B = mixture.mixture(lam, lH, lL, GH, GL, 0.2, 0.07, sg=sg, lam_gHL=both)
        lnB = mixture.ln_mixture(np.log(lam), np.log(lH), np.log(lL), np.log(GH), np.log(GL), 0.2, 0.07, sg=sg,
                                 ln_lam_gHL=None if both is None else np.log(both))
        assert B.shape == lam.shape and np.allclose(np.log(B), lnB, rtol=0, atol=1e-12)
        i, a, b = 7, 2, 3
        Bi = mixture.mixture(lam[i:i + 1, a, b], lH[i:i + 1, a, b], lL[i:i + 1, a, b], GH[i:i + 1], GL[i:i + 1], 0.2,
                             0.07, sg=sg, lam_gHL=None if both is None else both[i:i + 1, a, b])
        assert np.isclose(Bi[0], B[i, a, b], rtol=1e-14)
    assert np.allclose(mixture.mixture(lam, lH, lL, GH, GL, 0.0, 0.0), lam, rtol=1e-15)
    c = mixture.noise_ratio(GH, 0.3)
    assert np.allclose(np.log(c), mixture.ln_noise_ratio(np.log(GH), 0.3), rtol=0, atol=1e-13)
    B = lam[:, 0, 0]
    cH, cL = mixture.noise_ratio(GH, 0.2), mixture.noise_ratio(GL, 0.07)
    tot = np.sum(np.log(cH) + np.log(cL) + np.log(1 + 0.03 * (B - 1)))
    assert np.isclose(mixture.log_likelihood(0.03, B, cH, cL), tot, rtol=1e-13)
    assert np.isclose(mixture.log_likelihood(0.03, np.log(B), np.log(cH), np.log(cL), log=True), tot, rtol=1e-13)
    grid = np.linspace(0.0, 0.5, 201)
    prior = lambda x: np.exp(-x / 0.1)
    curve = mixture.time_curve(B, grid, prior, chunk=7)
    assert np.allclose(curve, mixture.time_curve(np.log(B), grid, prior, log=True), rtol=0, atol=1e-12)
    for m in (1, 13, n):
        f = lambda x: prior(x) * np.prod(1 + x * (B[:m] - 1))
        ref = quad(f, 0.0, 0.5, points=grid[::20], limit=400)[0] / quad(prior, 0.0, 0.5)[0]
        trap = np.trapezoid(prior(grid) * np.prod(1 + grid[:, None] * (B[None, :m] - 1), axis=1), grid)
        trap /= np.trapezoid(prior(grid), grid)
        assert np.isclose(curve[m - 1], np.log(trap), rtol=0, atol=1e-10), (m, curve[m - 1], np.log(trap))
        assert np.isclose(curve[m - 1], np.log(ref), rtol=0, atol=2e-3), (m, curve[m - 1], np.log(ref))


# ----------------------------------------------------------------------------------------------- the toy


def test_toy_table_unit_mean():
    """control 1 for the toy: every l_k of the table, as interpolated, averages to 1 over w ~ Exp(1), to 1e-6"""
    t = toy()
    v = np.arange(0.0, VMAX, 3e-4)
    lk = t.ln_lk(v)
    means = np.trapezoid(2 * v[:, None] * np.exp(lk - v[:, None] ** 2), v, axis=0)
    assert np.max(np.abs(means - 1)) < 1e-6, means - 1


def test_unit_mean():
    """over noise-only toy segments every evidence (Lambda_S, Lambda_S^I, E_I[k], G_I lambda_I) has mean 1, and over
    segments with glitches at eps (no merger) B has mean 1, each within 3 standard errors; Smith & Thrane's B and the
    Gaussian B = Lambda_S (glitches present) are not, by more than 10 standard errors.  Weak amplitudes and NT = 16:
    the mean of a loud evidence is carried by noise of probability ~exp(-rho^2/2), which no sample contains."""
    edges = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    t = Toy(edges, np.array([0.5, 1.0, 2.0]), np.array([0.5, 0.3, 0.2]), 16)
    pk = law(edges, 2.0, 0.5)
    d = t.draws(200000, SEED + 1)
    eH, eL = 0.3, 0.2
    lines, bad = [], []

    def check(x, tag, expect_one=True):
        m, se = x.mean(), x.std() / np.sqrt(x.size)
        lines.append('%s: mean %.4f +- %.4f (%+.1f se)' % (tag, m, se, (m - 1) / se))
        if (abs(m - 1) > 3 * se) if expect_one else (abs(m - 1) < 10 * se):
            bad.append(lines[-1])

    z, _ = t.data(d, 0.0, dict(H=0.0, L=0.0), dict(H=pk, L=pk))
    ev = t.evidences(z, [dict(H=pk, L=pk)])
    for k in ('S', 'SH', 'SL'):
        check(np.exp(ev[k]), 'Lambda ' + k)
    for I in 'HL':
        for k in range(edges.size - 1):
            check(np.exp(ev['E' + I][:, k]), 'E_%s[%d]' % (I, k))
        check(np.exp(ev['G' + I][0] + ev['g' + I][0]), 'G_%s lambda_%s' % (I, I))
    z, _ = t.data(d, 0.0, dict(H=eH, L=eL), dict(H=pk, L=pk))
    ev = t.evidences(z, [dict(H=pk, L=pk)])
    S, gH, gL = np.exp(ev['S']), np.exp(ev['gH'][0]), np.exp(ev['gL'][0])
    GH, GL = mixture.glitch_evidence(np.exp(ev['EH']), pk), mixture.glitch_evidence(np.exp(ev['EL']), pk)
    check(mixture.mixture(S, gH, gL, GH, GL, eH, eL, lam_gHL=np.exp(ev['HL'][0])), 'B (Sec. 4)')
    check(mixture.mixture(S, gH, gL, GH, GL, eH, eL), 'B, lambda_HL = 1')
    check(mixture.mixture(S, gH, gL, GH, GL, eH, eL, sg=False), 'B, Smith & Thrane', False)
    check(S, 'B = Lambda_S (Gaussian)', False)
    print('\n' + '\n'.join(lines))
    assert not bad, bad


# ----------------------------------------------------------------------------------------------- plant, Gaussian


def test_plant_gaussian_eps_free():
    """mergers at nbar in Gaussian noise, eps_H, eps_L free (weak law): nbar within 2 sigma of the truth, each eps
    within 2 sigma of 0"""
    ev, fl = plant('gauss')
    r = fit(ev, fixed=('gamma_H', 'gamma_L', 'rho_min_H', 'rho_min_L'), start=start('weak', eps_H=0.01, eps_L=0.01))
    msg = show('%d segments, %d mergers' % (NSEG, fl['m'].sum()), r, dict(nbar=NBAR, eps_H=0.0, eps_L=0.0))
    print('\n' + msg)
    assert abs(r['nbar'] - NBAR) < 2 * r['err']['nbar'] and r['eps_H'] < 2 * r['err']['eps_H'] \
        and r['eps_L'] < 2 * r['err']['eps_L'], msg


def test_plant_gaussian_eps_zero_is_plain_mixture():
    """eps = 0: the total equals sum ln(1 + nbar (Lambda_S - 1)) to 1e-12, the fit returns the maximum of that sum
    (found here by a scalar search) to 1e-6 relative and its error 1 / sqrt(sum (S - 1)^2 / (1 + nbar (S - 1))^2) to
    1e-3 relative, and nbar is within 2 sigma of the truth"""
    ev, _ = plant('gauss')
    S = np.exp(ev['S'])
    GH, GL = np.exp(ev['GH'][0]), np.exp(ev['GL'][0])
    for nb in (1e-3, 0.05, 0.3):
        B = mixture.mixture(S, np.exp(ev['gH'][0]), np.exp(ev['gL'][0]), GH, GL, 0.0, 0.0)
        tot = mixture.log_likelihood(nb, B, mixture.noise_ratio(GH, 0.0), mixture.noise_ratio(GL, 0.0))
        plain = np.sum(np.log1p(nb * (S - 1)))
        assert abs(tot - plain) <= 1e-12 * abs(plain), (nb, tot, plain)
    r = fit(ev, fixed=ONLY_NBAR, start=start('weak', eps_H=0.0, eps_L=0.0))
    ref = minimize_scalar(lambda x: -np.sum(np.log1p(x * (S - 1))), bounds=(1e-6, 0.5), method='bounded',
                          options=dict(xatol=1e-12)).x
    err = 1.0 / np.sqrt(np.sum((S - 1) ** 2 / (1 + ref * (S - 1)) ** 2))
    msg = show('eps = 0', r, dict(nbar=NBAR)) + '; scalar maximum of the plain mixture %.6g +- %.4g' % (ref, err)
    print('\n' + msg)
    assert abs(r['nbar'] - ref) < 1e-6 * ref and abs(r['err']['nbar'] / err - 1) < 1e-3 \
        and abs(r['nbar'] - NBAR) < 2 * r['err']['nbar'], msg


def _wrong_eps(name, eps):
    ev, _ = plant('gauss')
    j = ('weak', 'loud').index(name)
    r0 = fit(ev, j, fixed=ONLY_NBAR, start=start(name, eps_H=0.0, eps_L=0.0))
    r = fit(ev, j, fixed=ONLY_NBAR, start=start(name, eps_H=eps, eps_L=eps))
    return r, r0


def test_plant_gaussian_wrong_eps():
    """eps fixed at a wrong 0.2 with the weak law (lambda at that law): nbar within 2 sigma of the truth.  This holds
    for this seed only because its eps = 0 estimate is high: the shift itself is test_weak_glitches_change_nothing"""
    r, r0 = _wrong_eps('weak', 0.2)
    msg = show('eps fixed 0.2', r, dict(nbar=NBAR)) + '; eps = 0 on the same data: %.4g' % r0['nbar']
    print('\n' + msg)
    assert abs(r['nbar'] - NBAR) < 2 * r['err']['nbar'], msg


@pytest.mark.xfail(strict=True, reason='measured shift of nbar against eps = 0 on the same data, weak law: -0.7% at '
                                       'eps 0.01, -2.9% at 0.05, -8.7% at 0.2, -17% at 0.5 (6 seeds)')
def test_weak_glitches_change_nothing():
    """Sec. 4: weak glitches change nothing, whatever eps is.  Paired on the same Gaussian data, nbar with eps fixed
    at 0.2 (weak law) moves from its eps = 0 value by less than a quarter of its statistical error"""
    r, r0 = _wrong_eps('weak', 0.2)
    shift = r['nbar'] - r0['nbar']
    msg = 'eps 0.2: nbar %.4g, eps 0: %.4g, shift %+.4g = %+.1f%% = %+.2f sigma' % (
        r['nbar'], r0['nbar'], shift, 100 * shift / r0['nbar'], shift / r0['err']['nbar'])
    print('\n' + msg)
    assert abs(shift) < 0.25 * r0['err']['nbar'], msg


# ----------------------------------------------------------------------------------------------- plant, glitches


def test_plant_glitches_gaussian_likelihood_overestimates():
    """control: on the loud plant the Gaussian likelihood (eps = 0) overestimates nbar by more than 5 sigma"""
    ev, fl = plant('loud')
    r = fit(ev, fixed=ONLY_NBAR, start=start('loud', eps_H=0.0, eps_L=0.0))
    msg = show('Gaussian likelihood, %d glitches in H, %d in L' % (fl['gH'].sum(), fl['gL'].sum()), r, dict(nbar=NBAR))
    print('\n' + msg)
    assert (r['nbar'] - NBAR) > 5 * r['err']['nbar'], msg


@pytest.mark.parametrize('name', ['loud', 'weak'])
def test_plant_glitches_nbar(name):
    """mergers and glitches (with shared segments): the fit of nbar, eps_H, eps_L, gamma_H, gamma_L (lambda at the true
    law) returns nbar within 2.5 sigma of the truth"""
    ev, fl = plant(name)
    r = full_fit(name)
    msg = show('%s, %d mergers, %d with a glitch in H, %d in L' % (
        name, fl['m'].sum(), (fl['m'] & fl['gH']).sum(), (fl['m'] & fl['gL']).sum()), r, truth(name))
    print('\n' + msg)
    assert within(r, truth(name), 2.5, ['nbar']), msg


@pytest.mark.parametrize('name', ['loud', pytest.param('weak', marks=pytest.mark.xfail(
    strict=True, reason='lambda at a fixed law is not normalized as the law varies: score of gamma_H at the truth '
                        '-62 +- 12, zero with lambda recomputed (160000 segments); eps_H -2.9, gamma_H -4.0 sigma '
                        'over 12 seeds'))])
def test_plant_glitches_noise_model(name):
    """the same fit returns eps_H, eps_L, gamma_H, gamma_L within 2.5 sigma of the truth"""
    r = full_fit(name)
    msg = show(name, r, truth(name))
    print('\n' + msg)
    assert within(r, truth(name), 2.5, r['free']), msg


@pytest.mark.parametrize('name', ['loud', 'weak'])
def test_plant_glitches_per_bin_lambda(name):
    """lambda exact in the law (per-bin Y_I): nbar, eps_H, eps_L, gamma_H, gamma_L within 2.5 sigma of the truth"""
    r = full_fit(name, per_bin=True)
    msg = show(name + ', per-bin lambda', r, truth(name))
    print('\n' + msg)
    assert within(r, truth(name), 2.5, r['free']), msg


@pytest.mark.parametrize('name', ['loud', pytest.param('weak', marks=pytest.mark.xfail(
    strict=True, reason='the iteration converges in one step to eps_H 0.071, gamma_H 5.12 (truth 0.10, 6): a fixed '
                        'point of the fiducial-law likelihood, not the truth'))])
def test_lambda_at_fitted_law(name):
    """lambda recomputed once at the fitted law and the fit repeated: every parameter within 2.5 sigma of the truth"""
    ev, _ = plant(name)
    r = full_fit(name)
    z, _ = toy().data(draws(), NBAR, EPS, dict(H=pis(name), L=pis(name)))
    laws = [{I: mixture.amplitude_weights(EDGES, r['gamma_' + I], r['rho_min_' + I]) for I in 'HL'}]
    ev2 = dict(ev, **{k: v for k, v in toy().evidences(z, laws).items() if k in ('gH', 'gL', 'HL')})
    r2 = fit(ev2, start=start(name))
    msg = '%s\n%s' % (show(name + ', lambda at the true law', r, truth(name)),
                      show('lambda at the fitted law', r2, truth(name)))
    print('\n' + msg)
    assert within(r2, truth(name), 2.5, r2['free']), msg


def test_double_term_one():
    """lambda_HL = 1 (Z_{S+gH+gL} = G_H G_L, no merger information when both detectors have a glitch) against the exact
    lambda_HL on the weak plant, eps and law at the truth: nbar moves by less than a quarter of its statistical error
    (the term carries eps_H eps_L = 0.003 of the segments)"""
    ev, _ = plant('weak')
    r = fit(ev, fixed=ONLY_NBAR, start=start('weak'))
    one = fit(ev, double='one', fixed=ONLY_NBAR, start=start('weak'))
    d = one['nbar'] - r['nbar']
    msg = 'weak: nbar with lambda_HL = 1 %.5g, exact %.5g +- %.2g: shift %+.2f%% = %+.3f sigma' % (
        one['nbar'], r['nbar'], r['err']['nbar'], 100 * d / r['nbar'], d / r['err']['nbar'])
    print('\n' + msg)
    assert abs(one['nbar'] - r['nbar']) < 0.25 * r['err']['nbar'], msg


@pytest.mark.parametrize('name', ['loud', 'weak'])
def test_dropped_terms(name):
    """the same data with the three S+g terms dropped (Smith & Thrane), eps and law at the truth: nbar is below the
    estimate of the full mixture on the same data by more than 2.5 sigma; the ratios are reported"""
    ev, _ = plant(name)
    r = fit(ev, fixed=ONLY_NBAR, start=start(name))
    st = fit(ev, fixed=ONLY_NBAR, start=start(name), sg=False)
    msg = ('%s: Smith & Thrane nbar %.4g +- %.2g, full %.4g +- %.2g, truth %.3g; ratio to truth %.3f, to full %.3f, '
           'kappa %.3f' % (name, st['nbar'], st['err']['nbar'], r['nbar'], r['err']['nbar'], NBAR, st['nbar'] / NBAR,
                           st['nbar'] / r['nbar'], KAPPA))
    print('\n' + msg)
    assert r['nbar'] - st['nbar'] > 2.5 * r['err']['nbar'], msg


@pytest.mark.parametrize('name', ['weak'])
def test_whole_segment(name):
    """the whole-segment form (lambda_H = Lambda_S^L, lambda_L = Lambda_S^H, S+gH+gL = G_H G_L), eps and law at the
    truth, weak glitches: nbar is above the estimate of the full mixture on the same data by more than 2.5 sigma"""
    ev, _ = plant(name)
    r = fit(ev, fixed=ONLY_NBAR, start=start(name))
    w = mixture.fit(ev['S'], ev['SL'], ev['SH'], ev['EH'], ev['EL'], EDGES, log=True, fixed=ONLY_NBAR,
                    start=start(name))
    msg = ('%s: whole-segment nbar %.4g +- %.2g, full %.4g +- %.2g, truth %.3g; ratio to truth %.3f, to full %.3f, '
           '1/kappa %.3f' % (name, w['nbar'], w['err']['nbar'], r['nbar'], r['err']['nbar'], NBAR, w['nbar'] / NBAR,
                             w['nbar'] / r['nbar'], 1 / KAPPA))
    print('\n' + msg)
    assert w['nbar'] - r['nbar'] > 2.5 * r['err']['nbar'], msg


# ----------------------------------------------------------------------------------------------- time curve


def _curve(ev):
    pk = pis('loud')
    lnGH, lnGL = (mixture.ln_glitch_evidence(ev['E' + I], pk) for I in 'HL')
    lnB = mixture.ln_mixture(ev['S'], ev['gH'][0], ev['gL'][0], lnGH, lnGL, EPS['H'], EPS['L'], ln_lam_gHL=ev['HL'][0])
    return mixture.time_curve(lnB, GRID, PRIOR, log=True)


GRID, PRIOR, VILLE = np.linspace(0.0, 0.2, 401), np.ones(401), np.log(100.0)


def test_time_curve():
    """ln of the evidence ratio against the number of segments (prior uniform in nbar on [0, 0.2]): on glitches without
    mergers it ends negative and stays below ln 100 (Ville's bound at 1%); the Gaussian likelihood on the same data
    crosses ln 100 (control); with the mergers it ends above 50"""
    ev, _ = plant('glitch')
    c = _curve(ev)
    gauss = mixture.time_curve(ev['S'], GRID, PRIOR, log=True)
    cm = _curve(plant('loud')[0])
    msg = ('glitches only: final %.2f, max %.2f (bound %.2f); Gaussian likelihood: max %.1f; with mergers: final %.1f'
           % (c[-1], c.max(), VILLE, gauss.max(), cm[-1]))
    print('\n' + msg)
    assert c[-1] < 0 and c.max() < VILLE and gauss.max() > VILLE and cm[-1] > 50, msg


# ----------------------------------------------------------------------------------------------- noise model alone


def _kkt(lnE, omega):
    """the largest violation of the optimality conditions of fit_noise, from E and omega: mean_i Ebar_i[k] / D_i is 1
    where omega_k > 0 and at most 1 elsewhere"""
    X = np.exp(np.concatenate([np.zeros((lnE.shape[0], 1)), lnE], axis=1))
    g = np.mean(X / np.einsum('ik,k->i', X, omega)[:, None], axis=0)
    return max(np.max(np.abs(g[omega > 0] - 1)), np.max(g[omega == 0] - 1, initial=0.0))


def test_fit_noise_two_populations():
    """fit_noise on glitches in H from a law that is not a power law (half steep from rho = 2, half flat in 10-20,
    eps 0.2): the optimality conditions hold to 1e-6, the covariance is the inverse of minus the Hessian of ln L on
    the simplex (finite differences, 1e-3), and the true binned law lies inside the 99.9% likelihood-ratio region,
    0 <= 2 (ln L_max - ln L_true) < chi2_13(0.999) (conservative where weights sit at zero)"""
    ev, fl = plant('two')
    r = mixture.fit_noise(ev['EH'], log=True)
    X = np.exp(ev['EH'])
    on = np.nonzero(r['omega'][1:] > 0)[0]
    h = 0.01 * np.sqrt(np.diag(r['cov']))[on]                    # truncation error ~ (h / sigma)^2

    def f(dw):
        w = r['omega'][1:].copy()
        w[on] += dw
        return np.sum(np.log(1 - w.sum() + np.einsum('ik,k->i', X, w)))

    E1 = np.eye(on.size) * h
    hess = np.array([[(f(E1[a] + E1[b]) - f(E1[a] - E1[b]) - f(-E1[a] + E1[b]) + f(-E1[a] - E1[b]))
                      / (4 * h[a] * h[b]) for b in range(on.size)] for a in range(on.size)])
    cov_err = np.max(np.abs(np.linalg.inv(-hess) / r['cov'][np.ix_(on, on)] - 1))
    w = np.r_[1 - EPS_TWO, EPS_TWO * pis('two')]
    with np.errstate(divide='ignore'):
        terms = np.concatenate([np.full((NSEG, 1), np.log(w[0])), ev['EH'] + np.log(w[1:])], axis=1)
    lr = 2 * (r['lnl'] - logsumexp(terms, axis=1).sum())
    err = np.sqrt(np.diag(r['cov']))
    low = np.nonzero(r['omega'][1:5] > 0)[0]
    corr = r['cov'][np.ix_(low, low)] / np.outer(err[low], err[low])
    kkt, bound = _kkt(ev['EH'], r['omega']), chi2.ppf(0.999, EDGES.size - 1)
    msg = ('eps %.4f +- %.4f (truth %.2f, realized %.4f), %d iterations, KKT %.1e, covariance %.1e, 2 dlnL %.2f '
           '(bound %.1f)\nomega_k  %s\ntruth    %s\nerror    %s\n'
           'correlation of the weights above zero in rho 2-4:\n%s' % (
               r['eps'], r['err_eps'], EPS_TWO, fl['gH'].mean(), r['n_iter'], kkt, cov_err, lr, bound,
               ' '.join('%.4f' % v for v in r['omega'][1:]), ' '.join('%.4f' % v for v in w[1:]),
               ' '.join('%.4f' % v for v in err), np.round(corr, 2)))
    print('\n' + msg)
    assert kkt < 1e-6 and cov_err < 1e-3 and -1e-6 < lr < bound, msg


def test_fit_noise_gaussian():
    """fit_noise on Gaussian data (L of the same plant): the optimality conditions hold to 1e-6, and eps is consistent
    with 0: 2 (ln L_max - ln L(eps = 0)) < chi2_13(0.999)"""
    ev, _ = plant('two')
    r = mixture.fit_noise(ev['EL'], log=True)
    msg = 'eps %.4f +- %.4f, %d iterations, 2 dlnL against eps = 0: %.2f (bound %.1f), weights above zero: %s' % (
        r['eps'], r['err_eps'], r['n_iter'], 2 * r['lnl'], chi2.ppf(0.999, EDGES.size - 1),
        np.nonzero(r['omega'][1:])[0])
    print('\n' + msg)
    assert _kkt(ev['EL'], r['omega']) < 1e-6 and 2 * r['lnl'] < chi2.ppf(0.999, EDGES.size - 1), msg


@lru_cache(maxsize=None)
def stage_one(source):
    """fit_noise of each detector on plant(source)"""
    ev, _ = plant(source)
    return {I: mixture.fit_noise(ev['E' + I], log=True) for I in 'HL'}


@lru_cache(maxsize=None)
def stage_two(source, eps_free):
    """nbar on the loud plant with the binned laws of stage_one(source) fixed and lambda computed at them"""
    s1 = stage_one(source)
    z, _ = toy().data(draws(), NBAR, EPS, dict(H=pis('loud'), L=pis('loud')))
    ev = dict(plant('loud')[0])
    ev.update({k: v for k, v in toy().evidences(z, [{I: s1[I]['pi_k'] for I in 'HL'}]).items()
               if k in ('gH', 'gL', 'HL')})
    return mixture.fit(ev['S'], ev['gH'][0], ev['gL'][0], ev['EH'], ev['EL'], EDGES, log=True, lam_gHL=ev['HL'][0],
                       pi_k_H=s1['H']['pi_k'], pi_k_L=s1['L']['pi_k'], fixed=() if eps_free else ('eps_H', 'eps_L'),
                       start=dict(nbar=NBAR, eps_H=s1['H']['eps'], eps_L=s1['L']['eps']))


def _stage_msg(source):
    s1, a, b = stage_one(source), stage_two(source, False), stage_two(source, True)
    return ('stage 1 on %s: eps_H %.4f +- %.4f, eps_L %.4f +- %.4f (truth %.2f, %.2f); stage 2, eps fixed: nbar '
            '%.4f +- %.4f (%+.1f sigma); eps free: nbar %.4f +- %.4f (%+.1f sigma), eps_H %.4f, eps_L %.4f' % (
                source, s1['H']['eps'], s1['H']['err_eps'], s1['L']['eps'], s1['L']['err_eps'], EPS['H'], EPS['L'],
                a['nbar'], a['err']['nbar'], (a['nbar'] - NBAR) / a['err']['nbar'], b['nbar'], b['err']['nbar'],
                (b['nbar'] - NBAR) / b['err']['nbar'], b['eps_H'], b['eps_L']))


def test_fixed_bins_match_power_law():
    """fit with the amplitude weights given as bins (gamma_H 3, gamma_L 4 on the loud plant) returns the nbar and ln L
    of the power-law fit with the same laws held fixed, to 1e-9"""
    ev, _ = plant('loud')
    st = start('loud', gamma_L=4.0)
    a = fit(ev, fixed=ONLY_NBAR, start=st)
    b = fit(ev, fixed=('eps_H', 'eps_L'), start=st, pi_k_H=mixture.amplitude_weights(EDGES, 3.0, 6.0),
            pi_k_L=mixture.amplitude_weights(EDGES, 4.0, 6.0))
    msg = 'power law: nbar %.9g, ln L %.9g; bins: nbar %.9g, ln L %.9g' % (a['nbar'], a['lnl'], b['nbar'], b['lnl'])
    assert abs(a['nbar'] / b['nbar'] - 1) < 1e-9 and abs(a['lnl'] - b['lnl']) < 1e-9 * abs(a['lnl']), msg


def test_fit_rho_max():
    """fit with rho_max = 12 (gamma_H 3, gamma_L 4 held, loud plant) returns the nbar and ln L of the fit given the bins
    of amplitude_weights(..., 12) to 1e-9, and not those without rho_max (the loud law reaches 30); with rho_min free
    its box stops at 10, the last edge below rho_max, and the fit runs"""
    ev, _ = plant('loud')
    st = start('loud', gamma_L=4.0)
    a = fit(ev, fixed=ONLY_NBAR, start=st, rho_max=12.0)
    b = fit(ev, fixed=('eps_H', 'eps_L'), start=st, pi_k_H=mixture.amplitude_weights(EDGES, 3.0, 6.0, 12.0),
            pi_k_L=mixture.amplitude_weights(EDGES, 4.0, 6.0, 12.0))
    c = fit(ev, fixed=ONLY_NBAR, start=st)
    msg = 'rho_max 12: nbar %.9g, ln L %.9g; bins: nbar %.9g, ln L %.9g; no rho_max: nbar %.9g, ln L %.9g' % (
        a['nbar'], a['lnl'], b['nbar'], b['lnl'], c['nbar'], c['lnl'])
    assert abs(a['nbar'] / b['nbar'] - 1) < 1e-9 and abs(a['lnl'] - b['lnl']) < 1e-9 * abs(a['lnl']), msg
    assert abs(a['lnl'] - c['lnl']) > 1.0, msg
    r = fit(ev, fixed=('nbar', 'eps_H', 'eps_L', 'gamma_H', 'gamma_L'), start=dict(st, rho_min_H=9.5, rho_min_L=9.5),
            rho_max=12.0)
    assert 2.0 <= r['rho_min_H'] <= 10.0 and 2.0 <= r['rho_min_L'] <= 10.0, (r['rho_min_H'], r['rho_min_L'])
    v = (12.0, 0.1, 1.0)
    s = fit(ev, fixed=ONLY_NBAR, start=st, veto=v)
    t = fit(ev, fixed=('eps_H', 'eps_L'), start=st, pi_k_H=mixture.amplitude_weights(EDGES, 3.0, 6.0, veto=v),
            pi_k_L=mixture.amplitude_weights(EDGES, 4.0, 6.0, veto=v))
    msg = 'soft veto: nbar %.9g, ln L %.9g; bins: nbar %.9g, ln L %.9g' % (s['nbar'], s['lnl'], t['nbar'], t['lnl'])
    assert abs(s['nbar'] / t['nbar'] - 1) < 1e-9 and abs(s['lnl'] - t['lnl']) < 1e-9 * abs(s['lnl']), msg
    assert abs(s['lnl'] - a['lnl']) > 1.0, msg


def test_two_stage_loud():
    """two-stage route on the loud plant, stage 1 (fit_noise per detector) on the same glitches without the mergers,
    stage 2 nbar with eps and the binned laws fixed: nbar within 2.5 sigma of the truth, as is the joint power-law fit
    of the same data"""
    a, j = stage_two('glitch', False), full_fit('loud')
    msg = '%s\njoint power-law fit: nbar %.4f +- %.4f (%+.1f sigma)' % (
        _stage_msg('glitch'), j['nbar'], j['err']['nbar'], (j['nbar'] - NBAR) / j['err']['nbar'])
    print('\n' + msg)
    assert within(a, dict(nbar=NBAR), 2.5, ['nbar']) and within(j, dict(nbar=NBAR), 2.5, ['nbar']), msg


@pytest.mark.parametrize('eps_free', [
    pytest.param(False, marks=pytest.mark.xfail(strict=True, reason='stage 1 absorbs the single-detector part of the '
                 'mergers (eps_H 0.151, eps_L 0.090 for 0.10, 0.03): nbar -3.5% against the joint fit, -0.94 sigma at '
                 '40000 segments (6 seeds, loud; -4.8% weak)')),
    pytest.param(True, marks=pytest.mark.xfail(strict=True, reason='eps free in stage 2 with the stage-1 law: '
                 'nbar -2.9% against the joint fit, -0.78 sigma at 40000 segments (6 seeds, loud; +0.4% weak)'))])
def test_two_stage_mergers_ignored(eps_free):
    """two-stage route as on real data: stage 1 on the loud plant itself, its mergers taken for noise; nbar moves from
    the joint fit of the same data by less than a quarter of its statistical error"""
    r, j = stage_two('loud', eps_free), full_fit('loud')
    d = r['nbar'] - j['nbar']
    msg = '%s\njoint: nbar %.4f; shift %+.4f = %+.1f%% = %+.2f sigma' % (
        _stage_msg('loud'), j['nbar'], d, 100 * d / j['nbar'], d / r['err']['nbar'])
    print('\n' + msg)
    assert abs(d) < 0.25 * r['err']['nbar'], msg
