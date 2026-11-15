"""
the population likelihood of segments with glitches (docs/glitch_likelihood.md, Secs. 2 and 4), on per-segment
arrays.

Each detector I is Gaussian noise with probability 1 - eps_I, and Gaussian noise plus one glitch with probability
eps_I.  The glitch amplitude has the law dN/drho ~ rho^-gamma_I on [rho_min_I, rho_max], carried by the bins of
rho_edges with the weights pi_k; E_I[m, k] is the evidence of a glitch of bin k in template m:

    G_I = sum_m pi_m sum_k pi_k E_I[m, k]                     glitch evidence
    c_I = 1 - eps_I + eps_I G_I                               non-Gaussian over Gaussian noise density

    B = [ (1-eps_H)(1-eps_L) Lambda_S + eps_H (1-eps_L) G_H lambda_H + (1-eps_H) eps_L G_L lambda_L
          + eps_H eps_L G_H G_L lambda_HL ] / (c_H c_L)

    ln L = sum_i [ ln c_H,i + ln c_L,i + ln(1 + nbar (B_i - 1)) ]

Lambda_S is the coherent likelihood ratio, lambda_I the merger likelihood ratio given a glitch in I and lambda_HL
given a glitch in each detector, each averaged over the population first (population.py).  Without lambda_HL the
last term is G_H G_L (lambda_HL = 1: normalized, no merger information when both detectors have a glitch).
sg=False drops the three S+g terms (Smith & Thrane 2018), for comparison; lambda_H = Lambda_S^L, lambda_L =
Lambda_S^H discards the glitched detector over the whole segment.  B has mean 1 over the segments without a merger
when the noise follows the model.  The ln_ functions take and return logarithms, for segments loud enough to
overflow.

    fit          the maximum of ln L over nbar, eps_I and the amplitude law (power law, or fixed bins), with errors
    fit_noise    eps and free amplitude weights of one detector from its glitch evidences alone (no mergers)
    time_curve   ln of the evidence ratio against no mergers on the first n segments, for every n
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import minimize
from scipy.special import exprel, log_ndtr, logsumexp

PARAMETERS = ('nbar', 'eps_H', 'eps_L', 'gamma_H', 'gamma_L', 'rho_min_H', 'rho_min_L')
_GL = np.polynomial.legendre.leggauss(24)


def _log(x):
    with np.errstate(divide='ignore'):
        return np.log(x)


def _trail(a, ndim):
    """a (n_segment,) with trailing axes of length 1 up to ndim"""
    a = np.asarray(a, float)
    return a.reshape(a.shape + (1,) * (ndim - a.ndim))


def ln_amplitude_weights(rho_edges, gamma, rho_min, rho_max=None, veto=None):
    """ln pi_k (n_rho,), see amplitude_weights"""
    e = np.asarray(rho_edges, float)
    a, b = np.maximum(e[:-1], rho_min), e[1:] if rho_max is None else np.minimum(e[1:], rho_max)
    ok = b > a
    if not ok.any() or a[ok][0] <= 0.0:
        raise ValueError('rho_min = %g, rho_max = %s leave no bin of positive amplitude in [%g, %g]'
                         % (rho_min, rho_max, e[0], e[-1]))
    if veto is not None:
        cut, offset, width = veto
        la, lb = np.log(a[ok]), np.log(b[ok])
        x = 0.5 * (lb - la)[:, None] * _GL[0][None, :] + 0.5 * (lb + la)[:, None]
        lnf = (1.0 - gamma) * x + log_ndtr((cut - np.exp(x) - offset) / width) + np.log(_GL[1])[None, :]
        lnw = np.full(a.size, -np.inf)
        lnw[ok] = np.log(0.5 * (lb - la)) + logsumexp(lnf, axis=1)
        return lnw - logsumexp(lnw)
    la = np.log(np.where(ok, a, 1.0))
    d = np.log(np.where(ok, b, np.e)) - la
    x = (1.0 - gamma) * d                                   # int_a^b rho^-gamma = a^(1-gamma) d exprel(x)
    lnw = np.where(ok, (1.0 - gamma) * la + np.log(d) + np.maximum(x, 0.0) + np.log(exprel(-np.abs(x))), -np.inf)
    return lnw - logsumexp(lnw)


def amplitude_weights(rho_edges, gamma, rho_min, rho_max=None, veto=None):
    """pi_k (n_rho,): the probability of the bin [rho_edges[k], rho_edges[k+1]] for dN/drho ~ rho^-gamma on
    [rho_min, rho_max] and zero outside, normalized over the bins (rho_edges (n_rho + 1,) ascending; rho_max None: the
    last edge).  veto = (cut, offset, width): the law of the glitches that pass a veto on the measured SNR at cut, the
    measured SNR being rho + offset + width N(0, 1): rho^-gamma times Phi((cut - rho - offset) / width) integrated over
    each bin (Gauss-Legendre in ln rho, 24 nodes), in place of a hard cut"""
    return np.exp(ln_amplitude_weights(rho_edges, gamma, rho_min, rho_max, veto))


def _template_average(E, pi_m, log):
    """sum_m pi_m E[:, m, :] (n_segment, n_rho), of ln E when log"""
    E = np.asarray(E, float)
    E = E[:, None, :] if E.ndim == 2 else E
    pm = np.full(E.shape[1], 1.0 / E.shape[1]) if pi_m is None else np.asarray(pi_m, float)
    if log:
        with np.errstate(divide='ignore'):
            return logsumexp(E, axis=1, b=pm[None, :, None])
    return np.einsum('imk,m->ik', E, pm)


def glitch_evidence(E, pi_k, pi_m=None):
    """G (n_segment,) = sum_m pi_m sum_k pi_k E[i, m, k].

    E      (n_segment, n_template, n_rho), or (n_segment, n_rho) for one template
    pi_k   (n_rho,) amplitude weights
    pi_m   (n_template,) distribution of the glitches over the templates (uniform when None)"""
    return np.einsum('ik,k->i', _template_average(E, pi_m, False), np.asarray(pi_k, float))


def ln_glitch_evidence(ln_E, pi_k, pi_m=None):
    """ln G (n_segment,) from ln E, see glitch_evidence"""
    with np.errstate(divide='ignore'):
        return logsumexp(_template_average(ln_E, pi_m, True), axis=1, b=np.asarray(pi_k, float)[None, :])


def noise_ratio(G, eps):
    """c = 1 - eps + eps G, the non-Gaussian over the Gaussian noise density"""
    return 1.0 - eps + eps * np.asarray(G, float)


def ln_noise_ratio(ln_G, eps):
    """ln c from ln G"""
    return np.logaddexp(_log(1.0 - eps), _log(eps) + np.asarray(ln_G, float))


def mixture(lam, lam_gH, lam_gL, G_H, G_L, eps_H, eps_L, sg=True, lam_gHL=None):
    """B of Sec. 4 (the shape of lam).

    lam             (n_segment, ...) Lambda_S, population-averaged (n_segment,) or on a grid (n_segment, ...)
    lam_gH, lam_gL  (n_segment, ...) lambda_H, lambda_L: the merger likelihood ratio given a glitch in H, in L
    G_H, G_L        (n_segment,) glitch evidences
    eps_H, eps_L    glitch probabilities
    sg              False drops the three S+g terms (Smith & Thrane 2018); lam_g* are then unused
    lam_gHL         (n_segment, ...) lambda_HL, the merger likelihood ratio given a glitch in each detector
                    (S+gH+gL term G_H G_L lam_gHL); None: 1"""
    lam = np.asarray(lam, float)
    gH, gL = _trail(G_H, lam.ndim), _trail(G_L, lam.ndim)
    num = (1.0 - eps_H) * (1.0 - eps_L) * lam
    if sg:
        lH, lL = np.asarray(lam_gH, float), np.asarray(lam_gL, float)
        both = 1.0 if lam_gHL is None else np.asarray(lam_gHL, float)
        num = num + eps_H * (1.0 - eps_L) * gH * lH + (1.0 - eps_H) * eps_L * gL * lL + eps_H * eps_L * gH * gL * both
    return num / (noise_ratio(gH, eps_H) * noise_ratio(gL, eps_L))


def ln_mixture(ln_lam, ln_lam_gH, ln_lam_gL, ln_G_H, ln_G_L, eps_H, eps_L, sg=True, ln_lam_gHL=None):
    """ln B from the logarithms of the arguments of mixture"""
    return _ln_mixture(ln_lam, ln_lam_gH, ln_lam_gL, ln_G_H, ln_G_L, eps_H, eps_L, sg, ln_lam_gHL)[0]


def _ln_mixture(ln_lam, ln_lam_gH, ln_lam_gL, ln_G_H, ln_G_L, eps_H, eps_L, sg, ln_lam_gHL):
    """ln B, ln c_H, ln c_L"""
    L = np.asarray(ln_lam, float)
    gH, gL = _trail(ln_G_H, L.ndim), _trail(ln_G_L, L.ndim)
    aH, aL, bH, bL = _log(1.0 - eps_H), _log(1.0 - eps_L), _log(eps_H), _log(eps_L)
    num = aH + aL + L
    if sg:
        lH, lL = np.asarray(ln_lam_gH, float), np.asarray(ln_lam_gL, float)
        both = 0.0 if ln_lam_gHL is None else np.asarray(ln_lam_gHL, float)
        num = np.logaddexp(num, bH + aL + gH + lH)
        num = np.logaddexp(num, aH + bL + gL + lL)
        num = np.logaddexp(num, bH + bL + gH + gL + both)
    cH, cL = ln_noise_ratio(gH, eps_H), ln_noise_ratio(gL, eps_L)
    return num - cH - cL, cH, cL


def _ln_one_plus(nbar, ln_B):
    """ln(1 + nbar (B - 1)) from ln B, broadcast"""
    return np.logaddexp(_log(1.0 - nbar), _log(nbar) + ln_B)


def log_likelihood(nbar, B, c_H, c_L, log=False):
    """sum_i [ln c_H,i + ln c_L,i + ln(1 + nbar (B_i - 1))].

    B, c_H, c_L  (n_segment,), their logarithms when log"""
    B = np.asarray(B, float)
    cH, cL = (np.broadcast_to(np.asarray(c, float), B.shape) for c in (c_H, c_L))
    if log:
        return float(cH.sum() + cL.sum() + _ln_one_plus(nbar, B).sum())
    return float(np.log(cH).sum() + np.log(cL).sum() + np.log1p(nbar * (B - 1.0)).sum())


def time_curve(B, nbar_grid, prior, log=False, chunk=4096):
    """ln of the evidence ratio of mergers at a rate nbar ~ prior against no mergers on the first n segments,
    ln int d nbar prior(nbar) prod_{i<=n} (1 + nbar (B_i - 1)), for every n.

    B          (n_segment,) (ln B when log)
    nbar_grid  (n_grid,) ascending, the prior integrated on it by the trapezoid rule
    prior      (n_grid,) density on the grid, or a function of nbar; normalized on the grid here

    returns (n_segment,)"""
    lnB = np.asarray(B, float) if log else _log(np.asarray(B, float))
    g = np.asarray(nbar_grid, float)
    p = np.asarray(prior(g) if callable(prior) else prior, float)
    tw = np.zeros_like(g)
    tw[1:] += 0.5 * np.diff(g)
    tw[:-1] += 0.5 * np.diff(g)
    lw = _log(p * tw / np.sum(p * tw))
    out = np.empty(lnB.size)
    run = np.zeros(g.size)
    for c0 in range(0, lnB.size, chunk):
        S = run + np.cumsum(_ln_one_plus(g[None, :], lnB[c0:c0 + chunk, None]), axis=0)
        out[c0:c0 + S.shape[0]] = logsumexp(S + lw[None, :], axis=1)
        run = S[-1]
    return out


def _scaled(lnA):
    """(per-segment maximum, A / maximum) of a (n_segment, n_rho) array given as ln A"""
    mx = lnA.max(axis=1)
    return mx, np.exp(lnA - mx[:, None])


def _ln_dot(scaled, pi_k):
    mx, A = scaled
    return mx + _log(np.einsum('ik,k->i', A, pi_k))


class _Model:
    """ln L(theta) on fixed per-segment arrays; per-bin arrays are rescaled per segment for a fast pi_k product"""

    def __init__(self, lam, lam_gH, lam_gL, lam_gHL, E_H, E_L, rho_edges, pi_m, log, sg, pi_k, rho_max=None, veto=None):
        conv = (lambda a: np.asarray(a, float)) if log else _log
        avg = (lambda a: _template_average(a, pi_m, True)) if log else \
            (lambda a: _log(_template_average(a, pi_m, False)))
        self.lnS = conv(lam)
        self.lnHL = None if lam_gHL is None else conv(lam_gHL)
        self.edges, self.sg, self.pi_k, self.rho_max, self.veto = np.asarray(rho_edges, float), sg, pi_k, rho_max, veto
        self.A = {I: _scaled(avg(E)) for I, E in (('H', E_H), ('L', E_L))}
        self.lng, self.Y = {}, {}
        for I, a in (('H', lam_gH), ('L', lam_gL)):
            per_bin = np.ndim(a) > 1
            self.Y[I] = _scaled(avg(a)) if per_bin else None
            self.lng[I] = None if per_bin else conv(a)

    def __call__(self, th):
        lnG, lng = {}, {}
        for I in 'HL':
            pk = self.pi_k[I]
            if pk is None:
                pk = amplitude_weights(self.edges, th['gamma_' + I], th['rho_min_' + I], self.rho_max, self.veto)
            lnG[I] = _ln_dot(self.A[I], pk)
            lng[I] = self.lng[I] if self.Y[I] is None else _ln_dot(self.Y[I], pk) - lnG[I]
        lnB, cH, cL = _ln_mixture(self.lnS, lng['H'], lng['L'], lnG['H'], lnG['L'], th['eps_H'], th['eps_L'],
                                  self.sg, self.lnHL)
        return log_likelihood(th['nbar'], lnB, cH, cL, log=True)


def _logit(p):
    return np.log(p) - np.log1p(-p)


def _hessian(f, x, lo, hi, h, diagonal=False):
    """d^2 f / dx dx by central differences of steps h, the stencil shifted inside [lo, hi]"""
    n = x.size
    x = np.clip(x, np.minimum(lo + 2 * h, (lo + hi) / 2), np.maximum(hi - 2 * h, (lo + hi) / 2))
    E = np.diag(h)
    f0 = f(x)
    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = (f(x + E[i]) - 2.0 * f0 + f(x - E[i])) / h[i] ** 2
        for j in range(0 if diagonal else i):
            H[i, j] = H[j, i] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j]) - f(x - E[i] + E[j])
                                 + f(x - E[i] - E[j])) / (4.0 * h[i] * h[j])
    return H


def fit(lam, lam_gH, lam_gL, E_H, E_L, rho_edges, start=None, fixed=('rho_min_H', 'rho_min_L'), bounds=None,
        pi_m=None, log=False, sg=True, lam_gHL=None, pi_k_H=None, pi_k_L=None, rho_max=None, veto=None):
    """the maximum of ln L over PARAMETERS = (nbar, eps_H, eps_L, gamma_H, gamma_L, rho_min_H, rho_min_L).

    lam                  (n_segment,) Lambda_S averaged over the population
    lam_gH, lam_gL       (n_segment,) lambda_H, lambda_L averaged over the population, computed with a fiducial
                         glitch law and held fixed here; or (n_segment, [n_template,] n_rho) the per-bin Y_I[m, k]
                         with G_I lambda_I = sum_m pi_m sum_k pi_k Y_I[m, k], which keeps lambda_I exact in the
                         law (lambda_I = glitch_evidence(Y_I, pi_k) / glitch_evidence(E_I, pi_k))
    E_H, E_L             (n_segment, n_template, n_rho) or (n_segment, n_rho): E_I[m, k]
    rho_edges            (n_rho + 1,) edges of the amplitude bins
    start                dict of starting values (default nbar 0.01, eps 0.01, gamma 3, rho_min the lower
                         bound below: rho_edges[0], or 1e-3 rho_edges[1] when that is 0)
    fixed                names held at their start value; rho_min is fixed by default, ln L has kinks in it at
                         the edges
    bounds               dict name: (lo, hi) (default nbar, eps in [0, 1], gamma in [-10, 20], rho_min between
                         the first and the next-to-last edge, and not above the last edge below rho_max)
    pi_m                 (n_template,) distribution of the glitches over the templates (uniform when None)
    log                  the arrays are logarithms
    sg                   False drops the S+g terms (Smith & Thrane 2018)
    lam_gHL              (n_segment,) lambda_HL, with a fiducial law (None: 1, see mixture)
    pi_k_H, pi_k_L       (n_rho,) fixed amplitude weights of a detector (e.g. from fit_noise) in place of the power
                         law, whose gamma and rho_min are then fixed
    rho_max              upper end of the power law of both detectors, held fixed: no weight above it, the bins
                         renormalized (None: the last edge), e.g. sqrt(2 w) of a veto on the loudest w
    veto                 (cut, offset, width), held fixed: the law times the probability of passing a veto on the
                         measured SNR, a soft cut (amplitude_weights)

    L-BFGS-B in logit(nbar), logit(eps), gamma, ln rho_min; the covariance is the inverse of minus the Hessian in
    the parameters themselves, by central differences (moved inside the bounds for an estimate on one).

    returns dict: every parameter, err and free (the free ones; nan when the Hessian is singular), cov
    (n_free, n_free), lnl (the maximum), result (scipy OptimizeResult)"""
    pk = dict(H=pi_k_H, L=pi_k_L)
    model = _Model(lam, lam_gH, lam_gL, lam_gHL, E_H, E_L, rho_edges, pi_m, log, sg,
                   {I: None if v is None else np.asarray(v, float) for I, v in pk.items()}, rho_max, veto)
    fixed = tuple(fixed) + tuple(p + '_' + I for I in 'HL' if pk[I] is not None for p in ('gamma', 'rho_min'))
    e = model.edges
    rmin = e[0] if e[0] > 0 else 1e-3 * e[1]
    th0 = dict(nbar=0.01, eps_H=0.01, eps_L=0.01, gamma_H=3.0, gamma_L=3.0, rho_min_H=rmin, rho_min_L=rmin)
    th0.update(start or {})
    top = e[-2] if rho_max is None else min(e[-2], e[e < rho_max].max())
    box = dict(nbar=(0.0, 1.0), eps_H=(0.0, 1.0), eps_L=(0.0, 1.0), gamma_H=(-10.0, 20.0), gamma_L=(-10.0, 20.0),
               rho_min_H=(rmin, top), rho_min_L=(rmin, top))
    box.update(bounds or {})
    unknown = set(fixed) - set(PARAMETERS)
    if unknown:
        raise ValueError('unknown parameters %s' % sorted(unknown))
    free = [p for p in PARAMETERS if p not in fixed]
    kind = {p: 'p' if p[0] in 'ne' else ('g' if p[0] == 'g' else 'r') for p in PARAMETERS}
    to_phi = {'p': _logit, 'g': lambda v: v, 'r': np.log}
    from_phi = {'p': lambda f: 1.0 / (1.0 + np.exp(-f)), 'g': lambda f: f, 'r': np.exp}
    clip = {'p': lambda v: np.clip(v, 1e-10, 1.0 - 1e-10), 'g': lambda v: v, 'r': lambda v: v}

    def theta(x, natural):
        th = dict(th0)
        for p, v in zip(free, x):
            th[p] = v if natural else from_phi[kind[p]](v)
        return th

    n = np.asarray(lam).shape[0]
    x0 = np.array([to_phi[kind[p]](clip[kind[p]](np.clip(th0[p], *box[p]))) for p in free])
    pb = [tuple(to_phi[kind[p]](clip[kind[p]](np.asarray(box[p], float)))) for p in free]
    res = minimize(lambda x: -model(theta(x, False)) / n, x0, method='L-BFGS-B', jac='2-point', bounds=pb,
                   options=dict(ftol=1e-15, gtol=1e-10, maxiter=2000))
    est = theta(res.x, False)
    for p in free:                                          # an estimate on a bound is the bound itself
        if res.x[free.index(p)] <= pb[free.index(p)][0] + 1e-9:
            est[p] = box[p][0]
        elif res.x[free.index(p)] >= pb[free.index(p)][1] - 1e-9:
            est[p] = box[p][1]
    x = np.array([est[p] for p in free])
    lo = np.array([box[p][0] for p in free])
    hi = np.array([box[p][1] for p in free])
    f = lambda v: model(theta(v, True))
    h = np.array([1e-4 * max(abs(est[p]), 1e-2) for p in free])
    cov = np.zeros((0, 0))
    if free:
        d = np.diag(_hessian(f, x, lo, hi, h, diagonal=True))
        h = np.minimum(np.where(d < 0, 0.1 / np.sqrt(np.abs(d)), h), 0.25 * (hi - lo))
        try:
            cov = np.linalg.inv(-_hessian(f, x, lo, hi, h))
        except np.linalg.LinAlgError:                         # a direction ln L does not depend on
            cov = np.full((len(free), len(free)), np.nan)
    out = dict(est)
    out.update(err={p: float(np.sqrt(cov[i, i])) if cov[i, i] > 0 else np.nan for i, p in enumerate(free)},
               free=free, cov=cov, lnl=model(est), result=res)
    return out




def fit_noise(E, pi_m=None, log=False, n_iter=500, tol=1e-9, floor=1e-10):
    """eps and free amplitude weights of one detector from its glitch evidences, no mergers: the maximum of

        sum_i ln(omega_0 + sum_k omega_k Ebar_i[k]),   omega_0 = 1 - eps, omega_k = eps pi_k,  omega on the simplex

    (Ebar the template average of E with pi_m), concave in omega.  Each iteration is an EM step followed by a Newton
    step on the weights above zero, kept when it raises ln L.  A weight below floor (relative to eps) is set to zero
    and leaves the set; once the KKT conditions hold on the set, the zero weight with the largest gradient comes
    back.  Converged when mean_i Ebar_i[k] / D_i = 1 to tol for every weight above zero and <= 1 + tol for the
    others (D_i = omega_0 + sum_k omega_k Ebar_i[k], Ebar_i[0] = 1).

    E          (n_segment, n_template, n_rho) or (n_segment, n_rho), ln E when log
    returns dict: eps, pi_k (n_rho,), omega (n_rho + 1,), err_eps, err_pi (n_rho,), cov (n_rho, n_rho) of omega_k
    for k >= 1 (zero rows for weights at zero), lnl (= sum_i ln c_i), n_iter, converged"""
    lnA = _template_average(E, pi_m, True) if log else _log(_template_average(E, pi_m, False))
    m = np.maximum(lnA.max(axis=1), 0.0)
    X = np.concatenate([np.exp(-m)[:, None], np.exp(lnA - m[:, None])], axis=1)      # Ebar_i[k] / e^m_i, k = 0..K
    n, K1 = X.shape
    lnl = lambda w: float(np.sum(m) + np.sum(np.log(np.einsum('ik,k->i', X, w))))
    grad = lambda w: np.einsum('ik,i->k', X, 1.0 / np.einsum('ik,k->i', X, w)) / n
    w = np.full(K1, 0.01 / (K1 - 1))
    w[0] = 0.99
    converged = False
    for it in range(1, n_iter + 1):
        g = grad(w)
        on = w > 0
        if np.all(np.abs(g[on] - 1) < tol):
            k = np.argmax(np.where(on, -np.inf, g))
            if on.all() or g[k] <= 1 + tol:
                converged = True
                break
            w[k] = max(1e-8, 1e-3 * (1.0 - w[0]))
            w /= w.sum()
            continue
        w = w * g                                                                      # EM
        w[1:][w[1:] < floor * max(1.0 - w[0], floor)] = 0.0
        w /= w.sum()
        l = lnl(w)
        S = np.nonzero(w > 0)[0]
        while S.size > 1:                                                              # Newton on the set
            D = np.einsum('ik,k->i', X, w)
            g = np.einsum('ik,i->k', X[:, S], 1.0 / D)
            H = -np.einsum('ik,ij,i->kj', X[:, S], X[:, S], 1.0 / D ** 2)
            try:
                a, b = np.linalg.solve(H, g), np.linalg.solve(H, np.ones(S.size))
            except np.linalg.LinAlgError:
                break
            step = b * a.sum() / b.sum() - a                                           # sum(step) = 0
            neg = (step < 0) & (S > 0)
            r = np.where(neg, -w[S] / np.where(neg, step, -1.0), np.inf)
            j = np.argmin(r)
            if r[j] < 1 and w[S[j]] < 1e-6 * (1.0 - w[0]):                             # a tiny weight blocks it
                w[S[j]] = 0.0
                w /= w.sum()
                S = np.delete(S, j)
                continue
            t = min(1.0, r[j])
            for _ in range(10):
                cand = w.copy()
                cand[S] = np.maximum(w[S] + t * step, 0.0)
                if t == r[j]:
                    cand[S[j]] = 0.0
                if cand[0] > 0 and lnl(cand / cand.sum()) > l:
                    w = cand / cand.sum()
                    break
                t *= 0.5
            break
    D = np.einsum('ik,k->i', X, w)
    on = np.nonzero(w[1:] > 0)[0]
    cov = np.zeros((K1 - 1, K1 - 1))
    if on.size:
        Y = X[:, 1 + on] - X[:, :1]                                                  # d D_i / d omega_k, k >= 1
        try:
            cov[np.ix_(on, on)] = np.linalg.inv(np.einsum('ik,ij,i->kj', Y, Y, 1.0 / D ** 2))
        except np.linalg.LinAlgError:
            cov[np.ix_(on, on)] = np.nan
    eps = 1.0 - w[0]
    pi = w[1:] / eps if eps > 0 else np.zeros(K1 - 1)
    J = (np.eye(K1 - 1) - pi[:, None]) / eps if eps > 0 else np.zeros((K1 - 1, K1 - 1))
    var_pi = np.einsum('kj,jl,kl->k', J, cov, J)
    return dict(eps=eps, pi_k=pi, omega=w, err_eps=float(np.sqrt(max(cov.sum(), 0.0))),
                err_pi=np.sqrt(np.maximum(var_pi, 0.0)), cov=cov, lnl=lnl(w), n_iter=it, converged=converged)


class _Windows:
    """sum_i ln c_i of Sec. 4a on fixed window evidences and its derivatives, in x = (f, Om[0], .., Om[K-1]) with
    Om[s] = (1 - q_s, q_s a^s): the factor of a window in state s is D = Om[s] . (1, F), F_b = basis[b] . E_w, kept
    as Om[s] . Y, Y = (1, F) e^-h, h = max(0, ln F) of the row.  Segments in chunks, on threads"""

    def __init__(self, ln_E, basis, identity, chunk):
        self.n, self.nw, nk = ln_E.shape
        self.nb = basis.shape[0]
        if identity:
            lnF = ln_E.reshape(-1, nk)
        else:
            lb = _log(basis)
            lnF = np.concatenate([logsumexp(ln_E[a:a + 256, :, None, :] + lb, axis=-1).reshape(-1, self.nb)
                                  for a in range(0, self.n, 256)])
        self.lnF = lnF
        self.h = np.maximum(lnF.max(axis=1), 0.0)
        self.Y = np.empty((lnF.shape[0], self.nb + 1))
        self.Y[:, 0] = np.exp(-self.h)
        self.Y[:, 1:] = np.exp(lnF - self.h[:, None])
        self.hsum = self.h.reshape(self.n, self.nw).sum(axis=1)
        self.parts = [(a, min(a + chunk, self.n)) for a in range(0, self.n, chunk)]

    def _part(self, ab, f, Om, hess, tiny=1e-150):
        a, b = ab
        nw, K, P = self.nw, f.size, self.nb + 1
        rows = slice(a * nw, b * nw)
        Y = self.Y[rows]
        with np.errstate(all='ignore'):
            D = Y @ Om.T                                                    # (rows, K)
            lnD = np.log(D)
        i, k = np.nonzero(D < tiny)
        if i.size:                                                          # a law without weight where E is large:
            lnY = np.concatenate([-self.h[rows][i, None], self.lnF[rows][i] - self.h[rows][i, None]], axis=1)
            lnD[i, k] = logsumexp(lnY + _log(Om[k]), axis=1)                # these rows in the log domain
            rho = np.exp(np.minimum(lnY - lnD[i, k][:, None], 345.0))      # Y / D
            D[i, k] = np.inf
        lnC = lnD.reshape(b - a, nw, K).sum(axis=1) + self.hsum[a:b, None]
        ln_c = logsumexp(lnC + _log(f), axis=1)
        Q = np.exp(np.minimum(lnC - ln_c[:, None], 345.0))                  # C_s / c = d ln c_i / d f_s
        r = Q * f
        rw = np.repeat(r, nw, axis=0)
        with np.errstate(all='ignore'):
            G = Y.T @ (rw / D)                                              # sum r_s Y / D_s, (P, K)
        if i.size:
            np.add.at(G.T, k, rw[i, k][:, None] * rho)
        out = dict(lnl=math.fsum(ln_c), Q=Q.sum(axis=0), R=r.sum(axis=0), G=G)
        if hess:
            Gi, M = np.empty((b - a, P, K)), np.empty((K, P, P))           # sum_w Y / D_s; sum r_s Y Y^T / D_s^2
            for t in range(K):
                Z = Y / D[:, t:t + 1]
                if i.size:
                    Z[i[k == t]] = rho[k == t]
                Gi[:, :, t] = Z.reshape(b - a, nw, P).sum(axis=1)
                W = Z * np.sqrt(rw[:, t:t + 1])
                with np.errstate(all='ignore'):
                    M[t] = W.T @ W
            with np.errstate(all='ignore'):
                out.update(M=M, QQ=Q.T @ Q, QG=np.einsum('is,ips->sp', Q, Gi),
                           QrG=np.einsum('is,it,ipt->stp', Q, r, Gi), GG=np.einsum('is,ips,iqs->spq', r, Gi, Gi),
                           rrGG=np.einsum('is,it,ips,iqt->sptq', r, r, Gi, Gi))
        return out

    def __call__(self, x, K, hess=False):
        f, Om = x[:K], x[K:].reshape(K, self.nb + 1)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            parts = list(ex.map(lambda ab: self._part(ab, f, Om, hess), self.parts))
        out = {key: sum(p[key] for p in parts) for key in parts[0] if key != 'lnl'}
        out['lnl'] = math.fsum(p['lnl'] for p in parts)
        return out

    def gradient(self, ev):
        """d ln L / dx, and mu (n for f, n_w R_s for Om[s]): EM is x -> x g / mu"""
        K = ev['R'].size
        mu = np.concatenate([np.full(K, float(self.n)), np.repeat(self.nw * ev['R'], self.nb + 1)])
        return np.concatenate([ev['Q'], ev['G'].T.ravel()]), mu

    def hessian(self, ev):
        """d^2 ln L / dx dx: with Q_s = C_s / c, r_s = f_s Q_s and G_s = sum_w Y / D_s per segment, sum_i of
        -Q_s Q_t (f f), Q_s (delta_st - r_t) G_t (f Om), and delta_st r_s (G_s G_s^T - sum_w Y Y^T / D_s^2)
        - r_s r_t G_s G_t^T (Om Om)"""
        K, P = ev['R'].size, self.nb + 1
        H = np.zeros((K + K * P, K + K * P))
        H[:K, :K] = -ev['QQ']
        for s in range(K):
            for t in range(K):
                v = (ev['QG'][s] if s == t else 0.0) - ev['QrG'][s, t]
                H[s, K + t * P:K + (t + 1) * P] = H[K + t * P:K + (t + 1) * P, s] = v
                blk = -ev['rrGG'][s, :, t, :] + ((ev['GG'][s] - ev['M'][s]) if s == t else 0.0)
                H[K + s * P:K + (s + 1) * P, K + t * P:K + (t + 1) * P] = blk
        return H


def _tangent(free, blocks):
    """(n, m): a basis of the steps that keep every block summing to 1 and move only the free entries"""
    Z = []
    for bl in blocks:
        fb = bl[free[bl]]
        for j in fb[1:]:
            z = np.zeros(free.size)
            z[j], z[fb[0]] = 1.0, -1.0
            Z.append(z)
    return np.array(Z).T if Z else np.zeros((free.size, 0))


def fit_windows(ln_E, n_state=2, basis=None, start=None, tol=1e-9, max_iter=500, chunk=1024):
    """the glitch model of one detector, Sec. 4a, from its window evidences (no mergers): the maximum of

        sum_i ln c_i,   c_i = sum_s f_s prod_w (1 - q_s + q_s pi^s . E_iw),   pi^s = a^s @ basis

    over f and every a^s on the simplex and q in [0, 1].

    ln_E     (n_segment, n_w, n_k) ln E_w[k] (single.window_evidence)
    n_state  K
    basis    (n_basis, n_k) rows that are normalized laws over the columns (None: the identity, a free weight per
             column)
    start    dict f (K,), q (K,), a (K, n_basis) (default f uniform, q from 1e-3 to 0.1, a uniform)
    tol      the gain of the Newton step below which it has converged, nats
    max_iter the number of updates
    chunk    segments per task of the threads

    In x = (f, Om[0], .., Om[K-1]), Om[s] = (1 - q_s, q_s a^s), every block on a simplex, c_i is linear in each
    Om[s].  EM over the state of the segment, glitch or not in each window and the basis component of the glitch
    is x -> x g / mu (g = d ln L / dx, mu = n for f and n_w sum_i r_is for Om[s]), accelerated by SQUAREM
    (Varadhan & Roland 2008, the extrapolation kept only when it raises ln L), until a cycle gains less than 0.01;
    then Newton steps with the exact Hessian on the entries above zero and those at zero with g > mu, each block
    kept on its simplex (the eigenvalues of minus the Hessian taken in absolute value, so the step rises where ln L
    is not concave), a step kept only when it raises ln L (an EM cycle otherwise), so ln L never decreases.
    Converged when the gain the Newton step predicts, g . dx / 2, is below tol (nats; the distance to the maximum
    is then about sqrt(2 tol) standard errors), the entries of empty states (f_s = 0) aside.

    returns dict: f (K,), q (K,), pi (K, n_k), a (K, n_basis), lnl, n_iter (updates), converged, trace (ln L after
    every update), cov (covariance of x from the Hessian on the entries above zero, the same order), err_f, err_q,
    err_a; states ordered by increasing q"""
    lnE = np.asarray(ln_E, float)
    nk = lnE.shape[-1]
    B = np.eye(nk) if basis is None else np.asarray(basis, float)
    if B.ndim != 2 or B.shape[1] != nk or np.any(B < 0) or np.any(np.abs(B.sum(axis=1) - 1.0) > 1e-12):
        raise ValueError('the rows of basis must be normalized laws over the %d columns' % nk)
    model = _Windows(lnE, B, basis is None, chunk)
    K, nb = n_state, B.shape[0]
    P = nb + 1
    st = dict(f=np.full(K, 1.0 / K), q=np.geomspace(1e-3, 0.1, K) if K > 1 else np.array([0.01]),
              a=np.full((K, nb), 1.0 / nb))
    st.update(start or {})
    q0 = np.asarray(st['q'], float)
    x = np.concatenate([np.asarray(st['f'], float), np.concatenate([(1.0 - q0)[:, None],
                                                                    q0[:, None] * np.asarray(st['a'], float)], 1).ravel()])
    blocks = [np.arange(K)] + [K + s * P + np.arange(P) for s in range(K)]

    def norm(x):
        x = np.maximum(x, 0.0)
        for bl in blocks:
            x[bl] /= x[bl].sum()
        return x

    def em(x, ev):
        g, mu = model.gradient(ev)
        with np.errstate(all='ignore'):
            y = np.where((x > 0) & (mu > 0), x * g / mu, x)
        return norm(y)

    ev = model(x, K)
    trace, n_iter = [ev['lnl']], 0

    def accept(xn, evn):
        nonlocal x, ev, n_iter
        x, ev, n_iter = xn, evn, n_iter + 1
        trace.append(evn['lnl'])

    def squarem():
        """one SQUAREM cycle; False when the EM step does not raise ln L"""
        x1 = em(x, ev)
        ev1 = model(x1, K)
        if not ev1['lnl'] > ev['lnl']:
            return False
        x2 = em(x1, ev1)
        r, v = x1 - x, x2 - 2.0 * x1 + x
        nv = np.linalg.norm(v)
        al = min(-np.linalg.norm(r) / nv, -1.0) if nv > 0 else -1.0
        while al < -1.0:
            xe = x - 2.0 * al * r + al * al * v
            if np.all(xe >= 0):
                xe = norm(xe)
                eve = model(xe, K)
                accept(x1, ev1)
                if eve['lnl'] >= ev1['lnl']:
                    accept(xe, eve)
                return True
            al = (al - 1.0) / 2.0 if al < -1.01 else -1.0
        accept(x1, ev1)
        return True

    while n_iter < max_iter:
        l0 = ev['lnl']
        if not squarem() or ev['lnl'] - l0 < 1e-2:
            break
    converged = False
    ev = model(x, K, hess=True)
    while n_iter < max_iter:
        g, mu = model.gradient(ev)
        H = model.hessian(ev)
        free = np.concatenate([np.ones(K, bool), np.repeat(x[:K] > 0, P)]) & ((x > 0) | (g > mu))
        while True:                                                        # Newton on the free entries
            Z = _tangent(free, blocks)
            d = np.zeros(x.size)
            if Z.shape[1] == 0:
                break
            with np.errstate(all='ignore'):
                w, V = np.linalg.eigh(-(Z.T @ H @ Z))
                w = np.maximum(np.abs(w), 1e-12 * np.abs(w).max())         # an ascent also where not concave
                d = Z @ (V @ ((V.T @ (Z.T @ g)) / w))
            out = (x <= 0) & (d < 0) & free
            if not out.any():
                break
            free &= ~out
        dec = 0.5 * float(g @ d)                                           # the gain the step predicts
        if not np.isfinite(dec):
            break
        if dec < tol:
            converged = True
            break
        neg = (d < 0) & (x > 0)
        reach = np.where(neg, -x / np.where(neg, d, -1.0), np.inf)
        hit = np.argmin(reach)
        t, moved = min(1.0, reach[hit]), False
        for _ in range(30):
            xn = x + t * d
            if t == reach[hit]:
                xn[hit] = 0.0
            xn = norm(xn)
            evn = model(xn, K, hess=True)
            if evn['lnl'] > ev['lnl']:
                accept(xn, evn)
                moved = True
                break
            t *= 0.5
        if not moved:
            if not squarem():
                break
            ev = model(x, K, hess=True)
    H = model.hessian(ev)
    Z = _tangent(np.concatenate([np.ones(K, bool), np.repeat(x[:K] > 0, P)]) & (x > 0), blocks)
    cov = np.zeros((x.size, x.size))
    if Z.shape[1]:
        with np.errstate(all='ignore'):
            cov = Z @ np.linalg.pinv(-(Z.T @ H @ Z), hermitian=True) @ Z.T
    f, Om = x[:K], x[K:].reshape(K, P)
    o = np.argsort(1.0 - Om[:, 0], kind='stable')
    idx = np.concatenate([o, K + (o[:, None] * P + np.arange(P)).ravel()])
    f, Om, cov = f[o], Om[o], cov[np.ix_(idx, idx)]
    q = 1.0 - Om[:, 0]
    qd = np.where(q > 0, q, 1.0)[:, None]
    a = np.where(q[:, None] > 0, Om[:, 1:] / qd, 1.0 / nb)
    var_a = np.empty((K, nb))
    for s in range(K):                                                     # a = Om[1:] / (1 - Om[0])
        J = np.concatenate([Om[s, 1:, None] / qd[s] ** 2, np.eye(nb) / qd[s]], axis=1)
        c = cov[K + s * P:K + (s + 1) * P, K + s * P:K + (s + 1) * P]
        var_a[s] = np.einsum('kj,jl,kl->k', J, c, J)
    sd = lambda v: np.sqrt(np.maximum(v, 0.0))
    return dict(f=f, q=q, pi=a @ B, a=a, lnl=float(ev['lnl']), n_iter=n_iter, converged=converged,
                trace=np.array(trace), cov=cov, err_f=sd(np.diag(cov)[:K]), err_q=sd(np.diag(cov)[K::P]),
                err_a=sd(var_a))
