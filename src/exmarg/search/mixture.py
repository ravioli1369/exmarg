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
    start                dict of starting values (default nbar 0.01, eps 0.01, gamma 3, rho_min rho_edges[0])
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
    th0 = dict(nbar=0.01, eps_H=0.01, eps_L=0.01, gamma_H=3.0, gamma_L=3.0, rho_min_H=e[0], rho_min_L=e[0])
    th0.update(start or {})
    rmin = e[0] if e[0] > 0 else 1e-3 * e[1]
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
