"""
the single-detector evidences of the glitch-robust likelihood (docs/glitch_likelihood.md, Secs. 2, 3 and 5), on
the CPU (numpy) or the GPU (cupy), the array module passed as xp.

For one arrival time with w = |z_I|^2 / u_I (Exp(1) in Gaussian noise), the likelihood ratio of a transient of
power x and random phase is

  l(w; x)      = exp(-x) I0(2 sqrt(x w)) = exp(w - (sqrt(x) - sqrt(w))^2) i0e(2 sqrt(x w))

and the evidences of a template m are averages over the arrival times of columns of w,

  E[m, k]      = (1/n_t) sum_t L_k(w_m(t)),       L_k(w) = int d rho b_k(rho) l(w; rho^2 / 2)
  Lam^I(m, s)  = (1/n_t) sum_t Phi(w_m(t); s),    Phi(w; s) = < l(w; s |R|^2) >_{sky, psi, chi}

b_k uniform and normalized on the amplitude bin [rho_k, rho_k+1] (a glitch), s = u y^2 (a merger).  Every
column has unit mean over w ~ Exp(1).

The columns are tabulated once, as logarithms, on a grid uniform in v = sqrt(w) (dv = 0.01, v <= 150).  The
arrival times with v <= VLOUD go through the cloud-in-cell histogram H of v, E = H @ Lmat / n_t, which is linear
interpolation of the columns in v, less its curvature term (the deposit D of f (1 - f), f the fraction of the cell,
against the second differences of the columns); the louder ones, where the columns grow like exp(w) and linear
interpolation fails, are summed one by one with the logarithm of the columns interpolated linearly in v.  v beyond
the grid is clamped to its last point.  The evidences are returned as logarithms (a loud glitch overflows them).

|R|^2 = F^2 P is a sky factor F^2 = F+^2 + Fx^2 times a factor P of polarization and inclination, independent of
it.  In the frame of a detector with orthogonal arms F+ = A+(cos theta) cos 2phi and Fx = cos(theta) sin 2phi over
an isotropic sky, so F^2 has the law of P with chi = cos(theta).  P comes from tables._responses at the zenith of
H on a (psi, chi) quadrature, psi uniform (tables._quad) and chi Gauss-Legendre on panels that close in on
|chi| = 1, where |R| -> 1 and the loud arrival times read the distribution; each factor and their product are
compressed to two nodes per cell of |R| that keep the moments 0..3 of the points in the cell.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import i0e as _i0e_np

from ..detectors import u as _arm1, v as _arm2
from ..tables import _quad, _responses

DV = 0.01
NV = 15001                                   # v = DV j, j = 0 .. NV - 1, up to v = 150
V = DV * np.arange(NV)
VLOUD = 2.0                                  # arrival times with larger v are summed one by one
RHO_EDGES = np.concatenate([np.arange(0.0, 8.0, 0.5), np.geomspace(8.0, 200.0, 24)])
_CACHE = {}


def _i0e(xp):
    if xp is np:
        return _i0e_np
    from cupyx.scipy.special import i0e
    return i0e


def _map(f, items):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        return list(ex.map(f, items))


def _dot(A, x):
    with np.errstate(all="ignore"):         # some BLAS raise spurious flags; the operands are finite
        return A @ x


def ln_kernel(w, x, xp=np):
    """ln l(w; x), elementwise"""
    sw, sx = xp.sqrt(w), xp.sqrt(x)
    return w - (sx - sw) ** 2 + xp.log(_i0e(xp)(2.0 * sx * sw))


def kernel(w, x):
    """l(w; x) = exp(-x) I0(2 sqrt(x w)), elementwise (inf where it overflows; ln_kernel does not)"""
    return np.exp(ln_kernel(np.asarray(w, float), np.asarray(x, float)))


def _edges(n, last=1e-8):
    """cells of [0, 1]: width 1/n, and towards 1 width / distance to 1 = 0.1 from distance 10/n down to last"""
    d = np.geomspace(10.0 / n, last, int(np.ceil(np.log(10.0 / n / last) / np.log(1.1))) + 1)
    return np.concatenate([np.linspace(0.0, 1.0 - d[0], int(round(n * (1.0 - d[0]))) + 1), 1.0 - d[1:], [1.0]])


def _chi_quad(n_panel=24):
    """Gauss-Legendre nodes and weights on chi in [0, 1] (|R| is even in chi), panels 0.05 wide up to 0.95 and
    halving the distance to 1 down to 1e-8; sum weights = 1"""
    e = np.concatenate([np.linspace(0.0, 0.95, 20), 1.0 - 0.05 * 0.5 ** np.arange(1, 23), [1.0]])
    x, w = leggauss(n_panel)
    a, b = e[:-1, None], e[1:, None]
    return (a + (b - a) * (1.0 + x) / 2.0).ravel(), ((b - a) * w / 2.0).ravel()


def _compress(q, w, edges):
    """points q in [0, 1] with weights w -> two nodes per cell of edges with the moments 0..3 of its points"""
    nc = edges.size - 1
    c = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, nc - 1)
    width = np.diff(edges)
    x = (q - edges[c]) / width[c] - 0.5
    m0 = np.bincount(c, w, nc)
    ok = m0 > 0
    m1, m2, m3 = (np.bincount(c, w * x ** k, nc)[ok] / m0[ok] for k in (1, 2, 3))
    var = np.maximum(m2 - m1 ** 2, 0.0)
    k3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    one = var <= 1e-24                       # a single point: one node
    a = np.where(one, 0.0, 0.5 * k3 / np.where(one, 1.0, var))
    r = np.sqrt(a * a + var)
    p = np.where(one, 1.0, (r - a) / np.where(one, 1.0, 2.0 * r))          # weight of the upper node
    lo, wd = edges[:-1][ok], width[ok]
    nodes = np.concatenate([lo + wd * (0.5 + m1 + a + r), lo + wd * (0.5 + m1 + a - r)])
    wts = np.concatenate([m0[ok] * p, m0[ok] * (1.0 - p)])
    keep = wts > 0
    return np.clip(nodes[keep], 0.0, 1.0), wts[keep]


def response_distribution(n_psi=1024, n_cell=2000, n_panel=24):
    """the distribution of |R| of one LIGO detector over isotropic sky position, polarization and inclination:
    nodes q (n,) and weights (n,), two per cell of [0, 1] (n_cell uniform cells, finer towards |R| = 1), from
    n_psi polarizations times the chi panels of _chi_quad(n_panel).  sum weights = 1, sum weights q^2 = <|R|^2>
    = 4/25.  cached"""
    key = ("R", n_psi, n_cell, n_panel)
    if key not in _CACHE:
        z = np.cross(_arm1["H"], _arm2["H"])
        z /= np.linalg.norm(z)
        th, ph = np.array([np.arccos(z[2])]), np.array([np.arctan2(z[1], z[0])])
        psi = _quad(0, n_psi, 1)[0]
        cg, wc = _chi_quad(n_panel)
        R = _responses(th, ph, psi, cg)[1][0]                                   # (n_psi, n_chi) at the zenith
        F = abs(_responses(th, ph, np.zeros(1), np.ones(1))[1][0, 0, 0])        # 1 for orthogonal arms
        e = _edges(n_cell)
        q1, w1 = _compress(np.minimum(np.abs(R).ravel() / F, 1.0), np.broadcast_to(wc / n_psi, R.shape).ravel(), e)
        parts = _map(lambda a: _compress((q1[a:a + 256, None] * q1[None, :]).ravel(),
                                         (w1[a:a + 256, None] * w1[None, :]).ravel(), e), range(0, q1.size, 256))
        _CACHE[key] = _compress(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), e)
    return _CACHE[key]


def _cells(s):
    """uniform cells of |R| that resolve Phi(w; s): 16 sqrt(s), at least 32, rounded up to a power of 2"""
    return int(2 ** np.ceil(np.log2(np.maximum(32.0, 16.0 * np.sqrt(s)))))


def _response_nodes(n):
    """response_distribution on n uniform cells, refined towards |R| = 1 down to the scale 0.1 / (2 sqrt(s) V[-1])
    of the loudest arrival time for every s with _cells(s) <= n; nodes sorted"""
    if ("Rn", n) not in _CACHE:
        q, w = _compress(*response_distribution(), _edges(n, 5e-3 / n))
        o = np.argsort(q)
        _CACHE[("Rn", n)] = q[o], w[o]
    return _CACHE[("Rn", n)]


def _ln_sum(vv, t, w, xp):
    """ln sum_j w_j l(vv^2; t_j^2): vv (n, 1), t (n, m) or (1, m), w (m,)"""
    d = xp.maximum(vv - t.max(axis=1, keepdims=True), 0.0) + xp.maximum(t.min(axis=1, keepdims=True) - vv, 0.0)
    K = xp.exp(d * d - (t - vv) ** 2) * _i0e(xp)(2.0 * t * vv)               # <= 1, the largest term ~ 1
    return (vv * vv - d * d)[:, 0] + xp.log(_dot(K, w))


def _ln_phi(v, s, xp=np, chunk=2000000):
    """ln Phi(v^2; s) at the pairs (v[i], s[i]): v (n,) on the device, s (n,) on the host"""
    us, inv = np.unique(np.asarray(s, float), return_inverse=True)
    lev = np.array([_cells(x) for x in us])[inv]
    out = xp.empty(v.shape)
    for n in np.unique(lev):
        q, w = _response_nodes(int(n))
        qd, wd = xp.asarray(q), xp.asarray(w)
        sel = np.nonzero(lev == n)[0]
        step = max(1, chunk // q.size)
        for a in range(0, sel.size, step):
            i = sel[a:a + step]
            idx = xp.asarray(i)
            out[idx] = _ln_sum(v[idx][:, None], xp.asarray(np.sqrt(us[inv[i]]))[:, None] * qd[None, :], wd, xp)
    return out


def _phi_column(s, block=256, reach=10.0):
    """ln Phi(V^2; s) (NV,): per block of v only the nodes with t = |R| sqrt(s) within reach of the block (or of
    the largest t) enter; the others weigh less than exp(-reach^2) relative"""
    q, w = _response_nodes(_cells(s))
    t = q * np.sqrt(s)

    def part(a):
        vb = V[a:a + block]
        lo = np.searchsorted(t, min(vb[0], t[-1]) - reach)
        hi = np.searchsorted(t, max(vb[-1], t[0]) + reach, side="right")
        return _ln_sum(vb[:, None], t[None, lo:hi], w[lo:hi], np)
    return np.concatenate(_map(part, range(0, NV, block)))


def merger_columns(s):
    """ln Phi(V^2; s) for the signal powers s: (NV, n_s).  cached per s"""
    s = np.atleast_1d(np.asarray(s, float))
    for x in s:
        if ("Phi", x) not in _CACHE:
            _CACHE[("Phi", x)] = _phi_column(x)
    return np.stack([_CACHE[("Phi", x)] for x in s], axis=1)


class MergerTable:
    """ln Phi(V[j]^2; s[m]) for a signal power per template, s (n_template,) on the host: the table (n_template, n)
    on the device of xp of the first n grid points, grown when more are asked for (upto).  The columns of
    ln_evidence's own: Phi at a grid point does not depend on the data, so a bank computes it once and not at every
    segment (ln_merger_at computes it at the grid points each segment reaches)"""

    def __init__(self, s, xp=np, n=256):
        self.s, self.xp = np.asarray(s, float), xp
        self.P = xp.empty((self.s.size, 0))
        self.upto(n)

    def upto(self, n):
        """the table with at least the grid points j < n (at most NV of them): (n_template, >= n)"""
        have, n = self.P.shape[1], min(int(n), NV)
        if n > have:
            j = np.arange(have, min(NV, max(n, 2 * have)))
            add = _ln_phi(self.xp.asarray(np.tile(DV * j, self.s.size)), np.repeat(self.s, j.size), self.xp)
            self.P = self.xp.concatenate([self.P, add.reshape(self.s.size, j.size)], axis=1)
        return self.P


def _ln_bin(v, a, b, n=16, reach=25.0):
    """ln int_a^b d rho l(v^2; rho^2 / 2) for v (n_v,), by n-point Gauss-Legendre on the part of [a, b] within
    reach / |lam| of its larger end, lam the secant slope of ln l (cut only when |lam| (b - a) > reach; the rest
    then weighs about exp(-reach) relative)"""
    x, wx = leggauss(n)
    w = v * v
    lam = (ln_kernel(w, 0.5 * b * b) - ln_kernel(w, 0.5 * a * a)) / (b - a)
    c = np.minimum(b - a, reach / np.maximum(np.abs(lam), 1e-300))
    rho = np.where(lam >= 0, b - c, a)[:, None] + c[:, None] * (1.0 + x[None, :]) / 2.0
    g = ln_kernel(w[:, None], 0.5 * rho * rho)
    m = g.max(axis=1)
    return np.log(c / 2.0) + m + np.log(_dot(np.exp(g - m[:, None]), wx))


def glitch_columns(edges=RHO_EDGES, piece=2.0):
    """ln L_k(V^2) of the amplitude bins [edges[k], edges[k+1]]: (NV, n_rho), each bin integrated in pieces at most
    piece wide in rho.  cached"""
    edges = np.asarray(edges, float)
    key = ("L", tuple(edges), piece)
    if key not in _CACHE:
        cuts = [np.linspace(a, b, int(np.ceil((b - a) / piece)) + 1) for a, b in zip(edges[:-1], edges[1:])]
        jobs = [(k, c0, c1) for k, c in enumerate(cuts) for c0, c1 in zip(c[:-1], c[1:])]
        parts = _map(lambda j: _ln_bin(V, j[1], j[2]), jobs)
        cols = []
        for k in range(edges.size - 1):
            p = np.stack([parts[i] for i, j in enumerate(jobs) if j[0] == k])
            m = p.max(axis=0)
            cols.append(m + np.log(np.exp(p - m).sum(axis=0)) - np.log(edges[k + 1] - edges[k]))
        _CACHE[key] = np.stack(cols, axis=1)
    return _CACHE[key]


def _v(lq, ln_u, xp):
    return xp.exp(lq.astype(xp.float64) - 0.5 * xp.asarray(ln_u, xp.float64)[:, None])


def _cic(v, xp):
    """grid point below v and the fraction to the next, v clamped to the grid"""
    g = xp.minimum(v / DV, NV - 1.0)
    j = xp.minimum(xp.floor(g), NV - 2.0)
    return j.astype(xp.int64), g - j


def _weights(weights, nt, xp):
    return xp.ones(nt) if weights is None else xp.asarray(weights, xp.float64)


def _ln_weight_rows(weights, nt, xp, log):
    """the logarithms of the weights as rows (n_w, n_time), and whether they were given as one row"""
    W = xp.asarray(weights, xp.float64)
    if not log:
        with np.errstate(divide="ignore"):
            W = xp.log(W)
    return (W[None, :], True) if W.ndim == 1 else (W, False)


def _run_max(x, r, n, xp):
    """the largest row of x (n_row, n_col) over each run of equal r (r sorted, template rows 0 .. n - 1), by
    doubling: (n, n_col), -inf for the rows without any"""
    counts = xp.bincount(r, minlength=n)
    k, longest = 1, int(counts.max()) if r.size else 0
    while k < longest:
        same = (r[k:] == r[:-k])[:, None]
        x = xp.concatenate([xp.where(same, xp.maximum(x[:-k], x[k:]), x[:-k]), x[-k:]])
        k *= 2
    out = xp.full((n,) + x.shape[1:], -np.inf)
    has = counts > 0
    out[has] = x[(xp.cumsum(counts) - counts)[has]]
    return out


def _ln_loud(x, r, n, shift, lwt, xp, nc):
    """ln sum over the loud times of each template of exp x, x (n_loud, nc) = ln T + ln weight, as (n, nc): shifted
    by the template's largest ln T plus the largest ln weight when the weights of the loud times span less than
    600 nats (then no term within exp(-145) of the largest is lost), else by the largest x of each template"""
    lfin = lwt[lwt > -np.inf]
    if lfin.size and float(lfin.max() - lfin.min()) <= 600.0 and lfin.size == lwt.size:
        sh = shift + lfin.max()
    else:
        sh = _run_max(x, r, n, xp)
        sh = xp.where(sh > -np.inf, sh, 0.0)
    idx = (r[:, None] * nc + xp.arange(nc)[None, :]).ravel()
    tot = xp.bincount(idx, weights=xp.exp(x - sh[r]).ravel(), minlength=n * nc).reshape(n, nc)
    with np.errstate(divide="ignore"):
        return xp.log(tot) + sh


def _wmax(lq, ln_u, xp):
    return xp.exp(2.0 * lq.max(axis=1).astype(xp.float64) - xp.asarray(ln_u, xp.float64))


def histogram(lq, ln_u, weights=None, xp=np, block=256):
    """the cloud-in-cell histogram of v = sqrt(w) over the arrival times, w = exp(2 lq) / u.

    lq        (n_template, n_time) ln|z_I| (bank.trajectories), on the device of xp
    ln_u      (n_template,) ln u_I = ln E_0|z_I|^2
    weights   None or (n_time,) a weight per arrival time

    returns H (n_template, NV) float64, H[m, j] = sum_t weights[t] (cloud-in-cell of v_m(t) at V[j]), v beyond the
    grid clamped to its last point, and wmax (n_template,) the largest w of each template."""
    nm, nt = lq.shape
    wt = _weights(weights, nt, xp)
    H = xp.empty((nm, NV))
    for a in range(0, nm, block):
        b = min(a + block, nm)
        j, f = _cic(_v(lq[a:b], ln_u[a:b], xp), xp)
        idx = (j + NV * xp.arange(b - a)[:, None]).ravel()
        H[a:b] = (xp.bincount(idx, weights=(wt * (1.0 - f)).ravel(), minlength=(b - a) * NV)
                  + xp.bincount(idx + 1, weights=(wt * f).ravel(), minlength=(b - a) * NV)).reshape(b - a, NV)
    return H, _wmax(lq, ln_u, xp)


def _quiet(v, j, f, wt, xp, J):
    """the arrival times with v <= V[J - 2]: their histogram H (n, J) on the grid points and the deposit
    D (n, J - 1) of f (1 - f) on the cells, for the curvature term of the interpolation"""
    n = v.shape[0]
    q = v <= DV * (J - 2)
    wq = xp.where(q, wt, 0.0)
    idx = (xp.where(q, j, 0) + J * xp.arange(n)[:, None]).ravel()
    H = (xp.bincount(idx, weights=(wq * (1.0 - f)).ravel(), minlength=n * J)
         + xp.bincount(idx + 1, weights=(wq * f).ravel(), minlength=n * J)).reshape(n, J)
    D = xp.bincount(idx, weights=(wq * f * (1.0 - f)).ravel(), minlength=n * J).reshape(n, J)[:, :J - 1]
    return H, D


def _quiet_columns(lnT, J, xp):
    """the columns on the grid points below vloud scaled by their largest value there, B (J, n_col), the curvature
    term of the cells, Q (J - 1, n_col), and the scales top (n_col,)"""
    top = lnT[:J + 1].max(axis=0)
    T = xp.exp(lnT[:J + 1] - top[None, :])
    d2 = xp.concatenate([2.0 * (T[1:2] - T[:1]), T[2:] - 2.0 * T[1:-1] + T[:-2]])      # T(-dv) = T(dv)
    return T[:J], xp.minimum(0.25 * (d2[:-1] + d2[1:]), T[:J - 1] + T[1:J]), top


def _quiet_own(P, J, xp):
    """_quiet_columns for a column per template, P (n, >= J + 1) ln T_m at the grid points: B (n, J), Q (n, J - 1) and
    the scales top (n,)"""
    top = P[:, :J + 1].max(axis=1)
    T = xp.exp(P[:, :J + 1] - top[:, None])
    d2 = xp.concatenate([2.0 * (T[:, 1:2] - T[:, :1]), T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]], axis=1)
    return T[:, :J], xp.minimum(0.25 * (d2[:, :-1] + d2[:, 1:]), T[:, :J - 1] + T[:, 1:J]), top


def ln_evidence(lq, ln_u, lnT, weights=None, xp=np, block=256, vloud=VLOUD, log_weights=False, own=None):
    """ln E[m, c] = ln (1/n_time) sum_t weights[t] T_c(w_m(t)) for tabulated columns.

    lq, ln_u           as for histogram
    lnT                (NV, n_col) ln T_c(V^2), e.g. glitch_columns() and merger_columns(s) side by side,
                       non-decreasing in v (every column of this module is), on the device of xp
    weights            None, (n_time,) a weight per arrival time, or (n_w, n_time) n_w sets of them, read in
                       one pass over the trajectories; their logarithms with log_weights, so that a weight below
                       1e-308 next to a large T (a glitch rings both) is kept.  A row of unit weights is summed
                       exactly as without weights
    own                None, or MergerTables (of the n_template templates, on the device of xp): n_own more
                       columns after the n_col of lnT, column k of template m the merger column of the signal power
                       own[k].s[m], with the same interpolation and the same sums (what ln_merger_at(lq, ln_u,
                       own[k].s, weights) returns, from the same pass over the trajectories)

    below vloud T is interpolated linearly in v, E = (H @ T - D @ Q) / n_time with the histogram H and the
    curvature term D @ Q of the interpolation; above vloud (the grid point DV (J - 2), J = round(vloud / DV) + 2,
    nearest to it, so that every arrival time is in exactly one of the two sets) ln T is interpolated linearly in
    v, one arrival time at a time, the sum taken in the log domain.  returns (n_template, n_col + n_own), or
    (n_w, n_template, n_col + n_own), float64 on the device.

    Accuracy of the part below vloud for steep columns (the amplitude bins of glitch_columns, which grow by
    exp(sqrt(2) rho DV) per cell of the grid, and any column that grows as fast): the curvature-corrected linear
    interpolation errs by about (sqrt(2) rho DV)^4 / 70 in relative terms (within a factor 1.5 of the measurements).
    ln E of a row whose times are all below vloud is too small by 1e-5 at the last bin up to rho = 12.2 (and by less
    than 3e-7 in ln of the fiducial sum over the bins, amplitude_weights()), by 1e-3 at rho ~ 35, 1e-2 at 60 and
    0.3 at 140 (measured against the quadrature of the columns at each time, tests/test_single.py).  The bins above
    rho ~ 30 weigh exp(-450) and less next to the others in every sum over the bins that is used (pi_k E_k, pi_k ~
    rho^-3 or steeper), so those sums are unaffected; only the per-bin outputs of these bins are not to be read.  A
    row with times above vloud has the same relative error on the share of its sum that comes from below it."""
    nm, nt = lq.shape
    J = int(round(vloud / DV)) + 2
    B, Q, top = _quiet_columns(lnT, J, xp)              # every column spans less than 600 nats below vloud
    nc = lnT.shape[1]
    if weights is None:
        lnW, one = None, True
    else:
        lnW, one = _ln_weight_rows(weights, nt, xp, log_weights)
    unit = [True] if lnW is None else [bool((lw == 0.0).all()) for lw in lnW]
    own = [] if own is None else list(own)
    na = nc + len(own)                                  # the columns of lnT, then one per table of own
    out = xp.empty((len(unit), nm, na))
    for a in range(0, nm, block):
        b = min(a + block, nm)
        v = _v(lq[a:b], ln_u[a:b], xp)
        j, f = _cic(v, xp)
        jm, fm = _cic(v.max(axis=1), xp)                # the loudest time of each template sets its scale
        shift = (1.0 - fm)[:, None] * lnT[jm] + fm[:, None] * lnT[jm + 1]
        r, t = xp.nonzero(v > DV * (J - 2))
        jl, fl = j[r, t], f[r, t]
        val = (1.0 - fl)[:, None] * lnT[jl] + fl[:, None] * lnT[jl + 1]
        qo = []
        if own:                                         # the same interpolation in each template's own column
            n_grid, rows = max(J + 1, int(jm.max()) + 2), xp.arange(b - a)
            Po = [tab.upto(n_grid)[a:b] for tab in own]
            shift = xp.concatenate([shift] + [((1.0 - fm) * P[rows, jm] + fm * P[rows, jm + 1])[:, None] for P in Po],
                                   axis=1)
            val = xp.concatenate([val] + [((1.0 - fl) * P[r, jl] + fl * P[r, jl + 1])[:, None] for P in Po], axis=1)
            qo = [_quiet_own(P, J, xp) for P in Po]

        def ln_quiet(H, D):
            """ln of the sum below vloud of every column: (b - a, na)"""
            with np.errstate(divide="ignore"):
                return xp.concatenate([xp.log(_dot(H, B) - _dot(D, Q)) + top[None, :]] + [
                    (xp.log((H * Bo).sum(axis=1) - (D * Qo).sum(axis=1)) + to)[:, None] for Bo, Qo, to in qo], axis=1)
        for i, u in enumerate(unit):
            if u:
                H, D = _quiet(v, j, f, xp.ones(nt), xp, J)
                loud = xp.bincount((r[:, None] * na + xp.arange(na)[None, :]).ravel(),
                                   weights=xp.exp(val - shift[r]).ravel(), minlength=(b - a) * na)
                with np.errstate(divide="ignore"):
                    out[i, a:b] = _logaddexp(ln_quiet(H, D), xp.log(loud.reshape(b - a, na)) + shift, xp) - np.log(nt)
                continue
            lw = lnW[i]
            top_w = float(lw.max())
            if top_w == -np.inf:
                out[i, a:b] = -np.inf
                continue
            H, D = _quiet(v, j, f, xp.exp(lw - top_w), xp, J)   # T is bounded below vloud: no term lost here
            loud = _ln_loud(val + lw[t][:, None], r, b - a, shift, lw[t], xp, na)
            out[i, a:b] = _logaddexp(ln_quiet(H, D) + top_w, loud, xp) - np.log(nt)
    return out[0] if one else out


def ln_merger_at(lq, ln_u, s, weights=None, xp=np, block=256, vloud=VLOUD, log_weights=False):
    """ln Lam^I(m) = ln (1/n_time) sum_t weights[t] Phi(w_m(t); s[m]) at a signal power per template.

    s (n_template,) on the host, the rest as for ln_evidence, whose interpolation it repeats with Phi computed at
    the grid points it needs.  returns (n_template,), or (n_w, n_template), float64 on the device."""
    nm, nt = lq.shape
    if weights is None:
        lnW, one = None, True
    else:
        lnW, one = _ln_weight_rows(weights, nt, xp, log_weights)
    unit = [True] if lnW is None else [bool((lw == 0.0).all()) for lw in lnW]
    s = np.asarray(s, float)
    J = int(round(vloud / DV)) + 2
    out = xp.empty((len(unit), nm))
    for a in range(0, nm, block):
        b = min(a + block, nm)
        sb = s[a:b]
        v = _v(lq[a:b], ln_u[a:b], xp)
        j, f = _cic(v, xp)
        jm, _ = _cic(v.max(axis=1), xp)
        H, D = _quiet(v, j, f, xp.ones(nt), xp, J)     # the grid points and cells the times reach
        rq, cq = xp.nonzero(H)
        rd, cd = xp.nonzero(D)
        rl, tl = xp.nonzero(v > DV * (J - 2))
        jl = j[rl, tl]
        # Phi once per (template, grid point) needed; it grows with v, so the template's last point sets its scale
        rows = xp.arange(b - a)
        need = [rq * NV + cq, rl * NV + jl, rl * NV + jl + 1, rows * NV + jm + 1] + [rd * NV + xp.abs(cd + d)
                                                                                    for d in (-1, 0, 1, 2)]
        key, inv = xp.unique(xp.concatenate(need), return_inverse=True)
        val = _ln_phi(DV * (key % NV), sb[_host(key // NV)], xp)[inv.ravel()]
        part = np.cumsum([0] + [x.size for x in need])
        lnq, lj, lj1, shift = (val[part[i]:part[i + 1]] for i in range(4))
        P = [xp.exp(val[part[i]:part[i + 1]] - shift[rd]) for i in range(4, 8)]           # Phi at j-1, j, j+1, j+2
        Q = xp.minimum(0.25 * (P[0] - P[1] - P[2] + P[3]), P[1] + P[2])
        lin = (1.0 - f[rl, tl]) * lj + f[rl, tl] * lj1
        eq = xp.exp(lnq - shift[rq])
        if not all(unit):                                # the weighted quiet part has its own scale
            qtop = _run_max(lnq[:, None], rq, b - a, xp)[:, 0]
            eqw = xp.exp(lnq - qtop[rq])
            Pw = [xp.exp(val[part[i]:part[i + 1]] - qtop[rd]) for i in range(4, 8)]
            Qw = xp.minimum(0.25 * (Pw[0] - Pw[1] - Pw[2] + Pw[3]), Pw[1] + Pw[2])
        for i, u in enumerate(unit):
            if u:
                tot = (xp.bincount(rq, weights=H[rq, cq] * eq, minlength=b - a)
                       - xp.bincount(rd, weights=D[rd, cd] * Q, minlength=b - a)
                       + xp.bincount(rl, weights=xp.exp(lin - shift[rl]), minlength=b - a))
                with np.errstate(divide="ignore"):
                    out[i, a:b] = shift + xp.log(tot) - np.log(nt)
                continue
            lw = lnW[i]
            top_w = float(lw.max())
            if top_w == -np.inf:
                out[i, a:b] = -np.inf
                continue
            Hw, Dw = _quiet(v, j, f, xp.exp(lw - top_w), xp, J)
            with np.errstate(divide="ignore"):
                quiet = qtop + top_w + xp.log(xp.bincount(rq, weights=Hw[rq, cq] * eqw, minlength=b - a)
                                              - xp.bincount(rd, weights=Dw[rd, cd] * Qw, minlength=b - a))
            loud = _ln_loud((lin + lw[tl])[:, None], rl, b - a, shift[:, None], lw[tl], xp, 1)[:, 0]
            out[i, a:b] = _logaddexp(quiet, loud, xp) - np.log(nt)
    return out[0] if one else out


def _host(a):
    return a if isinstance(a, np.ndarray) else a.get()


def glitch_time_series(lq, ln_u, pi_k, pi_m=None, xp=np, edges=RHO_EDGES, block=256, vloud=VLOUD):
    """ln l_g(t) = ln sum_m pi_m[m] sum_k pi_k[k] L_k(w_m(t)), the glitch evidence per arrival time.

    lq, ln_u   as for histogram
    pi_k       (n_rho,) weights of the amplitude bins of edges
    pi_m       (n_template,) weights of the templates, uniform by default

    the interpolation of ln_evidence (below vloud of L_pi = Lmat @ pi_k with its curvature term, above it of
    ln L_k), so that the mean over t of l_g is sum_{m,k} pi_m pi_k E[m, k] to rounding.  returns (n_time,) on the
    device."""
    nm, nt = lq.shape
    J = int(round(vloud / DV)) + 2
    lnT = glitch_columns(edges)
    with np.errstate(divide="ignore"):
        lpk = np.log(np.asarray(pi_k, float))
        lpm = np.log(np.full(nm, 1.0 / nm) if pi_m is None else np.asarray(pi_m, float))
    B, Q, top = _quiet_columns(lnT, J, np)
    A = (top + lpk).max()
    c = np.exp(top + lpk - A)
    P, Qp = xp.asarray(_dot(B, c)), xp.asarray(_dot(Q, c))  # L_pi and its curvature term below vloud, / e^A
    lnTd, lpkd = xp.asarray(lnT), xp.asarray(lpk)
    out = xp.full(nt, -np.inf)
    for a in range(0, nm, block):
        b = min(a + block, nm)
        v = _v(lq[a:b], ln_u[a:b], xp)
        j, f = _cic(v, xp)
        jq = xp.minimum(j, J - 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = A + xp.log((1.0 - f) * P[jq] + f * P[jq + 1] - f * (1.0 - f) * Qp[jq])
        r, t = xp.nonzero(v > DV * (J - 2))
        if r.size:
            x = (1.0 - f[r, t])[:, None] * lnTd[j[r, t]] + f[r, t][:, None] * lnTd[j[r, t] + 1] + lpkd[None, :]
            mx = x.max(axis=1)
            val[r, t] = mx + xp.log(xp.exp(x - mx[:, None]).sum(axis=1))
        val += xp.asarray(lpm[a:b])[:, None]
        mx = val.max(axis=0)
        mx = xp.where(mx > -np.inf, mx, 0.0)             # a block of templates of weight 0
        with np.errstate(divide="ignore"):
            out = _logaddexp(out, mx + xp.log(xp.exp(val - mx[None, :]).sum(axis=0)), xp)
    return out


def _logaddexp(a, b, xp):
    """ln(exp a + exp b), also for two -inf on the device (where a - b is nan)"""
    if xp is np:
        return np.logaddexp(a, b)
    m = xp.maximum(a, b)
    return xp.where(m == -np.inf, m, m + xp.log1p(xp.exp(-xp.abs(a - b))))


def _ln_arcs(x, L, xp):
    """ln sum_{i < L} exp x[(a + i) mod n] for every start a (n,), 0 <= L < n: the sums over 2^k consecutive
    elements built by doubling and combined over the bits of L, every step a logaddexp, so no term is lost to the
    dynamic range"""
    out = xp.full(x.shape, -np.inf)
    S, size, off = x, 1, 0
    while size <= L:
        if L & size:
            out = _logaddexp(out, xp.roll(S, -off), xp)
            off += size
        S = _logaddexp(S, xp.roll(S, -size), xp)
        size *= 2
    return out


def glitch_time_weights(ln_lg, half_window, xp=np):
    """the glitch-time weights kbar(t') = sum_{|t - t'| > half_window} l_g(t) / sum_t l_g(t), the probability that
    the glitch is away from t' (circular in t), and 1 - kbar(t'), the share of the window.  Both sums are taken
    directly in the log domain (_ln_arcs), never as a difference, so each logarithm stays right when its value
    underflows: next to a loud glitch ln kbar ~ -ln l_g, far from it ln(1 - kbar) ~ -ln l_g.

    ln_lg        (n_time,) glitch_time_series(...)
    half_window  in samples; < 0: an empty window, kbar = 1

    Where the glitch evidence is zero at every time (ln_lg = -inf throughout: no glitch can have produced the
    data) the weights are those of a flat l_g, kbar = 1 - (2 half_window + 1) / n_time: no time is preferred, and
    the glitch hypothesis then carries no weight where they are used.

    returns ln kbar and ln(1 - kbar), each (n_time,) and <= 0."""
    n = ln_lg.size
    hw = int(half_window)
    if hw < 0:
        return xp.zeros(n), xp.full(n, -np.inf)
    if 2 * hw + 1 >= n:
        return xp.full(n, -np.inf), xp.zeros(n)
    ln_lg = xp.where(ln_lg.max() > -np.inf, ln_lg, 0.0)                         # zero evidence everywhere: flat
    m = ln_lg.max()
    total = m + xp.log(xp.exp(ln_lg - m).sum())
    away = xp.roll(_ln_arcs(ln_lg, n - 2 * hw - 1, xp), -(hw + 1)) - total       # from t' + hw + 1 round to t' - hw - 1
    near = xp.roll(_ln_arcs(ln_lg, 2 * hw + 1, xp), hw) - total                   # from t' - hw to t' + hw
    return xp.minimum(away, 0.0), xp.minimum(near, 0.0)


def ln_near_fraction(ln_near, lags, xp=np):
    """ln o(t), o(t) = <1 - kbar[(t + tau) mod n_time]>_tau the mean with equal weights over the lags tau (samples)
    of the probability that the glitch is near the time t + tau, from ln(1 - kbar) (n_time,) of
    glitch_time_weights, in the log domain: ln o_H(t) of the L arrival time t is ln_near_fraction(ln_near_H, lags),
    ln o_L(t') of the H arrival time t' ln_near_fraction(ln_near_L, -lags).  returns (n_time,)"""
    acc = xp.full(ln_near.shape, -np.inf)
    for tau in np.asarray(lags, int).ravel():
        acc = _logaddexp(acc, xp.roll(ln_near, -int(tau)), xp)
    return acc - np.log(np.asarray(lags).size)


def amplitude_weights(edges=RHO_EDGES, gamma=4.0, rho_min=4.0):
    """pi_k (n_rho,): the probability of the amplitude bin [edges[k], edges[k+1]] for dN/drho ~ rho^-gamma above
    rho_min and zero below, normalized over the bins; the default is the fiducial law of the glitch-time weights"""
    e = np.asarray(edges, float)
    a, b = np.maximum(e[:-1], rho_min), e[1:]
    ok = b > a
    a, b = np.where(ok, a, 1.0), np.where(ok, b, 1.0)
    w = np.log(b / a) if gamma == 1.0 else (a ** (1.0 - gamma) - b ** (1.0 - gamma)) / (gamma - 1.0)
    w = np.where(ok, w, 0.0)
    return w / w.sum()


def _ln_sum_exp(x, axis, xp):
    """ln sum exp x over axis, -inf where every term is -inf"""
    m = x.max(axis=axis, keepdims=True)
    m = xp.where(m > -np.inf, m, 0.0)
    with np.errstate(divide="ignore"):
        return xp.squeeze(m, axis=axis) + xp.log(xp.exp(x - m).sum(axis=axis))


def window_evidence(lq, ln_u, lnT, n_window, pi_m=None, xp=np, block=256):
    """ln E_w[k] = ln (1/n_win) sum_{t in w} sum_m pi_m[m] T_k(w_m(t)), the evidence of each window of the segment
    (docs/glitch_likelihood.md, Sec. 4a), the windows the n_window consecutive blocks of n_win = n_time / n_window
    arrival times.

    lq, ln_u   as for histogram
    lnT        (NV, n_col) as for ln_evidence, e.g. glitch_columns()
    n_window   a divisor of n_time
    pi_m       (n_template,) weights of the templates, uniform by default

    ln_evidence of the rows (template, window) of n_win times each, block templates at a time, then the mixture over
    the templates in the log domain: the mean over the windows of E_w is sum_m pi_m E[m] of the whole segment.
    returns (n_window, n_col), float64 on the device."""
    nm, nt = lq.shape
    if n_window < 1 or nt % n_window:
        raise ValueError("n_time = %d is not a multiple of n_window = %d" % (nt, n_window))
    rows = lq.reshape(nm * n_window, nt // n_window)
    lu = xp.repeat(xp.asarray(ln_u, xp.float64), n_window)
    lnE = ln_evidence(rows, lu, lnT, xp=xp, block=block * n_window).reshape(nm, n_window, -1)
    with np.errstate(divide="ignore"):
        lpm = np.log(np.full(nm, 1.0 / nm) if pi_m is None else np.asarray(pi_m, float))
    return _ln_sum_exp(lnE + xp.asarray(lpm)[:, None, None], 0, xp)


def window_merger(lq, ln_u, lnT, n_window, ln_W, xp=np, block=256):
    """ln M_w = ln sum_m sum_k W[m, k] (1/n_win) sum_{t in w} T_k(w_m(t)), the merger evidence of each window of the
    segment (docs/2026-11-15_mergers_in_stage1.md, Sec. 4): the windows those of window_evidence, T_k the merger
    columns Phi(V^2; s_k) of a ladder of signal powers (merger_columns) and W[m, k] the weight of the population on
    template m and node k (population.merger_weights).

    lq, ln_u   as for histogram, the templates of the rows of ln_W
    lnT        (NV, n_col) ln T_k(V^2), on the device of xp
    n_window   a divisor of n_time
    ln_W       (n_template, n_col) ln W[m, k], -inf where W is 0, on the device of xp

    ln_evidence of the rows (template, window) of n_win times each, as in window_evidence, then the sum over
    (m, k) in the log domain.  returns (n_window,), float64 on the device."""
    nm, nt = lq.shape
    if n_window < 1 or nt % n_window:
        raise ValueError("n_time = %d is not a multiple of n_window = %d" % (nt, n_window))
    if ln_W.shape != (nm, lnT.shape[1]):
        raise ValueError("ln_W has shape %s, not (n_template, n_col) = %s" % (ln_W.shape, (nm, lnT.shape[1])))
    rows = lq.reshape(nm * n_window, nt // n_window)
    lu = xp.repeat(xp.asarray(ln_u, xp.float64), n_window)
    lnE = ln_evidence(rows, lu, lnT, xp=xp, block=block * n_window).reshape(nm, n_window, -1)
    x = (lnE + ln_W[:, None, :]).transpose(1, 0, 2).reshape(n_window, -1)
    return _ln_sum_exp(x, -1, xp)


def _ln_mixed(lnE, pi, xp, tiny=1e-280):
    """ln sum_k pi[s, k] exp lnE[..., k] for every row s of pi: (..., K).  The terms are scaled by the largest of
    their row; where the law of s has (almost) no weight on the large ones the row is summed again with its own
    scale"""
    mx = lnE.max(axis=-1, keepdims=True)
    mx = xp.where(mx > -np.inf, mx, 0.0)
    y = _dot(xp.exp(lnE - mx), xp.asarray(pi.T))
    with np.errstate(divide="ignore"):
        out = mx + xp.log(y)
        low = xp.nonzero(y < tiny)
        if low[0].size:
            out[low] = _ln_sum_exp(lnE[low[:-1]] + xp.asarray(np.log(pi))[low[-1]], -1, xp)
    return out


def _ln_one_minus_exp(a, xp):
    """ln(1 - exp(-a)) for a >= 0, accurate at both ends"""
    with np.errstate(divide="ignore"):
        return xp.where(a < np.log(2.0), xp.log(-xp.expm1(-a)), xp.log1p(-xp.exp(-a)))


def window_posterior(ln_E, f, q, pi, zone=1, xp=np):
    """the noise ratio, the posterior of the state and the glitch-time weights of the windows of one detector
    (docs/glitch_likelihood.md, Sec. 4a), in the log domain.

    ln_E   (..., n_w, n_k) ln E_w[k] (window_evidence), any leading axes (segments), on the device of xp
    f, q   (K,) probability of each state, and of a glitch in a window given the state
    pi     (K, n_k) the amplitude law of each state over the columns
    zone   the zone of a window is the window and zone neighbours on each side, circular

        L_w^s = sum_k pi[s, k] E_w[k],   D_w^s = 1 - q_s + q_s L_w^s,   c = sum_s f_s prod_w D_w^s,
        r_s = f_s prod_w D_w^s / c,   kappa_w^s = prod_{w' in zone(w)} (1 - q_s) / D_w'^s,   kbar_w = sum_s r_s kappa_w^s

    ln D = ln(1 - q) + ln(1 + a), a = q L / (1 - q), and ln kappa = -sum_zone ln(1 + a); 1 - kbar = sum_s r_s
    (1 - kappa^s) with 1 - kappa = -expm1(ln kappa), so ln kbar next to a loud glitch and ln(1 - kbar) ~
    ln((2 zone + 1) q) in quiet data keep their relative accuracy.  A state with q_s = 0 (no glitches) or f_s = 0 is
    allowed.

    returns dict: ln_c (...,), r (..., K), ln_kbar (..., n_w), ln_near (..., n_w) = ln(1 - kbar)"""
    lnE = xp.asarray(ln_E, xp.float64)
    f, q, pi = (np.asarray(a, float) for a in (f, q, pi))
    nw = lnE.shape[-2]
    if zone < 0 or 2 * zone + 1 > nw:
        raise ValueError("a zone of %d windows on each side does not fit in %d windows" % (zone, nw))
    lnL = _ln_mixed(lnE, pi, xp)                                         # (..., n_w, K)
    with np.errstate(divide="ignore"):
        lq, l1q, lf = (xp.asarray(v) for v in (np.log(q), np.log1p(-q), np.log(f)))
    lnD = xp.logaddexp(l1q, lq + lnL)
    A1 = xp.logaddexp(0.0, lq - l1q + lnL)                                # ln(1 + a) = ln D - ln(1 - q)
    x = lnD.sum(axis=-2) + lf                                             # ln f_s C^s
    ln_c = _ln_sum_exp(x, -1, xp)
    ln_r = x - ln_c[..., None]
    A = sum(xp.roll(A1, d, axis=-2) for d in range(-zone, zone + 1))     # -ln kappa
    ln_kbar = _ln_sum_exp(ln_r[..., None, :] - A, -1, xp)
    ln_near = _ln_sum_exp(ln_r[..., None, :] + _ln_one_minus_exp(A, xp), -1, xp)
    return dict(ln_c=ln_c, r=xp.exp(ln_r), ln_kbar=ln_kbar, ln_near=ln_near)
