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


def _weight_rows(weights, nt, xp):
    """weights as rows (n_w, n_time), and whether they were given as one row"""
    W = _weights(weights, nt, xp)
    return (W[None, :], True) if W.ndim == 1 else (W, False)


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


def ln_evidence(lq, ln_u, lnT, weights=None, xp=np, block=256, vloud=VLOUD):
    """ln E[m, c] = ln (1/n_time) sum_t weights[t] T_c(w_m(t)) for tabulated columns.

    lq, ln_u           as for histogram
    lnT                (NV, n_col) ln T_c(V^2), e.g. glitch_columns() and merger_columns(s) side by side,
                       non-decreasing in v (every column of this module is), on the device of xp
    weights            None, (n_time,) a weight per arrival time, or (n_w, n_time) n_w sets of them, read in
                       one pass over the trajectories

    below vloud T is interpolated linearly in v, E = (H @ T - D @ Q) / n_time with the histogram H and the
    curvature term D @ Q of the interpolation; above vloud ln T is interpolated linearly in v, one arrival time
    at a time.  returns (n_template, n_col), or (n_w, n_template, n_col), float64 on the device."""
    nm, nt = lq.shape
    W, one = _weight_rows(weights, nt, xp)
    J = int(round(vloud / DV)) + 2
    B, Q, top = _quiet_columns(lnT, J, xp)              # every column spans less than 600 nats below vloud
    nc = lnT.shape[1]
    out = xp.empty((W.shape[0], nm, nc))
    for a in range(0, nm, block):
        b = min(a + block, nm)
        v = _v(lq[a:b], ln_u[a:b], xp)
        j, f = _cic(v, xp)
        jm, fm = _cic(v.max(axis=1), xp)                # the loudest time of each template sets its scale
        shift = (1.0 - fm)[:, None] * lnT[jm] + fm[:, None] * lnT[jm + 1]
        r, t = xp.nonzero(v > vloud)
        val = (1.0 - f[r, t])[:, None] * lnT[j[r, t]] + f[r, t][:, None] * lnT[j[r, t] + 1]
        X, idx = xp.exp(val - shift[r]), (r[:, None] * nc + xp.arange(nc)[None, :]).ravel()
        for i, wt in enumerate(W):
            H, D = _quiet(v, j, f, wt, xp, J)
            loud = xp.bincount(idx, weights=(wt[t][:, None] * X).ravel(), minlength=(b - a) * nc)
            with np.errstate(divide="ignore"):
                quiet = xp.log(_dot(H, B) - _dot(D, Q)) + top[None, :]
                out[i, a:b] = xp.logaddexp(quiet, xp.log(loud.reshape(b - a, nc)) + shift) - np.log(nt)
    return out[0] if one else out


def ln_merger_at(lq, ln_u, s, weights=None, xp=np, block=256, vloud=VLOUD):
    """ln Lam^I(m) = ln (1/n_time) sum_t weights[t] Phi(w_m(t); s[m]) at a signal power per template.

    s (n_template,) on the host, the rest as for ln_evidence, whose interpolation it repeats with Phi computed at
    the grid points it needs.  returns (n_template,), or (n_w, n_template), float64 on the device."""
    nm, nt = lq.shape
    W, one = _weight_rows(weights, nt, xp)
    s = np.asarray(s, float)
    J = int(round(vloud / DV)) + 2
    out = xp.empty((W.shape[0], nm))
    for a in range(0, nm, block):
        b = min(a + block, nm)
        sb = s[a:b]
        v = _v(lq[a:b], ln_u[a:b], xp)
        j, f = _cic(v, xp)
        jm, _ = _cic(v.max(axis=1), xp)
        H, D = _quiet(v, j, f, xp.ones(nt), xp, J)     # the grid points and cells the times reach
        rq, cq = xp.nonzero(H)
        rd, cd = xp.nonzero(D)
        rl, tl = xp.nonzero(v > vloud)
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
        eq, el = xp.exp(lnq - shift[rq]), xp.exp(lin - shift[rl])
        for i, wt in enumerate(W):
            H, D = _quiet(v, j, f, wt, xp, J)
            tot = (xp.bincount(rq, weights=H[rq, cq] * eq, minlength=b - a)
                   - xp.bincount(rd, weights=D[rd, cd] * Q, minlength=b - a)
                   + xp.bincount(rl, weights=wt[tl] * el, minlength=b - a))
            with np.errstate(divide="ignore"):
                out[i, a:b] = shift + xp.log(tot) - np.log(nt)
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
        r, t = xp.nonzero(v > vloud)
        if r.size:
            x = (1.0 - f[r, t])[:, None] * lnTd[j[r, t]] + f[r, t][:, None] * lnTd[j[r, t] + 1] + lpkd[None, :]
            mx = x.max(axis=1)
            val[r, t] = mx + xp.log(xp.exp(x - mx[:, None]).sum(axis=1))
        val += xp.asarray(lpm[a:b])[:, None]
        mx = val.max(axis=0)
        out = xp.logaddexp(out, mx + xp.log(xp.exp(val - mx[None, :]).sum(axis=0)))
    return out


def glitch_time_weights(ln_lg, half_window, xp=np):
    """kbar(t') = 1 - sum_{|t - t'| <= half_window} l_g(t) / sum_t l_g(t), the probability that the glitch is
    away from t', circular in t, clipped to [0, 1].  ln_lg (n_time,) = glitch_time_series(...), half_window in
    samples (< 0: an empty window, kbar = 1).  returns (n_time,)"""
    n = ln_lg.size
    hw = int(half_window)
    if hw < 0:
        return xp.ones(n)
    if 2 * hw + 1 >= n:
        return xp.zeros(n)
    p = xp.exp(ln_lg - ln_lg.max())
    c = xp.concatenate([xp.zeros(1), xp.cumsum(xp.concatenate([p[n - hw:], p, p[:hw]]))])
    return xp.clip(1.0 - (c[2 * hw + 1:] - c[:n]) / p.sum(), 0.0, 1.0)


def near_fraction(kbar, lags, xp=np):
    """o(t) = <1 - kbar[(t + tau) mod n_time]>_tau, the mean with equal weights over the lags tau (samples) of the
    probability that the glitch with the time weights kbar (n_time,) is near the time t + tau: o_H(t) of the L
    arrival time t is near_fraction(kbar_H, lags), o_L(t') of the H arrival time t' near_fraction(kbar_L, -lags).
    returns (n_time,)"""
    near = 1.0 - kbar
    acc = xp.zeros(near.shape)
    for tau in np.asarray(lags, int).ravel():
        acc += xp.roll(near, -int(tau))
    return acc / np.asarray(lags).size


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
