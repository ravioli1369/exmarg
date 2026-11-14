"""tests of the single-detector evidences, exmarg.search.single (docs/glitch_likelihood.md, Secs. 2, 3, 5 and
controls 1 to 3 of Sec. 7), on the CPU and without tables.  The expected values come from analytic results, from
scipy quadrature over the amplitude bins, from a Monte Carlo over sky, polarization and inclination written here
with detectors.Fplus_Fcross, or from the construction of the test data; the kernel is rewritten here with
scipy.special.ive."""

from pathlib import Path

import numpy as np
from scipy.integrate import quad, simpson
from scipy.special import ive, logsumexp

from exmarg.detectors import Fplus_Fcross
from exmarg.search import single

S_NODES = np.load(Path(__file__).resolve().parents[1] / "data" / "tables" / "s_grid.npz")["s_nodes"]


def ln_l(w, x):
    """ln [exp(-x) I0(2 sqrt(x w))]"""
    z = 2.0 * np.sqrt(x * w)
    return z - x + np.log(ive(0, z))


def trajectory(rng, nm, nt, loud=()):
    """lq (float32), ln_u and the w they give, w ~ Exp(1) with the (template, time, w) of loud set by hand"""
    w = rng.exponential(size=(nm, nt))
    for m, t, x in loud:
        w[m, t] = x
    ln_u = rng.normal(0.0, 1.0, nm)
    lq = (0.5 * (np.log(w) + ln_u[:, None])).astype(np.float32)
    return lq, ln_u, np.exp(2.0 * lq.astype(float) - ln_u[:, None])


def test_unit_mean():
    """control 1: l(w; x), every L_k and every Phi(w; s) at the store nodes average to 1 over w ~ Exp(1)"""
    dev = []
    for x in (0.0, 0.01, 0.5, 3.0, 20.0, 200.0, 2000.0, 20000.0):
        r = np.sqrt(x)
        f = lambda v: 2.0 * v * np.exp(single.ln_kernel(v * v, x) - v * v)
        dev.append(quad(f, 0.0, r + 40.0, points=[r] if r > 0 else None, limit=500, epsabs=1e-14, epsrel=1e-13)[0] - 1)
    print("unit mean of l: max |mean - 1| = %.1e over x from 0 to 2e4" % np.abs(dev).max())
    assert np.abs(dev).max() < 1e-6
    v = single.V
    for name, T in (("L_k", single.glitch_columns()), ("Phi", single.merger_columns(S_NODES))):
        mean = simpson(2.0 * v[:, None] * np.exp(T - v[:, None] ** 2), x=v, axis=0)
        print("unit mean of %s: max |mean - 1| = %.1e over %d columns" % (name, np.abs(mean - 1).max(), mean.size))
        assert np.abs(mean - 1.0).max() < 1e-6, name


def test_response_moments():
    """the distribution of |R|: normalized, <|R|^2> = 4/25 (<F+^2> = <Fx^2> = 1/5, <chi^2> = 1/3, <chi^4> = 1/5)
    and <|R|^4> = (8/35)^2 (F^2 and P have <.^2> = 8/35: 3/8 <A+^4> + 3/8 <chi^4> + 1/4 <A+^2 chi^2>)"""
    q, w = single.response_distribution()
    assert (w > 0).all() and q.min() >= 0.0 and q.max() <= 1.0
    m0, m2, m4 = w.sum(), (w * q ** 2).sum(), (w * q ** 4).sum()
    print("|R|: sum - 1 = %.1e, <|R|^2> - 4/25 = %.1e, <|R|^4> - (8/35)^2 = %.1e"
          % (m0 - 1, m2 - 0.16, m4 - (8 / 35) ** 2))
    assert abs(m0 - 1.0) < 1e-12
    assert abs(m2 - 4.0 / 25.0) < 1e-12
    assert abs(m4 - (8.0 / 35.0) ** 2) < 1e-12


def test_brute_force():
    """control 3: E of the amplitude bins against scipy.quad over each bin, and Lambda_S^L at signal powers s
    against a Monte Carlo over sky, polarization and inclination of L, both summed over the arrival times"""
    rng = np.random.default_rng(7)
    lq, ln_u, w = trajectory(rng, 2, 24, loud=[(0, 5, 6.0), (1, 9, 15.0)])
    nm, nt = w.shape
    edges = single.RHO_EDGES
    got = single.ln_evidence(lq, ln_u, single.glitch_columns())
    ref = np.empty(got.shape)
    for m in range(nm):
        for k in range(edges.size - 1):
            a, b = edges[k], edges[k + 1]
            rg = np.linspace(a, b, 401)
            terms = []
            for wt in w[m]:
                g = ln_l(wt, 0.5 * rg * rg)
                c, pk = g.max(), rg[np.argmax(g)]
                val = quad(lambda r: np.exp(ln_l(wt, 0.5 * r * r) - c), a, b, points=[pk], limit=200, epsabs=0.0,
                           epsrel=1e-12)[0]
                terms.append(c + np.log(val / (b - a)))
            ref[m, k] = logsumexp(terms) - np.log(nt)
    err = np.abs(got - ref).max()
    print("E of the amplitude bins against quad: max |d ln E| = %.1e" % err)
    assert err < 1e-5

    n = 400000
    th, ph = np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)
    psi, ci = rng.uniform(0, np.pi, n), rng.uniform(-1, 1, n)
    Fp, Fx = Fplus_Fcross("L", th, ph, psi)
    r2 = (0.5 * (1 + ci ** 2) * Fp) ** 2 + (ci * Fx) ** 2

    def mc(m, s):
        g = np.exp(ln_l(w[m][None, :], s * r2[:, None])).mean(axis=1)       # per sky point, mean over times
        return g.mean(), g.std() / np.sqrt(n)

    for s in (0.3, 3.0, 30.0):
        got = np.exp(single.ln_evidence(lq, ln_u, single.merger_columns([s]))[:, 0])
        for m in range(nm):
            mean, se = mc(m, s)
            print("Lambda_S^L s = %g template %d: %.6f, Monte Carlo %.6f +- %.6f" % (s, m, got[m], mean, se))
            assert abs(got[m] - mean) < 4.0 * se + 1e-4 * mean
    s = np.array([0.5, 7.0])
    got = np.exp(single.ln_merger_at(lq, ln_u, s))
    for m in range(nm):
        mean, se = mc(m, s[m])
        print("Lambda_S^L at s = %g per template %d: %.6f, Monte Carlo %.6f +- %.6f" % (s[m], m, got[m], mean, se))
        assert abs(got[m] - mean) < 4.0 * se + 1e-4 * mean


def test_interpolation():
    """the histogram route against the exact sum over the arrival times, with a loud time (w = 2000), times
    beyond the grid (w = 3e4, 1e6: clamped to v = 150) and a moderately loud one (w = 9), for columns l(w; x)
    built here, x from 0.01 to 20000; and the histogram itself, with and without weights per time, against
    np.interp of the columns at every time"""
    rng = np.random.default_rng(11)
    lq, ln_u, w = trajectory(rng, 4, 32768, loud=[(1, 100, 2000.0), (2, 7, 3.0e4), (2, 8, 1.0e6), (3, 50, 9.0)])
    nm, nt = w.shape
    xs = np.array([0.01, 0.5, 4.0, 25.0, 200.0, 2000.0, 20000.0])
    got = single.ln_evidence(lq, ln_u, ln_l(single.V[:, None] ** 2, xs[None, :]))
    v = np.minimum(np.sqrt(w), single.V[-1])
    ref = logsumexp(ln_l(v[:, :, None] ** 2, xs[None, None, :]), axis=1) - np.log(nt)
    err = np.abs(got - ref)
    print("ln E against the exact sum: max error per column", np.array2string(err.max(axis=0), precision=1))
    assert err.max() < 2e-5

    wt = rng.uniform(0.0, 1.0, nt)
    C = np.stack([np.cos(single.V), np.exp(-single.V), single.V ** 2 / (1 + single.V ** 2)], axis=1)
    for weights in (None, wt):
        H, wmax = single.histogram(lq, ln_u, weights)
        f = np.ones(nt) if weights is None else weights
        ref = np.stack([[(f * np.interp(v[m], single.V, c)).sum() for c in C.T] for m in range(nm)])
        assert np.allclose(np.einsum("mj,jc->mc", H, C), ref, rtol=1e-12, atol=1e-9)
        assert np.allclose(H.sum(axis=1), f.sum(), rtol=1e-12)
        assert np.allclose(wmax, w.max(axis=1), rtol=1e-12)


def test_weighted_evidence_and_glitch_time_series():
    """ln_evidence with weights per time against the weighted exact sum, and the mean over the times of the glitch
    time series against sum_{m,k} pi_m pi_k E[m, k] of the histogram route"""
    rng = np.random.default_rng(5)
    lq, ln_u, w = trajectory(rng, 3, 4096, loud=[(0, 10, 40.0), (2, 3000, 300.0)])
    nm, nt = w.shape
    wt = rng.uniform(0.0, 1.0, nt)
    xs = np.array([0.5, 4.0, 50.0])
    got = single.ln_evidence(lq, ln_u, ln_l(single.V[:, None] ** 2, xs[None, :]), weights=wt)
    ref = logsumexp(ln_l(w[:, :, None], xs[None, None, :]), b=wt[None, :, None], axis=1) - np.log(nt)
    print("weighted ln E against the weighted exact sum: max error %.1e" % np.abs(got - ref).max())
    assert np.abs(got - ref).max() < 2e-5

    pk = rng.uniform(0.0, 1.0, single.RHO_EDGES.size - 1)
    pk /= pk.sum()
    pm = np.array([0.5, 0.2, 0.3])
    lg = single.glitch_time_series(lq, ln_u, pk, pm)
    lnE = single.ln_evidence(lq, ln_u, single.glitch_columns())
    lhs = logsumexp(lg) - np.log(nt)
    rhs = logsumexp(lnE + np.log(pk)[None, :] + np.log(pm)[:, None])
    print("mean of l_g against sum pi_m pi_k E: %.1e" % (lhs - rhs))
    assert abs(lhs - rhs) < 1e-12
    lg0 = single.glitch_time_series(lq, ln_u, pk)
    assert abs(logsumexp(lg0) - np.log(nt) - logsumexp(lnE + np.log(pk)[None, :] - np.log(nm))) < 1e-12


def test_glitch_time_weights():
    """kbar by hand: one spike (0 inside the window, circularly, 1 outside), flat (1 - (2 W + 1) / n), and a
    random l_g against the window sums written out"""
    n, hw = 64, 5
    ln_lg = np.full(n, -1000.0)
    ln_lg[2] = 0.0
    k = single.glitch_time_weights(ln_lg, hw)
    inside = np.zeros(n, bool)
    inside[[(2 + d) % n for d in range(-hw, hw + 1)]] = True
    assert np.array_equal(k[inside], np.zeros(inside.sum())) and np.array_equal(k[~inside], np.ones((~inside).sum()))
    assert np.allclose(single.glitch_time_weights(np.zeros(n), hw), 1.0 - (2 * hw + 1) / n, rtol=0, atol=1e-14)
    rng = np.random.default_rng(2)
    ln_lg = rng.normal(0.0, 3.0, n)
    p = np.exp(ln_lg)
    ref = np.array([1.0 - sum(p[(t + d) % n] for d in range(-hw, hw + 1)) / p.sum() for t in range(n)])
    assert np.allclose(single.glitch_time_weights(ln_lg, hw), ref, rtol=0, atol=1e-13)
    assert np.array_equal(single.glitch_time_weights(ln_lg, n // 2), np.zeros(n))
