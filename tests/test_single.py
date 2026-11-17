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
    """kbar by hand: one spike (0 inside the window, circularly, 1 outside), flat (1 - (2 W + 1) / n), a random l_g
    against the window sums written out, an empty window (kbar = 1) and one covering everything (kbar = 0)"""
    n, hw = 64, 5
    ln_lg = np.full(n, -1000.0)
    ln_lg[2] = 0.0
    ln_k, ln_n = single.glitch_time_weights(ln_lg, hw)
    k = np.exp(ln_k)
    inside = np.zeros(n, bool)
    inside[[(2 + d) % n for d in range(-hw, hw + 1)]] = True
    assert np.array_equal(k[inside], np.zeros(inside.sum())) and np.array_equal(k[~inside], np.ones((~inside).sum()))
    assert np.allclose(np.exp(ln_n), 1.0 - k, rtol=0, atol=1e-14)
    ln_k, ln_n = single.glitch_time_weights(np.zeros(n), hw)
    assert np.allclose(np.exp(ln_k), 1.0 - (2 * hw + 1) / n, rtol=0, atol=1e-14)
    assert np.allclose(np.exp(ln_n), (2 * hw + 1) / n, rtol=1e-14, atol=0)
    rng = np.random.default_rng(2)
    ln_lg = rng.normal(0.0, 3.0, n)
    p = np.exp(ln_lg)
    ref = np.array([1.0 - sum(p[(t + d) % n] for d in range(-hw, hw + 1)) / p.sum() for t in range(n)])
    ln_k, ln_n = single.glitch_time_weights(ln_lg, hw)
    assert np.allclose(np.exp(ln_k), ref, rtol=0, atol=1e-13)
    assert np.allclose(np.exp(ln_n), 1.0 - ref, rtol=0, atol=1e-13)
    ln_k, ln_n = single.glitch_time_weights(ln_lg, -1)
    assert np.array_equal(ln_k, np.zeros(n)) and np.all(ln_n == -np.inf)
    ln_k, ln_n = single.glitch_time_weights(ln_lg, n // 2)
    assert np.all(ln_k == -np.inf) and np.array_equal(ln_n, np.zeros(n))


def test_glitch_time_weights_log_domain():
    """ln kbar and ln(1 - kbar) next to and away from loud glitches, where one of them underflows: against the sums
    inside and outside each window written here with scipy logsumexp (stable on its own, the largest term of each
    sum taken out) and against the closed forms for a spike of ln l_g = 3000 on a flat floor of 0, with
    ln sum_t l_g = ln(e^3000 + n - 1): inside its window ln kbar = ln(n - 2W - 1) - ln sum_t l_g and ln(1 - kbar) =
    0 to rounding, outside it ln kbar = 0 to rounding and ln(1 - kbar) = ln(2W + 1) - ln sum_t l_g; then two spikes of
    very different height and spikes at the wrap"""
    n, W = 2048, 40

    def ref(ln_lg):
        d = np.arange(n)[:, None]
        away = (d + W + 1 + np.arange(n - 2 * W - 1)[None, :]) % n
        near = (d - W + np.arange(2 * W + 1)[None, :]) % n
        return logsumexp(ln_lg[away], axis=1) - logsumexp(ln_lg), logsumexp(ln_lg[near], axis=1) - logsumexp(ln_lg)

    def inside(t0):
        d = np.abs(np.arange(n) - t0)
        return np.minimum(d, n - d) <= W

    total = np.logaddexp(3000.0, np.log(n - 1))
    for t0 in (700, 3, n - 2):                                  # in the middle, and at the wrap on either side
        ln_lg = np.zeros(n)
        ln_lg[t0] = 3000.0
        ln_k, ln_n = single.glitch_time_weights(ln_lg, W)
        i = inside(t0)
        err_k = np.abs(ln_k[i] / (np.log(n - 2 * W - 1) - total) - 1).max()
        err_n = np.abs(ln_n[~i] / (np.log(2 * W + 1) - total) - 1).max()
        print("spike 3000 at %d: inside ln kbar %.6f (relative error %.1e), outside ln(1 - kbar) %.6f (%.1e); "
              "max |ln kbar| outside %.1e, max |ln(1 - kbar)| inside %.1e"
              % (t0, ln_k[i][0], err_k, ln_n[~i][0], err_n, np.abs(ln_k[~i]).max(), np.abs(ln_n[i]).max()))
        assert i.sum() == 2 * W + 1 and err_k < 1e-9 and err_n < 1e-9
        assert np.abs(ln_k[~i]).max() < 1e-12 and np.abs(ln_n[i]).max() < 1e-12
        assert np.all(np.exp(ln_k[i]) == 0.0) and np.all(np.exp(ln_n[~i]) == 0.0)   # what a linear kbar gives
        for got, want in zip((ln_k, ln_n), ref(ln_lg)):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    rng = np.random.default_rng(9)
    for spikes in (((300, 3000.0), (1500, 400.0)), ((5, 2500.0), (n - 300, 900.0), (1000, 60.0))):
        ln_lg = rng.normal(0.0, 1.0, n)
        for t0, h in spikes:
            ln_lg[t0] = h
        ln_k, ln_n = single.glitch_time_weights(ln_lg, W)
        for got, want in zip((ln_k, ln_n), ref(ln_lg)):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
        (t1, h1), (t2, h2) = spikes[:2]
        assert abs(ln_k[t1] - (h2 - h1)) < 1.0 and abs(ln_k[t2]) < 1e-12    # the loudest sets the rest
        assert abs(ln_n[t2] - (h2 - h1)) < 1.0


def test_log_weights():
    """ln_evidence and ln_merger_at with weights given as logarithms, against the exact sums over the times in the
    log domain: a time with w = 2000 carries the weight exp(-1995) (a loud glitch: kbar tiny where T is huge), so
    the answer is set by a product that linear weights would round to 0; and a second row whose weights span less
    than 600 nats (the shift by the loudest time)"""
    rng = np.random.default_rng(13)
    lq, ln_u, w = trajectory(rng, 3, 4096, loud=[(1, 100, 2000.0), (2, 50, 30.0)])
    nm, nt = w.shape
    lw = np.stack([np.log(rng.uniform(0.01, 1.0, nt)), np.log(rng.uniform(0.01, 1.0, nt))])
    lw[0, 100] = -1995.0
    xs = np.array([0.5, 25.0, 2000.0])
    got = single.ln_evidence(lq, ln_u, ln_l(single.V[:, None] ** 2, xs[None, :]), lw, log_weights=True)
    ref = logsumexp(ln_l(w[None, :, :, None], xs) + lw[:, None, :, None], axis=2) - np.log(nt)
    print("ln_evidence with log weights: max error %.1e; template 1, x = 2000: %.3f (linear weights: %.3f)"
          % (np.abs(got - ref).max(), got[0, 1, 2],
             single.ln_evidence(lq, ln_u, ln_l(single.V[:, None] ** 2, xs[None, :]), np.exp(lw[0]))[1, 2]))
    assert np.abs(got - ref).max() < 2e-5 and got[0, 1, 2] > -10.0
    s = np.array([0.3, 3000.0, 30.0])
    got = single.ln_merger_at(lq, ln_u, s, lw, log_weights=True)
    phi = np.stack([single._ln_phi(np.sqrt(w[m]), np.full(nt, s[m])) for m in range(nm)])
    ref = logsumexp(phi[None] + lw[:, None, :], axis=2) - np.log(nt)
    print("ln_merger_at with log weights: max error %.1e" % np.abs(got - ref).max())
    assert np.abs(got - ref).max() < 2e-5


def test_own_columns_of_ln_evidence():
    """the column of each template's own signal power in ln_evidence (MergerTable: Phi at the grid points, computed
    once) against ln_merger_at, which computes Phi at the grid points each call reaches: the same interpolation and
    the same sums, so the two agree to rounding.  Without weights, with log weights (a unit row; a row with a time at
    w = 2000 that weighs exp(-1995); a row of -inf: no time at all) and with linear weights; a template with s = 0
    (ln Phi = 0); blocks that cut the templates; a vloud off the grid.  The table grows when a segment is louder than
    what it holds, and the columns of lnT are those of a call without own, bit for bit.  A table of another power is
    told apart (the comparison can fail)"""
    rng = np.random.default_rng(21)
    lq, ln_u, w = trajectory(rng, 5, 4096, loud=[(1, 100, 2000.0), (2, 50, 30.0), (4, 7, 9.0)])
    nm, nt = w.shape
    lw = np.stack([np.zeros(nt), np.log(rng.uniform(0.01, 1.0, nt)), np.log(rng.uniform(0.01, 1.0, nt)),
                   np.full(nt, -np.inf)])
    lw[1, 100] = -1995.0
    lnT = ln_l(single.V[:, None] ** 2, np.array([0.5, 25.0])[None, :])
    powers = [np.array([0.3, 3000.0, 30.0, 0.0, 2.0]), np.array([1.0, 0.02, 700.0, 5.0, 0.0])]
    worst = 0.0
    for weights, kw in ((None, {}), (lw, dict(log_weights=True)), (np.exp(lw[2]), {})):
        for block, vloud in ((256, 2.0), (2, 2.0), (256, 1.2349)):
            tabs = [single.MergerTable(s) for s in powers]
            assert tabs[0].P.shape == (nm, 256)
            got = single.ln_evidence(lq, ln_u, lnT, weights, block=block, vloud=vloud, own=tabs, **kw)
            plain = single.ln_evidence(lq, ln_u, lnT, weights, block=block, vloud=vloud, **kw)
            assert got.shape == plain.shape[:-1] + (4,) and np.array_equal(got[..., :2], plain)
            for k, s in enumerate(powers):
                ref = single.ln_merger_at(lq, ln_u, s, weights, block=block, vloud=vloud, **kw)
                fin = np.isfinite(ref)
                assert np.array_equal(np.isfinite(got[..., 2 + k]), fin) and fin.any()
                np.testing.assert_allclose(got[..., 2 + k][fin], ref[fin], rtol=1e-12, atol=1e-10)
                worst = max(worst, np.abs(got[..., 2 + k][fin] - ref[fin]).max())
            assert tabs[0].P.shape[1] >= 4474 and tabs[1].P.shape == tabs[0].P.shape     # w = 2000: grid point 4472
            if weights is lw:
                assert np.all(got[3] == -np.inf) and np.all(np.abs(got[0][:, 2 + 0][powers[0] == 0.0]) < 1e-12)
    print("own columns against ln_merger_at: largest difference %.1e" % worst)
    other = single.ln_evidence(lq, ln_u, lnT, own=[single.MergerTable(1.01 * powers[0])])[:, 2]
    assert np.abs(other - single.ln_merger_at(lq, ln_u, powers[0])).max() > 1e-4


def test_glitch_time_series_zero_weight_block():
    """templates with pi_m = 0 for a whole block (the log-sum-exp over the block has max -inf): the series is that of
    the templates with weight, whatever the block size, and glitch_time_weights of an all -inf series (no glitch
    evidence anywhere) is the weights of a flat one: no time is preferred, the probability that the glitch is away
    from t' is 1 - (2 W + 1) / n"""
    rng = np.random.default_rng(0)
    lq, ln_u, w = trajectory(rng, 6, 2048, loud=[(4, 100, 60.0)])
    pk = single.amplitude_weights()
    pm = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
    ref = single.glitch_time_series(lq[4:], ln_u[4:], pk, pm[4:])
    for block in (256, 2, 1):
        with np.errstate(invalid="raise"):                 # the nan came from -inf - -inf
            lg = single.glitch_time_series(lq, ln_u, pk, pm, block=block)
        assert np.isfinite(lg).all()
        np.testing.assert_allclose(lg, ref, rtol=0, atol=1e-12)
    lg = single.glitch_time_series(lq, ln_u, pk, np.zeros(6), block=2)          # no weight at all
    assert np.all(lg == -np.inf)
    n, hw = 64, 5
    ln_k, ln_n = single.glitch_time_weights(np.full(n, -np.inf), hw)
    assert np.allclose(np.exp(ln_k), 1.0 - (2 * hw + 1) / n, rtol=0, atol=1e-14)
    assert np.allclose(np.exp(ln_n), (2 * hw + 1) / n, rtol=1e-14, atol=0)


def test_vloud_off_grid():
    """a vloud that is not a multiple of DV: the quiet set is v <= DV (J - 2) and the loud set the rest, so every
    arrival time is counted once.  The unit column has ln E = 0 and so has a unit merger column of ln_merger_at;
    columns l(w; x) against the exact sum; and the mean of the glitch time series against sum pi_m pi_k E"""
    rng = np.random.default_rng(3)
    lq, ln_u, w = trajectory(rng, 2, 20000, loud=[(0, 10, 40.0)])
    nm, nt = w.shape
    unit = np.zeros((single.NV, 1))
    xs = np.array([0.5, 25.0])
    lnT = ln_l(single.V[:, None] ** 2, xs[None, :])
    ref = logsumexp(ln_l(w[:, :, None], xs[None, None, :]), axis=1) - np.log(nt)
    pk = rng.uniform(0.0, 1.0, single.RHO_EDGES.size - 1)
    pk /= pk.sum()
    for vloud in (2.0, 1.234, 1.236, 1.2349, 0.5):
        got = single.ln_evidence(lq, ln_u, unit, vloud=vloud)
        print("vloud %.4f: ln E of the unit column %+.1e" % (vloud, np.abs(got).max()))
        assert np.abs(got).max() < 1e-12
        assert np.abs(single.ln_evidence(lq, ln_u, lnT, vloud=vloud) - ref).max() < 2e-5
        lnE = single.ln_evidence(lq, ln_u, single.glitch_columns(), vloud=vloud)
        lg = single.glitch_time_series(lq, ln_u, pk, vloud=vloud)
        assert abs(logsumexp(lg) - np.log(nt) - logsumexp(lnE + np.log(pk)[None, :] - np.log(nm))) < 1e-12
    s = np.array([0.5, 4.0])
    phi = np.stack([single._ln_phi(np.sqrt(w[m]), np.full(nt, s[m])) for m in range(nm)])
    ref = logsumexp(phi, axis=1) - np.log(nt)
    for vloud in (1.234, 1.236):
        assert np.abs(single.ln_merger_at(lq, ln_u, s, vloud=vloud) - ref).max() < 2e-5


def test_ln_evidence_steep_columns_below_vloud():
    """accuracy of ln_evidence of the amplitude bins for rows whose arrival times are all below vloud, where the
    histogram route interpolates T linearly in v (less its curvature term): the reference is the quadrature of the
    columns themselves (single._ln_bin) at the v of every time.  Up to rho = 12.2 (the bins that are used) the worst
    error of ln E is 1.2e-5 (the last of these bins, times piled at v = 1.995), growing like rho^4 for the steeper
    bins (1e-3 at rho ~ 35, 1e-2 at 60, 0.33 at 140) which weigh exp(-450) and less in the sums over the bins; the
    fiducial sum G = sum_k pi_k E_k is right to 3e-7 in ln G"""
    e = single.RHO_EDGES
    nb = int(np.searchsorted(e, 12.2 + 1e-9)) - 1
    assert nb == 19 and e[nb] < 12.2
    pk = single.amplitude_weights()
    rng = np.random.default_rng(2)
    nt = 4096
    errs = []
    for v in (np.sqrt(np.minimum(rng.exponential(size=nt), 3.9)), rng.uniform(0.0, 2.0, nt), np.full(nt, 1.995)):
        lq = np.log(v).astype(np.float32)[None]
        v = np.exp(lq[0].astype(float))
        got = single.ln_evidence(lq, np.zeros(1), single.glitch_columns())[0]
        ref = np.empty(e.size - 1)
        for k, (a, b) in enumerate(zip(e[:-1], e[1:])):
            c = np.linspace(a, b, int(np.ceil((b - a) / 2.0)) + 1)
            p = np.stack([single._ln_bin(v, c0, c1) for c0, c1 in zip(c[:-1], c[1:])])
            ref[k] = logsumexp(logsumexp(p, axis=0) - np.log(b - a)) - np.log(nt)
        errs.append(np.abs(got[:nb] - ref[:nb]).max())
        print("rows below vloud: worst |d ln E| of the bins up to rho = 12.2: %.1e; of all bins %.1e; ln G error %.1e"
              % (errs[-1], np.abs(got - ref).max(), logsumexp(got, b=pk) - logsumexp(ref, b=pk)))
        assert errs[-1] < 5e-5
        assert abs(logsumexp(got, b=pk) - logsumexp(ref, b=pk)) < 1e-6
