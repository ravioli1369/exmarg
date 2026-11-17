"""gather_cpu against a slow float64 reimplementation of the gather.py docstring, with and without time weights.

Tolerance: gather_cpu accumulates in float64 like the reference, but takes the phase difference of the float32
trajectories in float32 and keeps beta and Delta 1e-6 of a grid step inside the table's last node; both move
ln Lambda by at most 4e-8 here, so RTOL = 1e-6 (relative and absolute) is loose for the CPU and tight against
any error in indexing or weighting (those move ln Lambda by 1e-2 or more).  Where two quantities must be the same
arithmetic (variant 0 against no weights, a zero weight against no weight) they are compared bit for bit.  The
GPU sums in float32 and is held to 2e-4 against gather_cpu by scripts/validate/gather_gpu_weights.py."""
import numpy as np
import pytest

from exmarg.search.gather import LK_MIN, gather_cpu, select_times

NPHI = 96
RTOL = 1e-6


def synthetic(seed=0, n_template=4, n_time=256, lags=(-3, 4), nb=12, nd=10, ntab=3, ns=2, nsel=3):
    """smooth random tables, random trajectories with a loud stretch, (template, slot) -> table, and weights"""
    rng = np.random.default_rng(seed)
    tau = np.arange(*lags)
    beta = -1.0 + 0.5 * np.arange(nb)
    delta = -2.0 + 0.4 * np.arange(nd)
    ph = np.cos(2 * np.pi * np.arange(NPHI // 2 + 1) / NPHI)[None, :, None, None]
    xb = np.linspace(0, 1, nb)[None, None, :, None]
    xd = np.linspace(-1, 1, nd)[None, None, None, :]
    xt = np.linspace(-1, 1, tau.size)[:, None, None, None]
    T = []
    for _ in range(ntab):
        c = rng.normal(size=6)
        t = (c[0] + c[1] * ph * (1 + xb) + 3 * xb + c[2] * xb ** 2 - 2 * xd ** 2 + c[3] * xd * xb
             + 0.3 * ph * xd + 0.2 * c[4] * xt + 0.1 * c[5] * xt * ph)
        T.append(np.ascontiguousarray(t, np.float32))
    lqH = rng.normal(0.0, 0.8, (n_template, n_time))
    lqL = rng.normal(0.0, 0.8, (n_template, n_time))
    lqH[:, 40:46] += 2.0
    lqL[:, 38:44] += 2.0
    traj = dict(lqH=lqH.astype(np.float32), lqL=lqL.astype(np.float32),
                phH=rng.uniform(-np.pi, np.pi, (n_template, n_time)).astype(np.float32),
                phL=rng.uniform(-np.pi, np.pi, (n_template, n_time)).astype(np.float32))
    args = dict(traj=traj, tables=dict(T=T, tau=tau, beta=beta, delta=delta),
                mass_ids=rng.choice(n_template, nsel, replace=False),
                lna=rng.uniform(-0.5, 2.0, (nsel, ns)), tab_ids=rng.integers(0, ntab, (nsel, ns)))
    lkH = np.log(rng.uniform(0.0, 1.0, n_time)).astype(np.float32)
    lkL = np.log(rng.uniform(0.0, 1.0, n_time)).astype(np.float32)
    return args, lkH, lkL


def _frac(x, n):
    """x clamped to [0, n - 1] -> (lower node, fraction), the last interval holding the top node"""
    x = np.clip(x, 0.0, n - 1.0)
    i = np.minimum(np.floor(x).astype(int), n - 2)
    return i, x - i


def _lnS(Tfull, lna, lqH, phH, lqL, phL, beta, delta):
    """lnS of one lag at each arrival time: bilinear in (beta, Delta), linear in the phase difference;
    beta clamped to the grid with ln q_L moved along and ln q_H kept"""
    db, dd = beta[1] - beta[0], delta[1] - delta[0]
    b = np.clip(lna + 2.0 * lqL, beta[0], beta[-1])
    lqLc = (b - lna) / 2.0
    ib, fb = _frac((b - beta[0]) / db, beta.size)
    jd, fd = _frac((lqH - lqLc - delta[0]) / dd, delta.size)
    x = np.mod((phH - phL) * NPHI / (2 * np.pi), NPHI)
    c0 = np.floor(x).astype(int)
    fc = x - c0
    out = 0.0
    for c, wc in ((c0 % NPHI, 1.0 - fc), ((c0 + 1) % NPHI, fc)):
        out = out + wc * ((1 - fb) * (1 - fd) * Tfull[c, ib, jd] + (1 - fb) * fd * Tfull[c, ib, jd + 1]
                          + fb * (1 - fd) * Tfull[c, ib + 1, jd] + fb * fd * Tfull[c, ib + 1, jd + 1])
    return out


def _lse(a):
    mx = a.max()
    return mx + np.log(np.exp(a - mx).sum())


def reference(traj, tables, mass_ids, lna, tab_ids, times=None, lkH=None, lkL=None):
    """(4, NS, nsel): ln( sum_tau sum_t exp(lnS + w_v) ) - ln n_time with w_0 = 0, w_1 = lkH(t + tau),
    w_2 = lkL(t), w_3 = lkH(t + tau) + lkL(t) (log weights floored at LK_MIN), over every t or over the selected t
    with their log weights"""
    n_time = traj["lqH"].shape[1]
    zero = np.zeros(n_time)
    wH = zero if lkH is None else np.maximum(np.asarray(lkH, np.float64), LK_MIN)
    wL = zero if lkL is None else np.maximum(np.asarray(lkL, np.float64), LK_MIN)
    beta, delta = np.asarray(tables["beta"], float), np.asarray(tables["delta"], float)
    ns = np.shape(tab_ids)[1]
    out = np.empty((4, ns, len(mass_ids)))
    for j, mid in enumerate(mass_ids):
        t = np.arange(n_time) if times is None else np.asarray(times[0][mid])
        lw = np.zeros(t.size) if times is None else np.asarray(times[1][mid], np.float64)
        q = {k: np.asarray(v[mid], np.float64) for k, v in traj.items()}
        for n in range(ns):
            Th = np.asarray(tables["T"][tab_ids[j, n]], np.float64)
            Tfull = np.concatenate([Th, Th[:, NPHI // 2 - 1:0:-1]], axis=1)     # even in the phase: full circle
            terms = [[], [], [], []]
            for r, tau in enumerate(tables["tau"]):
                i1 = (t + tau) % n_time
                s = _lnS(Tfull[r], lna[j, n], q["lqH"][i1], q["phH"][i1], q["lqL"][t], q["phL"][t], beta, delta) + lw
                for v, w in enumerate((zero[t], wH[i1], wL[t], wH[i1] + wL[t])):
                    terms[v].append(s + w)
            for v in range(4):
                out[v, n, j] = _lse(np.concatenate(terms[v])) - np.log(n_time)
    return out


@pytest.fixture(scope="module")
def case():
    return synthetic()


def _close(got, ref):
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=RTOL)


def test_no_weights(case):
    a, _, _ = case
    got = gather_cpu(**a)
    assert got.shape == (2, 3)
    _close(got, reference(**a)[0])


def test_variant0_is_unweighted(case):
    a, lkH, lkL = case
    got = gather_cpu(**a, time_weights=(lkH, lkL))
    assert got.shape == (4, 2, 3)
    np.testing.assert_array_equal(got[0], gather_cpu(**a))


def test_zero_weights(case):
    a, _, _ = case
    z = np.zeros(a["traj"]["lqH"].shape[1], np.float32)
    got = gather_cpu(**a, time_weights=(z, z))
    for v in (1, 2, 3):
        np.testing.assert_array_equal(got[v], got[0])


def test_lkH_only(case):
    a, lkH, _ = case
    z = np.zeros_like(lkH)
    got = gather_cpu(**a, time_weights=(lkH, z))
    np.testing.assert_array_equal(got[2], got[0])
    np.testing.assert_array_equal(got[3], got[1])
    _close(got, reference(**a, lkH=lkH))
    assert np.all(got[1] < got[0] - 0.05)                # the weights are < 1 and matter


def test_lkL_only(case):
    a, _, lkL = case
    z = np.zeros_like(lkL)
    got = gather_cpu(**a, time_weights=(z, lkL))
    np.testing.assert_array_equal(got[1], got[0])
    np.testing.assert_array_equal(got[3], got[2])
    _close(got, reference(**a, lkL=lkL))
    assert np.all(got[2] < got[0] - 0.05)


def test_both(case):
    a, lkH, lkL = case
    got = gather_cpu(**a, time_weights=(lkH, lkL))
    _close(got, reference(**a, lkH=lkH, lkL=lkL))
    assert np.all(got[3] < np.minimum(got[1], got[2]) - 0.05)    # variant 3 carries both weights


@pytest.mark.parametrize("shared", [False, True])
def test_selected_times(case, shared):
    a, lkH, lkL = case
    times = select_times(a["traj"], a["tables"]["tau"], 24, 40, seed=3, shared=shared)
    assert np.any(times[1] != 0)                         # the random draw carries a weight
    assert not shared or np.any(times[1] == np.float32(LK_MIN))    # and none where it falls on a loud time
    _close(gather_cpu(**a, times=times), reference(**a, times=times)[0])
    got = gather_cpu(**a, times=times, time_weights=(lkH, lkL))
    np.testing.assert_array_equal(got[0], gather_cpu(**a, times=times))
    _close(got, reference(**a, times=times, lkH=lkH, lkL=lkL))


def test_extreme_weights(case):
    a, lkH, lkL = case
    n_time = lkH.size
    # every H weight zero (-inf, clipped at LK_MIN): variant 1 is variant 0 + LK_MIN and variant 3 is variant 2 + LK_MIN
    got = gather_cpu(**a, time_weights=(np.full(n_time, -np.inf, np.float32), lkL))
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got[1], got[0] + LK_MIN, rtol=0, atol=1e-9)
    np.testing.assert_allclose(got[3], got[2] + LK_MIN, rtol=0, atol=1e-9)
    # both zero everywhere: variant 3 is variant 0 + 2 LK_MIN
    inf = np.full(n_time, -np.inf, np.float32)
    got = gather_cpu(**a, time_weights=(inf, inf))
    np.testing.assert_allclose(got[3], got[0] + 2 * LK_MIN, rtol=0, atol=1e-9)
    # zero weight on half the times, large negative weights (-1e4, above the floor; -1e6, below it) on a few more,
    # with the loud stretch (t = 38..45) knocked out in L
    wH, wL = lkH.copy(), lkL.copy()
    wH[::2] = -np.inf
    wH[1:20:4] = -1e4
    wH[3:20:4] = -1e6
    wL[30:50] = -np.inf
    got = gather_cpu(**a, time_weights=(wH, wL))
    assert np.all(np.isfinite(got))
    _close(got, reference(**a, lkH=wH, lkL=wL))


def _correlated_trajectories(rng, nm, nt, rho):
    """ln|z| of nm templates whose complex series are correlated at rho with a common one: their loud times differ"""
    c = lambda *sh: (rng.standard_normal(sh) + 1j * rng.standard_normal(sh)) / np.sqrt(2.0)
    out = {}
    for I in "HL":
        common = c(1, nt)
        out["lq" + I] = np.log(np.abs(rho * common + np.sqrt(1.0 - rho ** 2) * c(nm, nt))).astype(np.float32)
    return out


def test_select_times_shared():
    """shared=True: the random arrival times are the same times for every template, also when the templates differ
    in their loudest times (a draw that falls on one of a template's loudest times, which are summed exactly,
    carries no weight for that template); the selection stays an unbiased estimate of the sum over all times."""
    rng = np.random.default_rng(5)
    nm, nt, K, S = 3, 512, 32, 64
    traj = _correlated_trajectories(rng, nm, nt, 0.9)
    lags = np.arange(-3, 4)
    sel, lw = select_times(traj, lags, K, S, 11, shared=True)
    loud = [np.sort(sel[m][lw[m] == 0.0]) for m in range(nm)]
    assert all(x.size == K for x in loud)
    assert not np.array_equal(loud[0], loud[1]), "the fixture needs templates with different loudest times"
    drawn = [np.sort(sel[m][lw[m] != 0.0]) for m in range(nm)]
    assert np.array_equal(drawn[0], drawn[1]) and np.array_equal(drawn[0], drawn[2]), "the random times differ"
    hits = 0
    for m in range(nm):                                  # no weight on a loud time, ln(n_time / n_random) elsewhere
        r = lw[m] != 0.0
        on_loud = np.isin(sel[m][r], loud[m])
        hits += on_loud.sum()
        np.testing.assert_array_equal(lw[m][r][on_loud], np.float32(LK_MIN))
        np.testing.assert_allclose(lw[m][r][~on_loud], np.log(nt / S), rtol=1e-6)
    assert hits > 0, "the fixture needs a draw that falls on a loud time"
    sel0, lw0 = select_times(traj, lags, K, S, 11)
    assert not np.array_equal(np.sort(sel0[0][lw0[0] != 0.0]), np.sort(sel0[1][lw0[1] != 0.0])), \
        "the default draws once per template"
    f = np.exp(0.3 * rng.standard_normal((nm, nt)))                 # any positive function of the arrival time
    f[0, loud[0][:8]] *= 50.0                                       # with weight on the loudest times
    for shared in (True, False):
        est = []
        for seed in range(400):
            sel, lw = select_times(traj, lags, K, S, seed, shared=shared)
            est.append((np.take_along_axis(f, sel.astype(np.int64), axis=1) * np.exp(lw.astype(np.float64))).sum(axis=1))
        est = np.array(est)
        err = est.std(axis=0, ddof=1) / np.sqrt(est.shape[0])
        assert np.all(np.abs(est.mean(axis=0) - f.sum(axis=1)) < 4 * err), \
            "shared %s: selected sum %s +- %s, full sum %s" % (shared, est.mean(axis=0), err, f.sum(axis=1))
        assert np.all(err < 0.02 * f.sum(axis=1)), "the test resolves a bias of a few percent"


def test_float64_weights_deep_in_the_tail(case):
    """a log weight near -3000 reaches the CPU gather in float64: float32 has a spacing of 2.4e-4 there and would
    round -3000.00005 by 5e-5 (and each weight of a random set by up to 1.2e-4), 50 times RTOL"""
    a, lkH, lkL = case
    n_time = lkH.size
    d = -3000.00005
    flat = np.full(n_time, d)
    got = gather_cpu(**a, time_weights=(flat, np.zeros(n_time)))       # a constant weight is a shift of ln Lambda
    np.testing.assert_allclose(got[1], got[0] + d, rtol=0, atol=1e-9)
    wH, wL = lkH.astype(np.float64) + d, lkL.astype(np.float64) + d
    got = gather_cpu(**a, time_weights=(wH, wL))
    _close(got, reference(**a, lkH=wH, lkL=wL))


def test_nan_time_weights(case):
    """a NaN time weight is an error naming the weights (and the times); -inf is a zero weight, clipped at LK_MIN"""
    a, lkH, lkL = case
    bad = lkH.copy()
    bad[[7, 100]] = np.nan
    with pytest.raises(ValueError, match=r"lkH has 2 NaN.*\[7, 100\]"):
        gather_cpu(**a, time_weights=(bad, lkL))
    with pytest.raises(ValueError, match=r"lkL has 2 NaN"):
        gather_cpu(**a, time_weights=(lkH, bad))
    ok = lkH.copy()
    ok[7] = -np.inf
    assert np.all(np.isfinite(gather_cpu(**a, time_weights=(ok, lkL))))


def test_checked_weights(case):
    """the time weights checked once per segment (gather.checked_weights) are what _weights makes of the pair at
    every launch (float64 on the CPU, clipped at LK_MIN), and gather_cpu with them returns the same bytes; they are
    passed through for the same n_time and type only; None stays None and a NaN is the same error"""
    from exmarg.search import gather
    a, lkH, lkL = case
    lkL = lkL.copy()
    lkL[5] = -np.inf
    n = lkH.size
    chk = gather.checked_weights((lkH, lkL), n, np)
    plain = gather._weights((lkH, lkL), n, np, np.float64)
    assert isinstance(chk, tuple) and all(x.dtype == np.float64 and x.tobytes() == y.tobytes() for x, y in zip(chk, plain))
    assert chk[1][5] == gather.LK_MIN and all(x is y for x, y in zip(gather._weights(chk, n, np, np.float64), chk))
    assert gather._weights(chk, n, np, np.float32)[0].dtype == np.float32
    with pytest.raises(ValueError, match="shape"):
        gather._weights(chk, n + 1, np, np.float64)
    assert gather_cpu(**a, time_weights=chk).tobytes() == gather_cpu(**a, time_weights=(lkH, lkL)).tobytes()
    assert gather.checked_weights(None, n, np) is None
    bad = lkH.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match=r"lkH has 1 NaN.*\[3\]"):
        gather.checked_weights((bad, lkL), n, np)
