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


def test_selected_times(case):
    a, lkH, lkL = case
    times = select_times(a["traj"], a["tables"]["tau"], 24, 40, seed=3)
    assert np.any(times[1] != 0)                         # the random draw carries a weight
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
