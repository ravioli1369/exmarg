"""tests of the model with many glitches per segment inside pe_redshift and pe_intrinsic (docs/glitch_likelihood.md,
Sec. 4a): glitch=dict(model='windows', ...), its outputs ln_B, ln_c_I, state_I, ln_window_I, and window_evidences.

The store, bank and settings of the segments are those of test_pe_glitch.py (smooth random tables on the 53 nodes of
s_grid.npz, lags -6 to 14 of which the light travel time keeps -6 to 10, segments of 4 s up to 128 Hz, 4096 arrival
times, templates of 40, 60 and 90 M_sun), with n_window = 64 windows of 64 arrival times (62.5 ms).  The segments
(two simulated segments, template glitches added at given times):

    'merger'  a merger (M 60, network SNR about 18, w_L about 48 at sample 1598, w_H about 120), two template glitches
              in H (SNR 15 in the zone of the merger, SNR 20 1.3 s later) and one in L (SNR 12, 2.2 s after the merger)
    'clean'   noise alone (the second simulated segment, merger at y = 1e-8)
    'one'     the clean noise with one L glitch (SNR 12, template 60)
    'two'     the clean noise with two L glitches of the same SNR (12, template 60, 1.6 s apart)

The reference of ln_B (group b) is the reference of test_pe_glitch.py (every arrival time and lag, in the log
domain) with weights computed here from the window evidences that pe_redshift returns, by the formulas of Sec. 4a in
decimal arithmetic with 50 digits (not by single.window_posterior).

The synthetic tables are not a likelihood ratio (E_0 Lambda is not 1, and lnS grows like s w_L / 2, so that the
sidelobes of a glitch in L, beyond any window, carry ln Lambda).  Groups c and e use a store built here for an
incoherent merger, lnS(tau) = ln p(tau) + ln Phi(w_L; s) + ln Phi(w_H; s) with p(tau) uniform on five lags and Phi
of single.py, read through the same gather: its E_0 Lambda is 1.  For the unit mean (group e) the matched-filter
outputs are white (independent arrival times and templates, as in test_windows.py) so that a glitch rings one
template at one time and the windows are independent, which makes the model of Sec. 4a exact and E[B] = 1 a true
statement.
"""

import decimal
import functools
import importlib.util
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from exmarg.pp import simulate
from exmarg.profiles import profile
from exmarg.search import mass_bank, pe, read_segments, single
from exmarg.search.store import RedshiftStore, unbias
from test_pe_glitch import RTOL, REDSHIFT, case, reference  # noqa: F401  (case is a fixture)

ROOT = Path(__file__).resolve().parents[1]
HEAD = "d05cb8d"                            # the branch before model='windows': pe.py as it was
MASSES = np.array([40.0, 60.0, 90.0])
N_WINDOW, ZONE = 64, 1
NR = single.RHO_EDGES.size - 1


def law(gamma, rho_min):
    return single.amplitude_weights(gamma=gamma, rho_min=rho_min)


# two states in H, three in L (one without glitches): different K, different laws
P_H = dict(f=[0.85, 0.15], q=[0.01, 0.15], pi_k=np.stack([law(4.0, 4.0), law(2.5, 6.0)]))
P_L = dict(f=[0.6, 0.3, 0.1], q=[0.0, 0.02, 0.2], pi_k=np.stack([law(4.0, 4.0), law(4.0, 5.0), law(2.0, 8.0)]))
WINDOWS = dict(model="windows", n_window=N_WINDOW, zone=ZONE, H=P_H, L=P_L)
# the model of group c, the same in both detectors
P_C = dict(f=[0.9, 0.1], q=[0.01, 0.1], pi_k=np.stack([law(4.0, 4.0), law(2.0, 6.0)]))
WINDOWS_C = dict(model="windows", H=P_C, L=P_C)


def add_glitch(d, segs, t0, rho, mass, phase):
    """d (n_freq,) plus an equal-mass zero-spin template of total mass mass at time t0 with phase phase and
    single-detector SNR rho (as exmarg.glitches.add_template_glitches, at a given time)"""
    T, nu = segs["T"], segs["nu"]
    nu_t, _, a, psi0 = profile(mass / 2, mass / 2, 0.0, 0.0, T, segs["fmin"], segs["fmax"], 1.0)
    h = np.zeros(nu.size, complex)
    h[np.searchsorted(nu, nu_t)] = a * np.exp(1j * (psi0 + phase))
    return d + h * np.exp(-2j * np.pi * nu / T * t0) * rho / np.sqrt(2.0 * np.sum(np.abs(h) ** 2 / segs["sig2"]))


@pytest.fixture(scope="module")
def data(case):
    """the four segments of the module docstring with the store of test_pe_glitch, and what _node_values needs"""
    y = iter([12.0, 1e-8])
    raw, _, _ = simulate(2, 77, lambda r: (60.0, 1.0, 0.0), lambda r: next(y), 4.0, 20.0, 128.0)
    dt = raw["T"] / 4096
    tm = 1598 * dt                                                      # the merger, found on the L trajectory
    g = lambda d, t, rho, ph: add_glitch(d, raw, (t % raw["T"]) // dt * dt, rho, 60.0, ph)
    dH = [g(g(raw["d_H"][0], tm + 0.05, 15.0, 0.3), tm + 1.3, 20.0, 1.1), raw["d_H"][1], raw["d_H"][1], raw["d_H"][1]]
    one = g(raw["d_L"][1], 1.0, 12.0, 0.7)
    dL = [g(raw["d_L"][0], tm + 2.2, 12.0, 2.0), raw["d_L"][1], one, g(one, 2.6, 12.0, 2.5)]
    segs = dict(raw, d_H=np.array(dH), d_L=np.array(dL))
    bank = mass_bank(MASSES, segs)
    assert bank.n_time == 4096
    store = case["store"]
    seg = read_segments(segs)
    geometry = pe._node_geometry(bank, store, 0.1267)
    groups = [list(range(g0, min(g0 + 6, store.s.size))) for g0 in range(0, store.s.size, 6)]
    return dict(store=store, segs=segs, bank=bank, seg=seg, geometry=geometry, groups=groups,
                rows=dict(merger=0, clean=1, one=2, two=3))


def node_values(c, glitch, rows):
    """_node_values of the segments rows (every arrival time): lamN and the dict of _single_nodes"""
    sd = pe._single_setup(c["bank"], c["seg"], c["store"], c["geometry"])
    gd = pe._glitch_setup(glitch, c["bank"], c["seg"], c["store"])
    return pe._node_values(c["bank"], c["seg"], rows, c["store"], c["geometry"], c["groups"], 256, False, 0, 0, 0, sd,
                           gd)


@pytest.fixture(scope="module")
def runs(data):
    """pe_redshift of every segment, every arrival time: plain single=True, the default glitch model and windows"""
    args = (data["segs"], data["bank"], data["store"])
    return dict(single=pe.pe_redshift(*args, n_loud=0, single=True, **REDSHIFT),
                old=pe.pe_redshift(*args, n_loud=0, glitch=True, **REDSHIFT),
                windows=pe.pe_redshift(*args, n_loud=0, glitch=WINDOWS, **REDSHIFT))


# ----------------------------------------------------------------------------------------- the reference weights


def direct_weights(lnE, f, q, pi, zone):
    """from the definitions of Sec. 4a in decimal arithmetic (50 digits), for one detector-segment:

        L_w^s = sum_k pi[s, k] E_w[k],  D_w^s = 1 - q_s + q_s L_w^s,  C^s = prod_w D_w^s,  c = sum_s f_s C^s,
        r_s = f_s C^s / c,  kappa^s(w) = prod_{w' in zone(w)} (1 - q_s) / D_w'^s,  kbar = sum_s r_s kappa^s,
        1 - kbar = sum_s r_s (1 - kappa^s)

    returns ln c, r (K,), ln kbar (n_w,), ln(1 - kbar) (n_w,)"""
    D = decimal.Decimal
    with decimal.localcontext() as ctx:
        ctx.prec = 50
        nw, nk = lnE.shape
        E = [[D(float(x)).exp() if np.isfinite(x) else D(0) for x in row] for row in lnE]
        fC, kappa = [], []
        for s in range(len(f)):
            qs = D(float(q[s]))
            one = D(1) - qs
            Dw = [one + qs * sum(D(float(pi[s][k])) * E[w][k] for k in range(nk)) for w in range(nw)]
            fC.append(D(float(f[s])) * math.prod(Dw))
            kappa.append([math.prod(one / Dw[(w + j) % nw] for j in range(-zone, zone + 1)) for w in range(nw)])
        c = sum(fC)
        r = [x / c for x in fC]
        kbar = [sum(r[s] * kappa[s][w] for s in range(len(f))) for w in range(nw)]
        near = [sum(r[s] * (D(1) - kappa[s][w]) for s in range(len(f))) for w in range(nw)]
        ln = lambda x: float(x.ln()) if x > 0 else -np.inf
        return ln(c), np.array([float(x) for x in r]), np.array([ln(x) for x in kbar]), np.array([ln(x) for x in near])


def direct_time_weights(res, e, glitch, n_time):
    """{I: (ln kbar_I, ln(1 - kbar_I))} per arrival time of segment e, from the window evidences of the result res:
    the arrival time t is in the window t // n_win; and {I: (ln c, r)}"""
    win = np.arange(n_time) // (n_time // glitch["n_window"])
    kbar, post = {}, {}
    for I in "HL":
        p = glitch[I]
        lc, r, lk, ln = direct_weights(res["ln_window_" + I][e], p["f"], p["q"], p["pi_k"], glitch["zone"])
        kbar[I], post[I] = (lk[win], ln[win]), (lc, r)
    return kbar, post


# ----------------------------------------------------------------------------------------- a. no glitches


@pytest.mark.parametrize("K", [1, 2])
def test_no_glitches(data, K):
    """q = 0 in every state: kbar = 1, 1 - kbar = 0, and ln_B is ln_lambda.  With one state (f = 1) ln kbar is 0
    exactly, the weighted gather sum adds 0 to every lnS and ln_B equals ln_lambda bit for bit, in pe_redshift and
    pe_intrinsic; with two states ln kbar = ln(r_1 + r_2) is 0 only to rounding (ln c = 5.6e-17 here), which could
    move ln_B by an ulp: asserted to 1e-13, measured 0.  ln c = 0 and the posterior of the state is its prior"""
    f = [1.0] if K == 1 else [0.3, 0.7]
    p = dict(f=f, q=[0.0] * K, pi_k=np.stack([law(3.0, 4.0 + s) for s in range(K)]))
    gl = dict(model="windows", H=p, L=p)
    args = (data["segs"], data["bank"], data["store"])
    lines = []
    for fn, kw in ((pe.pe_redshift, REDSHIFT), (pe.pe_intrinsic, {})):
        r = fn(*args, n_loud=0, glitch=gl, **kw)
        fin = np.isfinite(r["ln_lambda"])
        assert fin.sum() >= (100 if fn is pe.pe_redshift else 12) and np.array_equal(np.isfinite(r["ln_B"]), fin)
        err = np.abs(r["ln_B"][fin] - r["ln_lambda"][fin]).max()
        lines.append("%s, K = %d: max |ln_B - ln_lambda| %.1e over %d values; ln c %s; state %s"
                     % (fn.__name__, K, err, fin.sum(), np.abs(r["ln_c_H"]).max(), r["state_H"][0]))
        if K == 1:
            assert np.array_equal(r["ln_B"], r["ln_lambda"])
        else:
            assert err < 1e-13
        for I in "HL":
            assert np.abs(r["ln_c_" + I]).max() < 1e-15 and np.allclose(r["state_" + I], f, rtol=1e-14, atol=0)
    print("\n" + "\n".join(lines))


# ----------------------------------------------------------------------------------------- b. the reference


def test_reference(data, runs):
    """the segment with a merger, two loud glitches in H (one in the zone of the merger) and one in L, two states in
    H and three in L: (1) ln_c_I and state_I of pe_redshift against the decimal evaluation of Sec. 4a on the window
    evidences it returns, to 1e-11 (relative where |ln| > 1); (2) the time weights, delay averages and ln c0 of the
    code against those built from the decimal weights, to the same; (3) ln_B at every node and boundary point, reduced like ln_lambda, against
    the reference of test_pe_glitch with the decimal weights (every arrival time and lag in the log domain), to
    RTOL = 1e-6 as there; and the four terms of B each within e^-12 of B somewhere on the nodes, so that each is
    seen by the comparison"""
    c, res, e = data, runs["windows"], data["rows"]["merger"]
    bank, store = c["bank"], c["store"]
    kbar, post = direct_time_weights(res, e, WINDOWS, bank.n_time)
    rel = lambda a, b: np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))
    err_c = max(rel(res["ln_c_" + I][e], post[I][0]) for I in "HL")
    err_r = max(np.abs(res["state_" + I][e] - post[I][1]).max() for I in "HL")
    sd = pe._single_setup(bank, c["seg"], store, c["geometry"])
    gd = pe._glitch_setup(WINDOWS, bank, c["seg"], store)
    traj = bank.trajectories({I: c["seg"]["d_" + I][e] for I in "HL"}, {I: c["seg"]["sig2_" + I] for I in "HL"})
    gw = pe._glitch_weights(traj, gd, sd, np)
    h = reference(c, e, kbar=kbar)
    err_w = []
    for got, want in ((gw["tw"][0], kbar["H"][0]), (gw["tw"][1], kbar["L"][0]), (gw["L"][0], h["oH"]),
                      (gw["H"][0], h["oL"]), (gw["L"][1], h["oH"] + kbar["L"][0]), (gw["H"][1], h["oL"] + kbar["H"][0])):
        assert np.array_equal(np.isfinite(got), np.isfinite(want))
        fin = np.isfinite(want)
        err_w.append(rel(got[fin], want[fin]))
    err_w.append(rel(gw["lc0"], h["lc0"]))
    _, one = node_values(c, WINDOWS, [e])
    ok = np.isfinite(h["gHL"])
    assert np.array_equal(np.isfinite(one["lamN_gHL"][0]), ok)
    dn, mag = np.abs(one["lamN_gHL"][0][ok] - h["gHL"][ok]), np.abs(h["gHL"][ok])
    print("\nnode values: %d pairs, max |d ln B| %.1e (at ln B %.0f), %.1e where |ln B| < 100 (%d pairs), relative %.1e"
          % (ok.sum(), dn.max(), h["gHL"][ok][np.argmax(dn)], dn[mag < 100].max(), (mag < 100).sum(),
             (dn / np.maximum(1, mag)).max()))
    ref = pe._to_source_frame(h["gHL"], c["geometry"][2], bank.M, pe.D0 / pe._dl_of_z(res["z"]), res["m1"], res["z"], np)
    got = res["ln_B"][e]
    fin = np.isfinite(got)
    terms = np.stack([h[k] for k in ("far_HL", "near_H_farL", "near_L_farH")] + [np.full(h["gHL"].shape, h["lc0"])])
    with np.errstate(invalid="ignore"):
        share = (terms - h["gHL"][None]).reshape(4, -1)
    small = (ok & (np.abs(h["gHL"]) < 100)).ravel()
    share_small = np.where(small[None], share, -np.inf).max(axis=1)
    share = np.where(ok.ravel()[None], share, -np.inf).max(axis=1)
    lkH, lkL = kbar["H"][0], kbar["L"][0]
    print("\nmerger segment: ln c_H %.4f ln c_L %.4f, state_H %s state_L %s; smallest ln kbar H %.1f L %.1f, largest "
          "ln(1 - kbar) H %.2e L %.2e\nerrors: ln c %.1e, state %.1e, weights %s; ln_B against the reference %.1e over "
          "%d grid points; largest share of each term of B (far, near H, near L, both near): %s; ln_B - ln_lambda from "
          "%.2f to %.2f" % (post["H"][0], post["L"][0], np.round(post["H"][1], 6), np.round(post["L"][1], 6), lkH.min(),
                            lkL.min(), kbar["H"][1].max(), kbar["L"][1].max(), err_c, err_r,
                            " ".join("%.1e" % x for x in err_w), np.abs(got[fin] - ref[fin]).max(), fin.sum(),
                            np.round(share, 2), (got[fin] - res["ln_lambda"][e][fin]).min(),
                            (got[fin] - res["ln_lambda"][e][fin]).max()))
    assert lkH.min() < -50 and lkL.min() < -50                       # loud glitches in both detectors
    assert err_c < 1e-11 and err_r < 1e-11 and max(err_w) < 1e-11
    assert fin.sum() > 100 and np.array_equal(fin, np.isfinite(ref))
    np.testing.assert_allclose(got[fin], ref[fin], rtol=RTOL, atol=RTOL)
    print("largest share of each term where |ln B| < 100: %s" % np.round(share_small, 2))
    assert (mag < 100).sum() >= 20 and dn[mag < 100].max() < 1e-7 and (dn / np.maximum(1, mag)).max() < RTOL
    assert np.all(share > -12.0) and np.all(share_small > -12.0)


# ----------------------------------------------------------------------------------------- c. the purpose


C_S = (2.0, 8.0, 32.0)                   # the nodes of the incoherent store of group c


def test_two_glitches(data):
    """two L glitches of the same SNR (12) and nothing in H, on the incoherent store (a likelihood ratio, so that
    the noise level of ln Lambda is about 0) at s = 2, 8, 32 and every template; node values, segments 'clean', 'one'
    and 'two'.  The model of Sec. 4 removes one glitch (ln lambda_gL of 'one' within 0.2 of ln Lambda of 'clean'), but
    with two it removes only the one of larger l_g and keeps the coherent sum of the other whole: ln lambda_gL of
    'two' is within 0.2 of ln Lambda of 'one' wherever that glitch raises ln Lambda by more than 2, and above the
    H-only value ln Lambda_H by more than 40 at s = 32 (by more than 2 at 6 of the 9 pairs).  ln B of the windows
    model is within 0.5 of its value on the clean noise with one glitch as with two, and below ln Lambda_H + 0.5,
    everywhere.  (With evidences exactly equal the old model would keep half of each glitch; the noise makes them
    unequal here, and it keeps the weaker one.)"""
    store = incoherent_store(C_S)
    c = dict(data, store=store, geometry=pe._node_geometry(data["bank"], store, 0.1),
             groups=[[k] for k in range(len(C_S))])           # one node per launch: each node has its own beta grid
    assert c["geometry"][0].all()
    rw = data["rows"]
    rows = [rw["clean"], rw["one"], rw["two"]]
    lam, old = node_values(c, True, rows)
    _, new = node_values(c, WINDOWS_C, rows)
    L, LH, gL, B = (a[:, :-1] for a in (lam, old["lamN_H"], old["lamN_gL"], new["lamN_gHL"]))   # (3, node, template)
    lines = ["s     M   | ln Lambda: clean   one    two | ln Lambda_H two | old ln lambda_gL: one   two | "
             "new ln B: clean   one   two"]
    for k, s in enumerate(C_S):
        for m, M in enumerate(MASSES):
            lines.append("%-5g %-3g | %15.2f %6.2f %6.2f | %15.2f | %22.2f %5.2f | %15.2f %5.2f %5.2f"
                         % (s, M, L[0, k, m], L[1, k, m], L[2, k, m], LH[2, k, m], gL[1, k, m], gL[2, k, m], B[0, k, m],
                            B[1, k, m], B[2, k, m]))
    lines.append("L window posterior: ln c_L %s, noisy state %s" % (np.round(new["ln_c_L"], 2),
                                                                  np.round(new["state_L"][:, 1], 4)))
    print("\n" + "\n".join(lines))
    raised = L[1] - L[0] > 2.0
    assert np.abs(gL[1] - L[0]).max() < 0.2                           # one glitch: removed by the old model
    assert raised.sum() >= 6 and np.abs(gL[2] - L[1])[raised].max() < 0.2   # two: the weaker one stays whole
    assert (gL[2] - LH[2]).max() > 40.0 and (gL[2] - LH[2] > 2.0).sum() >= 6
    assert np.abs(B[1:] - B[0]).max() < 0.5 and (B[2] - LH[2]).max() < 0.5  # the new model removes both


# ----------------------------------------------------------------------------------------- d. window evidences


def test_window_evidences(data, runs):
    """window_evidences gives the ln_window_* (and wmax_*) of pe_redshift bit for bit, with uniform and with given
    template weights; the mean of E_w over the windows is the mixture over the templates of ln_glitch_* (the whole
    segment) to 1e-12 (relative where |ln| > 1), including the windows of loud glitches"""
    c = data
    pm = np.array([0.2, 0.5, 0.3])
    lines = []
    for pi_m, res in ((None, runs["windows"]),
                      (pm, pe.pe_redshift(c["segs"], c["bank"], c["store"], n_loud=0, glitch=dict(WINDOWS, pi_m=2 * pm),
                                          **REDSHIFT))):
        w = pe.window_evidences(c["segs"], c["bank"], n_window=N_WINDOW, pi_m=None if pi_m is None else 2 * pm, chunk=3)
        assert set(w) == {"ln_window_H", "ln_window_L", "wmax_H", "wmax_L", "rho_edges", "rows"}
        assert w["ln_window_H"].shape == (4, N_WINDOW, NR) and np.array_equal(w["rows"], res["rows"])
        p = np.full(MASSES.size, 1.0 / MASSES.size) if pi_m is None else pm
        for I in "HL":
            assert np.array_equal(w["ln_window_" + I], res["ln_window_" + I])
            assert np.array_equal(w["wmax_" + I], res["wmax_" + I]) and np.array_equal(w["rho_edges"], res["rho_edges"])
            mean = logsumexp(res["ln_window_" + I], axis=1) - np.log(N_WINDOW)
            whole = logsumexp(res["ln_glitch_" + I] + np.log(p)[None, :, None], axis=1)
            err = (np.abs(mean - whole) / np.maximum(1.0, np.abs(whole))).max()
            lines.append("pi_m %s, %s: window_evidences = pe_redshift bit for bit; largest ln E_w %.1f; mean over the "
                         "windows against ln_glitch %.1e" % (pi_m, I, res["ln_window_" + I].max(), err))
            assert err < 1e-12
    print("\n" + "\n".join(lines))


# ----------------------------------------------------------------------------------------- outputs


def test_outputs(data, runs):
    """model='windows' adds ln_B, ln_c_I, state_I, ln_window_I and its settings, drops ln_lambda_g*, and leaves every
    output of single=True unchanged; pe_intrinsic returns the same arrays with ln_B distance integrated, and its
    per-segment arrays equal those of pe_redshift"""
    sgl, res = runs["single"], runs["windows"]
    new = {"ln_B", "ln_c_H", "ln_c_L", "state_H", "state_L", "ln_window_H", "ln_window_L", "glitch_model",
           "glitch_n_window", "glitch_zone"}
    assert set(res) == set(sgl) | new
    for k in sgl:
        assert np.array_equal(res[k], sgl[k]), k
    assert res["ln_B"].shape == res["ln_lambda"].shape and res["ln_c_L"].shape == (4,)
    assert res["state_H"].shape == (4, 2) and res["state_L"].shape == (4, 3)
    assert res["ln_window_L"].shape == (4, N_WINDOW, NR)
    assert (res["glitch_model"], res["glitch_n_window"], res["glitch_zone"]) == ("windows", N_WINDOW, ZONE)
    assert np.allclose(res["state_H"].sum(axis=1), 1.0, rtol=1e-12, atol=0)
    ri = pe.pe_intrinsic(data["segs"], data["bank"], data["store"], n_loud=0, glitch=WINDOWS)
    assert ri["ln_B"].shape == ri["ln_lambda"].shape == (4, MASSES.size)
    assert np.isfinite(ri["ln_B"]).all()
    for k in ("ln_c_H", "ln_c_L", "state_H", "state_L", "ln_window_H", "ln_window_L", "ln_glitch_H", "wmax_L"):
        assert np.array_equal(ri[k], res[k]), k
    assert not any(k.startswith("ln_lambda_g") for k in ri)


# ----------------------------------------------------------------------------------------- e. unit mean


E_S = np.array([0.5, 2.0])               # signal powers of the nodes of the incoherent store
E_TAU = np.arange(-2, 3)                 # its lags, all within the light travel time (2 ms)
E_H = 0.08                               # spacing of its beta and Delta grids
E_MODEL = dict(f=np.array([0.9, 0.1]), q=np.array([0.02, 0.2]), pi_k=np.stack([law(4.0, 4.0), law(2.0, 5.0)]))
E_N, E_CHUNK, E_SEED = 1000, 200, 20261114


@functools.lru_cache(maxsize=None)
def incoherent_store(s_nodes):
    """the redshift store of an incoherent merger: at node s, lnS(tau, phi, beta, Delta) = ln(1/5) + ln Phi(w_L; s) +
    ln Phi(w_H; s) with w_L = e^beta / s and w_H = e^(beta + 2 Delta) / s (beta = ln a + 2 ln q_L, Delta = ln q_H - ln
    q_L, a = s/u), so that E_0 S = p(tau) and the H (L) data integrated out leave p(tau) Phi(w_L) (p(tau) Phi(w_H)), the
    one-detector columns of single.py.  beta from ln s - 9 to ln s + 7 and Delta from -7 to 7, both in steps of E_H, so
    that beta + 2 Delta falls on a grid of the same step; the tables pass through store.unbias (the mean error of the
    bilinear read removed).  Each node has its own beta grid, so the gather must read one node per launch"""
    d = np.arange(-7.0, 7.0 + 1e-9, E_H)
    tables, betas = [], []
    E_S = np.asarray(s_nodes, float)
    for s in E_S:
        b = np.log(s) + np.arange(-9.0, 7.0 + 1e-9, E_H)
        x = b[0] + 2.0 * d[0] + E_H * np.arange(b.size + 2 * d.size - 2)
        lnPhi = lambda y: single._ln_phi(np.exp(0.5 * (y - np.log(s))), np.full(y.size, s))
        T2 = lnPhi(b)[:, None] + lnPhi(x)[np.arange(b.size)[:, None] + 2 * np.arange(d.size)[None, :]]
        T = np.broadcast_to(T2 - np.log(E_TAU.size), (E_TAU.size, 49, b.size, d.size))
        tables.append(unbias(np.ascontiguousarray(T, np.float32)))
        betas.append(b)
    n = E_S.size
    return RedshiftStore(s=E_S, alpha_lo=np.full(n, 1e-6), alpha_hi=np.full(n, 1e6), beta=betas, delta=[d] * n,
                         tau_idx=E_TAU, tables=tables, n_max=120)


def white_segment(seed, u, n_time, model):
    """the matched-filter outputs of one segment: white complex Gaussian z_I[m, t] of power u_m, independent over
    templates, times and detectors (noise), and the same with the glitches of the model added per window (state s ~ f,
    a glitch with probability q_s in each window, at a uniform time of the window, in a template of uniform weight, of
    SNR rho uniform in an amplitude bin drawn from pi_k[s], random phase).  returns (glitched, noise) trajectories
    as bank.trajectories gives them, and the number of glitches"""
    rng = np.random.default_rng(seed)
    nm, nw = u.size, N_WINDOW
    nwin = n_time // nw
    out = ({}, {})
    count = 0
    for I in "HL":
        z = (rng.standard_normal((nm, n_time)) + 1j * rng.standard_normal((nm, n_time))) / np.sqrt(2.0)
        zg = z.copy()
        s = int(rng.random() < model["f"][1])
        has = rng.random(nw) < model["q"][s]
        for w in np.nonzero(has)[0]:
            k = rng.choice(NR, p=model["pi_k"][s])
            rho = rng.uniform(single.RHO_EDGES[k], single.RHO_EDGES[k + 1])
            zg[rng.integers(nm), w * nwin + rng.integers(nwin)] += np.sqrt(0.5 * rho * rho) * np.exp(2j * np.pi * rng.random())
        count += has.sum()
        for o, v in zip(out, (zg, z)):
            v = v * np.sqrt(u)[:, None]
            o["lq" + I] = np.log(np.abs(v)).astype(np.float32)
            o["ph" + I] = np.angle(v).astype(np.float32)
    return out[0], out[1], count


@pytest.fixture(scope="module")
def unit(data):
    """E_N segments of white matched-filter outputs with glitches drawn from E_MODEL and the same noise without them,
    analysed at every (node, template) of the incoherent store with E_MODEL in both detectors ('B', and 'B noise' on
    the noise alone), and on the glitched data with wrong settings: q / 10, the laws of the two states swapped; and
    B without the term in which both detectors are near a glitch ('both near dropped', B - c0 from ln c0 recorded
    during the analysis).  Per segment the population average (the mean over the 2 x 3 nodes and templates) of B
    and of Lambda_S"""
    bank, store = data["bank"], incoherent_store(tuple(E_S))
    nt = bank.n_time
    segs = dict(data["segs"], d_H=np.zeros((E_CHUNK, data["segs"]["nu"].size), complex))
    segs["d_L"] = segs["d_H"]
    seg = read_segments(segs)
    geometry = pe._node_geometry(bank, store, 0.1)
    assert geometry[0].all()
    groups = [[k] for k in range(store.s.size)]                     # one node per launch: own beta grids
    sd = pe._single_setup(bank, seg, store, geometry)
    gl = lambda p: pe._glitch_setup(dict(model="windows", H=p, L=p), bank, seg, store)
    passes = (("B", 0, gl(E_MODEL)), ("B noise", 1, gl(E_MODEL)), ("q / 10", 0, gl(dict(E_MODEL, q=E_MODEL["q"] / 10))),
              ("laws swapped", 0, gl(dict(E_MODEL, pi_k=E_MODEL["pi_k"][::-1]))))
    out = {k: [] for k in [p[0] for p in passes] + ["Lambda glitched", "Lambda noise", "glitches", "lc0"]}
    seeds = np.random.SeedSequence(E_SEED).spawn(E_N)
    real_filter, real_time_weights = pe._filter, pe._time_weights
    lc0 = []

    def record(*args):
        w = real_time_weights(*args)
        lc0.append(w["lc0"])
        return w

    try:
        for a in range(0, E_N, E_CHUNK):
            trajs = [white_segment(seeds[a + i], bank.u, nt, E_MODEL) for i in range(E_CHUNK)]
            out["glitches"] += [t[2] for t in trajs]
            for key, which, gd in passes:
                pe._filter = lambda bank_, seg_, e, which=which: trajs[e][which]
                pe._time_weights = record if key == "B" else real_time_weights
                lamN, one = pe._node_values(bank, seg, range(E_CHUNK), store, geometry, groups, 256, False, 0, 0, 0, sd,
                                            gd)
                out[key].append(np.exp(one["lamN_gHL"][:, :-1]).mean(axis=(1, 2)))
                if key in ("B", "B noise"):
                    out["Lambda glitched" if key == "B" else "Lambda noise"].append(np.exp(lamN[:, :-1]).mean(axis=(1, 2)))
    finally:
        pe._filter, pe._time_weights = real_filter, real_time_weights
    res = {k: np.concatenate(v) for k, v in out.items() if k not in ("glitches", "lc0")}
    res["glitches"] = np.array(out["glitches"])
    res["both near dropped"] = res["B"] - np.exp(np.array(lc0))
    assert res["B"].size == E_N == len(lc0)
    return res


def stats(x):
    return x.mean(), x.std() / np.sqrt(x.size)


def test_unit_mean(unit):
    """E_N segments drawn from the windows model (white matched-filter outputs, f = (0.9, 0.1), q = (0.02, 0.2), laws
    rho^-4 above 4 and rho^-2 above 5), analysed with the same model: the population average of B has mean 1 within 3
    standard errors, raw and against the control variate Lambda_S of the same noise without the glitches (whose mean
    is 1 for the store, also checked to 3 standard errors); the control variate has a standard error below 1e-3.
    Printed: Lambda_S of the glitched data (the Gaussian likelihood), and B of the glitch model on the pure noise,
    whose mean is below 1 (the weights hide the share of a merger near where glitches could be)"""
    B, L0, Lg = unit["B"], unit["Lambda noise"], unit["Lambda glitched"]
    mB, sB = stats(B)
    mD, sD = stats(B - L0)
    m0, s0 = stats(L0)
    mN, sN = stats(unit["B noise"])
    mg = Lg.mean()
    print("\n%d segments, %.2f glitches per segment-pair: mean B %.5f +- %.5f (%+.1f se); B - Lambda_S(noise) %.6f +- "
          "%.6f (%+.1f se); Lambda_S(noise) %.5f +- %.5f (%+.1f se)\n  Lambda_S of the glitched data: mean %.3g, "
          "median %.4f; B of the glitch model on the pure noise %.5f +- %.5f (%+.1f se)"
          % (B.size, unit["glitches"].mean(), mB, sB, (mB - 1) / sB, mD, sD, mD / sD, m0, s0, (m0 - 1) / s0, mg,
             np.median(Lg), mN, sN, (mN - 1) / sN))
    assert abs(mB - 1.0) < 3.0 * sB and abs(mD) < 3.0 * sD and abs(m0 - 1.0) < 3.0 * s0 and sD < 1e-3


@pytest.mark.parametrize("name", ["q / 10", "laws swapped", "both near dropped"])
def test_unit_mean_detects_errors(unit, name):
    """control of test_unit_mean: on the same segments a wrong analysis moves the mean of B - Lambda_S(noise) from 0
    by more than 5 standard errors: q ten times too small, the laws of the two states swapped, or the term of B with
    both detectors near a glitch left out.  (Errors that amount to another valid merger model leave E[B] = 1 and are
    not seen here: the zone, or the weights of each window given to the window before, which is the zone {w, w + 1,
    w + 2}; measured +0.2 se.  test_reference sees those)"""
    m, s = stats(unit[name] - unit["Lambda noise"])
    print("\n%s: B - Lambda_S(noise) %.5f +- %.5f (%+.1f se)" % (name, m, s, m / s))
    assert abs(m) > 5.0 * s


# ----------------------------------------------------------------------------------------- f. settings


BAD = {
    "unknown model": (dict(model="many"), "unknown glitch model"),
    "unknown key": (dict(WINDOWS, window=0.1), "unknown glitch settings"),
    "H missing": (dict(model="windows", L=P_L), "needs glitch setting H"),
    "state key": (dict(WINDOWS, H=dict(P_H, eps=0.1)), "exactly the keys"),
    "f 2-d": (dict(WINDOWS, H=dict(P_H, f=[[0.85, 0.15]])), "one-dimensional"),
    "f ragged": (dict(WINDOWS, H=dict(P_H, f=[[0.85], 0.15])), "H.f is not an array of numbers"),
    "f sum": (dict(WINDOWS, H=dict(P_H, f=[0.8, 0.15])), "H.f must sum to 1"),
    "f negative": (dict(WINDOWS, H=dict(P_H, f=[1.1, -0.1])), "finite and >= 0"),
    "q length": (dict(WINDOWS, H=dict(P_H, q=[0.01, 0.1, 0.2])), "H.q has shape"),
    "q one": (dict(WINDOWS, L=dict(P_L, q=[0.0, 0.02, 1.0])), r"in \[0, 1\)"),
    "q nan": (dict(WINDOWS, L=dict(P_L, q=[0.0, np.nan, 0.2])), "finite"),
    "pi_k shape": (dict(WINDOWS, H=dict(P_H, pi_k=P_H["pi_k"][:, :-1])), "H.pi_k has shape"),
    "pi_k rows": (dict(WINDOWS, L=dict(P_L, pi_k=P_L["pi_k"][:2])), "L.pi_k has shape"),
    "pi_k sum": (dict(WINDOWS, H=dict(P_H, pi_k=P_H["pi_k"] * 1.001)), "H.pi_k must sum to 1"),
    "n_window divisor": (dict(WINDOWS, n_window=48), "must divide n_time"),
    "n_window float": (dict(WINDOWS, n_window=64.0), "must be an integer"),
    "zone": (dict(WINDOWS, n_window=2), "does not fit"),
    "zone negative": (dict(WINDOWS, zone=-1), "zone must be an integer >= 0"),
    "windows pi_m length": (dict(WINDOWS, pi_m=[0.5, 0.5]), "pi_m has shape"),
    "windows pi_m zero": (dict(WINDOWS, pi_m=[0.0, 0.0, 0.0]), "positive sum"),
    "pi_m length": (dict(pi_m=[0.5, 0.5]), "pi_m has shape"),
    "pi_m length 1": (dict(pi_m=[1.0]), "pi_m has shape"),          # broadcast silently before
    "pi_m 2-d": (dict(pi_m=[[0.2, 0.3, 0.5]]), "pi_m has shape"),
    "pi_m negative": (dict(pi_m=[0.5, 0.6, -0.1]), "finite and >= 0"),
    "pi_m zero": (dict(pi_m=[0.0, 0.0, 0.0]), "positive sum"),
    "pi_k length": (dict(pi_k=np.ones(NR - 1)), "pi_k has shape"),
    "pi_k negative": (dict(pi_k=-law(4.0, 4.0)), "finite and >= 0"),
    "window nan": (dict(window=np.nan), "window must be finite"),
    "unknown old key": (dict(eps=0.1), "unknown glitch settings"),
    "not a dict": ("windows", "must be None, False, True or a dict"),
}


@pytest.mark.parametrize("name", list(BAD))
def test_bad_settings(data, name):
    """malformed glitch settings raise ValueError with a message that names the setting, in pe_redshift (before any
    work) and in window_evidences (n_window, pi_m)"""
    glitch, msg = BAD[name]
    with pytest.raises(ValueError, match=msg):
        pe.pe_redshift(data["segs"], data["bank"], data["store"], rows=[1], n_loud=0, glitch=glitch, **REDSHIFT)
    if name in ("n_window divisor", "n_window float", "windows pi_m length", "windows pi_m zero"):
        with pytest.raises(ValueError, match=msg):
            pe.window_evidences(data["segs"], data["bank"], n_window=glitch["n_window"], pi_m=glitch.get("pi_m"))


def test_settings_normalized(data):
    """f, the rows of pi_k and the windows pi_m are divided by their sums (within 1e-6 of 1 for f and pi_k, any
    positive sum for pi_m): settings off by 1e-7 give the B of the normalized ones to 1e-9; pi_m given as 2 pi_m
    gives the same arrays"""
    c = data
    off = dict(WINDOWS, H=dict(P_H, f=np.array(P_H["f"]) * (1 + 1e-7)), L=dict(P_L, pi_k=P_L["pi_k"] * (1 - 1e-7)))
    a = pe.pe_redshift(c["segs"], c["bank"], c["store"], rows=[0], n_loud=0, glitch=WINDOWS, **REDSHIFT)
    b = pe.pe_redshift(c["segs"], c["bank"], c["store"], rows=[0], n_loud=0, glitch=off, **REDSHIFT)
    fin = np.isfinite(a["ln_B"])
    assert np.abs(a["ln_B"][fin] - b["ln_B"][fin]).max() < 1e-9
    pm = np.array([0.2, 0.5, 0.3])
    x = pe.pe_redshift(c["segs"], c["bank"], c["store"], rows=[3], n_loud=0, glitch=dict(WINDOWS, pi_m=pm), **REDSHIFT)
    y = pe.pe_redshift(c["segs"], c["bank"], c["store"], rows=[3], n_loud=0, glitch=dict(WINDOWS, pi_m=7 * pm),
                       **REDSHIFT)
    for k in ("ln_B", "ln_c_L", "ln_window_L"):
        assert np.array_equal(x[k], y[k]), k


@pytest.fixture(scope="module")
def pe_head(tmp_path_factory):
    """pe.py as committed at HEAD (before model='windows'), imported as a module of exmarg.search"""
    try:
        src = subprocess.run(["git", "show", HEAD + ":src/exmarg/search/pe.py"], cwd=ROOT, capture_output=True,
                             check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        pytest.skip("no git history to compare with (%s)" % err)
    path = tmp_path_factory.mktemp("head") / "pe_head.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("exmarg.search._pe_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "windows" not in src
    return mod


IDENTITY = [None, True, dict(window=0.05), dict(pi_k=2.0 * law(3.0, 5.0), pi_m=[0.2, 0.3, 0.5], window=0.12),
            dict(window=-1.0)]
NAMED = dict(model=None, window=0.05)                # the default model, named: the arrays of dict(window=0.05)


@pytest.mark.parametrize("which", range(len(IDENTITY)))
def test_default_path_unchanged(data, pe_head, which):
    """glitch None / True / dict without model: every output of pe_redshift and pe_intrinsic is the array of the
    committed pe.py bit for bit (same keys), with every arrival time and with the default selection of arrival
    times, single=True and False; an unnormalized pi_k is used as given (kbar does not depend on its scale)"""
    glitch = IDENTITY[which]
    args = (data["segs"], data["bank"], data["store"])
    n = 0
    for name, kw in (("pe_redshift", REDSHIFT), ("pe_intrinsic", {})):
        for extra in (dict(n_loud=0), dict(seed=3)):
            for sgl in ((False, True) if glitch is None else (False,)):
                a = getattr(pe, name)(*args, glitch=glitch, single=sgl, **extra, **kw)
                b = getattr(pe_head, name)(*args, glitch=glitch, single=sgl, **extra, **kw)
                assert set(a) == set(b)
                for k in a:
                    assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), (name, k)
                n += len(a)
    print("\nglitch setting %d: %d arrays bit for bit equal to the committed pe.py" % (which, n))


def test_default_model_named(data):
    """model=None is the default model: the arrays of the same settings without it"""
    args = (data["segs"], data["bank"], data["store"])
    a = pe.pe_redshift(*args, rows=[2], n_loud=0, glitch=NAMED, **REDSHIFT)
    b = pe.pe_redshift(*args, rows=[2], n_loud=0, glitch=dict(window=0.05), **REDSHIFT)
    assert set(a) == set(b) and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)
