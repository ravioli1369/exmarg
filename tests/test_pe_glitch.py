"""tests of the merger likelihood ratios given a glitch in pe_redshift and pe_intrinsic (docs/glitch_likelihood.md,
Sec. 4 and steps 3 to 5 of Sec. 5), on the CPU with a small synthetic store built here: smooth random tables on the
53 nodes of data/tables/s_grid.npz that grow like exp(beta) / 2 in beta = ln a + 2 ln q_L (as the real tables do,
so that a loud glitch in L makes lnS large at its time), with the lags -6 to 14 (asymmetric, so that the sign of
tau in the delay averages matters; the light travel time keeps -6 to 10 of them), and three segments of 4 s up to
128 Hz (4096 arrival times): a loud transient in H only, a louder one in L only (ln l_g up to about 300) and an
extreme one in L only (ln l_g about 1600, where kbar, Phi and the products of the near terms leave the range of
float64).  The expected values come from the construction (kbar = 1, kbar = 0) or from a reference written here in
the log domain: the table read of the gather.py docstring as a float64 loop over the lags, the glitch-time weights
as log-sum-exps over the arrival times inside and outside each window, the delay averages and the one-detector sums
as explicit log-sum-exps over the lags and the arrival times, with Phi(w; s) evaluated at each w (single._ln_phi,
validated in test_single.py)."""

from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from exmarg.detectors import c as C_LIGHT, x as SITES
from exmarg.pp import simulate
from exmarg.search import mass_bank, pe, read_segments, single
from exmarg.search.gather import LK_MIN
from exmarg.search.segments import D0
from exmarg.search.store import RedshiftStore

NPHI = 96
RTOL = 1e-6
S_NODES = np.load(Path(__file__).resolve().parents[1] / "data" / "tables" / "s_grid.npz")["s_nodes"]
REDSHIFT = dict(nz=40, nm1=30, m1_range=(10.0, 60.0))


@pytest.fixture(scope="module")
def case():
    """store, segments, bank and what _node_values needs, every arrival time summed (n_loud = 0)"""
    rng = np.random.default_rng(4)
    tau = np.arange(-6, 15)
    nb, nd = 12, 10
    beta, delta = np.linspace(-4.0, 10.0, nb), np.linspace(-3.0, 3.0, nd)
    ph = np.cos(2 * np.pi * np.arange(NPHI // 2 + 1) / NPHI)[None, :, None, None]
    xb = np.linspace(0, 1, nb)[None, None, :, None]
    xd = np.linspace(-1, 1, nd)[None, None, None, :]
    xt = np.linspace(-1, 1, tau.size)[:, None, None, None]
    grow = 0.5 * np.exp(beta)[None, None, :, None]
    tables = []
    for _ in S_NODES:
        k = rng.normal(size=5)
        T = (-2.0 + k[0] + grow + k[1] * ph * xb - xd ** 2 + 0.3 * k[2] * xd * xb + 0.2 * k[3] * xt
             + 0.1 * k[4] * xt * ph)
        tables.append(np.ascontiguousarray(np.broadcast_to(T, (tau.size, NPHI // 2 + 1, nb, nd)), np.float32))
    n = S_NODES.size
    store = RedshiftStore(s=S_NODES, alpha_lo=np.full(n, 0.16), alpha_hi=np.full(n, 4e4), beta=[beta] * n,
                          delta=[delta] * n, tau_idx=tau, tables=tables, n_max=120)
    y = iter([12.0, 18.0, 40.0])
    segs, _, _ = simulate(3, 21, lambda r: (60.0, 1.0, 0.0), lambda r: next(y), 4.0, 20.0, 128.0)
    noise = lambda: np.sqrt(segs["sig2"] / 2) * (rng.standard_normal(segs["sig2"].size)
                                                 + 1j * rng.standard_normal(segs["sig2"].size))
    segs["d_L"][0], segs["d_H"][1], segs["d_H"][2] = noise(), noise(), noise()    # transients in H alone, L alone
    bank = mass_bank(np.array([40.0, 60.0, 90.0]), segs)
    seg = read_segments(segs)
    geometry = pe._node_geometry(bank, store, 0.1267)
    groups = [list(range(g, min(g + 6, S_NODES.size))) for g in range(0, S_NODES.size, 6)]
    return dict(store=store, segs=segs, bank=bank, seg=seg, geometry=geometry, groups=groups)


def node_values(c, glitch=True):
    """_node_values of every segment with the glitch terms: lamN and the dict of _single_nodes"""
    sd = pe._single_setup(c["bank"], c["seg"], c["store"], c["geometry"])
    gd = pe._glitch_setup(glitch, c["bank"], c["seg"], c["store"])
    return pe._node_values(c["bank"], c["seg"], range(c["seg"]["rows"].size), c["store"], c["geometry"], c["groups"],
                           256, False, 0, 0, 0, sd, gd)


def ln_S(Tr, lna, lqH, phH, lqL, phL, beta, delta):
    """lnS of one lag at each arrival time: bilinear in (beta, Delta) with beta clamped to the table, ln q_L moved
    with it and ln q_H kept, linear in the phase difference on the full circle (the table is even in it)"""
    db, dd = beta[1] - beta[0], delta[1] - delta[0]
    b = np.clip(lna + 2.0 * lqL, beta[0], beta[-1])
    xb = (b - beta[0]) / db
    xd = np.clip((lqH - (b - lna) / 2.0 - delta[0]) / dd, 0.0, delta.size - 1.0)
    ib, jd = np.minimum(xb.astype(int), beta.size - 2), np.minimum(xd.astype(int), delta.size - 2)
    fb, fd = xb - ib, xd - jd
    x = np.mod((phH - phL) * NPHI / (2 * np.pi), NPHI)
    c0 = np.floor(x).astype(int)
    out = 0.0
    for ci, wc in ((c0 % NPHI, 1.0 - (x - c0)), ((c0 + 1) % NPHI, x - c0)):
        ci = np.where(ci > NPHI // 2, NPHI - ci, ci)
        out = out + wc * ((1 - fb) * (1 - fd) * Tr[ci, ib, jd] + (1 - fb) * fd * Tr[ci, ib, jd + 1]
                          + fb * (1 - fd) * Tr[ci, ib + 1, jd] + fb * fd * Tr[ci, ib + 1, jd + 1])
    return out


def reference(c, e, pairs=None, kbar=None, floor=LK_MIN):
    """every term of Sec. 4 for segment e in the log domain, at the (node, template) pairs given (node n_node: the
    boundary point) or at every pair of the node geometry.  kbar: {I: (ln kbar_I, ln(1 - kbar_I))} to use instead of
    the glitch-time weights of the fiducial law; floor: the floor of ln kbar in the far sums (that of the gather, or
    the old -80).  returns the per-segment arrays (ln kbar, ln(1 - kbar), ln o_H, ln o_L, ln c0, ln l_g) and, per
    term, (n_node + 1, n_template) arrays of ln values, -inf off the pairs"""
    bank, store, seg = c["bank"], c["store"], c["seg"]
    valid, _, _, bnode, blna = c["geometry"]
    nn, nm = store.s.size, bank.M.size
    traj = bank.trajectories({I: seg["d_" + I][e] for I in "HL"}, {I: seg["sig2_" + I] for I in "HL"})
    q = {key: np.asarray(v, np.float64) for key, v in traj.items()}
    n = q["lqH"].shape[1]
    dt = seg["T"] / n
    ln_u = np.log(bank.u)
    lags = store.tau_idx[np.abs(store.tau_idx) * dt <= np.linalg.norm(SITES["H"] - SITES["L"]) / C_LIGHT]
    a, b = np.maximum(single.RHO_EDGES[:-1], 4.0), single.RHO_EDGES[1:]
    pk = np.where(b > a, (np.minimum(a, b) ** -3 - b ** -3) / 3.0, 0.0)
    pk /= pk.sum()
    hw = int(round(0.1 / dt))
    t = np.arange(n)
    lk, ln, lg = {}, {}, {}
    for I in "HL":
        lg[I] = single.glitch_time_series(traj["lq" + I], ln_u, pk)
        if kbar is not None and I in kbar:
            lk[I], ln[I] = (np.asarray(x, float) for x in kbar[I])
            continue
        lk[I], ln[I] = np.empty(n), np.empty(n)
        for a0 in range(0, n, 512):
            d = np.arange(a0, min(a0 + 512, n))[:, None]
            lk[I][a0:a0 + 512] = logsumexp(lg[I][(d + hw + 1 + np.arange(n - 2 * hw - 1)) % n], axis=1)
            ln[I][a0:a0 + 512] = logsumexp(lg[I][(d - hw + np.arange(2 * hw + 1)) % n], axis=1)
        lk[I] -= logsumexp(lg[I])
        ln[I] -= logsumexp(lg[I])
    oH = logsumexp(ln["H"][(t[:, None] + lags[None, :]) % n], axis=1) - np.log(lags.size)
    oL = logsumexp(ln["L"][(t[:, None] - lags[None, :]) % n], axis=1) - np.log(lags.size)
    lc0 = logsumexp(oH + ln["L"]) - np.log(n)
    wH, wL = np.maximum(lk["H"], floor), np.maximum(lk["L"], floor)
    if pairs is None:
        pairs = [(k, m) for k in range(nn) for m in range(nm) if valid[k, m]] + [(nn, m) for m in range(nm)
                                                                                 if bnode[m] >= 0]
    keys = ("ln_lam", "far_H", "far_L", "far_HL", "near_H", "near_L", "near_H_farL", "near_L_farH", "gH", "gL", "gHL")
    res = {key: np.full((nn + 1, nm), -np.inf) for key in keys}
    for k, m in pairs:
        node, lna = (k, np.log(store.s[k] / bank.u[m])) if k < nn else (bnode[m], blna[m])
        Tk = np.asarray(store.tables[node], np.float64)
        terms = [[], [], [], []]
        for r, tau in enumerate(store.tau_idx):
            i1 = (t + tau) % n
            S = ln_S(Tk[r], lna, q["lqH"][m, i1], q["phH"][m, i1], q["lqL"][m], q["phL"][m],
                     np.asarray(store.beta[node]), np.asarray(store.delta[node]))
            for v, w in enumerate((0.0, wH[i1], wL, wH[i1] + wL)):
                terms[v].append(S + w)
        for key, tv in zip(("ln_lam", "far_H", "far_L", "far_HL"), terms):
            res[key][k, m] = logsumexp(np.concatenate(tv)) - np.log(n)
        s_km = np.exp(lna) * bank.u[m]
        phi = {I: single._ln_phi(np.exp(q["lq" + I][m] - 0.5 * ln_u[m]), np.full(n, s_km)) for I in "HL"}
        for key, I, lw in (("near_H", "L", oH), ("near_L", "H", oL), ("near_H_farL", "L", oH + lk["L"]),
                           ("near_L_farH", "H", oL + lk["H"])):
            res[key][k, m] = logsumexp(phi[I] + lw) - np.log(n)
        res["gH"][k, m] = np.logaddexp(res["far_H"][k, m], res["near_H"][k, m])
        res["gL"][k, m] = np.logaddexp(res["far_L"][k, m], res["near_L"][k, m])
        res["gHL"][k, m] = logsumexp([res["far_HL"][k, m], res["near_H_farL"][k, m], res["near_L_farH"][k, m], lc0])
    res.update(lk=lk, ln=ln, oH=oH, oL=oL, lc0=lc0, lg=lg, traj=traj, phi=phi)
    return res


def pair(c, e):
    """a (node, template) pair of segment e: the injected mass at the valid node nearest s = 1"""
    m = 1
    ks = np.nonzero(c["geometry"][0][:, m])[0]
    return ks[np.argmin(np.abs(np.log(c["store"].s[ks])))], m


def _close(got, ref, rtol=RTOL):
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=rtol)


def test_kbar_one(case):
    """an empty window (kbar = 1): lambda_H = lambda_L = lambda_HL = Lambda_S, through both reductions"""
    for f, kw in ((pe.pe_intrinsic, {}), (pe.pe_redshift, REDSHIFT)):
        r = f(case["segs"], case["bank"], case["store"], n_loud=0, glitch=dict(window=-1.0), **kw)
        fin = np.isfinite(r["ln_lambda"])
        assert fin.any()
        for X in ("H", "L", "HL"):
            assert np.array_equal(np.isfinite(r["ln_lambda_g" + X]), fin)
            _close(r["ln_lambda_g" + X][fin], r["ln_lambda"][fin], 1e-12)
        assert r["glitch_window"] == -1.0 and np.allclose(r["glitch_pi_k"], single.amplitude_weights())


@pytest.mark.parametrize("zero", ["H", "L"])
def test_kbar_zero(case, monkeypatch, zero):
    """kbar of one detector 0 everywhere: lambda of that detector is the other detector's Lambda_S^I (node values and
    pe_intrinsic), and the other two lambda are the far and near terms computed by hand"""
    kbar = pe._kbar
    monkeypatch.setattr(pe, "_kbar", lambda traj, I, gd, sd, xp: (xp.full(traj["lqH"].shape[1], -np.inf),
                                                                  xp.zeros(traj["lqH"].shape[1])) if I == zero
                        else kbar(traj, I, gd, sd, xp))           # ln kbar = -inf, ln(1 - kbar) = 0
    other = "L" if zero == "H" else "H"
    lamN, one = node_values(case)
    fin = np.isfinite(lamN)
    _close(one["lamN_g" + zero][fin], one["lamN_" + other][fin], 1e-12)
    r = pe.pe_intrinsic(case["segs"], case["bank"], case["store"], n_loud=0, glitch=True)
    _close(r["ln_lambda_g" + zero], r["ln_lambda_" + other], 1e-12)
    for e in range(3):
        k, m = pair(case, e)
        h = reference(case, e, [(k, m)], kbar={zero: (np.full(case["bank"].n_time, -np.inf),
                                                      np.zeros(case["bank"].n_time))})
        _close(one["lamN_g" + other][e, k, m], h["g" + other][k, m])
        _close(one["lamN_gHL"][e, k, m], h["gHL"][k, m])


def test_by_hand(case):
    """every term at one (node, template) pair of each segment against the reference written here"""
    lamN, one = node_values(case)
    sd = pe._single_setup(case["bank"], case["seg"], case["store"], case["geometry"])
    gd = pe._glitch_setup(True, case["bank"], case["seg"], case["store"])
    nr = single.RHO_EDGES.size - 1
    for e in range(3):
        k, m = pair(case, e)
        h = reference(case, e, [(k, m)])
        I, J = "HLL"[e], "LHH"[e]
        assert h["lk"][I].min() < -100.0 < -1.0 < h["lk"][J].min()                    # a glitch in one detector only
        gw = pe._glitch_weights(h["traj"], gd, sd, np)
        for i, X in enumerate("HL"):
            np.testing.assert_allclose(gw["tw"][i], h["lk"][X], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(gw["L"][0], h["oH"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(gw["H"][0], h["oL"], rtol=1e-12, atol=1e-12)
        _close(gw["lc0"], h["lc0"], 1e-12)
        sv = pe._single_values(h["traj"], sd, np, gw)
        near = dict(near_H=sv["L"][0][1, m, nr + k], near_L=sv["H"][0][1, m, nr + k],
                    near_H_farL=sv["L"][0][2, m, nr + k], near_L_farH=sv["H"][0][2, m, nr + k])
        for key, v in near.items():
            _close(v, h[key][k, m])
        _close(lamN[e, k, m], h["ln_lam"][k, m])
        for X in ("H", "L", "HL"):
            _close(one["lamN_g" + X][e, k, m], h["g" + X][k, m])
        print("segment %d, node %d, template %d: |d ln| of Lambda_S %.1e, lambda_H %.1e, lambda_L %.1e, lambda_HL %.1e,"
              " near terms %.1e; ln lambda - ln Lambda_S %.2f %.2f %.2f"
              % (e, k, m, abs(lamN[e, k, m] - h["ln_lam"][k, m]),
                 *(abs(one["lamN_g" + X][e, k, m] - h["g" + X][k, m]) for X in ("H", "L", "HL")),
                 max(abs(v - h[key][k, m]) for key, v in near.items()),
                 *(h["g" + X][k, m] - h["ln_lam"][k, m] for X in ("H", "L", "HL"))))
        assert abs(h["gH"][k, m] - h["ln_lam"][k, m]) > 1e-3 and abs(h["gL"][k, m] - h["ln_lam"][k, m]) > 1e-3

    lnW = np.full(case["geometry"][2].shape, -np.inf)
    yN = case["geometry"][2]
    for m in range(case["bank"].M.size):
        pts = np.nonzero(yN[:, m] > 0)[0]
        order = pts[np.argsort(yN[pts, m])]
        lnW[order, m] = pe.distance_weights(yN[order, m], "volume", 0.1267)
    r = pe.pe_intrinsic(case["segs"], case["bank"], case["store"], n_loud=0, glitch=True)
    for X in ("H", "L", "HL"):
        ref = [pe._distance_integral(lnW, one["lamN_g" + X][e]) for e in range(3)]
        np.testing.assert_array_equal(r["ln_lambda_g" + X], ref)


def test_loud_glitch(case):
    """a glitch in L alone with ln l_g above 200 (segment 1): ln_lambda_gL of pe_redshift against the reference in
    the log domain at every node and boundary point, reduced the same way; and far below what the old floor of the
    time weights (ln kbar clipped at -80) gave, ln_lambda - 80 wherever the glitch dominates Lambda_S"""
    bank, store = case["bank"], case["store"]
    r = pe.pe_redshift(case["segs"], bank, store, rows=[1], n_loud=0, glitch=True, **REDSHIFT)
    h = reference(case, 1)
    assert h["lg"]["L"].max() > 200.0 and h["lg"]["H"].max() < 50.0
    ref = pe._to_source_frame(h["gL"], case["geometry"][2], bank.M, D0 / pe._dl_of_z(r["z"]), r["m1"], r["z"], np)
    got = r["ln_lambda_gL"][0]
    fin = np.isfinite(got)
    assert fin.any() and np.array_equal(fin, np.isfinite(ref))
    _close(got[fin], ref[fin])
    k, m = pair(case, 1)
    old = reference(case, 1, [(k, m)], floor=-80.0)
    gap = ((r["ln_lambda"][0][fin] - 80.0) - got[fin]).max()
    print("loud L glitch, ln l_g peak %.0f: max |d ln lambda_L| against the reference %.1e; node %d, template %d: "
          "ln Lambda_S %.1f, ln lambda_L %.2f, with ln kbar clipped at -80 %.2f; on the grid ln Lambda_S - 80 exceeds "
          "ln lambda_L by up to %.1f" % (h["lg"]["L"].max(), np.abs(got[fin] - ref[fin]).max(), k, m, h["ln_lam"][k, m],
                                         h["gL"][k, m], old["gL"][k, m], gap))
    assert abs(old["gL"][k, m] - (old["ln_lam"][k, m] - 80.0)) < 1e-3       # the old floor set the answer
    assert h["gL"][k, m] < h["ln_lam"][k, m] - 80.0 - 10.0
    assert gap > 10.0


def test_extreme_glitch(case):
    """a glitch in L alone with ln l_g about 1600 (segment 2): kbar_L at its time and o_H far from it are far below
    1e-308 and Phi(w_L) at its time far above 1e308, so the one-detector sums need the weights as logarithms; every
    term at nodes where s reaches w of the glitch, against the reference (the near terms to 3e-5 in ln, the bound
    (dv^2 / 8) (ln Phi)'' of the interpolation of the columns in v), and the term near_H_farL far from what weights
    rounded to linear give"""
    lamN, one = node_values(case)
    sd = pe._single_setup(case["bank"], case["seg"], case["store"], case["geometry"])
    gd = pe._glitch_setup(True, case["bank"], case["seg"], case["store"])
    m, nr = 1, single.RHO_EDGES.size - 1
    h = reference(case, 2, [(k, m) for k in (40, 50, 52)])
    assert h["lg"]["L"].max() > 1000.0 and h["lk"]["L"].min() < -745.0 and h["phi"]["L"].max() > 745.0
    sv = pe._single_values(h["traj"], sd, np, pe._glitch_weights(h["traj"], gd, sd, np))
    for k in (40, 50, 52):
        terms = dict(near_H=sv["L"][0][1, m, nr + k], near_L=sv["H"][0][1, m, nr + k],
                     near_H_farL=sv["L"][0][2, m, nr + k], near_L_farH=sv["H"][0][2, m, nr + k])
        for key, v in terms.items():
            np.testing.assert_allclose(v, h[key][k, m], rtol=0, atol=3e-5)
        for X in ("H", "L", "HL"):
            _close(one["lamN_g" + X][2, k, m], h["g" + X][k, m])
    k = 52
    lq = np.asarray(h["traj"]["lqL"][m], float)
    phi = single._ln_phi(np.exp(lq - 0.5 * np.log(case["bank"].u[m])), np.full(lq.size, case["store"].s[k]))
    with np.errstate(divide="ignore"):
        linear = logsumexp(phi, b=np.exp(h["oH"]) * np.exp(h["lk"]["L"])) - np.log(lq.size)
    print("extreme L glitch, ln l_g peak %.0f: node %d (s %.0f), ln near_H_farL %.2f; with the weights rounded to "
          "linear %.2f" % (h["lg"]["L"].max(), k, case["store"].s[k], h["near_H_farL"][k, m], linear))
    assert h["near_H_farL"][k, m] > linear + 10.0


def test_glitch_leaves_the_rest(case):
    """glitch=None changes nothing, and with glitch every output of single=True and of the default path is the same
    array (variant 0 of the weighted read is the unweighted read)"""
    for f, kw in ((pe.pe_intrinsic, {}), (pe.pe_redshift, REDSHIFT)):
        args = (case["segs"], case["bank"], case["store"])
        base, sgl = f(*args, **kw), f(*args, single=True, **kw)
        none, gl = f(*args, glitch=None, **kw), f(*args, single=True, glitch=None, **kw)
        full = f(*args, glitch=dict(window=0.05), **kw)
        assert set(none) == set(base) and set(gl) == set(sgl) and set(sgl) < set(full)
        for k in base:
            assert np.array_equal(none[k], base[k]) and np.array_equal(sgl[k], base[k])
            assert np.array_equal(full[k], base[k])
        for k in sgl:
            assert np.array_equal(gl[k], sgl[k]) and np.array_equal(full[k], sgl[k])
        assert full["glitch_window"] == 0.05
