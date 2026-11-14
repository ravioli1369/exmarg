"""tests of the merger likelihood ratios given a glitch in pe_redshift and pe_intrinsic (docs/glitch_likelihood.md,
Sec. 4 and steps 3 to 5 of Sec. 5), on the CPU with a small synthetic store built here: smooth random tables on the
53 nodes of data/tables/s_grid.npz with the lags -6 to 14 (asymmetric, so that the sign of tau in the delay
averages matters; the light travel time keeps -6 to 10 of them), and two segments of 4 s up to 128 Hz (4096
arrival times), the first with a loud transient in H only, the second in L only.  The expected values come from
the construction (kbar = 1, kbar = 0) or from a reference written here: the table read of the gather.py docstring
as a float64 loop over the lags, the glitch-time weights, the delay averages and the one-detector sums as
explicit sums over the arrival times, with Phi(w; s) evaluated at each w (single._ln_phi, validated in
test_single.py)."""

from pathlib import Path

import numpy as np
import pytest

from exmarg.detectors import c as C_LIGHT, x as SITES
from exmarg.pp import simulate
from exmarg.search import mass_bank, pe, read_segments, single
from exmarg.search.store import RedshiftStore

NPHI = 96
LK_MIN = -80.0
RTOL = 1e-6
S_NODES = np.load(Path(__file__).resolve().parents[1] / "data" / "tables" / "s_grid.npz")["s_nodes"]


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
    tables = []
    for _ in S_NODES:
        k = rng.normal(size=5)
        T = (-2.0 + k[0] + 2.0 * xb + k[1] * ph * xb - xd ** 2 + 0.3 * k[2] * xd * xb + 0.2 * k[3] * xt
             + 0.1 * k[4] * xt * ph)
        tables.append(np.ascontiguousarray(np.broadcast_to(T, (tau.size, NPHI // 2 + 1, nb, nd)), np.float32))
    n = S_NODES.size
    store = RedshiftStore(s=S_NODES, alpha_lo=np.full(n, 0.16), alpha_hi=np.full(n, 4e4), beta=[beta] * n,
                          delta=[delta] * n, tau_idx=tau, tables=tables, n_max=120)
    segs, _, _ = simulate(2, 21, lambda r: (60.0, 1.0, 0.0), lambda r: 12.0, 4.0, 20.0, 128.0)
    noise = lambda: np.sqrt(segs["sig2"] / 2) * (rng.standard_normal(segs["sig2"].size)
                                                 + 1j * rng.standard_normal(segs["sig2"].size))
    segs["d_L"][0], segs["d_H"][1] = noise(), noise()            # a transient in H alone, then in L alone
    bank = mass_bank(np.array([40.0, 60.0, 90.0]), segs)
    seg = read_segments(segs)
    geometry = pe._node_geometry(bank, store, 0.1267)
    groups = [list(range(g, min(g + 6, S_NODES.size))) for g in range(0, S_NODES.size, 6)]
    return dict(store=store, segs=segs, bank=bank, seg=seg, geometry=geometry, groups=groups)


def node_values(c, glitch=True):
    """_node_values of both segments with the glitch terms: lamN and the dict of _single_nodes"""
    sd = pe._single_setup(c["bank"], c["seg"], c["store"], c["geometry"])
    gd = pe._glitch_setup(glitch, c["bank"], c["seg"], c["store"])
    return pe._node_values(c["bank"], c["seg"], range(2), c["store"], c["geometry"], c["groups"], 256, False, 0, 0, 0,
                           sd, gd)


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


def by_hand(c, e, k, m, kbar=None):
    """every term of Sec. 4 at node k and template m of segment e, from the trajectories: returns dict of kbar_H,
    kbar_L, o_H, o_L (n_time,), c0, the sums of S, S kbar_H, S kbar_L, S kbar_H kbar_L over (t, tau) / n_t, the
    near terms and ln Lambda_S, ln lambda_H, ln lambda_L, ln lambda_HL.  kbar: {I: (n_time,)} to use instead of
    the glitch-time weights of the fiducial law"""
    bank, store, seg = c["bank"], c["store"], c["seg"]
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
    kb = {}
    for I in "HL":
        if kbar is not None and I in kbar:
            kb[I] = np.asarray(kbar[I], float)
            continue
        lg = single.glitch_time_series(traj["lq" + I], ln_u, pk)
        p = np.exp(lg - lg.max())
        win = p[(np.arange(n)[:, None] + np.arange(-hw, hw + 1)[None, :]) % n].sum(axis=1)
        kb[I] = np.clip(1.0 - win / p.sum(), 0.0, 1.0)
    t = np.arange(n)
    oH = (1.0 - kb["H"][(t[:, None] + lags[None, :]) % n]).mean(axis=1)
    oL = (1.0 - kb["L"][(t[:, None] - lags[None, :]) % n]).mean(axis=1)
    kH, kL = np.maximum(kb["H"], np.exp(LK_MIN)), np.maximum(kb["L"], np.exp(LK_MIN))
    lna = np.log(store.s[k] / bank.u[m])
    Tk = np.asarray(store.tables[k], np.float64)
    sums = np.zeros(4)
    for r, tau in enumerate(store.tau_idx):
        i1 = (t + tau) % n
        S = np.exp(ln_S(Tk[r], lna, q["lqH"][m, i1], q["phH"][m, i1], q["lqL"][m], q["phL"][m],
                        np.asarray(store.beta[k]), np.asarray(store.delta[k])))
        sums += [S.sum(), (S * kH[i1]).sum(), (S * kL).sum(), (S * kH[i1] * kL).sum()]
    sums /= n
    phi = {I: np.exp(single._ln_phi(np.exp(q["lq" + I][m] - 0.5 * ln_u[m]), np.full(n, store.s[k]))) for I in "HL"}
    near = dict(H=(phi["L"] * oH).mean(), L=(phi["H"] * oL).mean(), H_farL=(phi["L"] * oH * kb["L"]).mean(),
                L_farH=(phi["H"] * oL * kb["H"]).mean())
    c0 = (oH * (1.0 - kb["L"])).mean()
    return dict(kbar=kb, oH=oH, oL=oL, c0=c0, sums=sums, near=near, ln_lam=np.log(sums[0]),
                gH=np.log(sums[1] + near["H"]), gL=np.log(sums[2] + near["L"]),
                gHL=np.log(sums[3] + near["H_farL"] + near["L_farH"] + c0), traj=traj)


def pair(c, e):
    """a (node, template) pair of segment e: the injected mass at the valid node nearest s = 1"""
    m = 1
    ks = np.nonzero(c["geometry"][0][:, m])[0]
    return ks[np.argmin(np.abs(np.log(c["store"].s[ks])))], m


def _close(got, ref, rtol=RTOL):
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=rtol)


def test_kbar_one(case):
    """an empty window (kbar = 1): lambda_H = lambda_L = lambda_HL = Lambda_S, through both reductions"""
    for f, kw in ((pe.pe_intrinsic, {}), (pe.pe_redshift, dict(nz=40, nm1=30, m1_range=(10.0, 60.0)))):
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
    monkeypatch.setattr(pe, "_kbar", lambda traj, I, gd, sd, xp: xp.zeros(traj["lqH"].shape[1]) if I == zero
                        else kbar(traj, I, gd, sd, xp))
    other = "L" if zero == "H" else "H"
    lamN, one = node_values(case)
    fin = np.isfinite(lamN)
    _close(one["lamN_g" + zero][fin], one["lamN_" + other][fin], 1e-12)
    r = pe.pe_intrinsic(case["segs"], case["bank"], case["store"], n_loud=0, glitch=True)
    _close(r["ln_lambda_g" + zero], r["ln_lambda_" + other], 1e-12)
    for e in range(2):
        k, m = pair(case, e)
        h = by_hand(case, e, k, m, kbar={zero: np.zeros(case["bank"].n_time)})
        _close(one["lamN_g" + other][e, k, m], h["g" + other])
        _close(one["lamN_gHL"][e, k, m], h["gHL"])


def test_by_hand(case):
    """every term at one (node, template) pair of each segment against the reference written here"""
    lamN, one = node_values(case)
    sd = pe._single_setup(case["bank"], case["seg"], case["store"], case["geometry"])
    gd = pe._glitch_setup(True, case["bank"], case["seg"], case["store"])
    nr = single.RHO_EDGES.size - 1
    for e in range(2):
        k, m = pair(case, e)
        h = by_hand(case, e, k, m)
        dips = {I: h["kbar"][I].min() for I in "HL"}
        assert dips["HL"[e]] < 0.5 < dips["LH"[e]]                    # the transient is seen in one detector only
        gw = pe._glitch_weights(h["traj"], gd, sd, np)
        _close(np.exp(gw["tw"][0]), h["kbar"]["H"], 1e-12)
        _close(np.exp(gw["tw"][1]), h["kbar"]["L"], 1e-12)
        _close(gw["L"][1], h["oH"], 1e-12)
        _close(gw["H"][1], h["oL"], 1e-12)
        _close(gw["c0"], h["c0"], 1e-12)
        sv = pe._single_values(h["traj"], sd, np, gw)
        _close(np.exp(sv["L"][0][1, m, nr + k]), h["near"]["H"])
        _close(np.exp(sv["H"][0][1, m, nr + k]), h["near"]["L"])
        _close(np.exp(sv["L"][0][2, m, nr + k]), h["near"]["H_farL"])
        _close(np.exp(sv["H"][0][2, m, nr + k]), h["near"]["L_farH"])
        _close(lamN[e, k, m], h["ln_lam"])
        for X in ("H", "L", "HL"):
            _close(one["lamN_g" + X][e, k, m], h["g" + X])
        print("segment %d, node %d, template %d: |d ln| of Lambda_S %.1e, lambda_H %.1e, lambda_L %.1e, lambda_HL %.1e;"
              " near terms rel %.1e; lambda - Lambda_S %.3f %.3f %.3f"
              % (e, k, m, abs(lamN[e, k, m] - h["ln_lam"]), *(abs(one["lamN_g" + X][e, k, m] - h["g" + X])
                                                               for X in ("H", "L", "HL")),
                 max(abs(np.exp(sv[I][0][i, m, nr + k]) / h["near"][key] - 1) for I, i, key in
                     (("L", 1, "H"), ("H", 1, "L"), ("L", 2, "H_farL"), ("H", 2, "L_farH"))),
                 *(h["g" + X] - h["ln_lam"] for X in ("H", "L", "HL"))))
        assert abs(h["gH"] - h["ln_lam"]) > 1e-3 and abs(h["gL"] - h["ln_lam"]) > 1e-3   # the terms matter here

    lnW = np.full(case["geometry"][2].shape, -np.inf)
    yN = case["geometry"][2]
    for m in range(case["bank"].M.size):
        pts = np.nonzero(yN[:, m] > 0)[0]
        order = pts[np.argsort(yN[pts, m])]
        lnW[order, m] = pe.distance_weights(yN[order, m], "volume", 0.1267)
    r = pe.pe_intrinsic(case["segs"], case["bank"], case["store"], n_loud=0, glitch=True)
    for X in ("H", "L", "HL"):
        ref = [pe._distance_integral(lnW, one["lamN_g" + X][e]) for e in range(2)]
        np.testing.assert_array_equal(r["ln_lambda_g" + X], ref)


def test_glitch_leaves_the_rest(case):
    """glitch=None changes nothing, and with glitch every output of single=True and of the default path is the same
    array (variant 0 of the weighted read is the unweighted read)"""
    for f, kw in ((pe.pe_intrinsic, {}), (pe.pe_redshift, dict(nz=40, nm1=30, m1_range=(10.0, 60.0)))):
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
