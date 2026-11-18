"""unequal noise curves in exmarg.search (docs/2026-11-15_unequal_noise_design.md, Secs. 2, 3, 5): the template power
of each detector, one store read at its asymmetry, the mixture of the two stores that bracket a template's asymmetry,
the noise mean, zmin, the error cases, and the GPU against the CPU.

Stores.  The incoherent store of tests/test_pe_windows.py generalized to an asymmetry w_j: at node sbar

    lnS(tau, phi, beta, Delta) = ln p(tau) + ln Phi(e^beta / s_L; s_L) + ln Phi(e^(beta + 2 Delta) / s_H; s_H),
    s_H = sbar (1 + w_j), s_L = sbar (1 - w_j),  p(tau) uniform on three lags.

Read at a = sbar / ubar with the data scaled to the store's model (|z_I''|^2 = |z_I|^2 ubar (1 +- w_j) / u_I),
e^beta / s_L = |z_L|^2 / u_L = w_L and e^(beta + 2 Delta) / s_H = w_H, so the table is the likelihood ratio of a merger
seen incoherently, with the closed form

    Lambda_j = (1/n_t) sum_{t, tau} p(tau) Phi(w_L(t); s_L) Phi(w_H(t + tau); s_H),

E_0 Lambda_j = 1 for every w_j whatever the template's true w (Sec. 3), and the single-detector marginals are the
columns of single.py at s_L and s_H.  With alpha_lo = 1 and ymin = 0.5 every template also has a boundary point at
a = 0.25 (read from the first node).  The stores of one test share their grids (beta from ln sbar - 9 to ln sbar + 8,
Delta from -8 to 8, steps 0.08, unbiased), as a mixture requires; each node has its own beta grid, so a launch reads
one node (nodes_per_launch=1).

References (no code of exmarg.search.pe): the table read of the gather in float64 (test_pe_glitch.ln_S) at the data
scaled by the formula of Sec. 3 written here from u_I and ubar (rounded to float32, as the data are); Phi at each w
(single._ln_phi, validated in test_single.py); the glitch-time weights of Sec. 4 as explicit log-sum-exps and those
of Sec. 4a in decimal arithmetic (test_pe_windows.direct_weights); the closed form above; and the mixture weights
f = (w - w_j) / (w_{j+1} - w_j) computed here from u_H, u_L.

Segments: 4 s up to 128 Hz (4096 arrival times), templates of 40, 60 and 90 M_sun, noise drawn on curves of each
detector: 'uniform' sig2_H = sig2 / (1 + W), sig2_L = sig2 / (1 - W) (every template has w = W), or 'shaped' (H worse
above 100 Hz, L below 40 Hz: w = -0.24, -0.27, -0.35 for the three templates, in two cells of the grid
-0.45, -0.30, -0.15).  Rows: a merger (y = 12) with a glitch in H in its zone and one in L 2.2 s later; noise; noise
with a glitch in L.
"""

import functools
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from exmarg.detectors import S as design, c as C_LIGHT, x as SITES
from exmarg.pp import simulate
from exmarg.profiles import profile
from exmarg.search import mass_bank, pe, read_segments, single
from exmarg.search.segments import D0
from exmarg.search.store import RedshiftStore, unbias
from test_pe_glitch import ln_S
from test_pe_windows import add_glitch, direct_time_weights, law

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "03361d4"                      # the branch before unequal noise curves: pe.py as it was
T4, FMIN, FMAX = 4.0, 20.0, 128.0
MASSES = np.array([40.0, 60.0, 90.0])
S_INC = (2.0, 8.0)                      # the nodes of the incoherent stores
TAU = np.arange(-1, 2)
H_GRID = 0.08
GRID = (-0.45, -0.30, -0.15)            # the asym of the mixture of the shaped curves
YMIN = 0.5                              # with alpha_lo = 1 the boundary points are at a = 0.25 (s = a ubar about 0.2 to 0.8)
RTOL = 1e-6                             # the table read
RTOL_S = 1e-5                           # what holds the single-detector columns, interpolated in v (dv = 0.01) by single.py
REDSHIFT = dict(nz=40, nm1=30, m1_range=(10.0, 60.0))
NR = single.RHO_EDGES.size - 1
P_H = dict(f=[0.85, 0.15], q=[0.01, 0.15], pi_k=np.stack([law(4.0, 4.0), law(2.5, 6.0)]))
P_L = dict(f=[0.6, 0.3, 0.1], q=[0.0, 0.02, 0.2], pi_k=np.stack([law(4.0, 4.0), law(4.0, 5.0), law(2.0, 8.0)]))
WINDOWS = dict(model="windows", n_window=64, zone=1, H=P_H, L=P_L)


@functools.lru_cache(maxsize=None)
def inc_store(asym, s_nodes=S_INC):
    """the incoherent store of asymmetry asym at the nodes s_nodes (module docstring)"""
    d = np.arange(-8.0, 8.0 + 1e-9, H_GRID)
    tables, betas = [], []
    for s in s_nodes:
        b = np.log(s) + np.arange(-9.0, 8.0 + 1e-9, H_GRID)
        x = b[0] + 2.0 * d[0] + H_GRID * np.arange(b.size + 2 * d.size - 2)          # beta + 2 Delta on the same step
        sL, sH = s * (1.0 - asym), s * (1.0 + asym)
        lnPhi = lambda y, si: single._ln_phi(np.exp(0.5 * (y - np.log(si))), np.full(y.size, si))
        T2 = lnPhi(b, sL)[:, None] + lnPhi(x, sH)[np.arange(b.size)[:, None] + 2 * np.arange(d.size)[None, :]]
        T = np.broadcast_to(T2 - np.log(TAU.size), (TAU.size, 49, b.size, d.size))
        tables.append(unbias(np.ascontiguousarray(T, np.float32)))
        betas.append(b)
    n = len(s_nodes)
    return RedshiftStore(s=np.array(s_nodes, float), alpha_lo=np.full(n, 1.0), alpha_hi=np.full(n, 1e6), beta=betas,
                         delta=[d] * n, tau_idx=TAU, tables=tables, n_max=120, asym=float(asym))


def design_sig2():
    nu = np.arange(int(np.ceil(FMIN * T4)), int(np.floor(FMAX * T4)) + 1)
    return nu / T4, T4 * design("H", nu / T4) / 2.0


def curves(kind):
    """{'H', 'L'}: the noise curve of each detector, 'shaped' or a uniform asymmetry W (a float)"""
    f, s2 = design_sig2()
    if kind == "shaped":
        return dict(H=1.6 * s2 * (1.0 + 10.0 * (f / 120.0) ** 3), L=s2 * (1.0 + 10.0 * (30.0 / f) ** 2))
    return dict(H=s2 / (1.0 + kind), L=s2 / (1.0 - kind))


@functools.lru_cache(maxsize=None)
def make_case(kind):
    """segments (merger with glitches, noise, noise with an L glitch) on the curves of kind, the bank built on them"""
    sig2 = curves(kind)
    y = iter([12.0, 1e-8])
    raw, _, _ = simulate(2, 77, lambda r: (60.0, 1.0, 0.0), lambda r: next(y), T4, FMIN, FMAX, sig2=sig2)
    dt = T4 / 4096
    tm = 1598 * dt
    g = lambda I, d, t, rho, ph: add_glitch(d, dict(raw, sig2=raw["sig2_" + I]), (t % T4) // dt * dt, rho, 60.0, ph)
    dH = [g("H", raw["d_H"][0], tm + 0.05, 15.0, 0.3), raw["d_H"][1], raw["d_H"][1]]
    dL = [g("L", raw["d_L"][0], tm + 2.2, 12.0, 2.0), raw["d_L"][1], g("L", raw["d_L"][1], 1.0, 12.0, 0.7)]
    segs = dict(raw, d_H=np.array(dH), d_L=np.array(dL))
    bank = mass_bank(MASSES, segs)
    return dict(segs=segs, seg=read_segments(segs), bank=bank)


def setup(c, stores, ymin=YMIN):
    """the case c with the stores (list), their node geometry and one node per launch"""
    geometry = pe._node_geometry(c["bank"], stores[0], ymin)
    return dict(c, stores=stores, geometry=geometry, groups=[[k] for k in range(stores[0].s.size)])


def node_values(c, glitch, rows, trim=True, asym_tol=0.0):
    """_node_values of the rows with every arrival time: lamN and the dict of _single_nodes"""
    st = c["stores"] if len(c["stores"]) > 1 else c["stores"][0]
    sd = pe._single_setup(c["bank"], c["seg"], st, c["geometry"], trim, asym_tol)
    gd = None if glitch is None else pe._glitch_setup(glitch, c["bank"], c["seg"], st)
    return pe._node_values(c["bank"], c["seg"], rows, st, c["geometry"], c["groups"], 256, False, 0, 0, 0, sd, gd,
                           False, trim, asym_tol)


# ----------------------------------------------------------------------------------------------- the references


def data(c, e):
    """the matched-filter outputs of row e in float64, and w_I = |z_I|^2 / u_I"""
    bank, seg = c["bank"], c["seg"]
    traj = bank.trajectories({I: seg["d_" + I][e] for I in "HL"}, {I: seg["sig2_" + I] for I in "HL"})
    q = {k: np.asarray(v, np.float64) for k, v in traj.items()}
    w = {I: np.exp(2.0 * q["lq" + I]) / getattr(bank, "u_" + I)[:, None] for I in "HL"}
    return traj, q, w


def scaled(c, q, asym):
    """Sec. 3: the data of every template scaled to the model of asymmetry asym, |z_I''|^2 = |z_I|^2 u_I' / u_I with
    u_H' = ubar (1 + asym), u_L' = ubar (1 - asym), rounded to float32 as the data are"""
    bank = c["bank"]
    ubar = 0.5 * (bank.u_H + bank.u_L)
    out = dict(q)
    for I, sg in (("H", 1.0), ("L", -1.0)):
        c_I = 0.5 * np.log(ubar * (1.0 + sg * asym) / getattr(bank, "u_" + I))
        out["lq" + I] = (q["lq" + I] + c_I[:, None]).astype(np.float32).astype(np.float64)
    return out


def old_model_weights(c, traj):
    """{I: (ln kbar_I, ln(1 - kbar_I))} of the default glitch model (fiducial law rho^-4 above 4, window 0.1 s) as
    explicit log-sum-exps, with the true u_I"""
    bank = c["bank"]
    n = bank.n_time
    hw = int(round(0.1 / (T4 / n)))
    a, b = np.maximum(single.RHO_EDGES[:-1], 4.0), single.RHO_EDGES[1:]
    pk = np.where(b > a, (np.minimum(a, b) ** -3 - b ** -3) / 3.0, 0.0)
    pk /= pk.sum()
    out = {}
    for I in "HL":
        lg = single.glitch_time_series(traj["lq" + I], np.log(getattr(bank, "u_" + I)), pk)
        d = np.arange(n)[:, None]
        lk = logsumexp(lg[(d + hw + 1 + np.arange(n - 2 * hw - 1)) % n], axis=1) - logsumexp(lg)
        ln = logsumexp(lg[(d - hw + np.arange(2 * hw + 1)) % n], axis=1) - logsumexp(lg)
        out[I] = (lk, ln)
    return out


def reference(c, e, st, kbar=None):
    """the node values of row e read from store st at every pair of the geometry (boundary points in row n_node):
    lam, lamH, lamL, and with kbar {I: (ln kbar, ln(1 - kbar))} per arrival time the terms of Sec. 4 gH, gL, gHL"""
    bank, (valid, _, _, bnode, blna) = c["bank"], c["geometry"]
    traj, q0, w = data(c, e)
    q = scaled(c, q0, st.asym)
    n, nn, nm = bank.n_time, st.s.size, bank.M.size
    ubar = 0.5 * (bank.u_H + bank.u_L)
    t = np.arange(n)
    lags = st.tau_idx[np.abs(st.tau_idx) * T4 / n <= np.linalg.norm(SITES["H"] - SITES["L"]) / C_LIGHT]
    keys = ("lam", "lamH", "lamL", "gH", "gL", "gHL")
    res = {k: np.full((nn + 1, nm), -np.inf) for k in keys}
    if kbar is not None:
        lk, ln = {I: kbar[I][0] for I in "HL"}, {I: kbar[I][1] for I in "HL"}
        oH = logsumexp(ln["H"][(t[:, None] + lags[None, :]) % n], axis=1) - np.log(lags.size)
        oL = logsumexp(ln["L"][(t[:, None] - lags[None, :]) % n], axis=1) - np.log(lags.size)
        lc0 = logsumexp(oH + ln["L"]) - np.log(n)
        wH, wL = np.maximum(lk["H"], -1e5), np.maximum(lk["L"], -1e5)
    pairs = [(k, m) for k in range(nn) for m in range(nm) if valid[k, m]] + [(nn, m) for m in range(nm) if bnode[m] >= 0]
    for k, m in pairs:
        node, lna = (k, np.log(st.s[k] / ubar[m])) if k < nn else (bnode[m], blna[m])
        Tk = np.asarray(st.tables[node], np.float64)
        terms = [[], [], [], []]
        for r, tau in enumerate(st.tau_idx):
            i1 = (t + tau) % n
            S = ln_S(Tk[r], lna, q["lqH"][m, i1], q["phH"][m, i1], q["lqL"][m], q["phL"][m],
                     np.asarray(st.beta[node]), np.asarray(st.delta[node]))
            for v, wt in enumerate((0.0,) if kbar is None else (0.0, wH[i1], wL, wH[i1] + wL)):
                terms[v].append(S + wt)
        res["lam"][k, m] = logsumexp(np.concatenate(terms[0])) - np.log(n)
        sbar = np.exp(lna) * ubar[m]
        phi = {I: single._ln_phi(np.sqrt(w[I][m]), np.full(n, sbar * (1.0 + sg * st.asym)))
               for I, sg in (("H", 1.0), ("L", -1.0))}
        res["lamH"][k, m] = logsumexp(phi["H"]) - np.log(n)
        res["lamL"][k, m] = logsumexp(phi["L"]) - np.log(n)
        if kbar is not None:
            far_H, far_L, far_HL = (logsumexp(np.concatenate(terms[v])) - np.log(n) for v in (1, 2, 3))
            near = lambda I, lw: logsumexp(phi[I] + lw) - np.log(n)
            res["gH"][k, m] = np.logaddexp(far_H, near("L", oH))
            res["gL"][k, m] = np.logaddexp(far_L, near("H", oL))
            res["gHL"][k, m] = logsumexp([far_HL, near("L", oH + lk["L"]), near("H", oL + lk["H"]), lc0])
    return res


def closed_form(c, e, st):
    """ln Lambda_j of the incoherent store at its nodes (n_node, n_template) from the formula of the module docstring"""
    bank = c["bank"]
    _, _, w = data(c, e)
    n = bank.n_time
    ubar = 0.5 * (bank.u_H + bank.u_L)
    out = np.full((st.s.size, bank.M.size), -np.inf)
    for k, s in enumerate(st.s):
        for m in range(bank.M.size):
            pL = single._ln_phi(np.sqrt(w["L"][m]), np.full(n, s * (1.0 - st.asym)))
            pH = single._ln_phi(np.sqrt(w["H"][m]), np.full(n, s * (1.0 + st.asym)))
            out[k, m] = logsumexp([pL + pH[(np.arange(n) + tau) % n] for tau in st.tau_idx]) - np.log(n * TAU.size)
    return out


def bracket(c, grid):
    """per template the index j of the store below its w in the sorted grid and f = (w - w_j) / (w_{j+1} - w_j),
    from u_H and u_L"""
    bank = c["bank"]
    w = (bank.u_H - bank.u_L) / (bank.u_H + bank.u_L)
    g = np.sort(np.asarray(grid, float))
    j = np.array([np.nonzero(g <= x)[0][-1] for x in w])
    j = np.minimum(j, g.size - 2)
    return j, (w - g[j]) / (g[j + 1] - g[j])


def mixed(refs, j, f, key):
    """ln((1 - f) e^X_j + f e^X_{j+1}) per template from the references of each store (sorted by asym)"""
    out = np.full(refs[0][key].shape, -np.inf)
    for m in range(out.shape[1]):
        with np.errstate(divide="ignore"):
            out[:, m] = np.logaddexp(np.log1p(-f[m]) + refs[j[m]][key][:, m], np.log(f[m]) + refs[j[m] + 1][key][:, m])
    return out


def close(got, want, rtol=RTOL):
    """the same -inf pattern, and |got - want| <= rtol max(1, |want|) elsewhere; returns the largest error"""
    assert np.array_equal(np.isfinite(got), np.isfinite(want))
    fin = np.isfinite(want)
    err = np.abs(got[fin] - want[fin]) / np.maximum(1.0, np.abs(want[fin]))
    assert fin.any() and err.max() <= rtol, err.max()
    return err.max()


# ----------------------------------------------------------------------------------------------- the bank


def test_bank_powers():
    """u_H, u_L are sum a^2 / sig2_I of the template profile (computed here), u their mean and asym (u_H - u_L)/(u_H +
    u_L); with equal curves u_H = u_L = u to the last bit and asym = 0"""
    c = make_case("shaped")
    bank, segs = c["bank"], c["segs"]
    for i, M in enumerate(MASSES):
        nu, _, a, _ = profile(M / 2, M / 2, 0.0, 0.0, T4, FMIN, FMAX, 1.0)
        j = np.searchsorted(segs["nu"], nu)
        for I in "HL":
            want = np.sum((a / D0) ** 2 / segs["sig2_" + I][j])
            assert abs(getattr(bank, "u_" + I)[i] / want - 1.0) < 1e-12
    assert np.allclose(bank.u, 0.5 * (bank.u_H + bank.u_L), rtol=1e-15, atol=0)
    assert np.allclose(bank.asym, (bank.u_H - bank.u_L) / (bank.u_H + bank.u_L), rtol=1e-15, atol=0)
    print("\nshaped curves: w = %s, u_H/u_L = %s" % (np.round(bank.asym, 4), np.round(bank.u_H / bank.u_L, 4)))
    assert np.all(np.diff(bank.asym) < 0) and -0.45 < bank.asym.min() and bank.asym.max() < -0.15
    assert ((bank.asym > -0.3) & (bank.asym < -0.15)).any() and ((bank.asym < -0.3) & (bank.asym > -0.45)).any()
    eq = mass_bank(MASSES, dict(segs, sig2=segs["sig2_L"], sig2_H=segs["sig2_L"]))
    assert np.array_equal(eq.u_H, eq.u_L) and np.array_equal(eq.u, eq.u_H) and np.array_equal(eq.asym, np.zeros(3))


# ----------------------------------------------------------------------------------------------- (i) equal curves


@pytest.fixture(scope="module")
def pe_before(tmp_path_factory):
    """pe.py as committed before unequal noise curves, imported as a module of exmarg.search"""
    try:
        src = subprocess.run(["git", "show", BEFORE + ":src/exmarg/search/pe.py"], cwd=ROOT, capture_output=True,
                             check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        pytest.skip("no git history to compare with (%s)" % err)
    path = tmp_path_factory.mktemp("before") / "pe_before.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("exmarg.search._pe_before", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "asym" not in src
    return mod


@pytest.mark.parametrize("glitch", [None, "single", True, "windows"])
def test_equal_curves_unchanged(pe_before, glitch):
    """equal curves and one store of asym 0 (the published one): every output of pe_redshift (with interp 'log': the
    committed pe.py takes ln Lambda linear between the nodes), pe_intrinsic and window_evidences is the array of the
    committed pe.py bit for bit (same keys), with every arrival time and with the default selection; also when the
    store is given as a list of one"""
    c = make_case(0.0)
    st = inc_store(0.0)
    kw = dict(single=True) if glitch == "single" else dict(glitch=WINDOWS if glitch == "windows" else glitch)
    n = 0
    for name, extra in (("pe_redshift", REDSHIFT), ("pe_intrinsic", {})):
        for sel in (dict(n_loud=0), dict(seed=3, n_loud=256, n_random=256)):
            log = dict(interp="log") if name == "pe_redshift" else {}
            a = getattr(pe, name)(c["segs"], c["bank"], st, ymin=0.1, nodes_per_launch=1, **kw, **sel, **extra, **log)
            b = getattr(pe_before, name)(c["segs"], c["bank"], st, ymin=0.1, nodes_per_launch=1, **kw, **sel, **extra)
            a1 = getattr(pe, name)(c["segs"], c["bank"], [st], ymin=0.1, nodes_per_launch=1, **kw, **sel, **extra,
                                   **log)
            assert set(a) == set(a1) and set(a) - {"wnet"} == set(b)      # wnet: added after BEFORE
            for k in a:
                assert np.array_equal(np.asarray(a1[k]), np.asarray(a[k])), (name, k)
            for k in b:
                assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), (name, k)
            n += len(b)
    w0, w1 = pe.window_evidences(c["segs"], c["bank"]), pe_before.window_evidences(c["segs"], c["bank"])
    assert all(np.array_equal(w0[k], w1[k]) for k in w0)
    print("\nglitch %s: %d arrays bit for bit equal to the committed pe.py" % (glitch, n + len(w0)))


# ----------------------------------------------------------------------------------------------- (ii) one store


@pytest.fixture(scope="module")
def uniform():
    """uniform asymmetry W = -0.3: the store of asym -0.3 alone, its node values with the default glitch model and with
    the window model"""
    c = setup(make_case(-0.3), [inc_store(-0.3)])
    rows = [0, 1, 2]
    return dict(c, rows=rows, old=node_values(c, True, rows), windows=node_values(c, WINDOWS, rows),
                res=pe.pe_redshift(c["segs"], c["bank"], c["stores"][0], ymin=YMIN, nodes_per_launch=1, n_loud=0, glitch=WINDOWS, **REDSHIFT))


def test_one_store_at_its_asym(uniform):
    """every template has w = -0.3 (to 1e-15) and reads the store of asym -0.3 alone: ln Lambda, ln Lambda_S^H,
    ln Lambda_S^L and the lambda terms of both glitch models at every node and boundary point against the references
    (table read at a = sbar / ubar, the columns at s_H = 0.7 sbar and s_L = 1.3 sbar), to 1e-6 (the table read) and
    1e-5 (what holds the single-detector columns); and ln Lambda at the nodes against the closed form of the
    incoherent store, to its interpolation error: 1e-3 of max(1, |ln Lambda|) (measured 1.7e-4 here; the store of asym 0
    on equal curves gives 5.3e-4)"""
    c, st = uniform, uniform["stores"][0]
    assert np.abs(c["bank"].asym + 0.3).max() < 1e-15 and c["geometry"][0].sum() >= 5 and (c["geometry"][3] >= 0).sum() >= 2
    lines, errs = [], []
    for e in c["rows"]:
        traj = data(c, e)[0]
        res_w = c["res"]
        h_old = reference(c, e, st, old_model_weights(c, traj))
        h_win = reference(c, e, st, direct_time_weights(res_w, e, WINDOWS, c["bank"].n_time)[0])
        (lam, one), (_, onew) = c["old"], c["windows"]
        err = [close(lam[e], h_old["lam"]), close(one["lamN_H"][e], h_old["lamH"], RTOL_S),
               close(one["lamN_L"][e], h_old["lamL"], RTOL_S)]
        err += [close(one["lamN_g" + X][e], h_old["g" + X], RTOL_S) for X in ("H", "L", "HL")]
        err.append(close(onew["lamN_gHL"][e], h_win["gHL"], RTOL_S))
        errs.append(max(err))
        cf = closed_form(c, e, st)
        on = c["geometry"][0]                                # the valid pairs (template 90 is below node 0)
        d = np.abs(lam[e][:-1][on] - cf[on]) / np.maximum(1.0, np.abs(cf[on]))
        lines.append("row %d: max error against the references %.1e; against the closed form %.1e relative (ln Lambda "
                     "%s)" % (e, max(err), d.max(), np.round(cf[on], 2)))
        assert d.max() < 1e-3
    print("\n" + "\n".join(lines))


def test_one_store_of_several(uniform):
    """the same bank with the stores of asym 0, -0.3 and -0.6 given (in any order): only the store of -0.3 is read
    (every table read is one of its tables), and every output of pe_redshift is the array of the one-store call bit
    for bit"""
    c = uniform
    stores = [inc_store(0.0), inc_store(-0.6), inc_store(-0.3)]
    seen = []
    real = pe._tables

    def spy(store, *args):
        seen.append(store.asym)
        return real(store, *args)
    pe._tables = spy
    try:
        r = pe.pe_redshift(c["segs"], c["bank"], stores, ymin=YMIN, nodes_per_launch=1, n_loud=0, glitch=WINDOWS, **REDSHIFT)
    finally:
        pe._tables = real
    assert set(seen) == {-0.3} and len(seen) > 0
    assert set(r) == set(c["res"]) and all(np.array_equal(np.asarray(r[k]), np.asarray(c["res"][k])) for k in r)


def test_asym_tol(uniform):
    """asym_tol: every template at w = -0.3, 0.02 above the range of the stores of asym -0.45 and -0.32.  With asym_tol
    = 0.05 each is read from the end store -0.32 alone with its data scaled to that store's model: ln Lambda, ln
    Lambda_S^H, ln Lambda_S^L and ln B at every node and boundary point equal the reference of that store (to 1e-6 and
    1e-5), pe_redshift returns asym_outside = 3, and the store of -0.32 alone with the same tolerance gives the same
    arrays bit for bit; with asym_tol = 0.01 or 0 (the strict check) a ValueError, for the pair and for the one store"""
    end = inc_store(-0.32)
    c = dict(uniform, stores=[inc_store(-0.45), end])
    rows = [0, 1, 2]
    lam, one = node_values(c, None, rows, asym_tol=0.05)
    _, onew = node_values(c, WINDOWS, rows, asym_tol=0.05)
    kw = dict(ymin=YMIN, nodes_per_launch=1, n_loud=0, glitch=WINDOWS, **REDSHIFT)
    r2 = pe.pe_redshift(c["segs"], c["bank"], c["stores"], asym_tol=0.05, **kw)
    r1 = pe.pe_redshift(c["segs"], c["bank"], end, asym_tol=0.05, **kw)
    errs = []
    for e in rows:
        h = reference(c, e, end, direct_time_weights(r2, e, WINDOWS, c["bank"].n_time)[0])
        errs += [close(lam[e], h["lam"]), close(one["lamN_H"][e], h["lamH"], RTOL_S),
                 close(one["lamN_L"][e], h["lamL"], RTOL_S), close(onew["lamN_gHL"][e], h["gHL"], RTOL_S)]
    print("\nw = -0.3 read from the end store -0.32: largest error against its reference %.1e" % max(errs))
    assert r2["asym_outside"] == 3 and set(r1) == set(r2) and all(np.array_equal(np.asarray(r1[k]), np.asarray(r2[k]))
                                                                  for k in r1)
    for st in (c["stores"], end):
        for tol in (0.01, 0.0):
            with pytest.raises(ValueError, match=r"cover u_H/u_L .*asym_tol %g" % tol):
                pe.pe_redshift(c["segs"], c["bank"], st, asym_tol=tol, **kw)


# ----------------------------------------------------------------------------------------------- (iii) the mixture


@pytest.fixture(scope="module")
def shaped():
    """the shaped curves and the three stores of GRID: node values of every row with the default glitch model and with
    the window model, the stores read per template, and pe_redshift with the window model"""
    c = setup(make_case("shaped"), [inc_store(a) for a in GRID[::-1]])     # given in descending order
    rows = [0, 1, 2]
    seen = []
    real = pe._store_values

    def spy(store, reads, *args):
        seen.extend((r["e"]["asym"], r["e"]["use"].copy()) for r in reads)
        return real(store, reads, *args)
    pe._store_values = spy
    try:
        windows = node_values(c, WINDOWS, rows)
    finally:
        pe._store_values = real
    return dict(c, rows=rows, seen=seen, old=node_values(c, True, rows), windows=windows, plain=node_values(c, None, rows),
                res=pe.pe_redshift(c["segs"], c["bank"], c["stores"], ymin=YMIN, nodes_per_launch=1, n_loud=0, glitch=WINDOWS, **REDSHIFT))


def test_mixture_reads_the_bracketing_stores(shaped):
    """each template is gathered from the two stores that bracket its w and from no other"""
    j, f = bracket(shaped, GRID)
    g = np.sort(GRID)
    print("\nw %s: stores %s, f %s" % (np.round(shaped["bank"].asym, 4), [(g[a], g[a + 1]) for a in j], np.round(f, 3)))
    assert np.all((f > 0.05) & (f < 0.95)) and len(set(j)) == 2
    use = {a: u for a, u in shaped["seen"]}
    assert set(use) == set(GRID)
    for m in range(MASSES.size):
        assert {a for a, u in use.items() if u[m]} == {g[j[m]], g[j[m] + 1]}


def test_mixture_node_values(shaped):
    """ln Lambda, ln Lambda_S^H, ln Lambda_S^L, the three lambda of the default glitch model and ln B of the window
    model at every node and boundary point: exactly (1 - f) X_j + f X_{j+1} with X of each store from the references
    and f from u_H, u_L, to 1e-6 (ln Lambda) and 1e-5 (the others); and ln Lambda without glitch terms"""
    c = shaped
    j, f = bracket(c, GRID)
    stores = sorted(c["stores"], key=lambda s: s.asym)
    lines = []
    for e in c["rows"]:
        traj = data(c, e)[0]
        kb = old_model_weights(c, traj)
        kw = direct_time_weights(c["res"], e, WINDOWS, c["bank"].n_time)[0]
        h_old = [reference(c, e, st, kb) for st in stores]
        h_win = [reference(c, e, st, kw) for st in stores]
        (lam, one), (_, onew) = c["old"], c["windows"]
        err = dict(lam=close(lam[e], mixed(h_old, j, f, "lam")), plain=close(c["plain"][0][e], mixed(h_old, j, f, "lam")),
                   H=close(one["lamN_H"][e], mixed(h_old, j, f, "lamH"), RTOL_S),
                   L=close(one["lamN_L"][e], mixed(h_old, j, f, "lamL"), RTOL_S),
                   B=close(onew["lamN_gHL"][e], mixed(h_win, j, f, "gHL"), RTOL_S))
        for X in ("H", "L", "HL"):
            err["g" + X] = close(one["lamN_g" + X][e], mixed(h_old, j, f, "g" + X), RTOL_S)
        spread = max(np.nanmax(np.abs(np.where(np.isfinite(a["lam"]), a["lam"] - b["lam"], np.nan)))
                     for a, b in zip(h_old[:-1], h_old[1:]))
        lines.append("row %d: errors %s; largest |X_j - X_j+1| of ln Lambda %.3f"
                     % (e, " ".join("%s %.1e" % kv for kv in err.items()), spread))
        assert spread > 0.05                         # the stores differ: a wrong f or a wrong store shows
    print("\n" + "\n".join(lines))


def test_mixture_on_the_grid(shaped):
    """pe_redshift with the three stores: ln_lambda, ln_lambda_H, ln_lambda_L and ln_B of every row are the reduction
    (_to_source_frame) of the mixed reference node values, to 1e-6; the per-segment arrays (window evidences, ln c,
    state, glitch evidences, wmax) do not depend on the stores: those of a call with the store at the w of the middle
    template"""
    c, res = shaped, shaped["res"]
    j, f = bracket(c, GRID)
    stores = sorted(c["stores"], key=lambda s: s.asym)
    yq = D0 / pe._dl_of_z(res["z"])
    for e in c["rows"]:
        kw = direct_time_weights(res, e, WINDOWS, c["bank"].n_time)[0]
        h = [reference(c, e, st, kw) for st in stores]
        for key, ref in (("ln_lambda", "lam"), ("ln_lambda_H", "lamH"), ("ln_lambda_L", "lamL"), ("ln_B", "gHL")):
            want = pe._to_source_frame(mixed(h, j, f, ref), c["geometry"][2], c["bank"].M, yq, res["m1"], res["z"], np)
            close(res[key][e], want, RTOL if key == "ln_lambda" else RTOL_S)
    st = inc_store(float(c["bank"].asym[1]))
    one = pe._store_list(st)
    bank = c["bank"]
    keep = bank.asym.copy()
    try:
        bank.asym = np.full(3, st.asym)                   # every template on that store: the data part is the same
        r1 = pe.pe_redshift(c["segs"], bank, one[0], ymin=YMIN, nodes_per_launch=1, n_loud=0, glitch=WINDOWS, **REDSHIFT)
    finally:
        bank.asym = keep
    for k in ("ln_window_H", "ln_window_L", "ln_c_H", "ln_c_L", "state_H", "state_L", "ln_glitch_H", "ln_glitch_L",
              "wmax_H", "wmax_L"):
        assert np.array_equal(res[k], r1[k]), k


def test_asym_tol_inside(shaped):
    """within the range of the stores a tolerance changes nothing: asym_tol = 0.05 gives every array of asym_tol = 0 bit
    for bit, and asym_outside = 0"""
    c = shaped
    r = pe.pe_redshift(c["segs"], c["bank"], c["stores"], ymin=YMIN, nodes_per_launch=1, n_loud=0, glitch=WINDOWS,
                       asym_tol=0.05, **REDSHIFT)
    assert r.pop("asym_outside") == 0 and set(r) == set(c["res"])
    assert all(np.array_equal(np.asarray(r[k]), np.asarray(c["res"][k])) for k in r)


def test_mixture_intrinsic(shaped):
    """pe_intrinsic with the three stores: ln_lambda, ln_lambda_H and ln_B are the distance integrals of the mixed
    reference node values, to 1e-6"""
    c = shaped
    j, f = bracket(c, GRID)
    stores = sorted(c["stores"], key=lambda s: s.asym)
    r = pe.pe_intrinsic(c["segs"], c["bank"], c["stores"], ymin=YMIN, nodes_per_launch=1, n_loud=0, glitch=WINDOWS)
    yN = c["geometry"][2]
    lnW = np.full(yN.shape, -np.inf)
    for m in range(MASSES.size):
        pts = np.nonzero(yN[:, m] > 0)[0]
        order = pts[np.argsort(yN[pts, m])]
        lnW[order, m] = pe.distance_weights(yN[order, m], "volume", YMIN)
    for e in c["rows"]:
        kw = direct_time_weights(r, e, WINDOWS, c["bank"].n_time)[0]
        h = [reference(c, e, st, kw) for st in stores]
        for key, ref in (("ln_lambda", "lam"), ("ln_lambda_H", "lamH"), ("ln_B", "gHL")):
            close(r[key][e], logsumexp(lnW + mixed(h, j, f, ref), axis=0), RTOL if key == "ln_lambda" else RTOL_S)


def test_window_evidences_true_powers(shaped):
    """the window evidences and wmax of each detector use its own u_I: window_evidences equals single.window_evidence
    with ln u_I (and differs from it with ln u of the other detector), wmax_I = max_t |z_I|^2 / u_I, and pe_redshift
    returns the same arrays"""
    c = shaped
    w = pe.window_evidences(c["segs"], c["bank"], n_window=64)
    lnG = single.glitch_columns()
    for e in c["rows"]:
        traj, _, wI = data(c, e)
        for I, J in (("H", "L"), ("L", "H")):
            mine = single.window_evidence(traj["lq" + I], np.log(getattr(c["bank"], "u_" + I)), lnG, 64)
            other = single.window_evidence(traj["lq" + I], np.log(getattr(c["bank"], "u_" + J)), lnG, 64)
            assert np.array_equal(w["ln_window_" + I][e], mine) and not np.allclose(mine, other)
            assert np.allclose(w["wmax_" + I][e], wI[I].max(axis=1), rtol=1e-6, atol=0)
            assert np.array_equal(c["res"]["ln_window_" + I][e], mine) and np.array_equal(c["res"]["wmax_" + I][e],
                                                                                          w["wmax_" + I][e])


# ----------------------------------------------------------------------------------------------- (iv) noise mean


N_W = (-0.12, -0.30, -0.47)             # the asymmetry of the three templates
N_GRID = (0.0, -0.3, -0.6)              # the stores (the design's grid)
N_S = (1.0, 4.0)                        # their nodes
N_SEG, N_CHUNK, N_SEED = 400, 100, 20261115
E_MODEL = dict(f=np.array([0.9, 0.1]), q=np.array([0.02, 0.2]), pi_k=np.stack([law(4.0, 4.0), law(2.0, 5.0)]))


def white(seed, uH, uL, n_time, model=None):
    """the matched-filter outputs of one segment: white complex Gaussian z_I of power u_I per template, independent over
    templates, times and detectors; with model, glitches of the window model added per window (as
    test_pe_windows.white_segment): w_I = |z_I|^2 / u_I = rho^2 / 2 at the glitch"""
    rng = np.random.default_rng(seed)
    nm, nw = uH.size, 64
    nwin = n_time // nw
    out = {}
    for I, u in (("H", uH), ("L", uL)):
        z = (rng.standard_normal((nm, n_time)) + 1j * rng.standard_normal((nm, n_time))) / np.sqrt(2.0)
        if model is not None:
            s = int(rng.random() < model["f"][1])
            for w in np.nonzero(rng.random(nw) < model["q"][s])[0]:
                k = rng.choice(NR, p=model["pi_k"][s])
                rho = rng.uniform(single.RHO_EDGES[k], single.RHO_EDGES[k + 1])
                z[rng.integers(nm), w * nwin + rng.integers(nwin)] += np.sqrt(0.5) * rho * np.exp(2j * np.pi * rng.random())
        v = z * np.sqrt(u)[:, None]
        out["lq" + I], out["ph" + I] = np.log(np.abs(v)).astype(np.float32), np.angle(v).astype(np.float32)
    return out


@pytest.fixture(scope="module")
def noise_mean():
    """N_SEG segments of white outputs with powers u_I = ubar (1 +- w) (w of N_W), analysed with the stores of N_GRID at
    every node: the population average (mean over nodes and templates) of Lambda on noise, and of B of the window
    model E_MODEL on data drawn from it; each with the data scaled to each store's model (Sec. 3) and with the scaling
    deliberately left out (pe._scaled the identity)"""
    c = make_case(0.0)
    bank = mass_bank(MASSES, c["segs"])
    w = np.array(N_W)
    bank.u_H, bank.u_L, bank.asym = bank.u * (1.0 + w), bank.u * (1.0 - w), w
    stores = [inc_store(a, N_S) for a in N_GRID]
    geometry = pe._node_geometry(bank, stores[0], 0.1)
    assert geometry[0].all()
    seg = read_segments(dict(c["segs"], d_H=np.zeros((N_CHUNK, c["segs"]["nu"].size), complex),
                             d_L=np.zeros((N_CHUNK, c["segs"]["nu"].size), complex)))
    groups = [[k] for k in range(len(N_S))]
    sd = pe._single_setup(bank, seg, stores, geometry, True)
    gd = pe._glitch_setup(dict(model="windows", H=E_MODEL, L=E_MODEL), bank, seg, stores)
    seeds = np.random.SeedSequence(N_SEED).spawn(N_SEG)
    out = {k: [] for k in ("Lambda", "Lambda unscaled", "B", "B unscaled")}
    real_filter, real_scaled = pe._filter, pe._scaled
    try:
        for a in range(0, N_SEG, N_CHUNK):
            for model, key in ((None, "Lambda"), (E_MODEL, "B")):
                trajs = [white(seeds[a + i], bank.u_H, bank.u_L, bank.n_time, model) for i in range(N_CHUNK)]
                pe._filter = lambda bank_, seg_, e: trajs[e]
                for scaled_, name in ((real_scaled, key), (lambda t, e, xp: t, key + " unscaled")):
                    pe._scaled = scaled_
                    if model is None:
                        lam = pe._node_values(bank, seg, range(N_CHUNK), stores, geometry, groups, 256, False, 0, 0, 0,
                                              None, None, False, True)
                    else:
                        lam = pe._node_values(bank, seg, range(N_CHUNK), stores, geometry, groups, 256, False, 0, 0, 0,
                                              sd, gd, False, True)[1]["lamN_gHL"]
                    out[name].append(np.exp(lam[:, :-1]).mean(axis=(1, 2)))
    finally:
        pe._filter, pe._scaled = real_filter, real_scaled
    return {k: np.concatenate(v) for k, v in out.items()}


def test_noise_mean(noise_mean):
    """with the mixture (two templates between stores, one on the store of -0.3) the population averages of Lambda on
    noise and of B on data drawn from the window model have mean 1 within 3 standard errors; with the scaling of the
    data left out both move away from 1 by more than 5 standard errors (the test can fail; printed also: the paired
    difference on the same segments)"""
    lines = []
    for key in ("Lambda", "B"):
        for name in (key, key + " unscaled"):
            x = noise_mean[name]
            m, s = x.mean(), x.std() / np.sqrt(x.size)
            d = x - noise_mean[key]
            lines.append("%-17s mean %.5f +- %.5f (%+.1f se); minus the scaled one on the same segments %.5f +- %.5f"
                         % (name, m, s, (m - 1.0) / s, d.mean(), d.std() / np.sqrt(d.size)))
            if name == key:
                assert abs(m - 1.0) < 3.0 * s
            else:
                assert abs(m - 1.0) > 5.0 * s
    print("\n%d segments, w %s, stores %s, nodes %s:\n" % (N_SEG, N_W, N_GRID, N_S) + "\n".join(lines))


# ----------------------------------------------------------------------------------------------- (v) zmin


def test_zmin():
    """pe_redshift(zmin=1) on the store of test_pe_glitch (53 nodes) with the window model: every grid equals the call
    without zmin bit for bit at z >= 1 and is -inf below, the per-segment arrays are equal, the population average
    over z >= 1 (plant.population(1)) is unchanged, ln B is the prior average over z >= 1; and fewer (node, template)
    pairs are gathered, from fewer nodes"""
    from test_pe_glitch import S_NODES
    from exmarg.priors import population_m1_z
    from exmarg.search import population
    rng = np.random.default_rng(4)
    tau = np.arange(-6, 15)
    beta, delta = np.linspace(-4.0, 10.0, 12), np.linspace(-3.0, 3.0, 10)
    tables = []
    for _ in S_NODES:
        k = rng.normal(size=3)
        T = (-2.0 + k[0] + 0.5 * np.exp(beta)[None, None, :, None] + k[1] * np.linspace(-1, 1, 10)[None, None, None, :]
             + 0.1 * k[2] * np.cos(np.linspace(0, np.pi, 49))[None, :, None, None])
        tables.append(np.ascontiguousarray(np.broadcast_to(T, (tau.size, 49, 12, 10)), np.float32))
    n = S_NODES.size
    store = RedshiftStore(s=S_NODES, alpha_lo=np.full(n, 0.16), alpha_hi=np.full(n, 4e4), beta=[beta] * n,
                          delta=[delta] * n, tau_idx=tau, tables=tables, n_max=120)
    c = make_case(0.0)
    zmin = 1.0
    pairs = {}
    real = pe._read

    def count(bank, traj, tab, ids, lna, *args):
        pairs[key] = pairs.get(key, 0) + np.size(lna)
        return real(bank, traj, tab, ids, lna, *args)
    pe._read = count
    try:
        res = {}
        for key, z0 in (("all", None), ("zmin", zmin)):
            res[key] = pe.pe_redshift(c["segs"], c["bank"], store, glitch=WINDOWS, zmin=z0, ymin=0.1267,
                                      **dict(REDSHIFT, nz=60))
    finally:
        pe._read = real
    a, b = res["all"], res["zmin"]
    above = a["z"] >= zmin
    assert above.any() and (~above).any() and set(a) == set(b)
    for k in ("ln_lambda", "ln_lambda_H", "ln_lambda_L", "ln_B"):
        assert np.array_equal(a[k][:, :, above], b[k][:, :, above]), k
        assert np.all(b[k][:, :, ~above] == -np.inf) and np.isfinite(a[k][:, :, ~above]).any(), k
    for k in ("ln_c_H", "state_L", "ln_window_H", "ln_glitch_L", "wmax_H"):
        assert np.array_equal(a[k], b[k]), k
    (m, pm), (z, pz) = population_m1_z()
    keep = z >= zmin
    popz = ((m, pm), (np.concatenate([[zmin - 1e-9], z[keep]]), np.concatenate([[0.0], pz[keep]])))
    for k in ("ln_lambda", "ln_B"):
        x = population.ln_lambda_bar(a[k], a["m1"], a["z"], population=popz)
        y = population.ln_lambda_bar(b[k], b["m1"], b["z"], population=popz)
        assert np.array_equal(x, y), k
    geo_a = pe._node_geometry(c["bank"], store, 0.1267)
    geo_b = pe._node_geometry(c["bank"], store, 0.1267, float(D0 / pe._dl_of_z(zmin)[0]))
    na, nb = geo_a[0].any(axis=1).sum(), geo_b[0].any(axis=1).sum()
    print("\nzmin %g: valid pairs %d -> %d, nodes read %d -> %d, values read by the gather %d -> %d; ln B (prior over z "
          ">= zmin) %s -> %s" % (zmin, geo_a[0].sum(), geo_b[0].sum(), na, nb, pairs["all"], pairs["zmin"],
                                 np.round(a["ln_b"], 4), np.round(b["ln_b"], 4)))
    assert geo_b[0].sum() < geo_a[0].sum() and nb < na and pairs["zmin"] < pairs["all"]
    ymax = D0 / pe._dl_of_z(zmin)[0]
    for mm in range(MASSES.size):                         # each template keeps its nodes below ymax and one above
        ya, yb = (g[2][:-1, mm][g[0][:, mm]] for g in (geo_a, geo_b))
        assert (ya > ymax).sum() > 1 and (yb > ymax).sum() == 1 and np.array_equal(ya[ya <= ymax], yb[yb <= ymax])
    dl = pe._dl_of_z(b["z"])
    lnp = np.log((1.0 + b["z"]) * dl ** 2 * np.gradient(dl, b["z"]))
    Md = 2.0 * b["m1"][:, None] * (1.0 + b["z"][None, :])
    wt = np.where((Md >= 10.0) & (Md <= 500.0) & above[None, :], np.exp(lnp)[None, :], 0.0) \
        * np.gradient(b["m1"])[:, None] * np.gradient(b["z"])[None, :]
    wt /= wt.sum()
    for e in range(3):
        L = b["ln_lambda"][e]
        fin = np.isfinite(L) & (wt > 0)
        assert abs(b["ln_b"][e] - logsumexp(L[fin], b=wt[fin])) < 1e-12


# ----------------------------------------------------------------------------------------------- (vi) errors


def test_errors():
    """ValueError (before any work): the published store with unequal curves, and a template outside the stores' range,
    each naming the u_H/u_L of the data and of the tables; stores of the same asym; stores with other nodes, grids or
    lags.  Grids that differ by rounding (4e-16, as the Delta grids of the published store between its nodes) are the
    same grids"""
    c = make_case("shaped")
    run = lambda st: pe.pe_redshift(c["segs"], c["bank"], st, ymin=0.1, nodes_per_launch=1, n_loud=0, **REDSHIFT)
    with pytest.raises(ValueError, match=r"u_H/u_L from 0\.4.* to 0\.6.*cover u_H/u_L = 1 only"):
        run(inc_store(0.0))
    with pytest.raises(ValueError, match=r"cover u_H/u_L from 0\.5385 to 0\.8182 \(w from -0\.3 to -0\.1, 2 stores\)"):
        run([inc_store(-0.3), inc_store(-0.1)])
    with pytest.raises(ValueError, match="cover u_H/u_L = 1 only"):
        pe.pe_intrinsic(c["segs"], c["bank"], inc_store(0.0), ymin=0.1)
    with pytest.raises(ValueError, match="distinct asym"):
        run([inc_store(-0.3), inc_store(-0.3)])
    base = inc_store(-0.45)
    for what, other in (("s", dict(s=base.s * 1.01)), ("lags", dict(tau_idx=TAU + 1)),
                        ("beta", dict(beta=[b + 0.01 for b in base.beta])), ("delta", dict(delta=[d[1:] for d in base.delta])),
                        ("alpha", dict(alpha_lo=base.alpha_lo * 2))):
        st = RedshiftStore(**dict(vars(inc_store(-0.15)), **other))
        with pytest.raises(ValueError, match="differ in %s" % what):
            run([base, st])
    rounded = RedshiftStore(**dict(vars(inc_store(-0.15)), delta=[np.nextafter(d, np.inf) for d in base.delta],
                                   s=base.s * (1.0 + 2e-16)))
    assert [st.asym for st in pe._store_list([rounded, base])] == [-0.45, -0.15]
    with pytest.raises(ValueError, match="u_H/u_L"):
        pe._single_setup(c["bank"], c["seg"], inc_store(0.0), pe._node_geometry(c["bank"], base, 0.1))


# ----------------------------------------------------------------------------------------------- (vii) GPU


def test_gpu_agrees_with_cpu(shaped):
    """on a GPU (skipped without one): pe_redshift with the three stores and the window model, and with zmin, gives the
    arrays of the CPU: ln_lambda, ln_lambda_H, ln_lambda_L, ln_B and the per-segment arrays within 2e-4 (relative where
    |x| > 1; the GPU filters and sums in float32)"""
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
    except Exception as err:
        pytest.skip("no GPU (%s)" % err)
    c = shaped
    bank_g = mass_bank(MASSES, c["segs"], "gpu")
    lines = []
    for zmin in (None, 2.0):
        kw = dict(ymin=YMIN, nodes_per_launch=1, n_loud=0, glitch=WINDOWS, zmin=zmin, **REDSHIFT)
        a = pe.pe_redshift(c["segs"], c["bank"], c["stores"], **kw)
        b = pe.pe_redshift(c["segs"], bank_g, c["stores"], **kw)
        for k in ("ln_lambda", "ln_lambda_H", "ln_lambda_L", "ln_B"):
            lines.append("zmin %s %s: %.1e" % (zmin, k, close(b[k], a[k], 2e-4)))
        for k in ("ln_c_H", "ln_c_L", "ln_window_H", "ln_window_L", "wmax_H", "wmax_L"):
            lines.append("zmin %s %s: %.1e" % (zmin, k, close(np.asarray(b[k]), np.asarray(a[k]), 2e-4)))
    print("\nGPU against CPU, largest error relative to max(1, |x|): " + ", ".join(lines))


def test_gpu_shortcuts(shaped, monkeypatch):
    """on a GPU (skipped without one), two shortcuts that must not change a result.
    (a) ln_near_fraction in one kernel (single.NEAR_KERNEL) is the lag-by-lag sum bit for bit, with -inf entries and
    lags of either sign, and the CPU's to rounding; the sum over the opposite lags differs (the comparison can fail).
    pe_redshift with the kernel and lag by lag: the same within the scatter of two GPU runs (1e-5 relative to
    max(1, |x|) for the gathers, whose float32 sums the kernel takes in an order it does not fix; 1e-10 for the
    arrays without a gather, float64 sums with atomic adds).
    (b) The time weights checked once per segment (gather.checked_weights) are the arrays _weights makes at every
    launch, they are passed through unchanged, and a NaN is still an error that names the weights"""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception as err:
        pytest.skip("no GPU (%s)" % err)
    from exmarg.search import gather
    c = shaped
    bank_g = mass_bank(MASSES, c["segs"], "gpu")
    kw = dict(ymin=YMIN, nodes_per_launch=2, n_loud=256, n_random=256, glitch=WINDOWS, **REDSHIFT)
    assert single.NEAR_KERNEL
    a = pe.pe_redshift(c["segs"], bank_g, c["stores"], **kw)
    monkeypatch.setattr(single, "NEAR_KERNEL", False)
    d = pe.pe_redshift(c["segs"], bank_g, c["stores"], **kw)
    lines = ["%s %.1e" % (k, close(a[k], d[k], 1e-5)) for k in ("ln_lambda", "ln_B")]
    for k in ("ln_lambda_H", "ln_lambda_L", "ln_c_H", "ln_c_L", "ln_window_H", "ln_window_L", "wmax_H", "wmax_L"):
        close(np.asarray(a[k]), np.asarray(d[k]), 1e-10)
    rng = np.random.default_rng(5)
    x = rng.normal(-3.0, 4.0, 1024)
    x[rng.integers(0, 1024, 40)] = -np.inf
    xg = cp.asarray(x)
    for lags in (np.arange(-20, 21), -np.arange(-20, 21), np.array([3]), np.array([-1000, 0, 2047])):
        monkeypatch.setattr(single, "NEAR_KERNEL", False)
        loop = single.ln_near_fraction(xg, lags, cp)
        monkeypatch.setattr(single, "NEAR_KERNEL", True)
        fused = single.ln_near_fraction(xg, lags, cp)
        assert fused.shape == loop.shape and cp.asnumpy(fused).tobytes() == cp.asnumpy(loop).tobytes()
        np.testing.assert_allclose(cp.asnumpy(fused), single.ln_near_fraction(x, lags), rtol=1e-13, atol=1e-13)
    assert not np.array_equal(cp.asnumpy(single.ln_near_fraction(xg, np.arange(1, 9), cp)),
                              cp.asnumpy(single.ln_near_fraction(xg, -np.arange(1, 9), cp)))
    lk = (cp.asarray(rng.normal(-5.0, 3.0, 1024)), cp.asarray(np.where(rng.random(1024) < 0.1, -np.inf, -1.0)))
    plain = gather._weights(lk, 1024, cp, cp.float32)
    chk = gather.checked_weights(lk, 1024, cp)
    again = gather._weights(chk, 1024, cp, cp.float32)
    assert all(y is z for y, z in zip(chk, again)) and all(y.dtype == cp.float32 for y in chk)
    assert all(cp.asnumpy(y).tobytes() == cp.asnumpy(z).tobytes() for y, z in zip(plain, chk))
    assert float(chk[1].min()) == np.float32(gather.LK_MIN) and gather.checked_weights(None, 1024, cp) is None
    other = gather._weights(chk, 1024, cp, cp.float64)                # another dtype: made again, not passed through
    assert other[0].dtype == cp.float64 and other[0] is not chk[0]
    bad = (cp.asarray(np.where(np.arange(1024) == 7, np.nan, -1.0)), lk[1])
    with pytest.raises(ValueError, match=r"lkH has 1 NaN.*\[7\]"):
        gather.checked_weights(bad, 1024, cp)
    print("\nGPU, ln_near_fraction in one kernel against lag by lag, relative to max(1, |x|): " + ", ".join(lines))
