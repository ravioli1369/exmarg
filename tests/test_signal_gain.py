"""the signal gain of the conditioning (exmarg.strain: estimate_moment, normalization_gain, hand_over_gain, signal_gain)
and the model that carries it (bank.MassBank(gain=...), pe, analyse.py --unequal), each against an independent
reference and with the deliberate break that makes it fail:

  - the moments and gains of the estimators against direct draws of exponential periodograms (break: the median of an
    even number of blocks taken as the odd one's order statistic; the gain without the weights of the matched filter);
  - signal_gain of epochs conditioned by strain.condition: the counts of the blocks behind each estimate, the loud
    blocks left out, the known spectrum, and the cases it refuses (break: a stored correction factor that does not
    belong to the count);
  - gains 1 are the code without gains bit for bit (every output of pe_redshift, pe_intrinsic, window_evidences against
    the pe.py of ee47f02), and the single-detector statistics of the data do not depend on the gains at all;
  - equal gains kappa are a rescaled distance: the node values at gain kappa are those at gain 1 and the node distances
    y / kappa (break: the gain in the model powers but not in the scale of the data);
  - the noise mean of Lambda with gains is 1 (break: as above);
  - analyse.py: the bank of an epoch takes its gains with --unequal, not with --no-gain; the injections of inj carry them;
    the campaign records signal_gain as a setting of the merger stage.
The strain-level check (a signal through the real conditioning on own curves against the model with and without the
gain) is tests/test_strain_plant.py::test_strain_injection_on_own_curves_carries_the_signal_gain.
"""
import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import strain_sim as sim
from exmarg import strain
from exmarg.search import mass_bank, pe, read_segments
import test_pe_unequal as tu

ROOT = Path(__file__).resolve().parents[1]
BASE = "ee47f02"                                   # the code before the signal gain
DRAWS = 300000


def draws(n, method, size, rng):
    """g = spectrum() of n exponential periodograms over its expectation, size draws"""
    P = rng.exponential(size=(size, n))
    if method == "mean":
        return P.mean(axis=1)
    return np.median(P, axis=1) / strain.median_bias(n)


def mean_se(x):
    return x.mean(), x.std() / np.sqrt(x.size)


def lower_middle_moment(N, p):
    """E[X_(k)^p] of the k-th of N = 2k independent Exp(1) (the break of test_estimate_moment_against_draws)"""
    from scipy import integrate, special
    k = N // 2
    lc = special.gammaln(N + 1) - special.gammaln(k) - special.gammaln(k + 1)
    dens = lambda x: np.exp(lc + (k - 1) * np.log(-np.expm1(-x)) - (k + 1) * x)
    return integrate.quad(lambda x: dens(x) * x ** p, 0.0, 60.0, points=(0.3, np.log(2.0), 1.5), limit=500)[0]


# ----------------------------------------------------------------------------------------------- the factors


@pytest.mark.parametrize("method, n", [("median", 8), ("median", 9), ("median", 32), ("median", 33), ("median", 66),
                                       ("mean", 8), ("mean", 32), ("mean", 66)])
def test_estimate_moment_against_draws(method, n):
    """E[g^p] for p = 1/2, -1/2, -1 within 4 standard errors of direct draws; E[g^-1] is correction() to 1e-12; the
    break: for an even n, the median taken as the lower middle order statistic X_(n/2) alone (without Renyi's half gap
    E / n, a quadrature written here) is rejected"""
    g = draws(n, method, DRAWS, np.random.default_rng(n + (1000 if method == "mean" else 0)))
    for p in (0.5, -0.5, -1.0):
        m, se = mean_se(g ** p)
        got = strain.estimate_moment(n, p, method)
        print("%s n %d p %+.1f: %.6f, draws %.6f +- %.6f" % (method, n, p, got, m, se))
        assert abs(got - m) < 4 * se
    assert strain.estimate_moment(n, -1.0, method) == pytest.approx(strain.correction(n, method), rel=1e-12)
    if method == "median" and n % 2 == 0:
        wrong = lower_middle_moment(n, -0.5) / strain.median_bias(n) ** -0.5
        m, se = mean_se(g ** -0.5)
        assert abs(wrong - m) > 4 * se


@pytest.mark.parametrize("method, N, n_ref", [("median", 32, 66), ("median", 32, 33), ("median", 8, 18),
                                              ("mean", 32, 66)])
def test_gains_against_draws(method, N, n_ref):
    """normalization_gain = E[(c g_P)^{-1/2}]; normalization_gain x hand_over_gain = sum gamma w / sum w with gamma =
    sqrt(g_s / (c g_P)) and the matched filter's weight w = 1 / g_s of data on their own curve; E[gamma^2] = 1.  The break:
    the unweighted mean of gamma, kappa_P E[g_s^{1/2}], is not what the matched filter recovers"""
    rng = np.random.default_rng(N * 100 + n_ref)
    c = strain.correction(N, method)
    gP, gs = draws(N, method, DRAWS, rng), draws(n_ref, method, DRAWS, rng)
    gam = np.sqrt(gs / (c * gP))
    m, se = mean_se(1.0 / np.sqrt(c * gP))
    assert abs(strain.normalization_gain(N, method) - m) < 4 * se
    w = 1.0 / gs
    kappa = strain.normalization_gain(N, method) * strain.hand_over_gain(n_ref, method)
    x = gam * w / w.mean()
    m, se = mean_se(x)
    print("%s N %d n_ref %d: kappa %.5f, draws %.5f +- %.5f; unweighted %.5f" % (method, N, n_ref, kappa, m, se,
                                                                              gam.mean()))
    assert abs(kappa - m) < 4 * se
    m2, se2 = mean_se(gam ** 2)
    assert abs(m2 - 1.0) < 4 * se2
    unweighted = strain.normalization_gain(N, method) * strain.estimate_moment(n_ref, 0.5, method)
    assert abs(unweighted - m) > 4 * se


def test_published_value():
    assert strain.normalization_gain(32) * strain.hand_over_gain(66) == pytest.approx(0.972829, abs=2e-6)
    assert strain.normalization_gain(32) * strain.hand_over_gain(33) == pytest.approx(0.953438, abs=2e-6)


# ----------------------------------------------------------------------------------------------- signal_gain


@pytest.fixture(scope="module")
def epochs():
    """three epochs of N = 8 conditioned from simulated strain (median and mean), one with a loud block in H; the same
    with the known spectrum"""
    fs, T, N, pad = sim.FS, 8.0, 8, 8.0
    n = int(round((3 * (N + 1) * T + 2 * pad) * fs))
    x = {I: sim.noise(I, n, [41, k]) for k, I in enumerate("HL")}
    loud = {I: v.copy() for I, v in x.items()}
    j0 = int((pad + (N + 1) * T + 3 * T + 2.0) * fs)                 # epoch 1, block 3
    loud["H"][j0:j0 + int(0.5 * fs)] += 0.3 * np.std(x["H"]) * np.hanning(int(0.5 * fs))
    kw = dict(T=T, N=N, notches=sim.NOTCHES, pad=pad, seed=3)
    out = {m: strain.condition(x["H"], x["L"], fs, 0.0, method=m, **kw) for m in ("median", "mean")}
    out["loud"] = strain.condition(loud["H"], loud["L"], fs, 0.0, method="median", **kw)
    out["known"] = strain.condition(x["H"], x["L"], fs, 0.0, psd=(sim.psd("H"), sim.psd("L")), **kw)
    out["epoch"] = strain.condition(x["H"], x["L"], fs, 0.0, method="mean", reference="epoch", **kw)
    return out


def test_signal_gain_of_conditioned_epochs(epochs):
    for method in ("median", "mean"):
        for ep in epochs[method]:
            g = strain.signal_gain(ep)
            for I in "HL":
                want = strain.normalization_gain(8, method) * strain.hand_over_gain(ep["n_reference_" + I], method)
                assert g[I] == pytest.approx(want, rel=1e-12)
        assert [ep["n_reference_H"] for ep in epochs[method]] == [9, 18, 9]
    for ep in epochs["known"]:
        assert strain.signal_gain(ep) == {"H": 1.0, "L": 1.0}
    lo = epochs["loud"][1]
    n_loud = int(lo["loud_H"].sum())
    assert 1 <= n_loud <= 3 and lo["n_used_H"] == 9 - n_loud and lo["n_used_L"] == 9 and not lo["loud_L"].any()
    g = strain.signal_gain(lo)
    assert g["H"] == pytest.approx(strain.normalization_gain(8 - n_loud) * strain.hand_over_gain(18), rel=1e-12)
    assert g["L"] == pytest.approx(strain.normalization_gain(8) * strain.hand_over_gain(18), rel=1e-12)
    nb = epochs["loud"][0]                                # a neighbour: its hand-over curve lacks the loud blocks
    assert nb["n_reference_H"] == 9 - n_loud and strain.signal_gain(nb)["H"] == pytest.approx(
        strain.normalization_gain(8) * strain.hand_over_gain(9 - n_loud), rel=1e-12)
    print("\nmedian N 8: gains %s (middle epoch), loud epoch %s" % (strain.signal_gain(epochs["median"][1]), g))


def test_signal_gain_refuses(epochs):
    ep = dict(epochs["median"][1])
    with pytest.raises(ValueError, match="equalized"):
        strain.signal_gain(ep, own=False)
    with pytest.raises(ValueError, match="reference"):
        strain.signal_gain(epochs["epoch"][0])
    with pytest.raises(ValueError, match="does not record"):
        strain.signal_gain({k: v for k, v in ep.items() if k != "n_reference_L"})
    bad = dict(ep, correction_H=np.asarray(ep["correction_H"]) * 1.01)     # the break of the consistency check
    with pytest.raises(ValueError, match="correction factors"):
        strain.signal_gain(bad)


# ----------------------------------------------------------------------------------------------- the model


@pytest.fixture(scope="module")
def pe_base(tmp_path_factory):
    """pe.py as committed at BASE (before the signal gain), a module of exmarg.search"""
    try:
        src = subprocess.run(["git", "show", BASE + ":src/exmarg/search/pe.py"], cwd=ROOT, capture_output=True,
                             check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        pytest.skip("no git history to compare with (%s)" % err)
    path = tmp_path_factory.mktemp("base") / "pe_base.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("exmarg.search._pe_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "bank.gain" not in src and "\"gain\"" not in src
    return mod


def same(a, b):
    """bit for bit: numeric arrays with NaN where NaN; anything else by its repr"""
    if a.dtype.kind in "biufc" and b.dtype.kind in "biufc":
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
    return repr(a.tolist()) == repr(b.tolist())


def all_outputs(mod, segs, bank, st, glitch):
    """every output of pe_redshift, pe_intrinsic and window_evidences of the module mod; pe.py itself with interp
    'log' (pe.py of BASE takes ln Lambda linear between the nodes and has no interp)"""
    kw = dict(single=True) if glitch == "single" else dict(glitch=tu.WINDOWS if glitch == "windows" else glitch)
    out = {}
    for name, extra in (("pe_redshift", dict(tu.REDSHIFT, **(dict(interp="log") if mod is pe else {}))),
                        ("pe_intrinsic", {})):
        r = getattr(mod, name)(segs, bank, st, ymin=0.1, nodes_per_launch=1, n_loud=0, **kw, **extra)
        out.update({"%s %s" % (name, k): np.asarray(v) for k, v in r.items()})
    out.update({"windows %s" % k: np.asarray(v) for k, v in mod.window_evidences(segs, bank).items()})
    return out


@pytest.mark.parametrize("glitch", [None, "single", True, "windows"])
def test_gains_one_bit_identical(pe_base, glitch):
    """gains 1 (explicit) and no gains: the bank's arrays and every output of pe_redshift, pe_intrinsic and
    window_evidences bit for bit those of pe.py before the change, for one store at the templates' asym and for the
    mixture of two stores (shaped curves)"""
    n = 0
    for kind, stores in ((-0.3, tu.inc_store(-0.3)), ("shaped", [tu.inc_store(a) for a in tu.GRID])):
        c = tu.make_case(kind)
        b0 = mass_bank(tu.MASSES, c["segs"])
        b1 = mass_bank(tu.MASSES, c["segs"], gain={"H": 1.0, "L": 1.0})
        for k in ("u_H", "u_L", "u", "asym"):
            assert np.array_equal(getattr(b0, k), getattr(b1, k))
        base = all_outputs(pe_base, c["segs"], b0, stores, glitch)
        for bank in (b0, b1):
            now = all_outputs(pe, c["segs"], bank, stores, glitch)
            assert set(now) - {"pe_redshift wnet", "pe_intrinsic wnet"} == set(base)     # wnet: added after BASE
            for k in base:
                assert same(now[k], base[k]), k
            n += len(base)
    print("\nglitch %s: %d arrays bit for bit equal to pe.py of %s" % (glitch, n, BASE))


def test_data_statistics_do_not_depend_on_the_gains():
    """with gains 0.9 and 0.8: the window evidences, the glitch evidences and wmax of pe_redshift are bit for bit those of
    gains 1 (they depend on the data alone), while ln Lambda changes"""
    c = tu.make_case("shaped")
    stores = [tu.inc_store(a) for a in tu.GRID]
    b0 = mass_bank(tu.MASSES, c["segs"])
    bg = mass_bank(tu.MASSES, c["segs"], gain={"H": 0.9, "L": 0.8})
    assert np.array_equal(b0.u_H, bg.u_H) and np.array_equal(b0.u_L, bg.u_L)
    assert np.allclose(bg.u * (1 + bg.asym), 0.81 * b0.u_H, rtol=1e-14)
    assert np.allclose(bg.u * (1 - bg.asym), 0.64 * b0.u_L, rtol=1e-14)
    w0, wg = pe.window_evidences(c["segs"], b0), pe.window_evidences(c["segs"], bg)
    assert all(np.array_equal(w0[k], wg[k]) for k in w0)
    r0 = pe.pe_redshift(c["segs"], b0, stores, ymin=0.1, nodes_per_launch=1, n_loud=0, glitch=True, asym_tol=0.2,
                        **tu.REDSHIFT)
    rg = pe.pe_redshift(c["segs"], bg, stores, ymin=0.1, nodes_per_launch=1, n_loud=0, glitch=True, asym_tol=0.2,
                        **tu.REDSHIFT)
    for k in ("ln_glitch_H", "ln_glitch_L", "wmax_H", "wmax_L"):
        assert np.array_equal(r0[k], rg[k]), k
    assert not np.allclose(r0["ln_lambda"][np.isfinite(r0["ln_lambda"])],
                           rg["ln_lambda"][np.isfinite(r0["ln_lambda"])])


def node_values(c, bank, st):
    cc = tu.setup(dict(c, bank=bank), [st])
    lam = tu.node_values(cc, None, [0, 1, 2])
    return (lam[0] if isinstance(lam, tuple) else lam), cc["geometry"]


@pytest.mark.parametrize("kappa", [0.9, 0.97])
def test_equal_gains_rescale_the_distance(kappa, monkeypatch):
    """equal gains kappa: the model at distance y is the model of gains 1 at kappa y, so on every node valid in both the
    node values are those of gains 1 (to the float32 rounding of the data's scale) and the node distance is y / kappa.
    The break: the gain in the model powers but not in the scale of the data (bank.gain reset after the powers) moves
    the node values far off"""
    c = tu.make_case(-0.3)
    st = tu.inc_store(-0.3)
    l1, g1 = node_values(c, mass_bank(tu.MASSES, c["segs"]), st)
    bk = mass_bank(tu.MASSES, c["segs"], gain={"H": kappa, "L": kappa})
    lk, gk = node_values(c, bk, st)
    nn = st.s.size
    both = g1[0] & gk[0]
    assert both.sum() >= 3
    err = np.abs(lk[:, :nn][:, both] - l1[:, :nn][:, both]).max()
    print("\nkappa %g: node values off by %.2e; y ratio %s" % (kappa, err, np.unique(np.round(gk[2][:nn][both] /
                                                                                         g1[2][:nn][both], 12))))
    assert err < 2e-5
    assert np.allclose(gk[2][:nn][both], g1[2][:nn][both] / kappa, rtol=1e-12)
    broken = mass_bank(tu.MASSES, c["segs"], gain={"H": kappa, "L": kappa})
    broken.gain = {"H": 1.0, "L": 1.0}
    lb, gb = node_values(c, broken, st)
    assert np.abs(lb[:, :nn][:, both] - l1[:, :nn][:, both]).max() > 1e-2


@pytest.fixture(scope="module")
def noise_mean_gain():
    """the noise mean of tests/test_pe_unequal.py (white outputs of powers u_I, the stores of N_GRID at every node) with
    gains 0.93 (H) and 0.97 (L) in the model, and with the break (the data's scale without the gains)"""
    c = tu.make_case(0.0)
    out = {}
    for name in ("gain", "gain, data not scaled"):
        bank = mass_bank(tu.MASSES, c["segs"], gain={"H": 0.93, "L": 0.97})
        w = np.array(tu.N_W)
        uH, uL = bank.u_H * (1.0 + w) / (1.0 + bank.asym), bank.u_L * (1.0 - w) / (1.0 - bank.asym)
        # true data powers u_I with the model's asymmetry N_W: kappa_I^2 u_I = ubar' (1 +- w)
        bank.u_H, bank.u_L = uH, uL
        mH, mL = 0.93 ** 2 * uH, 0.97 ** 2 * uL
        bank.u, bank.asym = (mH + mL) / 2.0, (mH - mL) / (mH + mL)
        if name != "gain":
            bank.gain = {"H": 1.0, "L": 1.0}
        stores = [tu.inc_store(a, tu.N_S) for a in tu.N_GRID]
        geometry = pe._node_geometry(bank, stores[0], 0.1)
        assert geometry[0].all()
        seg = read_segments(dict(c["segs"], d_H=np.zeros((tu.N_CHUNK, c["segs"]["nu"].size), complex),
                                 d_L=np.zeros((tu.N_CHUNK, c["segs"]["nu"].size), complex)))
        groups = [[k] for k in range(len(tu.N_S))]
        seeds = np.random.SeedSequence(tu.N_SEED + 1).spawn(tu.N_SEG)
        real_filter = pe._filter
        res = []
        try:
            for a in range(0, tu.N_SEG, tu.N_CHUNK):
                trajs = [tu.white(seeds[a + i], uH, uL, bank.n_time) for i in range(tu.N_CHUNK)]
                pe._filter = lambda bank_, seg_, e: trajs[e]
                lam = pe._node_values(bank, seg, range(tu.N_CHUNK), stores, geometry, groups, 256, False, 0, 0, 0,
                                      None, None, False, True)
                res.append(np.exp(lam[:, :-1]).mean(axis=(1, 2)))
        finally:
            pe._filter = real_filter
        out[name] = np.concatenate(res)
    return out


def test_noise_mean_with_gains(noise_mean_gain):
    """the population average of Lambda on noise has mean 1 within 3 standard errors with the gains in the model; with
    the data's scale left without them (the break) it is off by more than 5"""
    for name, x in noise_mean_gain.items():
        m, se = mean_se(x)
        print("\n%-24s mean %.5f +- %.5f (%+.1f se)" % (name, m, se, (m - 1) / se))
        if name == "gain":
            assert abs(m - 1.0) < 3 * se
        else:
            assert abs(m - 1.0) > 5 * se


# ----------------------------------------------------------------------------------------------- analyse.py


def _script(name, folder):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / folder / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ana():
    import test_analyse as ta
    return ta


def estimated_epoch(ta, gps, seed, n_used=(33, 31), n_ref=(66, 60)):
    """test_analyse's unequal epoch with the bookkeeping of an estimated spectrum (median, N 32, neighbours)"""
    ep, pre = ta.make_unequal_epoch(gps, seed)
    loud = {"H": np.zeros(33, bool), "L": np.zeros(33, bool)}
    loud["L"][[5, 6]] = n_used[1] < 33
    count = {I: (u - 1) for I, u in zip("HL", n_used)}
    ep.update(psd="estimated", method="median", reference="neighbours", N=32, correction=strain.correction(32, "median"),
              n_used_H=n_used[0], n_used_L=n_used[1], n_reference_H=n_ref[0], n_reference_L=n_ref[1],
              loud_H=loud["H"], loud_L=loud["L"],
              correction_H=np.full(33, strain.correction(count["H"], "median")),
              correction_L=np.where(loud["L"], strain.correction(n_used[1], "median"),
                                    strain.correction(count["L"], "median")))
    ep["analysable"] = ~ep["loud_H"] & ~ep["loud_L"] & ep["usable"]
    return ep, pre


def test_variants_carry_the_gain(ana):
    """own curves with gain: every variant's dict holds the epoch's signal gain; the injections of inj are kappa_I times
    those of the code without gains (in both stages: also when the model takes none); without own curves nothing"""
    ep, _ = estimated_epoch(ana, 1266934296, 101)
    kappa = strain.signal_gain(ep)
    assert kappa["H"] == pytest.approx(0.972829, abs=2e-6)
    assert kappa["L"] == pytest.approx(strain.normalization_gain(30) * strain.hand_over_gain(60), rel=1e-12)
    s = ana.plant.Sampler(ana.plant.population(ana.analyse.ZMIN), ana.plant.MDET)
    cfg = dict(seed=7, nbar_inj=0.5)
    var, rows, inj = ana.analyse.variants(cfg, ep, ana.analyse.VARIANTS, s, own=True, gain=True)
    for v in ana.analyse.VARIANTS:
        assert var[v]["gain"] == kappa
    var0, _, inj0 = ana.analyse.variants(cfg, ep, ("slide+0.5", "inj"), s, own=True, gain=False)
    assert "gain" not in var0["inj"]
    h = {I: var["inj"]["d_" + I] - var["slide+0.5"]["d_" + I] for I in "HL"}
    h0 = {I: var0["inj"]["d_" + I] - var0["slide+0.5"]["d_" + I] for I in "HL"}
    ep_known = dict(ep, psd="known")
    var1, _, _ = ana.analyse.variants(cfg, ep_known, ("slide+0.5", "inj"), s, own=True)
    h1 = {I: var1["inj"]["d_" + I] - var1["slide+0.5"]["d_" + I] for I in "HL"}
    m = inj["merger"]
    assert m.any()
    for I in "HL":
        assert np.allclose(h[I][m], kappa[I] * h1[I][m], rtol=1e-6, atol=1e-6 * np.abs(h1[I]).max())
        assert np.array_equal(h[I], h0[I])
    eq, _, _ = ana.analyse.variants(cfg, ep, ("zero",), s)
    assert "gain" not in eq["zero"]


def test_driver_bank_takes_the_epoch_gains(ana, tmp_path, monkeypatch):
    """--unequal: the factory of make_analysis builds each epoch's bank with its signal gain (mass_bank recorded), the
    campaign records signal_gain epoch; --no-gain: gains 1 and signal_gain none; --no-gain without --unequal is refused;
    a window stage does not record signal_gain"""
    seg_dir = tmp_path / "week"
    os.makedirs(seg_dir)
    lines = ["# estimated\n"]
    for i, gps in enumerate((1266933768, 1266934032)):
        ep, _ = estimated_epoch(ana, gps, 100 + i, n_used=(33, 33 - 2 * i), n_ref=(66 - i, 60))
        strain.write_epoch(str(seg_dir / ("epoch_%d.npz" % gps)), ep)
        lines.append("epoch_%d.npz %d 33 127 127 0 0 0 33 33 1 0.8\n" % (gps, gps))
    (seg_dir / "manifest.txt").write_text("".join(lines))
    seen = []

    class Bank:
        def __init__(self, gain):
            self.gain = gain

    import exmarg.search as search
    monkeypatch.setattr(search, "mass_bank", lambda b, seg, device, gain=None: seen.append(gain) or Bank(gain))
    factory = ana.analyse.make_analysis(None, "cpu")
    for no_gain in (False, True):
        seen.clear()
        argv = ["x", "cpu", str(seg_dir), "--out", str(tmp_path / ("o%d" % no_gain)), "--unequal", "--variants", "zero"]
        a = ana.analyse.parse(argv + (["--no-gain"] if no_gain else []))[1]
        for gps in (1266933768, 1266934032):
            ep = strain.read_epoch(str(seg_dir / ("epoch_%d.npz" % gps)))
            var, _, _ = ana.analyse.variants(dict(seed=7, nbar_inj=0.3), ep, ("zero",), None, True, not no_gain)
            factory(var["zero"])
            want = None if no_gain else strain.signal_gain(ep)
            assert seen[-1] == want
        cfg = ana.analyse.campaign(a, str(seg_dir), "zero")
        assert cfg["signal_gain"] == ("none" if no_gain else "epoch")
    assert seen[0] is None
    with pytest.raises(SystemExit):
        ana.analyse.parse(["x", "cpu", str(seg_dir), "--out", "o", "--no-gain"])
    aw = ana.analyse.parse(["x", "cpu", str(seg_dir), "--out", "o", "--unequal", "--stage", "windows"])[1]
    assert "signal_gain" not in ana.analyse.campaign(aw, str(seg_dir), "zero")
    assert "signal_gain" in ana.plant.MERGER_KEYS
