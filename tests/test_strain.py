"""exmarg.strain on simulated coloured strain (tests/strain_sim.py): tests 1, 2 and 4 of docs/real_data.md.

Data: K = 4 epochs of N + 1 = 33 segments of 8 s at 4096 Hz in each detector, with known spectra (wall below 15 Hz,
lines at 60, 120, 180 Hz and near 500 Hz, notched).  Expected values come from the simulation: unit power, the true
spectra, the injected amplitude, phase and time.

Tolerance of a mean power: a mean of M values of |d|^2 / sig2 has standard error sqrt(V / M), with V = 1.2 bounding
the variance of one value (|e|^2 = |W|^2 / (c g): 2 E[1/(c g)^2] - 1 = N / (N - 2) = 1.07 for the mean of N = 32
periodograms, about 1.14 for the median, 1 for a known spectrum).  The segments of an epoch share their estimates and
are anti-correlated, so this is conservative.  Every check is at 4 standard errors.  The power per bin cannot see a
sig2 that depends on the segment's own data (|d|^2 / sig2 = |e|^2 whatever sig2 is), so test 1 also measures the
matched-filter power E_0|z|^2 / u with the weights of the analysis.

Test 2: with known spectra the conditioned template is exact (|z| / u = A within 2e-3); with estimated ones each bin
carries the gain sqrt(s / (c Phat_{-i})), of mean 1 - O(1/8N) and rms about 0.1, which averages to a 0.6 percent
scatter over the ~1000 bins of the template: 3 percent is about 5 sigma.
"""
import numpy as np
import pytest

import strain_sim as sim
from exmarg import strain
from exmarg.profiles import profile
from exmarg.search import mass_bank, read_segments
from exmarg.search.segments import D0

FS, T, N, K, PAD, GPS0 = sim.FS, 8.0, 32, 4, 8.0, 1264316160.0
V, NSIG = 1.2, 4.0
PSD = (sim.psd("H"), sim.psd("L"))
BANDS = ((20.0, 22.0), (22.0, 40.0), (40.0, 80.0), (80.0, 160.0), (160.0, 320.0), (320.0, 512.0))
RUNS = {"mean": dict(method="mean", equal_noise=False), "median": dict(method="median", equal_noise=False),
        "mean c=1": dict(method="mean", correct=False, equal_noise=False),
        "median c=1": dict(method="median", correct=False, equal_noise=False),
        "known": dict(psd=PSD, equal_noise=False), "mean eq": dict(method="mean"), "known eq": dict(psd=PSD),
        "mean own": dict(method="mean", reference="epoch"), "median eq": dict(method="median"),
        "wrong": dict(psd=PSD[::-1], equal_noise=False)}

# test 2: a binary of detector-frame total mass 60 in epoch 1, segment 10, merging 5 s into it (L 31 samples later)
M_INJ, EPOCH, ROW = 60.0, 1, 10
T_SEG = GPS0 + PAD + (EPOCH * (N + 1) + ROW) * T
T_INJ = {"H": T_SEG + 5.0, "L": T_SEG + 5.0 + 31 / FS}
RHO, PHASE = {"H": 20.0, "L": 15.0}, {"H": 1.1, "L": -2.3}


@pytest.fixture(scope="module")
def noise():
    n = int(round((K * (N + 1) * T + 2 * PAD) * FS))
    return {I: sim.noise(I, n, [7, k]) for k, I in enumerate("HL")}


@pytest.fixture(scope="module")
def injection(noise):
    """the noise with the injection, the amplitudes A_I (relative to the template at D0) and the template power u with
    the true equalized spectrum"""
    nu, _, a, psi0 = profile(M_INJ / 2, M_INJ / 2, 0.0, 0.0, T, 20.0, 512.0, 1.0)
    f, h = sim.chirp(M_INJ)
    k = np.rint(nu / T * 64.0).astype(int)
    assert np.allclose(h[k], a * np.exp(1j * psi0), rtol=1e-10, atol=0.0)      # the injection is the template
    keep = strain.response(nu / T, notches=sim.NOTCHES) == 1.0
    sig2_c = np.maximum(T * PSD[0](nu / T) / 2, T * PSD[1](nu / T) / 2)
    u_true = np.sum((a[keep] / D0) ** 2 / sig2_c[keep])
    x, A = {}, {}
    for I in "HL":
        A[I] = RHO[I] / np.sqrt(2 * u_true)
        s = sim.time_series(f, A[I] * np.exp(-1j * PHASE[I]) * h / D0, 32.0)
        j0 = int(round((T_INJ[I] - GPS0) * FS)) - int(32 * FS)
        x[I] = noise[I].copy()
        x[I][j0:j0 + s.size] += s
        assert np.sum(s[: int(28 * FS)] ** 2) + np.sum(s[int(34 * FS):] ** 2) < 1e-8 * np.sum(s ** 2)  # 1 s from edges
    return x, A, u_true


@pytest.fixture(scope="module")
def run(noise, injection):
    cache = {}

    def get(name):
        if name not in cache:
            x = injection[0] if name.endswith("inj") else noise
            opts = RUNS[name.replace(" inj", "")]
            cache[name] = strain.condition(x["H"], x["L"], FS, GPS0, T=T, N=N, notches=sim.NOTCHES, pad=PAD, seed=3,
                                           **opts)
        return cache[name]
    return get


def normalized(eps, I):
    """d_I / sqrt(sig2) of every segment, and the bins that are not notched"""
    z = np.concatenate([e["d_" + I].astype(complex) / np.sqrt(e.get("sig2_" + I, e.get("sig2"))) for e in eps])
    return z, ~eps[0]["notched"]


def band_masks(f, notched, keep=None):
    """bins of each band of BANDS (the last one closed), the bins within 3 of a notched one, all; only bins in keep"""
    keep = ~notched if keep is None else keep
    near = np.zeros(f.size, bool)
    for s in range(1, 4):
        near |= np.roll(notched, s) | np.roll(notched, -s)
    out = {"%g-%g Hz" % (a, b): keep & (f >= a) & ((f <= b) if b == BANDS[-1][1] else (f < b)) for a, b in BANDS}
    out["next to notches"] = keep & near
    out["all"] = keep
    return out


def power_table(eps, I):
    """{band: (mean power, standard error)} of |d_I|^2 / sig2"""
    z, keep = normalized(eps, I)
    p = np.abs(z) ** 2
    return {k: (p[:, m].mean(), np.sqrt(V / p[:, m].size)) for k, m in band_masks(eps[0]["f"], ~keep).items()}


def neighbour_correlation(eps, I):
    """(rho, its standard error): mean of z_k conj(z_{k+1}) over bins (both kept) and segments, over mean |z|^2"""
    z, keep = normalized(eps, I)
    pair = keep[:-1] & keep[1:]
    prod = (z[:, :-1] * np.conj(z[:, 1:]))[:, pair]
    return abs(prod.mean()) / np.mean(np.abs(z[:, keep]) ** 2), 1.0 / np.sqrt(prod.size)


def report(name, table):
    print("\n%s: " % name + ", ".join("%s %.4f +- %.4f" % (k, m, s) for k, (m, s) in table.items()))


# test 1


@pytest.mark.parametrize("name", ["mean", "median", "known"])
def test_whitened_bins_have_unit_power(run, name):
    eps = run(name)
    for I in "HL":
        table = power_table(eps, I)
        report("%s %s" % (name, I), table)
        for band, (m, s) in table.items():
            assert abs(m - 1.0) < NSIG * s, (I, band, m, s)
        mon = np.concatenate([e["monitor_" + I] for e in eps]).mean(axis=0)
        assert np.all(np.abs(mon - 1.0) < 0.02), mon


@pytest.mark.parametrize("name", ["mean", "median", "known"])
def test_neighbouring_bins_are_uncorrelated(run, name):
    eps = run(name)
    for I in "HL":
        rho, s = neighbour_correlation(eps, I)
        z, keep = normalized(eps, I)
        lag = np.mean(z[1:, keep] * np.conj(z[:-1, keep]))           # same bin, consecutive segments
        print("\n%s %s: |rho(k, k+1)| = %.4f, |rho(i, i+1)| = %.4f, standard error %.4f" % (name, I, rho, abs(lag), s))
        assert rho < NSIG * s and abs(lag) < NSIG * s


@pytest.mark.parametrize("method", ["mean", "median"])
def test_uncorrected_estimate_measures_the_correction(run, method):
    """without c the power is E[1/g]: N/(N-1) for the mean, for the median from direct draws of N exponentials"""
    rng = np.random.default_rng(11)
    if method == "mean":
        expect = N / (N - 1.0)
    else:
        med = np.median(rng.exponential(size=(400000, N)), axis=1)
        expect = np.mean(1.0 / med) * np.mean(med)
    eps = run(method + " c=1")
    p = np.concatenate([np.abs(normalized(eps, I)[0][:, ~eps[0]["notched"]]) ** 2 for I in "HL"], axis=1)
    s = np.sqrt(V / p.size)
    print("\n%s without correction: power %.4f +- %.4f, expected E[1/g] = %.4f, correction() = %.4f"
          % (method, p.mean(), s, expect, strain.correction(N, method)))
    assert abs(p.mean() - expect) < NSIG * s + 1e-3
    assert abs(p.mean() - 1.0) > NSIG * s
    assert abs(strain.correction(N, method) - expect) < 2e-3


@pytest.mark.parametrize("name", ["mean eq", "median eq", "mean own", "known eq"])
def test_matched_filter_noise_power_is_the_template_power(run, name):
    """what the likelihood needs: E_0|z(t)|^2 = u for every template, with the weights 1/sig2 of the analysis (the
    unit power per bin above cannot see a sig2 that depends on the segment's own data).  The mean of |z|^2 over the
    arrival times is sum_k |W_k|^2 |d_k|^2 / sig2_k^2; its standard error over n segments is
    sqrt(V sum w^2 / (sum w)^2 / n), w = |W|^2 / sig2"""
    eps = run(name)
    out = {}
    for e in eps:
        bank = mass_bank(np.array([20.0, 60.0, 200.0]), e, "cpu")
        W2 = np.abs(bank.W.astype(complex)) ** 2
        w = W2 / e["sig2"]
        for i in range(e["d_H"].shape[0]):
            tr = bank.trajectories({I: e["d_" + I][i] for I in "HL"}, {I: e["sig2"] for I in "HL"})
            for I in "HL":
                out.setdefault(I, []).append(np.mean(np.exp(2.0 * tr["lq" + I].astype(float)), axis=1) / bank.u)
        neff = w.sum(1) ** 2 / (w ** 2).sum(1)
    for I in "HL":
        r = np.array(out[I])
        s = np.sqrt(V / neff / r.shape[0])
        print("\n%s %s: E|z|^2 / u for M = 20, 60, 200: %s +- %s (n_eff %s)"
              % (name, I, np.round(r.mean(0), 4), np.round(s, 4), np.round(neff).astype(int)))
        assert np.all(np.abs(r.mean(0) - 1.0) < NSIG * s)


def test_unit_power_fails_without_whitening_or_with_a_wrong_spectrum(noise, run):
    """the checks above fail for a rectangular window on the coloured strain and when whitening with a wrong spectrum"""
    n = int(T * FS)
    eps = run("known")
    keep = ~eps[0]["notched"]
    nu, f = eps[0]["nu"], eps[0]["f"]
    x = noise["H"][int(PAD * FS): int(PAD * FS) + K * (N + 1) * n].reshape(-1, n)
    z = np.fft.rfft(x, axis=1)[:, nu] / FS / np.sqrt(T * PSD[0](f) / 2)
    p = np.abs(z) ** 2
    low = keep & (f < 40.0)
    pair = keep[:-1] & keep[1:] & (f[:-1] < 40.0)
    rho = abs(np.mean((z[:, :-1] * np.conj(z[:, 1:]))[:, pair])) / p[:, low].mean()
    print("\nrectangular window on coloured data, 20-40 Hz: power %.3g, |rho(k, k+1)| = %.3f" % (p[:, low].mean(), rho))
    assert abs(p[:, low].mean() - 1.0) > NSIG * np.sqrt(V / p[:, low].size)
    assert rho > NSIG / np.sqrt(pair.sum() * p.shape[0])
    wrong = {I: power_table(run("wrong"), I) for I in "HL"}
    report("H whitened with the L spectrum", wrong["H"])
    report("L whitened with the H spectrum", wrong["L"])
    for I in "HL":
        assert any(abs(m - 1.0) > NSIG * s for m, s in wrong[I].values())


def test_bin_variance_next_to_notches():
    """the bin variance v of a rectangular cut is the Fejer-smoothed R^2 (direct integral), 0.9 at the edge of a notch;
    with a known spectrum the bins there have power v sig2 (10 epochs of known-spectrum conditioning)"""
    n = int(T * FS)
    nu = np.array([160, 476, 484, 3976])                          # 20 Hz, the two edges of the 60 Hz notch, 497 Hz
    phi = np.arange(-0.25, 0.25, 1.0 / (64 * n)) + 0.5 / (128 * n)
    fejer = np.sin(np.pi * n * phi) ** 2 / (n * np.sin(np.pi * phi) ** 2)
    direct = [np.sum(strain.response(np.abs(k / n + phi) * FS, notches=sim.NOTCHES) ** 2 * fejer) / (64 * n)
              for k in nu]
    v = strain.bin_variance(FS, T, nu, notches=sim.NOTCHES)
    print("\nbin variance: %s, direct integral %s" % (np.round(v, 5), np.round(direct, 5)))
    assert np.allclose(v, direct, rtol=0.0, atol=2e-4)
    m = int(round((10 * (N + 1) * T + 2 * PAD) * FS))
    x = {I: sim.noise(I, m, [8, k]) for k, I in enumerate("HL")}
    eps = strain.condition(x["H"], x["L"], FS, GPS0, T=T, N=N, notches=sim.NOTCHES, pad=PAD, psd=PSD, equal_noise=False)
    notched = eps[0]["notched"]
    edge = ~notched & (np.roll(notched, 1) | np.roll(notched, -1))
    for I in "HL":
        p = np.abs(normalized(eps, I)[0][:, edge]) ** 2
        s = np.sqrt(V / p.size)
        print("known spectrum, %s, the %d bins at notch edges: power %.4f +- %.4f" % (I, edge.sum(), p.mean(), s))
        assert abs(p.mean() - 1.0) < NSIG * s


def test_correction_factors_and_leave_one_out():
    rng = np.random.default_rng(5)
    assert strain.correction(N, "mean") == N / (N - 1.0) and strain.correction(N, "mean", "estimate") == N / (N - 2.0)
    for n in (15, 32, 33):
        med = np.median(rng.exponential(size=(400000, n)), axis=1)
        g = med / np.mean(med)
        assert abs(strain.median_bias(n) / np.mean(med) - 1) < 2e-3
        assert abs(strain.correction(n, "median") - np.mean(1 / g)) < 3e-3
        assert abs(strain.correction(n, "median", "estimate") - np.mean(g ** -2) / np.mean(1 / g)) < 5e-3
    P = rng.exponential(size=(7, 50))
    for method, fn in (("mean", np.mean), ("median", np.median)):
        brute = np.array([fn(np.delete(P, i, axis=0), axis=0) for i in range(7)])
        scale = strain.median_bias(6) if method == "median" else 1.0
        assert np.allclose(strain.loo_spectrum(P, method) * scale, brute)


# test 2


def peak(eps, d, I):
    """index, modulus over u, phase and modulus over sqrt(u) of the matched-filter maximum of one detector"""
    seg = eps[EPOCH]
    bank = mass_bank(np.array([M_INJ]), seg, "cpu")
    sig2 = seg["sig2"]
    tr = bank.trajectories({"H": d["H"], "L": d["L"]}, {"H": sig2, "L": sig2})
    j = int(np.argmax(tr["lq" + I][0]))
    q = float(np.exp(tr["lq" + I][0, j]))
    return j, q / bank.u[0], float(tr["ph" + I][0, j]), q / np.sqrt(bank.u[0])


@pytest.mark.parametrize("name, tol", [("known eq", 2e-3), ("mean eq", 0.03), ("mean own", 0.03)])
def test_injected_template_recovered_at_its_time_and_amplitude(run, injection, name, tol):
    _, A, _ = injection
    clean, inj = run(name), run(name + " inj")
    seg, seg0 = inj[EPOCH], clean[EPOCH]
    assert seg["gps"][ROW] == T_SEG
    n_time = 8 * int(seg["nu"].max())
    noiseless = {I: seg["d_" + I][ROW].astype(complex) - seg0["d_" + I][ROW].astype(complex) for I in "HL"}
    noisy = {I: seg["d_" + I][ROW].astype(complex) for I in "HL"}
    for I in "HL":
        expect_j = int(round((T_INJ[I] - T_SEG) * n_time / T))
        j, amp, ph, snr = peak(inj, noiseless, I)
        dphi = np.angle(np.exp(1j * (ph - PHASE[I])))
        print("\n%s %s noiseless: peak at %d (expected %d), |z|/u = %.5f (A = %.5f, ratio %.5f), phase error %.2e, "
              "rho = %.3f (injected %.1f)"
              % (name, I, j, expect_j, amp, A[I], amp / A[I], dphi, np.sqrt(2) * snr, RHO[I]))
        assert j == expect_j
        assert abs(amp / A[I] - 1.0) < tol and abs(dphi) < tol
        jn, _, _, snr_n = peak(inj, noisy, I)
        print("%s %s noisy: peak at %d, |z|/sqrt(u) = %.3f (expected %.3f)" % (name, I, jn, snr_n, RHO[I] / np.sqrt(2)))
        assert abs(jn - expect_j) <= 3
        assert abs(snr_n - RHO[I] / np.sqrt(2)) < NSIG * np.sqrt(0.5) + tol * RHO[I]


# test 4


@pytest.mark.parametrize("name", ["mean eq", "mean own", "known eq"])
def test_equalized_detectors_have_the_common_variance(run, name):
    eps = run(name)
    f, keep = eps[0]["f"], ~eps[0]["notched"]
    true_c = np.maximum(T * PSD[0](f) / 2, T * PSD[1](f) / 2)
    for I in "HL":
        table = power_table(eps, I)
        report("%s %s, |d|^2 / sig2" % (name, I), table)
        for band, (m, s) in table.items():
            assert abs(m - 1.0) < NSIG * s, (I, band, m, s)
    # against the true spectra where the bin variance v of the cut is 1 within 2e-3 (above 22 Hz and more than 16
    # bins from a notch).  Known spectra: the variance is the true larger spectrum.  Estimated: the curve of each
    # detector before step 5 is its true spectrum ((N + 1) / N times it for reference 'epoch', the sum over N), within
    # 1 / sqrt((N + 1) bins) per band; their max is biased up where H and L are close, so it is not compared.
    near = np.zeros(f.size, bool)
    for s in range(-16, 17):
        near |= np.roll(~keep, s)
    far = keep & ~near & (f > 22.0)
    for I in "HL":
        if name.startswith("known"):
            p = np.abs(np.concatenate([e["d_" + I].astype(complex) for e in eps])) ** 2 / true_c
            what, expect = "|d|^2 / true max spectrum", 1.0
        else:
            p = np.stack([e["sig2_ref_" + I] for e in eps]) / (T * PSD["HL".index(I)](f) / 2)
            what, expect = "sig2_ref / true spectrum", (N + 1.0) / N if name == "mean own" else 1.0
        for band, m in band_masks(f, ~keep, far).items():
            if not m.any():
                continue
            s = np.sqrt(V / p[:, m].size) if name.startswith("known") else np.sqrt(1.0 / ((N + 1) * m.sum()))
            print("%s %s, %s, %s: %.4f +- %.4f (expected %.4f)" % (name, I, what, band, p[:, m].mean(), s, expect))
            assert abs(p[:, m].mean() - expect) < NSIG * s, (I, band)
    if name.startswith("known"):
        assert np.allclose(eps[0]["sig2"][far] / true_c[far], 1.0, rtol=0.0, atol=2e-3)
    zH, zL = normalized(eps, "H")[0][:, keep], normalized(eps, "L")[0][:, keep]
    rho = abs(np.mean(zH * np.conj(zL)))
    assert rho < NSIG / np.sqrt(zH.size), rho


def test_equalization_adds_the_missing_variance(run):
    """before step 5 the better detector L is quieter than the common curve; the noise of step 5 is reproducible"""
    eps = run("known")
    f = eps[0]["f"]
    m = ~eps[0]["notched"] & (f > 22.0) & (f < 100.0)
    true_c = np.maximum(T * PSD[0](f) / 2, T * PSD[1](f) / 2)
    pL = np.mean(np.abs(np.concatenate([e["d_L"].astype(complex) for e in eps])[:, m]) ** 2 / true_c[m])
    print("\nL before step 5, 22-100 Hz: power over the H spectrum %.4f (1/1.44 = %.4f)" % (pL, 1 / 1.44))
    assert abs(pL - 1 / 1.44) < NSIG * np.sqrt(V / (m.sum() * len(eps) * (N + 1)))
    d = np.ones((2, 3), complex)
    a, b, c = (strain.equalize(d, d, np.full(3, 2.0), np.ones(3), s) for s in (9, 9, 10))
    assert np.array_equal(a[1], b[1]) and not np.array_equal(a[1], c[1]) and np.array_equal(a[0], d)
    with pytest.raises(ValueError):
        strain.condition(d[0].real, d[0].real, FS, GPS0, method="median", reference="epoch")


def test_epoch_file_round_trip(run, tmp_path):
    ep = run("mean eq")[0]
    path = tmp_path / "epoch.npz"
    strain.write_epoch(path, ep)
    seg = read_segments(str(path), rows=[0, 5])
    assert np.array_equal(seg["d_H"], ep["d_H"][[0, 5]].astype(complex)) and np.array_equal(seg["sig2_L"], ep["sig2"])
    back = strain.read_epoch(path)
    assert back["method"] == "mean" and back["N"] == N and back["notches"].shape == (len(sim.NOTCHES), 2)
    assert not back["event"].any() and back["gps"][0] == ep["gps_epoch"]
