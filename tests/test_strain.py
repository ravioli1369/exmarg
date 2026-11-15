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


# loud blocks (strain.loud_blocks): the same noise with blocks of H made 30, 900 and 1e6 times louder


PLANTS = ((0, 7, 30.0), (1, 12, 900.0), (2, 20, 1e6))       # (epoch, block, power factor), detector H


def plant(x, blocks, factor, seed):
    """x with each (epoch, block) made `factor` times louder: an independent realization of the line-free noise of H
    (high-passed below 20 Hz), scaled by sqrt(factor - 1), added under a window that is 1 over the block but its first
    and last second (sin^2 ramps), so that the whitening filter carries little of it into the next blocks"""
    m = int(T * FS)
    fk = np.fft.rfftfreq(m, 1.0 / FS)
    lines = sum(h * np.exp(-0.5 * ((fk - f0) / sim.WIDTH) ** 2) for f0, h in sim.LINES["H"])
    S = PSD[0](fk) / (1.0 + lines) * np.sin(0.5 * np.pi * np.clip((fk - 18.0) / 2.0, 0.0, 1.0)) ** 2
    S[0] = S[-1] = 0.0
    h = int(FS)
    w = np.ones(m)
    w[:h] = np.sin(0.5 * np.pi * (np.arange(h) + 0.5) / h) ** 2
    w[-h:] = w[:h][::-1]
    y = x.copy()
    rng = np.random.default_rng(seed)
    for e, b in blocks:
        g = rng.standard_normal((2, fk.size))
        z = np.fft.irfft(np.sqrt(m * FS * S / 4.0) * (g[0] + 1j * g[1]), m)
        j0 = int(PAD * FS) + (e * (N + 1) + b) * m
        y[j0:j0 + m] += np.sqrt(factor - 1.0) * w * z
    return y


@pytest.fixture(scope="module")
def loud(noise):
    """conditioned with and without flags: the three plants, and an epoch with 20 loud blocks next to one with 10"""
    x = noise["H"]
    for k, (e, b, factor) in enumerate(PLANTS):
        x = plant(x, [(e, b)], factor, [11, k])
    many = plant(noise["H"], [(1, b) for b in range(20)] + [(2, b) for b in range(3, 33, 3)], 30.0, 12)
    hidden = plant(plant(noise["H"], [(1, 5), (1, 15), (1, 25)], 900.0, 13), [(1, 10)], 1.25, 14)
    kw = dict(T=T, N=N, notches=sim.NOTCHES, pad=PAD, seed=3, method="median")
    return {"flag": strain.condition(x, noise["L"], FS, GPS0, **kw),
            "no flag": strain.condition(x, noise["L"], FS, GPS0, flag=False, **kw),
            "many": strain.condition(many, noise["L"], FS, GPS0, **kw),
            "hidden": strain.condition(hidden, noise["L"], FS, GPS0, **kw)}


def unflagged(eps, I, rows=None):
    """whitened data before step 5 of the analysable segments (or of `rows`, per epoch), and the kept bins"""
    keep = ~eps[0]["notched"]
    out = []
    for k, e in enumerate(eps):
        r = e["analysable"] if rows is None else rows[k]
        out.append(strain.unequalized(e)[I][r] / np.sqrt(np.where(keep, e["sig2_ref_" + I], 1.0)))
    return np.concatenate(out), keep


def test_loud_blocks_are_flagged(loud):
    """the planted blocks and nothing else, except the two blocks next to the 1e6 one: the whitening filter carries
    1e-4 of its power into them (80 to 160 times the noise), so they are loud too"""
    eps, plain = loud["flag"], loud["no flag"]
    flagged = {(e, int(b)) for e, ep in enumerate(eps) for b in np.flatnonzero(ep["loud_H"])}
    planted = {(e, b) for e, b, _ in PLANTS}
    leak = {(2, 19), (2, 21)}
    print("\nflagged %s; monitors without flags of the blocks next to the 1e6 one: %s" % (
        sorted(flagged), [round(float(plain[e]["monitor_H"][b].max()), 1) for e, b in sorted(leak)]))
    assert flagged == planted | leak
    assert all(plain[e]["monitor_H"][b].max() > 10.0 for e, b in leak)
    assert not any(ep["loud_L"].any() for ep in eps) and all(ep["usable"] for ep in eps)
    assert [ep["n_used_H"] for ep in eps] == [32, 32, 30, 33]
    assert [ep["n_reference_H"] for ep in eps] == [32, 62, 65, 30]
    assert all(np.array_equal(ep["analysable"], ~ep["loud_H"]) for ep in eps)


def test_other_segments_have_unit_power_next_to_loud_blocks(loud):
    """|e|^2 of the analysable segments, per band, is 1 within the sampling error; without flags the segments that
    share an epoch with a loud block are low (0.96 for 30, 0.95 for 900 or 1e6 times the power)"""
    eps, plain = loud["flag"], loud["no flag"]
    for I in "HL":
        z, keep = unflagged(eps, I)
        p = np.abs(z) ** 2
        table = {k: (p[:, m].mean(), np.sqrt(V / p[:, m].size)) for k, m in band_masks(eps[0]["f"], ~keep).items()}
        report("analysable segments, %s" % I, table)
        for band, (m, s) in table.items():
            assert abs(m - 1.0) < NSIG * s, (I, band, m, s)
    rows = [np.isin(np.arange(N + 1), [b for pe, b, _ in PLANTS if pe == e] + ([19, 21] if e == 2 else []),
                    invert=True) for e in range(K)]
    z, keep = unflagged(plain, "H", rows)
    per_epoch = [np.mean(np.abs(z[k * 30:(k + 1) * 30][:, keep]) ** 2) for k in range(3)]
    print("without flags, the other segments of the epochs with a loud block: %s" % np.round(per_epoch, 4))
    p = np.abs(z[:, keep]) ** 2
    assert p.mean() < 1.0 - NSIG * np.sqrt(V / p.size)


def test_matched_filter_noise_power_next_to_loud_blocks(loud):
    """E_0|z|^2 / u over the analysable segments after step 5 is 1 within its error (it was 0.96 without flags)"""
    for name in ("flag", "no flag"):
        eps = loud[name]
        out = {}
        for k, e in enumerate(eps):
            bank = mass_bank(np.array([20.0, 60.0, 200.0]), e, "cpu")
            w = np.abs(bank.W.astype(complex)) ** 2 / e["sig2"]
            neff = w.sum(1) ** 2 / (w ** 2).sum(1)
            rows = np.flatnonzero(loud["flag"][k]["analysable"])
            for i in rows:
                tr = bank.trajectories({I: e["d_" + I][i] for I in "HL"}, {I: e["sig2"] for I in "HL"})
                for I in "HL":
                    out.setdefault(I, []).append(np.mean(np.exp(2.0 * tr["lq" + I].astype(float)), axis=1) / bank.u)
        for I in "HL":
            r = np.array(out[I])
            s = np.sqrt(V / neff / r.shape[0])
            print("\n%s, %s: E|z|^2 / u for M = 20, 60, 200: %s +- %s" % (name, I, np.round(r.mean(0), 4),
                                                                       np.round(s, 4)))
            if name == "flag":
                assert np.all(np.abs(r.mean(0) - 1.0) < NSIG * s)
            elif I == "H":
                assert np.all(r.mean(0) < 1.0 - NSIG * s)


def test_hand_over_curve_leaves_out_loud_blocks(loud):
    """sig2_ref of every epoch is the true spectrum of H within the error of a median of its n_reference blocks
    (variance about pi / (4 ln2^2 n) = 1.6 / n per bin), on bins where the cut leaves it unchanged; with the loud
    blocks in it, the epochs next to them are high"""
    f, keep = loud["flag"][0]["f"], ~loud["flag"][0]["notched"]
    near = np.zeros(f.size, bool)
    for s in range(-16, 17):
        near |= np.roll(~keep, s)
    far = keep & ~near & (f > 22.0)
    for name in ("flag", "no flag"):
        ratios = []
        for e, ep in enumerate(loud[name]):
            r = (ep["sig2_ref_H"] / (T * PSD[0](f) / 2))[far].mean()
            s = np.sqrt(1.6 / (ep["n_reference_H"] * far.sum()))
            ratios.append((r, s))
        print("\n%s: sig2_ref_H / true spectrum per epoch: %s" % (name, ", ".join("%.4f +- %.4f" % t for t in ratios)))
        if name == "flag":
            assert all(abs(r - 1.0) < NSIG * s for r, s in ratios)
        else:
            assert ratios[1][0] - 1.0 > NSIG * ratios[1][1] and ratios[3][0] - 1.0 > NSIG * ratios[3][1]


def test_epoch_with_too_few_unflagged_blocks_is_unusable(loud):
    """20 loud blocks of 33: unusable (13 left, fewer than min_blocks + 1); 10 loud blocks: usable, and its 23
    unflagged blocks, normalized with the correction for 22, have unit power (with the factor for 32 they are 2.7 %
    low)"""
    eps = loud["many"]
    print("\nloud blocks per epoch %s, usable %s" % ([int(e["loud_H"].sum()) for e in eps], [e["usable"] for e in eps]))
    assert [int(e["loud_H"].sum()) for e in eps] == [0, 20, 10, 0]
    assert not eps[1]["usable"] and not eps[1]["analysable"].any() and eps[1]["n_used_H"] == 13
    assert eps[2]["usable"] and eps[2]["n_used_H"] == 23
    assert np.allclose(eps[2]["correction_H"][~eps[2]["loud_H"]], strain.correction(22, "median"))
    z, keep = unflagged([eps[2]], "H")
    p = np.abs(z[:, keep]) ** 2
    s = np.sqrt(V / p.size)
    print("epoch with 10 loud blocks, the other 23: power %.4f +- %.4f" % (p.mean(), s))
    assert abs(p.mean() - 1.0) < NSIG * s


def test_iteration_finds_a_block_hidden_by_louder_ones(loud):
    """three blocks 900 times louder pull the estimate up by about 13 %, which hides a block 1.25 times louder (band
    excess 0.25, above the thresholds of 160-512 Hz, 0.15 and 0.14) from the first pass; the second pass finds it"""
    ep = loud["hidden"][1]
    print("\nflagged %s in %d passes" % (list(np.flatnonzero(ep["loud_H"])), ep["flag_passes_H"]))
    assert list(np.flatnonzero(ep["loud_H"])) == [5, 10, 15, 25] and ep["flag_passes_H"] >= 2
    assert not any(e["loud_H"].any() for k, e in enumerate(loud["hidden"]) if k != 1)


def test_no_loud_blocks_in_gaussian_noise(run):
    """no flags in the stationary noise of the runs above (264 blocks each), nor among 16,500 blocks of simulated
    exponential periodograms on the bins of the analysis (one block in 1e4 would give 1.6 expected flags); and with
    nothing flagged, the numbers are those of flag=False"""
    for name in ("mean", "median", "mean eq", "median eq", "mean own"):
        assert not any(e["loud_H"].any() or e["loud_L"].any() for e in run(name)), name
    ep = run("median eq")[0]
    rng = np.random.default_rng(13)
    n = 0
    for _ in range(500):
        P = rng.exponential(size=(N + 1, ep["f"].size))
        n += int(strain.loud_blocks(P, ep["f"], ~ep["notched"], "median")[0].sum())
    print("\nfalse flags among %d simulated blocks: %d" % (500 * (N + 1), n))
    assert n == 0
    x = {I: sim.noise(I, int(round((2 * (N + 1) * T + 2 * PAD) * FS)), [7, k]) for k, I in enumerate("HL")}
    kw = dict(T=T, N=N, notches=sim.NOTCHES, pad=PAD, seed=3, method="median")
    a = strain.condition(x["H"], x["L"], FS, GPS0, **kw)
    b = strain.condition(x["H"], x["L"], FS, GPS0, flag=False, **kw)
    for ea, eb in zip(a, b):
        for key in ("d_H", "d_L", "sig2", "sig2_ref_H", "sig2_ref_L", "monitor_H", "monitor_L"):
            assert np.array_equal(ea[key], eb[key]), key


def test_data_quality_slice_must_be_inside_the_array():
    """np.bitwise_and.reduce of an empty slice is -1 (every bit set): a segment outside the data-quality array must
    stop the run, with the segment named, in strain.condition (dq shorter than the data) and in the helper"""
    bits = np.full(100, 3, np.int64)
    assert strain.segment_and(bits, 10, 20, 1234.0) == 3
    for a, b in ((100, 108), (-2, 5), (7, 7), (95, 108)):
        with pytest.raises(ValueError, match="1234.5"):
            strain.segment_and(bits, a, b, 1234.5)
    n = int(round((2 * (N + 1) * T + 2 * PAD) * FS))
    x = {I: sim.noise(I, n, [7, k]) for k, I in enumerate("HL")}
    short = np.full(int((n / FS) // 2), 127)
    with pytest.raises(ValueError, match="segment at GPS"):
        strain.condition(x["H"], x["L"], FS, GPS0, T=T, N=N, notches=sim.NOTCHES, pad=PAD, seed=3, method="median",
                         dq_H=short, dq_L=np.full(int(n / FS), 127))


def hard_case(seed=112, n_block=16):
    """exponential periodograms on the analysis bins with a few blocks made louder at random, in all bands or in 40 to
    80 Hz only (found by search: with this seed the blocks flagged in the second pass include one that the third pass
    clears again when the flags are recomputed from scratch)"""
    f = np.arange(160, 4097) / 8.0
    rng = np.random.default_rng(seed)
    P = rng.exponential(size=(n_block, f.size))
    for b in rng.choice(n_block, rng.integers(1, n_block // 2), replace=False):
        P[b] *= np.exp(rng.uniform(0, 3))
        if rng.random() < 0.5:
            P[b] = rng.exponential(size=f.size) * np.where((f > 40) & (f < 80), np.exp(rng.uniform(0, 3)), 1.0)
    return P, f, np.ones(f.size, bool)


def test_loud_flags_only_grow():
    """a block once flagged stays flagged (the docstring of loud_blocks): the flags after k passes are contained in
    those after k + 1, and the converged flags contain every earlier pass"""
    P, f, keep = hard_case()
    flags = [strain.loud_blocks(P, f, keep, "median", iterations=k)[0] for k in range(1, 7)]
    for k in range(5):
        assert not (flags[k] & ~flags[k + 1]).any(), (k + 1, np.flatnonzero(flags[k]), np.flatnonzero(flags[k + 1]))
    loud, passes = strain.loud_blocks(P, f, keep, "median")
    assert passes < 10 and np.array_equal(loud, flags[-1])


def test_one_loud_block_converges():
    """one block 900 times louder in every band, on 32 others: found, nothing else flagged, in two passes (the second
    confirms that the flags stopped changing)"""
    rng = np.random.default_rng(5)
    f = np.arange(160, 4097) / 8.0
    P = rng.exponential(size=(33, f.size))
    P[7] *= 900.0
    loud, passes = strain.loud_blocks(P, f, np.ones(f.size, bool), "median")
    assert list(np.flatnonzero(loud)) == [7] and passes == 2


def test_unequalized_regenerates_the_data_before_step_5(run):
    """strain.unequalized of an equalized run gives the data of the same run without step 5 (same seed)"""
    eq, plain = run("median eq"), run("median")
    keep = ~eq[0]["notched"]
    for a, b in zip(eq, plain):
        pre = strain.unequalized(a)
        for I in "HL":
            err = np.abs(pre[I][:, keep] - b["d_" + I][:, keep].astype(complex)) / np.sqrt(a["sig2_ref_" + I][keep])
            assert err.max() < 1e-5, (I, err.max())


@pytest.mark.parametrize("name", ["mean eq", "median eq", "known eq"])
def test_own_curves_have_the_power_of_their_data(run, name):
    """strain.own_curves of an equalized run (the data of each detector before step 5, the notched bins of the file):
    the curve of detector I is sig2_ref_I off the notches and the file's notch variance in them; |d_I|^2 / sig2_I has
    mean 1 in every band and in the notched bins, in both detectors; the matched-filter power E_0|z_I|^2 / u_I with the
    bank built on the two curves is 1 for each detector (what the search needs).  Control: L, the quieter detector
    below 100 Hz, against the equalized curve has power 1/1.44 there (fails)"""
    eps = run(name)
    notched = eps[0]["notched"]
    own = [strain.own_curves(e) for e in eps]
    for e, o in zip(eps, own):
        for I in "HL":
            assert np.array_equal(o[I][1][~notched], e["sig2_ref_" + I][~notched])
            assert np.array_equal(o[I][1][notched], e["sig2"][notched]) and np.all(o[I][1] > 0)
    as_eps = [dict(e, d_H=o["H"][0], d_L=o["L"][0], sig2_H=o["H"][1], sig2_L=o["L"][1]) for e, o in zip(eps, own)]
    for e in as_eps:
        del e["sig2"]
    for I in "HL":
        table = power_table(as_eps, I)
        z = np.concatenate([e["d_" + I][:, notched] / np.sqrt(e["sig2_" + I][notched]) for e in as_eps])
        table["notched bins"] = (np.mean(np.abs(z) ** 2), np.sqrt(V / z.size))
        report("%s own curve %s, |d|^2 / sig2_%s" % (name, I, I), table)
        for band, (m, s) in table.items():
            assert abs(m - 1.0) < NSIG * s, (I, band, m, s)
    out, neff = {}, None
    for e in as_eps:
        bank = mass_bank(np.array([20.0, 60.0, 200.0]), e, "cpu")
        for i in range(e["d_H"].shape[0]):
            tr = bank.trajectories({I: e["d_" + I][i] for I in "HL"}, {I: e["sig2_" + I] for I in "HL"})
            for I in "HL":
                out.setdefault(I, []).append(np.mean(np.exp(2.0 * tr["lq" + I].astype(float)), axis=1)
                                             / getattr(bank, "u_" + I))
        W2 = np.abs(bank.W.astype(complex)) ** 2
        neff = {I: (W2 / e["sig2_" + I]).sum(1) ** 2 / ((W2 / e["sig2_" + I]) ** 2).sum(1) for I in "HL"}
    for I in "HL":
        r = np.array(out[I])
        s = np.sqrt(V / neff[I] / r.shape[0])
        print("%s %s on its own curve: E|z|^2 / u_I for M = 20, 60, 200: %s +- %s; u_H/u_L %s"
              % (name, I, np.round(r.mean(0), 4), np.round(s, 4), np.round(bank.u_H / bank.u_L, 3)))
        assert np.all(np.abs(r.mean(0) - 1.0) < NSIG * s)
    f = eps[0]["f"]
    m = ~notched & (f > 22.0) & (f < 100.0)
    p = np.concatenate([np.abs(o["L"][0][:, m]) ** 2 / e["sig2"][m] for e, o in zip(eps, own)])
    print("control: L before step 5 against the equalized curve, 22-100 Hz: %.4f" % p.mean())
    assert abs(p.mean() - 1.0) > NSIG * np.sqrt(V / p.size)
