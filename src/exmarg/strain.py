"""
detector strain -> the segments exmarg.search reads (docs/real_data.md), on arrays (numpy, scipy).

units.  A block or segment is n = T fs samples x_j.  Its data on the bins nu are d_nu = dt rfft(x)_nu, the Fourier
transform over the segment, so that a stationary series of one-sided spectrum S has E|d_nu|^2 = sig2 = T S(nu/T) / 2
(without leakage): the sig2 of read_segments and pp.simulate, with templates in the units of profiles.profile.

whitening.  The epoch (its N + 1 blocks plus pad seconds on each side, the outer half of each pad tapered) is Fourier
transformed, multiplied by the real filter R(f) / sqrt(fs S0(f) / 2) and transformed back; the pads are discarded.  For
a series of spectrum S0 this gives unit-variance white noise inside the passband (two-sided spectrum R^2 per sample).
R (response) is 1 on band, falls to 0 with cosine tapers of width roll below and above, and inside every notch.

cutting.  The whitened series is cut into blocks of n samples (rectangular window), W = rfft(block) / sqrt(n) on the
bins nu = fmin T .. fmax T: E|W|^2 = 1 for unit white noise.  For whitened spectrum R^2 the bin variance is the Fejer
kernel smoothed R^2,
    v_nu = sum_{|tau| < n} (1 - |tau|/n) c(tau) e^{-2 pi i nu tau / n},      c = autocovariance   (bin_variance)
below 1 near the roll-offs and notches: by 1.0e-3 at 20 Hz (roll-off from 16 to 12 Hz), by 0.1 at the edge of a
1 Hz notch and still 1e-3 two Hz away from it.

spectrum of a segment (steps 2-3).  Block j of an epoch has the periodogram, in strain units and without leakage from
the wall below 16 Hz (it is cut from pre-whitened data),
    P_j = |W_j|^2 G,     G = (T S0 / 2) / R^2,        E P_j = v sig2,  P_j ~ exponential, independent between blocks.
S0 (smooth_spectrum: median of the Hann periodograms of the epoch's blocks, running median over `smooth` bins) only
removes the leakage: its level and shape cancel in everything below.  Segment i is normalized by the other N blocks,
    e_i = W_i sqrt(G) / sqrt(c Phat_{-i}),     Phat_{-i} = mean or median / E median of the P_j, j != i (loo_spectrum)
Phat_{-i} does not contain segment i, so given it e_i is complex Gaussian of variance 1 / (c g), g = Phat / E P, in
every bin.  The correction c = E[1/g] makes E|e_i|^2 = 1 (correction): N/(N-1) for the mean (g ~ Gamma(N, 1/N)),
quadrature over the order statistics for the median.  Doing this on the segment's own bins, instead of whitening the
epoch with each leave-one-out spectrum, keeps it exact: that spectrum fluctuates bin to bin, so its filter rings for
about T and mixes the neighbouring blocks, which are in the estimate, into the segment.

one sig2 per file (steps 4-5).  An epoch is handed over on one curve s_I per detector, segment i as sqrt(s_I) e_i, then
equalized:
    d_I = sqrt(s_I) e_I + sqrt(sig2 - s_I) xi_I,     sig2 = max(s_H, s_L),   xi complex unit normal (equalize)
If s_I is independent of the e_i of the file, E|d_I|^2 = sig2 in every bin, for both detectors whatever their spectra,
and E_0|z|^2 = u for every template: the noise mean of Lambda is 1 at first order in the estimation error (the second
order, Var(1/(c g)) x^2 / (2 n_eff) for a template of power x over n_eff bins, is what
scripts/validate/conditioned_noise.py measures).  Two curves are independent of the e_i:
 - reference='neighbours' (default): Phat over the blocks of the neighbouring epochs, disjoint from this one
   (E s_I = v sig2_I).  Exact for the mean and the median; a stretch needs two epochs, 2 (N + 1) T + 2 pad seconds.
 - reference='epoch', mean only: the sum of the P_j of the epoch over N.  The e_i depend on the P_j only through
   their proportions P_j / sum P, independent of the sum for independent exponentials (Lukacs).  The median of the
   epoch is not: it gives E_0|z|^2 / u = 1.034 at N = 32.
A signal in segment i arrives as h sqrt(s / (c Phat_{-i})), with E[s / (c Phat_{-i})] = 1 for both curves: a gain error
of rms about 1 / (2 sqrt N) per bin, and no bias of the noise mean.  Handing over the estimate itself as sig2
(notebook 7, c = N/(N-2) for the mean) is exact for one detector, but not with step 5: the factor must then be N/(N-2)
where the detector is the louder and 1 where it is the quieter (at N = 32 with spectra 1.44 apart, N/(N-2) on both
gives E_0|z|^2 / u = 0.993 and 0.958), and picking the louder from the estimates biases the bins where they cross.
With known spectra (psd=..., the reference of the tests) e_i = W_i / sqrt(v) and s_I = v T S_I / 2.

notches.  Bins where R < 1 get sig2 = notch_factor max(sig2) and d = sqrt(sig2) xi, in both detectors.

loud blocks.  A loud block (a glitch) in an epoch pulls the median of the others up: one block among N + 1 = 33 with
per-bin power 1 + delta lowers the mean |e|^2 of the other 32 by 0.2 % (delta = 0.1), 0.8 % (0.3), 1.8 % (1), and
4.5 % when it is far louder; the hand-over curve of the neighbouring epochs moves the same way.  loud_blocks flags,
per detector and epoch, the blocks whose whitened power in a monitor band exceeds what Gaussian noise reaches with
probability p (loud_thresholds), the others being normalized by the unflagged blocks only, and repeats until the
flags stop changing.  Every estimate then uses the unflagged blocks: masked_loo for the epoch, with correction(n) for
the count n behind each block, and the unflagged blocks of the neighbours for the hand-over curve.  The flags depend
on the data, block i included, but only through thresholds that Gaussian noise crosses with probability p, so the
leave-one-out argument above holds up to O(p).  An unflagged block just below the thresholds (p = 1e-6: band excess
0.49, 0.31, 0.21, 0.15, 0.14 from 20 to 512 Hz with N = 32) still moves the others by up to about 1 %.
"""

import numpy as np
from scipy import integrate, ndimage, special

MONITOR_EDGES = (20.0, 40.0, 80.0, 160.0, 320.0, 512.0)
LEAST = 6                     # fewest unflagged blocks an estimate is made from (the median correction needs 5)


def _ramp(t):
    """0 for t <= 0, sin^2(pi t / 2) between, 1 for t >= 1"""
    return np.sin(0.5 * np.pi * np.clip(t, 0.0, 1.0)) ** 2


def response(f, band=(16.0, 600.0), roll=(4.0, 100.0), notches=(), notch_taper=0.25):
    """amplitude response R of the whitening filter at the frequencies f (Hz): 1 on band, cosine tapers of width
    roll[0] below and roll[1] above it, and inside each notch interval (lo, hi) 0 with cosine tapers of width
    notch_taper at its two sides (R = 1 at lo and hi)"""
    f = np.asarray(f, float)
    lo, hi = band
    R = np.where(f < lo, _ramp((f - lo + roll[0]) / roll[0]),
                 np.where(f > hi, _ramp((hi + roll[1] - f) / roll[1]), 1.0))
    for a, b in notches:
        w = min(notch_taper, 0.5 * (b - a))
        inside = (f > a) & (f < b)
        R = np.where(inside, R * (1.0 - _ramp(np.minimum(f - a, b - f) / w)), R)
    return R


def bin_variance(fs, T, nu, oversample=8, **filt):
    """v (nu.size,): E|W_nu|^2 of blocks of length T cut with a rectangular window from a series of two-sided spectrum
    R^2 per sample (unit white noise whitened by response(**filt)), from its autocovariance on an oversampled grid"""
    n = int(round(T * fs))
    m = oversample * n
    c = np.fft.irfft(response(np.fft.rfftfreq(m, 1.0 / fs), **filt) ** 2, m)[:n]
    a = np.zeros(2 * n)
    a[:n] = (1.0 - np.arange(n) / n) * c
    a[n + 1:] = a[1:n][::-1]
    return np.fft.rfft(a).real[2 * np.asarray(nu)]


def periodograms(x, fs, T, window="hann"):
    """periodograms of the consecutive blocks of length T of x, |dt rfft(w x)|^2 / mean(w^2), so that E P = T S / 2
    for a stationary series of one-sided spectrum S smooth over the window's resolution.
    x (n_block n,) -> P (n_block, n // 2 + 1) on f = k / T; window 'hann' or 'rect'"""
    n = int(round(T * fs))
    x = np.asarray(x, float)
    b = x[: x.size // n * n].reshape(-1, n)
    w = np.sin(np.pi * np.arange(n) / n) ** 2 if window == "hann" else np.ones(n)
    return np.abs(np.fft.rfft(b * w, axis=1)) ** 2 / (fs * fs * np.mean(w * w))


def median_bias(n):
    """E[median of n independent Exp(1)] (numpy's median: mean of the two middle values for even n)"""
    k = np.arange(n)
    if n % 2:
        return float(np.sum(1.0 / (n - k[: n // 2 + 1])))
    return float(np.sum(1.0 / (n - k[: n // 2])) + 0.5 / (n - n // 2))


def spectrum(P, method="mean"):
    """estimate of E P from exponential periodograms P (n_block, n_freq): their mean, or their median / median_bias
    (both unbiased).  returns (n_freq,)"""
    P = np.asarray(P, float)
    if method == "mean":
        return P.mean(axis=0)
    if method == "median":
        return np.median(P, axis=0) / median_bias(P.shape[0])
    raise ValueError("method must be 'mean' or 'median'")


def loo_spectrum(P, method="mean"):
    """leave-one-out estimates: row i is spectrum() of every row of P (n_block, n_freq) but row i.  returns P.shape"""
    P = np.asarray(P, float)
    nb = P.shape[0]
    if method == "mean":
        return (P.sum(axis=0)[None, :] - P) / (nb - 1)
    if method != "median":
        raise ValueError("method must be 'mean' or 'median'")
    order = np.argsort(P, axis=0)
    srt = np.take_along_axis(P, order, axis=0)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(nb)[:, None].repeat(P.shape[1], axis=1), axis=0)

    def kth(k):                                    # k-th smallest of the others: skip the removed rank
        return np.take_along_axis(srt, k + (rank <= k), axis=0)

    m = nb - 1
    med = kth(m // 2) if m % 2 else 0.5 * (kth(m // 2 - 1) + kth(m // 2))
    return med / median_bias(m)


def masked_loo(P, use, method="mean"):
    """leave-one-out estimates from the rows in `use` only: row i is spectrum() of the rows j != i with use[j] (all
    of them for a row not in use).  P (n_block, n_freq), use (n_block,) bool -> (estimates P.shape, counts (n_block,)
    of the rows behind each); with every row in use exactly loo_spectrum(P)"""
    P = np.asarray(P, float)
    use = np.asarray(use, bool)
    if use.all():
        return loo_spectrum(P, method), np.full(P.shape[0], P.shape[0] - 1)
    idx = np.flatnonzero(use)
    out = np.empty_like(P)
    out[idx] = loo_spectrum(P[idx], method)
    out[~use] = spectrum(P[idx], method)[None, :]
    return out, np.where(use, idx.size - 1, idx.size)


def _median_moments(N):
    """E[m], E[1/m], E[1/m^2] of the median m of N independent Exp(1), by quadrature over the order statistics"""
    k = N // 2
    if N % 2:                                      # m = X_(k+1), density N!/(k!)^2 (1 - e^-x)^k e^-(k+1)x
        lc = special.gammaln(N + 1) - 2.0 * special.gammaln(k + 1)
        pw = k
        inner = (lambda x: x, lambda x: 1.0 / x, lambda x: 1.0 / (x * x))
    else:                                          # m = X_(k) + E / (2k), E ~ Exp(1) independent (Renyi)
        lc = special.gammaln(N + 1) - special.gammaln(k) - special.gammaln(k + 1)
        pw = k - 1
        b = 2.0 * k

        def ee1(z):                                # e^z E_1(z)
            return np.exp(z) * special.exp1(z) if z < 500.0 else (1.0 - 1.0 / z + 2.0 / z ** 2 - 6.0 / z ** 3) / z

        inner = (lambda x: x + 1.0 / b, lambda x: b * ee1(b * x), lambda x: b * b * (1.0 / (b * x) - ee1(b * x)))

    def dens(x):
        return np.exp(lc + pw * np.log(-np.expm1(-x)) - (k + 1) * x)

    out = []
    for g in inner:
        v = integrate.quad(lambda x: dens(x) * g(x), 0.0, 60.0, points=(0.3, np.log(2.0), 1.5), limit=500,
                           epsabs=0.0, epsrel=1e-11)[0]
        out.append(v)
    return out


def correction(N, method="mean", scheme="whiten"):
    """the factor c multiplying an estimate from N exponential periodograms (spectrum(), unbiased), with g the estimate
    over its expectation.

    scheme 'whiten'    the estimate normalizes the data (this module): c = E[1/g], so E|e|^2 = E[1/(c g)] = 1
    scheme 'estimate'  the estimate is handed over as sig2 (notebook 7): c = E[1/g^2] / E[1/g]
    mean: g ~ Gamma(N, 1/N), c = N/(N-1) and N/(N-2).  median: quadrature (N >= 5)."""
    if method == "mean":
        return N / (N - 1.0) if scheme == "whiten" else N / (N - 2.0)
    if method != "median":
        raise ValueError("method must be 'mean' or 'median'")
    if N < 5:
        raise ValueError("the inverse moments of the median need N >= 5")
    m1, mi1, mi2 = _median_moments(N)
    return m1 * mi1 if scheme == "whiten" else m1 * mi2 / mi1


def whitened_variance(N, method="mean"):
    """V = Var|e|^2 of a bin normalized by an estimate from N exponential periodograms and its correction (scheme
    'whiten'): 2 E[1/(c g)^2] - 1, N / (N - 2) for the mean, by quadrature for the median"""
    if method == "mean":
        return N / (N - 2.0)
    _, mi1, mi2 = _median_moments(N)
    return 2.0 * mi2 / mi1 ** 2 - 1.0


# the gain of the conditioning on a signal --------------------------------------------------------------------------
#
# A segment's bin is normalized by its leave-one-out estimate (sqrt(c Phat_{-i})) and handed over on the curve s_I of
# the neighbouring epochs (condition: d = sqrt(s_I) W sqrt(G / (c Phat_{-i}))).  A signal h in the segment (strain units)
# therefore arrives as gamma h, gamma = sqrt(g_s / (c g_P)), with g_P = Phat_{-i} / E P and g_s = s_I / E P the two
# estimates over the truth (independent of each other and of block i; P_j are independent exponentials in Gaussian
# noise, and S0 of the pre-whitening cancels in them, so nothing is smoothed over bins).  E[gamma^2] = 1 (c = E[1/g_P]),
# so the signal POWER is unbiased, but the AMPLITUDE is not, and the matched filter sees the amplitude.  On the own curve
# of detector I (sig2_I = s_I, no step 5) the matched filter weights bin nu by |a|^2 / s_I, which is correlated with
# gamma; averaged over the bins of a template it recovers kappa_I of the model's amplitude,
#     kappa_I = normalization_gain(N_I) hand_over_gain(n_ref,I),
#     normalization_gain(N) = E[(c g_P)^{-1/2}],     hand_over_gain(n) = E[g_s^{1/2} / g_s] / E[1 / g_s] = E[g_s^{-1/2}] / E[g_s^{-1}].
# 0.97283 for the median, N = 32 and n_ref = 66 blocks (0.95344 for n_ref = 33, the end epochs of a stretch); 0.98650 for
# the mean.  Exact: this mean amplitude, in every in-band bin that is not notched, for both detectors, given the counts.
# Neglected: (i) the scatter of gamma from bin to bin, under the matched filter's weights of variance
# 1 / E[g_s^{-1}] - kappa^2 (0.023 for the median, 32 and 66), independent between bins: it adds to the projected
# matched filter of a merger of SNR rho a variance about 0.023 rho^2 / n_eff, n_eff = (sum w)^2 / sum w^2 its effective
# number of bins (170 to 1300 for the bank on an O3b curve), below 0.01 at rho = 8 against the unit noise variance;
# (ii) signal power in the blocks of the estimates (other mergers, glitches) and drifts of the spectrum; (iii) the bins
# next to a notch or the band edges, where the bin variance v < 1 correlates the periodograms of neighbouring blocks
# slightly.  The data on the equalized curve (step 5, sig2 = max(s_H, s_L)) have another, spectrum-dependent gain that
# is not modelled (that configuration is retired); a spectrum that is known (psd='known') or segments drawn directly in
# the frequency domain (pp.simulate) have gain 1.  Measured on strain-level injections (scripts/validate/strain_plant.py):
# the equalized gain predicted the same way, 0.98067, against 0.98054 +- 0.00052.

def _ee1(z):
    """e^z E_1(z)"""
    return np.exp(z) * special.exp1(z) if z < 500.0 else (1.0 - 1.0 / z + 2.0 / z ** 2 - 6.0 / z ** 3) / z


def _median_moment(N, p):
    """E[m^p] of the median m of N independent Exp(1) (numpy's median), p in (1, -1, -2, 1/2, -1/2), by quadrature over
    the order statistics as _median_moments: m = X_(k+1) for odd N, m = X_(k) + E / (2k) with E ~ Exp(1) independent
    (Renyi) for even N = 2k, the expectation over E in closed form"""
    k = N // 2
    if N % 2:
        lc = special.gammaln(N + 1) - 2.0 * special.gammaln(k + 1)
        pw = k

        def inner(x):
            return x ** p
    else:
        lc = special.gammaln(N + 1) - special.gammaln(k) - special.gammaln(k + 1)
        pw = k - 1
        b = 2.0 * k
        inner = {1: lambda x: x + 1.0 / b, -1: lambda x: b * _ee1(b * x),
                 -2: lambda x: b * b * (1.0 / (b * x) - _ee1(b * x)),
                 0.5: lambda x: (np.sqrt(b * x) + 0.5 * np.sqrt(np.pi) * special.erfcx(np.sqrt(b * x))) / np.sqrt(b),
                 -0.5: lambda x: np.sqrt(np.pi * b) * special.erfcx(np.sqrt(b * x))}[p]

    def dens(x):
        return np.exp(lc + pw * np.log(-np.expm1(-x)) - (k + 1) * x)

    return integrate.quad(lambda x: dens(x) * inner(x), 0.0, 60.0, points=(0.3, np.log(2.0), 1.5), limit=500,
                          epsabs=0.0, epsrel=1e-11)[0]


def estimate_moment(N, p, method="median"):
    """E[g^p] for g = spectrum() of N independent exponential periodograms over its expectation: the mean, g ~ Gamma(N,
    1/N), Gamma(N + p) / (Gamma(N) N^p); the median, E[m^p] / median_bias(N)^p by quadrature (_median_moment, p in 1, -1,
    -2, 1/2, -1/2; N >= 5 for p = -2)"""
    if method == "mean":
        return float(np.exp(special.gammaln(N + p) - special.gammaln(N) - p * np.log(N)))
    if method != "median":
        raise ValueError("method must be 'mean' or 'median'")
    return _median_moment(N, p) / median_bias(N) ** p


_GAINS = {}


def normalization_gain(N, method="median", correct=True):
    """E[(c g)^{-1/2}]: the mean amplitude of a signal in a segment normalized by the leave-one-out estimate from N
    blocks with its correction c = correction(N, method) (c = 1 with correct False), relative to the truth"""
    key = ("P", int(N), method, bool(correct))
    if key not in _GAINS:
        c = correction(N, method) if correct else 1.0
        _GAINS[key] = estimate_moment(N, -0.5, method) / np.sqrt(c)
    return _GAINS[key]


def hand_over_gain(n, method="median"):
    """E[g^{-1/2}] / E[g^{-1}] for the hand-over curve, an estimate from n blocks: the amplitude factor of a signal
    handed over on that curve, as the matched filter of data on that same curve (weights 1 / s_I) recovers it"""
    key = ("s", int(n), method)
    if key not in _GAINS:
        _GAINS[key] = estimate_moment(n, -0.5, method) / estimate_moment(n, -1.0, method)
    return _GAINS[key]


def signal_gain(epoch, own=True):
    """{'H': kappa_H, 'L': kappa_L}: the amplitude a signal in the segments of the epoch (a dict of condition, or read
    back by read_epoch) carries in the data of each detector on its own curve, relative to the true strain (see above:
    normalization_gain of the blocks behind the estimate of an unflagged segment, hand_over_gain of n_reference_I).
    1.0 for both when the spectrum was known (psd 'known').  own False (the equalized curve of step 5): ValueError, its
    gain depends on the two spectra and is not modelled.  The epoch must record psd, method, reference ('neighbours'),
    N, correction and n_used_I, n_reference_I; the stored per-segment factors correction_I of the unflagged segments
    are checked against the count used (ValueError otherwise)"""
    if str(epoch.get("psd", "")) == "known":
        return {"H": 1.0, "L": 1.0}
    if not own:
        raise ValueError("the gain of data on the equalized curve (step 5, sig2 = max(s_H, s_L)) is not modelled: "
                         "analyse the epoch on its own curves")
    need = ("psd", "method", "reference", "N", "correction", "n_used_H", "n_used_L", "n_reference_H", "n_reference_L")
    missing = [k for k in need if k not in epoch]
    if missing:
        raise ValueError("the epoch does not record %s: its signal gain cannot be derived" % ", ".join(missing))
    if str(epoch["reference"]) != "neighbours":
        raise ValueError("reference %r: the hand-over curve contains the segment itself; its gain is not modelled"
                         % str(epoch["reference"]))
    method, N = str(epoch["method"]), int(epoch["N"])
    correct = float(epoch["correction"]) != 1.0
    out = {}
    for I in "HL":
        n_used = int(epoch["n_used_" + I])
        count = n_used - 1 if n_used >= LEAST else N          # condition: an unflagged block's others; all if too few
        if "correction_" + I in epoch and "loud_" + I in epoch:
            ok = ~np.asarray(epoch["loud_" + I], bool)
            want = correction(count, method) if correct else 1.0
            got = np.asarray(epoch["correction_" + I], float)[ok]
            if got.size and not np.allclose(got, want, rtol=1e-12, atol=0.0):
                raise ValueError("detector %s: the stored correction factors %s are not those of %d blocks (%s)"
                                 % (I, np.unique(got), count, want))
        out[I] = float(normalization_gain(count, method, correct) * hand_over_gain(int(epoch["n_reference_" + I]),
                                                                                    method))
    return out


def loud_thresholds(nbins, N, method="mean", p=1e-6):
    """upper thresholds of the band monitors of a block (mean of |e|^2 over nbins bins, normalized by an estimate
    from N periodograms), each exceeded with probability p by stationary Gaussian noise: the Gamma law of mean 1 and
    variance V / nbins (V = whitened_variance).  Exact for a known spectrum (V = 1); on simulated noise through the
    median of N = 32 the per-band rate is within a factor 2 of p down to p = 1e-5.  nan where nbins is 0"""
    nb = np.asarray(nbins, float)
    V = whitened_variance(N, method)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nb > 0, special.gammainccinv(nb / V, p) * V / np.where(nb > 0, nb, 1.0), np.nan)


def loud_blocks(P, f, keep, method="mean", edges=MONITOR_EDGES, p=1e-6, correct=True, iterations=10, least=6):
    """the loud blocks of one epoch and detector.  Every block is normalized by the leave-one-out estimate of the
    blocks not flagged (masked_loo, with the correction for its count), and flagged when its monitor exceeds
    loud_thresholds (probability p per band on Gaussian noise) in any band; repeated from the new flags until they
    stop changing (at most `iterations` passes; the flags only grow, as removing a loud block lowers the estimate of
    the others).  Stops, with the flags reached, when fewer than `least` blocks remain unflagged.
    P (n_block, n_freq) periodograms, f (n_freq,) Hz, keep (n_freq,) bool -> (loud (n_block,) bool, passes)"""
    P = np.asarray(P, float)
    nb = np.array([np.sum(keep & (f >= edges[k]) & ((f <= edges[k + 1]) if k == len(edges) - 2 else
                                                     (f < edges[k + 1]))) for k in range(len(edges) - 1)])
    use = np.ones(P.shape[0], bool)
    flagged = np.zeros(P.shape[0], bool)
    cache = {}
    for it in range(1, iterations + 1):
        est, count = masked_loo(P, use, method)
        for n in np.unique(count):
            if n not in cache:
                cache[n] = (correction(n, method) if correct else 1.0, loud_thresholds(nb, n, method, p))
        c = np.array([cache[n][0] for n in count])
        thr = np.array([cache[n][1] for n in count])
        mon = monitors(np.sqrt(P / (c[:, None] * est)), f, edges, keep)
        loud = flagged | np.any(np.nan_to_num(mon, nan=-np.inf) > np.nan_to_num(thr, nan=np.inf), axis=1)
        if np.array_equal(loud, ~use) or (~loud).sum() < least:
            return loud, it
        use, flagged = ~loud, loud
    return loud, iterations


def segment_and(bits, a, b, gps):
    """the bits set over all of bits[a:b] (the data-quality seconds of the segment starting at GPS gps).  np.bitwise_and
    .reduce of an empty slice is -1, every bit set, so a slice that is empty or reaches outside the array is an error"""
    if not 0 <= a < b <= len(bits):
        raise ValueError("the data-quality array (%d values) does not cover the segment at GPS %.3f: indices %d to %d"
                         % (len(bits), gps, a, b))
    return int(np.bitwise_and.reduce(bits[a:b]))


def smooth_spectrum(x, fs, T, smooth=17, pad=0.0, highpass=None):
    """a smooth one-sided spectrum of x for the pre-whitening: the median over the blocks of length T of their Hann
    periodograms, then a running median of its log over `smooth` bins.  With highpass = (lo, hi) the series is first
    high-passed (0 below lo, cosine ramp to 1 at hi), so that the wall below the band does not leak into the Hann
    periodograms (their sidelobes fall as only 1/k^6); pad seconds at each end are tapered and removed as in whiten.
    returns a callable f -> S0(f), meaningful above hi"""
    if highpass is not None:
        lo, hi = highpass
        x = whiten(x, fs, lambda f: np.full(np.shape(f), 2.0 / fs), pad, band=(hi, np.inf), roll=(hi - lo, 1.0))
    elif pad:
        p = int(round(pad * fs))
        x = np.asarray(x)[p:len(x) - p]
    P = periodograms(x, fs, T, "hann")
    s = spectrum(P, "median") * 2.0 / T
    ls = ndimage.median_filter(np.log(np.maximum(s, 1e-300)), size=smooth, mode="nearest")
    fk = np.arange(s.size) / T
    return lambda f: np.exp(np.interp(f, fk, ls))


def whiten(x, fs, S, pad=0.0, **filt):
    """x whitened in the frequency domain: rfft, times response(**filt) / sqrt(fs S / 2), irfft (zero phase).  S is a
    callable f -> one-sided spectrum.  The outer half of each pad (seconds) is tapered first and the pads are removed.
    x (n,) -> (n - 2 pad fs,), unit-variance white noise in the passband for a series of spectrum S"""
    x = np.array(x, float)
    n, p = x.size, int(round(pad * fs))
    if p:
        h = p // 2
        r = _ramp((np.arange(h) + 0.5) / h)
        x[:h] *= r
        x[n - h:] *= r[::-1]
    fk = np.fft.rfftfreq(n, 1.0 / fs)
    R = response(fk, **filt)
    on = R > 0
    s = np.asarray(S(fk[on]), float)
    H = np.zeros(fk.size)
    H[on] = np.where(s > 0, R[on] / np.sqrt(0.5 * fs * np.where(s > 0, s, 1.0)), 0.0)
    return np.fft.irfft(np.fft.rfft(x) * H, n)[p:n - p]


def cut(w, fs, T, nu):
    """blocks of length T of the series w (rectangular window), W = rfft(block) / sqrt(n) on the bins nu:
    E|W|^2 = 1 for unit white noise.  w (n_block n,) -> W (n_block, nu.size) complex"""
    n = int(round(T * fs))
    b = np.asarray(w)[: w.size // n * n].reshape(-1, n)
    return np.fft.rfft(b, axis=1)[:, np.asarray(nu)] / np.sqrt(n)


def equalize(d_H, d_L, sig2_H, sig2_L, seed):
    """step 5: independent complex Gaussian noise of variance sig2 - sig2_I added to every bin of detector I,
    sig2 = max(sig2_H, sig2_L).  d_I (n_segment, n_freq), sig2_I (n_freq,), seed an int, sequence or Generator.
    returns d_H, d_L, sig2"""
    rng = np.random.default_rng(seed)
    sig2 = np.maximum(sig2_H, sig2_L)
    out = []
    for d, s in ((d_H, sig2_H), (d_L, sig2_L)):
        g = rng.standard_normal((2,) + np.shape(d))
        out.append(d + np.sqrt(0.5 * (sig2 - s)) * (g[0] + 1j * g[1]))
    return out[0], out[1], sig2


def monitors(e, f, edges=MONITOR_EDGES, keep=None):
    """whitened variance of each segment in sub-bands: the mean of |e|^2 over the bins of f in [edges[k], edges[k+1])
    (the last band closed), bins with keep False left out.  e (n_segment, n_freq) -> (n_segment, len(edges) - 1)"""
    keep = np.ones(f.size, bool) if keep is None else keep
    out = np.empty((e.shape[0], len(edges) - 1))
    for k in range(len(edges) - 1):
        top = f <= edges[k + 1] if k == len(edges) - 2 else f < edges[k + 1]
        sel = keep & (f >= edges[k]) & top
        out[:, k] = np.mean(np.abs(e[:, sel]) ** 2, axis=1) if sel.any() else np.nan
    return out


def condition(x_H, x_L, fs, gps0, T=8.0, fmin=20.0, fmax=512.0, N=32, notches=(), method="mean", correct=True,
              reference="neighbours", psd=None, equal_noise=True, seed=0, pad=None, band=(16.0, 600.0),
              roll=(4.0, 100.0), notch_taper=0.25, smooth=17, notch_factor=1e10, dq_H=None, dq_L=None, dq_rate=1.0,
              events=(), event_margin=1.0, monitor_edges=MONITOR_EDGES, flag=True, flag_p=1e-6, flag_iterations=10,
              min_blocks=16):
    """the epochs of one coincident stretch of H and L.

    x_H, x_L       (n_sample,) strain over the same times, first sample at GPS gps0, sample rate fs (Hz)
    T, fmin, fmax  segment length (s) and band of the bins (Hz)
    N              an epoch holds N + 1 segments; each is normalized by the spectrum of the other N
    notches        (lo, hi) frequency intervals (Hz) removed from both detectors
    method         'mean' or 'median' of the periodograms
    correct        multiply the leave-one-out estimate by correction(N, method) (False: by 1)
    reference      the curve an epoch is handed over on: 'neighbours' (the blocks of the epochs before and after) or,
                   for the mean only, 'epoch' (its own blocks, sum over N; see the module docstring)
    psd            None (spectra estimated from the data) or (S_H, S_L), callables f -> one-sided spectrum: whiten
                   with these known spectra instead (the reference of the tests)
    equal_noise    step 5 (one sig2 for both); False keeps sig2_H, sig2_L
    seed           of the added noise, combined with the GPS time of the epoch
    pad            data (s) on each side of an epoch for the whitening filter, default T
    band, roll     passband of the whitening filter and widths of its roll-offs below and above (Hz)
    notch_taper    width of the cosine tapers inside each notch (Hz)
    smooth         bins of the running median of the pre-whitening spectrum
    notch_factor   sig2 of a notched bin over the largest sig2 of the band
    dq_H, dq_L     (n_second,) int data-quality bits at dq_rate (Hz) from gps0; a segment keeps the bits set over all
                   of it (bitwise and)
    events         GPS times of catalogued events: a segment is flagged when one lies within event_margin (s) of it
    monitor_edges  band edges (Hz) of the whitened-variance monitors
    flag           with estimated spectra, find the loud blocks of each epoch and detector (loud_blocks: a band
                   monitor above its threshold, probability flag_p per band on Gaussian noise, at most
                   flag_iterations passes) and leave them out of every estimate: the leave-one-out spectrum of the
                   epoch (with the correction for the count used) and the hand-over curve of its neighbours.  Loud
                   blocks are still conditioned and written.  Nothing flagged: the same numbers as flag=False
    min_blocks     an epoch is usable when, in both detectors, at least this many blocks enter the leave-one-out
                   estimate of an unflagged block and the hand-over curve

    After pad seconds the stretch is cut into consecutive epochs of (N + 1) T seconds; the rest is dropped.  With
    reference 'neighbours' and estimated spectra a stretch shorter than two epochs gives [], otherwise one shorter
    than one epoch.  returns a list of dicts, one per epoch, with the keys read_segments reads
        d_H, d_L       (n_segment, n_freq) complex64       f, nu (n_freq,)     T, fmin, fmax
        sig2           (n_freq,) common noise variance (or sig2_H, sig2_L when equal_noise is False)
    and the bookkeeping
        sig2_ref_H, sig2_ref_L  (n_freq,) the curve of each detector before step 5
        notched        (n_freq,) bool                       gps (n_segment,) GPS start of each segment
        monitor_H, monitor_L (n_segment, n_band), monitor_edges (n_band + 1,)
        event          (n_segment,) bool                    dq_H, dq_L (n_segment,) int (when given)
        gps_epoch, gps_reference (start of the epochs that gave sig2_ref), and the settings: fs, N, method,
        correction, reference, psd ('estimated' or 'known'), equal_noise, seed, pad, band, roll, notches,
        notch_taper, smooth, notch_factor
    and the loud blocks
        loud_H, loud_L (n_segment,) bool                    analysable (n_segment,) bool: loud in neither, usable
        n_used_H, n_used_L  unflagged blocks of the epoch   n_reference_H, n_reference_L  blocks of sig2_ref
        correction_H, correction_L (n_segment,) the factor applied to each block's estimate
        flag_passes_H, flag_passes_L, usable, and the settings flag, flag_p, flag_iterations, min_blocks
    The random draws of an epoch (step 5, then the noise of the notched bins) follow the order unequalized
    regenerates."""
    if reference not in ("neighbours", "epoch") or (reference == "epoch" and method != "mean"):
        raise ValueError("reference must be 'neighbours', or 'epoch' with method 'mean'")
    own = reference == "epoch" or psd is not None
    x = {"H": np.asarray(x_H, float), "L": np.asarray(x_L, float)}
    if x["H"].shape != x["L"].shape or x["H"].ndim != 1:
        raise ValueError("x_H and x_L must be 1-d arrays over the same times")
    if not np.all(np.isfinite(x["H"])) or not np.all(np.isfinite(x["L"])):
        raise ValueError("the strain must be finite over the stretch")
    n = int(round(T * fs))
    if abs(n - T * fs) > 1e-6:
        raise ValueError("T fs must be an integer")
    pad = T if pad is None else pad
    p = int(round(pad * fs))
    L = (N + 1) * n
    K = (x["H"].size - 2 * p) // L
    if K < (1 if own else 2):
        return []
    nu = np.arange(int(np.ceil(fmin * T - 1e-9)), int(np.floor(fmax * T + 1e-9)) + 1)
    f = nu / T
    filt = dict(band=band, roll=roll, notches=tuple(map(tuple, notches)), notch_taper=notch_taper)
    R = response(f, **filt)
    notched = R < 1.0
    R2 = np.where(notched, 1.0, R * R)
    c = correction(N, method) if (correct and psd is None) else 1.0
    v = np.where(notched, 1.0, bin_variance(fs, T, nu, **filt)) if psd is not None else None

    W, G = {}, {}
    for I in "HL":
        for e in range(K):
            xe = x[I][e * L: e * L + L + 2 * p]
            S0 = psd["HL".index(I)] if psd is not None else smooth_spectrum(
                xe, fs, T, smooth, pad, highpass=(band[0] - 2.0 * roll[0], band[0] - roll[0]))
            W[I, e] = cut(whiten(xe, fs, S0, pad, **filt), fs, T, nu)
            G[I, e] = 0.5 * T * np.where(notched, 1.0, np.asarray(S0(f), float)) / R2

    P, loud, passes = {}, {}, {}
    for I in "HL":
        for e in range(K):
            loud[I, e], passes[I, e] = np.zeros(N + 1, bool), 0
            if psd is None:
                P[I, e] = np.abs(W[I, e]) ** 2 * G[I, e]
                if flag:
                    loud[I, e], passes[I, e] = loud_blocks(P[I, e], f, ~notched, method, monitor_edges, flag_p,
                                                           correct, flag_iterations, LEAST)
    factor = {}

    def correction_of(n):
        if n not in factor:
            factor[n] = correction(n, method) if correct else 1.0
        return factor[n]

    out = []
    for e in range(K):
        gps_epoch = gps0 + (p + e * L) / fs
        refs = [] if psd is not None else [e] if own else [k for k in (e - 1, e + 1) if 0 <= k < K]
        rng = np.random.default_rng([int(seed), int(round(gps_epoch * 1000))])
        white, ref, cfac, n_used, n_ref, usable = {}, {}, {}, {}, {}, {}
        for I in "HL":
            use = ~loud[I, e]
            n_used[I] = int(use.sum())
            if psd is not None:
                white[I] = W[I, e] / np.sqrt(v)
                ref[I] = v * G[I, e]
                cfac[I], n_ref[I], usable[I] = np.ones(N + 1), 0, True
            else:
                base = use if n_used[I] >= LEAST else np.ones(N + 1, bool)      # too few: all blocks, unusable
                if base.all():
                    white[I] = W[I, e] * np.sqrt(G[I, e] / (c * loo_spectrum(P[I, e], method)))
                    cfac[I] = np.full(N + 1, c)
                else:
                    est, count = masked_loo(P[I, e], base, method)
                    cfac[I] = np.array([correction_of(k) for k in count])
                    white[I] = W[I, e] * np.sqrt(G[I, e] / (cfac[I][:, None] * est))
                if reference == "epoch":
                    ref[I] = P[I, e][base].sum(axis=0) / (base.sum() - 1)
                    n_ref[I] = int(base.sum()) - 1
                else:
                    blocks = np.concatenate([P[I, k][~loud[I, k]] for k in refs])
                    if blocks.shape[0] < LEAST:
                        blocks = np.concatenate([P[I, k] for k in refs])
                    ref[I] = spectrum(blocks, method)
                    n_ref[I] = int(sum((~loud[I, k]).sum() for k in refs))
                usable[I] = n_used[I] >= LEAST and n_used[I] - 1 >= min_blocks and n_ref[I] >= min_blocks
            ref[I] = np.where(notched, 0.0, ref[I])
        d = {I: np.sqrt(ref[I]) * white[I] for I in "HL"}
        if equal_noise:
            d["H"], d["L"], sig2 = equalize(d["H"], d["L"], ref["H"], ref["L"], rng)
            sig2 = {"H": sig2, "L": sig2}
        else:
            sig2 = dict(ref)
        big = notch_factor * max(float(np.max(sig2[I][~notched])) for I in "HL") if notched.any() else 0.0
        for I in "HL":
            sig2[I] = np.where(notched, big, sig2[I])
            g = rng.standard_normal((2, d[I].shape[0], int(notched.sum())))
            d[I][:, notched] = np.sqrt(0.5 * big) * (g[0] + 1j * g[1])

        gps = gps_epoch + np.arange(N + 1) * T
        ep = dict(d_H=d["H"].astype(np.complex64), d_L=d["L"].astype(np.complex64), f=f, nu=nu.astype(np.int64),
                  T=float(T), fmin=float(fmin), fmax=float(fmax))
        if equal_noise:
            ep["sig2"] = sig2["H"]
        else:
            ep["sig2_H"], ep["sig2_L"] = sig2["H"], sig2["L"]
        ep.update(sig2_ref_H=ref["H"], sig2_ref_L=ref["L"], notched=notched, gps=gps,
                  monitor_H=monitors(white["H"], f, monitor_edges, ~notched),
                  monitor_L=monitors(white["L"], f, monitor_edges, ~notched),
                  monitor_edges=np.asarray(monitor_edges, float),
                  event=np.array([np.any((np.asarray(events) >= t - event_margin) &
                                         (np.asarray(events) < t + T + event_margin)) for t in gps], bool))
        for I, dq in (("H", dq_H), ("L", dq_L)):
            if dq is not None:
                dq = np.asarray(dq, np.int64)
                a = np.floor((gps - gps0) * dq_rate + 1e-9).astype(int)
                b = np.ceil((gps + T - gps0) * dq_rate - 1e-9).astype(int)
                ep["dq_" + I] = np.array([segment_and(dq, i, j, t) for i, j, t in zip(a, b, gps)], np.int64)
        ep.update(gps_epoch=float(gps_epoch), gps_reference=np.array([gps0 + (p + k * L) / fs for k in refs]),
                  fs=float(fs), N=int(N), method=method, correction=float(c), reference=reference,
                  psd="known" if psd is not None else "estimated", equal_noise=bool(equal_noise), seed=int(seed),
                  pad=float(pad), band=np.asarray(band, float), roll=np.asarray(roll, float),
                  notches=np.asarray(notches, float).reshape(-1, 2), notch_taper=float(notch_taper),
                  smooth=int(smooth), notch_factor=float(notch_factor))
        ep.update(loud_H=loud["H", e].copy(), loud_L=loud["L", e].copy(), n_used_H=n_used["H"], n_used_L=n_used["L"],
                  n_reference_H=n_ref["H"], n_reference_L=n_ref["L"], correction_H=cfac["H"], correction_L=cfac["L"],
                  flag_passes_H=int(passes["H", e]), flag_passes_L=int(passes["L", e]),
                  usable=bool(usable["H"] and usable["L"]), flag=bool(flag and psd is None), flag_p=float(flag_p),
                  flag_iterations=int(flag_iterations), min_blocks=int(min_blocks))
        ep["analysable"] = ~ep["loud_H"] & ~ep["loud_L"] & ep["usable"]
        out.append(ep)
    return out


def one_settings_hash(hashes):
    """hashes {epoch file: settings hash or None} of a folder of epochs: the hash every epoch carries, None for a
    legacy folder in which none does; ValueError (naming files) for a mixture of hashed and unhashed epochs or of
    different hashes"""
    found = {}
    for name, h in sorted(hashes.items()):
        found.setdefault(h, []).append(name)
    if len(found) <= 1:
        return next(iter(found), None)
    what = ", ".join("%s: %s" % (("no settings hash" if h is None else "settings hash %s" % str(h)[:12]),
                                 ", ".join(v[:3]) + (" and %d more" % (len(v) - 3) if len(v) > 3 else ""))
                     for h, v in sorted(found.items(), key=lambda kv: str(kv[0])))
    raise ValueError("the epochs of one folder were not conditioned with one set of settings (%s)" % what)


def unequalized(epoch):
    """the data of each detector before step 5, {I: (n_segment, n_freq) complex128}, 0 in the notched bins: the
    noise of equalize regenerated and subtracted.  The draws of an epoch come from default_rng([seed, round(1000
    gps_epoch)]) in this order, which condition and this function must keep: equalize (standard_normal((2,
    n_segment, n_freq)) for H, then for L), then the noise of the notched bins (H, then L).  Exact to the complex64
    rounding of d (6e-8 relative to sqrt(sig2 / sig2_ref_I))"""
    keep = ~np.asarray(epoch["notched"], bool)
    if not epoch["equal_noise"]:
        return {I: np.where(keep, np.asarray(epoch["d_" + I]).astype(complex), 0.0) for I in "HL"}
    rng = np.random.default_rng([int(epoch["seed"]), int(round(epoch["gps_epoch"] * 1000))])
    sig2 = np.maximum(epoch["sig2_ref_H"], epoch["sig2_ref_L"])
    out = {}
    for I in "HL":
        g = rng.standard_normal((2,) + np.shape(epoch["d_" + I]))
        d = np.asarray(epoch["d_" + I]).astype(complex) - np.sqrt(0.5 * (sig2 - epoch["sig2_ref_" + I])) * (
            g[0] + 1j * g[1])
        out[I] = np.where(keep, d, 0.0)
    return out


def own_curves(epoch):
    """the segments of an epoch on the noise curve of each detector (no step 5; docs/2026-11-15_unequal_noise_design.md):
    {I: (d_I (n_segment, n_freq) complex128, sig2_I (n_freq,))}.  Off the notches d_I = unequalized(epoch)[I] and
    sig2_I = sig2_ref_I, its variance (the curve each detector was handed over on); in the notched bins, where
    unequalized gives 0 and sig2_ref is 0, the data and variance of the file (noise drawn with variance
    notch_factor max sig2, independent of the rest), so that E|d_I|^2 = sig2_I in every bin and neither the data nor the
    templates carry weight there.  An epoch conditioned with equal_noise=False is its own data and curves."""
    sig2 = own_sig2(epoch)
    if not epoch["equal_noise"]:
        return {I: (np.asarray(epoch["d_" + I]).astype(complex), sig2[I]) for I in "HL"}
    notched = np.asarray(epoch["notched"], bool)
    pre = unequalized(epoch)
    return {I: (np.where(notched, np.asarray(epoch["d_" + I]).astype(complex), pre[I]), sig2[I]) for I in "HL"}


def own_sig2(epoch):
    """{I: (n_freq,)} the noise curves of own_curves: sig2_ref_I off the notches and the file's sig2 in them (sig2_I
    of an epoch conditioned with equal_noise=False)"""
    if not epoch["equal_noise"]:
        return {I: np.asarray(epoch["sig2_" + I], float) for I in "HL"}
    notched = np.asarray(epoch["notched"], bool)
    return {I: np.where(notched, np.asarray(epoch["sig2"], float), np.asarray(epoch["sig2_ref_" + I], float))
            for I in "HL"}


def write_epoch(path, epoch):
    """one epoch (a dict of condition) to an uncompressed .npz, so that read_segments maps only the rows it uses.
    The data are complex64: their rounding (6e-8 relative) is far below the noise, but at |d| ~ 1e-23 their squares
    underflow in float32, so convert to complex128 (as read_segments does) before squaring."""
    np.savez(path, **epoch)


def read_epoch(path):
    """the dict write_epoch wrote, scalars as Python numbers and strings"""
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
