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
    cache = {}
    for it in range(1, iterations + 1):
        est, count = masked_loo(P, use, method)
        for n in np.unique(count):
            if n not in cache:
                cache[n] = (correction(n, method) if correct else 1.0, loud_thresholds(nb, n, method, p))
        c = np.array([cache[n][0] for n in count])
        thr = np.array([cache[n][1] for n in count])
        mon = monitors(np.sqrt(P / (c[:, None] * est)), f, edges, keep)
        loud = np.any(np.nan_to_num(mon, nan=-np.inf) > np.nan_to_num(thr, nan=np.inf), axis=1)
        if np.array_equal(loud, ~use) or (~loud).sum() < least:
            return loud, it
        use = ~loud
    return loud, iterations


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
                ep["dq_" + I] = np.array([np.bitwise_and.reduce(dq[i:j]) for i, j in zip(a, b)], np.int64)
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


def write_epoch(path, epoch):
    """one epoch (a dict of condition) to an uncompressed .npz, so that read_segments maps only the rows it uses.
    The data are complex64: their rounding (6e-8 relative) is far below the noise, but at |d| ~ 1e-23 their squares
    underflow in float32, so convert to complex128 (as read_segments does) before squaring."""
    np.savez(path, **epoch)


def read_epoch(path):
    """the dict write_epoch wrote, scalars as Python numbers and strings"""
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
