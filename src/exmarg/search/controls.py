"""
control data sets of an analysis: the same segments with the astrophysical correlation between the detectors removed,
or with the mergers made unlike any template (time reversal), so that a search run on them must find no background.

    slid = time_slide(segments, 0.0488)      # L against H by a whole number of matched-filter samples, circularly
    slid = segment_slide(segments, 1)        # H of segment i with L of segment i + 1
    pr = phase_randomize(segments, seed=3)   # the PR data of the paper
    rev = time_reverse(segments)             # d -> conj(d): each detector's series reversed in time

Each function returns a new segments dict (same keys) and leaves its input unmodified.
"""

import numpy as np

MIN_SHIFT = 0.02      # s: twice the 0.010 s light travel time between the sites


def time_step(segments):
    """spacing T / n_time of the arrival times of the matched filter (bank.MassBank: n_time = 8 nu_max, nu_max the last
    bin), 1 / (8 fmax) when fmax T is an integer"""
    return float(segments['T']) / (8 * int(np.asarray(segments['nu']).max()))


def time_slide(segments, shift):
    """shift the L data against the H data by `shift` seconds, circularly inside each segment: the L data are
    multiplied by exp(-2 pi i f shift), f = nu / T, so a signal in L arrives `shift` later and wraps around at T.

    The shift is a whole number of matched-filter time samples when shift / time_step(segments) is an integer, that is
    shift * 8 nu_max / T (= 8 fmax shift when fmax T is an integer); otherwise the peak of the filter moves to the
    nearest sample and loses a little power.  Raises ValueError when the shift, taken modulo T, is closer than 0.02 s
    to zero: the signal of a merger reaches the two sites 0.010 s apart at most, and a slide that small keeps it
    coherent.

    segments  dict with d_H, d_L (n_segment, n_freq) complex, nu (n_freq,), T and the other keys, kept as they are"""
    T = float(segments['T'])
    wrapped = (float(shift) + T / 2) % T - T / 2
    if abs(wrapped) < MIN_SHIFT:
        raise ValueError('shift %g s is within %g s of a multiple of T = %g s' % (shift, MIN_SHIFT, T))
    f = np.asarray(segments['nu']) / T
    out = dict(segments)
    out['d_H'] = np.array(segments['d_H'])
    out['d_L'] = np.asarray(segments['d_L']) * np.exp(-2j * np.pi * f * float(shift))[None, :]
    return out


def segment_slide(segments, k):
    """pair H of segment i with L of segment i + k, cyclically: the new d_L[i] is the old d_L[(i + k) % n_segment];
    d_H is unchanged.  k is a nonzero integer, not a multiple of n_segment.  The noise spectra are the same for every
    segment, so nothing else changes.

    segments  dict with d_H, d_L (n_segment, n_freq) complex"""
    n = np.asarray(segments['d_H']).shape[0]
    if int(k) != k or int(k) % n == 0:
        raise ValueError('k must be an integer that is not a multiple of n_segment = %d' % n)
    out = dict(segments)
    out['d_H'] = np.array(segments['d_H'])
    out['d_L'] = np.roll(np.asarray(segments['d_L']), -int(k), axis=0)
    return out


def phase_randomize(segments, seed):
    """the phase-randomized (PR) data of Boybeyi, Mandic and Romano, Sec. III F: every frequency bin of the data of a
    segment is rotated by a phase phi_nu, uniform on [0, 2 pi) and independent between bins and segments, the same in
    H and L: d_I,nu -> exp(i phi_nu) d_I,nu.  The power |d_I,nu|^2 of every bin and detector and the cross spectrum
    d_H,nu conj(d_L,nu) of every bin are exactly preserved; a product of two different bins gets the phase
    exp(i (phi_nu - phi_nu')), so all correlations between bins are lost.

    The paper rotates the signal h_I,nu of each merger and keeps the noise.  Here the whole datum is rotated, which
    is the same for Gaussian noise (independent circular noise in the two detectors has the same distribution after a
    common rotation that does not depend on it) and also scrambles the glitches and mergers of real data.

    segments  dict with d_H, d_L (n_segment, n_freq) complex"""
    d_H, d_L = np.asarray(segments['d_H']), np.asarray(segments['d_L'])
    rot = np.exp(2j * np.pi * np.random.default_rng(seed).uniform(size=d_H.shape))
    out = dict(segments)
    out['d_H'], out['d_L'] = d_H * rot, d_L * rot
    return out


def time_reverse(segments):
    """the time-reversed data, control 9 of docs/glitch_likelihood.md: d_I,nu -> conj(d_I,nu) in both detectors.  For
    the one-sided spectrum of a real series x(t), conj(d) is the spectrum of x(-t), circularly inside the segment: a
    transient at t moves to T - t (mod T) in each detector, so the two detectors stay aligned and the delay between the
    sites changes sign.  The power of every bin is kept and Gaussian noise keeps its distribution; a chirp becomes an
    anti-chirp, which the templates no longer match, while a glitch symmetric in time keeps its shape (Kou et al. 2025).

    segments  dict with d_H, d_L (n_segment, n_freq) complex"""
    out = dict(segments)
    out['d_H'], out['d_L'] = np.conj(segments['d_H']), np.conj(segments['d_L'])
    return out
