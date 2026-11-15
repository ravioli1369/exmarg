"""
SIMULATION ONLY: glitches added to synthetic segments, for tests and calibration of the glitch-robust search.

nothing here is imported by the analysis (exmarg.search), which reads segments from files: generation and processing
never mix.

    segments, truth, snr = exmarg.pp.simulate(...)                       # the dict read_segments takes
    draw_rho = power_law_snr(4.0, 5.0, 30.0)                             # dN/d rho ~ rho^-4 on [5, 30]
    glitched, rec = add_template_glitches(segments, {'H': 0.05, 'L': 0.02}, draw_rho, [30.0, 60.0, 90.0], seed=7)
    glitched, rec = add_burst_glitches(segments, 0.05, draw_rho, 'sine-gaussian', seed=7)
    h, p = glitch_signal(segments, 'H', 'ringdown', 9.0, 2.5, np.random.default_rng(3))   # one glitch, given time

A glitch is in one detector only.  Segment k draws from np.random.default_rng([seed, k]), so the glitches of a
segment do not depend on the other segments (or on how many there are), and a seed that differs from the one of the
noise gives glitches independent of it.  The input arrays are never modified.
"""

import numpy as np

from .profiles import profile

KINDS = ('sine-gaussian', 'gaussian', 'ringdown')


def power_law_snr(gamma, rho_min, rho_max):
    """sampler draw_rho(rng) of the single-detector SNR rho with dN/d rho proportional to rho^-gamma on
    [rho_min, rho_max] (inverse cdf)"""
    if not 0 < rho_min < rho_max:
        raise ValueError('need 0 < rho_min < rho_max')
    if gamma == 1.0:
        return lambda rng: float(rho_min * (rho_max / rho_min) ** rng.uniform())
    a, b = rho_min ** (1.0 - gamma), rho_max ** (1.0 - gamma)
    return lambda rng: float((a + rng.uniform() * (b - a)) ** (1.0 / (1.0 - gamma)))


def add_template_glitches(segments, eps, draw_rho, masses, seed, time_step=None):
    """the glitch of the model: with probability eps[I], independently per segment and detector, one equal-mass
    zero-spin template waveform (total mass drawn uniformly from masses, profiles.profile) with random phase and
    arrival time uniform over the segment, scaled to single-detector optimal SNR draw_rho(rng), in detector I only.

    segments  dict with d_H, d_L (n_segment, n_freq) complex, nu (n_freq,), sig2 or sig2_H, sig2_L, T, fmin, fmax
    eps       {'H': eps_H, 'L': eps_L}, or one number for both
    draw_rho  draw_rho(rng) -> rho (see power_law_snr)
    masses    detector-frame total masses (M_sun) to draw from
    time_step if given, arrival times are rounded to a multiple of it (matched-filter grid: bank.T / bank.n_time)

    returns a new segments dict and the record of the glitches, record[I] for I in 'HL': dict of (n_segment,)
    arrays present (bool), kind ('' where absent), rho, t0 (s), mass, phase (nan where absent)"""
    return _add(segments, eps, draw_rho, seed, 'template', *_template_shape(segments, masses), time_step)


def _template_shape(segments, masses):
    """(names, draw, waveform) of the template glitch: mass uniform from masses and phase uniform, the profile at
    arrival time 0 (cached per mass)"""
    masses = np.atleast_1d(np.asarray(masses, float))
    T, fmin, fmax = float(segments['T']), float(segments['fmin']), float(segments['fmax'])
    nu = np.asarray(segments['nu'])
    cache = {}

    def draw(rng):
        return dict(mass=float(rng.choice(masses)), phase=rng.uniform(0.0, 2.0 * np.pi))

    def waveform(p, f):
        if p['mass'] not in cache:
            nu_t, _, a, psi0 = profile(p['mass'] / 2, p['mass'] / 2, 0.0, 0.0, T, fmin, fmax, 1.0)
            j = np.searchsorted(nu, nu_t)
            if j.max() >= nu.size or np.any(nu[j] != nu_t):
                raise ValueError('template bins are not on the segment grid: check T, fmin, fmax')
            w = np.zeros(nu.size, complex)
            w[j] = a * np.exp(1j * psi0)
            cache[p['mass']] = w
        return cache[p['mass']] * np.exp(1j * p['phase'])

    return ('mass', 'phase'), draw, waveform


def add_burst_glitches(segments, eps, draw_rho, kind, seed, f_range=(30.0, 400.0), Q_range=(2.0, 40.0),
                       width_range=(1e-3, 5e-3), time_step=None):
    """glitches outside the model, the same as add_template_glitches with analytic waveforms (their Fourier transform
    at the bins f = nu / T, h(f) = int h(t) exp(-2 pi i f t) dt, the one-sided spectrum of a real signal).  Central
    frequency, quality factor and width are log-uniform on their ranges.

    kind  'sine-gaussian'  exp(-t^2 / 2 tau^2) cos(2 pi f0 t + phase), tau = Q / (2 pi f0), peak at t0
          'gaussian'       exp(-t^2 / 2 sigma^2) times +-1 (phase 0 or pi), sigma = width, peak at t0
          'ringdown'       t >= 0: exp(-t / tau) cos(2 pi f0 t + phase), tau = Q / (pi f0), starts at t0

    record[I] has present, kind, rho, t0, phase and, by kind, f0 and Q, width, f0 and Q"""
    if kind not in KINDS:
        raise ValueError('kind must be one of %s' % (KINDS,))
    return _add(segments, eps, draw_rho, seed, kind, *_burst_shape(kind, f_range, Q_range, width_range), time_step)


def _burst_shape(kind, f_range, Q_range, width_range):
    """(names, draw, waveform) of a burst of one of KINDS, see add_burst_glitches"""
    logu = lambda rng, r: float(np.exp(rng.uniform(np.log(r[0]), np.log(r[1]))))

    def draw(rng):
        if kind == 'gaussian':
            return dict(width=logu(rng, width_range), phase=float(rng.integers(2)) * np.pi)
        return dict(f0=logu(rng, f_range), Q=logu(rng, Q_range), phase=rng.uniform(0.0, 2.0 * np.pi))

    def waveform(p, f):
        if kind == 'gaussian':
            return np.cos(p['phase']) * np.sqrt(2.0 * np.pi) * p['width'] * np.exp(-2.0 * (np.pi * p['width'] * f) ** 2)
        e, f0 = np.exp(1j * p['phase']), p['f0']
        if kind == 'sine-gaussian':
            tau = p['Q'] / (2.0 * np.pi * f0)
            G = lambda g: np.sqrt(2.0 * np.pi) * tau * np.exp(-2.0 * (np.pi * tau * g) ** 2)
            return 0.5 * (e * G(f - f0) + np.conj(e) * G(f + f0))
        r = np.pi * f0 / p['Q']
        return 0.5 * (e / (r + 2j * np.pi * (f - f0)) + np.conj(e) / (r + 2j * np.pi * (f + f0)))

    names = ('width', 'phase') if kind == 'gaussian' else ('f0', 'Q', 'phase')
    return names, draw, waveform


def glitch_signal(segments, I, kind, rho, t0, rng, masses=None, f_range=(30.0, 400.0), Q_range=(2.0, 40.0),
                  width_range=(1e-3, 5e-3)):
    """one glitch at a given arrival time, for several glitches in a segment: 'template' (a mass drawn from masses)
    or one of KINDS, its shape parameters drawn from rng as add_template_glitches and add_burst_glitches draw them,
    moved to t0 (s) and scaled to single-detector optimal SNR rho on the noise curve of detector I.

    returns (the signal (n_freq,) complex, to add to d_I, the dict of shape parameters)"""
    if kind == 'template':
        names, draw, waveform = _template_shape(segments, masses)
    elif kind in KINDS:
        names, draw, waveform = _burst_shape(kind, f_range, Q_range, width_range)
    else:
        raise ValueError('kind must be template or one of %s' % (KINDS,))
    p = draw(rng)
    f = np.asarray(segments['nu']) / float(segments['T'])
    return _scaled(waveform(p, f), f, t0, rho, _sig2(segments, I)), p


def _sig2(segments, I):
    return np.asarray(segments['sig2_' + I] if 'sig2_' + I in segments else segments['sig2'], float)


def _scaled(h, f, t0, rho, sig2):
    """h moved to the arrival time t0 and scaled to the optimal SNR rho: h exp(-2 pi i f t0) rho / sqrt(2 sum |h|^2
    / sig2)"""
    norm = 2.0 * np.sum(np.abs(h) ** 2 / sig2)
    if not norm > 0:
        raise ValueError('glitch has no power inside the band')
    return h * np.exp(-2j * np.pi * f * t0) * (rho / np.sqrt(norm))


def _add(segments, eps, draw_rho, seed, kind, names, draw, waveform, time_step):
    """the engine: per segment and detector one draw of (present, rho, t0, parameters), always in the same order"""
    T = float(segments['T'])
    f = np.asarray(segments['nu']) / T
    eps = {I: float(eps[I] if isinstance(eps, dict) else eps) for I in 'HL'}
    d = {I: np.array(segments['d_' + I], complex) for I in 'HL'}
    n = d['H'].shape[0]
    sig2 = {I: _sig2(segments, I) for I in 'HL'}
    rec = {I: dict(present=np.zeros(n, bool), kind=np.full(n, '', dtype='<U16'),
                   **{key: np.full(n, np.nan) for key in ('rho', 't0') + names}) for I in 'HL'}
    for k in range(n):
        rng = np.random.default_rng([seed, k])
        for I in 'HL':
            present = rng.uniform() < eps[I]
            rho, t0, p = float(draw_rho(rng)), rng.uniform(0.0, T), draw(rng)
            if not present:
                continue
            if time_step is not None:
                t0 = (np.rint(t0 / time_step) * time_step) % T
            d[I][k] += _scaled(waveform(p, f), f, t0, rho, sig2[I])
            rec[I]['present'][k], rec[I]['kind'][k] = True, kind
            rec[I]['rho'][k], rec[I]['t0'][k] = rho, t0
            for key in names:
                rec[I][key][k] = p[key]
    out = dict(segments)
    out['d_H'], out['d_L'] = d['H'], d['L']
    return out, rec
