"""
SIMULATION ONLY: coloured Gaussian strain of a known spectrum and a time-domain chirp, for tests/test_strain.py and
scripts/validate/conditioned_noise.py.  Written without exmarg.strain, so that the tests do not check the code with
itself.

noise: the whole series is synthesized at once in the frequency domain, X_k = sqrt(n fs S(f_k) / 4) (g + i g'),
x = irfft(X): stationary (circular over the series), Gaussian, E|X_k|^2 = n fs S / 2, exactly the spectrum S.

spectra: the O4 curve of exmarg.detectors (aligo_O4high) above 15 Hz; below, S(15) (f / 15)^-20 up to 1e8 S(15)
(reached at 6 Hz, flat under it); Gaussian lines of width WIDTH at 60, 120, 180 Hz and one violin-like
line near 500 Hz.  L is H / 1.44 (20 percent better in amplitude) except for a bump around 200 Hz, where L is worse
than H, so that the equalization switches between the detectors.
"""

import numpy as np
import lal
import lalsimulation as ls

from exmarg import detectors
from exmarg.profiles import MPC_SI

FS = 4096.0
WIDTH = 0.05
LINES = {"H": ((60.0, 300.0), (120.0, 100.0), (180.0, 30.0), (502.25, 1000.0)),
         "L": ((60.0, 300.0), (120.0, 100.0), (180.0, 30.0), (497.75, 1000.0))}
NOTCHES = ((59.5, 60.5), (119.5, 120.5), (179.5, 180.5), (497.0, 498.5), (501.5, 503.0))


def _o4(f):
    old = detectors.CURVE
    detectors.set_curve("o4")
    try:
        return detectors.S("H", f)
    finally:
        detectors.set_curve(old)


def psd(I):
    """the one-sided spectrum (1/Hz) of detector I, a callable of f (Hz)"""
    def S(f):
        f = np.asarray(f, float)
        wall = np.minimum((np.clip(f, 1.0, 15.0) / 15.0) ** -20.0, 1e8)
        base = _o4(np.maximum(f, 15.0)) * wall
        if I == "L":
            base = base / 1.44 * (1.0 + 1.2 * np.exp(-((f - 200.0) / 40.0) ** 2))
        lines = sum(h * np.exp(-0.5 * ((f - f0) / WIDTH) ** 2) for f0, h in LINES[I])
        return base * (1.0 + lines)
    return S


def noise(I, n, seed, fs=FS):
    """(n,) samples at fs of Gaussian noise of spectrum psd(I)"""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    s = psd(I)(f)
    s[0] = s[-1] = 0.0
    g = rng.standard_normal((2, f.size))
    return np.fft.irfft(np.sqrt(n * fs * s / 4.0) * (g[0] + 1j * g[1]), n)


def chirp(M, duration=64.0, fs=FS, f_ref=20.0, taper=(14.0, 19.5, 520.0, 900.0)):
    """the plus polarization at 1 Mpc, face on, of an equal-mass nonspinning binary of detector-frame total mass M
    (IMRPhenomXAS) on f = k / duration, with the reference frequency f_ref of profiles.profile(..., fmin=f_ref, ...);
    tapered to 0 below taper[0] and above taper[3], untouched on [taper[1], taper[2]].  returns f, h (one-sided)"""
    hp, _ = ls.SimInspiralChooseFDWaveform(0.5 * M * lal.MSUN_SI, 0.5 * M * lal.MSUN_SI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           MPC_SI, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / duration, taper[0] - 2.0,
                                           taper[3] + 50.0, f_ref, None, ls.IMRPhenomXAS)
    m = int(round(duration * fs)) // 2 + 1
    f = np.arange(m) / duration
    h = np.zeros(m, complex)
    k = min(m, hp.data.data.size)
    h[:k] = hp.data.data[:k]
    ramp = lambda t: np.sin(0.5 * np.pi * np.clip(t, 0.0, 1.0)) ** 2
    return f, h * ramp((f - taper[0]) / (taper[1] - taper[0])) * ramp((taper[3] - f) / (taper[3] - taper[2]))


def time_series(f, h, t0, duration=64.0, fs=FS):
    """the real series s(t) = int h(f) e^{2 pi i f (t - t0)} df on t = j / fs, 0 <= t < duration (one-sided h)"""
    n = int(round(duration * fs))
    return fs * np.fft.irfft(h * np.exp(-2j * np.pi * f * t0), n)
