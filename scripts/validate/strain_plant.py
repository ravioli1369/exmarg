"""Strain-level plants: the edge loss and the template coverage of the search, measured with mergers of known truth in
simulated continuous strain of H1 and L1, conditioned by the real conditioning (exmarg.strain.condition) and analysed by
the standard driver (scripts/real_data/analyse.py, variant zero: the data are simulated, there is no recorded week).

    python scripts/validate/strain_plant.py spectrum <O3b week folder> spectrum.npz [--epochs 50]
    python scripts/validate/strain_plant.py generate spectrum.npz <root> [--stretches 18] [--epochs 69] [--workers 12]
           [--seed 1] [--nbar 0.3] [--arms N,A,B,C,D] [--only 0,1] [--no-correct --tag nocorr]
    python scripts/validate/strain_plant.py controls spectrum.npz out.json [--seed 1]     # controls i and ii
    python scripts/run analyse.py on <root>/simstrain_<arm> into <runs>/simstrain_<arm> (--variants zero)
    python scripts/validate/strain_plant.py factors <root> <runs> out.json                 # Nbar, factors, iii, iv

Noise.  Stationary Gaussian strain, the SAME one-sided spectrum S in both detectors (spectrum: the median over --epochs
epochs of a conditioned week of the hand-over curve sig2_ref_L, S = 2 sig2 / (T v) with v the bin variance of the cut
(strain.bin_variance), notched bins filled by interpolation of ln S, continued below the band with the log slope of its
first 4 Hz down to 20 Hz, then a wall S(20) (f / 20)^-20 capped at 1e8 S(20), and flat above the band), sample rate
4096 Hz, synthesized in the frequency domain over a whole stretch (circular, as tests/strain_sim.py).  Stretch s starts
at GPS 2e9 + 32768 s (simulated time, no real data there) and holds --epochs epochs of N + 1 = 33 segments of 8 s plus
the 8 s pad of the conditioning at each end.  Conditioning: strain.condition with the settings of the O3b week
(median, N = 32, neighbours, equal noise, the 7 notches of scripts/real_data/notches_O3b.txt, loud blocks flagged) and
seed 20261115 + seed.

Mergers.  Segment of GPS g holds a merger with probability --nbar: presence, source-frame m1 and z from plant.mergers
keyed by g (plant.population(0.99119), plant.Sampler: detector-frame total mass 2 m1 (1 + z) in 20 to 400), y = D0 /
D_L(z).  The extrinsic parameters are those pp.simulate draws from plant.stream(seed, g, 'noise') (the generator of
analyse.py's inj variant): theta, phi (isotropic), psi, cos iota, phase, and a uniform t on [0, T).  ARRIVAL TIME: t is
taken as the time, from the segment start, of the peak of the strain amplitude in H, t_H = t: the peak of |z(t)| with
z the analytic signal of the plus polarization of IMRPhenomXAS (the (2,2) mode, full band) as generated here (from 9 Hz
up to the 1600-2000 Hz taper).  The template of the model, y R_I e^{-i phase} h(f) e^{-2 pi i f (t_a + tau_I)} with
tau_I = n.(x_I - x_L) / c (pp.simulate, profiles.profile), peaks in detector I at t_a + tau_I + t_pk0, t_pk0 the peak
offset of the waveform (found numerically, per waveform); so t_a = t_H - tau_H - t_pk0, and L peaks at t_H - tau_H.
Mass ratio and spin of arms B, D: q from p_q (q^1.1 on [0.3, 1]), chi from p_chi (normal 0.06, 0.12 on [-0.6, 0.6]) of
src/exmarg/data/demo_population.npz, inverse cdf, generator SeedSequence([seed, g, QCHI_KEY]); m1_det = m1 (1 + z),
m2_det = q m1_det, the same chi on both bodies.  The same mergers (segments, m1, z, sky, orientation, phase, t_H) in
every arm.

Arms, one folder of epoch files each (<root>/simstrain_<arm>), the same noise in all:
    N  noise only, conditioned
    A  N plus the segment-level signal (equal mass, zero spin) on the segment's bins: pp.simulate's signal (with y
       less with y = 0, as analyse.py's inj variant) times e^{-2 pi i f dt}, dt = -(tau_H + t_pk0) (the shift to the
       arrival-time definition above); checked bin by bin against model_signal for every merger
    B  N plus the segment-level signal of (m1_det, q m1_det, chi)
    C  the equal-mass mergers of A injected into the strain of both detectors as time series, then conditioned
    D  the mergers of B injected as in C
Strain injection: the waveform of SimInspiralChooseFDWaveform (IMRPhenomXAS, the plus polarization at iota 0, phiRef 0,
f_ref 20 Hz as profiles.profile) on f = k / Tg, Tg a power of 2 above the chirp time from 9 Hz plus 10 s, zero below
9 Hz, sin^2 ramp to 1 at 11 Hz (the whitening filter is 0 below 12 Hz, so the conditioned data are those of an endless
chirp), ramp down from 1600 to 2000 Hz; in detector I the series 2 Re int y R_I e^{-i phase} h e^{2 pi i f (t - t0_I)} df,
t0_I = segment start + t_a + tau_I, added to the strain where it overlaps the stretch.

Per merger the truth file also holds the in-band whitened energy of the strain signal (20 to 512 Hz, notched bins out,
true spectrum) that falls in the merger's segment, the previous and the next (frac_own, frac_prev, frac_next) and, for
each arm X and detector I against the planted segment-level signal s (A's for A, C; B's for B, D), with
<a, b> = sum conj(a) b / sig2 over the bins: opt = sqrt(2 <s, s>) (sig2 of N), sig_X = sqrt(2) Re <s, d_X - d_N> /
sqrt(<s, s>) (the signal in the conditioned segment projected on the planted template, the noise of N subtracted),
snr0_X = sqrt(2) |<s, d_X>| / sqrt(<s, s>) (noisy, at the planted time) and, for C, D, lag_X: the time of the maximum of
|<s e^{2 pi i f tau}, d_X - d_N>| in matched-filter samples (T / 32768).

Output of generate: <root>/simstrain_<arm>[_<tag>]/epoch_<gps>.npz (strain.write_epoch, the keys of strain.condition,
u_ratio_60, settings_hash and settings of the arm), manifest.txt (condition.write_manifest), and
<root>/simstrain_truth[_<tag>]/truth_<stretch>.npz (per segment: see stretch_job).  A stretch whose files all exist is
skipped.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import plant
from exmarg import strain
from exmarg.detectors import Fplus_Fcross, basis, c as C_LIGHT, x as X_DET
from exmarg.pp import simulate
from exmarg.profiles import MPC_SI, profile
from exmarg.search.segments import D0


def _load_script(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


condition = _load_script('exmarg_condition', os.path.join(ROOT, 'scripts', 'real_data', 'condition.py'))

FS = 4096.0
T, FMIN, FMAX, N = 8.0, 20.0, 512.0, 32
PAD = 8.0
F_START, F_FULL = 9.0, 11.0          # Hz: waveform 0 below F_START, sin^2 ramp to 1 at F_FULL
F_HIGH = (1600.0, 2000.0)            # Hz: sin^2 ramp down
F_REF = 20.0                         # reference frequency of profiles.profile (its fmin)
T_POST = 2.0                         # s of the waveform grid after the peak
GPS0, STRIDE = 2000000000, 32768     # simulated GPS of stretch 0, and between stretch starts
NBAR, ZMIN, SEED_COND = 0.30, 0.99119, 20261115
QCHI_KEY, NOISE_KEY = 0x51434849, 0x4E4F4953
ARMS = ('N', 'A', 'B', 'C', 'D')
WALL, WALL_CAP = 20.0, 1e8
NOTCHES_O3B = ((17.375, 18.25), (19.375, 22.8125), (299.0, 300.3125), (301.75, 303.8125), (305.625, 307.9375),
               (314.4375, 315.625), (500.6875, 517.1875))
# strain.condition settings of the O3b week (data/segments/O3b_v2/manifest.txt), notches and seed added per run
COND = dict(T=T, fmin=FMIN, fmax=FMAX, N=N, method='median', reference='neighbours', equal_noise=True, pad=PAD,
            band=(16.0, 600.0), roll=(4.0, 100.0), smooth=17, event_margin=2.0, flag=True, flag_p=1e-6,
            flag_iterations=10, min_blocks=16)
POPULATION = os.path.join(ROOT, 'src', 'exmarg', 'data', 'demo_population.npz')


def ramp(t):
    """0 for t <= 0, sin^2(pi t / 2) between, 1 for t >= 1"""
    return np.sin(0.5 * np.pi * np.clip(t, 0.0, 1.0)) ** 2


# ---------------------------------------------------------------------------------------------------------------------
# the noise spectrum
def median_curve(seg_dir, n_epochs=50, key='sig2_ref_L'):
    """the median over n_epochs usable epochs, evenly spread over the manifest of seg_dir, of the curve `key`:
    (f, curve, notched, the epoch files used, the notches of the epochs)"""
    rows = []
    with open(os.path.join(seg_dir, 'manifest.txt')) as fh:
        for line in fh:
            if line.strip() and not line.startswith('#'):
                p = line.split()
                if p[10] == '1':
                    rows.append((float(p[1]), p[0]))
    rows.sort()
    pick = sorted(set(np.linspace(0, len(rows) - 1, n_epochs).round().astype(int)))
    curves, names, f, notched, notches = [], [], None, None, None
    for i in pick:
        with np.load(os.path.join(seg_dir, rows[i][1])) as z:
            if f is None:
                f, notched, notches = np.asarray(z['f'], float), np.asarray(z['notched'], bool), np.asarray(z['notches'])
            elif not (np.array_equal(f, z['f']) and np.array_equal(notched, z['notched'])):
                raise SystemExit('%s: another frequency grid or other notches' % rows[i][1])
            curves.append(np.asarray(z[key], float))
            names.append(rows[i][1])
    return f, np.median(np.array(curves), axis=0), notched, names, notches


def build_spectrum(f, sig2, notched, notches, fs=FS, T_=T, cond=COND):
    """(fk, S): the one-sided spectrum on fk = k / T_ from 0 to fs / 2, from a hand-over curve sig2 on the band bins f
    (module docstring): S = 2 sig2 / (T_ v) on the bins not notched, ln S interpolated across the notched ones, the log
    slope of the first 4 Hz continued down to FMIN, the wall below, flat above the last bin not notched"""
    nu = np.rint(np.asarray(f) * T_).astype(int)
    v = strain.bin_variance(fs, T_, nu, band=cond['band'], roll=cond['roll'], notches=tuple(map(tuple, notches)))
    good = ~np.asarray(notched, bool) & (np.asarray(sig2) > 0)
    fg = np.asarray(f, float)[good]
    lnS = np.log(2.0 * np.asarray(sig2, float)[good] / (T_ * v[good]))
    fk = np.arange(int(round(fs / 2 * T_)) + 1) / T_
    out = np.interp(fk, fg, lnS)
    first = fg < fg[0] + 4.0
    slope = np.polyfit(np.log(fg[first]), lnS[first], 1)[0]
    low = (fk >= FMIN) & (fk < fg[0])
    out[low] = lnS[0] + slope * np.log(fk[low] / fg[0])
    s20 = lnS[0] + slope * np.log(FMIN / fg[0]) if fg[0] > FMIN else np.interp(FMIN, fg, lnS)
    below = fk < FMIN
    with np.errstate(divide='ignore'):
        out[below] = s20 + np.minimum(-WALL * np.log(np.maximum(fk[below], 1e-30) / FMIN), np.log(WALL_CAP))
    return fk, np.exp(out)


def spectrum_callable(fk, S):
    lnS = np.log(S)
    return lambda f: np.exp(np.interp(np.asarray(f, float), fk, lnS))


def load_spectrum(path):
    """(S callable f -> one-sided spectrum, the dict of the spectrum file)"""
    with np.load(path) as z:
        meta = {k: z[k] for k in z.files}
    return spectrum_callable(meta['fk'], meta['S']), meta


def noise(S, n, seed, fs=FS):
    """(n,) samples of stationary Gaussian noise of one-sided spectrum S (circular over the series), as
    tests/strain_sim.noise: X_k = sqrt(n fs S / 4) (g + i g'), x = irfft(X)"""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    s = S(f)
    s[0] = s[-1] = 0.0
    X = rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)
    X *= np.sqrt(n * fs * s / 4.0)
    return np.fft.irfft(X, n)


# ---------------------------------------------------------------------------------------------------------------------
# the mergers
class QChi:
    """(q, chi) of the three-parameter set-up (demo_population.npz), inverse cdf of p_q and of p_chi"""

    def __init__(self, path=POPULATION):
        with np.load(path) as z:
            self.q = plant._inverse_cdf(z['q'], z['p_q'])
            self.chi = plant._inverse_cdf(z['chi'], z['p_chi'])

    def __call__(self, seed, gps):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(gps), QCHI_KEY]))
        return float(np.interp(rng.uniform(), *self.q)), float(np.interp(rng.uniform(), *self.chi))


def extrinsic(seed_pp):
    """the extrinsic parameters pp.simulate draws for one event from default_rng(seed_pp) when its draw functions use
    no random numbers (analyse.py's inj variant): theta, phi, psi, cos_iota, phase, t (uniform on [0, T))"""
    rng = np.random.default_rng(seed_pp)
    theta, phi = np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi)
    psi, cos_iota = rng.uniform(0, np.pi), rng.uniform(-1, 1)
    phase, t = rng.uniform(0, 2 * np.pi), rng.uniform(0, T)
    return dict(theta=float(theta), phi=float(phi), psi=float(psi), cos_iota=float(cos_iota), phase=float(phase),
                t=float(t))


def responses(ext):
    """{I: (R_I, tau_I)} of pp.simulate: R = (1 + cos^2 iota) / 2 F+ + i cos iota Fx, tau_I = n.(x_I - x_L) / c"""
    th, ph, ps = np.array([ext['theta']]), np.array([ext['phi']]), np.array([ext['psi']])
    n_hat = basis(th, ph)[0][0]
    out = {}
    for I in 'HL':
        Fp, Fx = Fplus_Fcross(I, th, ph, ps)
        R = complex((0.5 * (1 + ext['cos_iota'] ** 2) * Fp + 1j * ext['cos_iota'] * Fx)[0])
        out[I] = (R, float(n_hat @ (X_DET[I] - X_DET['L']) / C_LIGHT))
    return out


def draw_mergers(seed, nbar, gps, sampler, qchi):
    """the truth of the segments of GPS gps (n,): dict of (n,) arrays (module docstring), NaN where no merger"""
    gps = [int(g) for g in gps]
    present, m1, z, y = plant.mergers(dict(seed=seed, nbar=nbar), gps, sampler)
    n = len(gps)
    out = dict(gps=np.array(gps, np.int64), present=present, m1=np.where(present, m1, np.nan),
               z=np.where(present, z, np.nan), y=y)
    keys = ('theta', 'phi', 'psi', 'cos_iota', 'phase', 't_H', 'tau_H', 'q', 'chi')
    for key in keys:
        out[key] = np.full(n, np.nan)
    for j in np.flatnonzero(present):
        ext = extrinsic(plant.stream(seed, gps[j], 'noise'))
        for key in ('theta', 'phi', 'psi', 'cos_iota', 'phase'):
            out[key][j] = ext[key]
        out['t_H'][j] = ext['t']
        out['tau_H'][j] = responses(ext)['H'][1]
        out['q'][j], out['chi'][j] = qchi(seed, gps[j])
    out['mdet_A'] = 2.0 * out['m1'] * (1.0 + out['z'])
    out['m1_det'] = out['m1'] * (1.0 + out['z'])
    out['mdet_B'] = out['m1_det'] * (1.0 + out['q'])
    return out


def ext_of(truth, j):
    return dict(theta=truth['theta'][j], phi=truth['phi'][j], psi=truth['psi'][j], cos_iota=truth['cos_iota'][j],
                phase=truth['phase'][j], t=truth['t_H'][j])


# ---------------------------------------------------------------------------------------------------------------------
# the waveform on a long grid, its peak, and the series in a detector
def waveform(m1d, m2d, chi, fs=FS, f_start=F_START, f_full=F_FULL, f_high=F_HIGH, t_post=T_POST):
    """the plus polarization of IMRPhenomXAS at D0, face on, phiRef 0, f_ref F_REF (profiles.profile: its value at
    f = nu / T is a e^{i psi0} / D0), on f = k / Tg from 0 to fs / 2, with the tapers of the module docstring, and the
    time of its amplitude peak t_pk0 relative to the time origin of the template (a template e^{-2 pi i f t_a} peaks at
    t_a + t_pk0).  returns dict Tg, f, h, t_pk0, tc (chirp time bound from f_start), edge (fraction of the analytic
    power within 1 s of the two ends of the grid when the peak sits at Tg - t_post: 0 for no wrap)"""
    import lal
    import lalsimulation as ls
    tc = ls.SimInspiralChirpTimeBound(f_start, m1d * lal.MSUN_SI, m2d * lal.MSUN_SI, chi, chi)
    Tg = max(32.0, 2.0 ** np.ceil(np.log2(tc + t_post + 8.0)))
    hp, _ = ls.SimInspiralChooseFDWaveform(m1d * lal.MSUN_SI, m2d * lal.MSUN_SI, 0.0, 0.0, chi, 0.0, 0.0, chi, MPC_SI,
                                           0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / Tg, f_start - 1.0, fs / 2.0, F_REF, None,
                                           ls.IMRPhenomXAS)
    m = int(round(Tg * fs)) // 2 + 1
    f = np.arange(m) / Tg
    h = np.zeros(m, complex)
    k = min(m, hp.data.length)
    h[:k] = hp.data.data[:k]
    h *= ramp((f - f_start) / (f_full - f_start)) * ramp((f_high[1] - f) / (f_high[1] - f_high[0])) / D0
    n = 2 * (m - 1)
    t_ref = Tg - t_post
    full = np.zeros(n, complex)
    full[:m] = h * np.exp(-2j * np.pi * f * t_ref)
    a2 = np.abs(np.fft.ifft(full)) ** 2
    j = int(np.argmax(a2))
    den = a2[j - 1] - 2.0 * a2[j] + a2[(j + 1) % n]
    delta = 0.5 * (a2[j - 1] - a2[(j + 1) % n]) / den if den < 0 else 0.0
    one = int(round(fs))
    edge = float((a2[:one].sum() + a2[n - one:].sum()) / a2.sum())
    return dict(Tg=Tg, f=f, h=h, t_pk0=(j + delta) / fs - t_ref, tc=float(tc), edge=edge, n=n)


def series(wf, A, t_shift, fs=FS):
    """the real series 2 Re int A h(f) e^{2 pi i f (t - t_shift)} df on t = j / fs, 0 <= t < Tg"""
    return fs * np.fft.irfft(A * wf['h'] * np.exp(-2j * np.pi * wf['f'] * t_shift), wf['n'])


def detector_factors(ext, y, delay_sign=1.0, phase_sign=1.0):
    """{I: (complex amplitude y R_I e^{-i phase}, tau_I)}; the breaks of control i: delay_sign -1 (tau -> -tau),
    phase_sign -1 (the complex conjugate amplitude)"""
    out = {}
    for I, (R, tau) in responses(ext).items():
        A = y * R * np.exp(-1j * ext['phase'])
        out[I] = (A if phase_sign > 0 else np.conj(A), delay_sign * tau)
    return out


def inject(x, t0, wf, ext, y, t_a, fs=FS, delay_sign=1.0, phase_sign=1.0, reverse=False, start=None):
    """add one merger to the strain x = {I: (n,)} whose sample 0 is at time t0 (s): template origin at t_a in L (the
    time of the segment start plus the model's t_a), each detector's own response and delay.  Breaks: delay_sign,
    phase_sign (detector_factors), reverse (the series reversed in time about its peak: the opposite sign of the
    Fourier convention), start (s): no strain before that time (the injection cut at a segment start).  returns
    {I: (first sample, series)} as added"""
    g0 = int(np.floor((t_a + wf['t_pk0'] - (wf['Tg'] - T_POST) - t0) * fs))
    tg = t0 + g0 / fs
    out = {}
    for I, (A, tau) in detector_factors(ext, y, delay_sign, phase_sign).items():
        s = series(wf, A, t_a + tau - tg, fs)
        if reverse:
            j = int(np.argmax(np.abs(s)))
            idx = 2 * j - np.arange(s.size)
            s = np.where((idx >= 0) & (idx < s.size), s[np.clip(idx, 0, s.size - 1)], 0.0)
        if start is not None:
            s = np.where(tg + np.arange(s.size) / fs >= start, s, 0.0)
        a, b = max(g0, 0), min(g0 + s.size, x[I].size)
        if a < b:
            x[I][a:b] += s[a - g0:b - g0]
        out[I] = (g0, s)
    return out


def band_mask(f, notches, fmin=FMIN, fmax=FMAX):
    """the frequencies of the analysis: fmin <= f <= fmax and not notched (strain.response == 1)"""
    f = np.asarray(f, float)
    return (f >= fmin) & (f <= fmax) & (strain.response(f, notches=tuple(map(tuple, notches))) == 1.0)


def whitened_split(wf, ext, y, t_a, seg_start, S, notches, fs=FS, T_=T):
    """{I: (frac_prev, frac_own, frac_next, rho2)}: the fractions of the in-band whitened energy of the strain signal
    (f in band_mask, whitened by the true spectrum S) in the segment [seg_start, seg_start + T_), the one before and
    the one after, and rho2 = 4 int |A h|^2 / S df over the band (the optimal SNR^2 with the true spectrum).  Times
    relative to the template origin: seg_start - t_a is the segment start on the grid of the series"""
    f = wf['f']
    m = band_mask(f, notches)
    wfil = np.where(m, 1.0 / np.sqrt(np.where(m, S(f), 1.0) * fs / 2.0), 0.0)
    g = np.floor((wf['t_pk0'] - (wf['Tg'] - T_POST)) * fs) / fs        # grid start relative to t_a
    t = g + np.arange(wf['n']) / fs
    out = {}
    df = 1.0 / wf['Tg']
    for I, (A, tau) in detector_factors(ext, y).items():
        w = fs * np.fft.irfft(A * wf['h'] * wfil * np.exp(-2j * np.pi * f * (tau - g)), wf['n'])
        e = w * w
        tot = e.sum()
        rel = t - (seg_start - t_a)
        fr = [float(e[(rel >= lo) & (rel < lo + T_)].sum() / tot) if tot > 0 else np.nan for lo in (-T_, 0.0, T_)]
        rho2 = float(4.0 * np.sum(np.abs(A * wf['h'][m]) ** 2 / S(f[m])) * df)
        out[I] = (fr[0], fr[1], fr[2], rho2)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the segment-level signal
def model_signal(m1d, m2d, chi, y, ext, t_a, nu, T_=T, fmin=FMIN, fmax=FMAX):
    """{I: (n_freq,)} the signal of the model on the bins nu: y R_I e^{-i phase} a e^{i psi0} / D0 e^{-2 pi i f (t_a +
    tau_I)} (pp.simulate's expression, written out here)"""
    nu = np.asarray(nu)
    nu_t, _, a1, psi0 = profile(m1d, m2d, chi, chi, T_, fmin, fmax, 1.0)
    h = np.zeros(nu.size, complex)
    h[np.searchsorted(nu, nu_t)] = a1 / D0 * np.exp(1j * psi0)
    f = nu / T_
    return {I: A * h * np.exp(-2j * np.pi * f * (t_a + tau)) for I, (A, tau) in detector_factors(ext, y).items()}


def plant_signal(seed, gps, mdet, q, chi, y, nu, T_=T, fmin=FMIN, fmax=FMAX):
    """{I: (n_freq,)} pp.simulate's signal as analyse.py's inj variant takes it: the data of simulate(1,
    plant.stream(seed, gps, 'noise'), (mdet, q, chi), y) less those with y = 0; its arrival time is the t of
    extrinsic(), at the template origin in L"""
    def one(yj):
        seg, _, _ = simulate(1, plant.stream(seed, gps, 'noise'), lambda rng: (mdet, q, chi), lambda rng: yj, T_, fmin,
                             fmax)
        if not np.array_equal(seg['nu'], np.asarray(nu)):
            raise ValueError('the bins of pp.simulate are not those of the epoch')
        return seg
    a, b = one(y), one(0.0)
    return {I: a['d_' + I][0] - b['d_' + I][0] for I in 'HL'}


def segment_signal(seed, gps, m1d, m2d, chi, y, ext, t_pk0, nu, T_=T, check=True):
    """the segment-level signal of one merger with the arrival-time definition of the module docstring: pp.simulate's
    (plant_signal) shifted by dt = -(tau_H + t_pk0); checked against model_signal at t_a = t_H - tau_H - t_pk0 (relative
    difference below 1e-6 of the largest bin).  returns ({I: (n_freq,)}, t_a)"""
    tau_H = responses(ext)['H'][1]
    t_a = ext['t'] - tau_H - t_pk0
    q = m2d / m1d
    s = plant_signal(seed, gps, m1d * (1.0 + q), q, chi, y, nu, T_)
    f = np.asarray(nu) / T_
    s = {I: v * np.exp(2j * np.pi * f * (tau_H + t_pk0)) for I, v in s.items()}
    if check:
        mine = model_signal(m1d, m2d, chi, y, ext, t_a, nu, T_)
        for I in 'HL':
            scale = np.abs(mine[I]).max()
            if not np.abs(s[I] - mine[I]).max() <= 1e-6 * scale:
                raise RuntimeError('segment %d, %s: pp.simulate and model_signal differ by %.3g of the largest bin'
                                   % (gps, I, np.abs(s[I] - mine[I]).max() / scale))
    return s, t_a


def mf_numbers(s, d, d0, sig2, nu, n_time=None):
    """the matched-filter numbers of one detector-segment against the planted segment-level signal s (module
    docstring): opt, sig (noiseless, projected), snr0 (noisy, at the planted time), lag (samples of T / n_time)"""
    w = 1.0 / np.asarray(sig2, float)
    s = np.asarray(s, complex)
    norm = np.sqrt(np.sum(np.abs(s) ** 2 * w))
    if not norm > 0:
        return dict(opt=0.0, sig=np.nan, snr0=np.nan, lag=np.nan)
    diff = np.asarray(d, complex) - np.asarray(d0, complex)
    out = dict(opt=float(np.sqrt(2.0) * norm), sig=float(np.sqrt(2.0) * np.sum(np.conj(s) * diff * w).real / norm),
               snr0=float(np.sqrt(2.0) * abs(np.sum(np.conj(s) * np.asarray(d, complex) * w)) / norm))
    n_time = n_time or 8 * int(np.max(nu))
    pad = np.zeros(n_time, complex)
    pad[np.asarray(nu)] = np.conj(s) * diff * w
    z = np.abs(np.fft.ifft(pad))
    j = int(np.argmax(z))
    out['lag'] = float(j if j < n_time // 2 else j - n_time)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# one stretch
def settings_of(cfg, arm):
    """(SHA-256, JSON) of what fixes the content of an arm's epochs"""
    rec = dict(cfg['cond'], notches=np.asarray(cfg['cond']['notches'], float).tolist(), arm=arm, seed=cfg['seed'],
               nbar=cfg['nbar'], spectrum_sha256=cfg['spectrum_sha256'], epochs=cfg['epochs'], fs=FS, gps0=GPS0,
               stride=STRIDE, version=1)
    rec = json.loads(json.dumps(rec))
    text = json.dumps(rec, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest(), text


def folder(root, arm, tag=''):
    return os.path.join(root, 'simstrain_%s%s' % (arm, '_' + tag if tag else ''))


def stretch_geometry(cfg, s):
    """(GPS of the first sample, number of samples, GPS of the segments (n_seg,)) of stretch s"""
    K, n1 = cfg['epochs'], cfg['cond']['N'] + 1
    n = int(round((K * n1 * T + 2 * PAD) * FS))
    start = GPS0 + s * STRIDE
    return start, n, start + PAD + np.arange(K * n1) * T


def stretch_files(cfg, s):
    start, _, _ = stretch_geometry(cfg, s)
    names = [condition.epoch_name(start + PAD + e * (cfg['cond']['N'] + 1) * T) for e in range(cfg['epochs'])]
    paths = [os.path.join(folder(cfg['root'], a, cfg['tag']), nm) for a in cfg['arms'] for nm in names]
    return paths + [os.path.join(cfg['root'], 'simstrain_truth' + ('_' + cfg['tag'] if cfg['tag'] else ''),
                                 'truth_%03d.npz' % s)]


def write_arm(cfg, arm, eps, s):
    """the epochs of one arm of stretch s, with u_ratio_60 and the arm's settings hash"""
    h, text = cfg['hashes'][arm]
    out = folder(cfg['root'], arm, cfg['tag'])
    for ep in eps:
        ep = dict(ep, u_ratio_60=condition.u_ratio(ep), settings_hash=h, settings=text, arm=arm, stretch=s,
                  notch_file='notches_O3b (strain_plant.NOTCHES_O3B)')
        condition.write(out, ep)


def stretch_job(job):
    """generate, inject and condition stretch s; write its epochs of every arm and its truth file.  returns timings"""
    cfg, s = job
    t_begin = time.time()
    if all(os.path.exists(p) for p in stretch_files(cfg, s)):
        return dict(stretch=s, skipped=True)
    S, _ = load_spectrum(cfg['spectrum'])
    start, n, seg_gps = stretch_geometry(cfg, s)
    notches = cfg['cond']['notches']
    kw = dict(cfg['cond'], seed=SEED_COND + cfg['seed'], correct=not cfg['no_correct'])
    if cfg.get('psd_known'):
        kw['psd'] = (S, S)
    sampler = plant.Sampler(plant.population(ZMIN), plant.MDET)
    truth = draw_mergers(cfg['seed'], cfg['nbar'], seg_gps, sampler, QChi())
    nseg = seg_gps.size
    n1 = cfg['cond']['N'] + 1
    truth.update(stretch=np.full(nseg, s), row=np.arange(nseg) % n1, k=s * nseg + np.arange(nseg),
                 epoch_gps=start + PAD + (np.arange(nseg) // n1) * n1 * T)
    timing = {}
    t0 = time.time()
    x = {I: noise(S, n, np.random.SeedSequence([cfg['seed'], NOISE_KEY, s, j])) for j, I in enumerate('HL')}
    timing['noise'] = time.time() - t0
    t0 = time.time()
    epN = strain.condition(x['H'], x['L'], FS, float(start), **kw)
    timing['condition_N'] = time.time() - t0
    if len(epN) != cfg['epochs']:
        raise RuntimeError('stretch %d: %d epochs, expected %d' % (s, len(epN), cfg['epochs']))
    present = np.flatnonzero(truth['present'])
    wanted = set(cfg['arms'])
    masses = dict(A=lambda j: (truth['m1_det'][j], truth['m1_det'][j], 0.0),
                  B=lambda j: (truth['m1_det'][j], truth['q'][j] * truth['m1_det'][j], truth['chi'][j]))
    # per merger: the waveform on its long grid (regenerated where needed, not kept), the segment-level signals (A, B),
    # the in-band whitened split of the strain signals (C, D)
    t0 = time.time()
    sigs, t_a = {}, {}
    for a in ('A', 'B'):
        if not (wanted & ({'A', 'C'} if a == 'A' else {'B', 'D'})):
            continue
        for key in ('t_pk0_', 'Tg_', 'edge_', 'tc_', 't_a_', 'opt_true_H_', 'opt_true_L_') + tuple(
                'frac_%s_%s_' % (w, I) for w in ('prev', 'own', 'next') for I in 'HL'):
            truth[key + a] = np.full(nseg, np.nan)
        for j in present:
            m1d, m2d, chi = masses[a](j)
            ext = ext_of(truth, j)
            wf = waveform(m1d, m2d, chi)
            truth['t_pk0_' + a][j], truth['Tg_' + a][j], truth['edge_' + a][j], truth['tc_' + a][j] = (
                wf['t_pk0'], wf['Tg'], wf['edge'], wf['tc'])
            sigs[j, a], t_a[j, a] = segment_signal(cfg['seed'], truth['gps'][j], m1d, m2d, chi, truth['y'][j], ext,
                                                   wf['t_pk0'], epN[0]['nu'])
            truth['t_a_' + a][j] = t_a[j, a]
            split = whitened_split(wf, ext, truth['y'][j], t_a[j, a], 0.0, S, notches)
            for I in 'HL':
                pv, own, nx, rho2 = split[I]
                truth['frac_prev_%s_%s' % (I, a)][j] = pv
                truth['frac_own_%s_%s' % (I, a)][j] = own
                truth['frac_next_%s_%s' % (I, a)][j] = nx
                truth['opt_true_%s_%s' % (I, a)][j] = np.sqrt(rho2)
            del wf
    timing['signals'] = time.time() - t0
    rows = {int(j): (int(j) // n1, int(j) % n1) for j in present}
    eps = {'N': epN}
    for a in ('A', 'B'):
        if a not in wanted:
            continue
        out = []
        for e, ep in enumerate(epN):
            d = {I: np.asarray(ep['d_' + I]).astype(complex) for I in 'HL'}
            for j, (ee, r) in rows.items():
                if ee == e:
                    for I in 'HL':
                        d[I][r] += sigs[j, a][I]
            out.append(dict(ep, d_H=d['H'].astype(np.complex64), d_L=d['L'].astype(np.complex64)))
        eps[a] = out
    for a, base in (('C', 'A'), ('D', 'B')):
        if a not in wanted:
            continue
        t0 = time.time()
        xs = {I: x[I].copy() for I in 'HL'}
        for j in present:
            seg_start = PAD + j * T
            wf = waveform(*masses[base](j))
            if wf['t_pk0'] != truth['t_pk0_' + base][j]:
                raise RuntimeError('stretch %d, segment %d: the waveform is not reproducible' % (s, j))
            inject(xs, 0.0, wf, ext_of(truth, j), truth['y'][j], seg_start + t_a[j, base])
        timing['inject_' + a] = time.time() - t0
        t0 = time.time()
        eps[a] = strain.condition(xs['H'], xs['L'], FS, float(start), **kw)
        timing['condition_' + a] = time.time() - t0
        del xs
    del x
    # the matched-filter numbers of every merger in every arm, and the flags
    nu = epN[0]['nu']
    for a in eps:
        truth['analysable_' + a] = np.concatenate([np.asarray(ep['analysable'], bool) & bool(ep['usable'])
                                                   for ep in eps[a]])
    for a in ('A', 'B', 'C', 'D'):
        if a not in eps:
            continue
        ref = 'A' if a in 'AC' else 'B'
        for key in ('opt', 'sig', 'snr0', 'lag'):
            for I in 'HL':
                truth['%s_%s_%s' % (key, I, a)] = np.full(nseg, np.nan)
        for j, (e, r) in rows.items():
            for I in 'HL':
                sig2 = eps['N'][e]['sig2'] if a in 'AB' else eps[a][e]['sig2']
                mf = mf_numbers(sigs[j, ref][I], eps[a][e]['d_' + I][r], eps['N'][e]['d_' + I][r], sig2, nu)
                for key in ('opt', 'sig', 'snr0', 'lag'):
                    truth['%s_%s_%s' % (key, I, a)][j] = mf[key]
    for a in cfg['arms']:
        write_arm(cfg, a, eps[a], s)
    tdir = os.path.join(cfg['root'], 'simstrain_truth' + ('_' + cfg['tag'] if cfg['tag'] else ''))
    tmp = os.path.join(tdir, 'truth_%03d.tmp.npz' % s)
    np.savez(tmp, **{k: np.asarray(v) for k, v in truth.items()})
    os.replace(tmp, os.path.join(tdir, 'truth_%03d.npz' % s))
    timing['total'] = time.time() - t_begin
    return dict(stretch=s, skipped=False, mergers=int(present.size), timing=timing,
                flagged={a: int((~truth['analysable_' + a]).sum()) for a in eps})


def spectrum_sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def make_config(spectrum, root, seed=1, nbar=NBAR, epochs=69, arms=ARMS, tag='', no_correct=False, cond=None,
                psd_known=False):
    """the settings of a generate run (cond: overrides of COND, the tests' small epochs; psd_known: whiten with the true
    spectrum instead of estimating it, strain.condition(psd=(S, S)), the exact reference of the tests)"""
    arms = list(arms)
    if not set(arms) <= set(ARMS) or 'N' not in arms:
        raise SystemExit('--arms: a subset of %s that holds N' % ','.join(ARMS))
    _, meta = load_spectrum(spectrum)
    cfg = dict(root=root, spectrum=os.path.abspath(spectrum), spectrum_sha256=spectrum_sha(spectrum), seed=int(seed),
               nbar=float(nbar), epochs=int(epochs), arms=arms, tag=tag, no_correct=bool(no_correct),
               psd_known=bool(psd_known),
               cond=dict(COND, notches=[tuple(map(float, p)) for p in np.asarray(meta['notches']).reshape(-1, 2)],
                         **(cond or {})))
    cfg['hashes'] = {arm: settings_of(dict(cfg, cond=dict(cfg['cond'], correct=not no_correct,
                                                          psd='known' if psd_known else 'estimated')), arm)
                     for arm in ARMS}
    return cfg


def generate(cfg, todo, workers=1, log=print):
    """the stretches todo of the run cfg (stretch_job), then the manifest of every arm folder"""
    root, tag = cfg['root'], cfg['tag']
    for arm in cfg['arms']:
        os.makedirs(folder(root, arm, tag), exist_ok=True)
        for p in glob.glob(os.path.join(folder(root, arm, tag), 'epoch_*.npz')):
            if p.endswith('.tmp.npz'):
                os.remove(p)
                continue
            with np.load(p) as z:
                if str(z['settings_hash']) != cfg['hashes'][arm][0]:
                    raise SystemExit('%s: conditioned with other settings; use another root or tag' % p)
    os.makedirs(os.path.join(root, 'simstrain_truth' + ('_' + tag if tag else '')), exist_ok=True)
    log('%s  stretches %s, %d epochs each, arms %s, %d workers, seed %d, nbar %g%s' % (
        time.strftime('%Y-%m-%d %H:%M:%S'), todo, cfg['epochs'], ','.join(cfg['arms']), workers, cfg['seed'],
        cfg['nbar'], ', NO correction (control iv break)' if cfg['no_correct'] else ''), flush=True)
    jobs = [(cfg, s) for s in todo]
    t0 = time.time()
    if workers <= 1:
        results = map(stretch_job, jobs)
    else:
        # spawn, not fork: exmarg.priors holds an open npz from import time, whose file offset forked workers share
        pool = mp.get_context('spawn').Pool(workers, maxtasksperchild=1)
        results = pool.imap_unordered(stretch_job, jobs)
    done = []
    for r in results:
        done.append(r)
        log('%s  stretch %d: %s (%.0f s elapsed)' % (
            time.strftime('%Y-%m-%d %H:%M:%S'), r['stretch'], 'present, skipped' if r['skipped'] else
            '%d mergers, not analysable %s; %s' % (r['mergers'], r['flagged'], ', '.join(
                '%s %.0f s' % kv for kv in r['timing'].items())), time.time() - t0), flush=True)
    if workers > 1:
        pool.close()
        pool.join()
    for arm in cfg['arms']:
        out = folder(root, arm, tag)
        head = ['simulated strain, scripts/validate/strain_plant.py, arm %s, %s' % (arm, time.strftime('%Y-%m-%d %H:%M')),
                'spectrum %s (sha256 %s)' % (cfg['spectrum'], cfg['spectrum_sha256'][:12]),
                'settings ' + cfg['hashes'][arm][1]]
        n_ep, n_seg, _ = condition.write_manifest(out, head)
        log('%s: %d epochs, %d segments' % (out, n_ep, n_seg), flush=True)
    return done


def run_generate(a):
    cfg = make_config(a.spectrum, a.root, a.seed, a.nbar, a.epochs, [x for x in a.arms.split(',') if x], a.tag,
                      a.no_correct)
    generate(cfg, [int(x) for x in a.only.split(',')] if a.only else list(range(a.stretches)), a.workers)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# controls i and ii: one loud merger in a short stretch
def control_case(S, notches, t_H, breaks, seed=1, psd_known=False, K=3, row=None, m=(15.0, 15.0), chi=0.0,
                 rho_H=25.0, ext=None, cond=None):
    """one loud merger (detector-frame masses m, H peak at t_H after the start of segment `row` (default the middle
    one) of epoch 1) in noise of
    spectrum S, conditioned with and without it (same noise); for each break in breaks (dicts of inject's keywords,
    plus 'circular': the segment-level signal added to the conditioned noise instead) the numbers of mf_numbers in that
    segment and the one before against the planted segment-level signal, the peak lags, the predicted split of
    whitened_split.  returns dict"""
    cond = dict(COND if cond is None else cond, notches=notches)
    kw = dict(cond, seed=SEED_COND + seed)
    if psd_known:
        kw['psd'] = (S, S)
    n1 = cond['N'] + 1
    row = n1 // 2 if row is None else row
    n = int(round((K * n1 * T + 2 * PAD) * FS))
    x0 = {I: noise(S, n, np.random.SeedSequence([seed, NOISE_KEY, 999, j])) for j, I in enumerate('HL')}
    ep0 = strain.condition(x0['H'], x0['L'], FS, 0.0, **kw)
    nu = ep0[1]['nu']
    ext = dict(theta=1.4, phi=1.7, psi=0.7, cos_iota=0.6, phase=1.1, t=t_H) if ext is None else dict(ext, t=t_H)
    wf = waveform(m[0], m[1], chi)
    tau_H = responses(ext)['H'][1]
    t_a = t_H - tau_H - wf['t_pk0']
    s1 = model_signal(m[0], m[1], chi, 1.0, ext, t_a, nu)
    y = rho_H / np.sqrt(2.0 * np.sum(np.abs(s1['H']) ** 2 / ep0[1]['sig2']))
    s = {I: y * v for I, v in s1.items()}
    seg = n1 + row
    seg_start = PAD + seg * T
    split = whitened_split(wf, ext, y, t_a, 0.0, S, notches)
    out = dict(t_H=t_H, tau_H=tau_H, t_pk0=wf['t_pk0'], y=float(y), masses=list(m), chi=chi,
               lag_delay_flipped_H=float(-2.0 * tau_H * 8 * nu.max() / T),
               predicted={I: dict(prev=split[I][0], own=split[I][1], next=split[I][2], opt_true=np.sqrt(split[I][3]))
                          for I in 'HL'}, cases={})
    for name, br in breaks.items():
        br = dict(br)
        if br.pop('circular', False):
            eps = [dict(e) for e in ep0]
            for I in 'HL':
                d = np.asarray(eps[1]['d_' + I]).astype(complex)
                d[row] += s[I]
                eps[1]['d_' + I] = d
        else:
            x = {I: v.copy() for I, v in x0.items()}
            if 'start' in br:
                br['start'] = seg_start
            inject(x, 0.0, wf, ext, y, seg_start + t_a, **br)
            eps = strain.condition(x['H'], x['L'], FS, 0.0, **kw)
        res = {}
        for where, (e, r) in (('own', (1, row)), ('prev', (1, row - 1))):
            for I in 'HL':
                res['%s_%s' % (where, I)] = mf_numbers(s[I], eps[e]['d_' + I][r], ep0[e]['d_' + I][r], eps[e]['sig2'],
                                                       nu)
        out['cases'][name] = res
    return out


def run_controls(a):
    S, meta = load_spectrum(a.spectrum)
    notches = [tuple(map(float, p)) for p in np.asarray(meta['notches']).reshape(-1, 2)]
    breaks_i = {'correct': {}, 'delay sign flipped': dict(delay_sign=-1.0), 'phase sign flipped': dict(phase_sign=-1.0),
                'time reversed (Fourier sign)': dict(reverse=True)}
    breaks_ii = {'correct': {}, 'cut at the segment start': dict(start=True), 'segment-level (circular)':
                 dict(circular=True)}
    out = {}
    for known in (False, True):
        label = 'known spectrum' if known else 'estimated spectrum (O3b settings)'
        t0 = time.time()
        out['i, ' + label] = control_case(S, notches, 4.0, breaks_i, a.seed, known)
        out['ii, ' + label] = control_case(S, notches, 0.5, breaks_ii, a.seed, known)
        print('%s: %.0f s' % (label, time.time() - t0), flush=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    for key, r in out.items():
        print('\n%s: t_H %.2f s, tau_H %.5f s, t_pk0 %.5f s; predicted (prev, own, next) H %s L %s' % (
            key, r['t_H'], r['tau_H'], r['t_pk0'],
            ', '.join('%.4f' % r['predicted']['H'][k] for k in ('prev', 'own', 'next')),
            ', '.join('%.4f' % r['predicted']['L'][k] for k in ('prev', 'own', 'next'))))
        for name, c in r['cases'].items():
            print('  %-30s ' % name + '; '.join('%s %s: opt %.2f sig %.3f snr0 %.2f lag %+d' % (
                w, I, c['%s_%s' % (w, I)]['opt'], c['%s_%s' % (w, I)]['sig'], c['%s_%s' % (w, I)]['snr0'],
                c['%s_%s' % (w, I)]['lag']) for w in ('own', 'prev') for I in 'HL'))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# the fits
def load_truth(root, tag=''):
    files = sorted(p for p in glob.glob(os.path.join(root, 'simstrain_truth' + ('_' + tag if tag else ''), 'truth_*.npz'))
                   if not p.endswith('.tmp.npz'))
    parts = []
    for p in files:
        with np.load(p) as z:
            parts.append({k: z[k] for k in z.files})
    keys = set.intersection(*(set(p) for p in parts))
    return {k: np.concatenate([p[k] for p in parts]) for k in sorted(keys)}


def jackknife(fn, idx, blocks=64):
    """(estimate, standard error) of fn(indices) by the block jackknife over `blocks` contiguous blocks of idx"""
    est = fn(idx)
    parts = np.array_split(idx, blocks)
    vals = np.array([fn(np.concatenate([parts[i] for i in range(blocks) if i != j])) for j in range(blocks)])
    se = np.sqrt((blocks - 1) / blocks * np.sum((vals - vals.mean(axis=0)) ** 2, axis=0))
    return est, se


def run_factors(a):
    pf = _load_script('exmarg_plant_fit', os.path.join(HERE, 'plant_fit.py'))
    tr = load_truth(a.root)
    where = {int(g): i for i, g in enumerate(tr['gps'])}
    res = {}
    arms = [x for x in a.arms.split(',') if x]
    data = {}
    for arm in arms:
        d = os.path.join(a.runs, 'simstrain_' + arm, 'zero')
        cfg, _, dd, runs, _ = pf.load(d)
        idx = np.array([where[int(round(g))] for g in dd['gps']])
        data[arm] = dict(lnL=dd['ln_lambda_bar'], keep=(dd['wmax_H'] <= pf.VETO_W) & (dd['wmax_L'] <= pf.VETO_W),
                         idx=idx, t_pe=float(np.mean([r['t_pe'] for r in runs])), wmax_H=dd['wmax_H'],
                         wmax_L=dd['wmax_L'], w_inf=dd['w_inf'])
    n = tr['gps'].size
    present = tr['present']
    full = {}
    for arm in arms:
        lnL = np.full(n, np.nan)
        keep = np.zeros(n, bool)
        lnL[data[arm]['idx']] = data[arm]['lnL']
        keep[data[arm]['idx']] = data[arm]['keep']
        full[arm] = (lnL, keep)
        own = np.flatnonzero(keep)
        r = pf.nbar_fit(lnL[own])
        truth_frac = 0.0 if arm.startswith('N') else float(present[own].mean())
        x = np.expm1(lnL[own][~present[own]]) if not arm.startswith('N') else np.expm1(lnL[own])
        res[arm] = dict(analysed=int(data[arm]['idx'].size), kept=int(own.size), nbar=r['nbar'], err=r['err']['nbar'],
                        edge=r['edge'], truth=truth_frac, pull=(r['nbar'] - truth_frac) / r['err']['nbar'],
                        unit_mean=float(x.mean()), unit_se=float(x.std(ddof=1) / np.sqrt(x.size)), unit_n=int(x.size),
                        t_pe=data[arm]['t_pe'], w_inf_max=float(data[arm]['w_inf'].max()))
    main = [x for x in ('A', 'B', 'C', 'D') if x in arms]
    common = np.ones(n, bool)
    for arm in main + [x for x in arms if x == 'N']:
        common &= full[arm][1]
    cidx = np.flatnonzero(common)
    res['common'] = dict(n=int(cidx.size), mergers=int(present[cidx].sum()), truth=float(present[cidx].mean()))

    def nb(arm, ii):
        return pf.nbar_fit(full[arm][0][ii])['nbar']

    pairs = {k: v for k, v in (('edge loss C/A', ('C', 'A')), ('coverage B/A', ('B', 'A')), ('both D/A', ('D', 'A')))
             if set(v) <= set(main)}
    full_set = set(main) == set('ABCD')
    fac = {}

    def factors(ii):
        v = {arm: nb(arm, ii) for arm in main}
        out = [v['A'] / 1.0]
        out += [v[p] / v[q] if v[q] > 0 else np.nan for p, q in pairs.values()]
        if full_set:
            out += [v['C'] * v['B'] / v['A'] ** 2 if v['A'] > 0 else np.nan]
            out += [v['D'] / v['A'] - v['C'] * v['B'] / v['A'] ** 2 if v['A'] > 0 else np.nan]
        out += [v[x] for x in main]
        return np.array(out)

    names = ['nbar A'] + list(pairs) + (['product (C/A)(B/A)', 'D/A - product'] if full_set else []) + [
        'nbar ' + x for x in main]
    if 'A' in main and pairs:
        est, se = jackknife(factors, cidx, a.blocks)
        fac['all'] = dict(n=int(cidx.size), truth=float(present[cidx].mean()),
                          **{nm: [float(e), float(s)] for nm, e, s in zip(names, est, se)})
        # by detector-frame total mass (of A), edges at the quartiles of the planted mergers fixed before any fit
        mA = tr['mdet_A']
        q_edges = np.quantile(mA[present], [0.25, 0.5, 0.75])
        edges = np.concatenate([[20.0], q_edges, [400.0]])
        fac['mass_edges'] = edges.tolist()
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = common & (~present | ((mA >= lo) & (mA < hi if hi < 400 else mA <= hi)))
            ii = np.flatnonzero(sel)
            est, se = jackknife(factors, ii, a.blocks)
            fac['M %.1f-%.1f' % (lo, hi)] = dict(n=int(ii.size), mergers=int(present[ii].sum()),
                                                  truth=float(present[ii].mean()),
                                                  **{nm: [float(e), float(s)] for nm, e, s in zip(names, est, se)})
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)):
            sel = common & (~present | ((tr['t_H'] >= lo) & (tr['t_H'] < hi)))
            ii = np.flatnonzero(sel)
            est, se = jackknife(factors, ii, a.blocks)
            fac['t_H %.1f-%.1f s' % (lo, hi)] = dict(n=int(ii.size), mergers=int(present[ii].sum()),
                                                     truth=float(present[ii].mean()),
                                                     **{nm: [float(e), float(s)] for nm, e, s in zip(names, est, se)})
    res['factors'] = fac
    # control iii: C against A (D against B), segment by segment, for mergers whose in-band signal is in the segment
    ctl = {}
    for x, ref in (('C', 'A'), ('D', 'B')):
        if 'sig_H_' + x not in tr:
            continue
        for thr in (0.999, 0.99):
            inside = present & (tr['frac_own_H_' + ref] >= thr) & (tr['frac_own_L_' + ref] >= thr)
            for snr_min in (0.0, 3.0, 6.0):
                optn = np.sqrt(tr['opt_H_' + ref] ** 2 + tr['opt_L_' + ref] ** 2)
                sel = inside & (optn >= snr_min)
                sx = np.sqrt(tr['sig_H_' + x] ** 2 + tr['sig_L_' + x] ** 2)[sel]
                sr = np.sqrt(tr['sig_H_' + ref] ** 2 + tr['sig_L_' + ref] ** 2)[sel]
                ratio = sx / sr
                rn = np.sqrt(tr['snr0_H_' + x] ** 2 + tr['snr0_L_' + x] ** 2)[sel] / np.sqrt(
                    tr['snr0_H_' + ref] ** 2 + tr['snr0_L_' + ref] ** 2)[sel]
                lag = np.concatenate([tr['lag_H_' + x][sel], tr['lag_L_' + x][sel]])
                if ratio.size:
                    ctl['%s/%s, frac_own >= %g, opt >= %g' % (x, ref, thr, snr_min)] = dict(
                        n=int(ratio.size), ratio_quantiles=np.quantile(ratio, [0.01, 0.16, 0.5, 0.84, 0.99]).tolist(),
                        ratio_mean=float(ratio.mean()), ratio_sd=float(ratio.std()),
                        noisy_ratio_quantiles=np.quantile(rn, [0.01, 0.16, 0.5, 0.84, 0.99]).tolist(),
                        lag_nonzero=int(np.sum(lag != 0)), lag_abs_max=float(np.abs(lag).max()))
    res['control iii'] = ctl
    # bookkeeping: B's masses outside the bank, segments the conditioning flagged, and the merger-free segments whose
    # next segment holds a merger (they carry its inspiral in C and D)
    m = present
    res['counts'] = dict(segments=int(n), mergers=int(m.sum()),
                         B_mdet_below_20=int(np.sum(m & (tr['mdet_B'] < 20))),
                         B_mdet_above_400=int(np.sum(m & (tr['mdet_B'] > 400))),
                         not_analysable={x: int(np.sum(~tr['analysable_' + x])) for x in ARMS if 'analysable_' + x in tr},
                         vetoed={arm: int(data[arm]['idx'].size - data[arm]['keep'].sum()) for arm in arms})
    nxt = np.zeros(n, bool)
    nxt[:-1] = m[1:] & (tr['stretch'][1:] == tr['stretch'][:-1])
    um = {}
    for arm in arms:
        lnL, keep = full[arm]
        for label, sel in (('next segment holds a merger', keep & ~m & nxt), ('next segment without', keep & ~m & ~nxt)):
            x = np.expm1(lnL[sel])
            um['%s, %s' % (arm, label)] = [float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)), int(x.size)]
    res['unit mean of merger-free segments'] = um
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    return 0


def run_control3(a):
    """control iii with its breaks, from the epoch files: for the mergers whose in-band whitened energy is inside their
    segment (frac_own >= --inside in both detectors) and of network opt >= --snr-min, the projection of d_C - d_N on
    the planted segment-level signal s = d_A - d_N (sig of mf_numbers), over that of s itself; then the same with the
    template of a flipped delay (s_H e^{+4 pi i f tau_H}: tau -> -tau) and of the conjugate phase convention (s e^{-2 i
    arg A_I}, A_I = y R_I e^{-i phase})"""
    tr = load_truth(a.root)
    where = {int(g): i for i, g in enumerate(tr['gps'])}
    sel = tr['present'] & (tr['frac_own_H_A'] >= a.inside) & (tr['frac_own_L_A'] >= a.inside) & (
        np.hypot(tr['opt_H_A'], tr['opt_L_A']) >= a.snr_min)
    out = {k: [] for k in ('correct', 'delay flipped', 'phase conjugate')}
    if a.own:                       # each detector on its own curve (strain.own_curves), over the epoch's signal gain
        out['own curves, over the signal gain'] = []
        out['own curves, without the gain'] = []
    if a.spectrum:                  # the per-bin conditional gain kappa_P sqrt(s_I / sigma^2), sigma^2 = v T S / 2
        S = None if a.spectrum == 'epochs' else load_spectrum(a.spectrum)[0]
        kP = gain_numbers(N, 66, 'median')['kappa_P']
        out['over the conditional gain'] = []
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder(a.root, 'N'), 'epoch_*.npz'))
                   if not p.endswith('.tmp.npz'))
    vbin = None
    for name in files:
        eps = {x: strain.read_epoch(os.path.join(folder(a.root, x), name)) for x in 'NAC'}
        f = eps['N']['nu'] / eps['N']['T']
        w = 1.0 / eps['C']['sig2']
        if a.spectrum and vbin is None:
            if a.spectrum == 'epochs':  # E P per bin: the mean of the hand-over curves of every epoch, both detectors
                acc = [0.0, 0]
                for nm in files:
                    with np.load(os.path.join(folder(a.root, 'N'), nm)) as z:
                        acc[0] = acc[0] + z['sig2_ref_H'] + z['sig2_ref_L']
                        acc[1] += 2
                sig2_true = acc[0] / acc[1]
                vbin = True
            else:
                vbin = strain.bin_variance(FS, T, eps['N']['nu'], band=tuple(eps['N']['band']),
                                           roll=tuple(eps['N']['roll']), notches=tuple(map(tuple, eps['N']['notches'])))
                sig2_true = vbin * T * S(f) / 2.0
        for r, g in enumerate(eps['N']['gps']):
            j = where[int(round(g))]
            if not sel[j]:
                continue
            fac = detector_factors(ext_of(tr, j), tr['y'][j])
            # network projected SNR: sqrt(2) sum_I Re <t_I, d_C - d_N> / sqrt(sum_I <s_I, s_I>), over that of s itself
            num, den = {k: 0.0 for k in out}, 0.0
            for I in 'HL':
                s = eps['A']['d_' + I][r].astype(complex) - eps['N']['d_' + I][r].astype(complex)
                dc = eps['C']['d_' + I][r].astype(complex) - eps['N']['d_' + I][r].astype(complex)
                A_I, tau = fac[I]
                tmpl = {'correct': s, 'delay flipped': s * np.exp(4j * np.pi * f * tau),
                        'phase conjugate': s * np.exp(-2j * np.angle(A_I))}
                den += np.sum(np.abs(s) ** 2 * w)
                for key, t in tmpl.items():
                    num[key] += np.sum(np.conj(t) * dc * w).real
                if a.spectrum:
                    gam = kP * np.sqrt(eps['C']['sig2_ref_' + I] / np.where(sig2_true > 0, sig2_true, np.inf))
                    num['gain'] = num.get('gain', 0.0) + np.sum(np.abs(s) ** 2 * gam * w)
            for key in ('correct', 'delay flipped', 'phase conjugate'):
                out[key].append(num[key] / den)
            if a.own:
                own = {x: strain.own_curves(eps[x]) for x in 'NAC'}
                kap = strain.signal_gain(eps['C'])
                nu_, de_, dg_ = 0.0, 0.0, 0.0
                for I in 'HL':
                    s = own['A'][I][0][r] - own['N'][I][0][r]
                    dc = own['C'][I][0][r] - own['N'][I][0][r]
                    wo = 1.0 / own['C'][I][1]
                    nu_ += np.sum(np.conj(s) * dc * wo).real
                    de_ += np.sum(np.abs(s) ** 2 * wo)
                    dg_ += kap[I] * np.sum(np.abs(s) ** 2 * wo)
                out['own curves, over the signal gain'].append(nu_ / dg_)
                out['own curves, without the gain'].append(nu_ / de_)
            if a.spectrum:
                out['over the conditional gain'].append(num['correct'] / num['gain'])
    res = {'n': int(sel.sum()), 'inside': a.inside, 'snr_min': a.snr_min}
    for k, v in out.items():
        v = np.array(v)
        res[k] = dict(n=int(v.size), quantiles=np.quantile(v, [0.01, 0.16, 0.5, 0.84, 0.99]).tolist(),
                      mean=float(v.mean()), sd=float(v.std()))
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# the gain of the conditioning on a signal, derived
#
# strain.condition (estimated spectra) hands over, in bin nu of segment i of detector I,
#     d_I = sqrt(s_I) W_i sqrt(G / (c Phat_{-i})) + sqrt(sig2 - s_I) xi,      sig2 = max(s_H, s_L)
# with P_j = |W_j|^2 G the periodogram of block j in strain units (the smooth pre-whitening S0 cancels in it), Phat_{-i}
# the median (or mean) of the P_j of the other N blocks of the epoch over its expectation's bias, c = strain.correction,
# and s_I the same estimate from the n_ref blocks of the neighbouring epochs.  A signal h (true strain) arrives as
#     gamma h,   gamma = sqrt(g_s / (c g_P)),   g_P = Phat_{-i} / sigma^2,   g_s = s_I / sigma^2,   sigma^2 = E P_j,
# g_P and g_s independent and independent of block i.  E[gamma^2] = E[g_s] E[1 / (c g_P)] = 1 (c = E[1/g_P]): the
# signal POWER is unbiased; its AMPLITUDE is not:
#     kappa = E[gamma] = kappa_P kappa_s,   kappa_P = E[g_P^{-1/2}] / sqrt(E[g_P^{-1}]),   kappa_s = E[g_s^{1/2}].
# The matched filter of the analysis weights bin nu by |h|^2 / sig2 with sig2 = sigma^2 max(g_sH, g_sL), correlated with
# g_sI, so the amplitude it recovers is, over many bins,
#     kappa_eff = sum gamma w / sum w -> kappa_P E[g_sH^{1/2} / max(g_sH, g_sL)] / E[1 / max(g_sH, g_sL)].
# Bin to bin, gamma scatters with variance 1 - kappa^2, independently between bins (the periodograms of different bins
# are independent): the projected matched filter of a merger of SNR rho then carries an extra variance about
# (E[gamma^2 w^2] ... ) ~ (1 - kappa^2) rho^2 / n_eff, n_eff = (sum w)^2 / sum w^2 its effective number of bins.

def estimate_density(n, method, x):
    """density, on the uniform grid x (from 0), of g = estimate / truth for an estimate from n independent exponential
    periodograms: 'mean' Gamma(n, 1/n); 'median' numpy's median of n Exp(1) over strain.median_bias(n): for odd n the
    order statistic X_(k+1) (k = n // 2), for even n X_(k) + E / n with E ~ Exp(1) independent (Renyi's representation,
    as strain._median_moments), whose density is that of X_(k) convolved with n e^{-n t} (recursion on the grid)"""
    from scipy import special, stats
    from scipy.signal import lfilter
    x = np.asarray(x, float)
    if method == 'mean':
        return stats.gamma.pdf(x, a=n, scale=1.0 / n)
    mu = strain.median_bias(n)
    m = mu * x
    k = n // 2
    with np.errstate(divide='ignore'):
        lp = np.log(-np.expm1(-m))
    if n % 2:
        f = np.exp(special.gammaln(n + 1) - 2.0 * special.gammaln(k + 1) + k * lp - (k + 1) * m)
    else:
        fk = np.exp(special.gammaln(n + 1) - special.gammaln(k) - special.gammaln(k + 1) + (k - 1) * lp - (k + 1) * m)
        h = m[1] - m[0]
        a = np.exp(-n * h)
        inp = 0.5 * h * (fk + a * np.concatenate([[0.0], fk[:-1]]))
        f = n * lfilter([1.0], [1.0, -a], inp)
    return mu * np.where(m > 0, f, 0.0)


def gain_numbers(N=32, n_ref=66, method='median', xmax=6.0, ngrid=600000):
    """the gain of the conditioning on a signal (see above), by quadrature over the densities of estimate_density:
    dict c (= strain.correction(N, method), recomputed), kappa_P, kappa_s, kappa, kappa_eff, var (1 - kappa^2), and the
    normalization checks"""
    from scipy.integrate import cumulative_trapezoid, trapezoid
    x = np.linspace(0.0, xmax, ngrid + 1)
    fP, fs = estimate_density(N, method, x), estimate_density(n_ref, method, x)
    xs = np.where(x > 0, x, np.inf)
    c = trapezoid(fP / xs, x)
    kP = trapezoid(fP / np.sqrt(xs), x) / np.sqrt(c)
    ks = trapezoid(fs * np.sqrt(x), x)
    Fs = cumulative_trapezoid(fs, x, initial=0.0)
    tail = trapezoid(fs / xs, x) - cumulative_trapezoid(fs / xs, x, initial=0.0)
    num = trapezoid(fs * np.sqrt(x) * (Fs / xs + tail), x)
    den = 2.0 * trapezoid(fs * Fs / xs, x)
    return dict(N=N, n_ref=n_ref, method=method, c=float(c), c_strain=float(strain.correction(N, method)),
                kappa_P=float(kP), kappa_s=float(ks), kappa=float(kP * ks), kappa_eff=float(kP * num / den),
                var=float(1.0 - (kP * ks) ** 2), checks=dict(norm_P=float(trapezoid(fP, x)),
                                                             mean_P=float(trapezoid(x * fP, x)),
                                                             norm_s=float(trapezoid(fs, x)),
                                                             mean_s=float(trapezoid(x * fs, x))))


def gain_monte_carlo(N=32, n_ref=66, method='median', draws=400000, seed=1):
    """the same numbers from direct draws of exponential periodograms (the reference of the quadrature)"""
    rng = np.random.default_rng(seed)
    est = (lambda P: np.median(P, axis=1) / strain.median_bias(P.shape[1])) if method == 'median' else \
        (lambda P: P.mean(axis=1))
    c = strain.correction(N, method)
    gP = est(rng.exponential(size=(draws, N)))
    gs = {I: est(rng.exponential(size=(draws, n_ref))) for I in 'HL'}
    gam = np.sqrt(gs['H'] / (c * gP))
    w = 1.0 / np.maximum(gs['H'], gs['L'])
    se = lambda v: float(v.std() / np.sqrt(v.size))
    return dict(kappa=float(gam.mean()), kappa_se=se(gam), kappa_eff=float((gam * w).mean() / w.mean()),
                kappa_eff_se=se(gam * w) / float(w.mean()), E_gamma2=float((gam ** 2).mean()), E_gamma2_se=se(gam ** 2))


def n_eff_of(m1d, m2d, chi, S, notches, T_=T):
    """the effective number of bins of a template in one detector, (sum w)^2 / sum w^2 with w = a^2 / sig2 on the bins
    that are not notched, sig2 = T S / 2 of the spectrum S"""
    nu, f, a, _ = profile(m1d, m2d, chi, chi, T_, FMIN, FMAX, 1.0)
    keep = band_mask(f, notches)
    w = a[keep] ** 2 / (T_ * S(f[keep]) / 2.0)
    return float(w.sum() ** 2 / (w ** 2).sum())


def run_gain(a):
    """(a) the gain kappa and kappa_eff for the estimators of the run, quadrature against Monte Carlo; (b) its scatter
    and the extra variance of the matched filter of a merger of network SNR 8; (c) on the arms of <root>: the projected
    matched-filter amplitude of the strain-level injection (C) over that of the segment-level one (A) times kappa_eff
    of each merger's epoch; (d) the prediction of Nbar_C / Nbar_A in the linear regime"""
    out = {}
    tab = {}
    for method in ('median', 'mean'):
        for n_ref in (66, 33):
            q = gain_numbers(32, n_ref, method)
            mc = gain_monte_carlo(32, n_ref, method)
            tab['%s, N 32, n_ref %d' % (method, n_ref)] = dict(quadrature=q, monte_carlo=mc)
    if True:                                          # closed forms of the mean: Gamma(N, 1/N) and Gamma(n, 1/n)
        from scipy.special import gammaln
        N_, n_ = 32, 66
        kP = np.exp(gammaln(N_ - 0.5) - gammaln(N_)) * np.sqrt(N_) / np.sqrt(N_ / (N_ - 1.0))
        ks = np.exp(gammaln(n_ + 0.5) - gammaln(n_)) / np.sqrt(n_)
        tab['mean, N 32, n_ref 66']['closed_form'] = dict(kappa_P=float(kP), kappa_s=float(ks), kappa=float(kP * ks))
    out['(a) gain'] = tab
    S, meta = load_spectrum(a.spectrum)
    notches = [tuple(map(float, p)) for p in np.asarray(meta['notches']).reshape(-1, 2)]
    var = tab['median, N 32, n_ref 66']['quadrature']['var']
    rows = {}
    for M in (20.0, 30.0, 60.0, 100.0, 200.0, 400.0):
        ne = n_eff_of(M / 2, M / 2, 0.0, S, notches)
        rows['M_det %g' % M] = dict(n_eff_one_detector=ne, extra_variance_rho8_one_detector=var * 64.0 / ne,
                                    extra_variance_rho8_equal_detectors=var * 64.0 / (2 * ne))
    out['(b) scatter'] = dict(var_gamma=var, sd_gamma=float(np.sqrt(var)), rho=8.0, templates=rows)
    if a.root:
        tr = load_truth(a.root)
        m = tr['present']
        n_ep = (N + 1)
        per = 69 * n_ep
        e = (tr['k'] % per) // n_ep
        n_ref = np.where((e == 0) | (e == 68), 33, 66)
        keff = {nr: tab['median, N 32, n_ref %d' % nr]['quadrature']['kappa_eff'] for nr in (33, 66)}
        kap = {nr: tab['median, N 32, n_ref %d' % nr]['quadrature']['kappa'] for nr in (33, 66)}
        ke = np.where(n_ref == 33, keff[33], keff[66])
        kk = np.where(n_ref == 33, kap[33], kap[66])
        optn = np.hypot(tr['opt_H_A'], tr['opt_L_A'])
        inside = m & (tr['frac_own_H_A'] >= 0.999) & (tr['frac_own_L_A'] >= 0.999)
        num = tr['sig_H_C'] * tr['opt_H_A'] + tr['sig_L_C'] * tr['opt_L_A']
        den = tr['opt_H_A'] ** 2 + tr['opt_L_A'] ** 2
        r = num / np.where(den > 0, den, np.nan)
        cres = {}
        for smin in (2.0, 3.0, 4.0, 6.0):
            s = inside & (optn >= smin)
            ne = np.array([2 * n_eff_of(tr['m1_det'][j], tr['m1_det'][j], 0.0, S, notches) for j in np.flatnonzero(s)])
            x = r[s] / ke[s]
            cres['opt >= %g' % smin] = dict(
                n=int(s.sum()), ratio_over_kappa_eff=[float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))],
                ratio_over_kappa=[float((r[s] / kk[s]).mean()), float((r[s] / kk[s]).std(ddof=1) / np.sqrt(x.size))],
                ratio_raw=[float(r[s].mean()), float(r[s].std(ddof=1) / np.sqrt(x.size))],
                sd_measured=float(x.std(ddof=1)), sd_predicted_gain_scatter=float(np.sqrt(np.mean(var / ne))),
                n_ref_33=int(np.sum(n_ref[s] == 33)))
        out['(c) C / (kappa_eff A), mergers inside their segment'] = cres
        # (d) linear regime: E[Lbar - 1 | merger] proportional to the projected power, kappa_eff^2 (a^2 + b^2 + c^2) rho^2
        # for C (own, previous, next segment), rho^2 for A; Nbar proportional to the sum over mergers
        pw_A = tr['opt_H_A'] ** 2 + tr['opt_L_A'] ** 2
        split = sum((tr['frac_%s_%s_A' % (w, I)] ** 2) * tr['opt_%s_A' % I] ** 2 for w in ('prev', 'own', 'next')
                    for I in 'HL')
        own = sum((tr['frac_own_%s_A' % I] ** 2) * tr['opt_%s_A' % I] ** 2 for I in 'HL')
        mm = m & np.isfinite(pw_A)
        out['(d) prediction'] = dict(
            kappa_eff_mean=float(np.mean(ke[mm] ** 2 * pw_A[mm]) / np.mean(pw_A[mm])),
            split_own_only=float(own[mm].sum() / pw_A[mm].sum()),
            split_with_neighbours=float(split[mm].sum() / pw_A[mm].sum()),
            nbar_C_over_A_linear=float((ke[mm] ** 2 * split[mm]).sum() / pw_A[mm].sum()),
            own_power_measured_noiseless=float(np.sum(tr['sig_H_C'][mm] ** 2 + tr['sig_L_C'][mm] ** 2) /
                                               pw_A[mm].sum()))
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


def run_owncheck(a):
    """the arms on own curves (strain.own_curves, as analyse.py --unequal reads them): per band, the mean power of the
    own-curve data of N and C over sig2_ref_I (1 for noise; C carries the mergers' tiny power), and of A - N against the
    equalized difference (the planted signal, the same: the noise of step 5 is common); the signal gains of the epochs;
    the asymmetry w of the model powers kappa_I^2 u_I of the bank (plant.MASSES) in every epoch"""
    from exmarg.search.segments import D0 as D0_
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder(a.root, 'N'), 'epoch_*.npz'))
                   if not p.endswith('.tmp.npz'))[::a.every]
    edges = strain.MONITOR_EDGES
    acc = {}
    a2 = None
    w_all, gains, diff_max = [], [], 0.0
    for name in files:
        eps = {x: strain.read_epoch(os.path.join(folder(a.root, x), name)) for x in 'NAC'}
        f, nu = eps['N']['f'], eps['N']['nu']
        keep = ~np.asarray(eps['N']['notched'], bool)
        if a2 is None:
            a2 = np.zeros((plant.MASSES.size, nu.size))
            for i, M in enumerate(plant.MASSES):
                nu_t, _, a1, _ = profile(M / 2, M / 2, 0.0, 0.0, T, FMIN, FMAX, 1.0)
                a2[i, np.searchsorted(nu, nu_t)] = (a1 / D0_) ** 2
        own = {x: strain.own_curves(eps[x]) for x in 'NAC'}
        for x in 'NC':
            for I in 'HL':
                p = np.abs(own[x][I][0]) ** 2 / np.where(keep, own[x][I][1], np.inf)
                for k in range(len(edges) - 1):
                    sel = keep & (f >= edges[k]) & (f < edges[k + 1] if k < len(edges) - 2 else f <= edges[k + 1])
                    s_ = acc.setdefault((x, I, k), [0.0, 0])
                    s_[0] += p[:, sel].sum()
                    s_[1] += p[:, sel].size
        for I in 'HL':
            d_own = own['A'][I][0] - own['N'][I][0]
            d_eq = eps['A']['d_' + I].astype(complex) - eps['N']['d_' + I].astype(complex)
            scale = np.sqrt(own['N'][I][1])[None, :]
            diff_max = max(diff_max, float(np.abs((d_own - d_eq) / scale)[:, keep].max()))
        g = strain.signal_gain(eps['C'])
        gains.append((g['H'], g['L'], int(eps['C']['n_reference_H']), int(eps['C']['n_reference_L'])))
        s2 = strain.own_sig2(eps['C'])
        u = {I: (a2[:, keep] / s2[I][keep]).sum(axis=1) for I in 'HL'}
        mH, mL = g['H'] ** 2 * u['H'], g['L'] ** 2 * u['L']
        w_all.append((mH - mL) / (mH + mL))
    w_all = np.array(w_all)
    gains = np.array(gains)
    power = {'%s %s %g-%g Hz' % (x, I, edges[k], edges[k + 1]): [v[0] / v[1], np.sqrt(1.14 / v[1])]
             for (x, I, k), v in sorted(acc.items())}
    out = dict(epochs=len(files), power_over_sig2_ref=power,
               own_A_minus_N_vs_equalized_max_rel=diff_max,
               gains=dict(values=sorted(set(map(tuple, gains[:, :2].round(8).tolist()))),
                          n_reference=sorted(set(map(tuple, gains[:, 2:].astype(int).tolist())))),
               asym=dict(min=float(w_all.min()), max=float(w_all.max()), abs_max=float(np.abs(w_all).max()),
                         quantiles=np.quantile(w_all, [0.001, 0.5, 0.999]).tolist()))
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


def run_spectrum(a):
    f, curve, notched, names, notches = median_curve(a.week, a.epochs)
    fk, S = build_spectrum(f, curve, notched, notches)
    np.savez(a.out, fk=fk, S=S, f_band=f, sig2_median=curve, notched=notched, notches=notches,
             epochs=np.array(names), source=os.path.abspath(a.week), key='sig2_ref_L')
    print('%s: median of sig2_ref_L over %d epochs of %s; S at 20, 30, 100, 300, 512 Hz: %s' % (
        a.out, len(names), a.week, ', '.join('%.3e' % v for v in spectrum_callable(fk, S)([20, 30, 100, 300, 512]))))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('spectrum')
    q.add_argument('week')
    q.add_argument('out')
    q.add_argument('--epochs', type=int, default=50)
    q = sub.add_parser('generate')
    q.add_argument('spectrum')
    q.add_argument('root')
    q.add_argument('--stretches', type=int, default=18)
    q.add_argument('--epochs', type=int, default=69)
    q.add_argument('--workers', type=int, default=12)
    q.add_argument('--seed', type=int, default=1)
    q.add_argument('--nbar', type=float, default=NBAR)
    q.add_argument('--arms', default=','.join(ARMS))
    q.add_argument('--only', default='', help='comma-separated stretch indices')
    q.add_argument('--no-correct', action='store_true', help='condition without the correction factor (break iv)')
    q.add_argument('--tag', default='', help='suffix of the folders (simstrain_<arm>_<tag>)')
    q = sub.add_parser('controls')
    q.add_argument('spectrum')
    q.add_argument('out')
    q.add_argument('--seed', type=int, default=1)
    q = sub.add_parser('control3')
    q.add_argument('root')
    q.add_argument('out')
    q.add_argument('--inside', type=float, default=0.999)
    q.add_argument('--snr-min', type=float, default=3.0)
    q.add_argument('--own', action='store_true', help='also on own curves, over strain.signal_gain of the epoch')
    q.add_argument('--spectrum', default='', help='also the ratio over the per-bin conditional gain: sigma^2 from this spectrum file (v T S / 2), or '
                        '"epochs": the mean of the hand-over curves of all epochs')
    q = sub.add_parser('owncheck')
    q.add_argument('root')
    q.add_argument('out')
    q.add_argument('--every', type=int, default=1, help='every so many epochs')
    q = sub.add_parser('gain')
    q.add_argument('spectrum')
    q.add_argument('out')
    q.add_argument('--root', default='', help='the generated arms (truth tables) for parts c and d')
    q = sub.add_parser('factors')
    q.add_argument('root')
    q.add_argument('runs')
    q.add_argument('out')
    q.add_argument('--arms', default='N,A,B,C,D')
    q.add_argument('--blocks', type=int, default=64)
    a = p.parse_args(argv)
    return dict(spectrum=run_spectrum, generate=run_generate, controls=run_controls, control3=run_control3,
                gain=run_gain, owncheck=run_owncheck, factors=run_factors)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
