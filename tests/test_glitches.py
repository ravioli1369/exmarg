"""simulated glitches: the SNR, the detector, the rate and the arrival time are the requested ones"""
import numpy as np
import pytest

from exmarg.detectors import S
from exmarg.glitches import add_burst_glitches, add_template_glitches, power_law_snr
from exmarg.search import mass_bank

T, FMIN, FMAX = 4.0, 25.0, 256.0
KINDS = ['template', 'sine-gaussian', 'gaussian', 'ringdown']


def segments(n, seed=0, T=T, fmin=FMIN, fmax=FMAX, noise=True):
    """n segments of Gaussian noise (zeros if noise is False); the L noise curve is 1.7 times the H one"""
    nu = np.arange(int(np.ceil(fmin * T)), int(np.floor(fmax * T)) + 1)
    f = nu / T
    sig2_H = T * S('H', f) / 2.0
    sig2_L = 1.7 * sig2_H
    rng = np.random.default_rng(seed)
    d = {}
    for I, s2 in (('H', sig2_H), ('L', sig2_L)):
        g = rng.standard_normal((2, n, nu.size)) * noise
        d['d_' + I] = np.sqrt(s2 / 2.0) * (g[0] + 1j * g[1])
    return dict(d, f=f, nu=nu, sig2_H=sig2_H, sig2_L=sig2_L, T=T, fmin=fmin, fmax=fmax)


def add(kind, seg, eps, draw_rho, seed, **kw):
    if kind == 'template':
        return add_template_glitches(seg, eps, draw_rho, [30.0, 45.0, 60.0], seed, **kw)
    return add_burst_glitches(seg, eps, draw_rho, kind, seed, **kw)


def snr2(h, sig2):
    return 2.0 * np.sum(np.abs(h) ** 2 / sig2, axis=-1)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('I', 'HL')
def test_snr_detector_and_input(kind, I):
    seg = segments(30, seed=1)
    before = {key: np.array(v) for key, v in seg.items()}
    J = 'L' if I == 'H' else 'H'
    new, rec = add(kind, seg, {I: 1.0, J: 0.0}, power_law_snr(3.0, 4.0, 40.0), seed=5)
    for key in before:                                             # the input is not modified
        assert np.array_equal(seg[key], before[key])
    assert rec[I]['present'].all() and not rec[J]['present'].any()
    assert np.array_equal(new['d_' + J], seg['d_' + J])            # the other detector is bit-identical
    assert new['d_' + J] is not seg['d_' + J]
    got = snr2(new['d_' + I] - seg['d_' + I], seg['sig2_' + I])
    assert np.max(np.abs(got / rec[I]['rho'] ** 2 - 1)) < 1e-9
    assert (rec[I]['rho'] >= 4.0).all() and (rec[I]['rho'] <= 40.0).all()
    assert (rec[I]['kind'] == kind).all()
    assert np.array_equal(rec[J]['rho'] != rec[J]['rho'], np.ones(30, bool))      # nan where absent


def test_absent_segments_unchanged_and_sig2_shared():
    seg = segments(60, seed=2)
    seg['sig2'] = seg.pop('sig2_H')                                # one noise curve for both detectors
    del seg['sig2_L']
    new, rec = add_template_glitches(seg, 0.5, lambda rng: 9.0, [40.0], seed=3)
    for I in 'HL':
        p = rec[I]['present']
        assert 10 < p.sum() < 50
        assert np.array_equal(new['d_' + I][~p], seg['d_' + I][~p])
        assert np.all(new['d_' + I][p] != seg['d_' + I][p])
        diff = new['d_' + I][p] - seg['d_' + I][p]
        assert np.max(np.abs(snr2(diff, seg['sig2']) / 81.0 - 1)) < 1e-9


def test_rate_and_independence():
    n, eps = 6000, {'H': 0.30, 'L': 0.15}
    seg = segments(n, seed=3, T=2.0, fmin=25.0, fmax=64.0, noise=False)
    _, rec = add_template_glitches(seg, eps, lambda rng: 8.0, [60.0], seed=11)
    present = {I: rec[I]['present'] for I in 'HL'}
    for I in 'HL':
        sd = np.sqrt(eps[I] * (1 - eps[I]) / n)
        assert abs(present[I].mean() - eps[I]) < 4 * sd, (I, present[I].mean())
    both = np.sum(present['H'] & present['L'])                     # independent: n eps_H eps_L
    p = eps['H'] * eps['L']
    assert abs(both - n * p) < 4 * np.sqrt(n * p * (1 - p))
    r = np.corrcoef(present['H'], present['L'])[0, 1]
    assert abs(r) < 4 / np.sqrt(n)
    for e in (0.0, 1.0):
        _, rec = add_template_glitches(seg, e, lambda rng: 8.0, [60.0], seed=11)
        assert all(rec[I]['present'].all() == bool(e) and rec[I]['present'].any() == bool(e) for I in 'HL')


@pytest.mark.parametrize('kind', KINDS)
def test_reproducible_and_independent_of_other_segments(kind):
    seg = segments(24, seed=4)
    small = {key: (v[:8] if key in ('d_H', 'd_L') else v) for key, v in seg.items()}
    a, ra = add(kind, seg, 0.6, power_law_snr(2.0, 5.0, 20.0), seed=9)
    b, rb = add(kind, seg, 0.6, power_law_snr(2.0, 5.0, 20.0), seed=9)
    c, rc = add(kind, small, 0.6, power_law_snr(2.0, 5.0, 20.0), seed=9)
    d, rd = add(kind, seg, 0.6, power_law_snr(2.0, 5.0, 20.0), seed=10)
    assert np.array_equal(a['d_H'], b['d_H']) and np.array_equal(a['d_L'], b['d_L'])
    assert np.array_equal(a['d_H'][:8], c['d_H']) and np.array_equal(a['d_L'][:8], c['d_L'])
    assert np.array_equal(ra['H']['t0'][:8], rc['H']['t0'], equal_nan=True)
    assert not np.array_equal(ra['H']['present'], rd['H']['present'])


@pytest.mark.parametrize('kind', KINDS)
def test_segment_streams_are_separate(kind):
    """a sampler that consumes extra random numbers in its first call changes segment 0 and no other segment"""
    seg = segments(12, seed=5)
    calls = []

    def greedy(rng):
        calls.append(1)
        if len(calls) == 1:
            rng.uniform(size=7)
        return 9.0

    a, ra = add(kind, seg, 1.0, lambda rng: 9.0, seed=4)
    b, rb = add(kind, seg, 1.0, greedy, seed=4)
    assert not np.array_equal(a['d_H'][0], b['d_H'][0])
    assert np.array_equal(a['d_H'][1:], b['d_H'][1:]) and np.array_equal(a['d_L'][1:], b['d_L'][1:])
    assert ra['H']['present'].any()


def test_power_law_sampler():
    rng = np.random.default_rng(0)
    for gamma in (4.0, 1.0, 0.0):
        x = np.array([power_law_snr(gamma, 5.0, 30.0)(rng) for _ in range(20000)])
        assert x.min() >= 5.0 and x.max() <= 30.0
        cdf = (lambda r: (r ** (1 - gamma) - 5.0 ** (1 - gamma)) / (30.0 ** (1 - gamma) - 5.0 ** (1 - gamma))) \
            if gamma != 1.0 else (lambda r: np.log(r / 5.0) / np.log(6.0))
        for q in (0.1, 0.5, 0.9):
            level = np.mean(x < np.quantile(x, q))
            assert abs(cdf(np.quantile(x, q)) - level) < 0.02             # the empirical quantile sits on the cdf
    with pytest.raises(ValueError):
        power_law_snr(2.0, 10.0, 5.0)


def test_template_matched_filter_peak():
    """a noiseless template glitch of SNR rho and arrival time t0 has z(t0) = c u e^{i phase}, with c^2 u = rho^2 / 2
    and u = sum a^2 / sig2, so |z|^2 / u = rho^2 / 2 at the sample of t0 and below it elsewhere (Cauchy-Schwarz)"""
    seg = segments(1, noise=False)
    seg['sig2_L'] = seg['sig2_H'].copy()
    M = 40.0
    bank = mass_bank(np.array([M, 70.0]), seg, 'cpu')
    dt = bank.T / bank.n_time
    rho = 11.0
    sig2 = {'H': seg['sig2_H'], 'L': seg['sig2_L']}
    for I in 'HL':
        J = 'L' if I == 'H' else 'H'
        for n_try, step in ((0, dt), (1, None)):
            seg1 = segments(40, noise=False)
            seg1['sig2_L'] = seg1['sig2_H'].copy()
            new, rec = add_template_glitches(seg1, {I: 1.0, J: 0.0}, lambda rng: rho, [M], seed=20 + n_try,
                                             time_step=step)
            for k in range(0, 40, 8):
                lq = bank.trajectories({'H': new['d_H'][k], 'L': new['d_L'][k]}, sig2)['lq' + I]
                power = np.exp(2.0 * lq.astype(float)) / bank.u[0]
                pk = int(np.argmax(power[0]))
                i0 = int(np.rint(rec[I]['t0'][k] / dt)) % bank.n_time
                if step is not None:
                    assert pk == i0
                    assert abs(power[0, pk] / (rho ** 2 / 2) - 1) < 1e-4
                else:
                    assert abs((pk - i0 + bank.n_time // 2) % bank.n_time - bank.n_time // 2) <= 1
                    assert np.cos(np.pi / 8) ** 2 * rho ** 2 / 2 <= power[0, pk] <= rho ** 2 / 2 * (1 + 1e-4)
                assert power[0, pk] > 5 * np.sort(power[0])[-30]                  # a clear peak
                other = bank.trajectories({'H': new['d_H'][k], 'L': new['d_L'][k]}, sig2)['lq' + J]
                assert np.all(np.asarray(other) < -43)             # zero data: the floor 0.5 ln 1e-38 of the filter


def reference_fft(kind, p, t0, seg, fs=8192.0):
    """one-sided spectrum on the segment bins of the time-domain waveform sampled at fs, centred (or starting) at t0"""
    Tn = seg['T']
    t = np.arange(int(Tn * fs)) / fs
    s = (t - t0 + Tn / 2) % Tn - Tn / 2                                 # time from t0, wrapped
    if kind == 'gaussian':
        h = np.cos(p['phase']) * np.exp(-s ** 2 / (2 * p['width'] ** 2))
    elif kind == 'sine-gaussian':
        tau = p['Q'] / (2 * np.pi * p['f0'])
        h = np.exp(-s ** 2 / (2 * tau ** 2)) * np.cos(2 * np.pi * p['f0'] * s + p['phase'])
    else:
        tau = p['Q'] / (np.pi * p['f0'])
        step = np.where(s > 0, np.exp(-np.abs(s) / tau), np.where(s == 0, 0.5, 0.0))
        h = step * np.cos(2 * np.pi * p['f0'] * s + p['phase'])
    return np.fft.rfft(h)[seg['nu']] / fs


@pytest.mark.parametrize('kind,tol', [('gaussian', 1e-6), ('sine-gaussian', 1e-6), ('ringdown', 2e-3)])
def test_burst_spectrum_against_time_domain(kind, tol):
    """the analytic Fourier transform is the one of the sampled time-domain waveform (convention exp(-2 pi i f t))"""
    seg = segments(12, noise=False, seed=6)
    new, rec = add_burst_glitches(seg, {'H': 1.0, 'L': 0.0}, lambda rng: 10.0, kind, seed=8, f_range=(60.0, 200.0),
                                  Q_range=(4.0, 12.0), width_range=(2e-3, 4e-3), time_step=1 / 8192.0)
    r = rec['H']
    for k in range(12):
        p = {key: r[key][k] for key in ('f0', 'Q', 'width', 'phase') if key in r}
        ref = reference_fft(kind, p, r['t0'][k], seg)
        ref *= 10.0 / np.sqrt(snr2(ref, seg['sig2_H']))
        assert np.linalg.norm(new['d_H'][k] - ref) < tol * np.linalg.norm(ref)


def test_sine_gaussian_peaks_at_f0_and_ranges():
    seg = segments(20, noise=False, seed=7)
    new, rec = add_burst_glitches(seg, 1.0, lambda rng: 10.0, 'sine-gaussian', seed=2, f_range=(40.0, 240.0),
                                  Q_range=(30.0, 40.0))
    r = rec['H']
    assert (r['f0'] >= 40).all() and (r['f0'] <= 240).all() and (r['Q'] >= 30).all() and (r['Q'] <= 40).all()
    for k in range(20):
        assert abs(seg['f'][np.argmax(np.abs(new['d_H'][k]))] - r['f0'][k]) <= 1.0 / T     # the spectrum peaks at f0
    with pytest.raises(ValueError):
        add_burst_glitches(seg, 1.0, lambda rng: 10.0, 'chirp', seed=2)
