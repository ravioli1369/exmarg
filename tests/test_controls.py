"""control data sets: time slides, segment slides and phase randomization"""
import numpy as np
import pytest

from exmarg.detectors import S
from exmarg.profiles import profile
from exmarg.search import mass_bank
from exmarg.search.controls import phase_randomize, segment_slide, time_slide, time_step

T, FMIN, FMAX, M = 4.0, 25.0, 256.0, 40.0


def grid():
    nu = np.arange(int(np.ceil(FMIN * T)), int(np.floor(FMAX * T)) + 1)
    f = nu / T
    return nu, f, T * S('H', f) / 2.0


def template(nu, scale=1.0):
    nu_t, _, a, psi0 = profile(M / 2, M / 2, 0.0, 0.0, T, FMIN, FMAX, 1.0)
    w = np.zeros(nu.size, complex)
    w[np.searchsorted(nu, nu_t)] = a * np.exp(1j * psi0)
    return w * scale


def signal_segments(i_H, i_L, rho=15.0):
    """one noiseless segment with the template in H at matched-filter sample i_H and in L at i_L"""
    nu, f, sig2 = grid()
    w = template(nu)
    w *= rho / np.sqrt(2 * np.sum(np.abs(w) ** 2 / sig2))
    dt = T / (8 * int(nu.max()))
    d = {I: (w * np.exp(-2j * np.pi * f * i * dt))[None, :] for I, i in (('H', i_H), ('L', i_L))}
    return dict(d_H=d['H'], d_L=d['L'], f=f, nu=nu, sig2=sig2, T=T, fmin=FMIN, fmax=FMAX)


def peaks(seg):
    bank = mass_bank(np.array([M]), seg, 'cpu')
    out = {}
    for I in 'HL':
        z = bank.trajectories({'H': seg['d_H'][0], 'L': seg['d_L'][0]}, {'H': seg['sig2'], 'L': seg['sig2']})['lq' + I]
        power = np.exp(2.0 * z[0].astype(float)) / bank.u[0]
        out[I] = (int(np.argmax(power)), float(power.max()), bank.n_time)
    return out


def test_time_step_and_slide_peak():
    seg = signal_segments(1500, 6000)
    nu = seg['nu']
    dt = time_step(seg)
    assert dt == 1 / (8 * FMAX) and dt == T / (8 * nu.max())
    before = peaks(seg)
    assert before['H'][0] == 1500 and before['L'][0] == 6000
    n = before['H'][2]
    for shift_samples in (100, -100, 3000, 2500):                       # the last two wrap around T
        shift = shift_samples * dt
        slid = time_slide(seg, shift)
        after = peaks(slid)
        assert after['L'][0] == (6000 + shift_samples) % n
        assert after['H'][0] == 1500
        assert np.array_equal(slid['d_H'], seg['d_H'])
        assert abs(after['L'][1] / before['L'][1] - 1) < 1e-4          # peak value of a whole-sample shift is unchanged
        assert np.abs(np.abs(slid['d_L']) - np.abs(seg['d_L'])).max() < 1e-12 * np.abs(seg['d_L']).max()


def test_time_slide_roundtrip_input_and_keys():
    rng = np.random.default_rng(0)
    nu, f, sig2 = grid()
    d = {I: rng.standard_normal((5, nu.size)) + 1j * rng.standard_normal((5, nu.size)) for I in 'HL'}
    seg = dict(d_H=d['H'], d_L=d['L'], f=f, nu=nu, sig2=sig2, T=T, fmin=FMIN, fmax=FMAX)
    keep = {key: np.array(v) for key, v in seg.items()}
    slid = time_slide(seg, 0.37)                                       # not a whole number of samples: still exact
    back = time_slide(slid, -0.37)
    assert np.max(np.abs(back['d_L'] - seg['d_L'])) < 1e-13 * np.abs(seg['d_L']).max()
    assert np.array_equal(back['d_H'], seg['d_H']) and set(slid) == set(seg)
    assert slid['d_H'] is not seg['d_H']
    for key in keep:
        assert np.array_equal(seg[key], keep[key])
    assert np.allclose(np.abs(slid['d_L']) ** 2, np.abs(seg['d_L']) ** 2, rtol=1e-13, atol=0)
    assert not np.allclose(slid['d_L'], seg['d_L'])


def test_time_slide_refuses_short_shifts():
    seg = signal_segments(10, 20)
    for bad in (0.0, 0.0199, -0.0199, 0.01, T, T + 0.01, -T + 0.019):
        with pytest.raises(ValueError):
            time_slide(seg, bad)
    time_slide(seg, 0.0201)
    time_slide(seg, -0.0201)
    time_slide(seg, T / 2)


def test_segment_slide_pairing():
    n, nf = 7, 5
    d_H = (np.arange(n)[:, None] + 1j * np.arange(nf)[None, :]).astype(complex)
    d_L = (100 + 10 * np.arange(n)[:, None] + 1j * np.arange(nf)[None, :]).astype(complex)
    seg = dict(d_H=d_H, d_L=d_L, nu=np.arange(nf), T=1.0)
    for k in (1, -1, 3, -2, n + 2):
        new = segment_slide(seg, k)
        assert np.array_equal(new['d_H'], d_H)
        for i in range(n):
            assert np.array_equal(new['d_L'][i], d_L[(i + k) % n])
        assert new['d_H'] is not seg['d_H'] and new['d_L'] is not seg['d_L']
    assert np.array_equal(seg['d_L'], d_L)
    for bad in (0, n, -n, 1.5):
        with pytest.raises(ValueError):
            segment_slide(seg, bad)


def test_segment_slide_kills_coherence():
    """segment i of H carries the signal of the loud segment, the slid L of segment i + k does not: the pair of the
    slid data has the template in H only"""
    s0 = signal_segments(1500, 6000)
    zero = np.zeros_like(s0['d_L'])
    seg = dict(s0, d_H=np.concatenate([s0['d_H'], zero]), d_L=np.concatenate([s0['d_L'], zero]))
    new = segment_slide(seg, 1)
    assert np.array_equal(new['d_H'][0], s0['d_H'][0]) and np.all(new['d_L'][0] == 0)
    assert np.array_equal(new['d_L'][1], s0['d_L'][0]) and np.all(new['d_H'][1] == 0)


def test_phase_randomize_power_and_cross_spectrum():
    rng = np.random.default_rng(1)
    nu, f, sig2 = grid()
    d = {I: rng.standard_normal((20, nu.size)) + 1j * rng.standard_normal((20, nu.size)) for I in 'HL'}
    seg = dict(d_H=d['H'], d_L=d['L'], f=f, nu=nu, sig2=sig2, T=T, fmin=FMIN, fmax=FMAX)
    keep = {key: np.array(v) for key, v in seg.items()}
    pr = phase_randomize(seg, 4)
    for I in 'HL':
        assert np.max(np.abs(np.abs(pr['d_' + I]) - np.abs(seg['d_' + I]))) < 1e-14 * np.abs(seg['d_' + I]).max()
    cross = lambda s: s['d_H'] * np.conj(s['d_L'])                    # same phase in both detectors: kept per bin
    assert np.max(np.abs(cross(pr) - cross(seg))) < 1e-13 * np.abs(cross(seg)).max()
    for key in keep:
        assert np.array_equal(seg[key], keep[key])
    again = phase_randomize(seg, 4)
    assert np.array_equal(again['d_H'], pr['d_H']) and not np.array_equal(phase_randomize(seg, 5)['d_H'], pr['d_H'])
    ph = np.angle(pr['d_H'] / seg['d_H'])                              # the applied phases: uniform, independent
    assert abs(np.mean(np.exp(1j * ph))) < 4 / np.sqrt(ph.size)
    assert abs(np.mean(np.exp(1j * (ph[:, 1:] - ph[:, :-1])))) < 4 / np.sqrt(ph[:, 1:].size)
    assert abs(np.mean(np.exp(1j * (ph[1:] - ph[:-1])))) < 4 / np.sqrt(ph[1:].size)


def test_phase_randomize_destroys_the_peak():
    """before: |z|^2 / u = rho^2 / 2 at the signal.  after: z(t) = c sum_nu w_nu exp(i (phi_nu - 2 pi nu t / T)) with
    w_nu = |a|^2 / sig2, E|z|^2 / u = (rho^2 / 2) / N_eff, N_eff = (sum w)^2 / sum w^2.  Criterion: the maximum over
    the n_time samples stays below (rho^2 / 2 / N_eff) (ln n_time + 12) (a sum of random phases is a Gaussian
    variable, so |z|^2 is exponential: the chance of exceeding this is below 1e-5 even over the samples) and below
    a tenth of the original peak."""
    rho = 30.0
    seg = signal_segments(1500, 6000, rho=rho)
    before = peaks(seg)
    assert abs(before['H'][1] / (rho ** 2 / 2) - 1) < 1e-4
    nu, sig2 = seg['nu'], seg['sig2']
    w = np.abs(template(nu)) ** 2 / sig2
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    bound = rho ** 2 / 2 / n_eff * (np.log(before['H'][2]) + 12)
    assert bound < 0.1 * rho ** 2 / 2
    for seed in range(4):
        after = peaks(phase_randomize(seg, seed))
        for I in 'HL':
            assert after[I][1] < bound, (seed, I, after[I][1], bound)
