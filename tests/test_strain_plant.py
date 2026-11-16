"""scripts/validate/strain_plant.py in miniature (CPU, no tables): the segment-level signal is pp.simulate's (the code
path of analyse.py's inj variant), the long-grid waveform is profiles.profile on the segment bins, the arrival time is
the peak of the strain amplitude in H, and the controls of the strain-level plants with their breaks:
  i    a loud strain-level merger in the middle of a segment is the planted segment-level signal after the
       conditioning: amplitude, phase and time in both detectors; a flipped delay, a flipped phase and a reversed time
       axis each make it fail;
  ii   at 0.5 s after a segment start, the matched filter of the segment and of the one before recover the fractions of
       the in-band whitened energy that fall in each; cut at the segment start, the previous segment is empty, and the
       segment-level (circular) injection loses nothing;
  iii  end to end on two short stretches (N = 8): arm A is exactly the planted signal, arm C agrees with it segment by
       segment for the mergers whose in-band signal is inside their segment, and loses what the split predicts for the
       others; the same noise and mergers in every arm.
Spectrum: tests/strain_sim.py's H curve and notches, on the 1/8 Hz grid of a spectrum file.
"""
import importlib.util
import os

import numpy as np
import pytest
from scipy.signal import hilbert

import strain_sim as sim
from exmarg import strain
from exmarg.profiles import profile
from exmarg.search.segments import D0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts', 'validate', name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _script('strain_plant')
plant = sp.plant
SMALL = dict(N=8)
NU = np.arange(160, 4097)
EXT = dict(theta=1.4, phi=1.7, psi=0.7, cos_iota=0.6, phase=1.1)      # tau_H = +7.1 ms


@pytest.fixture(scope='module')
def spectrum(tmp_path_factory):
    path = tmp_path_factory.mktemp('spectrum') / 'spectrum.npz'
    fk = np.arange(int(2048 * 8) + 1) / 8.0
    np.savez(path, fk=fk, S=sim.psd('H')(fk), notches=np.array(sim.NOTCHES, float))
    S, meta = sp.load_spectrum(str(path))
    return str(path), S, [tuple(p) for p in meta['notches']]


# ---------------------------------------------------------------------------------------------------------------------
# the pieces


def test_segment_signal_is_the_plant_signal():
    """model_signal (written out) equals pp.simulate's signal taken as analyse.py takes it, for equal and unequal
    masses, at pp.simulate's own arrival time; another generator gives other extrinsic parameters (the check bites)"""
    for gps, (m1d, m2d, chi) in ((2000000008, (40.0, 40.0, 0.0)), (2000000016, (40.0, 14.0, 0.35)),
                                 (2000000024, (9.0, 9.0, 0.0))):
        ext = sp.extrinsic(plant.stream(1, gps, 'noise'))
        q = m2d / m1d
        mine = sp.model_signal(m1d, m2d, chi, 0.3, ext, ext['t'], NU)
        theirs = sp.plant_signal(1, gps, m1d * (1 + q), q, chi, 0.3, NU)
        other = sp.model_signal(m1d, m2d, chi, 0.3, sp.extrinsic(plant.stream(2, gps, 'noise')), ext['t'], NU)
        for I in 'HL':
            scale = np.abs(mine[I]).max()
            assert np.abs(mine[I] - theirs[I]).max() < 1e-9 * scale
            assert np.abs(other[I] - theirs[I]).max() > 0.1 * scale
        wf = sp.waveform(m1d, m2d, chi)
        s, t_a = sp.segment_signal(1, gps, m1d, m2d, chi, 0.3, ext, wf['t_pk0'], NU)
        tau_H = sp.responses(ext)['H'][1]
        assert t_a == pytest.approx(ext['t'] - tau_H - wf['t_pk0'], abs=1e-12)
        f = NU / 8.0
        for I in 'HL':
            assert np.allclose(s[I], theirs[I] * np.exp(2j * np.pi * f * (tau_H + wf['t_pk0'])), rtol=0,
                               atol=1e-9 * np.abs(s[I]).max())


def test_waveform_is_the_profile_on_the_segment_bins():
    """the long-grid waveform at f = nu / T is profiles.profile's a e^{i psi0} / D0 (the same template, phase and time
    origin whatever the grid and the start frequency), and fits on its grid"""
    for m1d, m2d, chi in ((40.0, 40.0, 0.0), (30.0, 12.0, 0.4), (10.0, 10.0, 0.0)):
        wf = sp.waveform(m1d, m2d, chi)
        nu, _, a, psi0 = profile(m1d, m2d, chi, chi, 8.0, 20.0, 512.0, 1.0)
        k = np.rint(nu * wf['Tg'] / 8.0).astype(int)
        assert np.allclose(wf['h'][k], a * np.exp(1j * psi0) / D0, rtol=1e-9, atol=0)
        assert wf['edge'] < 1e-6 and abs(wf['t_pk0']) < 0.01


def test_arrival_time_is_the_peak_of_the_strain_in_H():
    """injected into zero strain, the amplitude of the series (Hilbert transform, independent of the code's peak
    finder) peaks at segment start + t_H in H and at t_H - tau_H in L"""
    for (m1d, m2d, chi), t_H in (((40.0, 40.0, 0.0), 3.3), ((25.0, 9.0, -0.3), 7.95), ((15.0, 15.0, 0.0), 0.02)):
        wf = sp.waveform(m1d, m2d, chi)
        ext = dict(EXT, t=t_H)
        tau = {I: v[1] for I, v in sp.responses(ext).items()}
        seg_start = 300.0
        x = {I: np.zeros(int(400 * sp.FS)) for I in 'HL'}
        sp.inject(x, 0.0, wf, ext, 1.0, seg_start + t_H - tau['H'] - wf['t_pk0'])
        for I, expect in (('H', t_H), ('L', t_H - tau['H'])):
            a = np.abs(hilbert(x[I]))
            assert abs(np.argmax(a) / sp.FS - seg_start - expect) <= 1.0 / sp.FS


def test_whitened_split_sums_to_one(spectrum):
    _, S, notches = spectrum
    wf = sp.waveform(20.0, 20.0, 0.0)
    for t_H, inside in ((4.0, True), (0.5, False)):
        ext = dict(EXT, t=t_H)
        t_a = t_H - sp.responses(ext)['H'][1] - wf['t_pk0']
        sp_ = sp.whitened_split(wf, ext, 1.0, t_a, 0.0, S, notches)
        for I in 'HL':
            prev, own, nxt, rho2 = sp_[I]
            assert prev + own + nxt == pytest.approx(1.0, abs=1e-3) and rho2 > 0     # the band edges ring
            assert (own > 0.999) if inside else (0.05 < own < 0.95 and prev > 0.05)


def test_mf_numbers():
    rng = np.random.default_rng(4)
    sig2 = np.full(NU.size, 2.0)
    s = rng.standard_normal(NU.size) + 1j * rng.standard_normal(NU.size)
    d0 = rng.standard_normal(NU.size) + 1j * rng.standard_normal(NU.size)
    out = sp.mf_numbers(s, d0 + s, d0, sig2, NU)
    assert out['sig'] == pytest.approx(out['opt'], rel=1e-12) and out['lag'] == 0
    late = s * np.exp(-2j * np.pi * NU * 3 / 32768)
    assert sp.mf_numbers(s, d0 + late, d0, sig2, NU)['lag'] == 3
    assert sp.mf_numbers(s, d0 - s, d0, sig2, NU)['sig'] == pytest.approx(-out['opt'], rel=1e-12)


def test_qchi_follows_the_three_parameter_set_up():
    qc = sp.QChi()
    with np.load(sp.POPULATION) as z:
        mq = np.trapezoid(z['q'] * z['p_q'], z['q']) / np.trapezoid(z['p_q'], z['q'])
        mc = np.trapezoid(z['chi'] * z['p_chi'], z['chi']) / np.trapezoid(z['p_chi'], z['chi'])
    d = np.array([qc(1, 2000000000 + 8 * i) for i in range(20000)])
    assert d[:, 0].min() >= 0.3 and d[:, 0].max() <= 1.0 and np.abs(d[:, 1]).max() <= 0.6
    for col, m in ((0, mq), (1, mc)):
        assert abs(d[:, col].mean() - m) < 4 * d[:, col].std() / np.sqrt(len(d))
    assert qc(1, 2000000008) == qc(1, 2000000008) != qc(1, 2000000016) and qc(1, 2000000008) != qc(2, 2000000008)


def test_build_spectrum():
    """S = 2 sig2 / (T v) on the bins kept, ln S interpolated across a notch, the wall below 20 Hz up to its cap, flat
    above the band"""
    notches = ((59.5, 60.5),)
    f = NU / 8.0
    true = 1e-46 * (f / 100.0) ** -1.5
    v = strain.bin_variance(sp.FS, 8.0, NU, band=(16.0, 600.0), roll=(4.0, 100.0), notches=notches)
    notched = strain.response(f, notches=notches) < 1
    sig2 = np.where(notched, 0.0, 4.0 * true * v)
    fk, S = sp.build_spectrum(f, sig2, notched, notches)
    on = (fk >= 20) & (fk <= 512)
    inside = ~notched
    assert np.allclose(S[on][inside], true[inside], rtol=1e-12)
    assert np.allclose(S[on][notched], 1e-46 * (f[notched] / 100.0) ** -1.5, rtol=1e-3)     # power law: exact in log
    s20 = S[fk == 20.0][0]
    assert S[fk == 10.0][0] == pytest.approx(s20 * 2.0 ** 20, rel=1e-9)
    assert S[fk == 5.0][0] == pytest.approx(s20 * 1e8, rel=1e-9) and S[0] == pytest.approx(s20 * 1e8, rel=1e-9)
    assert np.all(S[fk > 512] == S[fk == 512.0][0])


def test_noise_has_its_spectrum(spectrum):
    _, S, _ = spectrum
    n = int(64 * 4 * sp.FS)
    x = sp.noise(S, n, 5)
    P = strain.periodograms(x, sp.FS, 8.0, 'hann')                 # E P = T S / 2 where S is smooth
    f = np.arange(P.shape[1]) / 8.0
    for lo, hi in ((25.0, 50.0), (100.0, 200.0), (300.0, 500.0)):
        b = (f >= lo) & (f < hi) & (strain.response(f, notches=sim.NOTCHES) == 1)
        r = P[:, b] / (4.0 * S(f[b]))
        assert abs(r.mean() - 1.0) < 4.0 * 1.5 / np.sqrt(r.size)


# ---------------------------------------------------------------------------------------------------------------------
# control i


@pytest.fixture(scope='module')
def control_i(spectrum):
    _, S, notches = spectrum
    breaks = {'correct': {}, 'delay': dict(delay_sign=-1.0), 'phase': dict(phase_sign=-1.0), 'reverse': dict(reverse=True)}
    return {known: sp.control_case(S, notches, 4.0, breaks, seed=2, psd_known=known, cond=SMALL, m=(30.0, 30.0),
                                   ext=EXT) for known in (True, False)}


@pytest.mark.parametrize('known, tol', [(True, 2e-3), (False, 0.08)])
def test_control_i_strain_injection_is_the_segment_signal(control_i, known, tol):
    """known spectrum: exact.  Estimated (median of N = 8 here): each bin carries the gain sqrt(s / (c P)), whose square
    has mean 1 but whose mean is below 1 (Jensen), a few percent at N = 8 (about 1 percent at N = 32)"""
    r = control_i[known]['cases']['correct']
    for I in 'HL':
        c = r['own_' + I]
        print('\n%s %s: opt %.3f sig %.4f snr0 %.3f lag %d' % ('known' if known else 'estimated', I, c['opt'], c['sig'],
                                                             c['snr0'], c['lag']))
        assert abs(c['sig'] / c['opt'] - 1.0) < tol and c['lag'] == 0
        assert abs(c['snr0'] - c['opt']) < 4.0
        assert abs(r['prev_' + I]['sig']) < 0.01 * c['opt']


@pytest.mark.parametrize('known', [True, False])
def test_control_i_breaks(control_i, known):
    r = control_i[known]
    c = r['cases']
    assert abs(c['delay']['own_H']['lag'] - r['lag_delay_flipped_H']) <= 1 and abs(r['lag_delay_flipped_H']) > 40
    assert c['delay']['own_H']['sig'] / c['delay']['own_H']['opt'] < 0.5
    assert c['delay']['own_L']['sig'] / c['delay']['own_L']['opt'] > 0.95          # tau_L = 0: L does not move
    assert min(c['phase']['own_' + I]['sig'] / c['phase']['own_' + I]['opt'] for I in 'HL') < 0.8
    for I in 'HL':
        assert abs(c['reverse']['own_' + I]['sig']) < 0.2 * c['reverse']['own_' + I]['opt']


# ---------------------------------------------------------------------------------------------------------------------
# control ii


@pytest.fixture(scope='module')
def control_ii(spectrum):
    _, S, notches = spectrum
    breaks = {'correct': {}, 'cut': dict(start=True), 'circular': dict(circular=True)}
    return {known: sp.control_case(S, notches, 0.5, breaks, seed=3, psd_known=known, cond=SMALL, m=(15.0, 15.0),
                                   ext=EXT) for known in (True, False)}


@pytest.mark.parametrize('known, tol', [(True, 0.01), (False, 0.08)])
def test_control_ii_loss_appears_in_the_previous_segment(control_ii, known, tol):
    r = control_ii[known]
    c = r['cases']['correct']
    for I in 'HL':
        p = r['predicted'][I]
        own, prev = c['own_' + I]['sig'] / c['own_' + I]['opt'], c['prev_' + I]['sig'] / c['prev_' + I]['opt']
        print('\n%s %s: own %.4f (predicted %.4f), previous %.4f (predicted %.4f), sum %.4f' % (
            'known' if known else 'estimated', I, own, p['own'], prev, p['prev'], own + prev))
        assert p['prev'] > 0.1 and p['own'] < 0.9
        assert abs(own - p['own']) < tol and abs(prev - p['prev']) < tol and abs(own + prev - 1.0) < 2 * tol


@pytest.mark.parametrize('known', [True, False])
def test_control_ii_breaks(control_ii, known):
    r = control_ii[known]
    cut, circ = r['cases']['cut'], r['cases']['circular']
    for I in 'HL':
        assert abs(cut['prev_' + I]['sig']) < 0.01 * cut['prev_' + I]['opt']           # nothing in the previous one
        assert abs(circ['own_' + I]['sig'] / circ['own_' + I]['opt'] - 1.0) < 1e-5      # the model's injection: no loss
        assert abs(circ['prev_' + I]['sig']) < 1e-9 * circ['own_' + I]['opt']


# ---------------------------------------------------------------------------------------------------------------------
# end to end, control iii


@pytest.fixture(scope='module')
def run(spectrum, tmp_path_factory):
    """two stretches whitened with the true spectrum (exact), one with estimated spectra (the O3b settings at N = 8), one
    noise-only without the correction factor"""
    root = str(tmp_path_factory.mktemp('simstrain'))
    quiet = dict(workers=1, log=lambda *a, **k: None)
    cfg = sp.make_config(spectrum[0], root, seed=3, nbar=0.5, epochs=3, cond=SMALL, psd_known=True)
    done = sp.generate(cfg, [0, 1], **quiet)
    again = sp.generate(cfg, [0, 1], **quiet)
    est = sp.make_config(spectrum[0], root, seed=3, nbar=0.5, epochs=3, tag='est', cond=SMALL)
    sp.generate(est, [0], **quiet)
    nc = sp.make_config(spectrum[0], root, seed=3, nbar=0.5, epochs=3, arms=('N',), tag='nc', no_correct=True,
                        cond=SMALL)
    sp.generate(nc, [0], **quiet)
    return root, cfg, done, again, sp.load_truth(root), sp.load_truth(root, 'est')


def test_end_to_end_files(run):
    root, cfg, done, again, tr, _ = run
    assert all(not r['skipped'] for r in done) and all(r['skipped'] for r in again)
    n = 2 * 3 * 9
    assert tr['gps'].size == n and np.unique(tr['gps']).size == n
    for arm in sp.ARMS:
        files = sorted(f for f in os.listdir(sp.folder(root, arm)) if f.startswith('epoch_'))
        assert len(files) == 6 and os.path.exists(os.path.join(sp.folder(root, arm), 'manifest.txt'))
        eps = [strain.read_epoch(os.path.join(sp.folder(root, arm), f)) for f in files]
        assert np.array_equal(np.concatenate([e['gps'] for e in eps]), tr['gps'])
        assert all(e['arm'] == arm and e['settings_hash'] == cfg['hashes'][arm][0] for e in eps)
    # the same noise in every arm: A - N is the planted signal alone, zero where there is no merger
    N = [strain.read_epoch(os.path.join(sp.folder(root, 'N'), f)) for f in sorted(os.listdir(sp.folder(root, 'N')))
         if f.startswith('epoch_')]
    A = [strain.read_epoch(os.path.join(sp.folder(root, 'A'), f)) for f in sorted(os.listdir(sp.folder(root, 'A')))
         if f.startswith('epoch_')]
    dA = np.concatenate([a['d_H'].astype(complex) - b['d_H'].astype(complex) for a, b in zip(A, N)])
    assert np.all(dA[~tr['present']] == 0) and np.all(np.abs(dA[tr['present']]).max(axis=1) > 0)


def test_end_to_end_truth(run):
    _, _, _, _, tr, est = run
    m = tr['present']
    assert 0.3 < m.mean() < 0.7
    for key in ('present', 'm1', 'z', 'q', 'chi', 'theta', 't_H', 't_pk0_A', 't_pk0_B'):     # the same mergers
        assert np.array_equal(tr[key][:est['gps'].size], est[key], equal_nan=True)
    assert np.allclose(tr['mdet_A'][m], 2 * tr['m1'][m] * (1 + tr['z'][m]))
    assert np.allclose(tr['mdet_B'][m], tr['m1'][m] * (1 + tr['z'][m]) * (1 + tr['q'][m]))
    for a in 'AB':
        for I in 'HL':
            assert np.allclose(tr['sig_%s_%s' % (I, a)][m], tr['opt_%s_%s' % (I, a)][m], rtol=1e-5)
            # the segment-level optimal SNR (bins, notched ones out) against that of the strain signal, true spectrum
            assert np.allclose(tr['opt_%s_%s' % (I, a)][m], tr['opt_true_%s_%s' % (I, a)][m], rtol=0.01)


@pytest.mark.parametrize('which, tol', [('known', 2e-3), ('estimated', 0.15)])
def test_control_iii_C_agrees_with_A_inside_the_segment(run, which, tol):
    """known spectrum: C is A to the ringing of the band edges for every merger inside its segment, and the matched
    filter of the others recovers the fraction of their whitened energy inside.  Estimated (N = 8): the median ratio,
    within the Jensen loss of the per-bin gains at N = 8"""
    tr = run[4] if which == 'known' else run[5]
    for x, ref in (('C', 'A'), ('D', 'B')):
        inside = tr['present'] & (tr['frac_own_H_' + ref] >= 0.999) & (tr['frac_own_L_' + ref] >= 0.999)
        ratio = np.hypot(tr['sig_H_' + x], tr['sig_L_' + x])[inside] / np.hypot(tr['sig_H_' + ref],
                                                                               tr['sig_L_' + ref])[inside]
        lag = np.concatenate([tr['lag_H_' + x][inside], tr['lag_L_' + x][inside]])
        print('\n%s %s/%s inside: %d mergers, ratio %s, lags %s' % (which, x, ref, inside.sum(),
                                                                    np.round(np.sort(ratio), 4), np.unique(lag)))
        assert inside.sum() >= 5
        if which == 'known':
            assert np.all(np.abs(ratio - 1) < tol) and np.all(lag == 0)
            part = tr['present'] & ((tr['frac_own_H_' + ref] < 0.99) | (tr['frac_own_L_' + ref] < 0.99))
            for I in 'HL':
                r = tr['sig_%s_%s' % (I, x)][part] / tr['opt_%s_%s' % (I, ref)][part]
                print('%s partial: measured %s, predicted %s' % (I, np.round(r, 4),
                                                                 np.round(tr['frac_own_%s_%s' % (I, ref)][part], 4)))
                assert np.all(np.abs(r - tr['frac_own_%s_%s' % (I, ref)][part]) < 0.02)
        else:
            assert abs(np.median(ratio) - 1) < tol


def test_control_iv_break_is_the_missing_correction(run):
    """CPU proxy of control iv: the whitened power of the noise-only segments before step 5 is 1 with the correction
    and c(8) without"""
    root = run[0]
    p = {}
    for tag in ('est', 'nc'):
        d = sp.folder(root, 'N', tag)
        eps = [strain.read_epoch(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith('epoch_')]
        keep = ~eps[0]['notched']
        p[tag] = np.mean([np.mean(np.abs(strain.unequalized(e)['H'][:, keep]) ** 2 / e['sig2_ref_H'][keep])
                          for e in eps])
    c = strain.correction(8, 'median')
    print('\nwhitened power: with the correction %.4f, without %.4f (c = %.4f)' % (p['est'], p['nc'], c))
    assert abs(p['est'] - 1.0) < 0.05 and abs(p['nc'] / p['est'] - c) < 0.05


# ---------------------------------------------------------------------------------------------------------------------
# the gain of the conditioning (strain_plant.gain_numbers): quadrature against direct draws, and a strain-level injection
# against the segment-level signal times the gain


@pytest.mark.parametrize('method, N, n_ref', [('median', 32, 66), ('median', 32, 33), ('mean', 32, 66),
                                              ('median', 8, 18)])
def test_gain_quadrature_against_monte_carlo(method, N, n_ref):
    q = sp.gain_numbers(N, n_ref, method, ngrid=200000)
    mc = sp.gain_monte_carlo(N, n_ref, method, draws=200000, seed=7)
    print('\n%s N %d n_ref %d: kappa %.5f (MC %.5f +- %.5f), kappa_eff %.5f (MC %.5f +- %.5f), E[gamma^2] MC %.4f'
          % (method, N, n_ref, q['kappa'], mc['kappa'], mc['kappa_se'], q['kappa_eff'], mc['kappa_eff'],
             mc['kappa_eff_se'], mc['E_gamma2']))
    assert abs(q['c'] / q['c_strain'] - 1) < 1e-6 and abs(q['checks']['mean_P'] - 1) < 1e-6
    assert abs(q['kappa'] - mc['kappa']) < 4 * mc['kappa_se']
    assert abs(q['kappa_eff'] - mc['kappa_eff']) < 4 * mc['kappa_eff_se']
    assert abs(mc['E_gamma2'] - 1) < 4 * mc['E_gamma2_se'] + 1e-4


def test_gain_mean_closed_form():
    from scipy.special import gammaln
    q = sp.gain_numbers(32, 66, 'mean', ngrid=200000)
    kP = np.exp(gammaln(31.5) - gammaln(32)) * np.sqrt(32) / np.sqrt(32 / 31.0)
    ks = np.exp(gammaln(66.5) - gammaln(66)) / np.sqrt(66)
    assert q['kappa_P'] == pytest.approx(kP, rel=1e-7) and q['kappa_s'] == pytest.approx(ks, rel=1e-7)


def test_strain_injection_is_the_segment_signal_times_kappa_eff(spectrum):
    """estimated spectra (median, N = 8): mergers inside a segment of epochs 1, 3, 5 of 7 (their own blocks and their
    neighbours' clean, so d_inj - d_noise is the conditioned signal alone, no noise): the matched-filter projection on the
    segment-level signal s, over <s, s>, is kappa_eff(8, 18) within its scatter, and not kappa"""
    _, S, notches = spectrum
    cond = dict(sp.COND, notches=notches, N=8, seed=5)
    K, n1 = 7, 9
    n = int(round((K * n1 * sp.T + 2 * sp.PAD) * sp.FS))
    wf = sp.waveform(10.0, 10.0, 0.0)
    rng = np.random.default_rng(11)
    ratios = []
    for st in range(5):
        x0 = {I: sp.noise(S, n, [31, st, j]) for j, I in enumerate('HL')}
        ep0 = strain.condition(x0['H'], x0['L'], sp.FS, 0.0, **cond)
        x = {I: v.copy() for I, v in x0.items()}
        planted = []
        for e in (1, 3, 5):
            row = int(rng.integers(n1))
            ext = dict(theta=np.arccos(rng.uniform(-1, 1)), phi=rng.uniform(0, 2 * np.pi), psi=rng.uniform(0, np.pi),
                       cos_iota=rng.uniform(-1, 1), phase=rng.uniform(0, 2 * np.pi), t=6.5)
            t_a = ext['t'] - sp.responses(ext)['H'][1] - wf['t_pk0']
            s = sp.model_signal(10.0, 10.0, 0.0, 1.0, ext, t_a, ep0[e]['nu'])
            y = 5.0 / np.sqrt(sum(2 * np.sum(np.abs(s[I]) ** 2 / ep0[e]['sig2']) for I in 'HL'))
            sp.inject(x, 0.0, wf, ext, y, sp.PAD + (e * n1 + row) * sp.T + t_a)
            planted.append((e, row, {I: y * v for I, v in s.items()}))
        eps = strain.condition(x['H'], x['L'], sp.FS, 0.0, **cond)
        for e, row, s in planted:
            w = 1.0 / eps[e]['sig2']
            num = sum(np.sum(np.conj(s[I]) * (eps[e]['d_' + I][row].astype(complex) -
                                              ep0[e]['d_' + I][row].astype(complex)) * w).real for I in 'HL')
            ratios.append(num / sum(np.sum(np.abs(s[I]) ** 2 * w) for I in 'HL'))
    r = np.array(ratios)
    q = sp.gain_numbers(8, 18, 'median', ngrid=200000)
    se = r.std(ddof=1) / np.sqrt(r.size)
    print('\n%d injections: ratio %.4f +- %.4f; kappa_eff %.4f, kappa %.4f; ratio / kappa_eff = %.4f +- %.4f'
          % (r.size, r.mean(), se, q['kappa_eff'], q['kappa'], r.mean() / q['kappa_eff'], se / q['kappa_eff']))
    assert abs(r.mean() - q['kappa_eff']) < 4 * se
    assert abs(r.mean() - q['kappa']) > 4 * se


def test_strain_injection_on_own_curves_carries_the_signal_gain(spectrum):
    """the configuration of the search: each detector on its own curve (strain.own_curves, no step 5).  Mergers inside a
    segment of epochs 1, 3, 5 of 7, conditioned with estimated spectra (median, N = 8), so that d_inj - d_noise is the
    conditioned signal alone: its matched-filter projection on the segment-level signal s with the weights 1 / sig2_I of
    the own curves, sum_I Re <s_I, d_I> / sum_I kappa_I <s_I, s_I>, is 1 within its scatter with kappa_I =
    strain.signal_gain of the epoch (the model with the gain), and kappa (not 1) without the gain: the model without it
    fails by the predicted amount"""
    _, S, notches = spectrum
    cond = dict(sp.COND, notches=notches, N=8, seed=5)
    K, n1 = 7, 9
    n = int(round((K * n1 * sp.T + 2 * sp.PAD) * sp.FS))
    wf = sp.waveform(10.0, 10.0, 0.0)
    rng = np.random.default_rng(13)
    with_gain, without = [], []
    for st in range(5):
        x0 = {I: sp.noise(S, n, [37, st, j]) for j, I in enumerate('HL')}
        ep0 = strain.condition(x0['H'], x0['L'], sp.FS, 0.0, **cond)
        x = {I: v.copy() for I, v in x0.items()}
        planted = []
        for e in (1, 3, 5):
            row = int(rng.integers(n1))
            ext = dict(theta=np.arccos(rng.uniform(-1, 1)), phi=rng.uniform(0, 2 * np.pi), psi=rng.uniform(0, np.pi),
                       cos_iota=rng.uniform(-1, 1), phase=rng.uniform(0, 2 * np.pi), t=6.5)
            t_a = ext['t'] - sp.responses(ext)['H'][1] - wf['t_pk0']
            s = sp.model_signal(10.0, 10.0, 0.0, 1.0, ext, t_a, ep0[e]['nu'])
            curves = strain.own_sig2(ep0[e])
            y = 5.0 / np.sqrt(sum(2 * np.sum(np.abs(s[I]) ** 2 / curves[I]) for I in 'HL'))
            sp.inject(x, 0.0, wf, ext, y, sp.PAD + (e * n1 + row) * sp.T + t_a)
            planted.append((e, row, {I: y * v for I, v in s.items()}))
        eps = strain.condition(x['H'], x['L'], sp.FS, 0.0, **cond)
        for e, row, s in planted:
            own, own0 = strain.own_curves(eps[e]), strain.own_curves(ep0[e])
            kappa = strain.signal_gain(eps[e])
            assert kappa == strain.signal_gain(ep0[e])
            num = den_g = den = 0.0
            for I in 'HL':
                w = 1.0 / own[I][1]
                num += np.sum(np.conj(s[I]) * (own[I][0][row] - own0[I][0][row]) * w).real
                den += np.sum(np.abs(s[I]) ** 2 * w)
                den_g += kappa[I] * np.sum(np.abs(s[I]) ** 2 * w)
            with_gain.append(num / den_g)
            without.append(num / den)
    r, r0 = np.array(with_gain), np.array(without)
    se, se0 = r.std(ddof=1) / np.sqrt(r.size), r0.std(ddof=1) / np.sqrt(r0.size)
    k = strain.normalization_gain(8) * strain.hand_over_gain(18)
    print('\nown curves, %d injections: with the gain %.4f +- %.4f; without %.4f +- %.4f (kappa %.4f)'
          % (r.size, r.mean(), se, r0.mean(), se0, k))
    assert abs(r.mean() - 1.0) < 4 * se
    assert abs(r0.mean() - 1.0) > 4 * se0 and abs(r0.mean() - k) < 4 * se0
