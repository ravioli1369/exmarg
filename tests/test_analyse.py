"""the analysis of the conditioned week without the tables (scripts/real_data/analyse.py, summarize.py, and plant_fit.py
on real-data directories): the variants of a synthetic epoch file (strain.write_epoch), the selection and k across
--part splits and --variants subsets with a stub in place of pe_redshift, plant_fit on directories without a known
truth, and the guard that keeps the recorded week closed until the controls pass"""
import glob
import importlib.util
import json
import os
import re
import shutil

import numpy as np
import pytest

from exmarg import strain
from exmarg.detectors import S
from exmarg.search import RHO_EDGES, controls, mass_bank, mixture
from test_mixture import EDGES, LAWS, Toy, law
from test_plant import make_model, synthetic_B, window_stub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name, folder):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts', folder, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyse = _script('analyse', 'real_data')
summarize = _script('summarize', 'real_data')
windows_fit = _script('windows_fit', 'validate')
plant_fit = summarize.plant_fit
plant = analyse.plant

T, FMIN, FMAX, NSEG = 8.0, 20.0, 512.0, 33
NOTCH = slice(1200, 1210)


def make_epoch(gps_epoch, seed, loud_H=(), loud_L=(), event=(), usable=True):
    """an epoch dict of strain.condition's keys: Gaussian noise on 1.7 times the H design curve (not pp.simulate's) in
    both detectors, ten notched bins"""
    nu = np.arange(int(np.ceil(FMIN * T)), int(np.floor(FMAX * T)) + 1)
    f = nu / T
    sig2 = 1.7 * T * S('H', f) / 2.0
    notched = np.zeros(nu.size, bool)
    notched[NOTCH] = True
    sig2[notched] = 1e10 * sig2.max()
    rng = np.random.default_rng(seed)
    d = {I: (np.sqrt(sig2 / 2) * (rng.standard_normal((NSEG, nu.size)) + 1j * rng.standard_normal((NSEG, nu.size))))
         .astype(np.complex64) for I in 'HL'}
    flags = {}
    for name, rows in (('loud_H', loud_H), ('loud_L', loud_L), ('event', event)):
        flags[name] = np.isin(np.arange(NSEG), rows)
    ep = dict(d_H=d['H'], d_L=d['L'], f=f, nu=nu, sig2=sig2, T=T, fmin=FMIN, fmax=FMAX, notched=notched,
              gps=gps_epoch + T * np.arange(NSEG), gps_epoch=float(gps_epoch), usable=bool(usable), equal_noise=True,
              dq_H=np.full(NSEG, 127), dq_L=np.full(NSEG, 127), inj_H=np.full(NSEG, 31), inj_L=np.full(NSEG, 31),
              monitor_H=1 + 0.01 * rng.standard_normal((NSEG, 5)), monitor_L=1 + 0.01 * rng.standard_normal((NSEG, 5)),
              u_ratio_60=0.8, psd='known', **flags)                    # drawn on known curves: signal gain 1
    ep['analysable'] = ~ep['loud_H'] & ~ep['loud_L'] & ep['usable']
    return ep


EPOCHS = [dict(gps=1266933768, loud_L=(3, 30)), dict(gps=1266934032, usable=False),
          dict(gps=1266934296, event=(8, 9), loud_H=(20,)), dict(gps=1266934560)]


def write_week(folder):
    """four epoch files and their manifest (one unusable, one with event and loud segments), in reverse order on disk"""
    os.makedirs(folder, exist_ok=True)
    lines = ['# synthetic week\n']
    for i, e in enumerate(EPOCHS[::-1]):
        ep = make_epoch(e['gps'], 100 + i, e.get('loud_H', ()), e.get('loud_L', ()), e.get('event', ()),
                        e.get('usable', True))
        name = 'epoch_%d.npz' % e['gps']
        strain.write_epoch(os.path.join(folder, name), ep)
        lines.append('%s %d %d 127 127 %d 0 0 33 33 %d 0.8\n' % (name, e['gps'], NSEG, ep['event'].sum(),
                                                                 int(ep['usable'])))
    with open(os.path.join(folder, 'manifest.txt'), 'w') as fh:
        fh.writelines(lines)
    return folder


@pytest.fixture(scope='module')
def week(tmp_path_factory):
    return write_week(str(tmp_path_factory.mktemp('week') / 'segments'))


def series(seg, I):
    """the band-limited real series of each segment on the n_time = 8 nu_max samples of the matched filter"""
    nu = np.asarray(seg['nu'])
    n = 8 * int(nu.max())
    full = np.zeros((seg['d_' + I].shape[0], n), complex)
    full[:, nu] = seg['d_' + I]
    return np.real(np.fft.ifft(full, axis=1)), n


# ----------------------------------------------------------------------------------------------- the variants


def test_variants_built(week):
    """from an epoch file read back: the rows are the analysable segments without an event (none of an unusable epoch);
    zero is the recorded data; pr rotates every bin by the phases of controls.phase_randomize with the seed [seed, GPS
    of the epoch, 1] (power and cross spectrum kept); rev is the conjugate; the slides leave H as it is and move the
    series of L by exactly 2048 and -7168 matched-filter samples; inj is slide+0.5 plus a signal only in the segments
    with an injection, which match plant.mergers keyed by the GPS second, in the population (z >= 0.99119, M_det in
    [20, 400]); the recorded optimal SNR equals sqrt(sum_I 2 |h_I|^2 / sig2) of that signal and the matched filter's
    sum_I 2 max_t |z_I(t)|^2 / u of the injected template on it (2 percent: arrival times between the samples)"""
    ep = strain.read_epoch(os.path.join(week, 'epoch_1266934296.npz'))
    s = plant.Sampler(plant.population(analyse.ZMIN), plant.MDET)
    params = dict(seed=7, nbar_inj=0.5)
    var, rows, inj = analyse.variants(params, ep, analyse.VARIANTS, s)
    expect = np.setdiff1d(np.arange(NSEG), [8, 9, 20])
    assert np.array_equal(rows, expect)
    zero = var['zero']
    for I in 'HL':
        assert np.array_equal(zero['d_' + I], np.asarray(ep['d_' + I])[expect].astype(complex))
    pr = controls.phase_randomize(zero, [7, 1266934296, 1])
    assert np.array_equal(var['pr']['d_H'], pr['d_H']) and np.array_equal(var['pr']['d_L'], pr['d_L'])
    assert np.allclose(np.abs(var['pr']['d_L']), np.abs(zero['d_L']), rtol=1e-12, atol=0)
    assert np.allclose(var['pr']['d_H'] * np.conj(var['pr']['d_L']), zero['d_H'] * np.conj(zero['d_L']), rtol=1e-10)
    other = analyse.variants(params, strain.read_epoch(os.path.join(week, 'epoch_1266934560.npz')), ('zero', 'pr'), s)
    assert not np.allclose(np.angle(other[0]['pr']['d_H'][0] / other[0]['zero']['d_H'][0]),
                           np.angle(var['pr']['d_H'][0] / zero['d_H'][0]))
    assert np.array_equal(var['rev']['d_H'], np.conj(zero['d_H'])) and np.array_equal(var['rev']['d_L'],
                                                                                       np.conj(zero['d_L']))
    for v, samples in (('slide+0.5', 2048), ('slide-1.75', -7168)):
        assert np.array_equal(var[v]['d_H'], zero['d_H'])
        x, n = series(zero, 'L')
        xs, _ = series(var[v], 'L')
        assert np.max(np.abs(xs - np.roll(x, samples, axis=1))) < 1e-12 * np.abs(x).max(), v
        assert analyse.whole_samples(zero, analyse.SHIFTS[v]) == samples
    h = {I: var['inj']['d_' + I] - var['slide+0.5']['d_' + I] for I in 'HL'}
    on = inj['merger']
    gps = [int(g) for g in np.asarray(ep['gps'])[rows]]
    present, m1, z, y = plant.mergers(dict(seed=7, nbar=0.5), gps, s)
    assert np.array_equal(on, present) and 3 <= on.sum() <= rows.size - 3, on
    assert np.array_equal(inj['m1'][on], m1[on]) and np.array_equal(inj['z'][on], z[on]) and np.all(inj['y'][~on] == 0)
    assert np.all(z[on] >= analyse.ZMIN) and np.all((2 * m1 * (1 + z) >= 20) & (2 * m1 * (1 + z) <= 400))
    for I in 'HL':
        assert np.all(h[I][~on] == 0) and np.all(np.abs(h[I][on]).max(axis=1) > 0)
    sig2 = ep['sig2']
    direct = np.sqrt(sum(2 * np.sum(np.abs(h[I]) ** 2 / sig2, axis=1) for I in 'HL'))
    assert np.allclose(inj['snr'], direct, rtol=1e-6, atol=0), (inj['snr'], direct)
    seg = dict(zero, sig2_H=sig2, sig2_L=sig2)
    lines = []
    for j in np.flatnonzero(on):
        bank = mass_bank(np.array([2 * m1[j] * (1 + z[j])]), seg, 'cpu')
        traj = bank.trajectories({I: h[I][j] for I in 'HL'}, {'H': sig2, 'L': sig2})
        mf = np.sqrt(sum(2 * np.exp(2 * traj['lq' + I].astype(float)).max() / bank.u[0] for I in 'HL'))
        lines.append('injection %d: snr %.4f, matched filter %.4f' % (j, inj['snr'][j], mf))
        assert abs(mf / inj['snr'][j] - 1) < 0.02, lines[-1]
    print('\n' + '\n'.join(lines))
    none = analyse.variants(dict(seed=7, nbar_inj=0.0), ep, ('inj',), s)[2]
    assert not none['merger'].any()
    unusable = strain.read_epoch(os.path.join(week, 'epoch_1266934032.npz'))
    assert analyse.variants(params, unusable, ('zero',), s)[1].size == 0


# ----------------------------------------------------------------------------------------------- the driver


class Stub:
    """pe_redshift in place: arrays of its shapes for the rows asked, functions of those rows' data and of seed + row;
    records the calls"""

    def __init__(self):
        self.calls, self.glitches = [], []

    def factory(self, seg0):
        return self.pe

    def pe(self, seg, seed, rows, glitch=True):
        rows = np.asarray(rows)
        self.calls.append((seed, rows.copy()))
        self.glitches.append(glitch)
        n = rows.size
        g = plant.GRID
        m1, z = np.geomspace(*g['m1_range'], g['nm1']), np.linspace(0.02, g['zmax'], g['nz'])
        q = {I: np.mean(np.abs(seg['d_' + I][rows]) ** 2 / seg['sig2'], axis=1) - 1.0 for I in 'HL'}
        a = 0.1 * (q['H'] + q['L']) + 1e-3 * np.array([(seed + int(r)) % 101 for r in rows])
        Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
        shape = np.where((Md >= 20) & (Md <= 400), 0.1 * np.log1p(z)[None, :] - m1[:, None] / 500, -np.inf)
        res = dict(m1=m1, z=z, rows=rows, ln_lambda=a[:, None, None] + shape[None])
        windows = isinstance(glitch, dict)                       # the window model: its outputs, functions of it
        for j, key in enumerate(('ln_B', 'ln_lambda_H', 'ln_lambda_L') if windows else plant.KEYS[1:]):
            res[key] = res['ln_lambda'] + 0.1 * (j + 1)
        nr, nm = RHO_EDGES.size - 1, plant.MASSES.size
        for I in 'HL':
            res['ln_glitch_' + I] = q[I][:, None, None] * np.linspace(0, 1, nm)[None, :, None] \
                * np.linspace(-1, 2, nr)[None, None, :]
            res['wmax_' + I] = (np.abs(seg['d_' + I][rows]) ** 2 / seg['sig2']).max(axis=1)[:, None] \
                * np.linspace(0.5, 1, nm)[None]
            if windows:
                res['ln_c_' + I] = q[I] * glitch[I]['q'][-1]
                res['state_' + I] = np.tile(np.asarray(glitch[I]['f']), (n, 1))
        if windows:
            res.update(rho_edges=RHO_EDGES.copy(), ln_B=res['ln_B'] - 10.0 * glitch['H']['q'][0])
        else:
            res.update(rho_edges=RHO_EDGES.copy(), glitch_pi_k=np.full(nr, 1.0 / nr), glitch_window=0.1)
        return res


def drive(week, out, *extra, stub=None):
    stub = stub or Stub()
    assert analyse.main(['nostore', 'cpu', week, '--out', str(out), '--seed', '7'] + list(extra),
                        factory=stub.factory) == 0
    return stub


def test_driver_parts_and_variants(week, tmp_path):
    """analyse.main with the stub: one pe call per epoch with a row, holding every requested variant at the fixed rows
    VARIANTS.index(v) 33 + row and the seed stream(seed, GPS of the epoch, 'pe'); the unusable epoch, the event and loud
    segments are left out; k runs over the selected segments of the week in time order (the manifest is in reverse
    order on disk) and gps grows with it; --part 0/2 + 1/2 and 0/1 (epochs per file 1) give the same arrays bit for bit
    in every variant, as does a run of --variants pr,inj then the rest; a second run calls nothing; the meta marks the
    truth: control = variant, nbar null for zero, 0 for the controls, --nbar-inj for inj, glitch_truth false; and
    selection.txt records the k of each epoch"""
    one, two, three = tmp_path / 'one', tmp_path / 'two', tmp_path / 'three'
    s0 = drive(week, one, '--part', '0/2', '--epochs-per-file', '1')
    s1 = drive(week, one, '--part', '1/2', '--epochs-per-file', '1')
    calls = s0.calls + s1.calls
    assert [c[0] for c in s0.calls] == [plant.stream(7, 1266933768, 'pe')]             # part 0/2: the first two epochs
    gps = [1266933768, 1266934296, 1266934560]
    sel = {1266933768: np.setdiff1d(np.arange(NSEG), [3, 30]), 1266934296: np.setdiff1d(np.arange(NSEG), [8, 9, 20]),
           1266934560: np.arange(NSEG)}
    assert [c[0] for c in calls] == [plant.stream(7, g, 'pe') for g in gps]
    for (seed, rows), g in zip(calls, gps):
        assert np.array_equal(rows, np.concatenate([i * NSEG + sel[g] for i in range(6)])), g
    drive(week, two, '--epochs-per-file', '1')
    drive(week, three, '--epochs-per-file', '1', '--variants', 'pr,inj')
    drive(week, three, '--epochs-per-file', '1', '--variants', 'zero,rev,slide+0.5,slide-1.75,inj')
    again = drive(week, two, '--epochs-per-file', '1')
    assert again.calls == []
    n_sel = sum(v.size for v in sel.values())
    want_gps = np.concatenate([g + T * sel[g] for g in gps])
    for v in analyse.VARIANTS:
        cfg, _, a, runs, cells = plant_fit.load(str(one / v))
        b, c = plant_fit.load(str(two / v))[2], plant_fit.load(str(three / v))[2]
        assert np.array_equal(a['k'], np.arange(n_sel)) and np.array_equal(a['gps'], want_gps), v
        assert np.all(np.diff(a['gps']) > 0) and not a['event'].any() and a['analysable'].all()
        for key in a:
            assert np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == 'f'), (v, key)
            assert np.array_equal(a[key], c[key], equal_nan=a[key].dtype.kind == 'f'), (v, key)
        assert cfg['control'] == v and cfg['glitch_truth'] is False and not a['glitch_H'].any()
        assert cfg['nbar'] == (None if v == 'zero' else 0.3 if v == 'inj' else 0.0), (v, cfg['nbar'])
        tr, kind = plant_fit.truth(cfg)
        assert kind == ('recorded' if v == 'zero' else 'plant' if v == 'inj' else 'control')
        assert a['merger'].any() == (v == 'inj')
    with open(one / 'selection.txt') as fh:
        lines = fh.read().split()
    assert lines == ['epoch_1266933768.npz', '1266933768', '33', '31', '0', 'epoch_1266934032.npz', '1266934032', '33',
                     '0', '31', 'epoch_1266934296.npz', '1266934296', '33', '30', '31', 'epoch_1266934560.npz',
                     '1266934560', '33', '33', '61']


def with_hashes(week, folder, hashes):
    """a copy of the week's folder in which the epoch files (in time order) carry settings_hash = hashes[i] (None: left
    without)"""
    import shutil
    shutil.copytree(week, folder)
    for name, h in zip(sorted(f for f in os.listdir(folder) if f.startswith('epoch_')), hashes):
        ep = strain.read_epoch(os.path.join(folder, name))
        if h is not None:
            ep['settings_hash'] = h
        strain.write_epoch(os.path.join(folder, name), ep)
    return str(folder)


def test_settings_hash_one_across_the_epochs(week, tmp_path):
    """selection (so run) reads the settings hash of every epoch: none anywhere (a folder conditioned before it was
    stored) and one everywhere are accepted, with the same selection; hashed with unhashed, or two hashes, stop with a
    message naming the epochs"""
    base = analyse.selection(week)
    same = with_hashes(week, tmp_path / 'same', ['a' * 64] * 4)
    assert analyse.selection(same) == base
    with pytest.raises(SystemExit, match='epoch_1266934296.npz') as err:
        analyse.selection(with_hashes(week, tmp_path / 'mixed', ['a' * 64, 'a' * 64, None, 'a' * 64]))
    assert 'settings' in str(err.value)
    with pytest.raises(SystemExit, match='settings'):
        analyse.selection(with_hashes(week, tmp_path / 'two', ['a' * 64, 'a' * 64, 'b' * 64, 'b' * 64]))
    a = analyse.parse(['x', 'cpu', str(tmp_path / 'mixed'), '--out', str(tmp_path / 'o')])[1]
    with pytest.raises(SystemExit, match='settings'):
        analyse.run(a, Stub().factory)


def test_manifest_hash_is_that_of_the_epoch_rows(week, tmp_path):
    """the manifest hash of a campaign does not change when condition.py is run again and writes another timestamp in
    the header (the earlier outputs would otherwise be refused), but does when a row changes"""
    import shutil
    a = analyse.parse(['x', 'cpu', week, '--out', 'x'])[1]
    digest = analyse.campaign(a, week, 'pr')['manifest_sha256']
    again = tmp_path / 'again'
    shutil.copytree(week, again)
    with open(os.path.join(week, 'manifest.txt')) as fh:
        lines = fh.readlines()
    with open(again / 'manifest.txt', 'w') as fh:
        fh.writelines(['# exmarg conditioned segments, 2030-01-01 00:00\n', '# a second header line\n'] + lines[1:])
    assert analyse.campaign(a, str(again), 'pr')['manifest_sha256'] == digest
    lines[2] = lines[2].replace(' 33 33 ', ' 33 32 ')
    with open(again / 'manifest.txt', 'w') as fh:
        fh.writelines(lines)
    assert analyse.campaign(a, str(again), 'pr')['manifest_sha256'] != digest


# ----------------------------------------------------------------------------------------------- plant_fit on the week


def write_toy_variant(folder, seg_dir, variant, nbar, eps, n=6000, seed=20261116):
    """toy evidences (test_mixture, loud law) as one variant of analyse.py: the meta of analyse.campaign, merger fields
    set only for inj, glitch fields False (not known)"""
    a = analyse.parse(['x', 'cpu', seg_dir, '--out', 'x', '--seed', '7', '--nbar-inj', '0.05'])[1]
    cfg = analyse.campaign(a, seg_dir, variant)
    toy = Toy()
    pis = law(EDGES, *LAWS['loud'])
    z, fl = toy.data(toy.draws(n, seed), nbar, dict(H=eps, L=eps / 3), dict(H=pis, L=pis))
    ev = toy.evidences(z, [dict(H=pis, L=pis)])
    m = fl['m'] if variant == 'inj' else np.zeros(n, bool)
    per = dict(ln_lambda_bar=ev['S'], ln_lambda_gH_bar=ev['gH'][0], ln_lambda_gL_bar=ev['gL'][0],
               ln_lambda_gHL_bar=ev['HL'][0], ln_lambda_H_bar=ev['SH'], ln_lambda_L_bar=ev['SL'],
               ln_glitch_H=ev['EH'], ln_glitch_L=ev['EL'], wmax_H=np.abs(z['H']).max(axis=1) ** 2,
               wmax_L=np.abs(z['L']).max(axis=1) ** 2, w_inf=np.zeros(n), k=np.arange(n), merger=m, merger_H=m,
               merger_L=m, coherent=m, m1=np.full(n, np.nan), z=np.full(n, np.nan), y=np.zeros(n), snr=np.zeros(n),
               gps=1.2669e9 + 8.0 * np.arange(n))
    for I in 'HL':
        per.update({'glitch_' + I: np.zeros(n, bool), 'kind_' + I: np.full(n, ''), 'rho_' + I: np.full(n, np.nan),
                    't0_' + I: np.full(n, np.nan)})
    os.makedirs(folder, exist_ok=True)
    plant.write_batch(os.path.join(folder, 'plant_%010d_%010d.npz' % (0, n)), cfg, dict(first=0, end=n),
                      dict(per, rho_edges=EDGES, w_inf_cells=np.zeros((0, 2), np.int32)))
    return ev, fl, per


TOY_VARIANTS = dict(pr=(0.0, 0.0), rev=(0.0, 0.10), **{'slide+0.5': (0.0, 0.10), 'slide-1.75': (0.0, 0.10)},
                    inj=(0.05, 0.10), zero=(0.05, 0.10))


@pytest.fixture(scope='module')
def toy_week(week, tmp_path_factory):
    out = tmp_path_factory.mktemp('toy_week')
    made = {v: write_toy_variant(str(out / v), week, v, *TOY_VARIANTS[v]) for v in TOY_VARIANTS}
    with open(out / 'selection.txt', 'w') as fh:                       # one epoch of the 6000 toy segments
        fh.write('epoch_1266933768.npz 1266933768 6000 6000 0\n')
    return str(out), made


def test_plant_fit_on_the_week(toy_week):
    """plant_fit on directories in the format of analyse.py.  zero: the header says it is the week as recorded, no
    pull and no criterion on nbar, no truth for eps, and the joint fit for rho_min 4.5, 5 and 6 side by side (5 the
    default for every fit) equals direct mixture.fit calls at each value on the kept segments with the law up to 12.
    The slid week: the mixture criteria count, the Gaussian ones are printed as the control of the test and do not;
    the phase-randomized week: the Gaussian criteria count too.  inj: nbar within 2.5 sigma of the injected rate"""
    out, made = toy_week
    assert plant_fit._consistent_with_zero(dict(nbar=0.0, err=dict(nbar=np.nan)))[0]       # at the lower edge
    assert not plant_fit._consistent_with_zero(dict(nbar=0.01, err=dict(nbar=0.003)))[0]
    r = plant_fit.report(os.path.join(out, 'zero'))
    text = '\n'.join(r['lines'])
    assert r['kind'] == 'recorded' and 'THE WEEK AS RECORDED' in text and 'truth 0.' not in text
    assert 'pull +' not in text and 'pull -' not in text and 'pull nan' not in text
    assert [c[0] for c in r['criteria']] == ['population covered: w_inf < 0.0005'] and r['rho_min'] == 5.0
    assert 'the joint fit for each lower end of the glitch law (side by side)' in text
    ev, fl, per = made['zero']
    keep = (per['wmax_H'] <= 72) & (per['wmax_L'] <= 72)
    for rm, fitted, curve in r['scan']:
        d = mixture.fit(ev['S'][keep], ev['gH'][0][keep], ev['gL'][0][keep], ev['EH'][keep], ev['EL'][keep], EDGES,
                        start=dict(nbar=0.01, eps_H=0.01, eps_L=0.01, gamma_H=3.0, gamma_L=3.0, rho_min_H=rm,
                                   rho_min_L=rm), fixed=('rho_min_H', 'rho_min_L'), log=True,
                        lam_gHL=ev['HL'][0][keep], rho_max=12.0)
        assert abs(fitted['nbar'] - d['nbar']) <= 1e-12 * d['nbar'] and abs(fitted['lnl'] - d['lnl']) < 1e-9 * abs(
            d['lnl']), (rm, fitted['nbar'], d['nbar'])
        assert '%s +- %s' % (plant_fit._f(d['nbar'], '%.4g'), plant_fit._f(d['err']['nbar'], '%.2g')) in text, rm
    assert [rm for rm, _, _ in r['scan']] == [4.5, 5.0, 6.0]
    assert len({round(f['nbar'], 9) for _, f, _ in r['scan']}) == 3
    print('\n' + text)
    for v, counted_gauss in (('slide+0.5', False), ('pr', True)):
        r = plant_fit.report(os.path.join(out, v))
        crit = {name: (ok, counted) for name, ok, _, counted in r['criteria']}
        assert r['kind'] == 'control' and crit['mixture: nbar consistent with 0'][1] and crit[
            'mixture: evidence below ln 100'][1]
        assert crit['Gaussian: nbar consistent with 0'][1] == counted_gauss
        assert crit['Gaussian: evidence below ln 100'][1] == counted_gauss
        assert r['ok'] == all(ok for ok, counted in crit.values() if counted)
        if not counted_gauss:
            assert 'control of the test, not a failure of the run' in '\n'.join(r['lines'])
    r = plant_fit.report(os.path.join(out, 'inj'))
    J = r['fits']['joint']
    assert r['kind'] == 'plant' and r['truth']['nbar'] == 0.05
    crit = {name: ok for name, ok, _, _ in r['criteria']}
    # 300 injected mergers in 5814 kept segments: the rate is recovered (0.0442 +- 0.0047, 1.2 sigma low) and the
    # criterion passes ...
    assert 0.035 < J['nbar'] < 0.065 and 0.003 < J['err']['nbar'] < 0.007, J
    assert crit['mixture nbar within 2.5 sigma of the truth'] is True and r['ok']
    # ... and it can fail: the same files claimed to hold mergers at 0.2 (4 times the injected rate, 33 sigma away)
    wrong = os.path.join(out, 'inj_wrong_rate')
    os.makedirs(wrong)
    for path in glob.glob(os.path.join(out, 'inj', 'plant_*.npz')):
        with np.load(path) as f:
            arrays = {key: f[key] for key in f.files}
        meta = json.loads(str(arrays.pop('meta')))
        np.savez(os.path.join(wrong, os.path.basename(path)), meta=np.array(json.dumps(dict(meta, nbar=0.2))), **arrays)
    r = plant_fit.report(wrong)
    crit = {name: ok for name, ok, _, _ in r['criteria']}
    assert crit['mixture nbar within 2.5 sigma of the truth'] is False and not r['ok']


def test_plant_fit_reports_segments_missing_at_the_end(toy_week):
    """plant_fit counted only the gaps in between: with the expected total, the segments missing at the start and at
    the end are reported (and the line is that of the plants without it)"""
    out, _ = toy_week
    k = np.arange(10, 5990)
    assert plant_fit.missing_text(k) == '0 missing in between'
    assert plant_fit.missing_text(np.r_[k[:100], k[200:]]) == '100 missing in between'
    assert plant_fit.missing_text(k, 6000) == '0 missing in between, 10 at the start and 10 at the end of the 6000 ' \
        'expected'
    d = os.path.join(out, 'pr')
    assert '(0 missing in between)' in plant_fit.report(d, rho_mins=[5.0])['lines'][1]
    line = plant_fit.report(d, rho_mins=[5.0], expected=6100)['lines'][1]
    assert '0 missing in between, 0 at the start and 100 at the end of the 6100 expected' in line, line


# ----------------------------------------------------------------------------------------------- summarize


def fake_report(log, failing=(), cfgs=None):
    """plant_fit.report in place: the campaign (meta) of each variant is cfgs[variant] (default: none)"""
    def report(path, **kw):
        v = os.path.basename(path)
        log.append(v)
        ok = v not in failing
        crit = [('mixture: nbar consistent with 0', ok, '', True), ('Gaussian: nbar consistent with 0', False, '',
                                                                     v == 'pr')]
        fitted = dict(nbar=0.01, err=dict(nbar=0.01, eps_H=0.001, eps_L=0.001), eps_H=0.01, eps_L=0.02, gamma_H=3.0,
                      gamma_L=3.5)
        return dict(kind='recorded' if v == 'zero' else 'control', ok=ok, criteria=crit, rho_min=5.0,
                    cfg=(cfgs or {}).get(v, {}),
                    fits=dict(joint=fitted, gauss=fitted), curves=dict(gauss=np.zeros(3), mixture=np.zeros(3)),
                    keep=np.ones(10, bool), lines=['the report of %s' % v])
    return report


def guard_week(root, week, files=None, only=None):
    """<root>/selection.txt of the synthetic week and, per variant, files covering k 0 .. n_selected - 1 in two halves
    (files: {variant: [(first, end)]} instead); the files of controls hold k and the meta of analyse.campaign, those of
    zero are empty (never to be opened).  returns n_selected"""
    sel = analyse.selection(week)
    n = sum(e['n_selected'] for e in sel)
    os.makedirs(root, exist_ok=True)
    analyse.record_selection(str(root), sel)
    args = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7'])[1]
    for v in analyse.VARIANTS if only is None else only:
        os.makedirs(os.path.join(root, v), exist_ok=True)
        for first, end in (files or {}).get(v, [(0, n // 2), (n // 2, n)]):
            path = os.path.join(root, v, 'plant_%010d_%010d.npz' % (first, end))
            if v == 'zero':
                open(path, 'wb').close()
            else:
                plant.write_batch(path, analyse.campaign(args, week, v), dict(first=first, end=end),
                                  dict(k=np.arange(first, end)))
    return n


def test_summarize_guard(week, tmp_path, monkeypatch, capsys):
    """summarize: the controls in the fixed order (pr, rev, slide+0.5, slide-1.75, inj); without --recorded the
    recorded week is never opened, whatever the controls; with --recorded it is opened last and only when every control
    passes, and says so; a failing control keeps it closed, exit status 1"""
    guard_week(tmp_path, week)
    args = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7'])[1]
    cfgs = {v: analyse.campaign(args, week, v) for v in analyse.VARIANTS}
    order = list(summarize.ORDER)
    for argv, failing, opened, code, says in (
            ([], (), order, 0, 'is not opened'),
            (['--recorded'], (), order + ['zero'], 0, 'all pass: opening the week as recorded'),
            (['--recorded'], ('slide-1.75',), order, 1, 'NOT all pass: the week as recorded stays closed'),
            ([], ('pr',), order, 1, 'NOT all pass')):
        log = []
        monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log, failing, cfgs))
        assert summarize.main([str(tmp_path)] + argv) == code, (argv, failing)
        out = capsys.readouterr().out
        assert log == opened and says in out, (argv, failing, log, out)
    # the recorded week of another campaign (opened last, once the controls pass): its numbers are not shown
    log = []
    monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log, (), dict(cfgs, zero=dict(cfgs['zero'], seed=9))))
    assert summarize.main([str(tmp_path), '--recorded']) == 1
    out = capsys.readouterr().out
    assert 'another campaign' in out and 'seed' in out and 'the report of zero' not in out and log[-1] == 'zero'


def refused(root, monkeypatch, *argv):
    """summarize.main on root: the SystemExit message, and the variants fitted before it (must be none)"""
    log = []
    monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log))
    with pytest.raises(SystemExit) as err:
        summarize.main([str(root)] + list(argv))
    assert log == [], 'something was fitted before the guard stopped the run: %s' % log
    return str(err.value)


def test_summarize_guard_covers_every_segment(week, tmp_path, monkeypatch):
    """before any fit, every variant that is used must cover k = 0 .. n_selected - 1 once each: the first of two files
    of each control only (the missing end), a gap in the middle, a segment in two files, a variant with no files, no
    selection.txt, are hard errors that name the variant and what is missing; zero is checked from its file names alone
    (its files here are empty and cannot be read), and only with --recorded"""
    root = tmp_path / 'week'
    n = guard_week(root, week)
    h = n // 2
    partial = tmp_path / 'partial'
    guard_week(partial, week, {v: [(0, h)] for v in summarize.ORDER}, only=summarize.ORDER)
    msg = refused(partial, monkeypatch)
    assert all(v in msg for v in summarize.ORDER) and 'k %d to %d' % (h, n - 1) in msg and 'missing' in msg, msg
    gap = tmp_path / 'gap'
    guard_week(gap, week, {'rev': [(0, 10), (20, n)]})
    msg = refused(gap, monkeypatch)
    assert 'rev' in msg and 'k 10 to 19' in msg and len(msg.splitlines()) == 2, msg
    twice = tmp_path / 'twice'
    guard_week(twice, week, {'slide+0.5': [(0, h + 5), (h, n)]})
    msg = refused(twice, monkeypatch)
    assert 'slide+0.5' in msg and 'k %d to %d' % (h, h + 4) in msg and 'more than one file' in msg, msg
    none = tmp_path / 'none'
    guard_week(none, week, only=[v for v in summarize.ORDER if v != 'inj'])
    msg = refused(none, monkeypatch)
    assert 'inj' in msg and 'no files' in msg, msg
    os.remove(root / 'selection.txt')
    assert 'selection.txt' in refused(root, monkeypatch)
    # zero: not looked at without --recorded, by its names only with it
    short_zero = tmp_path / 'short_zero'
    guard_week(short_zero, week, {'zero': [(0, h)]})
    log = []
    monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log))
    assert summarize.main([str(short_zero)]) == 0 and 'zero' not in log
    msg = refused(short_zero, monkeypatch, '--recorded')
    assert 'zero' in msg and 'k %d to %d' % (h, n - 1) in msg, msg


def test_summarize_guard_one_campaign(week, tmp_path, monkeypatch):
    """the campaign keys of the files (everything of the meta that the variant does not define: manifest hash, seed,
    segment folder, ymin, zmin, ...) are the same in every variant: one file of slide-1.75 with another ymin, seed or
    manifest hash stops the run before a fit, naming the variant and the key; the keys that belong to the variant
    (control, nbar, shift, ...) may differ"""
    n = guard_week(tmp_path / 'ok', week)
    log = []
    monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log))
    assert summarize.main([str(tmp_path / 'ok')]) == 0 and log == list(summarize.ORDER)
    args = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7'])[1]
    for key, value in (('ymin', 0.5), ('seed', 8), ('manifest_sha256', 'f' * 64), ('segments', 'other'),
                       ('zmin', 0.5), ('epochs_per_file', 3)):
        root = tmp_path / ('bad_' + key)
        guard_week(root, week, {'slide-1.75': [(0, n)]})
        plant.write_batch(os.path.join(root, 'slide-1.75', 'plant_%010d_%010d.npz' % (0, n)),
                          dict(analyse.campaign(args, week, 'slide-1.75'), **{key: value}), dict(first=0, end=n),
                          dict(k=np.arange(n)))
        msg = refused(root, monkeypatch)
        assert 'slide-1.75' in msg and key in msg, (key, msg)


def test_summarize_table(toy_week, capsys):
    """summarize on the toy week: a row per control and the injections with the numbers of plant_fit.report, the
    recorded week not opened without --recorded"""
    out, _ = toy_week
    code = summarize.main([out, '--rho-min', '5'])
    text = capsys.readouterr().out
    for v in summarize.ORDER:
        r = plant_fit.report(os.path.join(out, v), rho_mins=[5.0])
        J = r['fits']['joint']
        line = [l for l in text.splitlines() if l.startswith(v + ' ')][0]
        assert '%s+-%s' % (plant_fit._f(J['nbar'], '%.3g'), plant_fit._f(J['err']['nbar'], '%.2g')) in line, (v, line)
        assert line.split()[1] == str(int(r['keep'].sum()))
    assert not any(l.startswith('zero ') for l in text.splitlines()) and 'is not opened' in text
    assert code == 0 and 'all pass' in text                  # the toy controls have no mergers: all pass
    assert all(l.split(None, 11)[-1].startswith('PASS') for l in text.splitlines() if l.split(' ')[0] in summarize.ORDER)
    print('\n' + text)


# ----------------------------------------------------------------------------------------------- the window model


class WinStub:
    """pe.window_evidences in place for the week: test_plant.window_stub of the rows asked; records the calls"""

    def __init__(self, n_window=plant.N_WINDOW):
        self.calls, self.n_window = [], n_window

    def factory(self, seg0):
        return self.win

    def win(self, seg, rows):
        self.calls.append(np.asarray(rows).copy())
        return window_stub(seg, np.asarray(rows), self.n_window)


def test_windows_stage_week(week, tmp_path):
    """analyse.main --stage windows next to the merger stage: in every variant the win files have the names (k ranges)
    of the plant files and the same k, gps, flags, monitors, truth and wmax; ln_window_I float32 (n, 64, 14) is the
    stub's on the selected rows of the variant's data; one call per epoch with the rows of every variant (those of the
    merger stage); the meta is the campaign without the merger settings, with stage 'windows', and campaign_key that of
    the plant files; --part splits and --variants subsets give the same files bit for bit; a second run calls nothing;
    a run of another seed or n_window into the folder stops.  windows_fit reads a variant whole, and refuses it with a
    file missing at the end (the selection)"""
    out = tmp_path / 'w'
    merger = drive(week, out, '--epochs-per-file', '1')
    win = drive(week, out, '--epochs-per-file', '1', '--stage', 'windows', stub=WinStub())
    assert len(win.calls) == 3 and all(np.array_equal(a, b) for a, b in zip(win.calls, [c[1] for c in merger.calls]))
    s = plant.Sampler(plant.population(analyse.ZMIN), plant.MDET)
    bins = plant.window_bins(3.0, 12.2)
    a = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7', '--epochs-per-file', '1'])[1]
    sel = analyse.selection(week)
    for v in analyse.VARIANTS:
        plants = sorted(os.path.basename(f) for f in glob.glob(str(out / v / 'plant_*.npz')))
        wins = sorted(os.path.basename(f) for f in glob.glob(str(out / v / 'win_*.npz')))
        assert [w[4:] for w in wins] == [p[6:] for p in plants] and len(wins) == 3, v
        for p, w, e in zip(plants, wins, [e for e in sel if e['n_selected']]):
            mp, mw = plant.read_meta(out / v / p), plant.read_meta(out / v / w)
            assert mw['stage'] == 'windows' and plant.campaign_key(mw) == plant.campaign_key(mp) and 'ymin' not in mw
            mw.pop('run')
            a.stage = 'windows'
            assert mw == analyse.campaign(a, week, v)
            a.stage = 'merger'
            ep = strain.read_epoch(os.path.join(week, e['name']))
            var, rows, inj = analyse.variants(dict(seed=7, nbar_inj=0.3), ep, (v,), s)
            ref = window_stub(var[v], np.arange(rows.size), 64)
            with np.load(out / v / p) as fp, np.load(out / v / w) as fw:
                for key in fp.files:
                    if key in fw.files and key != 'meta':
                        assert np.array_equal(fp[key], fw[key], equal_nan=fp[key].dtype.kind == 'f'), (v, key)
                assert {'gps', 'monitor_H', 'event', 'merger', 'wmax_L'} <= set(fw.files)
                for I in 'HL':
                    assert fw['ln_window_' + I].dtype == np.float32
                    assert fw['ln_window_' + I].shape == (rows.size, 64, 14)
                    assert np.array_equal(fw['ln_window_' + I], ref['ln_window_' + I][:, :, bins].astype(np.float32))
    two = tmp_path / 'two'
    drive(week, two, '--epochs-per-file', '1', '--stage', 'windows', '--part', '0/2', stub=WinStub())
    drive(week, two, '--epochs-per-file', '1', '--stage', 'windows', '--part', '1/2', '--variants', 'pr,inj',
          stub=WinStub())
    drive(week, two, '--epochs-per-file', '1', '--stage', 'windows', '--part', '1/2', stub=WinStub())
    for v in analyse.VARIANTS:
        for f in glob.glob(str(out / v / 'win_*.npz')):
            with np.load(f) as x, np.load(two / v / os.path.basename(f)) as y:
                for key in x.files:
                    if key != 'meta':
                        assert np.array_equal(x[key], y[key], equal_nan=x[key].dtype.kind == 'f'), (v, key)
    assert drive(week, out, '--epochs-per-file', '1', '--stage', 'windows', stub=WinStub()).calls == []
    with pytest.raises(SystemExit, match='another campaign.*seed'):
        drive(week, out, '--epochs-per-file', '1', '--stage', 'windows', '--seed', '8', stub=WinStub())
    with pytest.raises(SystemExit, match='another campaign.*n_window'):
        drive(week, out, '--epochs-per-file', '1', '--stage', 'windows', '--n-window', '32', stub=WinStub(32))
    cfg, d, shared, paths = windows_fit.load(str(out / 'pr'))
    n_sel = sum(e['n_selected'] for e in sel)
    assert np.array_equal(d['k'], np.arange(n_sel)) and len(paths) == 3
    os.remove(sorted(paths)[-1])
    with pytest.raises(SystemExit, match='cover k 0 to 60 of the 94 selected'):
        windows_fit.load(str(out / 'pr'))


def windows_week(week, out):
    """the windows stage of the synthetic week into out and a made-up model per variant (distinct q) in
    <out>/<variant>/glitch_model.json with the campaign key of its files: {variant: SHA-256}"""
    drive(week, out, '--epochs-per-file', '2', '--stage', 'windows', stub=WinStub())
    shas = {}
    for i, v in enumerate(analyse.VARIANTS):
        meta = plant.read_meta(sorted(glob.glob(str(out / v / 'win_*.npz')))[0])
        shas[v] = make_model(out / v / 'glitch_model.json', plant.campaign_key(meta), q_scale=1.0 + i)
    return shas


def test_model_stage_week(week, tmp_path):
    """--glitch-model auto: every variant is one pe call per epoch with its own model (q of its file) at its rows of
    the merger stage and the seed stream(seed, GPS of the epoch, 'pe'); its files record glitch_model 'windows', that
    model's SHA-256 and n_window, and hold ln_B_bar, ln_c_I, state_I and no ln_lambda_g*.  A given file is used for
    every variant (fitted on a variant of the same week).  Refused before any call: a variant without its model, the
    model of another variant in auto, another n_window, a model of another week as a given file, a modified file, and
    --glitch-model with --stage windows"""
    out = tmp_path / 'm'
    shas = windows_week(week, out)
    stub = drive(week, out, '--epochs-per-file', '2', '--glitch-model', 'auto')
    sel = {1266933768: np.setdiff1d(np.arange(NSEG), [3, 30]), 1266934296: np.setdiff1d(np.arange(NSEG), [8, 9, 20]),
           1266934560: np.arange(NSEG)}
    expect = [(plant.stream(7, g, 'pe'), analyse.VARIANTS.index(v) * NSEG + sel[g], v)
              for g in sel for v in analyse.VARIANTS]
    assert len(stub.calls) == len(expect)
    for (seed, rows), glitch, (seed_e, rows_e, v) in zip(stub.calls, stub.glitches, expect):
        model = plant.read_model(str(out / v / 'glitch_model.json'))[0]
        assert seed == seed_e and np.array_equal(rows, rows_e) and glitch['model'] == 'windows', v
        assert all(np.array_equal(glitch[I][p], model[I][p]) for I in 'HL' for p in ('f', 'q', 'pi_k')), v
    for v in analyse.VARIANTS:
        for f in glob.glob(str(out / v / 'plant_*.npz')):
            meta = plant.read_meta(f)
            assert meta['glitch_model'] == 'windows' and meta['glitch_model_sha256'] == shas[v]
            assert meta['n_window'] == 64
            with np.load(f) as x:
                assert {'ln_B_bar', 'ln_c_H', 'state_L', 'ln_lambda_bar'} <= set(x.files)
                assert not [k for k in x.files if k.startswith('ln_lambda_g')] and x['state_H'].shape[1] == 2
    assert drive(week, out, '--epochs-per-file', '2', '--glitch-model', 'auto').calls == []
    given = tmp_path / 'given'
    drive(week, given, '--epochs-per-file', '2', '--glitch-model', str(out / 'slide+0.5' / 'glitch_model.json'))
    assert {plant.read_meta(f)['glitch_model_sha256'] for f in glob.glob(str(given / '*' / 'plant_*.npz'))} == {
        shas['slide+0.5']}
    bad = tmp_path / 'bad'
    windows_week(week, bad)
    os.remove(bad / 'rev' / 'glitch_model.json')
    for extra, match in ((('--glitch-model', 'auto'), 'rev/glitch_model.json: no glitch model'),
                         (('--glitch-model', 'auto', '--n-window', '32'), 'n_window = 64, the run 32')):
        stub = Stub()
        with pytest.raises(SystemExit, match=match):
            drive(week, bad, '--epochs-per-file', '2', *extra, stub=stub)
        assert stub.calls == []
    import shutil
    shutil.copy(bad / 'pr' / 'glitch_model.json', bad / 'rev' / 'glitch_model.json')
    with pytest.raises(SystemExit, match='rev/glitch_model.json: the model was fitted on another campaign.*control'):
        drive(week, bad, '--epochs-per-file', '2', '--glitch-model', 'auto', stub=Stub())
    meta = plant.read_meta(sorted(glob.glob(str(bad / 'pr' / 'win_*.npz')))[0])
    make_model(tmp_path / 'other_week.json', dict(plant.campaign_key(meta), manifest_sha256='f' * 64))
    with pytest.raises(SystemExit, match='another campaign.*manifest_sha256'):
        drive(week, tmp_path / 'x', '--epochs-per-file', '2', '--glitch-model', str(tmp_path / 'other_week.json'))
    changed = json.loads(open(bad / 'pr' / 'glitch_model.json').read())
    changed['H']['f'] = [0.7, 0.3]
    with open(tmp_path / 'changed.json', 'w') as fh:
        json.dump(changed, fh)
    with pytest.raises(SystemExit, match='modified'):
        drive(week, tmp_path / 'x', '--epochs-per-file', '2', '--glitch-model', str(tmp_path / 'changed.json'))
    with pytest.raises(SystemExit):
        analyse.parse(['x', 'cpu', week, '--out', 'x', '--stage', 'windows', '--glitch-model', 'auto'])
    assert not glob.glob(str(tmp_path / 'x' / '*' / 'plant_*.npz'))


def test_variant_keys_agree():
    """the keys a variant defines are the same in analyse.py and summarize.py (which adds the model's SHA-256)"""
    assert set(summarize.VARIANT_KEYS) == set(analyse.VARIANT_KEYS) | {'glitch_model_sha256'}


def fake_report_w(log, failing=(), cfgs=None):
    """plant_fit.report of files of the window model in place"""
    def report(path, **kw):
        v = os.path.basename(path)
        log.append(v)
        ok = v not in failing
        crit = [('window model: nbar consistent with 0', ok, '', True),
                ('Gaussian: nbar consistent with 0', False, '', v == 'pr')]
        fitted = dict(nbar=0.01, err=dict(nbar=0.01))
        model = dict(n_window=64, H=dict(f=[0.8, 0.2], q=[0.004, 0.05]), L=dict(f=[0.9, 0.1], q=[0.002, 0.08]))
        return dict(kind='recorded' if v == 'zero' else 'control', ok=ok, criteria=crit, rho_min=None, model='windows',
                    cfg=(cfgs or {}).get(v, {}), window_model=model, fits=dict(windows=fitted, gauss=fitted),
                    curves=dict(gauss=np.zeros(3), windows=np.zeros(3)), keep=np.ones(10, bool),
                    lines=['the report of %s' % v])
    return report


def guard_week_w(root, week, files=None, only=None, change=None):
    """guard_week with the meta of the merger stage with the window model (a SHA-256 per variant); change: {variant:
    dict of meta entries replaced}.  returns n_selected and the campaigns"""
    sel = analyse.selection(week)
    n = sum(e['n_selected'] for e in sel)
    os.makedirs(root, exist_ok=True)
    analyse.record_selection(str(root), sel)
    args = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7'])[1]
    cfgs = {v: dict(analyse.campaign(args, week, v, '%064x' % (i + 1)), **(change or {}).get(v, {}))
            for i, v in enumerate(analyse.VARIANTS)}
    for v in analyse.VARIANTS if only is None else only:
        os.makedirs(os.path.join(root, v), exist_ok=True)
        for first, end in (files or {}).get(v, [(0, n // 2), (n // 2, n)]):
            path = os.path.join(root, v, 'plant_%010d_%010d.npz' % (first, end))
            if v == 'zero':
                open(path, 'wb').close()
            else:
                plant.write_batch(path, cfgs[v], dict(first=first, end=end), dict(k=np.arange(first, end)))
    return n, cfgs


def test_summarize_guard_window_model(week, tmp_path, monkeypatch, capsys):
    """summarize on variants of the window model (each its own SHA-256, which is allowed): the controls in the fixed
    order with the table of the window model; zero never opened without --recorded, opened last with it when every
    control passes, and not opened when one fails; incomplete controls, a variant analysed with the mixture (no
    glitch_model) or with another n_window are refused before any fit"""
    n, cfgs = guard_week_w(tmp_path / 'ok', week)
    order = list(summarize.ORDER)
    for argv, failing, opened, code, says in (
            ([], (), order, 0, 'is not opened'),
            (['--recorded'], (), order + ['zero'], 0, 'all pass: opening the week as recorded'),
            (['--recorded'], ('rev',), order, 1, 'NOT all pass: the week as recorded stays closed'),
            ([], ('inj',), order, 1, 'NOT all pass')):
        log = []
        monkeypatch.setattr(summarize.plant_fit, 'report', fake_report_w(log, failing, cfgs))
        assert summarize.main([str(tmp_path / 'ok')] + argv) == code, (argv, failing)
        text = capsys.readouterr().out
        assert log == opened and says in text, (argv, failing, log, text)
        assert 'window nbar' in text and 'glitches/seg H' in text and '0.8448' in text and '0.6272' in text
        assert [l.split()[7] for l in text.splitlines() if l.split(' ')[0] in order] == [
            cfgs[v]['glitch_model_sha256'][:12] for v in order]
    for name, files, change, match in (
            ('partial', {v: [(0, n // 2)] for v in summarize.ORDER}, None, 'missing'),
            ('mixture', None, {'slide-1.75': dict(glitch_model=None)}, 'slide-1.75.*glitch_model'),
            ('n_window', None, {'rev': dict(n_window=32)}, 'rev.*n_window')):
        root = tmp_path / name
        guard_week_w(root, week, files, change=change)
        log = []
        monkeypatch.setattr(summarize.plant_fit, 'report', fake_report_w(log))
        with pytest.raises(SystemExit, match=match):
            summarize.main([str(root), '--recorded'])
        assert log == [], (name, log)


def test_summarize_window_model_table(week, tmp_path, capsys):
    """summarize end to end on synthetic files of the window model in every variant (Bbar of test_plant.synthetic_B:
    no mergers in the controls, mergers at 0.05 injected): each row has the numbers of plant_fit.report, the controls
    and the injections pass, and the recorded week is not opened"""
    out = tmp_path / 'toy'
    os.makedirs(out)
    with open(out / 'selection.txt', 'w') as fh:
        fh.write('epoch_1266933768.npz 1266933768 6000 6000 0\n')
    a = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7', '--nbar-inj', '0.05'])[1]
    rng = np.random.default_rng(3)
    for i, v in enumerate(analyse.VARIANTS):
        lnB, m = synthetic_B(6000, 0.05 if v in ('inj', 'zero') else 0.0, 30 + i)
        m = m if v == 'inj' else np.zeros(6000, bool)
        n = lnB.size
        per = dict(k=np.arange(n), ln_B_bar=lnB, ln_lambda_bar=lnB + 2.0 * (rng.random(n) < 0.05),
                   ln_lambda_H_bar=0.5 * lnB, ln_lambda_L_bar=0.5 * lnB, w_inf=np.zeros(n), merger=m, merger_H=m,
                   merger_L=m, coherent=m, m1=np.full(n, np.nan), z=np.full(n, np.nan), y=np.zeros(n),
                   snr=np.zeros(n), gps=1.2669e9 + 8.0 * np.arange(n))
        for I in 'HL':
            per.update({'ln_c_' + I: np.zeros(n), 'state_' + I: np.tile([0.9, 0.1], (n, 1)),
                        'ln_glitch_' + I: np.zeros((n, RHO_EDGES.size - 1)), 'wmax_' + I: rng.exponential(5.0, n),
                        'glitch_' + I: np.zeros(n, bool), 'kind_' + I: np.full(n, ''), 'rho_' + I: np.full(n, np.nan),
                        't0_' + I: np.full(n, np.nan)})
        os.makedirs(out / v)
        plant.write_batch(str(out / v / ('plant_%010d_%010d.npz' % (0, n))), analyse.campaign(a, week, v, '%064x' % 5),
                          dict(first=0, end=n), dict(per, rho_edges=RHO_EDGES, glitch_pi_k=np.array(np.nan),
                                                     glitch_window=np.array(np.nan),
                                                     w_inf_cells=np.zeros((0, 2), np.int32)))
    code = summarize.main([str(out)])
    text = capsys.readouterr().out
    print('\n' + text)
    for v in summarize.ORDER:
        r = plant_fit.report(str(out / v))
        W = r['fits']['windows']
        line = [l for l in text.splitlines() if l.startswith(v + ' ')][0]
        assert '%s+-%s' % (plant_fit._f(W['nbar'], '%.3g'), plant_fit._f(W['err']['nbar'], '%.2g')) in line, (v, line)
        assert line.split()[1] == str(int(r['keep'].sum())) and 'PASS' in line, line
    assert code == 0 and 'is not opened' in text and not any(l.startswith('zero ') for l in text.splitlines())


# ----------------------------------------------------------------------------------------------- more time slides

slides_mean = _script('slides_mean', 'real_data')
EXTRA = ('slide+2.25', 'slide-3.0')
EXTRA_SAMPLES = {'slide+2.25': 9216, 'slide-3.0': -12288}            # shift x 4096 matched-filter samples per second
EVERY = analyse.VARIANTS + EXTRA


def sampler():
    return plant.Sampler(plant.population(analyse.ZMIN), plant.MDET)


def test_generic_slide_is_time_slide(week):
    """a slide named slide<signed number> is controls.time_slide by that many seconds, whatever else is asked: H as it
    is, L moved by exactly the matched-filter samples of the shift (9216 and -12288 for +2.25 and -3.0 s, the series
    rolled), array for array equal to controls.time_slide of the recorded data; the two built-in slides and inj are bit
    for bit what they are without the extra slides, and equal to controls.time_slide by their shifts; the name as typed
    (slide-3.0) is the key"""
    ep = strain.read_epoch(os.path.join(week, 'epoch_1266934296.npz'))
    params = dict(seed=7, nbar_inj=0.5)
    var, rows, _ = analyse.variants(params, ep, EVERY, sampler())
    plain, _, _ = analyse.variants(params, ep, analyse.VARIANTS, sampler())
    alone = analyse.variants(params, ep, ('slide-3.0',), sampler())[0]
    zero = var['zero']
    assert list(var) == list(EVERY) and set(alone) == {'slide-3.0'}
    for v, shift in dict(analyse.SHIFTS, **{'slide+2.25': 2.25, 'slide-3.0': -3.0}).items():
        ref = controls.time_slide(zero, shift)
        for I in 'HL':
            assert np.array_equal(var[v]['d_' + I], ref['d_' + I]), (v, I)
        assert np.array_equal(var[v]['d_H'], zero['d_H'])
    for v, samples in EXTRA_SAMPLES.items():
        assert analyse.whole_samples(zero, analyse.slide_shift(v)) == samples
        x, _ = series(zero, 'L')
        xs, _ = series(var[v], 'L')
        assert np.max(np.abs(xs - np.roll(x, samples, axis=1))) < 1e-12 * np.abs(x).max(), v
    assert np.array_equal(alone['slide-3.0']['d_L'], var['slide-3.0']['d_L'])
    for v in analyse.VARIANTS:
        for key in ('d_H', 'd_L'):
            assert np.array_equal(var[v][key], plain[v][key]), (v, key)
    assert analyse.slide_shift('slide-3.0') == -3.0 and analyse.slide_shift('slide+0.5') == 0.5


def test_slide_names_and_shifts_refused(week, tmp_path):
    """names that are not slide, a sign and digits[.digits] are refused by the command line (no exponent, no inf, no
    missing sign, no other case, no non-ASCII digit); the names as typed are kept (slide+2.250 is not slide+2.25) and
    repeated names merged; a name with the shift of another (as written: slide+0.50 and the built-in slide+0.5, or
    modulo the 8 s segment: slide+8.5, slide+10.5 and slide+2.5) is refused, naming both; a shift that is not a whole
    number of samples, within 0.02 s of zero or of T, stops the run with the variant's name before any call; a folder of
    another shift under the same name (the files of slide+2.25 in slide+2.5) is refused as another campaign"""
    for bad in ('slide', 'slide+', 'slide2.25', 'slide+1e3', 'slide+2.', 'slide+.5', 'slide++2', 'Slide+2.25',
                'slide+2.25s', 'slide+nan', 'slide+inf', 'slide+ 2', 'slide+0x10', 'sliding', 'slide+٣',
                'slide+' + '9' * 400):
        with pytest.raises(SystemExit):
            analyse.parse(['x', 'cpu', 'segs', '--out', 'o', '--variants', 'zero,' + bad])
    a = analyse.parse(['x', 'cpu', 'segs', '--out', 'o', '--variants', 'slide+2.250, slide-3.0,slide-3.0,pr'])[1]
    assert a.variants == ['slide+2.250', 'slide-3.0', 'pr']
    for names in ('slide+0.50', 'slide+2.25,slide+2.250', 'slide-1.750'):
        with pytest.raises(SystemExit):
            analyse.parse(['x', 'cpu', 'segs', '--out', 'o', '--variants', names])
    ep = strain.read_epoch(os.path.join(week, 'epoch_1266934296.npz'))
    params = dict(seed=7, nbar_inj=0.5)
    for names, match in ((('slide+0.0001',), r'slide\+0\.0001.*whole number'),
                         (('zero', 'slide+0.015625'), r'slide\+0\.015625.*within'),
                         (('slide+8.0',), r'slide\+8\.0.*within'), (('slide-0.0',), r'slide-0\.0.*within'),
                         (('slide+8.5',), r'slide\+0\.5 and slide\+8\.5 are the same shift modulo'),
                         (('slide+2.5', 'slide+10.5'), r'slide\+2\.5 and slide\+10\.5 are the same shift modulo'),
                         (('slide+2.5', 'slide-5.5'), r'slide\+2\.5 and slide-5\.5 are the same shift modulo')):
        with pytest.raises(ValueError, match=match):
            analyse.variants(params, ep, names, sampler())
    stub = Stub()
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match=r'slide\+0\.0001.*whole number'):
        drive(week, out, '--variants', 'zero,slide+0.0001', stub=stub)
    assert stub.calls == [] and not glob.glob(str(out / '*' / 'plant_*.npz'))
    drive(week, out, '--epochs-per-file', '2', '--variants', 'slide+2.25')
    import shutil
    shutil.copytree(out / 'slide+2.25', out / 'slide+2.5')
    stub = Stub()
    with pytest.raises(SystemExit, match='another campaign.*shift'):
        drive(week, out, '--epochs-per-file', '2', '--variants', 'slide+2.5', stub=stub)
    assert stub.calls == []


def same_data(a, b):
    """the campaign and every per-segment array of two folders of plant files equal, bit for bit"""
    ca, _, da, _, _ = plant_fit.load(str(a))
    cb, _, db, _, _ = plant_fit.load(str(b))
    assert ca == cb, (a, b)
    assert set(da) == set(db)
    for key in da:
        assert np.array_equal(da[key], db[key], equal_nan=da[key].dtype.kind == 'f'), (a, key)


def test_extra_slides_driver(week, tmp_path):
    """analyse.main with extra slides and the stub: the built-in variants are one call per epoch exactly as without the
    extras (same rows, seed, and files bit for bit) and each extra slide a call of its own on the epoch's rows with the
    seed stream(seed, GPS + 2^24 c, 'pe') (c = 2 |m| - [m < 0] = 18432 for +2.25 s, 24575 for -3.0 s) -- distinct from
    each other and from the built-ins' --; an extra slide's files are the same alone, in another order, in another
    part split and with the built-ins (so they do not depend on the other variants asked for), and hold k and gps of the
    other variants, control = its name, nbar 0, shift as typed, no merger, other data than slide+0.5 (another seed)"""
    gps = [1266933768, 1266934296, 1266934560]
    sel = {1266933768: np.setdiff1d(np.arange(NSEG), [3, 30]), 1266934296: np.setdiff1d(np.arange(NSEG), [8, 9, 20]),
           1266934560: np.arange(NSEG)}
    one, base = tmp_path / 'one', tmp_path / 'base'
    s_all = drive(week, one, '--epochs-per-file', '1', '--variants', ','.join(EVERY))
    drive(week, base, '--epochs-per-file', '1')
    code = {'slide+2.25': 18432, 'slide-3.0': 24575}
    want = []
    for g in gps:
        want.append((plant.stream(7, g, 'pe'), np.concatenate([i * NSEG + sel[g] for i in range(6)])))
        want += [(plant.stream(7, g + code[v] * 2 ** 24, 'pe'), sel[g]) for v in EXTRA]
    assert len(s_all.calls) == len(want) == 9
    for (seed, rows), (seed_e, rows_e) in zip(s_all.calls, want):
        assert seed == seed_e and np.array_equal(rows, rows_e)
    assert len({seed for seed, _ in s_all.calls}) == 9
    for v in analyse.VARIANTS:
        same_data(one / v, base / v)
    n_sel = sum(v.size for v in sel.values())
    a0 = plant_fit.load(str(one / 'slide+0.5'))[2]
    for v in ('slide-3.0', 'slide+2.25'):                                                  # each alone
        alone = tmp_path / ('alone_' + v)
        drive(week, alone, '--epochs-per-file', '1', '--variants', v)
        same_data(one / v, alone / v)
    other = tmp_path / 'other'
    drive(week, other, '--epochs-per-file', '1', '--part', '0/2', '--variants', 'slide-3.0,pr,slide+2.25')
    drive(week, other, '--epochs-per-file', '1', '--part', '1/2', '--variants', 'slide+2.25,slide-3.0')
    drive(week, other, '--epochs-per-file', '1', '--part', '0/2', '--variants', 'slide+2.25,slide-3.0')
    for v in EXTRA:
        same_data(one / v, other / v)
        cfg, _, d, _, _ = plant_fit.load(str(one / v))
        assert cfg['control'] == cfg['variant'] == v and cfg['nbar'] == 0.0 and cfg['shift'] == float(v[5:])
        assert cfg['merger_truth'] == 'none' and plant_fit.truth(cfg)[1] == 'control'
        assert np.array_equal(d['k'], np.arange(n_sel)) and np.array_equal(d['gps'], a0['gps']) and not d['merger'].any()
        assert not np.array_equal(d['ln_lambda_bar'], a0['ln_lambda_bar'])


def test_extra_slides_windows_and_model_stage(week, tmp_path):
    """the windows stage with extra slides: one call per epoch holding the rows of every variant (the extras' slots after
    the six of the built-ins), win files of the extras named like the others' with the stub's window evidences of the
    slid data, those of the built-ins unchanged by the extras; the model stage (--glitch-model auto, a model per
    variant) gives each variant a call of its own in the order of the variants, the built-ins' with their slots and
    seed as before, each extra slide's with the epoch's rows and the seed of the merger stage, and files with ln_B_bar,
    the SHA-256 of its model and its shift"""
    out, base = tmp_path / 'w', tmp_path / 'base'
    win = drive(week, out, '--epochs-per-file', '2', '--stage', 'windows', '--variants', ','.join(EVERY), stub=WinStub())
    drive(week, base, '--epochs-per-file', '2', '--stage', 'windows', stub=WinStub())
    assert [c.size for c in win.calls] == [8 * 31, 8 * 30, 8 * 33]
    ep = strain.read_epoch(os.path.join(week, 'epoch_1266933768.npz'))
    var, rows, _ = analyse.variants(dict(seed=7, nbar_inj=0.3), ep, EXTRA, sampler())
    bins = plant.window_bins(3.0, 12.2)
    for v in EXTRA:
        ref = window_stub(var[v], np.arange(rows.size), 64)
        f = sorted(glob.glob(str(out / v / 'win_*.npz')))[0]
        assert os.path.basename(f) == os.path.basename(sorted(glob.glob(str(out / 'pr' / 'win_*.npz')))[0])
        with np.load(f) as fw:
            for I in 'HL':
                assert np.array_equal(fw['ln_window_' + I][:rows.size],
                                      ref['ln_window_' + I][:, :, bins].astype(np.float32)), (v, I)
    for v in analyse.VARIANTS:
        for f in glob.glob(str(base / v / 'win_*.npz')):
            with np.load(f) as x, np.load(out / v / os.path.basename(f)) as y:
                for key in x.files:
                    if key != 'meta':
                        assert np.array_equal(x[key], y[key], equal_nan=x[key].dtype.kind == 'f'), (v, key)
    shas = {}
    for i, v in enumerate(EVERY):
        meta = plant.read_meta(sorted(glob.glob(str(out / v / 'win_*.npz')))[0])
        shas[v] = make_model(out / v / 'glitch_model.json', plant.campaign_key(meta), q_scale=1.0 + i)
    stub = drive(week, out, '--epochs-per-file', '2', '--glitch-model', 'auto', '--variants', ','.join(EVERY))
    sel = {1266933768: np.setdiff1d(np.arange(NSEG), [3, 30]), 1266934296: np.setdiff1d(np.arange(NSEG), [8, 9, 20]),
           1266934560: np.arange(NSEG)}
    code = {'slide+2.25': 18432, 'slide-3.0': 24575}
    expect = []
    for g in sel:
        for v in analyse.VARIANTS:
            expect.append((v, plant.stream(7, g, 'pe'), analyse.VARIANTS.index(v) * NSEG + sel[g]))
        expect += [(v, plant.stream(7, g + code[v] * 2 ** 24, 'pe'), sel[g]) for v in EXTRA]
    assert len(stub.calls) == len(expect) == 24
    for (seed, rows), glitch, (v, seed_e, rows_e) in zip(stub.calls, stub.glitches, expect):
        model = plant.read_model(str(out / v / 'glitch_model.json'))[0]
        assert seed == seed_e and np.array_equal(rows, rows_e) and glitch['model'] == 'windows', v
        assert all(np.array_equal(glitch[I][p], model[I][p]) for I in 'HL' for p in ('f', 'q', 'pi_k')), v
    for v in EXTRA:
        files = glob.glob(str(out / v / 'plant_*.npz'))
        assert len(files) == 2
        for f in files:
            meta = plant.read_meta(f)
            assert meta['glitch_model_sha256'] == shas[v] and meta['shift'] == float(v[5:]) and meta['n_window'] == 64
            with np.load(f) as x:
                assert {'ln_B_bar', 'ln_c_H', 'state_L'} <= set(x.files)


def guard_extra(root, week, names, files=None, change=None):
    """next to guard_week: folders of the slides names with files covering k 0 .. n_selected - 1 in two halves (files:
    {name: [(first, end)]}) and the meta of analyse.campaign (change: {name: entries replaced}); k only"""
    n = sum(e['n_selected'] for e in analyse.selection(week))
    args = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7'])[1]
    for v in names:
        os.makedirs(os.path.join(root, v), exist_ok=True)
        for first, end in (files or {}).get(v, [(0, n // 2), (n // 2, n)]):
            plant.write_batch(os.path.join(root, v, 'plant_%010d_%010d.npz' % (first, end)),
                              dict(analyse.campaign(args, week, v), **(change or {}).get(v, {})),
                              dict(first=first, end=end), dict(k=np.arange(first, end)))
    return n


def frozen_rows(text):
    return [l for l in text.splitlines() if l.split(' ')[0] in summarize.ORDER]


def test_summarize_extra_slides(week, tmp_path, monkeypatch, capsys):
    """summarize --extra-slides: the other slide* folders are fitted after the frozen five, by shift, as rows marked
    (extra), and the recorded week is not opened; the frozen rows, their verdicts and the exit status of the frozen set
    are the same with and without the option when the extras pass, and without it the extras are not looked at; an
    extra that fails gives exit status 1 and is named, and with --recorded keeps the week closed; passing extras let it
    open"""
    guard_week(tmp_path, week)
    guard_extra(tmp_path, week, EXTRA)
    args = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7'])[1]
    cfgs = {v: analyse.campaign(args, week, v) for v in EVERY}
    order, extra = list(summarize.ORDER), ['slide-3.0', 'slide+2.25']                 # by shift
    log = []
    monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log, (), cfgs))
    assert summarize.main([str(tmp_path)]) == 0 and log == order
    plain = capsys.readouterr().out
    assert 'extra' not in plain
    for argv, failing, opened, code, says in (
            (['--extra-slides'], (), order + extra, 0, 'all pass'),
            (['--extra-slides', '--recorded'], (), order + extra + ['zero'], 0, 'all pass: opening the week'),
            (['--extra-slides'], ('slide+2.25',), order + extra, 1, 'extra slides that FAIL: slide+2.25: FAIL'),
            (['--extra-slides', '--recorded'], ('slide-3.0',), order + extra, 1, 'and extra slides checked again'),
            (['--extra-slides'], ('rev',), order + extra, 1, 'NOT all pass')):
        log = []
        monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log, failing, cfgs))
        assert summarize.main([str(tmp_path)] + argv) == code, (argv, failing)
        text = capsys.readouterr().out
        assert log == opened and says in text, (argv, failing, log, text)
        for v in extra:
            line = [l for l in text.splitlines() if l.startswith(v + ' (extra) ')]
            assert len(line) == 1 and ('FAIL' in line[0]) == (v in failing), (v, line)
        if not [v for v in failing if v in order]:
            assert frozen_rows(text) == frozen_rows(plain)
        assert any(l.startswith('zero ') for l in text.splitlines()) == ('zero' in log)
    # a folder without extra slides: said, and the frozen set as before
    plain_root = tmp_path / 'plain'
    guard_week(plain_root, week)
    log = []
    monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log, (), cfgs))
    assert summarize.main([str(plain_root), '--extra-slides']) == 0 and log == order
    assert 'none: no slide* folder besides those above' in capsys.readouterr().out


def test_summarize_extra_slides_refused(week, tmp_path, monkeypatch):
    """an extra slide that is incomplete (the second half missing), of another campaign (seed), whose files say they are
    another variant, or a folder slide_old, is an error that names it, before any fit, and only with --extra-slides
    (without it the frozen set is fitted and the folder not opened); the recorded folder is not opened for the extras
    (its files here are empty)"""
    n = guard_week(tmp_path / 'ok', week)
    h = n // 2
    for name, kw, match in (
            ('partial', dict(files={'slide+2.25': [(0, h)]}), r'slide\+2\.25: .*missing.*k %d to %d' % (h, n - 1)),
            ('seed', dict(change={'slide-3.0': dict(seed=8)}), r'slide-3\.0: .* another campaign.*seed'),
            ('variant', dict(change={'slide-3.0': dict(variant='slide+2.25', control='slide+2.25')}),
             r"slide-3\.0: .* says it is variant 'slide\+2\.25'")):
        root = tmp_path / name
        guard_week(root, week)
        guard_extra(root, week, EXTRA, **kw)
        refused(root, monkeypatch, '--extra-slides', '--recorded')
        msg = refused(root, monkeypatch, '--extra-slides')
        assert re.search(match, msg), (name, msg)
        log = []
        monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log))
        assert summarize.main([str(root)]) == 0 and log == list(summarize.ORDER) and 'zero' not in log
    old = tmp_path / 'old'
    guard_week(old, week)
    os.makedirs(old / 'slide_old')
    assert 'slide_old' in refused(old, monkeypatch, '--extra-slides')
    log = []
    monkeypatch.setattr(summarize.plant_fit, 'report', fake_report(log))
    assert summarize.main([str(old)]) == 0 and 'zero' not in log


def test_summarize_extra_slide_criteria_real_fit(week, toy_week, tmp_path, capsys):
    """with the real fits on toy evidences (no mergers): an extra slide of the same null passes the criteria of the
    slid weeks, its row has the numbers of plant_fit.report, the frozen rows are the same lines as without the option;
    an extra slide whose data hold mergers (nbar 0.05 but the meta of a control) fails the criterion of nbar consistent
    with 0 and gives exit status 1, naming it"""
    out, _ = toy_week
    root = str(tmp_path / 'toy')
    shutil.copytree(out, root, ignore=shutil.ignore_patterns('zero', 'inj_wrong_rate'))
    write_toy_variant(os.path.join(root, 'slide+2.25'), week, 'slide+2.25', 0.0, 0.10, seed=20261201)
    assert summarize.main([root, '--rho-min', '5']) == 0
    plain = capsys.readouterr().out
    assert summarize.main([root, '--rho-min', '5', '--extra-slides']) == 0
    text = capsys.readouterr().out
    assert frozen_rows(text) == frozen_rows(plain)
    r = plant_fit.report(os.path.join(root, 'slide+2.25'), rho_mins=[5.0])
    line = [l for l in text.splitlines() if l.startswith('slide+2.25 (extra) ')][0]
    J = r['fits']['joint']
    assert '%s+-%s' % (plant_fit._f(J['nbar'], '%.3g'), plant_fit._f(J['err']['nbar'], '%.2g')) in line and 'PASS' in line
    write_toy_variant(os.path.join(root, 'slide-3.0'), week, 'slide-3.0', 0.05, 0.10, seed=20261202)
    assert summarize.main([root, '--rho-min', '5', '--extra-slides']) == 1
    text = capsys.readouterr().out
    assert 'extra slides that FAIL: slide-3.0: FAIL' in text and 'NOT all pass' in text
    assert frozen_rows(text) == frozen_rows(plain)
    print('\n' + text)


# ----------------------------------------------------------------------------------------------- slides_mean

SLIDES = ('slide+0.5', 'slide-1.75') + EXTRA
N_SLIDE, S_LN, RHO_SLIDE, DELTA = 6400, 0.5, 0.9, 0.02


def slide_lnB(n, seed, n_slide=len(SLIDES), s=S_LN, rho=RHO_SLIDE, delta=DELTA):
    """ln Bbar (n_slide, n) of slides that share the single-detector data: B_ij = (1 + delta) exp(s z_ij - s^2 / 2),
    z_ij = rho z_i + sqrt(1 - rho^2) e_ij (z_i common to the slides of segment i), so that E(B - 1) = delta and two slides
    of one segment have the log-correlation rho^2"""
    rng = np.random.default_rng(seed)
    z = rho * rng.standard_normal(n)[None] + np.sqrt(1 - rho ** 2) * rng.standard_normal((n_slide, n))
    return s * z - 0.5 * s ** 2 + np.log1p(delta)


def analytic_se(n, n_slide=len(SLIDES), s=S_LN, rho=RHO_SLIDE):
    """the standard error of the mean of B - 1 over the n_slide n pairs: the segments are independent; the pairs of one
    segment have variance (1 + delta)^2 (exp(s^2) - 1) and covariance (1 + delta)^2 (exp(s^2 rho^2) - 1)"""
    v, c = [(1 + DELTA) ** 2 * (np.exp(s ** 2 * r) - 1.0) for r in (1.0, rho ** 2)]
    return float(np.sqrt((v + (n_slide - 1) * c) / n_slide / n))


def write_slides(root, week, lnB, names=SLIDES, stat='window', veto=None, others=(), halves=2):
    """selection.txt of the n segments and a folder per slide with ln_B_bar (or ln_lambda_bar: stat 'gauss') of lnB[i], k,
    wmax_H and wmax_L (veto {name: {k: wmax}} raised above 72 at those k), in `halves` files, with the meta of
    analyse.campaign; folders of `others` (pr, rev, inj, zero) with unreadable files"""
    n = lnB.shape[1]
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'selection.txt'), 'w') as fh:
        fh.write('epoch_1266933768.npz 1266933768 %d %d 0\n' % (n, n))
    args = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7'])[1]
    rng = np.random.default_rng(5)
    cuts = np.linspace(0, n, halves + 1).astype(int)
    for i, v in enumerate(names):
        os.makedirs(os.path.join(root, v), exist_ok=True)
        w = {I: rng.exponential(5.0, n) for I in 'HL'}
        for k, x in (veto or {}).get(v, {}).items():
            w['H'][k] = x
        for a, b in zip(cuts[:-1], cuts[1:]):
            per = dict(k=np.arange(a, b), wmax_H=w['H'][a:b], wmax_L=w['L'][a:b],
                       **{'ln_B_bar' if stat == 'window' else 'ln_lambda_bar': lnB[i, a:b]})
            plant.write_batch(os.path.join(root, v, 'plant_%010d_%010d.npz' % (a, b)),
                              analyse.campaign(args, week, v, '%064x' % (i + 1) if stat == 'window' else None),
                              dict(first=int(a), end=int(b)), per)
    for v in others:
        os.makedirs(os.path.join(root, v), exist_ok=True)
        with open(os.path.join(root, v, 'plant_%010d_%010d.npz' % (0, n)), 'wb') as fh:
            fh.write(b'not an npz file')


def block_jackknife(x_by_slide, k_by_slide, n, G):
    """the delete-one-block jackknife of the mean over the pairs, by hand: (mean, se)"""
    pairs = np.concatenate(x_by_slide)
    blk = np.concatenate([(k * G) // n for k in k_by_slide])
    reps = np.array([pairs[blk != b].mean() for b in range(G)])
    return pairs.mean(), np.sqrt((G - 1) / G * np.sum((reps - reps.mean()) ** 2))


def test_slides_mean_numbers(week, tmp_path, capsys):
    """slides_mean on 4 slides of 6400 segments that share a common component (log-correlation 0.81 between the pairs of a
    segment; B has the known mean 1.02): per slide the kept segments (those with wmax above 72 in H set aside at known k),
    mean of B - 1, standard error, the half-sum count, the largest ln B with its k and nbar with its error are those of
    the direct computation and plant_fit.nbar_fit; for all slides the mean of B - 1 over the 25600 pairs and its block
    jackknife error equal a hand-written jackknife (32 blocks of consecutive k, the same blocks in every slide);
    the mean is within 3 jackknife errors of 0.02, and the jackknife error is within 30 percent of the exact one of the
    correlated pairs (analytic_se) and at least 1.4 times the naive error of the pairs; the pooled nbar is the nbar_fit of
    every B of every slide with a jackknife error within 30 percent of the spread of the pooled nbar over 200 replicas;
    the mixture's files (no ln_B_bar) give the same on ln_lambda_bar, saying so; the output prints each number"""
    lnB = slide_lnB(N_SLIDE, 11)
    veto = {'slide+0.5': {10: 100.0, 4000: 100.0}, 'slide+2.25': {7: 90.0}}
    root = tmp_path / 'week'
    write_slides(root, week, lnB, veto=veto)
    r = slides_mean.analyse(str(root))
    capsys.readouterr()
    xs, ks = [], []
    lines = []
    for i, v in enumerate(SLIDES):
        keep = np.ones(N_SLIDE, bool)
        keep[list(veto.get(v, {}))] = False
        ln, k = lnB[i][keep], np.arange(N_SLIDE)[keep]
        x = np.exp(ln) - 1.0
        s = r['slides'][v]
        fit = plant_fit.nbar_fit(ln)
        j = int(np.argmax(ln))
        assert s['kept'] == keep.sum() and s['aside'] == (~keep).sum()
        assert abs(s['mean'] - x.mean()) < 1e-12 and abs(s['se'] - x.std(ddof=1) / np.sqrt(x.size)) < 1e-12
        assert s['half'] == plant_fit.heavy_tail(ln) and s['max'] == ln[j] and s['k_max'] == k[j]
        assert abs(s['nbar'] - fit['nbar']) < 1e-12 and abs(s['err'] - fit['err']['nbar']) < 1e-12
        xs.append(x)
        ks.append(k)
        lines.append('%s: %d kept, mean %+.4f +- %.4f, half-sum %d, max ln B %.3f at k %d, nbar %.4f +- %.4f' % (
            v, s['kept'], s['mean'], s['se'], s['half'], s['max'], s['k_max'], s['nbar'], s['err']))
    P = r['pooled']
    mean, jse = block_jackknife(xs, ks, N_SLIDE, 32)
    assert P['pairs'] == sum(x.size for x in xs) and P['blocks'] == 32
    assert abs(P['mean'] - mean) < 1e-12 and abs(P['jack_mean'] - jse) < 1e-10 * jse
    naive = np.concatenate(xs).std(ddof=1) / np.sqrt(P['pairs'])
    exact = analytic_se(N_SLIDE)
    lines.append('pooled: mean %+.4f +- %.4f (jackknife), naive %.4f, exact %.4f; nbar %.4f +- %.4f (jackknife), '
                 '%.4f (curvature)' % (P['mean'], P['jack_mean'], naive, exact, P['nbar'], P['jack_nbar'], P['curv_nbar']))
    print('\n' + '\n'.join(lines))
    assert abs(P['mean'] - DELTA) < 3 * P['jack_mean'], lines[-1]
    assert 0.7 < P['jack_mean'] / exact < 1.3 and P['jack_mean'] > 1.4 * naive, lines[-1]
    pooled = np.concatenate([lnB[i][[k for k in range(N_SLIDE) if k not in veto.get(v, {})]] for i, v in enumerate(SLIDES)])
    assert abs(P['nbar'] - plant_fit.nbar_fit(pooled)['nbar']) < 1e-12
    reps = [plant_fit.nbar_fit(slide_lnB(N_SLIDE, 1000 + i).ravel())['nbar'] for i in range(200)]
    assert 0.7 < P['jack_nbar'] / np.std(reps, ddof=1) < 1.3, (P['jack_nbar'], np.std(reps, ddof=1))
    assert P['curv_nbar'] < 0.7 * P['jack_nbar']                 # the curvature error ignores the shared data
    text = '\n'.join(r['lines'])
    for v in SLIDES:
        assert v in text
    assert 'Bbar of the window model (ln_B_bar)' in text and 'jackknife over 32 blocks' in text
    assert slides_mean.main([str(root), '--blocks', '16']) == 0
    assert 'jackknife over 16 blocks' in capsys.readouterr().out
    gauss = tmp_path / 'gauss'
    write_slides(gauss, week, lnB, stat='gauss', veto=veto)
    g = slides_mean.analyse(str(gauss))
    assert g['stat'] == 'gauss' and 'only the Gaussian likelihood is used' in '\n'.join(g['lines'])
    assert abs(g['pooled']['mean'] - P['mean']) < 1e-12 and g['slides']['slide+0.5']['kept'] == N_SLIDE - 2


def test_slides_mean_reads_only_slides_and_refuses_the_rest(week, tmp_path, monkeypatch):
    """the recorded week, pr, rev and inj are never opened (their files here are not even npz, and np.load is watched);
    refused, naming it and before any number: the folder of one variant given as the week, a folder slide_old, a folder
    without slide*, an incomplete slide, a slide of another campaign, a slide that is an injection run (nbar 0.3), slides
    of the window model with slides of the mixture; the slide folders of an earlier good run are untouched"""
    lnB = slide_lnB(N_SLIDE, 12)
    root = tmp_path / 'week'
    write_slides(root, week, lnB, others=('zero', 'pr', 'rev', 'inj'))
    opened = []
    real = np.load
    monkeypatch.setattr(np, 'load', lambda path, *a, **kw: opened.append(str(path)) or real(path, *a, **kw))
    r = slides_mean.analyse(str(root))
    assert set(r['slides']) == set(SLIDES) and opened
    assert not [p for p in opened if re.search(r'/(zero|pr|rev|inj)/', p)], opened
    with pytest.raises(SystemExit, match='give the folder of the week'):
        slides_mean.analyse(str(root / 'slide+0.5'))
    cases = []
    bad = tmp_path / 'old'
    shutil.copytree(root, bad)
    os.makedirs(bad / 'slide_old')
    cases.append((bad, 'slide_old'))
    empty = tmp_path / 'empty'
    write_slides(empty, week, lnB[:0].reshape(0, N_SLIDE), names=(), others=('pr',))
    cases.append((empty, 'no folder slide'))
    part = tmp_path / 'part'
    shutil.copytree(root, part)
    os.remove(sorted(glob.glob(str(part / 'slide-3.0' / 'plant_*.npz')))[1])
    cases.append((part, r'slide-3\.0: .*missing.*k 3200 to 6399'))
    args = analyse.parse(['x', 'cpu', week, '--out', 'x', '--seed', '7'])[1]
    for tag, change, match in (('seed', dict(seed=8), r'slide\+2\.25: .*another campaign.*seed'),
                               ('inj', dict(nbar=0.3, merger_truth='injections'), r'not a time slide control')):
        d = tmp_path / tag
        shutil.copytree(root, d)
        for f in glob.glob(str(d / 'slide+2.25' / 'plant_*.npz')):
            with real(f) as z:
                arrays = {key: z[key] for key in z.files}
            meta = json.loads(str(arrays.pop('meta')))
            np.savez(f, meta=np.array(json.dumps(dict(meta, **change))), **arrays)
        cases.append((d, match))
    mixed = tmp_path / 'mixed'
    shutil.copytree(root, mixed)
    for f in glob.glob(str(mixed / 'slide-3.0' / 'plant_*.npz')):
        with real(f) as z:
            arrays = {key: z[key] for key in z.files if key != 'ln_B_bar'}
        arrays['ln_lambda_bar'] = arrays['wmax_H']
        np.savez(f, **arrays)
    cases.append((mixed, r'window model \(.*\) and of the mixture \(slide-3\.0\) cannot be pooled'))
    for d, match in cases:
        with pytest.raises(SystemExit, match=match):
            slides_mean.analyse(str(d))
    assert slides_mean.analyse(str(root))['pooled']['pairs'] == 4 * N_SLIDE


def test_slide_names_agree():
    """the form of a slide name is the same in analyse.py and summarize.py (slides_mean.py uses summarize's)"""
    assert summarize.SLIDE.pattern == analyse.SLIDE.pattern
    assert slides_mean.summarize.SLIDE.pattern == analyse.SLIDE.pattern


# ----------------------------------------------------------------------------------------------- own noise curves


def make_unequal_epoch(gps_epoch, seed, cond_seed=11, **flags):
    """an epoch as strain.condition writes it with step 5: data on the curves sig2_ref_H (1.7 times the H design curve)
    and sig2_ref_L (L better below 100 Hz), equalized by strain.equalize with the draws of condition, then the notch
    noise; returns (epoch, the data before step 5)"""
    ep = make_epoch(gps_epoch, seed, **flags)
    f, notched = ep['f'], ep['notched']
    ref = {'H': np.where(notched, 0.0, 1.7 * T * S('H', f) / 2.0)}
    ref['L'] = np.where(notched, 0.0, ref['H'] * np.where(f < 100.0, 0.55, 1.1))
    rng = np.random.default_rng(seed + 1)
    pre = {I: np.sqrt(ref[I] / 2) * (rng.standard_normal((NSEG, f.size)) + 1j * rng.standard_normal((NSEG, f.size)))
           for I in 'HL'}
    cond = np.random.default_rng([cond_seed, int(round(gps_epoch * 1000))])
    d_H, d_L, sig2 = strain.equalize(pre['H'], pre['L'], ref['H'], ref['L'], cond)
    big = 1e10 * sig2.max()
    sig2 = np.where(notched, big, sig2)
    d = {'H': d_H, 'L': d_L}
    for I in 'HL':
        g = cond.standard_normal((2, NSEG, int(notched.sum())))
        d[I][:, notched] = np.sqrt(0.5 * big) * (g[0] + 1j * g[1])
    ep.update(d_H=d['H'].astype(np.complex64), d_L=d['L'].astype(np.complex64), sig2=sig2, sig2_ref_H=ref['H'],
              sig2_ref_L=ref['L'], seed=cond_seed, equal_noise=True)
    return ep, pre


def write_unequal_week(folder):
    """the four epochs of write_week, conditioned with step 5 (make_unequal_epoch)"""
    os.makedirs(folder, exist_ok=True)
    lines = ['# synthetic week, unequal curves\n']
    for i, e in enumerate(EPOCHS[::-1]):
        ep, _ = make_unequal_epoch(e['gps'], 100 + i, loud_H=e.get('loud_H', ()), loud_L=e.get('loud_L', ()),
                                   event=e.get('event', ()), usable=e.get('usable', True))
        name = 'epoch_%d.npz' % e['gps']
        strain.write_epoch(os.path.join(folder, name), ep)
        lines.append('%s %d %d 127 127 %d 0 0 33 33 %d 0.8\n' % (name, e['gps'], NSEG, ep['event'].sum(),
                                                                 int(ep['usable'])))
    with open(os.path.join(folder, 'manifest.txt'), 'w') as fh:
        fh.writelines(lines)
    return folder


@pytest.fixture(scope='module')
def uweek(tmp_path_factory):
    return write_unequal_week(str(tmp_path_factory.mktemp('uweek') / 'segments'))


class UStub(Stub):
    """the Stub for segments on two curves: the same arrays with sig2_I in place of sig2; records the curves"""

    def __init__(self):
        super().__init__()
        self.curves = []

    def pe(self, seg, seed, rows, glitch=True):
        self.curves.append({I: np.asarray(seg['sig2_' + I]).copy() for I in 'HL'})
        one = {I: dict(seg, sig2=seg['sig2_' + I]) for I in 'HL'}
        res = {I: Stub.pe(self, one[I], seed, rows, glitch) for I in 'HL'}
        del self.calls[-1], self.glitches[-1]
        out = dict(res['H'], asym_outside=2)
        for key in ('ln_glitch_L', 'wmax_L') + (('ln_c_L', 'state_L') if isinstance(glitch, dict) else ()):
            out[key] = res['L'][key]
        return out


def test_unequal_variants(uweek):
    """variants with own curves: every variant's data are those of strain.own_curves on the selected rows, the data
    before step 5 to the complex64 rounding of the file off the notches, the file's notch noise in them; sig2_H, sig2_L
    the curves of own_curves (sig2_ref_I off the notches) and no sig2; the controls keep them; the injections' optimal
    SNR uses the curve of each detector; without own curves the variants are those of the equalized data"""
    ep = strain.read_epoch(os.path.join(uweek, 'epoch_1266934296.npz'))
    s = plant.Sampler(plant.population(analyse.ZMIN), plant.MDET)
    var, rows, inj = analyse.variants(dict(seed=7, nbar_inj=0.5), ep, analyse.VARIANTS, s, own=True)
    _, pre = make_unequal_epoch(1266934296, 101, loud_H=(20,), event=(8, 9))
    own = strain.own_curves(ep)
    keep = ~ep['notched']
    for I in 'HL':
        assert np.array_equal(var['zero']['d_' + I], own[I][0][rows]) and 'sig2' not in var['zero']
        err = np.abs(var['zero']['d_' + I][:, keep] - pre[I][rows][:, keep]) / np.sqrt(ep['sig2_ref_' + I][keep])
        assert err.max() < 1e-5, err.max()
        assert np.array_equal(var['zero']['d_' + I][:, ~keep], np.asarray(ep['d_' + I])[rows][:, ~keep].astype(complex))
        assert np.array_equal(var['zero']['sig2_' + I][keep], ep['sig2_ref_' + I][keep])
        for v in analyse.VARIANTS:
            assert np.array_equal(var[v]['sig2_' + I], own[I][1]), v
    h = {I: var['inj']['d_' + I] - var['slide+0.5']['d_' + I] for I in 'HL'}
    snr = np.sqrt(sum(2.0 * np.sum(np.abs(h[I]) ** 2 / own[I][1], axis=1) for I in 'HL'))
    assert inj['merger'].any() and np.allclose(inj['snr'], snr, rtol=1e-12, atol=0)
    eq, rows_eq, _ = analyse.variants(dict(seed=7, nbar_inj=0.5), ep, ('zero',), s)
    assert np.array_equal(rows_eq, rows) and np.array_equal(eq['zero']['sig2'], ep['sig2'])


def test_unequal_driver(uweek, tmp_path):
    """--unequal with two store folders (asym 0 and -0.6), --zmin 0 and --zmin-cut: pe gets each epoch's own-curve
    data and curves; the files record noise 'own curves', tables_asym, zmin 0, zmin_cut and asym_tol 0.05 (the
    default), and per segment the number of templates pe read from an end store (asym_outside); plant_fit reads them;
    the injections are drawn from the whole population (some below z = 0.99); a run of the equalized data into the
    same folder stops; the campaign of the equalized data with one store records none of these"""
    from test_store import write_store
    stores = '%s,%s' % (write_store(str(tmp_path / 's0')), write_store(str(tmp_path / 's6'), asym=-0.6))
    out = tmp_path / 'u'
    stub = UStub()
    assert analyse.main([stores, 'cpu', uweek, '--out', str(out), '--seed', '7', '--unequal', '--zmin', '0',
                         '--zmin-cut', '--epochs-per-file', '2'], factory=stub.factory) == 0
    gps = [1266933768, 1266934296, 1266934560]
    assert len(stub.calls) == 3
    for curves, g in zip(stub.curves, gps):
        want = strain.own_sig2(strain.read_epoch(os.path.join(uweek, 'epoch_%d.npz' % g)))
        assert all(np.array_equal(curves[I], want[I]) for I in 'HL')
    for v in analyse.VARIANTS:
        cfg, _, a, _, _ = plant_fit.load(str(out / v))
        assert cfg['noise'] == 'own curves' and cfg['tables_asym'] == [-0.6, 0.0] and cfg['zmin'] == 0.0
        assert cfg['zmin_cut'] is True and cfg['asym_tol'] == 0.05 and a['k'].size == 94
        assert np.array_equal(a['asym_outside'], np.full(94, 2))
        if v == 'inj':
            assert a['merger'].any() and np.nanmin(a['z']) < analyse.ZMIN
    with pytest.raises(SystemExit, match='another campaign.*noise'):
        analyse.main([stores, 'cpu', uweek, '--out', str(out), '--seed', '7', '--zmin', '0', '--zmin-cut',
                      '--epochs-per-file', '2'], factory=Stub().factory)
    plain = analyse.campaign(analyse.parse(['x', 'cpu', uweek, '--out', 'x'])[1], uweek, 'pr')
    assert not {'noise', 'tables_asym', 'asym_tol', 'zmin_cut'} & set(plain)


def test_zmin_reuses_the_window_model(week, tmp_path):
    """the window models of a windows stage at the default --zmin serve a merger stage at --zmin 0: with auto for every
    variant but inj (whose injections are drawn from the population, so its data change with zmin: refused), with a
    given file for every variant; the merger files record zmin 0 and lie next to the window files of the other zmin"""
    out = tmp_path / 'm'
    shas = windows_week(week, out)
    controls_ = 'pr,rev,slide+0.5,slide-1.75,zero'
    stub = drive(week, out, '--epochs-per-file', '2', '--glitch-model', 'auto', '--zmin', '0', '--variants', controls_)
    assert len(stub.calls) == 15
    for v in controls_.split(','):
        for f in glob.glob(str(out / v / 'plant_*.npz')):
            meta = plant.read_meta(f)
            assert meta['zmin'] == 0.0 and meta['glitch_model_sha256'] == shas[v]
        assert glob.glob(str(out / v / 'win_*.npz'))
    with pytest.raises(SystemExit, match='inj/glitch_model.json: the model was fitted on another campaign.*zmin'):
        drive(week, out, '--epochs-per-file', '2', '--glitch-model', 'auto', '--zmin', '0', '--variants', 'inj')
    given = tmp_path / 'given'
    drive(week, given, '--epochs-per-file', '2', '--glitch-model', str(out / 'slide+0.5' / 'glitch_model.json'),
          '--zmin', '0.5')
    assert {plant.read_meta(f)['zmin'] for f in glob.glob(str(given / '*' / 'plant_*.npz'))} == {0.5}


def test_analysis_wiring_stores_and_zmin(week, monkeypatch):
    """make_analysis passes the store (one, or the list) and, only when given, zmin and asym_tol to pe_redshift"""
    import exmarg.search as search
    seen = []
    monkeypatch.setattr(search, 'pe_redshift', lambda seg, bank, store, **kw: seen.append((store, kw)))
    ep = strain.read_epoch(os.path.join(week, 'epoch_1266934560.npz'))
    seg = analyse.variants(dict(seed=7, nbar_inj=0.0), ep, ('pr',), None)[0]['pr']
    analyse.make_analysis(['a', 'b'], 'cpu', zmin=0.5, asym_tol=0.05)(seg)(seg, 17, [0, 1])
    analyse.make_analysis('a', 'cpu')(seg)(seg, 17, [0, 1])
    (st1, kw1), (st2, kw2) = seen
    assert st1 == ['a', 'b'] and kw1['zmin'] == 0.5 and kw1['asym_tol'] == 0.05 and st2 == 'a'
    assert 'zmin' not in kw2 and 'asym_tol' not in kw2
    assert {k: v for k, v in kw1.items() if k not in ('zmin', 'asym_tol')} == kw2


def test_interp_recorded_and_passed(week, tmp_path, monkeypatch):
    """analyse.py: the files of the merger stage record tables 'unbiased' and interp 'linear' by default and, with
    --interp log, no interp entry (as the files written before 2026-11-18, read with ln Lambda linear between the
    nodes), the meta being otherwise the same; --as-built records tables 'as built'.  A run with the default into a
    folder of files without the entry stops as another campaign (differs in interp), as does a run with --interp log
    into the folder of a default run; a run with --interp log into the first writes the rest of one campaign, which
    plant_fit.load reads.  make_analysis gives interp to pe_redshift and to pe_redshift_epochs ('linear' when none is
    given), and main gives --interp to make_analysis.  Fires on: a missing entry taken as 'linear'"""
    import exmarg.search as search
    from test_store import write_store
    lin, log = tmp_path / 'lin', tmp_path / 'log'
    args = ('--epochs-per-file', '1', '--variants', 'pr,inj')
    drive(week, lin, *args, '--part', '0/2')
    drive(week, log, *args, '--part', '0/2', '--interp', 'log')
    for v in ('pr', 'inj'):
        a, b = (plant.read_meta(sorted(glob.glob(str(d / v / 'plant_*.npz')))[0]) for d in (lin, log))
        assert a['tables'] == 'unbiased' and a.pop('interp') == 'linear' and 'interp' not in b
        a.pop('run'), b.pop('run')
        assert a == b
    with pytest.raises(SystemExit, match='another campaign .differs in interp.'):
        drive(week, log, *args, '--part', '1/2')
    with pytest.raises(SystemExit, match='another campaign .differs in interp.'):
        drive(week, lin, *args, '--part', '1/2', '--interp', 'log')
    drive(week, log, *args, '--part', '1/2', '--interp', 'log')
    cfg, _, arrays, _, _ = plant_fit.load(str(log / 'pr'))
    assert 'interp' not in cfg and cfg['tables'] == 'unbiased' and np.array_equal(arrays['k'], np.arange(94))
    built = analyse.campaign(analyse.parse(['x', 'cpu', week, '--out', 'x', '--as-built'])[1], week, 'pr')
    assert built['tables'] == 'as built' and built['interp'] == 'linear'
    seen = []
    monkeypatch.setattr(search, 'pe_redshift', lambda seg, bank, store, **kw: seen.append(kw))
    monkeypatch.setattr(search, 'pe_redshift_epochs', lambda epochs, store, **kw: seen.append(kw))
    ep = strain.read_epoch(os.path.join(week, 'epoch_1266934560.npz'))
    seg = analyse.variants(dict(seed=7, nbar_inj=0.0), ep, ('pr',), None)[0]['pr']
    for interp, kw in (('linear', {}), ('log', dict(interp='log'))):
        pe = analyse.make_analysis('a', 'cpu', **kw)(seg)
        pe(seg, 17, [0, 1]), pe.batch([pe.epoch(seg, 17, [0, 1])])
        assert seen[-2]['interp'] == seen[-1]['interp'] == interp
    given = []
    monkeypatch.setattr(analyse.plant, 'load_stores', lambda a: 'the stores')
    monkeypatch.setattr(analyse, 'make_analysis', lambda *a: given.append(a) or Stub().factory)
    store = write_store(str(tmp_path / 'store'))
    for interp, extra in (('linear', []), ('log', ['--interp', 'log'])):
        assert analyse.main([store, 'cpu', week, '--out', str(tmp_path / ('wired_' + interp)), '--seed', '7',
                             '--variants', 'pr'] + extra) == 0
        assert given[-1][0] == 'the stores' and given[-1][-1] == interp and len(given[-1]) == 8
