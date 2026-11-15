"""the analysis of the conditioned week without the tables (scripts/real_data/analyse.py, summarize.py, and plant_fit.py
on real-data directories): the variants of a synthetic epoch file (strain.write_epoch), the selection and k across
--part splits and --variants subsets with a stub in place of pe_redshift, plant_fit on directories without a known
truth, and the guard that keeps the recorded week closed until the controls pass"""
import glob
import importlib.util
import json
import os

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
              u_ratio_60=0.8, **flags)
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
