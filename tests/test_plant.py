"""the planted-truth campaign (scripts/validate/plant.py and plant_fit.py) without the tables: the merger sampler
against the population of the analysis, segment k the same data in every split of a campaign, the random streams
independent, the driver end to end with a stub in place of pe_redshift, the coverage diagnostic, and the bookkeeping of
plant_fit on the toy evidences of tests/test_mixture.py against direct calls of exmarg.search.mixture"""
import glob
import importlib.util
import os

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import chi2

from exmarg.priors import population_m1_z
from exmarg.search import RHO_EDGES, mixture
from exmarg.search.population import ln_lambda_bar, population_weights
from test_mixture import EDGES, LAWS, Toy, law

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts', 'validate', name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plant, plant_fit = _script('plant'), _script('plant_fit')


def config(*extra):
    """a campaign of plant.py from command-line arguments (store and out unused)"""
    return plant.campaign(plant.parser().parse_args(['store', 'cpu', '--n', '1', '--out', 'x'] + list(extra)))


MIXED = ('--nbar', '0.5', '--eps-H', '0.5', '--eps-L', '0.5', '--glitch-kind', 'mixed')


def grid():
    g = plant.GRID
    return np.geomspace(*g['m1_range'], g['nm1']), np.linspace(0.02, g['zmax'], g['nz'])


def cells(x):
    """edges of the cells np.gradient gives the points x"""
    return np.concatenate([[x[0] - 0.5 * (x[1] - x[0])], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]])


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')


# ----------------------------------------------------------------------------------------------- the sampler


def test_sampler_matches_population():
    """20000 mergers of plant.mergers (nbar = 1, one stream per segment) against the restricted population as the
    analysis weighs it (population_weights with plant.population() on the 150 x 150 grid of pe_redshift), on bins of
    15 x 15 cells of that grid, bins expecting fewer than 5 draws pooled into one: chi-square p-value above 1e-3.  Every
    draw has z >= 1 and 20 <= 2 m1 (1 + z) <= 400.  The analysis weights vanish below z = 1 and are the weights of the
    unrestricted population, renormalized, above"""
    n = 20000
    pop = plant.population()
    present, m1, z, y = plant.mergers(dict(seed=11, nbar=1.0), range(n), plant.Sampler(pop))
    Md = 2 * m1 * (1 + z)
    assert present.all() and z.min() >= 1.0 and Md.min() >= 20.0 and Md.max() <= 400.0, (z.min(), Md.min(), Md.max())
    gm, gz = grid()
    w = population_weights(gm, gz, plant.MDET, pop)
    full = population_weights(gm, gz, plant.MDET, population_m1_z())
    hi = gz >= 1.0
    assert np.all(w[:, ~hi] == 0) and np.allclose(w[:, hi], full[:, hi] / full[:, hi].sum(), rtol=1e-12, atol=0)
    group = np.arange(0, gm.size, 15)
    em, ez = cells(gm)[np.r_[group, gm.size]], cells(gz)[np.r_[group, gz.size]]
    expect = n * np.add.reduceat(np.add.reduceat(w, group, axis=0), group, axis=1)
    got = np.histogram2d(m1, z, bins=[em, ez])[0]
    big = expect >= 5
    e = np.r_[expect[big], expect[~big].sum()]
    o = np.r_[got[big], got[~big].sum()]
    stat, dof = np.sum((o - e) ** 2 / e), e.size - 1
    p = chi2.sf(stat, dof)
    print('\nchi-square %.1f for %d degrees of freedom, p = %.3f; %d draws, %d bins' % (stat, dof, p, n, e.size))
    assert got.sum() == n and p > 1e-3, (stat, dof, p)


# ----------------------------------------------------------------------------------------------- the splits


@pytest.mark.parametrize('control', plant.CONTROLS)
def test_split_reproducible(control):
    """segment k is the same data, bit for bit in both detectors and in every truth array, whether simulated in one
    batch from 0 or in the splits [3, 9) and [3, 5) + [5, 9) (mergers and glitches of every kind in about half the
    segments, each control); every segment differs from every other"""
    cfg = config(*MIXED, '--control', control)
    s = plant.Sampler(plant.population())
    ref, tref = plant.simulate_batch(cfg, 0, 9, s)
    assert 0 < tref['merger'].sum() < 9 and 0 < tref['glitch_H'].sum() < 9 and 0 < tref['glitch_L'].sum() < 9
    assert np.array_equal(tref['k'], np.arange(9))
    for split in ([(3, 9)], [(3, 5), (5, 9)]):
        parts = [plant.simulate_batch(cfg, a, b, s) for a, b in split]
        for I in 'HL':
            assert np.array_equal(np.concatenate([p[0]['d_' + I] for p in parts]), ref['d_' + I][3:]), (split, I)
        for key in tref:
            assert same(np.concatenate([p[1][key] for p in parts]), tref[key][3:]), (split, key)
    for I in 'HL':
        d = ref['d_' + I]
        assert all(not np.array_equal(d[i], d[j]) for i in range(9) for j in range(i))


def test_streams_independent():
    """no two (seed, k, use) share a stream integer; pure noise (nbar = eps = 0) of segment k + 1 of seed s differs from
    segment k of seed s + 1 and from every segment of seed s + 1; the noise does not depend on nbar or eps (equal where
    neither a merger nor a glitch was added), the merger parameters not on nbar, and presence, SNR and time of the
    glitches not on the kind or on the eps of the other detector; over 1000 segments with nbar = eps_H = eps_L = 1/2
    the presences of a merger, a glitch in H and a glitch in L are pairwise uncorrelated, as are the glitch times in H
    and L (|r| < 4 / sqrt(1000))"""
    ids = [plant.stream(s, k, u) for s in (0, 1, 2, 2 ** 20 - 1) for k in (0, 1, 2, 999, 2 ** 40 - 1)
           for u in plant.USES]
    assert len(set(ids)) == len(ids) and max(ids) < 2 ** 63
    s = plant.Sampler(plant.population())
    quiet = {seed: plant.simulate_batch(config('--nbar', '0', '--seed', str(seed)), 0, 6, s)[0]['d_H']
             for seed in (1, 2)}
    assert all(not np.array_equal(quiet[1][i], quiet[2][j]) for i in range(6) for j in range(6))
    seg, tr = plant.simulate_batch(config(*MIXED), 0, 12, s)
    seg0, _ = plant.simulate_batch(config('--nbar', '0'), 0, 12, s)
    for I in 'HL':
        clean = ~tr['merger'] & ~tr['glitch_' + I]
        assert clean.any() and np.array_equal(seg['d_' + I][clean], seg0['d_' + I][clean])
    a = plant.mergers(config('--nbar', '0.3'), range(200), s)
    b = plant.mergers(config('--nbar', '0.7'), range(200), s)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) and np.all(a[0] <= b[0])
    _, t1 = plant.simulate_batch(config('--nbar', '0.5', '--eps-H', '0.5', '--eps-L', '0.5'), 0, 30, s)
    _, t2 = plant.simulate_batch(config('--nbar', '0.5', '--eps-H', '0.5', '--glitch-kind', 'ringdown'), 0, 30, s)
    assert t2['glitch_H'].any() and not t2['glitch_L'].any()
    for key in ('glitch_H', 'rho_H', 't0_H'):
        assert same(t1[key], t2[key]), key
    n = 1000
    _, t = plant.simulate_batch(config(*MIXED), 0, n, s)
    flags = np.stack([t['merger'], t['glitch_H'], t['glitch_L']]).astype(float)
    r = np.corrcoef(flags)
    both = t['glitch_H'] & t['glitch_L']
    rt = np.corrcoef(t['t0_H'][both], t['t0_L'][both])[0, 1]
    bound = 4 / np.sqrt(n)
    msg = 'correlations of merger, glitch H, glitch L presence:\n%s\nglitch times H, L: %.3f (%d); bound %.3f' % (
        np.round(r, 3), rt, both.sum(), bound)
    print('\n' + msg)
    assert np.all(np.abs(r[np.triu_indices(3, 1)]) < bound) and abs(rt) < 4 / np.sqrt(both.sum()), msg
    assert abs(flags.mean(axis=1) - 0.5).max() < bound, flags.mean(axis=1)


# ----------------------------------------------------------------------------------------------- the driver


class Stub:
    """pe_redshift in place: arrays of its shapes on plant.GRID, deterministic functions of the data and of the seed it
    is given, so that wrong data or a wrong seed shows; records its calls"""

    def __init__(self):
        self.calls, self.factories = [], 0

    def factory(self, seg0):
        self.factories += 1
        return self.pe

    def pe(self, seg, seed, glitch):
        n = seg['d_H'].shape[0]
        self.calls.append((n, seed, glitch))
        m1, z = grid()
        q = {I: np.mean(np.abs(seg['d_' + I]) ** 2 / seg['sig2'], axis=1) - 1.0 for I in 'HL'}
        a = q['H'] + q['L'] + 1e-3 * ((seed + np.arange(n)) % 101)
        Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
        shape = np.where((Md >= 20) & (Md <= 400), 0.1 * np.log1p(z)[None, :] - m1[:, None] / 500, -np.inf)
        res = dict(m1=m1, z=z, rows=np.arange(n), ln_b=a, ln_lambda=a[:, None, None] + shape[None])
        if glitch:
            for j, key in enumerate(plant.KEYS[1:]):
                res[key] = res['ln_lambda'] + 0.1 * (j + 1)
            nr, nm = RHO_EDGES.size - 1, plant.MASSES.size
            for I in 'HL':
                res['ln_glitch_' + I] = q[I][:, None, None] * np.linspace(0, 1, nm)[None, :, None] \
                    * np.linspace(-1, 2, nr)[None, None, :]
                res['wmax_' + I] = (np.abs(seg['d_' + I]) ** 2 / seg['sig2']).max(axis=1)[:, None] \
                    * np.linspace(0.5, 1, nm)[None]
            res.update(rho_edges=RHO_EDGES.copy(), glitch_pi_k=np.full(nr, 1.0 / nr))
        return res


def drive(out, *extra, stub=None):
    stub = stub or Stub()
    args = ['nostore', 'cpu', '--nbar', '0.3', '--eps-H', '0.3', '--eps-L', '0.3', '--glitch-kind', 'mixed',
            '--control', 'segment-slide', '--seed', '5', '--out', str(out)] + list(extra)
    assert plant.main(args, factory=stub.factory) == 0
    return stub


def test_driver_with_stub(tmp_path, capsys):
    """plant.main end to end with the stub (segments 2 to 8, batches of 3, --plain, segment slide, mixed glitches):
    the files cover [2, 3), [3, 6), [6, 9) with no temporary file left; the stub is built once and called once per batch
    with glitch and once without, with the seed stream(seed, first, 'pe'); every member has its shape and the truth is
    that of simulate_batch; ln_glitch_I is the template average of the stub's ln E; a second run finds every file and
    calls nothing; the same segments in one batch of 9 from 0 give the same per-segment arrays bit for bit; a run of
    another campaign into the same directory stops; plant_fit reads the directory and prints its criteria"""
    one, two = tmp_path / 'one', tmp_path / 'two'
    stub = drive(one, '--start', '2', '--n', '7', '--batch', '3', '--plain')
    names = sorted(os.path.basename(f) for f in glob.glob(str(one / '*')))
    hidden = [f for f in os.listdir(one) if f.startswith('.')]
    assert names == ['plant_%010d_%010d.npz' % r for r in ((2, 3), (3, 6), (6, 9))] and not hidden, (names, hidden)
    cfg = config('--nbar', '0.3', '--eps-H', '0.3', '--eps-L', '0.3', '--glitch-kind', 'mixed', '--control',
                 'segment-slide', '--seed', '5')
    seeds = [plant.stream(5, a, 'pe') for a in (2, 3, 6)]
    assert stub.factories == 1 and stub.calls == [(m, s, g) for m, s in zip((1, 3, 3), seeds) for g in (True, False)]
    seg, truth = plant.simulate_batch(cfg, 3, 6, plant.Sampler(plant.population()))
    with np.load(one / names[1]) as f:
        for key in truth:
            assert same(f[key], truth[key]), key
        for key in [k + '_bar' for k in plant.KEYS] + ['ln_lambda_bar_plain', 'w_inf', 'wmax_H', 'wmax_L']:
            assert f[key].shape == (3,), key
        assert f['ln_glitch_H'].shape == (3, RHO_EDGES.size - 1) and np.array_equal(f['rho_edges'], RHO_EDGES)
        res = Stub().pe(seg, seeds[1], True)
        assert np.allclose(f['ln_glitch_L'], logsumexp(res['ln_glitch_L'], axis=1) - np.log(plant.MASSES.size),
                           rtol=0, atol=1e-12)
        assert np.all(f['w_inf'] == 0) and np.array_equal(f['ln_lambda_bar_plain'], f['ln_lambda_bar'])
    stamp = {n: os.path.getmtime(one / n) for n in names}
    again = drive(one, '--start', '2', '--n', '7', '--batch', '3', '--plain')
    assert again.calls == [] and again.factories == 0 and stamp == {n: os.path.getmtime(one / n) for n in names}
    drive(two, '--start', '0', '--n', '9', '--batch', '9')
    _, _, a, _ = plant_fit.load(str(one))
    _, _, b, _ = plant_fit.load(str(two))
    assert np.array_equal(a['k'], np.arange(2, 9)) and np.array_equal(b['k'], np.arange(9))
    for key in a:
        if key != 'ln_lambda_bar_plain':
            assert same(a[key], b[key][2:]), key
    with pytest.raises(SystemExit):
        drive(one, '--start', '0', '--n', '3', '--batch', '3', '--nbar', '0.2')
    capsys.readouterr()
    code = plant_fit.main([str(two)])
    text = capsys.readouterr().out
    assert code in (0, 1) and 'population covered' in text and 'time curve' in text, text
    with np.load(one / names[2]) as f:                                  # another fiducial law: not the same campaign
        arrays = {key: f[key] for key in f.files}
    np.savez(one / names[2], **dict(arrays, glitch_pi_k=0.5 * arrays['glitch_pi_k']))
    with pytest.raises(SystemExit, match='glitch_pi_k'):
        plant_fit.load(str(one))


def test_analysis_wiring(monkeypatch):
    """make_analysis on the CPU: the bank is the 120 masses of plant.MASSES built on the segments, and pe_redshift is
    called with the store, chunk 128, the given seed and ymin, the grid of notebook 7, glitch True or None"""
    import exmarg.search as search
    seen = []
    monkeypatch.setattr(search, 'pe_redshift', lambda seg, bank, store, **kw: seen.append((seg, bank, store, kw)))
    seg, _ = plant.simulate_batch(config('--nbar', '0'), 0, 2, plant.Sampler(plant.population()))
    pe = plant.make_analysis('the store', 'cpu', ymin=0.0982)(seg)
    pe(seg, 17, True)
    pe(seg, 17, False)
    (s, bank, store, kw), (_, bank2, _, kw2) = seen
    assert s is seg and store == 'the store' and bank is bank2 and bank.device == 'cpu'
    assert np.array_equal(bank.M, plant.MASSES) and np.all(bank.q == 1) and np.all(bank.chi == 0)
    assert kw == dict(chunk=128, resident=False, seed=17, ymin=0.0982, mdet_range=(20.0, 400.0), glitch=True,
                      m1_range=(4.0, 200.0), nm1=150, zmax=10.0, nz=150), kw
    assert kw2 == dict(kw, glitch=None)


def test_reduce_coverage():
    """reduce: with ln Lambda = 0 on every cell of the grid but -inf above z = 8 (a store that stops at y(z = 8)),
    w_inf is the population weight above z = 8 (0.0067 of the restricted population) and Lambda_bar = 1 - w_inf"""
    m1, z = grid()
    pop = plant.population()
    L = np.broadcast_to(np.where(z > 8.0, -np.inf, 0.0), (2, m1.size, z.size))
    res = dict(m1=m1, z=z, **{key: L for key in plant.KEYS})
    for I in 'HL':
        res['ln_glitch_' + I], res['wmax_' + I] = np.zeros((2, 3, 4)), np.ones((2, 3))
    out = plant.reduce(res, pop)
    w = population_weights(m1, z, plant.MDET, pop)
    above = w[:, z > 8.0].sum()
    print('\npopulation weight above z = 8: %.5f' % above)
    assert 0.005 < above < 0.01 and np.allclose(out['w_inf'], above, rtol=1e-12)
    assert np.allclose(np.exp(out['ln_lambda_bar']), 1 - above, rtol=1e-12)
    assert np.allclose(out['ln_lambda_bar'], ln_lambda_bar(L, m1, z, plant.MDET, pop), rtol=0, atol=0)


# ----------------------------------------------------------------------------------------------- plant_fit


TOY = dict(n=12000, nbar=0.05, eps_H=0.10, eps_L=0.03, seed=20261115)
GAMMA, RHO_MIN = LAWS['loud']


def toy_campaign(nbar, *extra):
    return config('--nbar', str(nbar), '--eps-H', str(TOY['eps_H']), '--eps-L', str(TOY['eps_L']), '--gamma',
                  str(GAMMA), '--rho-min', str(RHO_MIN), '--rho-max', str(EDGES[-1]), *extra)


def write_toy(directory, nbar_plant, cfg, splits=((0, 2500), (2500, 9000), (9000, 12000))):
    """the toy of test_mixture (loud law) in the files of plant.py, mergers at nbar_plant, under the campaign cfg;
    returns the evidences and the flags"""
    toy = Toy()
    pis = law(EDGES, GAMMA, RHO_MIN)
    z, fl = toy.data(toy.draws(TOY['n'], TOY['seed']), nbar_plant, dict(H=TOY['eps_H'], L=TOY['eps_L']),
                     dict(H=pis, L=pis))
    ev = toy.evidences(z, [dict(H=pis, L=pis)])
    n = TOY['n']
    per = dict(ln_lambda_bar=ev['S'], ln_lambda_gH_bar=ev['gH'][0], ln_lambda_gL_bar=ev['gL'][0],
               ln_lambda_gHL_bar=ev['HL'][0], ln_lambda_H_bar=ev['SH'], ln_lambda_L_bar=ev['SL'], ln_glitch_H=ev['EH'],
               ln_glitch_L=ev['EL'], wmax_H=np.abs(z['H']).max(axis=1) ** 2, wmax_L=np.abs(z['L']).max(axis=1) ** 2,
               w_inf=np.zeros(n), k=np.arange(n), merger=fl['m'], merger_H=fl['m'], merger_L=fl['m'],
               coherent=fl['m'] & (cfg['control'] == 'none'), m1=np.full(n, np.nan), z=np.full(n, np.nan),
               y=np.zeros(n), snr=np.zeros(n))
    for I in 'HL':
        per.update({'glitch_' + I: fl['g' + I], 'kind_' + I: np.where(fl['g' + I], 'template', ''),
                    'rho_' + I: np.full(n, np.nan), 't0_' + I: np.full(n, np.nan)})
    os.makedirs(directory, exist_ok=True)
    for a, b in splits:
        arrays = {key: v[a:b] for key, v in per.items()}
        plant.write_batch(os.path.join(directory, 'plant_%010d_%010d.npz' % (a, b)), cfg,
                          dict(first=a, end=b, t_sim=1e-3, t_pe=0.1, t_plain=None), dict(arrays, rho_edges=EDGES))
    return ev, fl


@pytest.fixture(scope='module')
def toy_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp('toy')
    out = {}
    for name, nbar_plant, cfg in (('plant', TOY['nbar'], toy_campaign(TOY['nbar'])),
                                  ('wrong truth', TOY['nbar'], toy_campaign(0.2)),
                                  ('control', TOY['nbar'], toy_campaign(TOY['nbar'], '--control', 'time-slide'))):
        path = str(root / name.replace(' ', '_'))
        out[name] = (path,) + write_toy(path, nbar_plant, cfg)
    return out


def direct(ev, sg=True, free=('nbar', 'eps_H', 'eps_L', 'gamma_H', 'gamma_L'), start=None, pi_k=(None, None)):
    st = dict(nbar=0.01, eps_H=0.01, eps_L=0.01, gamma_H=3.0, gamma_L=3.0, rho_min_H=RHO_MIN, rho_min_L=RHO_MIN)
    st.update(start or {})
    return mixture.fit(ev['S'], ev['gH'][0], ev['gL'][0], ev['EH'], ev['EL'], EDGES, start=st, log=True, sg=sg,
                       fixed=tuple(p for p in mixture.PARAMETERS if p not in free), lam_gHL=ev['HL'][0],
                       pi_k_H=pi_k[0], pi_k_L=pi_k[1])


def test_plant_fit_bookkeeping(toy_dirs, capsys):
    """plant_fit on the toy written as three result files: the counts are those of the toy's flags; nbar and its error
    from the Gaussian, Smith and Thrane, joint and two-stage fits, eps and gamma of the joint fit, the unit means and
    the ends and maxima of the time curves equal direct calls of mixture.fit, fit_noise, ln_mixture and time_curve on
    the toy evidences (to 1e-12), and the numbers printed are those; each criterion is the comparison it states, and the
    exit status is 0 exactly when every criterion passes"""
    path, ev, fl = toy_dirs['plant']
    r = plant_fit.report(path)
    n = TOY['n']
    got = {name: v for name, _, v in r['counts']}
    assert got['segments'] == n and got['mergers planted'] == fl['m'].sum() and got['glitches in H'] == fl['gH'].sum()
    assert got['glitches in L'] == fl['gL'].sum() and got['merger signal and glitch in H'] == (fl['m'] & fl['gH']).sum()
    assert got['merger signal and glitch in L'] == (fl['m'] & fl['gL']).sum()
    F = r['fits']
    s1 = {I: mixture.fit_noise(ev['E' + I], log=True) for I in 'HL'}
    s2, pk = dict(eps_H=s1['H']['eps'], eps_L=s1['L']['eps']), (s1['H']['pi_k'], s1['L']['pi_k'])
    ref = dict(gauss=direct(ev, free=('nbar',), start=dict(eps_H=0.0, eps_L=0.0)), smith_thrane=direct(ev, sg=False),
               joint=direct(ev), two_stage=direct(ev, free=('nbar',), start=s2, pi_k=pk),
               two_stage_eps=direct(ev, free=('nbar', 'eps_H', 'eps_L'), start=s2, pi_k=pk))
    assert F['joint']['held'] == [] and F['smith_thrane']['held'] == []
    close = lambda x, y: abs(x - y) <= 1e-12 * max(abs(y), 1e-300)
    for key, d in ref.items():
        for p in d['free']:
            assert close(F[key][p], d[p]) and close(F[key]['err'][p], d['err'][p]), (key, p, F[key][p], d[p])
        assert close(F[key]['lnl'], d['lnl'])
    for I in 'HL':
        assert close(F['stage1'][I]['eps'], s1[I]['eps'])
    text = '\n'.join(r['lines'])
    for key, d in ref.items():
        assert '%.5g +- %.5g' % (d['nbar'], d['err']['nbar']) in text, key
    J = ref['joint']
    for I in 'HL':
        for p in ('eps_' + I, 'gamma_' + I):
            assert '%s %.4g +- %.2g' % (p, J[p], J['err'][p]) in text, p
    lnG = lambda E, pi: mixture.ln_glitch_evidence(E, pi)
    pis = law(EDGES, GAMMA, RHO_MIN)
    B_true = mixture.ln_mixture(ev['S'], ev['gH'][0], ev['gL'][0], lnG(ev['EH'], pis), lnG(ev['EL'], pis),
                                TOY['eps_H'], TOY['eps_L'], ln_lam_gHL=ev['HL'][0])
    lawJ = {I: mixture.amplitude_weights(EDGES, J['gamma_' + I], RHO_MIN) for I in 'HL'}
    B_fit = mixture.ln_mixture(ev['S'], ev['gH'][0], ev['gL'][0], lnG(ev['EH'], lawJ['H']), lnG(ev['EL'], lawJ['L']),
                               J['eps_H'], J['eps_L'], ln_lam_gHL=ev['HL'][0])
    assert np.allclose(r['lnB_true'], B_true, rtol=0, atol=1e-10)
    assert np.allclose(r['lnB_fit'], B_fit, rtol=0, atol=1e-10)
    quiet = ~fl['m']
    glitched = fl['gH'] | fl['gL']
    for sample, sel in (('all', quiet), ('with a glitch', quiet & glitched), ('without', quiet & ~glitched)):
        for name, v in (('Gaussian Lambda_bar - 1', ev['S']), ('B - 1, planted noise model', B_true),
                        ('B - 1, joint fit', B_fit)):
            x = np.exp(v[sel]) - 1
            mu, se, k = r['unit'][sample][name]
            assert k == sel.sum() and np.isclose(mu, x.mean(), rtol=1e-9) and np.isclose(se, x.std(ddof=1) / np.sqrt(k),
                                                                                          rtol=1e-9), (sample, name)
            assert '%+.3e +- %.2e  (%d,' % (x.mean(), x.std(ddof=1) / np.sqrt(k), k) in text, (sample, name)
    grid_, prior = np.geomspace(1e-4, 0.5, 600), lambda g: 1 / g
    for key, v in (('gauss', ev['S']), ('mixture', B_fit)):
        c = mixture.time_curve(v, grid_, prior, log=True)
        assert np.allclose(r['curves'][key], c, rtol=0, atol=1e-9)
        assert 'end %+.3f, max %+.3f' % (c[-1], c.max()) in text, key
    crit = {name: (ok, msg) for name, ok, msg in r['criteria']}
    e = J['err']['nbar']
    assert crit['mixture nbar within 2.5 sigma of the truth'][0] == (abs(J['nbar'] - TOY['nbar']) < 2.5 * e)
    assert not any(name.startswith('control') for name in crit) and crit['population covered: w_inf < 0.0001'][0]
    print('\n' + text)
    capsys.readouterr()
    assert plant_fit.main([path]) == (0 if all(ok for ok, _ in crit.values()) else 1)
    assert capsys.readouterr().out.strip() == text.strip()


def test_plant_fit_failure_and_control(toy_dirs, capsys):
    """the same toy declared at nbar 0.2 fails the nbar criterion (pull printed) and plant_fit exits with 1; declared a
    time-slide control (truth 0, no coherent merger, the merger signals still in it), it fails 'nbar consistent with 0'
    and states the maximum of the mixture time curve against ln 100, both computed as stated, and its unit mean runs
    over the segments without a merger signal, not over those without a coherent merger"""
    path, ev, fl = toy_dirs['wrong truth']
    r = plant_fit.report(path)
    crit = {name: (ok, msg) for name, ok, msg in r['criteria']}
    J = r['fits']['joint']
    pull = (J['nbar'] - 0.2) / J['err']['nbar']
    assert not crit['mixture nbar within 2.5 sigma of the truth'][0] and abs(pull) > 2.5
    assert 'pull %+.2f' % pull in crit['mixture nbar within 2.5 sigma of the truth'][1]
    assert plant_fit.main([path]) == 1
    path, ev, fl = toy_dirs['control']
    r = plant_fit.report(path)
    crit = {name: (ok, msg) for name, ok, msg in r['criteria']}
    J = r['fits']['joint']
    assert r['truth']['nbar'] == 0.0
    ok, msg = crit['control: nbar consistent with 0']
    assert ok == (J['nbar'] < 2.5 * J['err']['nbar']) and not ok, msg
    curve = r['curves']['mixture']
    ok, msg = crit['control: time curve below ln 100']
    assert ok == (curve.max() < np.log(100)) and 'max %+.3f' % curve.max() in msg, msg
    assert not r['data']['coherent'].any() and r['unit']['all']['B - 1, joint fit'][2] == (~fl['m']).sum()
    capsys.readouterr()
    assert plant_fit.main([path]) == 1
