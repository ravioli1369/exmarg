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
    is given, so that wrong data or a wrong seed shows, -inf outside the bank and on the row z = 10 (as the analysis
    with ymin = D0 / D_L(10) exactly); records its calls"""

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
        shape[:, -1] = -np.inf
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
    another campaign into the same directory stops; w_inf is the population weight of the row z = 10 and w_inf_cells
    its 53 cells, which plant_fit prints with the criteria; a file with another fiducial glitch law is refused"""
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
        m1, z = grid()
        w = population_weights(m1, z, plant.MDET, plant.population())
        assert np.allclose(f['w_inf'], w[:, -1].sum(), rtol=1e-12, atol=0)
        assert np.array_equal(f['w_inf_cells'], np.argwhere(np.isin(np.arange(z.size), [z.size - 1])[None] & (w > 0)))
        assert f['w_inf_cells'].shape == (53, 2) and np.array_equal(f['ln_lambda_bar_plain'], f['ln_lambda_bar'])
    stamp = {n: os.path.getmtime(one / n) for n in names}
    again = drive(one, '--start', '2', '--n', '7', '--batch', '3', '--plain')
    assert again.calls == [] and again.factories == 0 and stamp == {n: os.path.getmtime(one / n) for n in names}
    drive(two, '--start', '0', '--n', '9', '--batch', '9')
    a, b = plant_fit.load(str(one))[2], plant_fit.load(str(two))[2]
    assert np.array_equal(a['k'], np.arange(2, 9)) and np.array_equal(b['k'], np.arange(9))
    for key in a:
        if key != 'ln_lambda_bar_plain':
            assert same(a[key], b[key][2:]), key
    with pytest.raises(SystemExit):
        drive(one, '--start', '0', '--n', '3', '--batch', '3', '--nbar', '0.2')
    capsys.readouterr()
    code = plant_fit.main([str(two)])
    text = capsys.readouterr().out
    assert code in (0, 1) and 'population covered: w_inf < 0.0005' in text and 'time curve' in text, text
    assert '53 cells, z 10.000 to 10.000, m1 4.561 to 17.865, M_det 100.3 to 393.0' in text, text
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
    assert kw == dict(chunk=128, resident=False, seed=17, ymin=0.0982, mdet_range=(20.0, 400.0), glitch=True, shared_times=False,
                      m1_range=(4.0, 200.0), nm1=150, zmax=10.0, nz=150), kw
    assert kw2 == dict(kw, glitch=None)


def test_reduce_coverage():
    """reduce: with ln Lambda = 0 on every cell of the grid but -inf above z = 8 (a store that stops at y(z = 8)),
    w_inf is the population weight above z = 8 (0.0067 of the restricted population), w_inf_cells the cells above z = 8
    with weight, and Lambda_bar = 1 - w_inf"""
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
    assert np.array_equal(out['w_inf_cells'], np.argwhere((z[None, :] > 8.0) & (w > 0)))


def test_default_ymin_covers_grid():
    """pe_redshift's node geometry and interpolation to (m1, z) (pe._node_geometry, pe._to_source_frame; no table is
    read) on a store with the nodes s of data/tables/s_grid.npz and the amplitude axis of the published tables
    (sqrt(alpha_lo) = 0.4 at every node, from their headers; alpha_hi unbounded here): with plant's default ymin no cell
    of positive population weight is left without ln Lambda; with ymin = D0 / D_L(10) exactly the row z = 10 is (53
    cells, 1.3187e-4 of the population with z >= 0.99119, the w_inf of the pilot), and with pe_redshift's default
    0.1267 every cell above z = 8 is (6.7e-3)"""
    from exmarg.search import mass_bank, pe
    from exmarg.search.store import RedshiftStore
    s = np.load(os.path.join(ROOT, 'data', 'tables', 's_grid.npz'))['s_nodes']
    st = RedshiftStore(s=s, alpha_lo=np.full(s.size, 0.16), alpha_hi=np.full(s.size, np.inf), beta=None, delta=None,
                       tau_idx=None, tables=None, n_max=120)
    seg, _ = plant.simulate_batch(config('--nbar', '0'), 0, 1, plant.Sampler(plant.population()))
    bank = mass_bank(plant.MASSES, seg, 'cpu')
    m1, z = grid()
    w = population_weights(m1, z, plant.MDET, plant.population(0.99119))

    def missing(ymin):
        valid, _, yN, bnode, _ = pe._node_geometry(bank, st, ymin)
        lamN = np.where(np.concatenate([valid, (bnode >= 0)[None]]), 0.0, -np.inf)
        return (w > 0) & ~np.isfinite(pe._to_source_frame(lamN, yN, bank.M, pe.D0 / pe._dl_of_z(z), m1, z, np))

    default = config('--nbar', '0')['ymin']
    exact = float(pe.D0 / pe._dl_of_z(10.0)[0])
    row, above8 = missing(exact), missing(0.1267)
    msg = 'default ymin %.10f: %d cells; D0 / D_L(10) = %.10f: %d cells, weight %.4e; 0.1267: %d cells, weight %.4e' % (
        default, missing(default).sum(), exact, row.sum(), w[row].sum(), above8.sum(), w[above8].sum())
    print('\n' + msg)
    assert default < exact and not missing(default).any(), msg
    assert row.sum() == 53 and np.array_equal(row[:, -1], w[:, -1] > 0) and abs(w[row].sum() - 1.3187e-4) < 1e-8, msg
    assert np.array_equal(above8, (w > 0) & (z[None, :] > 8.0)), msg


# ----------------------------------------------------------------------------------------------- plant_fit


TOY = dict(n=12000, nbar=0.05, eps_H=0.10, eps_L=0.03, seed=20261115)
GAMMA, RHO_MIN = LAWS['loud']


def toy_campaign(nbar, *extra):
    return config('--nbar', str(nbar), '--eps-H', str(TOY['eps_H']), '--eps-L', str(TOY['eps_L']), '--gamma',
                  str(GAMMA), '--rho-min', str(RHO_MIN), '--rho-max', str(EDGES[-1]), *extra)


def write_toy(directory, nbar_plant, cfg, splits=((0, 2500), (2500, 9000), (9000, 12000))):
    """the toy of test_mixture (loud law, glitches up to rho 30) in the files of plant.py, mergers at nbar_plant, under
    the campaign cfg; wmax_I = max_t |z_I|^2 (u = 1) and rho_I the planted SNR (as Toy.data draws it).  returns the
    evidences, the flags and the per-segment arrays written"""
    toy = Toy()
    pis = law(EDGES, GAMMA, RHO_MIN)
    draws = toy.draws(TOY['n'], TOY['seed'])
    z, fl = toy.data(draws, nbar_plant, dict(H=TOY['eps_H'], L=TOY['eps_L']), dict(H=pis, L=pis))
    ev = toy.evidences(z, [dict(H=pis, L=pis)])
    n = TOY['n']
    per = dict(ln_lambda_bar=ev['S'], ln_lambda_gH_bar=ev['gH'][0], ln_lambda_gL_bar=ev['gL'][0],
               ln_lambda_gHL_bar=ev['HL'][0], ln_lambda_H_bar=ev['SH'], ln_lambda_L_bar=ev['SL'], ln_glitch_H=ev['EH'],
               ln_glitch_L=ev['EL'], wmax_H=np.abs(z['H']).max(axis=1) ** 2, wmax_L=np.abs(z['L']).max(axis=1) ** 2,
               w_inf=np.zeros(n), k=np.arange(n), merger=fl['m'], merger_H=fl['m'], merger_L=fl['m'],
               coherent=fl['m'] & (cfg['control'] == 'none'), m1=np.full(n, np.nan), z=np.full(n, np.nan),
               y=np.zeros(n), snr=np.zeros(n))
    for I in 'HL':
        g = draws['g'][I]
        k = np.searchsorted(np.cumsum(pis) / pis.sum(), g['uk'], side='right')
        rho = np.where(fl['g' + I], EDGES[np.minimum(k, EDGES.size - 2)] + g['ur'] * np.diff(EDGES)[np.minimum(
            k, EDGES.size - 2)], np.nan)
        per.update({'glitch_' + I: fl['g' + I], 'kind_' + I: np.where(fl['g' + I], 'template', ''), 'rho_' + I: rho,
                    't0_' + I: np.full(n, np.nan)})
    os.makedirs(directory, exist_ok=True)
    for a, b in splits:
        arrays = {key: v[a:b] for key, v in per.items()}
        plant.write_batch(os.path.join(directory, 'plant_%010d_%010d.npz' % (a, b)), cfg,
                          dict(first=a, end=b, t_sim=1e-3, t_pe=0.1, t_plain=None),
                          dict(arrays, rho_edges=EDGES, w_inf_cells=np.zeros((0, 2), np.int32)))
    return ev, fl, per


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


def kept(ev, keep):
    """the toy evidences of the segments keep, by the names of direct"""
    return dict(S=ev['S'][keep], gH=ev['gH'][0][keep], gL=ev['gL'][0][keep], HL=ev['HL'][0][keep], EH=ev['EH'][keep],
                EL=ev['EL'][keep])


def direct(e, sg=True, free=('nbar', 'eps_H', 'eps_L', 'gamma_H', 'gamma_L'), start=None, pi_k=(None, None),
           rho_max=None):
    st = dict(nbar=0.01, eps_H=0.01, eps_L=0.01, gamma_H=3.0, gamma_L=3.0, rho_min_H=RHO_MIN, rho_min_L=RHO_MIN)
    st.update(start or {})
    return mixture.fit(e['S'], e['gH'], e['gL'], e['EH'], e['EL'], EDGES, start=st, log=True, sg=sg,
                       fixed=tuple(p for p in mixture.PARAMETERS if p not in free), lam_gHL=e['HL'],
                       pi_k_H=pi_k[0], pi_k_L=pi_k[1], rho_max=rho_max)


def direct_fits(e, sel, rho_max):
    """the five estimates and stage 1 of plant_fit, by direct calls on the evidences e"""
    s1 = {I: mixture.fit_noise(e['E' + I][:, sel], log=True) for I in 'HL'}
    pk = []
    for I in 'HL':
        pk.append(np.zeros(EDGES.size - 1))
        pk[-1][sel] = s1[I]['pi_k']
    s2 = dict(eps_H=s1['H']['eps'], eps_L=s1['L']['eps'])
    return s1, dict(gauss=direct(e, free=('nbar',), start=dict(eps_H=0.0, eps_L=0.0)),
                    smith_thrane=direct(e, sg=False, rho_max=rho_max), joint=direct(e, rho_max=rho_max),
                    two_stage=direct(e, free=('nbar',), start=s2, pi_k=pk),
                    two_stage_eps=direct(e, free=('nbar', 'eps_H', 'eps_L'), start=s2, pi_k=pk))


def same_fits(F, s1, ref):
    close = lambda x, y: abs(x - y) <= 1e-12 * max(abs(y), 1e-300)
    assert F['joint']['held'] == [] and F['smith_thrane']['held'] == []
    for key, d in ref.items():
        for p in d['free']:
            assert close(F[key][p], d[p]) and close(F[key]['err'][p], d['err'][p]), (key, p, F[key][p], d[p])
        assert close(F[key]['lnl'], d['lnl']), key
    for I in 'HL':
        assert close(F['stage1'][I]['eps'], s1[I]['eps']) and F['stage1'][I]['converged']


def test_plant_fit_bookkeeping(toy_dirs, capsys):
    """plant_fit on the toy written as three result files.  The counts are those of the toy's flags; the veto (wmax_H or
    wmax_L above 72) sets aside the segments it states, and its printed numbers (by truth, by detector, the glitches
    within 1 of rho_cut = 12 and across it) are those of the planted rho and wmax; stage 1 gets the 4 bins of [6, 12).
    On the kept segments nbar and its error from the Gaussian, Smith and Thrane, joint and two-stage fits, eps and gamma
    of the joint fit (law up to 12), the unit means, the largest ln Lambda_bar and ln B, and the time curves equal
    direct calls of mixture.fit, fit_noise, ln_mixture and time_curve on the toy evidences (to 1e-12), the truth of eps
    is the realized fraction among the kept, and the numbers printed are those; each criterion is the comparison it
    states and the exit status is 0 exactly when every one passes.  With --veto-w inf nothing is set aside and the
    joint fit is the direct one on every segment with the law up to the last edge"""
    path, ev, fl, per = toy_dirs['plant']
    r = plant_fit.report(path)
    n = TOY['n']
    got = {name: v for name, _, v in r['counts']}
    assert got['segments'] == n and got['mergers planted'] == fl['m'].sum() and got['glitches in H'] == fl['gH'].sum()
    assert got['glitches in L'] == fl['gL'].sum() and got['merger signal and glitch in H'] == (fl['m'] & fl['gH']).sum()
    assert got['merger signal and glitch in L'] == (fl['m'] & fl['gL']).sum()
    loud = {I: per['wmax_' + I] > 72 for I in 'HL'}
    keep = ~(loud['H'] | loud['L'])
    out = ~keep
    sel = (EDGES[1:] > RHO_MIN) & (EDGES[:-1] < 12.0)
    assert np.array_equal(r['keep'], keep) and np.array_equal(r['bins'], sel) and sel.sum() == 4 and r['rho_cut'] == 12
    text = '\n'.join(r['lines'])
    gH, gL, m = fl['gH'], fl['gL'], fl['m']
    assert ('set aside %d of %d segments; by truth (a segment can count twice): glitch in H %d, glitch in L %d, merger '
            '%d, neither glitch nor merger %d' % (out.sum(), n, (out & gH).sum(), (out & gL).sum(), (out & m).sum(),
                                                  (out & ~gH & ~gL & ~m).sum())) in text
    assert 40 < out.sum() < 400 and (out & ~gH & ~gL).sum() == 0
    for I, g in (('H', gH), ('L', gL)):
        rho = per['rho_' + I]
        near = g & (np.abs(rho - 12) < 1)
        assert ('%s: planted glitches within 1 of rho_cut %d (set aside %d); rho above the cut but kept %d, below it '
                'but set aside %d' % (I, near.sum(), (near & loud[I]).sum(), (g & (rho > 12) & ~loud[I]).sum(),
                                  (g & (rho <= 12) & loud[I]).sum())) in text, I
    e = kept(ev, keep)
    s1, ref = direct_fits(e, sel, 12.0)
    F = r['fits']
    same_fits(F, s1, ref)
    eps = {I: fl['g' + I][keep].mean() for I in 'HL'}
    assert r['truth']['eps_H'] == eps['H'] and r['truth']['eps_L'] == eps['L']
    for key, d in ref.items():
        assert '%.5g +- %.5g' % (d['nbar'], d['err']['nbar']) in text, key
    J = ref['joint']
    for I in 'HL':
        assert 'eps_%s %.4g +- %.2g (truth %.4g' % (I, J['eps_' + I], J['err']['eps_' + I], eps[I]) in text, I
        assert 'gamma_%s %.4g +- %.2g (truth 3' % (I, J['gamma_' + I], J['err']['gamma_' + I]) in text, I
    lnG = lambda E, pi: mixture.ln_glitch_evidence(E, pi)
    pis = mixture.amplitude_weights(EDGES, GAMMA, RHO_MIN, 12.0)
    B_true = mixture.ln_mixture(e['S'], e['gH'], e['gL'], lnG(e['EH'], pis), lnG(e['EL'], pis), eps['H'], eps['L'],
                                ln_lam_gHL=e['HL'])
    lawJ = {I: mixture.amplitude_weights(EDGES, J['gamma_' + I], RHO_MIN, 12.0) for I in 'HL'}
    B_fit = mixture.ln_mixture(e['S'], e['gH'], e['gL'], lnG(e['EH'], lawJ['H']), lnG(e['EL'], lawJ['L']),
                               J['eps_H'], J['eps_L'], ln_lam_gHL=e['HL'])
    assert np.allclose(r['lnB_true'], B_true, rtol=0, atol=1e-10)
    assert np.allclose(r['lnB_fit'], B_fit, rtol=0, atol=1e-10)
    kk = np.arange(n)[keep]
    for name, v in (('ln Lambda_bar', e['S']), ('ln B, planted noise model', B_true), ('ln B, joint fit', B_fit)):
        i = int(np.argmax(v))
        assert re_line(text, name).startswith('%10.3f   k %d:' % (v[i], kk[i])), (name, re_line(text, name))
    quiet = ~fl['m'][keep]
    glitched = (fl['gH'] | fl['gL'])[keep]
    for sample, sel_ in (('all', quiet), ('with a glitch', quiet & glitched), ('without', quiet & ~glitched)):
        for name, v in (('Gaussian Lambda_bar - 1', e['S']), ('B - 1, planted noise model', B_true),
                        ('B - 1, joint fit', B_fit)):
            x = np.exp(v[sel_]) - 1
            mu, se, c = r['unit'][sample][name]
            assert c == sel_.sum() and np.isclose(mu, x.mean(), rtol=1e-9), (sample, name)
            assert np.isclose(se, x.std(ddof=1) / np.sqrt(c), rtol=1e-9), (sample, name)
            assert '%+.3e +- %.2e  (%d,' % (x.mean(), x.std(ddof=1) / np.sqrt(c), c) in text, (sample, name)
    grid_, prior = np.geomspace(1e-4, 0.5, 600), lambda g: 1 / g
    for key, v in (('gauss', e['S']), ('mixture', B_fit)):
        c = mixture.time_curve(v, grid_, prior, log=True)
        assert np.allclose(r['curves'][key], c, rtol=0, atol=1e-9)
        assert 'end %+.3f, max %+.3f at segment %d of %d' % (c[-1], c.max(), np.argmax(c) + 1, keep.sum()) in text, key
    crit = {name: (ok, msg) for name, ok, msg, _ in r['criteria']}
    err = J['err']['nbar']
    assert crit['mixture nbar within 2.5 sigma of the truth'][0] == (abs(J['nbar'] - TOY['nbar']) < 2.5 * err)
    assert not any(name.startswith('control') for name in crit) and crit['population covered: w_inf < 0.0005'][0]
    assert 'no cell' in text
    print('\n' + text)
    capsys.readouterr()
    assert plant_fit.main([path]) == (0 if all(ok for ok, _ in crit.values()) else 1)
    assert capsys.readouterr().out.strip() == text.strip()
    r = plant_fit.report(path, veto_w=np.inf)
    s1, ref = direct_fits(kept(ev, np.ones(n, bool)), EDGES[1:] > RHO_MIN, None)
    assert r['keep'].all() and np.array_equal(r['bins'], EDGES[1:] > RHO_MIN)
    same_fits(r['fits'], s1, ref)


def re_line(text, name):
    """what follows the name on its line of 'largest among the kept segments'"""
    line = [l for l in text.splitlines() if l.startswith('  ' + name + ' ')][0]
    return line[2 + 26 + 1:]


def test_plant_fit_failure_and_control(toy_dirs, capsys):
    """the same toy declared at nbar 0.2 fails the nbar criterion (pull printed) and plant_fit exits with 1; declared a
    time-slide control (truth 0, no coherent merger, the merger signals still in it), it fails 'nbar consistent with 0'
    and states the maximum of the mixture time curve against ln 100, both computed as stated, and its unit mean runs
    over the kept segments without a merger signal, not over those without a coherent merger"""
    path, ev, fl, per = toy_dirs['wrong truth']
    r = plant_fit.report(path)
    crit = {name: (ok, msg) for name, ok, msg, _ in r['criteria']}
    J = r['fits']['joint']
    pull = (J['nbar'] - 0.2) / J['err']['nbar']
    assert not crit['mixture nbar within 2.5 sigma of the truth'][0] and abs(pull) > 2.5
    assert 'pull %+.2f' % pull in crit['mixture nbar within 2.5 sigma of the truth'][1]
    assert plant_fit.main([path]) == 1
    path, ev, fl, per = toy_dirs['control']
    r = plant_fit.report(path)
    crit = {name: (ok, msg, counted) for name, ok, msg, counted in r['criteria']}
    J = r['fits']['joint']
    assert r['truth']['nbar'] == 0.0 and r['kind'] == 'control'
    ok, msg, counted = crit['mixture: nbar consistent with 0']
    assert counted and ok == (J['nbar'] < 2.5 * J['err']['nbar']) and not ok, msg
    assert not crit['Gaussian: nbar consistent with 0'][2] and not crit['Gaussian: evidence below ln 100'][2]
    curve = r['curves']['mixture']
    ok, msg, counted = crit['mixture: evidence below ln 100']
    assert ok == (curve.max() < np.log(100)) and 'max %+.3f' % curve.max() in msg, msg
    keep = (per['wmax_H'] <= 72) & (per['wmax_L'] <= 72)
    assert not r['data']['coherent'].any() and r['unit']['all']['B - 1, joint fit'][2] == (~fl['m'] & keep).sum()
    capsys.readouterr()
    assert plant_fit.main([path]) == 1


def test_plant_fit_soft_veto(toy_dirs):
    """--veto-law soft: the joint and Smith and Thrane fits on the kept segments equal direct mixture.fit calls with
    veto = (12, 0.1, 1.0) to 1e-12, the planted model of the unit mean has the planted law times the same probability
    of passing the veto, and the output says which cut it used"""
    path, ev, fl, per = toy_dirs['plant']
    r = plant_fit.report(path, veto_law='soft')
    keep = (per['wmax_H'] <= 72) & (per['wmax_L'] <= 72)
    e = kept(ev, keep)
    v = (12.0, 0.1, 1.0)
    st = dict(nbar=0.01, eps_H=0.01, eps_L=0.01, gamma_H=3.0, gamma_L=3.0, rho_min_H=RHO_MIN, rho_min_L=RHO_MIN)
    for key, sg in (('joint', True), ('smith_thrane', False)):
        d = mixture.fit(e['S'], e['gH'], e['gL'], e['EH'], e['EL'], EDGES, start=st, log=True, sg=sg,
                        fixed=('rho_min_H', 'rho_min_L'), lam_gHL=e['HL'], veto=v)
        for p in d['free']:
            assert abs(r['fits'][key][p] - d[p]) <= 1e-12 * abs(d[p]), (key, p, r['fits'][key][p], d[p])
        assert abs(r['fits'][key]['lnl'] - d['lnl']) <= 1e-12 * abs(d['lnl'])
    eps = {I: fl['g' + I][keep].mean() for I in 'HL'}
    pis = mixture.amplitude_weights(EDGES, GAMMA, RHO_MIN, EDGES[-1], veto=v)
    lnG = lambda E: mixture.ln_glitch_evidence(E, pis)
    B = mixture.ln_mixture(e['S'], e['gH'], e['gL'], lnG(e['EH']), lnG(e['EL']), eps['H'], eps['L'], ln_lam_gHL=e['HL'])
    assert np.allclose(r['lnB_true'], B, rtol=0, atol=1e-10)
    assert 'soft cut' in '\n'.join(r['lines']) and 'the law up to 12 (soft)' in '\n'.join(r['lines'])
