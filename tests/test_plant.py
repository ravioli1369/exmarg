"""the planted-truth campaign (scripts/validate/plant.py and plant_fit.py) without the tables: the merger sampler
against the population of the analysis, segment k the same data in every split of a campaign, the random streams
independent, the driver end to end with a stub in place of pe_redshift, the coverage diagnostic, and the bookkeeping of
plant_fit on the toy evidences of tests/test_mixture.py against direct calls of exmarg.search.mixture"""
import glob
import importlib.util
import json
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


plant, plant_fit, windows_fit = _script('plant'), _script('plant_fit'), _script('windows_fit')


def config(*extra):
    """a campaign of plant.py from command-line arguments (store and out unused)"""
    return plant.campaign(plant.parser().parse_args(['store', 'cpu', '--n', '1', '--out', 'x'] + list(extra)))


MIXED = ('--nbar', '0.5', '--eps-H', '0.5', '--eps-L', '0.5', '--glitch-kind', 'mixed')
STATES = ('--glitch-states', '0.7:0.01,0.3:0.1', '--glitch-kind', 'mixed', '--rho-min', '4')


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
            windows = isinstance(glitch, dict)                          # the window model: its outputs, which
            for j, key in enumerate(('ln_B', 'ln_lambda_H', 'ln_lambda_L') if windows else plant.KEYS[1:]):
                res[key] = res['ln_lambda'] + 0.1 * (j + 1)                # depend on its parameters
            nr, nm = RHO_EDGES.size - 1, plant.MASSES.size
            for I in 'HL':
                res['ln_glitch_' + I] = q[I][:, None, None] * np.linspace(0, 1, nm)[None, :, None] \
                    * np.linspace(-1, 2, nr)[None, None, :]
                res['wmax_' + I] = (np.abs(seg['d_' + I]) ** 2 / seg['sig2']).max(axis=1)[:, None] \
                    * np.linspace(0.5, 1, nm)[None]
                if windows:
                    res['ln_c_' + I] = q[I] * glitch[I]['q'][-1]
                    res['state_' + I] = np.tile(np.asarray(glitch[I]['f']), (n, 1))
            res['rho_edges'] = RHO_EDGES.copy()
            if windows:
                res['ln_B'] = res['ln_B'] - 10.0 * glitch['H']['q'][0]
            else:
                res['glitch_pi_k'] = np.full(nr, 1.0 / nr)
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
    its 53 cells, which plant_fit prints with the criteria (a control with the stub's positive evidences: the fit runs
    to nbar = 1 and the run fails, exit status 1, on that criterion alone); a file with another fiducial glitch law is
    refused"""
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
    assert code == 1 and 'time curve' in text, text            # the stub's ln Lambda is positive in every segment: the
    assert 'FAIL  mixture: nbar consistent with 0: nbar 1 +- nan' in text, text        # fit runs to nbar = 1, and a
    assert 'PASS  population covered: w_inf < 0.0005' in text, text                    # control (truth 0) fails
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


# ----------------------------------------------------------------------------------------------- window-model glitches


def test_glitch_states_rates():
    """the window model planted (states 0.7:0.01, 0.3:0.1 in both detectors, 64 windows, mixed kinds, rho^-3 on [4,
    200]) over 600 segments: the fraction of segments in each state and the mean number of glitches per segment in each
    state within 4 standard errors of f and 64 q; glitch_I is n_glitches_I > 0, rho_I (the loudest) inside [4, 200],
    t0_I on the matched-filter grid; every kind is the loudest somewhere; the states of H and L uncorrelated
    (|r| < 4 / sqrt(600)).  With q doubled the states, and in each window the presence, are those of the first
    campaign (the same uniforms), so that every segment has at least as many glitches; another law for L leaves the
    data and the glitches of H as they were"""
    n, nw = 600, plant.N_WINDOW
    s = plant.Sampler(plant.population())
    cfg = config('--nbar', '0.3', *STATES)
    assert cfg['glitch_states'] == dict(H=[[0.7, 0.01], [0.3, 0.1]], L=[[0.7, 0.01], [0.3, 0.1]], n_window=64)
    seg, tr = plant.simulate_batch(cfg, 0, n, s)
    dt = plant.controls.time_step(seg)
    lines = []
    for I in 'HL':
        st, ng = tr['glitch_state_' + I], tr['n_glitches_' + I]
        for j, (f, q) in enumerate(((0.7, 0.01), (0.3, 0.1))):
            frac, sd = (st == j).mean(), np.sqrt(f * (1 - f) / n)
            mean, sdn = ng[st == j].mean(), np.sqrt(nw * q * (1 - q) / (st == j).sum())
            lines.append('%s state %d: fraction %.4f (f %.2f, sd %.4f), glitches per segment %.3f (64 q %.2f, sd %.3f)'
                         % (I, j, frac, f, sd, mean, nw * q, sdn))
            assert abs(frac - f) < 4 * sd and abs(mean - nw * q) < 4 * sdn, lines[-1]
        g = tr['glitch_' + I]
        assert np.array_equal(g, ng > 0) and np.all((tr['rho_' + I][g] >= 4) & (tr['rho_' + I][g] <= 200))
        assert np.all(np.isnan(tr['rho_' + I][~g])) and set(tr['kind_' + I][g]) == set(plant.KINDS)
        assert np.allclose(tr['t0_' + I][g] / dt, np.rint(tr['t0_' + I][g] / dt), rtol=0, atol=1e-6)
    r = np.corrcoef(tr['glitch_state_H'], tr['glitch_state_L'])[0, 1]
    print('\n' + '\n'.join(lines) + '\nstates of H and L: r = %.3f' % r)
    assert abs(r) < 4 / np.sqrt(n)
    seg2, tr2 = plant.simulate_batch(config('--nbar', '0.3', '--glitch-states', '0.7:0.02,0.3:0.2', '--glitch-kind',
                                            'mixed', '--rho-min', '4'), 0, 200, s)
    for I in 'HL':
        assert np.array_equal(tr2['glitch_state_' + I], tr['glitch_state_' + I][:200])
        assert np.all(tr2['n_glitches_' + I] >= tr['n_glitches_' + I][:200])
        assert (tr2['n_glitches_' + I] > tr['n_glitches_' + I][:200]).any()
    seg3, tr3 = plant.simulate_batch(config('--nbar', '0.3', *STATES, '--glitch-states-L', '0.5:0.3,0.5:0.0'), 0, 200,
                                     s)
    assert np.array_equal(seg3['d_H'], seg['d_H'][:200]) and not np.array_equal(seg3['d_L'], seg['d_L'][:200])
    for key in ('glitch_state_H', 'n_glitches_H', 'rho_H', 't0_H', 'kind_H'):
        assert same(tr3[key], tr[key][:200]), key


@pytest.mark.parametrize('control', plant.CONTROLS)
def test_split_reproducible_states(control):
    """with the window model planted, segment k is the same data and truth bit for bit in one batch from 0 and in the
    splits [3, 9) and [3, 5) + [5, 9), for every control"""
    cfg = config('--nbar', '0.5', '--glitch-states', '0.5:0.02,0.5:0.2', '--glitch-kind', 'mixed', '--control',
                 control)
    s = plant.Sampler(plant.population())
    ref, tref = plant.simulate_batch(cfg, 0, 9, s)
    assert tref['glitch_H'].sum() > 2 and tref['glitch_L'].sum() > 2
    for split in ([(3, 9)], [(3, 5), (5, 9)]):
        parts = [plant.simulate_batch(cfg, a, b, s) for a, b in split]
        for I in 'HL':
            assert np.array_equal(np.concatenate([p[0]['d_' + I] for p in parts]), ref['d_' + I][3:]), (split, I)
        for key in tref:
            assert same(np.concatenate([p[1][key] for p in parts]), tref[key][3:]), (split, key)


@pytest.mark.parametrize('control', ('none', 'time-slide', 'segment-slide'))
def test_paired_campaigns_share_noise_and_mergers(control):
    """three campaigns of one seed, without glitches, with one glitch per segment (--eps, mixed) and with the window
    model (mixed): the mergers (presence, m1, z, y, optimal SNR, signals after the control) are the same, a detector
    segment without a glitch holds the same data bit for bit in all three, and one with glitches differs from the
    glitch-free data by the glitches alone: with exactly one, 2 sum |d - d_0|^2 / sig2 = rho^2 (1e-9).  Another seed
    shares nothing"""
    s = plant.Sampler(plant.population())
    base = ('--nbar', '0.3', '--seed', '9', '--control', control)
    n = 40
    seg0, t0 = plant.simulate_batch(config(*base), 0, n, s)
    other, _ = plant.simulate_batch(config('--nbar', '0.3', '--seed', '10', '--control', control), 0, n, s)
    assert not any(np.array_equal(seg0['d_H'][i], other['d_H'][j]) for i in range(n) for j in range(n))
    counts = []
    for extra in (('--eps-H', '0.3', '--eps-L', '0.3', '--glitch-kind', 'mixed', '--rho-min', '4'),
                  ('--glitch-states', '0.6:0.005,0.4:0.03', '--glitch-kind', 'mixed', '--rho-min', '4')):
        seg, tr = plant.simulate_batch(config(*base, *extra), 0, n, s)
        for key in ('merger', 'm1', 'z', 'y', 'snr', 'merger_H', 'merger_L', 'coherent'):
            assert same(tr[key], t0[key]), key
        for I in 'HL':
            g = tr['glitch_' + I]
            one = tr['n_glitches_' + I] == 1 if 'n_glitches_' + I in tr else g
            assert 0 < g.sum() < n and one.any(), (extra, I)
            assert np.array_equal(seg['d_' + I][~g], seg0['d_' + I][~g])
            assert all(not np.array_equal(seg['d_' + I][i], seg0['d_' + I][i]) for i in np.flatnonzero(g))
            diff = seg['d_' + I][one] - seg0['d_' + I][one]
            snr2 = 2.0 * np.sum(np.abs(diff) ** 2 / seg['sig2'], axis=1)
            assert np.allclose(snr2, tr['rho_' + I][one] ** 2, rtol=1e-9, atol=0), (extra, I)
            counts.append(int(g.sum()))
    print('\nsegments with glitches (eps H, L; states H, L): %s of %d' % (counts, n))


def test_glitch_states_options(capsys):
    """the command line: --glitch-states with bad pairs, f not summing to 1 or q outside [0, 1], or with --eps, a
    --n-window that does not divide the 32768 samples, the options of the merger stage with --stage windows, and an
    empty range of bins are refused before anything runs; one detector's states alone give the other no glitch"""
    base = ['s', 'cpu', '--n', '3', '--nbar', '0.1', '--out', 'never']
    for extra in (['--glitch-states', '0.5:0.1,0.5'], ['--glitch-states', '0.6:0.1,0.6:0.2'],
                  ['--glitch-states', '1:1.5'], ['--glitch-states', 'a:b'],
                  ['--glitch-states', '1:0.1', '--eps-H', '0.1'], ['--n-window', '48'],
                  ['--stage', 'windows', '--plain'], ['--stage', 'windows', '--glitch-model', 'm.json'],
                  ['--stage', 'windows', '--ymin', '0.1'], ['--stage', 'windows', '--window-rho-min', '5',
                                                           '--window-rho-max', '4']):
        with pytest.raises(SystemExit):
            plant.main(base + extra, factory=lambda seg: None)
        assert 'error' in capsys.readouterr().err, extra
    assert not os.path.exists('never')
    cfg = config('--nbar', '0', '--glitch-states-H', '1:0.05')
    assert cfg['glitch_states']['L'] == [[1.0, 0.0]] and cfg['glitch_states']['H'] == [[1.0, 0.05]]


# ----------------------------------------------------------------------------------------------- the windows stage


def window_stub(seg, rows, n_window):
    """pe.window_evidences in place: ln E_w (n, n_window, n_rho), a deterministic function of the data of the rows (the
    excess power of consecutive blocks of bins, one block per window, times a ramp over the bins), and wmax as the
    Stub's"""
    nr, nm = RHO_EDGES.size - 1, plant.MASSES.size
    out = dict(rho_edges=RHO_EDGES.copy(), rows=np.asarray(rows))
    for I in 'HL':
        p = np.abs(np.asarray(seg['d_' + I])[rows]) ** 2 / seg['sig2']
        pw = np.stack([c.mean(axis=1) for c in np.array_split(p, n_window, axis=1)], axis=1) - 1.0
        out['ln_window_' + I] = pw[:, :, None] * np.linspace(-1, 2, nr)[None, None, :]
        out['wmax_' + I] = p.max(axis=1)[:, None] * np.linspace(0.5, 1, nm)[None]
    return out


class WinStub:
    """the windows stage's factory in place: builds once, records the number of segments of each call"""

    def __init__(self, n_window=plant.N_WINDOW):
        self.calls, self.factories, self.n_window = [], 0, n_window

    def factory(self, seg0):
        self.factories += 1
        return self.win

    def win(self, seg):
        self.calls.append(seg['d_H'].shape[0])
        return window_stub(seg, np.arange(seg['d_H'].shape[0]), self.n_window)


def test_windows_stage_driver(tmp_path):
    """--stage windows next to the merger stage of the same campaign (segments 2 to 8, batches of 3, mixed glitches,
    segment slide): the win files cover the ranges of the plant files, [2, 3), [3, 6), [6, 9), with the same truth and
    wmax; ln_window_I float32 (n, 64, 14) are the stub's on the 14 bins that overlap [3, 12.2] (window_bin0 6,
    window_rho_edges 3 to 14.0); the meta is the campaign without the merger settings, with stage, n_window,
    window_rho, and the campaign_key of both stages is one; the stub is built once and called once per batch; a second
    run calls nothing; one batch of 9 from 0 gives the same per-segment arrays; and a run of another campaign (seed) or
    with another n_window into the folder stops, as does the merger stage of another campaign next to win files only"""
    one = tmp_path / 'one'
    drive(one, '--start', '2', '--n', '7', '--batch', '3')
    win = drive(one, '--start', '2', '--n', '7', '--batch', '3', '--stage', 'windows', stub=WinStub())
    plants = sorted(os.path.basename(f) for f in glob.glob(str(one / 'plant_*.npz')))
    wins = sorted(os.path.basename(f) for f in glob.glob(str(one / 'win_*.npz')))
    assert [w[4:] for w in wins] == [p[6:] for p in plants] == ['%010d_%010d.npz' % r for r in ((2, 3), (3, 6), (6, 9))]
    assert win.factories == 1 and win.calls == [1, 3, 3]
    assert not [f for f in os.listdir(one) if f.startswith('.')]
    cfg = config('--nbar', '0.3', '--eps-H', '0.3', '--eps-L', '0.3', '--glitch-kind', 'mixed', '--control',
                 'segment-slide', '--seed', '5', '--stage', 'windows')
    bins = plant.window_bins(3.0, 12.2)
    for p, w in zip(plants, wins):
        meta_p, meta_w = plant.read_meta(one / p), plant.read_meta(one / w)
        meta_w.pop('run')
        assert meta_w == cfg and meta_w['stage'] == 'windows' and meta_w['n_window'] == 64
        assert meta_w['window_rho'] == [3.0, 12.2] and not set(plant.MERGER_KEYS) & set(meta_w)
        assert plant.campaign_key(meta_w) == plant.campaign_key(meta_p)
        a, b = int(w[4:14]), int(w[15:25])
        seg, truth = plant.simulate_batch(cfg, a, b, plant.Sampler(plant.population()))
        ref = window_stub(seg, np.arange(b - a), 64)
        with np.load(one / p) as fp, np.load(one / w) as fw:
            for key in list(truth) + ['wmax_H', 'wmax_L']:
                assert same(fw[key], fp[key]), key
            for I in 'HL':
                assert fw['ln_window_' + I].dtype == np.float32 and fw['ln_window_' + I].shape == (b - a, 64, 14)
                assert np.array_equal(fw['ln_window_' + I], ref['ln_window_' + I][:, :, bins].astype(np.float32))
            assert int(fw['window_bin0']) == 6 and np.array_equal(fw['window_rho_edges'], RHO_EDGES[6:21])
    again = drive(one, '--start', '2', '--n', '7', '--batch', '3', '--stage', 'windows', stub=WinStub())
    assert again.calls == [] and again.factories == 0
    two = tmp_path / 'two'
    drive(two, '--start', '0', '--n', '9', '--batch', '9', '--stage', 'windows', stub=WinStub())
    with np.load(two / ('win_%010d_%010d.npz' % (0, 9))) as f9:
        for w in wins:
            a, b = int(w[4:14]), int(w[15:25])
            with np.load(one / w) as fw:
                for key in fw.files:
                    if key != 'meta' and fw[key].ndim and fw[key].shape[0] == b - a and key != 'window_rho_edges':
                        assert same(fw[key], f9[key][a:b]), key
    with pytest.raises(SystemExit, match='another campaign.*seed'):
        drive(one, '--start', '0', '--n', '3', '--batch', '3', '--stage', 'windows', '--seed', '6', stub=WinStub())
    with pytest.raises(SystemExit, match='another campaign.*n_window'):
        drive(one, '--start', '2', '--n', '1', '--batch', '3', '--stage', 'windows', '--n-window', '32',
              stub=WinStub(32))
    with pytest.raises(SystemExit, match='another campaign.*nbar'):
        drive(two, '--start', '0', '--n', '3', '--batch', '3', '--nbar', '0.2')
    assert sorted(os.listdir(two)) == ['win_%010d_%010d.npz' % (0, 9)]


def test_windows_analysis_wiring(monkeypatch):
    """make_windows: the bank of plant.MASSES on the segments, and pe.window_evidences called with it, n_window and
    chunk, and no tables"""
    from exmarg.search import pe as pe_module
    seen = []
    monkeypatch.setattr(pe_module, 'window_evidences', lambda seg, bank, **kw: seen.append((seg, bank, kw)) or 'r',
                        raising=False)
    seg, _ = plant.simulate_batch(config('--nbar', '0'), 0, 2, plant.Sampler(plant.population()))
    win = plant.make_windows('cpu', 32, 4)(seg)
    assert win(seg) == 'r'
    (s, bank, kw), = seen
    assert s is seg and np.array_equal(bank.M, plant.MASSES) and kw == dict(chunk=4, n_window=32)


# ----------------------------------------------------------------------------------------------- stage 2, the model


def make_model(path, key, n_window=64, q_scale=1.0):
    """a window model in the format of windows_fit.py, fitted (here: made up) on the campaign of campaign_key key;
    returns its SHA-256"""
    pi = np.zeros((2, RHO_EDGES.size - 1))
    pi[0, 8:19], pi[1, 8:19] = np.linspace(1, 2, 11), np.linspace(2, 1, 11)
    pi /= pi.sum(axis=1, keepdims=True)
    det = lambda f, q: dict(f=f, q=[q[0] * q_scale, q[1]], pi_k=pi.tolist())
    model = dict(model='windows', n_window=n_window, zone=1, pi_m=None, rho_edges=RHO_EDGES.tolist(),
                 H=det([0.8, 0.2], [0.004, 0.05]), L=det([0.9, 0.1], [0.002, 0.08]),
                 settings=dict(fitted_edges=RHO_EDGES[8:20].tolist()), campaign=key)
    return windows_fit.write_model(str(path), json.loads(json.dumps(model)))


def test_model_stage_driver(tmp_path, capsys):
    """--glitch-model with the model of the campaign's own window evidences: pe gets glitch=dict(model 'windows',
    n_window, zone, pi_m, H and L f, q, pi_k of the file); the files hold ln_B_bar, ln_lambda_bar, ln_lambda_H_bar,
    ln_lambda_L_bar (population averages of the grids), ln_c_I (n,), state_I (n, K) and no ln_lambda_g*; the meta
    records glitch_model 'windows', its SHA-256 and n_window; a second run calls nothing.  Refused: another model
    (resuming), a model of another campaign, of another n_window or a run with another --n-window, a modified or a
    missing file.  plant_fit fits these files with the window model and describes the model of the folder"""
    out = tmp_path / 'm'
    drive(out, '--start', '0', '--n', '6', '--batch', '3', '--stage', 'windows', stub=WinStub())
    key = plant.campaign_key(plant.read_meta(out / ('win_%010d_%010d.npz' % (0, 3))))
    path = out / 'glitch_model.json'
    sha = make_model(path, key)
    stub = drive(out, '--start', '0', '--n', '6', '--batch', '3', '--glitch-model', str(path))
    model = plant.read_model(str(path))[0]
    assert len(stub.calls) == 2
    for n, seed, g in stub.calls:
        assert g['model'] == 'windows' and g['n_window'] == 64 and g['zone'] == 1 and g['pi_m'] is None
        for I in 'HL':
            for p in ('f', 'q', 'pi_k'):
                assert np.array_equal(g[I][p], np.asarray(model[I][p])), (I, p)
    cfg = plant.with_model(config('--nbar', '0.3', '--eps-H', '0.3', '--eps-L', '0.3', '--glitch-kind', 'mixed',
                                  '--control', 'segment-slide', '--seed', '5'), sha, 64)
    m1, z = grid()
    for a, b in ((0, 3), (3, 6)):
        meta = plant.read_meta(out / ('plant_%010d_%010d.npz' % (a, b)))
        meta.pop('run')
        assert meta == cfg and meta['glitch_model'] == 'windows' and meta['glitch_model_sha256'] == sha
        seg, _ = plant.simulate_batch(cfg, a, b, plant.Sampler(plant.population()))
        res = Stub().pe(seg, plant.stream(5, a, 'pe'), plant.glitch_model(str(path), cfg, 64)[0])
        with np.load(out / ('plant_%010d_%010d.npz' % (a, b))) as f:
            assert not [k for k in f.files if k.startswith('ln_lambda_g')]
            for k in plant.KEYS_W:
                assert np.allclose(f[k + '_bar'], ln_lambda_bar(res[k], m1, z, plant.MDET, plant.population()),
                                   rtol=0, atol=1e-12), k
            for I in 'HL':
                assert np.array_equal(f['ln_c_' + I], res['ln_c_' + I]) and f['state_' + I].shape == (3, 2)
    again = drive(out, '--start', '0', '--n', '6', '--batch', '3', '--glitch-model', str(path))
    assert again.calls == []
    other = tmp_path / 'other.json'
    make_model(other, key, q_scale=2.0)
    with pytest.raises(SystemExit, match='another campaign.*glitch_model_sha256'):
        drive(out, '--start', '0', '--n', '3', '--batch', '3', '--glitch-model', str(other))
    make_model(other, dict(key, seed=6))
    with pytest.raises(SystemExit, match='fitted on another campaign.*seed'):
        drive(tmp_path / 'x', '--start', '0', '--n', '3', '--batch', '3', '--glitch-model', str(other))
    make_model(other, key, n_window=32)
    with pytest.raises(SystemExit, match='n_window = 32'):
        drive(tmp_path / 'x', '--start', '0', '--n', '3', '--batch', '3', '--glitch-model', str(other))
    with pytest.raises(SystemExit, match='n_window = 64, the run 32'):
        drive(tmp_path / 'x', '--start', '0', '--n', '3', '--batch', '3', '--glitch-model', str(path), '--n-window',
              '32')
    changed = json.loads(open(path).read())
    changed['L']['q'][1] = 0.0801
    with open(other, 'w') as fh:
        json.dump(changed, fh)
    with pytest.raises(SystemExit, match='modified'):
        drive(tmp_path / 'x', '--start', '0', '--n', '3', '--batch', '3', '--glitch-model', str(other))
    with pytest.raises(SystemExit, match='no glitch model'):
        drive(tmp_path / 'x', '--start', '0', '--n', '3', '--batch', '3', '--glitch-model', str(tmp_path / 'none'))
    assert not os.path.exists(tmp_path / 'x') or not os.listdir(tmp_path / 'x')
    r = plant_fit.report(str(out))
    text = '\n'.join(r['lines'])
    assert r['model'] == 'windows' and set(r['fits']) == {'gauss', 'windows'} and r['window_model']['sha256'] == sha
    assert 'window model, ln(1 + nbar (Bbar - 1))' in text and 'state 2 fraction 0.2, 3.2 glitches per segment' in text
    os.remove(path)
    assert 'no glitch_model.json in the folder' in '\n'.join(plant_fit.report(str(out))['lines'])
    make_model(path, key, q_scale=2.0)
    assert 'is another model' in '\n'.join(plant_fit.report(str(out))['lines'])


# ----------------------------------------------------------------------------------------------- the fit of nbar


def synthetic_B(n, nbar, seed, mus=(1.5, 2.5, 3.5), w=(0.5, 0.3, 0.2)):
    """ln Bbar of n segments, a merger in each with probability nbar: x ~ N(mu, 1), mu drawn from mus with weights w,
    with a merger and N(0, 1) without; Bbar = sum_j w_j exp(mu_j x - mu_j^2 / 2), the exact population-averaged
    likelihood ratio of x (its mean over segments without a merger is 1).  returns (ln Bbar, merger flags)"""
    rng = np.random.default_rng(seed)
    mus, w = np.asarray(mus, float), np.asarray(w, float)
    m = rng.random(n) < nbar
    mu = mus[np.minimum(np.searchsorted(np.cumsum(w), rng.random(n), side='right'), mus.size - 1)]
    x = rng.standard_normal(n) + np.where(m, mu, 0.0)
    return logsumexp(np.log(w)[None] + mus[None] * x[:, None] - 0.5 * mus[None] ** 2, axis=1), m


def test_nbar_fit_recovers_and_covers():
    """nbar_fit on synthetic Bbar from a known mixture: 20000 segments at nbar 0.05 recover it within 2.5 standard
    errors and noise alone gives 0 within 2.5 (or the lower edge); the estimate is the maximum of sum ln(1 + nbar (B -
    1)) on a grid of 2e5 points and its error 1 / sqrt(-d^2 ln L / d nbar^2) by finite differences (1e-4); over 400
    replicas of 4000 segments the pulls have mean within 4 / sqrt(400) of 0 and 91 to 99.5 percent within 2"""
    lnB, m = synthetic_B(20000, 0.05, 1)
    r = plant_fit.nbar_fit(lnB)
    msg = 'nbar %.5f +- %.5f (truth 0.05, %d mergers), ln L %.3f' % (r['nbar'], r['err']['nbar'], m.sum(), r['lnl'])
    assert abs(r['nbar'] - 0.05) < 2.5 * r['err']['nbar'] and r['edge'] is None, msg
    lnl = lambda x: mixture.log_likelihood(x, lnB, 0.0, 0.0, log=True)
    g = np.linspace(0.0, 0.2, 200001)
    best = g[np.argmax([lnl(x) for x in g[np.abs(g - r['nbar']) < 0.002]]) + np.flatnonzero(
        np.abs(g - r['nbar']) < 0.002)[0]]
    assert abs(best - r['nbar']) < 2e-6 and abs(r['lnl'] - lnl(r['nbar'])) < 1e-9
    h = 1e-4
    curv = (lnl(r['nbar'] + h) - 2 * lnl(r['nbar']) + lnl(r['nbar'] - h)) / h ** 2
    assert abs(r['err']['nbar'] * np.sqrt(-curv) - 1) < 1e-4, (r['err']['nbar'], 1 / np.sqrt(-curv))
    z0 = plant_fit.nbar_fit(synthetic_B(20000, 0.0, 2)[0])
    msg += '; noise: nbar %.5f +- %.5f' % (z0['nbar'], z0['err']['nbar'])
    assert plant_fit._consistent_with_zero(z0)[0], msg
    pulls = []
    for i in range(400):
        lb, _ = synthetic_B(4000, 0.05, 100 + i)
        f = plant_fit.nbar_fit(lb)
        pulls.append((f['nbar'] - 0.05) / f['err']['nbar'])
    pulls = np.array(pulls)
    msg += '; 400 replicas: mean pull %+.3f, sd %.3f, within 2: %.3f' % (pulls.mean(), pulls.std(), np.mean(
        np.abs(pulls) < 2))
    print('\n' + msg)
    assert abs(pulls.mean()) < 4 / np.sqrt(400) and 0.91 <= np.mean(np.abs(pulls) < 2) <= 0.995, msg


def test_nbar_fit_edges():
    """B below 1 everywhere: nbar 0 at the lower edge with the curvature error; B above 1 everywhere: nbar 1 at the
    upper edge; one segment with B overflowing (ln B = 800) among noise: a finite estimate inside (0, 1); B exactly 1
    counts nothing; ln B nan raises"""
    low = plant_fit.nbar_fit(np.full(100, -0.5))
    assert low['nbar'] == 0.0 and low['edge'] == 'lower' and np.isclose(low['err']['nbar'], 1 / np.sqrt(100 * np.expm1(
        -0.5) ** 2))
    high = plant_fit.nbar_fit(np.full(100, 2.0))
    assert high['nbar'] == 1.0 and high['edge'] == 'upper'
    lnB, _ = synthetic_B(5000, 0.0, 3)
    huge = plant_fit.nbar_fit(np.r_[lnB, 800.0])
    assert 0 < huge['nbar'] < 1 and np.isfinite(huge['err']['nbar']) and huge['edge'] is None
    assert plant_fit.nbar_fit(np.r_[lnB, np.zeros(50)])['nbar'] == pytest.approx(
        plant_fit.nbar_fit(lnB)['nbar'], abs=1e-12)
    with pytest.raises(ValueError):
        plant_fit.nbar_fit(np.r_[lnB, np.nan])


def write_windows_files(directory, cfg, lnB, lnL, m, wmax, splits):
    """synthetic files of the merger stage with the window model (the members of plant.py) under the campaign cfg"""
    n = lnB.size
    nr = RHO_EDGES.size - 1
    per = dict(k=np.arange(n), ln_B_bar=lnB, ln_lambda_bar=lnL, ln_lambda_H_bar=0.5 * lnL, ln_lambda_L_bar=0.5 * lnL,
               w_inf=np.zeros(n), merger=m, merger_H=m, merger_L=m, coherent=m & (cfg['control'] == 'none'),
               m1=np.full(n, np.nan), z=np.full(n, np.nan), y=np.zeros(n), snr=np.zeros(n))
    for I in 'HL':
        per.update({'ln_c_' + I: np.zeros(n), 'state_' + I: np.tile([0.9, 0.1], (n, 1)), 'ln_glitch_' + I:
                    np.zeros((n, nr)), 'wmax_' + I: wmax[I], 'glitch_' + I: np.zeros(n, bool),
                    'kind_' + I: np.full(n, ''), 'rho_' + I: np.full(n, np.nan), 't0_' + I: np.full(n, np.nan)})
    os.makedirs(directory, exist_ok=True)
    for a, b in splits:
        plant.write_batch(os.path.join(directory, 'plant_%010d_%010d.npz' % (a, b)), cfg,
                          dict(first=a, end=b, t_sim=1e-3, t_pe=0.1, t_plain=None),
                          dict({key: v[a:b] for key, v in per.items()}, rho_edges=RHO_EDGES,
                               glitch_pi_k=np.array(np.nan), glitch_window=np.array(np.nan),
                               w_inf_cells=np.zeros((0, 2), np.int32)))


@pytest.fixture(scope='module')
def windows_dirs(tmp_path_factory):
    """synthetic window-model campaigns: Bbar of synthetic_B at nbar 0.05 (12000 segments, some loud), declared at the
    truth, at 0.2 and as a time-slide control; noise alone declared a control.  The Gaussian likelihood is Bbar times a
    factor e^3 in the 5 percent of segments that hold a 'glitch'"""
    root = tmp_path_factory.mktemp('wmodel')
    n = 12000
    lnB, m = synthetic_B(n, 0.05, 7)
    rng = np.random.default_rng(8)
    lnL = lnB + 3.0 * (rng.random(n) < 0.05)
    wmax = {I: rng.exponential(15.0, n) for I in 'HL'}
    out = {}
    for name, nbar, control, data in (('plant', 0.05, 'none', (lnB, m)), ('wrong', 0.2, 'none', (lnB, m)),
                                      ('control', 0.05, 'time-slide', (lnB, m)),
                                      ('null', 0.05, 'time-slide', synthetic_B(n, 0.0, 9))):
        cfg = plant.with_model(config('--nbar', str(nbar), '--control', control), 'ab' * 32, 64)
        path = str(root / name)
        write_windows_files(path, cfg, data[0], lnL, data[1], wmax, ((0, 5000), (5000, n)))
        out[name] = (path, data[0], lnL, data[1], wmax)
    return out


def test_plant_fit_window_model(windows_dirs, capsys):
    """plant_fit on files of the window model: the segments with wmax above 72 set aside; the window-model row is
    nbar_fit on ln Bbar of the kept segments, the Gaussian row mixture.fit at eps = 0 on ln Lambda_bar (the same with
    any lambda given a glitch: they do not enter); the time curve that of Bbar; the unit mean of Bbar - 1 over the kept
    segments without a merger, with the number of segments that carry half of sum |Bbar - 1|, each printed; the
    criterion 'window model nbar within 2.5 sigma of the truth' is that comparison (passes at the truth, fails declared
    at 0.2, exit status 1); declared a control with mergers in it, 'window model: nbar consistent with 0' fails, and
    noise alone passes both control criteria (exit status 0)"""
    path, lnB, lnL, m, wmax = windows_dirs['plant']
    r = plant_fit.report(path)
    keep = (wmax['H'] <= 72) & (wmax['L'] <= 72)
    assert np.array_equal(r['keep'], keep) and 20 < (~keep).sum() < 400 and r['model'] == 'windows'
    W, G = r['fits']['windows'], r['fits']['gauss']
    ref = plant_fit.nbar_fit(lnB[keep])
    assert W['nbar'] == ref['nbar'] and W['err']['nbar'] == ref['err']['nbar']
    for lam_g in (lnL[keep], np.random.default_rng(1).normal(0, 3, keep.sum())):
        d = mixture.fit(lnL[keep], lam_g, lam_g, np.zeros((keep.sum(), RHO_EDGES.size - 1)),
                        np.zeros((keep.sum(), RHO_EDGES.size - 1)), RHO_EDGES, log=True, lam_gHL=lam_g,
                        fixed=tuple(p for p in mixture.PARAMETERS if p != 'nbar'),
                        start=dict(plant_fit.START, eps_H=0.0, eps_L=0.0, rho_min_H=4.0, rho_min_L=4.0))
        assert G['nbar'] == d['nbar'] and G['err']['nbar'] == d['err']['nbar'] and G['lnl'] == d['lnl']
    text = '\n'.join(r['lines'])
    for F in (W, G):
        assert '%.5g +- %.5g' % (F['nbar'], F['err']['nbar']) in text
    assert np.allclose(r['curves']['windows'], mixture.time_curve(lnB[keep], plant_fit.NBAR_GRID, lambda g: 1 / g,
                                                                  log=True), rtol=0, atol=1e-12)
    quiet = ~m[keep]
    x = np.expm1(lnB[keep][quiet])
    dev = np.sort(np.abs(x))[::-1]
    carry = int(np.searchsorted(np.cumsum(dev), 0.5 * dev.sum())) + 1
    mu, se, c = r['unit']['all']['Bbar - 1, window model']
    assert c == quiet.sum() and np.isclose(mu, x.mean(), rtol=1e-12) and np.isclose(se, x.std(ddof=1) / np.sqrt(c))
    assert '%+.3e +- %.2e  (%d, %+.1f sigma); half of sum |x - 1| in %d' % (x.mean(), se, c, x.mean() / se, carry) \
        in text and abs(x.mean()) < 3 * se
    crit = {name: ok for name, ok, _, _ in r['criteria']}
    assert crit['window model nbar within 2.5 sigma of the truth'] == (abs(W['nbar'] - 0.05) < 2.5 * W['err']['nbar'])
    assert crit['window model nbar within 2.5 sigma of the truth'] and r['ok']
    print('\n' + text)
    r = plant_fit.report(windows_dirs['wrong'][0])
    assert not {name: ok for name, ok, _, _ in r['criteria']}['window model nbar within 2.5 sigma of the truth']
    capsys.readouterr()
    assert plant_fit.main([windows_dirs['wrong'][0]]) == 1
    r = plant_fit.report(windows_dirs['control'][0])
    crit = {name: (ok, counted) for name, ok, _, counted in r['criteria']}
    assert r['kind'] == 'control' and crit['window model: nbar consistent with 0'] == (False, True)
    c = r['curves']['windows']
    assert crit['window model: evidence below ln 100'] == (bool(c.max() < np.log(100)), True)
    r = plant_fit.report(windows_dirs['null'][0])
    crit = {name: (ok, counted) for name, ok, _, counted in r['criteria']}
    assert crit['window model: nbar consistent with 0'] == (True, True) and crit[
        'window model: evidence below ln 100'] == (True, True), r['lines'][-6:]
    assert plant_fit.main([windows_dirs['null'][0]]) == 0


# ----------------------------------------------------------------------------------------------- unequal noise curves


def test_unequal_noise_plant(tmp_path):
    """--asym W: the noise of detector I is that of the campaign without it times sqrt(sig2_I / sig2) (the same random
    numbers), sig2_H = sig2 / (1 + W), sig2_L = sig2 / (1 - W), the mergers the same signals with their optimal SNR on
    those curves, the template glitches scaled to their recorded SNR on the curve of their detector, and every template
    of the bank at w = W.  --curves: the curves of a conditioned epoch file (strain.own_sig2), refused once the file
    changes"""
    from exmarg.search import mass_bank
    from test_analyse import make_unequal_epoch
    s = plant.Sampler(plant.population())
    W = -0.3
    base = ('--nbar', '1', '--seed', '3')
    seg0, tr0 = plant.simulate_batch(config(*base), 0, 4, s)
    seg1, tr1 = plant.simulate_batch(config(*base, '--asym', str(W)), 0, 4, s)
    k = {'H': 1.0 / (1.0 + W), 'L': 1.0 / (1.0 - W)}
    assert 'sig2' not in seg1 and np.allclose(seg1['sig2_H'], seg0['sig2'] * k['H'], rtol=1e-15, atol=0)
    snr = 0.0
    for I in 'HL':                                    # d1 = h + sqrt(k) n, d0 = h + n
        h = (seg1['d_' + I] - np.sqrt(k[I]) * seg0['d_' + I]) / (1.0 - np.sqrt(k[I]))
        snr = snr + np.sum(2.0 * np.abs(h) ** 2 / seg1['sig2_' + I], axis=1)
    assert tr0['merger'].all() and np.allclose(np.sqrt(snr), tr1['snr'], rtol=1e-8)
    assert np.array_equal(tr0['m1'], tr1['m1']) and not np.allclose(tr0['snr'], tr1['snr'])
    bank = mass_bank(plant.MASSES[::30], seg1)
    assert np.abs(bank.asym - W).max() < 1e-12
    g0, _ = plant.simulate_batch(config('--nbar', '0', '--asym', str(W), '--seed', '3'), 0, 6, s)
    g1, tg = plant.simulate_batch(config('--nbar', '0', '--asym', str(W), '--seed', '3', '--eps-H', '1', '--eps-L', '1',
                                         '--rho-min', '8'), 0, 6, s)
    for I in 'HL':
        rho = np.sqrt(np.sum(2.0 * np.abs(g1['d_' + I] - g0['d_' + I]) ** 2 / g1['sig2_' + I], axis=1))
        assert tg['glitch_' + I].all() and np.allclose(rho, tg['rho_' + I], rtol=1e-10)
    ep, _ = make_unequal_epoch(1266933768, 5)
    path = str(tmp_path / 'epoch.npz')
    np.savez(path, **ep)
    cfg = config('--nbar', '0', '--curves', path)
    assert cfg['noise']['curves'] == path and len(cfg['noise']['sha256']) == 64
    seg2, _ = plant.simulate_batch(cfg, 0, 2, s)
    from exmarg import strain
    want = strain.own_sig2(ep)
    assert all(np.array_equal(seg2['sig2_' + I], want[I]) for I in 'HL')
    np.savez(path, **dict(ep, sig2_ref_L=ep['sig2_ref_L'] * 1.01))
    plant._CURVES.clear()
    with pytest.raises(SystemExit, match='changed since the campaign'):
        plant.simulate_batch(cfg, 0, 2, s)


class UStub(Stub):
    """the Stub for segments on two curves (sig2_H, sig2_L): its arrays with the curve of each detector"""

    def pe(self, seg, seed, glitch):
        res = {I: Stub.pe(self, dict(seg, sig2=seg['sig2_' + I]), seed, glitch) for I in 'HL'}
        del self.calls[-1]
        out = dict(res['H'], asym_outside=1)
        if glitch:
            out.update(ln_glitch_L=res['L']['ln_glitch_L'], wmax_L=res['L']['wmax_L'])
        return out


def test_unequal_driver(tmp_path, capsys, monkeypatch):
    """plant.main --asym with two store folders, --zmin-cut and --asym-tol 0.03: the files record noise, tables_asym,
    zmin_cut, asym_tol and per segment asym_outside, and plant_fit reads and fits them; another --asym into the folder
    stops; make_analysis passes the list of stores, zmin and asym_tol to pe_redshift; a default campaign records none of
    them"""
    from test_store import write_store
    stores = '%s,%s' % (write_store(str(tmp_path / 's3'), asym=-0.3), write_store(str(tmp_path / 's6'), asym=-0.6))
    out = tmp_path / 'u'
    stub = UStub()
    assert plant.main([stores, 'cpu', '--nbar', '0.3', '--eps-H', '0.3', '--n', '6', '--batch', '3', '--seed', '5',
                       '--asym', '-0.45', '--zmin-cut', '--asym-tol', '0.03', '--out', str(out)], factory=stub.factory) == 0
    assert len(stub.calls) == 2
    cfg, _, a, _, _ = plant_fit.load(str(out))
    assert cfg['noise'] == dict(asym=-0.45) and cfg['tables_asym'] == [-0.6, -0.3] and cfg['zmin_cut'] is True
    assert cfg['asym_tol'] == 0.03 and a['k'].size == 6 and np.array_equal(a['asym_outside'], np.ones(6))
    assert not {'noise', 'tables_asym', 'asym_tol', 'zmin_cut'} & set(config('--nbar', '0'))
    capsys.readouterr()
    plant_fit.main([str(out)])
    assert 'nbar' in capsys.readouterr().out
    with pytest.raises(SystemExit, match='another campaign.*noise'):
        plant.main([stores, 'cpu', '--nbar', '0.3', '--eps-H', '0.3', '--n', '6', '--batch', '3', '--seed', '5',
                    '--asym', '-0.4', '--zmin-cut', '--asym-tol', '0.03', '--out', str(out)], factory=UStub().factory)
    import exmarg.search as search
    seen = []
    monkeypatch.setattr(search, 'pe_redshift', lambda seg, bank, store, **kw: seen.append((store, kw)))
    seg, _ = plant.simulate_batch(config('--nbar', '0', '--asym', '-0.45'), 0, 2, plant.Sampler(plant.population()))
    plant.make_analysis(['a', 'b'], 'cpu', zmin=1.0, asym_tol=0.05)(seg)(seg, 17, True)
    assert seen[0][0] == ['a', 'b'] and seen[0][1]['zmin'] == 1.0 and seen[0][1]['asym_tol'] == 0.05
