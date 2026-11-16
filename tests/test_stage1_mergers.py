"""controls 1 to 4 of the mergers in the single-detector fit (docs/2026-11-15_mergers_in_stage1.md, Sec. 5):

1. N = 0 changes nothing: fit_windows, window_evidences and the model file of windows_fit.py are those of the
   committed code bit for bit, of the branch before the mergers (ee47f02) and of the main branch it was merged with
   (a79e07d: the batched tables and the signal gain; HEADS below);
2. unit mean: the merger evidence M_w averages to 1 over Gaussian noise (Monte Carlo on white matched-filter outputs
   drawn here, and quadrature of the columns), and the weights W of the population sum to the fraction of the
   population that stage 2 covers (computed by stage 2's own node geometry and reduction);
3. the toy with known truth of test_windows.py (one template, white noise, 32 windows, two states, the strong mergers
   (X_S, P_S) at NBAR = 0.05, data as the model states): the fit without mergers is biased in the loud bins; the fit
   at the true N recovers the planted parameters; the iteration of Sec. 3 from N = 0 reaches the N of the analysis
   with the true glitch parameters in two passes, and a third does not move it.  M_w there is the toy's own exact
   single-detector ratio, (1/NWIN) sum_{t in w} S^I(t) (test_windows.evidences), and stage 2 the test's own ln_B
   (test_windows.ln_B); the objective with mergers is also computed here from single.window_posterior of one state
   at a time, not by mixture.py;
4. marginal identity: B of stage 2 (pe._node_values on the incoherent store of test_pe_windows.py, whose H data
   integrated out leave the L-only columns), averaged over H noise drawn from its window model, equals the
   single-detector ratio of L, c_L(theta, 1) / c_L(theta, 0) = sum_s r_s (1 + b^s), of the objective that stage 1
   maximizes.

Each test names, in its docstring, the deliberate error it was seen to fire on (applied to a scratch copy of the code).
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import chi2

from exmarg.pp import simulate
from exmarg.search import mass_bank, mixture, pe, population as popavg, read_segments, single
import test_windows as tw
import test_pe_windows as tpw

ROOT = Path(__file__).resolve().parents[1]
HEADS = ('ee47f02', 'a79e07d')          # the branch before the mergers in stage 1; the main branch merged into it
S_LADDER = np.load(ROOT / 'data' / 'tables' / 's_grid.npz')['s_nodes']       # the 53 nodes of the redshift tables


def _script(name, folder='validate'):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / folder / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=None)
def committed(ref=HEADS[0]):
    """the tree of the commit ref (src and scripts) exported to a temporary folder, or None without git history; the
    environment variable EXMARG_TREE_<ref> (EXMARG_HEAD_TREE for the first of HEADS) may name such an export instead (a
    copy of the code outside the repository)"""
    import tempfile
    env = os.environ.get('EXMARG_TREE_' + ref) or (os.environ.get('EXMARG_HEAD_TREE') if ref == HEADS[0] else None)
    if env:
        return Path(env)
    out = Path(tempfile.mkdtemp(prefix='exmarg_%s_' % ref))
    try:
        tar = subprocess.run(['git', 'archive', ref, 'src', 'scripts'], cwd=ROOT, capture_output=True, check=True).stdout
        subprocess.run(['tar', '-x', '-C', str(out)], input=tar, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out


def committed_module(rel, name, ref=HEADS[0]):
    root = committed(ref)
    if root is None:
        pytest.skip('no git history to compare with')
    spec = importlib.util.spec_from_file_location(name + '_' + ref, root / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    return set(a) == set(b) and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


# ================================================================================ 1. N = 0 changes nothing


@pytest.mark.parametrize('ref', HEADS)
def test_c1_fit_windows_unchanged(ref):
    """fit_windows without mergers (ln_M None, or nbar 0 with ln_M given) returns every number of the committed
    function bit for bit: the first 8000 segments of the toy of test_windows.py without and with its mergers (H, the
    default start), from a poor
    start, one and three states, a basis of three laws, and the small data of test_windows.stall_data.  Fires on: the
    SQUAREM cycles run until one gains less than 1e-3 nats (in place of 1e-2), which changes the path of the updates
    and with it the last bits of the maximum"""
    old = committed_module('src/exmarg/search/mixture.py', 'exmarg.search._mixture_head', ref)
    rec = {k: {I: v['E' + I][:8000] for I in 'HL'} for k, v in tw.recovery_data().items() if k != 'flags'}
    rec = {k: {'E' + I: v[I] for I in 'HL'} for k, v in rec.items()}
    stall = tw.stall_data()
    a = np.full((2, tw.NK), 1e-3)
    a[:, -1] = 1.0 - 1e-3 * (tw.NK - 1)
    basis = np.array([tw.law(tw.EDGES, g, 2.0) for g in (2.0, 3.0, 5.0)])
    cases = [(rec['free']['EH'], {}), (rec['model']['EH'], {}),
             (rec['model']['EL'], dict(start=dict(f=[0.5, 0.5], q=[0.3, 1e-4], a=a))),
             (rec['free']['EL'], dict(n_state=1)), (stall, dict(n_state=3)), (rec['model']['EH'], dict(basis=basis)),
             (stall, dict(start=dict(f=[0.7, 0.3], q=[0.02, 0.2])))]
    lines = []
    for E, kw in cases:
        want = old.fit_windows(E, **kw)
        lnM = np.zeros(E.shape[:2])
        for extra in ({}, dict(ln_M=lnM, nbar=0.0), dict(ln_M=None, nbar=0.3)):
            got = mixture.fit_windows(E, **kw, **extra)
            assert same(got, want), (list(kw), list(extra))
        lines.append('%s: ln L %.6f, %d updates, converged %s: bit for bit' % (sorted(kw), want['lnl'], want['n_iter'],
                                                                             want['converged']))
    print('\n' + '\n'.join(lines))


def tiny_population(bank, ladder, zmin=0.3):
    """a population on a small (m1, z) grid over the masses of a bank (the merger evidence of control 1)"""
    m1 = np.geomspace(bank.M[0] / 2.2, bank.M[-1] / 2.0, 24)
    z = np.linspace(0.05, 2.0, 30)
    w = np.where(z[None, :] >= zmin, np.exp(-z)[None, :] / m1[:, None], 0.0)
    Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
    w = np.where((Md >= bank.M[0]) & (Md <= bank.M[-1]), w, 0.0)
    return popavg.merger_population(m1, z, w / w.sum(), ladder, 0.01)


@pytest.mark.parametrize('ref', HEADS)
def test_c1_window_evidences_unchanged(tpw_data, ref):
    """pe.window_evidences without a population returns the arrays of the committed pe.py bit for bit (the four
    segments of test_pe_windows.py, uniform and given template weights); with a population the same arrays bit for
    bit, and ln_merger_H, ln_merger_L (4, 64) besides.  Fires on: the
    template weights pi_m dropped (uniform) when a population is given"""
    old = committed_module('src/exmarg/search/pe.py', 'exmarg.search._pe_head_c1', ref)
    c = tpw_data
    pop = tiny_population(c['bank'], S_LADDER)
    for pm in (None, np.array([0.2, 0.5, 0.3])):
        want = old.window_evidences(c['segs'], c['bank'], n_window=64, pi_m=pm, chunk=3)
        assert same(pe.window_evidences(c['segs'], c['bank'], n_window=64, pi_m=pm, chunk=3), want)
        got = pe.window_evidences(c['segs'], c['bank'], n_window=64, pi_m=pm, chunk=3, population=pop)
        extra = {'ln_merger_H', 'ln_merger_L', 'merger_covered_H', 'merger_covered_L'}
        assert set(got) == set(want) | extra and same({k: got[k] for k in want}, want)
        assert got['ln_merger_H'].shape == (4, 64) and np.isfinite(got['ln_merger_L']).all()
    print('\nwindow_evidences: the arrays of %s bit for bit, with and without a population; covered fraction %.6f'
          % (ref, got['merger_covered_H']))


@pytest.fixture(scope='module')
def tpw_data():
    """the segments, bank and store of test_pe_windows.py"""
    y = iter([12.0, 1e-8])
    raw, _, _ = simulate(2, 77, lambda r: (60.0, 1.0, 0.0), lambda r: next(y), 4.0, 20.0, 128.0)
    dt = raw['T'] / 4096
    tm = 1598 * dt
    g = lambda d, t, rho, ph: tpw.add_glitch(d, raw, (t % raw['T']) // dt * dt, rho, 60.0, ph)
    dH = [g(g(raw['d_H'][0], tm + 0.05, 15.0, 0.3), tm + 1.3, 20.0, 1.1), raw['d_H'][1], raw['d_H'][1], raw['d_H'][1]]
    one = g(raw['d_L'][1], 1.0, 12.0, 0.7)
    dL = [g(raw['d_L'][0], tm + 2.2, 12.0, 2.0), raw['d_L'][1], one, g(one, 2.6, 12.0, 2.5)]
    segs = dict(raw, d_H=np.array(dH), d_L=np.array(dL))
    return dict(segs=segs, bank=mass_bank(tpw.MASSES, segs))


def write_toy_folder(folder, with_mergers):
    """the toy campaign of test_windows_fit.py (2000 segments, three files) in win files; with_mergers: also
    ln_merger_I and the merger_population of the campaign, as the windows stage with the store writes them"""
    import test_windows_fit as twf
    sim = twf.simulate(2000, 20261116)
    cfg = twf.config()
    if with_mergers:
        plant = twf.plant
        cfg = dict(cfg, merger_population=plant.population_settings(cfg['zmin'], plant.default_ymin(), S_LADDER))
    twf.write(folder, cfg, sim, ((0, 700), (700, 1400), (1400, 2000)))
    if with_mergers:
        rng = np.random.default_rng(1)
        for path in sorted(Path(folder).glob('win_*.npz')):
            with np.load(path) as z:
                arrays = {key: z[key] for key in z.files if key != 'meta'}
                meta = json.loads(str(z['meta']))
            n = arrays['k'].size
            for I in 'HL':
                arrays['ln_merger_' + I] = rng.normal(0.0, 0.3, (n, twf.NW))
            twf.plant.write_batch(str(path), {k: v for k, v in meta.items() if k != 'run'}, meta['run'], arrays)
    return cfg


@pytest.mark.parametrize('ref', HEADS)
def test_c1_model_file_unchanged(tmp_path, capsys, ref):
    """windows_fit.py with --nbar 0 (and without --nbar) writes, byte for byte, the glitch_model.json that the
    committed windows_fit.py writes on win files of the windows stage as they were (no merger evidence); and the same
    file on files that also hold ln_merger_I and the merger_population of the campaign (the windows stage with the
    store: merger_population is a setting of the stage, not of the campaign key the model records).  Fires on: the
    model recording nbar 0 (the entry written whatever nbar)"""
    root = committed(ref)
    if root is None:
        pytest.skip('no git history to compare with')
    wf = _script('windows_fit')
    files = {}
    for tag, mergers in (('old', False), ('new', True)):
        folder = tmp_path / tag
        write_toy_folder(str(folder), mergers)
        if not mergers:
            subprocess.run([sys.executable, str(root / 'scripts' / 'validate' / 'windows_fit.py'), str(folder),
                            '--record', '1'], check=True, capture_output=True)
            files[tag, 'head'] = (folder / 'glitch_model.json').read_bytes()
        for argv in ([], ['--nbar', '0']):
            if (folder / 'glitch_model.json').exists():
                os.remove(folder / 'glitch_model.json')
            assert wf.main([str(folder), '--record', '1'] + argv) == 0
            files[tag, 'now' + ''.join(argv)] = (folder / 'glitch_model.json').read_bytes()
    capsys.readouterr()
    ref = files['old', 'head']
    assert all(v == ref for v in files.values()), [k for k, v in files.items() if v != ref]
    print('\nglitch_model.json: %d versions byte for byte equal (SHA-256 %s)' % (len(files), json.loads(ref)['sha256']))


# ================================================================================ 2. unit mean


def plant_bank():
    """the bank of plant.py (120 templates, 20 to 400 M_sun) on its noise curve"""
    plant = _script('plant')
    seg, _, _ = simulate(1, 5, lambda r: (60.0, 1.0, 0.0), lambda r: 1e-8, plant.T, plant.FMIN, plant.FMAX)
    return plant, mass_bank(plant.MASSES, seg)


def stage2_covered(plant, bank, zmin):
    """the population weight of the cells where stage 2 has a finite ln Lambda, from its own node geometry and
    reduction (pe._node_geometry, pe._to_source_frame on zero node values), store of the 53 nodes and the a-range of
    the published tables, plant's grid and ymin"""
    class Store:
        s, alpha_lo, alpha_hi = S_LADDER, np.full(S_LADDER.size, 0.16), np.full(S_LADDER.size, 4e4)
    g = plant.GRID
    z, m1 = np.linspace(0.02, g['zmax'], g['nz']), np.geomspace(*g['m1_range'], g['nm1'])
    geo = pe._node_geometry(bank, Store, plant.default_ymin())
    lam = np.concatenate([np.where(geo[0], 0.0, -np.inf), np.where(geo[3] >= 0, 0.0, -np.inf)[None]])
    surf = pe._to_source_frame(lam, geo[2], bank.M, pe.D0 / pe._dl_of_z(z), m1, z, np)
    w = popavg.population_weights(m1, z, plant.MDET, plant.population(zmin))
    return float(w[np.isfinite(surf)].sum())


def test_c2_weights_sum_to_covered_fraction():
    """the weights of the plant's population on the bank of plant.py and the 53 nodes of the tables sum to the
    population weight of the cells that stage 2 covers, to 1e-12: z >= 0.02 (the cells of small z where the last node
    does not reach y for the heavy templates are left out: 1 - 3.068e-5, the w_inf that the plant W_F02 recorded) and
    z >= 0.99119 (everything covered); W >= 0, the same in H and L (equal curves).  Fires on: the cell weights given to
    both bracketing templates in full (the factor 1 - t dropped), sum 1.4"""
    plant, bank = plant_bank()
    lines = []
    for zmin in (0.02, 0.99119):
        pop = plant.merger_population(plant.population_settings(zmin, plant.default_ymin(), S_LADDER))
        W = popavg.merger_weights(bank.M, bank.u_H, pop)
        assert np.all(W >= 0) and np.array_equal(W, popavg.merger_weights(bank.M, bank.u_L, pop))
        cov = stage2_covered(plant, bank, zmin)
        lines.append('zmin %g: sum W %.12f, stage 2 covers %.12f (1 - %.4g); %d nodes and %d templates with weight'
                     % (zmin, W.sum(), cov, 1 - cov, (W.sum(0) > 0).sum(), (W.sum(1) > 0).sum()))
        assert abs(W.sum() - cov) < 1e-12
    print('\n' + '\n'.join(lines))
    assert abs(1 - stage2_covered(plant, bank, 0.02) - 3.068e-5) < 1e-8


def test_c2_weights_concentrated():
    """a population on one cell whose M_det is that of template m and whose y = D0 / D_L(z) is node k of template m
    (s_k = u_m y^2): W is 1 at (m, k) and 0 elsewhere (to 1e-12); a cell midway in M_det between templates m and m + 1
    gives 1/2 to each template, shared between the two nodes of each that bracket y, linearly in y, the nodes of each
    template its own (computed here from s / u); a cell above the bank or beyond the ladder gives nothing.  Fires on: the nodes bracketed linearly
    in s instead of y"""
    plant, bank = plant_bank()
    from exmarg.search.segments import D0
    m, k, z0 = 57, 20, 1.3
    y0 = float(D0 / pe._dl_of_z(z0)[0])
    s = bank.u_H[m] * y0 ** 2 * 1.3 ** (np.arange(40) - k)
    one = lambda m1, z: popavg.merger_population(np.array([m1 / 1.1, m1, m1 * 1.1]), np.array([z - 0.1, z, z + 0.1]),
                                                 np.eye(3)[1][:, None] * np.eye(3)[1][None, :], s, 0.0)
    W = popavg.merger_weights(bank.M, bank.u_H, one(bank.M[m] / (2 * (1 + z0)), z0))
    want = np.zeros_like(W)
    want[m, k] = 1.0
    assert np.abs(W - want).max() < 1e-12, np.abs(W - want).max()
    # midway: M_det between templates m, m + 1 and y midway between the bracketing nodes of each template
    Md = 0.5 * (bank.M[m] + bank.M[m + 1])
    W = popavg.merger_weights(bank.M, bank.u_H, one(Md / (2 * (1 + z0)), z0))
    want = np.zeros_like(W)
    for mm in (m, m + 1):
        yk = np.sqrt(s / bank.u_H[mm])
        j = min(np.searchsorted(yk, y0, side='right') - 1, s.size - 2)
        tau = (y0 - yk[j]) / (yk[j + 1] - yk[j])
        want[mm, j] += 0.5 * (1 - tau)
        want[mm, j + 1] += 0.5 * tau
    assert np.abs(W - want).max() < 1e-12 and abs(W.sum() - 1.0) < 1e-12
    assert popavg.merger_weights(bank.M, bank.u_H, one(bank.M[-1] * 1.2 / 2.6, z0)).sum() == 0.0      # above the bank
    s_short = s[:k - 5]
    pop = popavg.merger_population(np.array([1.0, 2.0, 3.0]) * bank.M[m] / (4 * (1 + z0)), np.array([z0 - 0.1, z0,
                                   z0 + 0.1]), np.eye(3)[1][:, None] * np.eye(3)[1][None, :], s_short, 0.0)
    assert popavg.merger_weights(bank.M, bank.u_H, pop).sum() == 0.0                   # beyond the ladder
    print('\nconcentrated population: W = 1 at (m %d, k %d); the cell midway between two templates as computed here' % (m, k))


def white_trajectories(rng, u, n_time):
    """white complex Gaussian matched-filter outputs of power u_m per template (bank.trajectories' layout)"""
    z = (rng.standard_normal((u.size, n_time)) + 1j * rng.standard_normal((u.size, n_time))) / np.sqrt(2.0)
    v = z * np.sqrt(u)[:, None]
    return np.log(np.abs(v)).astype(np.float32), np.abs(z) ** 2


def test_c2_unit_mean():
    """M_w averages to the covered fraction over Gaussian noise.  Quadrature: every merger column of the 53 nodes has
    unit mean over w ~ Exp(1) to 1e-6 (Simpson's rule on the grid of single.V), so the mean of M_w is sum W.  Monte Carlo:
    single.window_merger on 2000 segments of white matched-filter outputs (3 templates of powers 2.3, 4.2, 9.1, 4096
    arrival times, 64 windows), W uniform on the 3 templates and the nodes s = 0.5, 2 (the incoherent store of
    test_pe_windows.py; sum W = 1): the mean over the windows is 1 within 3 standard errors (se < 1e-3); and ln M_w
    agrees with the direct sum of Phi at each arrival time (single._ln_phi, no interpolation) to 3e-6.  Fires on: the
    weights read as W / n_template (each template's evidence not weighted by the population: mean 1/3)"""
    from scipy.integrate import simpson
    lnT = single.merger_columns(S_LADDER)
    with np.errstate(divide='ignore'):
        x = lnT + (np.log(2.0 * single.V) - single.V ** 2)[:, None]
    mx = x.max(axis=0)
    mean = np.exp(mx) * simpson(np.exp(x - mx), x=single.V, axis=0)
    assert np.abs(mean - 1).max() < 1e-6, np.abs(mean - 1).max()
    u = np.array([2.29695571, 4.24624625, 9.11659647])
    s = np.array([0.5, 2.0])
    cols = single.merger_columns(s)
    lnW = np.full((3, 2), np.log(1.0 / 6.0))
    rng = np.random.default_rng(20261116)
    Ms, err = [], 0.0
    for i in range(2000):
        lq, w = white_trajectories(rng, u, 4096)
        lnM = single.window_merger(lq, np.log(u), cols, 64, lnW)
        Ms.append(np.exp(lnM))
        if i < 5:
            wq = np.exp(2.0 * lq.astype(float) - np.log(u)[:, None])                 # what the float32 data hold
            phi = np.stack([single._ln_phi(np.sqrt(wq.ravel()), np.full(wq.size, sk)).reshape(wq.shape) for sk in s])
            direct = logsumexp(phi.reshape(2, 3, 64, 64) + np.log(1 / 6.0), axis=(0, 1, 3)) - np.log(64)
            err = max(err, np.abs(direct - lnM).max())
    M = np.concatenate(Ms)
    se = M.reshape(2000, 64).mean(axis=1).std() / np.sqrt(2000)
    print('\ncolumns: largest |mean - 1| %.1e; M_w over %d windows: mean %.6f +- %.6f (%+.1f se); against the direct '
          'sum of Phi %.1e' % (np.abs(mean - 1).max(), M.size, M.mean(), se, (M.mean() - 1) / se, err))
    assert abs(M.mean() - 1.0) < 3 * se and se < 1e-3 and err < 3e-6


def test_c2_merger_power_with_signal_gain(tpw_data):
    """the merger evidence of a detector whose data carry a signal with kappa_I times the amplitude of the true strain
    (the signal gain of the conditioning, bank.gain) is that of a signal of power kappa_I^2 u_I y^2 in the data's
    statistic w_I = |z_I|^2 / u_I.  Planted truth: signals of strain amplitude y0 arriving with amplitude kappa_I (|R|
    from the response distribution, uniform phase, a time uniform in window 0, white noise) give E[1/M_0] = 1, the
    reciprocal identity E_signal[p_0 / p_1] = 1 of an exact likelihood ratio, within 3 standard errors, with the weights
    of pe._merger_setup on a bank with gains (kappa_H 0.8, kappa_L 1.25); with the weights of the same bank without gains
    (the power u_I y0^2) E[1/M_0] is off by more than 5.  The population is one cell whose y0 lands on a node of the
    ladder for either power (W = 1 there), so that both M_0 are exact likelihood ratios of their models.  With real
    segments, window_evidences on a bank with gains gives the window evidences of gains 1 bit for bit and other merger
    evidences.  Fires on: _merger_setup reading bank.u_I (the data's power) in place of kappa_I^2 u_I"""
    from types import SimpleNamespace
    from exmarg.search.segments import D0
    gains = dict(H=0.8, L=1.25)
    u, M, m, z0 = np.array([4.0, 10.0, 25.0]), np.array([30.0, 60.0, 90.0]), 1, 3.0
    y0 = float(D0 / pe._dl_of_z(z0)[0])
    powers = sorted({u[m] * y0 ** 2} | {gains[I] ** 2 * u[m] * y0 ** 2 for I in 'HL'})
    s = np.array([powers[0] / 2] + powers + [2 * powers[-1]])
    m1 = M[m] / (2 * (1 + z0))
    pop = popavg.merger_population(np.array([m1 / 1.1, m1, m1 * 1.1]), np.array([z0 - 0.1, z0, z0 + 0.1]),
                                   np.eye(3)[1][:, None] * np.eye(3)[1][None, :], s, 0.0)
    banks = dict(gain=SimpleNamespace(M=M, u_H=u, u_L=u, gain=gains), plain=SimpleNamespace(M=M, u_H=u, u_L=u))
    q, wq = single.response_distribution()
    rng = np.random.default_rng(20261116)
    n, nwin = 40000, 8
    lines, ok = [], True
    for I in 'HL':
        R = q[rng.choice(q.size, size=n, p=wq / wq.sum())]
        z = (rng.standard_normal((n, nwin)) + 1j * rng.standard_normal((n, nwin))) / np.sqrt(2.0)
        z[np.arange(n), rng.integers(nwin, size=n)] += gains[I] * y0 * np.sqrt(u[m]) * R * np.exp(2j * np.pi * rng.random(n))
        lq = np.log(np.abs(z * np.sqrt(u[m]))).astype(np.float32)                 # bank.trajectories' layout
        for name, bank in banks.items():
            mg = pe._merger_setup(pop, bank, np)
            Wfull = popavg.merger_weights(M, pe._model_power(bank, I), pop)
            k = int(np.argmax(Wfull[m]))
            row = list(mg['use_' + I]).index(m)
            lnE = single.ln_evidence(lq, np.full(n, np.log(u[m])), mg['lnT'])
            lnM = logsumexp(lnE + mg['ln_W_' + I][row][None, :], axis=1)
            x = np.exp(-lnM)
            mu, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
            lines.append('%s, bank %-5s: W = %.12f at the node of power %.4f (the signal: kappa^2 u y0^2 = %.4f); '
                         'E[1/M_0] over %d planted signals %.5f +- %.5f (%+.1f se)' % (
                             I, name, Wfull[m, k], s[k], gains[I] ** 2 * u[m] * y0 ** 2, n, mu, se, (mu - 1) / se))
            assert abs(Wfull.sum() - 1.0) < 1e-12 and abs(Wfull[m, k] - 1.0) < 1e-12
            assert abs(s[k] / (u[m] * y0 ** 2) - (gains[I] ** 2 if name == 'gain' else 1.0)) < 1e-12
            ok &= abs(mu - 1) < 3 * se if name == 'gain' else abs(mu - 1) > 5 * se
    c = tpw_data
    plain = pe.window_evidences(c['segs'], c['bank'], n_window=64, population=tiny_population(c['bank'], S_LADDER))
    gb = mass_bank(tpw.MASSES, c['segs'], gain=gains)
    with_gain = pe.window_evidences(c['segs'], gb, n_window=64, population=tiny_population(gb, S_LADDER))
    for I in 'HL':
        assert np.array_equal(plain['ln_window_' + I], with_gain['ln_window_' + I])
        assert np.array_equal(plain['wmax_' + I], with_gain['wmax_' + I])
        dev = np.abs(plain['ln_merger_' + I] - with_gain['ln_merger_' + I]).max()
        lines.append('%s, real segments: window evidences of gains 1 bit for bit; ln M_w differs by up to %.3g' % (I, dev))
        assert dev > 1e-3
    print('\n' + '\n'.join(lines))
    assert ok, lines


# ================================================================================ 3. the toy with known truth


NB = tw.NBAR
LOUD = tw.EDGES[:-1] >= 4.0                     # the bins where the toy's mergers (SNR up to 5.7) look like glitches


def toy_lnM(ev, I):
    """ln M_w of the toy: the mean over the window of its exact single-detector merger ratio S^I"""
    return ev['S' + I + 's'] - np.log(tw.NWIN)


def objective(ln_E, ln_M, f, q, pi, nbar, zone=tw.ZONE):
    """sum_i ln c_i(theta, N), c_i = sum_s f_s C^s [1 + N b^s], from single.window_posterior of each state alone
    (ln C^s, ln kappa^s_w), not from mixture.py; per segment (n,)"""
    terms = []
    for s in range(len(f)):
        p = single.window_posterior(ln_E, [1.0], [q[s]], [pi[s]], zone)
        b = np.mean(np.exp(p['ln_kbar']) * np.expm1(ln_M), axis=1)
        terms.append(np.log(f[s]) + p['ln_c'] + np.log1p(nbar * b))
    return logsumexp(np.stack(terms), axis=0)


def rate(r, bins):
    """the glitches per window in the bins, sum_s f_s q_s sum_k pi_sk = sum_s f_s Om[s, 1 + k] (linear in x), and its
    error from the covariance of the fit (delta method, exact for a linear function)"""
    P = tw.NK + 1
    Om = np.concatenate([(1 - r['q'])[:, None], r['q'][:, None] * r['a']], 1)
    J = np.zeros(r['cov'].shape[0])
    for s in range(2):
        J[s] = Om[s, 1:][bins].sum()
        J[2 + s * P + 1 + np.flatnonzero(bins)] = r['f'][s]
    return float(r['f'] @ Om[:, 1:][:, bins].sum(axis=1)), float(np.sqrt(J @ r['cov'] @ J))


def true_rate(bins):
    return float(tw.TRUTH['f'] @ (tw.TRUTH['q'] * tw.TRUTH['pi'][:, bins].sum(axis=1)))


@lru_cache(maxsize=None)
def toy_fits(nbar):
    """fit_windows of each detector on the toy's data with mergers ('model'), at nbar (0: the present fit)"""
    ev = tw.recovery_data()['model']
    if nbar == 0:
        return {I: tw.stage_one('model')[I] for I in 'HL'}
    return {I: mixture.fit_windows(ev['E' + I], ln_M=toy_lnM(ev, I), nbar=nbar) for I in 'HL'}


def stage2(fits):
    """N of the test's own stage 2 (test_windows.ln_B, nbar_fit) with the glitch parameters of fits"""
    ev = tw.recovery_data()['model']
    return tw.nbar_fit(tw.ln_B(ev, 's', tw.post(ev['EH'], tw.params(fits['H'])), tw.post(ev['EL'], tw.params(fits['L']))))


def test_c3_objective_and_maximum():
    """the objective that fit_windows maximizes with mergers is that of the specification: its ln L at the fitted
    parameters equals the sum of objective() (from single.window_posterior of each state, not mixture.py) to 1e-11
    relative, on the toy with mergers at the true N; and its maximum is that of an independent optimizer: on the first
    4000 segments with a basis of three laws (rho^-1.5, -2.5, -4 from 2, so that the planted laws are inside the
    simplex of the basis weights and the maximum is interior), scipy's L-BFGS-B on objective() in softmax
    coordinates from another start reaches a ln L at most 1e-6 above fit_windows', with f, q and the basis weights
    within 3e-3 of it.  Its gradient and Hessian agree with central differences of ln L and of the gradient to 1e-8
    and 1e-6 relative.  Fires on: the zone of kappa taken as the window alone in mixture.py (zone 0): ln L off by 0.15
    nats (3e-7 relative)"""
    ev = tw.recovery_data()['model']
    lines = []
    for I in 'HL':
        r = toy_fits(NB)[I]
        lo = math.fsum(objective(ev['E' + I], toy_lnM(ev, I), r['f'], r['q'], r['pi'], NB))
        lines.append('%s: ln L of the fit %.6f, objective() at its parameters %.6f' % (I, r['lnl'], lo))
        assert abs(r['lnl'] / lo - 1) < 1e-11, lines[-1]
    n = 4000
    E, lnM = ev['EH'][:n], toy_lnM(ev, 'H')[:n]
    basis = np.array([tw.law(tw.EDGES, g, 2.0) for g in (1.5, 2.5, 4.0)])
    r = mixture.fit_windows(E, basis=basis, ln_M=lnM, nbar=NB)

    def unpack(v):
        f = softmax(np.r_[0.0, v[0]])
        q = 1.0 / (1.0 + np.exp(-v[1:3]))
        a = np.stack([softmax(np.r_[0.0, v[3:5]]), softmax(np.r_[0.0, v[5:7]])])
        return f, q, a
    fun = lambda v: -math.fsum(objective(E, lnM, *unpack(v)[:2], unpack(v)[2] @ basis, NB)) / n
    res = minimize(fun, np.array([-1.0, -5.0, -1.0, 0.5, 0.0, 0.0, 0.5]), method='L-BFGS-B',
                   options=dict(ftol=1e-15, gtol=1e-10, maxiter=5000))
    f, q, a = unpack(res.x)
    o = np.argsort(q)
    f, q, a = f[o], q[o], a[o]
    gap = -res.fun * n - r['lnl']
    dpar = max(np.abs(f - r['f']).max(), np.abs(q - r['q']).max(), np.abs(a - r['a']).max())
    lines.append('basis of 3 laws, %d segments: fit_windows ln L %.6f (%d updates, converged %s); L-BFGS-B %.6f (gap '
                 '%+.2e), largest parameter difference %.1e' % (n, r['lnl'], r['n_iter'], r['converged'], -res.fun * n,
                                                                gap, dpar))
    assert r['converged'] and gap < 1e-6 and dpar < 3e-3, lines[-1]
    base = mixture._Windows(E, basis, False, 1024)
    m = mixture._WindowsMerger(base, lnM, NB, tw.ZONE)
    rng = np.random.default_rng(3)
    x = np.concatenate([[0.8, 0.2], np.r_[0.99, 0.01 * rng.dirichlet(np.ones(3))], np.r_[0.8, 0.2 * rng.dirichlet(np.ones(3))]])
    ev0 = m(x, 2, hess=True)
    g = np.concatenate([ev0['Q'], ev0['G'].T.ravel()])
    H = m.hessian(ev0)
    I_ = np.eye(x.size)
    gl = lambda y: np.concatenate([m(y, 2)['Q'], m(y, 2)['G'].T.ravel()])
    gfd = np.array([(m(x + 1e-6 * I_[j], 2)['lnl'] - m(x - 1e-6 * I_[j], 2)['lnl']) / 2e-6 for j in range(x.size)])
    Hfd = np.array([(gl(x + 1e-6 * I_[j]) - gl(x - 1e-6 * I_[j])) / 2e-6 for j in range(x.size)])
    eg, eh = np.abs(g - gfd).max() / np.abs(g).max(), np.abs(H - Hfd).max() / np.abs(H).max()
    lines.append('gradient against central differences %.1e, Hessian %.1e (relative to the largest entry)' % (eg, eh))
    print('\n' + '\n'.join(lines))
    assert eg < 1e-8 and eh < 1e-6


def test_c3_present_fit_biased():
    """(a) the present fit (no mergers) on the toy's data with mergers rejects the planted glitch parameters: 2 (ln L_max
    - ln L_truth) of its own objective (N = 0) is above chi2_27(0.999) in each detector, where on the same noise and
    glitches without mergers it is below (the parameters are biased by more than their errors; where the bias sits,
    bin by bin, is printed: the glitches per window of each bin, summed over the states, of each fit minus the
    merger-free fit; its size over several seeds is measured by scripts outside the tests).  (b) the fit at the true
    N = NBAR with the toy's merger evidence: 2 (ln L_max - ln L_truth) of the objective with mergers below
    chi2_27(0.999), f and q within 3.5 standard errors of the planted ones, converged, ln L never decreasing.  Fires
    on (b): the merger term with M_w - 1 replaced by M_w (b^s = mean kappa M)"""
    ev = tw.recovery_data()
    lines, ok = [], True
    bound = chi2.ppf(0.999, 1 + 2 * tw.NK)
    truth_lnl = lambda data, I, nbar: math.fsum(objective(data['E' + I], toy_lnM(data, I), tw.TRUTH['f'], tw.TRUTH['q'],
                                                          tw.TRUTH['pi'], nbar))
    for I in 'HL':
        free, biased, fixed = tw.stage_one('free')[I], toy_fits(0)[I], toy_fits(NB)[I]
        lr = dict(free=2 * (free['lnl'] - truth_lnl(ev['free'], I, 0.0)),
                  biased=2 * (biased['lnl'] - truth_lnl(ev['model'], I, 0.0)),
                  fixed=2 * (fixed['lnl'] - truth_lnl(ev['model'], I, NB)))
        lines.append('%s 2 (ln L_max - ln L_truth), bound %.1f: no mergers in the data, fit at N = 0: %.1f; mergers, fit '
                     'at N = 0 (present): %.1f; mergers, fit at the true N: %.1f' % (I, bound, lr['free'], lr['biased'],
                                                                                     lr['fixed']))
        cells = []
        for k in range(tw.NK):
            bk = np.arange(tw.NK) == k
            (x0, e0), (x1, e1), (x2, e2) = rate(free, bk), rate(biased, bk), rate(fixed, bk)
            cells.append('%5g-%-5g %+.2e %+.2e  (%.1e)' % (tw.EDGES[k], tw.EDGES[k + 1], x1 - x0, x2 - x0, e1))
        lines.append('%s glitches per window in each bin minus the merger-free fit: present fit, fit at the true N (error '
                     'of one fit):\n    %s' % (I, '\n    '.join(cells)))
        r = fixed
        pf, pq = (r['f'] - tw.TRUTH['f']) / r['err_f'], (r['q'] - tw.TRUTH['q']) / r['err_q']
        lines.append('%s fit at N = %g: pulls f %s q %s; quiet q: merger-free %.5f, present %.5f, true N %.5f (truth %g); '
                     '%d updates' % (I, NB, np.round(pf, 2), np.round(pq, 2), free['q'][0], biased['q'][0], r['q'][0],
                                     tw.TRUTH['q'][0], r['n_iter']))
        ok &= (lr['free'] < bound < lr['biased'] and 0.0 <= lr['fixed'] < bound and r['converged']
               and np.all(np.abs(pf) < 3.5) and np.all(np.abs(pq) < 3.5) and np.all(np.diff(r['trace']) >= 0))
    print('\n' + '\n'.join(lines))
    assert ok, lines


def delta_error(fits, ev):
    """the standard error of N from the fitted glitch parameters against N from the true ones: the derivative of
    stage 2's N along each parameter of x (central differences) against the covariance of each detector's fit"""
    var = 0.0
    for I in 'HL':
        r = fits[I]
        K, P = 2, tw.NK + 1
        x = np.concatenate([r['f'], np.concatenate([(1 - r['q'])[:, None], r['q'][:, None] * r['a']], 1).ravel()])

        def nb(y):
            f, Om = y[:K], y[K:].reshape(K, P)
            p = dict(f=f / f.sum(), q=1 - Om[:, 0] / Om.sum(axis=1), pi=Om[:, 1:] / Om[:, 1:].sum(axis=1, keepdims=True))
            other = 'L' if I == 'H' else 'H'
            pp = {I: tw.post(ev['E' + I], p), other: tw.post(ev['E' + other], tw.params(fits[other]))}
            return tw.nbar_fit(tw.ln_B(ev, 's', pp['H'], pp['L']))[0]
        on = np.flatnonzero(np.diag(r['cov']) > 0)
        J = np.zeros(x.size)
        for j in on:
            h = 1e-4 * max(x[j], 1e-3)
            e = np.zeros(x.size)
            e[j] = h
            J[j] = (nb(x + e) - nb(np.maximum(x - e, 0.0))) / (h + min(h, x[j]))
        var += J @ r['cov'] @ J
    return float(np.sqrt(var))


def test_c3_iteration():
    """(c) the iteration of Sec. 3 on the toy's data with mergers: N1 = stage 2 with the fit at N = 0, N2 = stage 2
    with the fit at N1, N3 with the fit at N2; N_true = stage 2 with the planted glitch parameters.  sigma_d, the
    standard error of N from fitted against true glitch parameters (delta method: the derivative of stage 2's N along
    each parameter, by central differences, against the covariance of the fits at N2): N1 - N_true is beyond 3
    sigma_d (the defect: stage 1 takes loud mergers for glitches), N2 - N_true within 3 sigma_d, |N3 - N2| below
    sigma_d (its Monte Carlo error) and below |N2 - N1| / 10.  Fires on: stage 1 always fitted at N = 0 (no
    iteration), N2 = N1"""
    ev = tw.recovery_data()['model']
    nt, et = tw.nbar_fit(tw.ln_B(ev, 's', tw.post(ev['EH'], tw.TRUTH), tw.post(ev['EL'], tw.TRUTH)))
    n1, e1 = stage2(toy_fits(0))
    f1 = {I: mixture.fit_windows(ev['E' + I], ln_M=toy_lnM(ev, I), nbar=n1) for I in 'HL'}
    n2, e2 = stage2(f1)
    f2 = {I: mixture.fit_windows(ev['E' + I], ln_M=toy_lnM(ev, I), nbar=n2) for I in 'HL'}
    n3, e3 = stage2(f2)
    sd = delta_error(f2, ev)
    msg = ('N_true %.6f +- %.6f (truth %g); N1 %.6f (%+.2f%% of N_true), N2 %.6f (%+.3f%%), N3 %.6f (%+.3f%%); sigma_d '
           '%.6f: N1 - N_true %+.1f sigma_d, N2 - N_true %+.2f sigma_d, N3 - N2 %+.3f sigma_d'
           % (nt, et, NB, n1, 100 * (n1 / nt - 1), n2, 100 * (n2 / nt - 1), n3, 100 * (n3 / nt - 1), sd, (n1 - nt) / sd,
              (n2 - nt) / sd, (n3 - n2) / sd))
    print('\n' + msg)
    assert abs(n1 - nt) > 3 * sd and abs(n2 - nt) < 3 * sd and abs(n3 - n2) < sd and abs(n3 - n2) < abs(n2 - n1) / 10, msg


# ================================================================================ 4. marginal identity


@pytest.fixture(scope='module')
def marginal():
    """stage 2 on the incoherent store (nodes s = 0.5, 2, the three templates of test_pe_windows.py, every arrival
    time, the window model E_MODEL in both detectors) for three fixed L segments of white matched-filter outputs (no
    glitch; one glitch of SNR 9; 14 glitches of SNR 5 to 9, the noisy state) each against N_H draws of H from its noise
    model (white noise with the glitches of E_MODEL, test_pe_windows.white_segment); B per pair is the mean over the
    (node, template) pairs of the stage-2 ratio (the population of the store, W = 1/6 each).  Per L segment: the mean
    of B over the draws and its standard error, and the single-detector ratio of L from the stage-1 objective,
    exp(ln c_L(theta, 1) - ln c_L(theta, 0)) (mixture._WindowsMerger and _Windows at the parameters of E_MODEL), with
    M_w from single.window_merger on the same population"""
    raw, _, _ = simulate(2, 77, lambda r: (60.0, 1.0, 0.0), lambda r: 1e-8, 4.0, 20.0, 128.0)
    bank = mass_bank(tpw.MASSES, raw)
    store = tpw.incoherent_store(tuple(tpw.E_S))
    nh, nt, nw = 600, bank.n_time, tpw.N_WINDOW
    segs = dict(raw, d_H=np.zeros((nh, raw['nu'].size), complex))
    segs['d_L'] = segs['d_H']
    seg = read_segments(segs)
    geometry = pe._node_geometry(bank, store, 0.1)
    assert geometry[0].all()
    groups = [[k] for k in range(store.s.size)]
    sd = pe._single_setup(bank, seg, store, geometry)
    gd = pe._glitch_setup(dict(model='windows', H=tpw.E_MODEL, L=tpw.E_MODEL), bank, seg, store)
    rng = np.random.default_rng(11)

    def white(glitches=()):
        z = (rng.standard_normal((3, nt)) + 1j * rng.standard_normal((3, nt))) / np.sqrt(2.0)
        for m, t, rho in glitches:
            z[m, t] += np.sqrt(0.5 * rho * rho) * np.exp(2j * np.pi * rng.random())
        v = z * np.sqrt(bank.u)[:, None]
        return dict(lqL=np.log(np.abs(v)).astype(np.float32), phL=np.angle(v).astype(np.float32))
    Ls = {'quiet': white(), 'one glitch, SNR 9': white([(1, 1000, 9.0)]),
          'noisy, 14 glitches': white([(int(m), int(t), float(r)) for m, t, r in zip(
              rng.integers(3, size=14), rng.choice(nt, 14, replace=False), rng.uniform(5.0, 9.0, 14))])}
    seeds = np.random.SeedSequence(12).spawn(nh)
    Hs = [tpw.white_segment(seeds[i], bank.u, nt, tpw.E_MODEL)[0] for i in range(nh)]
    P = tpw.E_MODEL
    x = np.concatenate([P['f'], np.concatenate([(1 - P['q'])[:, None], P['q'][:, None] * P['pi_k']], 1).ravel()])
    cols = single.merger_columns(np.asarray(tpw.E_S, float))
    lnW = np.full((3, 2), np.log(1.0 / 6.0))
    out = {}
    real = pe._filter
    try:
        for name, L in Ls.items():
            trajs = [dict(lqH=h['lqH'], phH=h['phH'], **L) for h in Hs]
            pe._filter = lambda b, s, e, trajs=trajs: trajs[e]
            _, one = pe._node_values(bank, seg, range(nh), store, geometry, groups, 256, False, 0, 0, 0, sd, gd)
            B = np.exp(one['lamN_gHL'][:, :-1]).mean(axis=(1, 2))
            lnE = single.window_evidence(L['lqL'], np.log(bank.u_L), single.glitch_columns(), nw)
            lnM = single.window_merger(L['lqL'], np.log(bank.u_L), cols, nw, lnW)
            base = mixture._Windows(lnE[None], np.eye(lnE.shape[1]), True, 16)
            ratio = np.exp(mixture._WindowsMerger(base, lnM[None], 1.0, tpw.ZONE)(x, 2)['lnl'] - base(x, 2)['lnl'])
            post = single.window_posterior(lnE, P['f'], P['q'], P['pi_k'], tpw.ZONE)
            lamL = float(np.exp(one['lamN_L'][0, :-1]).mean())
            out[name] = dict(B=B, ratio=ratio, lnM=lnM, post=post, lamL=lamL)
    finally:
        pe._filter = real
    return out


def test_c4_marginal_identity(marginal):
    """for each L segment the mean of stage 2's B over the H draws equals the single-detector ratio of L of the
    stage-1 objective within 3 standard errors (600 draws, standard errors 7e-5); the mean of M_w over the windows
    equals stage 2's own L-only ratio Lambda_S^L of the segment (the same population) to 1e-6 relative; the L segments
    differ in what is tested: the loud glitch puts kappa near 0 where M_w is e^5, the noisy one the state's posterior
    at the noisy state.  Fires on: kappa taken as 1 in mixture._WindowsMerger (the merger seen through every glitch):
    the ratio of the loud glitch segment 4.08 against a mean B of 0.9998"""
    lines, ok = [], True
    for name, m in marginal.items():
        B = m['B']
        mu, se = B.mean(), B.std(ddof=1) / np.sqrt(B.size)
        lam = float(np.exp(m['lnM']).mean())
        lines.append('%-20s mean B over %d H draws %.6f +- %.6f; single-detector ratio %.6f (%+.2f se); mean M_w %.6f, '
                     'stage 2 Lambda_S^L %.6f; smallest ln kbar %.1f, largest ln M_w %.2f, state posterior %s'
                     % (name, B.size, mu, se, m['ratio'], (mu - m['ratio']) / se, lam, m['lamL'],
                        m['post']['ln_kbar'].min(), m['lnM'].max(), np.round(m['post']['r'], 4)))
        ok &= abs(mu - m['ratio']) < 3 * se and abs(lam / m['lamL'] - 1) < 1e-6
    print('\n' + '\n'.join(lines))
    assert ok, lines
    assert marginal['noisy, 14 glitches']['post']['r'][1] > 0.99 and marginal['one glitch, SNR 9']['lnM'].max() > 4
