"""stage 1 of the window model (scripts/validate/windows_fit.py) on window evidences simulated here, in the files of
plant.py --stage windows.

The toy is that of tests/test_windows.py on the amplitude bins that plant.py stores (those of single.RHO_EDGES that
overlap [3, 12.2]): one template, NT = 256 arrival times of white noise z_I(t) (E|z|^2 = 1) cut into NW = 32 windows of
8; each detector-segment draws a state s with probability f_s, each window then a glitch with probability q_s at a
uniform time in the window, of SNR rho uniform in a bin drawn from the planted law (dN/drho ~ rho^-3 on [4, 9.2017],
the law of the campaign, between bin edges), adding sqrt(rho^2 / 2) e^{i phi}.  E_w[k] = (1/8) sum_{t in w} l_k(t) with
l_k the glitch ratio of bin k (test_mixture.Toy, exact), wmax = max_t |z|^2.  Nothing here uses mixture.py or single.py
to build the data.
"""
import glob
import importlib.util
import json
import os
import shutil

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import chi2

from exmarg.search import mixture, single
from test_mixture import Toy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts', 'validate', name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plant, windows_fit = _script('plant'), _script('windows_fit')

NT, NW = 256, 32
NWIN = NT // NW
STORED = plant.window_bins(*plant.WINDOW_RHO)
B0 = int(np.flatnonzero(STORED)[0])
EDGES = single.RHO_EDGES[B0:B0 + STORED.sum() + 1]
NK = EDGES.size - 1
LAW = (3.0, 4.0, float(EDGES[11]))                        # gamma, rho_min, rho_max (9.2017, a bin edge) of the law
STATES = dict(H=((0.8, 0.004), (0.2, 0.06)), L=((0.9, 0.002), (0.1, 0.08)))
N, SEED = 16000, 20261114


def law(gamma, lo, hi, edges=EDGES):
    """bin probabilities of dN/drho ~ rho^-gamma on [lo, hi] over the bins of edges"""
    m = lambda a, b: 0.0 if b <= a else (a ** (1 - gamma) - b ** (1 - gamma)) / (gamma - 1)
    p = np.array([m(max(a, lo), min(b, hi)) for a, b in zip(edges[:-1], edges[1:])])
    return p / p.sum()


def config(*extra, states=STATES):
    """a windows-stage campaign of plant.py with the window model planted (store and out unused)"""
    st = {I: ','.join('%g:%g' % p for p in states[I]) for I in 'HL'}
    return plant.campaign(plant.parser().parse_args(
        ['s', 'cpu', '--n', '1', '--out', 'x', '--stage', 'windows', '--n-window', str(NW), '--nbar', '0',
         '--glitch-states-H', st['H'], '--glitch-states-L', st['L'], '--gamma', str(LAW[0]), '--rho-min', str(LAW[1]),
         '--rho-max', str(LAW[2])] + list(extra)))


def simulate(n, seed, states=STATES, pi=None):
    """the window evidences (n, NW, NK) and truth of each detector: {I: dict(E, wmax, state, n, rho)}"""
    toy = Toy(edges=EDGES)
    pi = law(*LAW) if pi is None else pi
    cum = np.cumsum(pi) / pi.sum()
    rng = np.random.default_rng(seed)
    out = {}
    for I in 'HL':
        f, q = np.array(states[I], float).T
        parts = []
        for a in range(0, n, 2000):
            m = min(2000, n - a)
            z = (rng.standard_normal((m, NT)) + 1j * rng.standard_normal((m, NT))) / np.sqrt(2.0)
            s = np.minimum(np.searchsorted(np.cumsum(f), rng.random(m), side='right'), f.size - 1)
            has = rng.random((m, NW)) < q[s][:, None]
            i, w = np.nonzero(has)
            k = np.minimum(np.searchsorted(cum, rng.random(i.size), side='right'), NK - 1)
            rho = EDGES[k] + rng.random(i.size) * np.diff(EDGES)[k]
            z[i, w * NWIN + rng.integers(NWIN, size=i.size)] += np.sqrt(0.5 * rho ** 2) * np.exp(
                2j * np.pi * rng.random(i.size))
            top = np.full(m, np.nan)
            np.fmax.at(top, i, rho)
            E = logsumexp(toy.ln_lk(np.abs(z)).reshape(m, NW, NWIN, NK), axis=2) - np.log(NWIN)
            parts.append(dict(E=E, wmax=(np.abs(z) ** 2).max(axis=1), state=s, n=has.sum(axis=1), rho=top))
        out[I] = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return out


def write(folder, cfg, sim, splits):
    """the toy in win files of plant.py, one per (first, end) of splits"""
    os.makedirs(folder, exist_ok=True)
    n = sim['H']['E'].shape[0]
    per = dict(k=np.arange(n), merger=np.zeros(n, bool), merger_H=np.zeros(n, bool), merger_L=np.zeros(n, bool),
               coherent=np.zeros(n, bool), m1=np.full(n, np.nan), z=np.full(n, np.nan), y=np.zeros(n), snr=np.zeros(n))
    for I in 'HL':
        g = sim[I]
        per.update({'ln_window_' + I: g['E'].astype(np.float32), 'wmax_' + I: g['wmax'], 'glitch_' + I: g['n'] > 0,
                    'kind_' + I: np.where(g['n'] > 0, 'template', ''), 'rho_' + I: g['rho'],
                    't0_' + I: np.full(n, np.nan), 'glitch_state_' + I: g['state'], 'n_glitches_' + I: g['n']})
    for a, b in splits:
        plant.write_batch(os.path.join(folder, 'win_%010d_%010d.npz' % (a, b)), cfg, dict(first=a, end=b),
                          dict({key: v[a:b] for key, v in per.items()}, **plant.window_shared(STORED)))


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    """the campaign written in three files, and windows_fit.fit on it (bins basis, default settings)"""
    folder = str(tmp_path_factory.mktemp('windows') / 'states')
    cfg, sim = config(), simulate(N, SEED)
    write(folder, cfg, sim, ((0, 5000), (5000, 11000), (11000, N)))
    return folder, cfg, sim, windows_fit.fit(folder)


def full(lnE_stored, cols=None):
    """ln E over every bin of RHO_EDGES: the stored bins in place, -inf elsewhere (no law has weight there)"""
    out = np.full(lnE_stored.shape[:-1] + (single.RHO_EDGES.size - 1,), -np.inf)
    out[..., B0:B0 + NK] = lnE_stored
    return out


def test_windows_fit_recovers_states(toy, capsys):
    """windows_fit on N segments of the planted window model (two states per detector, the law of the campaign): every
    fit converges; for each detector the fraction and the glitch rate n_window q of each state within 3.5 standard
    errors of the planted ones (all planted glitches are inside the fitted bins); 2 states gain more than 300 nats over
    1, 3 gain fewer than 15 over 2 (12 more parameters); the segments set aside are those with wmax above 72; the report
    prints each of these numbers; main writes glitch_model.json and exits 0"""
    folder, cfg, sim, r = toy
    assert r['ok']
    keep = (sim['H']['wmax'] <= 72) & (sim['L']['wmax'] <= 72)
    assert np.array_equal(r['keep'], keep) and (~keep).sum() < 20
    text = '\n'.join(r['lines'])
    print('\n' + text)
    lines = []
    for I in 'HL':
        f, q = np.array(STATES[I], float).T
        R = r['fits'][I]
        fit = R[2]
        pf, pq = (fit['f'] - f) / fit['err_f'], (fit['q'] - q) / fit['err_q']
        lines.append('%s: f %s +- %s (pulls %s), n_w q %s +- %s (pulls %s); gains %.1f, %.1f' % (
            I, np.round(fit['f'], 4), np.round(fit['err_f'], 4), np.round(pf, 2), np.round(NW * fit['q'], 4),
            np.round(NW * fit['err_q'], 4), np.round(pq, 2), R[2]['lnl'] - R[1]['lnl'], R[3]['lnl'] - R[2]['lnl']))
        assert np.all(np.abs(pf) < 3.5) and np.all(np.abs(pq) < 3.5), lines[-1]
        assert R[2]['lnl'] - R[1]['lnl'] > 300 and 0 <= R[3]['lnl'] - R[2]['lnl'] < 15, lines[-1]
        for s in range(2):
            assert '%15.4g +- %-6.2g %24.4g +- %-9.2g' % (fit['f'][s], fit['err_f'][s], NW * fit['q'][s],
                                                         NW * fit['err_q'][s]) in text, (I, s)
        assert '2 over 1 states %+.3f for %+d parameters' % (R[2]['lnl'] - R[1]['lnl'], 12) in text
    print('\n'.join(lines))
    assert 'set aside %d segments' % (~keep).sum() in text and 'fitted bins: 11, rho 4 to 12.17' in text
    capsys.readouterr()
    assert windows_fit.main([folder]) == 0
    assert 'written %s' % os.path.join(folder, 'glitch_model.json') in capsys.readouterr().out


def test_glitch_model_file(toy):
    """glitch_model.json: SHA-256 of its content (plant.read_model accepts it, refuses q changed by 1e-12 relative and a
    file that is not JSON); n_window,
    zone, the campaign key of the files, the settings and the counts of segments; pi_k with a row per state over every
    bin of RHO_EDGES, zero outside the fitted bins [4, 12.17], each row summing to 1; and single.window_posterior with
    f, q, pi_k of the file on the evidences of every bin (stored bins in place) gives sum ln c equal to the ln L of the
    2-state fit of each detector to 1e-10 (relative)"""
    folder, cfg, sim, r = toy
    path = os.path.join(folder, 'glitch_model.json')
    if not os.path.exists(path):
        windows_fit.write_model(path, r['model'])
    model, sha = plant.read_model(path)
    assert model['n_window'] == NW and model['zone'] == 1 and model['pi_m'] is None
    assert model['campaign'] == plant.campaign_key(cfg) and 'stage' not in model['campaign']
    assert model['settings']['fitted_bins'] == [8, 19] and model['settings']['basis'] == 'bins'
    seg = model['segments']
    assert seg['total'] == N and seg['used'] + seg['set_aside'] == N and seg['used'] == r['keep'].sum()
    assert [f['n_state'] for f in model['fits']['H']] == [1, 2, 3]
    keep = r['keep']
    for I in 'HL':
        pi = np.array(model[I]['pi_k'])
        assert pi.shape == (2, single.RHO_EDGES.size - 1) and np.all(pi[:, :8] == 0) and np.all(pi[:, 19:] == 0)
        assert np.allclose(pi.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        lnE = full(sim[I]['E'].astype(np.float32).astype(float)[keep])
        post = single.window_posterior(lnE, model[I]['f'], model[I]['q'], pi, zone=model['zone'])
        lnl = float(np.sum(post['ln_c']))
        assert abs(lnl / model[I]['lnl'] - 1) < 1e-10, (I, lnl, model[I]['lnl'])
    changed = json.loads(open(path).read())
    changed['H']['q'][0] *= 1.0 + 1e-12
    with open(path, 'w') as fh:
        json.dump(changed, fh)
    with pytest.raises(SystemExit, match='modified'):
        plant.read_model(path)
    with open(path, 'w') as fh:
        fh.write('{"H": ')
    with pytest.raises(SystemExit, match='not a JSON'):
        plant.read_model(path)
    windows_fit.write_model(path, r['model'])


def test_windows_fit_power_basis(tmp_path):
    """--basis power with the fitted range [4, 9.21] (bins up to 9.2017), on which the planted law is the row g = 3 of
    the basis, so that the planted model is in the fitted family: every fit converges, f and n_window q of each state
    within 3.5 standard errors of the planted ones, 0 <= 2 (ln L_max - ln L_planted) < chi2_13(0.999) (ln L_planted by
    single.window_posterior), the law of each state its basis weights times the rows; the free bins on the same range
    reach a ln L at least as high (they contain the power laws).  The single weights are printed, not tested: a flat
    and a steep law mimic g = 3, and the quiet state has few glitches"""
    folder = str(tmp_path / 'power')
    sim = simulate(8000, SEED + 1)
    write(folder, config(), sim, ((0, 8000),))
    r = windows_fit.fit(folder, basis='power', rho_max=9.21)
    rb = windows_fit.fit(folder, basis='bins', rho_max=9.21)
    assert r['ok'] and rb['ok']
    fe = EDGES[(EDGES >= 4.0) & (EDGES <= 9.21)]
    B = windows_fit.power_basis(fe)
    assert B.shape == (6, fe.size - 1) and np.allclose(B[1], law(*LAW, fe)) and fe.size == 10
    bound, msg = chi2.ppf(0.999, 1 + 2 * 6), []
    for I in 'HL':
        f, q = np.array(STATES[I], float).T
        fit = r['fits'][I][2]
        E = sim[I]['E'].astype(np.float32).astype(float)[r['keep']][:, :, r['cols']]
        lt = float(np.sum(single.window_posterior(E, f, q, np.array([B[1], B[1]]))['ln_c']))
        lr = 2.0 * (fit['lnl'] - lt)
        msg.append('%s: f %s +- %s, n_w q %s +- %s, weights %s; 2 dlnL against the planted model %.2f (bound %.1f); '
                   'ln L power %.3f, bins %.3f' % (
                       I, np.round(fit['f'], 4), np.round(fit['err_f'], 4), np.round(NW * fit['q'], 3),
                       np.round(NW * fit['err_q'], 3), np.round(fit['a'], 3), lr, bound, fit['lnl'],
                       rb['fits'][I][2]['lnl']))
        assert np.all(np.abs(fit['f'] - f) < 3.5 * fit['err_f']) and np.all(np.abs(fit['q'] - q) < 3.5 * fit['err_q'])
        assert -1e-6 <= lr < bound and np.allclose(fit['pi'], fit['a'] @ B, rtol=0, atol=1e-12), msg[-1]
        assert rb['fits'][I][2]['lnl'] >= fit['lnl'] - 1e-6
        assert r['model'][I]['n_parameters'] == 1 + 2 * 6 and rb['model'][I]['n_parameters'] == 1 + 2 * 9
    print('\n' + '\n'.join(msg))
    assert r['model']['settings']['gammas'] == [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    assert 'weights of rho^-g' in '\n'.join(r['lines'])


@pytest.fixture(scope='module')
def small(tmp_path_factory):
    """600 segments of the toy in three files"""
    folder = str(tmp_path_factory.mktemp('small') / 'ok')
    cfg, sim = config(), simulate(600, SEED + 2)
    write(folder, cfg, sim, ((0, 200), (200, 400), (400, 600)))
    return folder, cfg, sim


def broken(small, tmp_path, name):
    folder = str(tmp_path / name)
    shutil.copytree(small[0], folder)
    return folder


def test_windows_fit_refuses(small, tmp_path, monkeypatch, capsys):
    """windows_fit stops, naming the problem, before any fit: no files; a gap (k 200 to 399 missing); a segment in two
    files; a file whose k is not that of its name; a file of another campaign (seed); a file of the merger stage named
    win; a fitted range outside the stored bins.  A fit that does not converge: exit status 1, NOT CONVERGED, no file"""
    folder, cfg, sim = small
    assert windows_fit.load(folder)[1]['k'].size == 600
    with pytest.raises(SystemExit, match='no win'):
        windows_fit.load(str(tmp_path))
    gap = broken(small, tmp_path, 'gap')
    os.remove(os.path.join(gap, 'win_%010d_%010d.npz' % (200, 400)))
    with pytest.raises(SystemExit, match='k 200 to 399'):
        windows_fit.load(gap)
    twice = broken(small, tmp_path, 'twice')
    write(twice, cfg, sim, ((350, 450),))
    with pytest.raises(SystemExit, match='more than one file: k 350 to 449'):
        windows_fit.load(twice)
    moved = broken(small, tmp_path, 'moved')
    os.rename(os.path.join(moved, 'win_%010d_%010d.npz' % (400, 600)), os.path.join(moved, 'win_%010d_%010d.npz'
                                                                                     % (401, 601)))
    with pytest.raises(SystemExit, match='other than k 401 to 600'):
        windows_fit.load(moved)
    other = broken(small, tmp_path, 'other')
    write(other, config('--seed', '2'), sim, ((200, 400),))
    with pytest.raises(SystemExit, match='another campaign.*seed'):
        windows_fit.load(other)
    merger = broken(small, tmp_path, 'merger')
    cfg_m = {key: v for key, v in cfg.items() if key not in plant.STAGE_KEYS}
    plant.write_batch(os.path.join(merger, 'win_%010d_%010d.npz' % (0, 200)), cfg_m, dict(first=0, end=200),
                      dict(k=np.arange(200)))
    with pytest.raises(SystemExit, match='not a file of the windows stage'):
        windows_fit.load(merger)
    with pytest.raises(SystemExit, match='hold the amplitude bins from rho 3 to 14'):
        windows_fit.fit(folder, rho_max=20.0)
    real = mixture.fit_windows
    monkeypatch.setattr(windows_fit.mixture, 'fit_windows', lambda E, n_state=2, basis=None: real(
        E, n_state=n_state, basis=basis, max_iter=2))
    capsys.readouterr()
    assert windows_fit.main([folder]) == 1
    out = capsys.readouterr().out
    assert 'NOT CONVERGED' in out and 'no model written' in out
    assert not os.path.exists(os.path.join(folder, 'glitch_model.json'))


def week_folder(small, tmp_path):
    """the toy of `small` as one variant of a week: files of analyse.py --stage windows (its campaign meta, which has no
    segment length T, source 'strain'), <out>/slide+0.5/win_*.npz and the selection.txt of <out> that windows_fit reads
    for a week.  returns (the variant's folder, the campaign)"""
    spec = importlib.util.spec_from_file_location('analyse', os.path.join(ROOT, 'scripts', 'real_data', 'analyse.py'))
    analyse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyse)
    segs = tmp_path / 'segments'
    os.makedirs(segs)
    with open(segs / 'manifest.txt', 'w') as fh:
        fh.write('# toy week\nepoch_1266933768.npz 1266933768 600 127 127 0 0 0 33 33 1 0.8\n')
    a = analyse.parse(['x', 'cpu', str(segs), '--out', 'x', '--stage', 'windows', '--n-window', str(NW)])[1]
    cfg = analyse.campaign(a, str(segs), 'slide+0.5')
    out = tmp_path / 'week_w'
    folder = out / 'slide+0.5'
    os.makedirs(folder)
    with open(out / 'selection.txt', 'w') as fh:
        fh.write('epoch_1266933768.npz 1266933768 600 600 0\n')
    for path in sorted(glob.glob(os.path.join(small[0], 'win_*.npz'))):
        with np.load(path) as z:
            arrays = {key: z[key] for key in z.files if key != 'meta'}
        first, end = (int(x) for x in os.path.basename(path)[4:-4].split('_'))
        plant.write_batch(str(folder / os.path.basename(path)), cfg, dict(first=first, end=end), arrays)
    return str(folder), cfg


def test_windows_fit_week_folder_has_no_segment_length(small, tmp_path, capsys):
    """the campaign meta of a week (analyse.py --stage windows) has no T, a plant's has: windows_fit fits the folder of a
    week (the report line of the segments has no window length, and says everything else, and the files are those of
    source 'strain'), writes the model of the campaign key of the files and exits 0; the same toy with a plant's meta
    gives the line with the window length T / n_window = 0.25 s"""
    folder, cfg = week_folder(small, tmp_path)
    assert 'T' not in cfg and cfg['source'] == 'strain' and cfg['variant'] == 'slide+0.5'
    capsys.readouterr()
    r = windows_fit.fit(folder, record=(1,))
    assert r['ok'] and 'T' not in r['cfg']
    line = [l for l in r['lines'] if 'segments in' in l][0]
    assert line == '600 segments in 3 files, k 0 to 599; 32 windows; evidences stored on rho 3 to 14', line
    assert r['lines'][1].startswith('recorded strain segments (manifest sha256 ') and 'variant slide+0.5' in r['lines'][1]
    assert windows_fit.main([folder, '--record', '1']) == 0
    out = capsys.readouterr().out
    model, sha = plant.read_model(os.path.join(folder, 'glitch_model.json'))
    assert 'written %s' % os.path.join(folder, 'glitch_model.json') in out and sha in out
    assert model['campaign'] == plant.campaign_key(cfg) and model['n_window'] == NW and model['segments']['total'] == 600
    plain = windows_fit.fit(small[0], record=(1,))
    assert '600 segments in 3 files, k 0 to 599; 32 windows of 0.25 s; evidences stored on rho 3 to 14' in plain['lines']


REAL_FIT_WINDOWS = mixture.fit_windows


def count_fits(monkeypatch, unconverged=()):
    """mixture.fit_windows as windows_fit calls it, at most 3 updates, reporting no convergence for the numbers of
    states in `unconverged`; the numbers of states of its calls are appended to the returned list"""
    real, calls = REAL_FIT_WINDOWS, []

    def fit_windows(E, n_state=2, basis=None):
        calls.append(n_state)
        return dict(real(E, n_state=n_state, basis=basis, max_iter=3), converged=n_state not in unconverged)
    monkeypatch.setattr(windows_fit.mixture, 'fit_windows', fit_windows)
    return calls


def test_windows_fit_record_and_convergence(small, tmp_path, monkeypatch, capsys):
    """--record lists the numbers of states fitted besides --n-state (default 1,2,3), and nothing else is fitted (one
    call per detector and number): --record 1 --n-state 2 fits 1 and 2, --record '' only 2, --record 1,3 --n-state 2
    fits 1, 2 and 3, --record 2 --n-state 3 fits 2 and 3.  A fit of a number of states that is only a record that does
    not converge does not stop the model (written, exit 0, said in the report as for the record only), a fit of the
    model written that does not converge does (exit 1, NOT CONVERGED naming the detectors and the number of states, no
    file, and an existing file is left as it was)"""
    folder = broken(small, tmp_path, 'record')
    model_path = os.path.join(folder, 'glitch_model.json')
    for argv, fitted in ((['--record', '1', '--n-state', '2'], [1, 2]), (['--record', '', '--n-state', '2'], [2]),
                         (['--record', '1,3', '--n-state', '2'], [1, 2, 3]), (['--record', '2', '--n-state', '3'], [2, 3]),
                         ([], [1, 2, 3])):
        calls = count_fits(monkeypatch)
        capsys.readouterr()
        assert windows_fit.main([folder] + argv) == 0, argv
        assert sorted(calls) == sorted(fitted * 2) and os.path.exists(model_path), (argv, calls)
        assert [f['n_state'] for f in json.loads(open(model_path).read())['fits']['H']] == fitted
        os.remove(model_path)
    count_fits(monkeypatch, unconverged=(1, 3))                  # the record fits do not converge: the model is written
    capsys.readouterr()
    assert windows_fit.main([folder]) == 0
    out = capsys.readouterr().out
    assert 'among the fits kept for the record only: H with 1 states, H with 3 states, L with 1 states, L with 3 states' \
        in out and 'NOT CONVERGED' not in out and os.path.exists(model_path)
    sha = plant.read_model(model_path)[1]
    count_fits(monkeypatch, unconverged=(2,))                    # the model's own does not: no model, the old one stays
    assert windows_fit.main([folder]) == 1
    out = capsys.readouterr().out
    assert 'NOT CONVERGED: H with 2 states, L with 2 states; no model written' in out and 'written %s' % model_path not in out
    assert plant.read_model(model_path)[1] == sha
    os.remove(model_path)
    count_fits(monkeypatch, unconverged=(2,))
    assert windows_fit.main([folder, '--n-state', '3']) == 0 and os.path.exists(model_path)   # 2 is a record now
    os.remove(model_path)
    count_fits(monkeypatch, unconverged=(3,))
    assert windows_fit.main([folder, '--n-state', '3']) == 1 and not os.path.exists(model_path)


def test_script_fits_with_one_blas_thread():
    """windows_fit.py run as a script sets one BLAS thread before numpy is loaded, whatever the environment says (the
    model then does not depend on the number of cores: measured on 32 cores, where the model of 32 threads differs
    from that of one by 1e-15 to 1e-12 and one thread repeats bit for bit); imported as a module it leaves the
    environment as it is.  Fires on: the setting removed, or placed after the import of numpy"""
    import subprocess
    import sys
    names = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
    script = os.path.join(ROOT, 'scripts', 'validate', 'windows_fit.py')
    code = ('import builtins, os, runpy, sys\n'
            'real = builtins.__import__\n'
            'seen = []\n'
            'def spy(name, *a, **k):\n'
            '    if name == "numpy" and not seen:\n'
            '        seen.append(",".join(os.environ.get(v, "unset") for v in %r))\n'
            '    return real(name, *a, **k)\n'
            'builtins.__import__ = spy\n'
            'sys.argv = ["windows_fit.py", "--help"]\n'
            'try:\n'
            '    runpy.run_path(%r, run_name=%%r)\n'
            'except SystemExit:\n'
            '    pass\n'
            'print("AT_NUMPY " + seen[0])\n' % (names, script))
    env = dict(os.environ, **{v: '7' for v in names})
    for run_name, want in (('__main__', '1,1,1,1'), ('windows_fit', '7,7,7,7')):
        out = subprocess.run([sys.executable, '-c', code % run_name], env=env, capture_output=True, text=True, check=True)
        line = [l for l in out.stdout.splitlines() if l.startswith('AT_NUMPY ')]
        assert line == ['AT_NUMPY ' + want], (run_name, out.stdout[-300:], out.stderr[-300:])

