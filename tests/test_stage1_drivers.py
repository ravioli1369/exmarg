"""the bookkeeping of the mergers in stage 1 (docs/2026-11-15_mergers_in_stage1.md) in the drivers: plant.py's windows
stage with a store writes the merger evidence and records its population; windows_fit.py --nbar fits with it and records
nbar and the population in the model, and refuses controls of the data and files without it; the merger stages refuse a
model fitted with the mergers of another population, and analyse.py such a model for its control variants; iterate.py
chains the passes.  The store here is a folder of store headers (bd_*.npz with the small arrays read_header reads, no
tables): the windows stage reads the ladder alone."""
import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from exmarg.search import population as popavg
import test_stage1_mergers as tm

ROOT = Path(__file__).resolve().parents[1]
S_LADDER = tm.S_LADDER


def _script(name, folder='validate'):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / folder / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plant, windows_fit, iterate = _script('plant'), _script('windows_fit'), _script('iterate')
analyse = _script('analyse', 'real_data')


def header_store(folder, s=S_LADDER):
    """a store folder of headers only: bd_n120_s<s>.npz with s_node, alpha, beta, delta, tau_idx"""
    os.makedirs(folder, exist_ok=True)
    for x in s:
        np.savez(os.path.join(folder, 'bd_n120_s%012.6f.npz' % x), s_node=x, alpha=np.array([0.16, 4e4]),
                 beta=np.linspace(-1, 1, 3), delta=np.linspace(-1, 1, 3), tau_idx=np.arange(-2, 3))
    return folder


CAMPAIGN = ['--nbar', '0.02', '--seed', '3', '--glitch-states-H', '0.9:0.002,0.1:0.05', '--glitch-states-L',
            '0.9:0.002,0.1:0.05', '--gamma', '4', '--rho-min', '5', '--zmin', '0.5']


def test_windows_stage_writes_merger_evidence(tmp_path, capsys):
    """plant.py --stage windows with a store folder: the win files hold ln_merger_H, ln_merger_L (n, 64) float64 next
    to the arrays of the stage without a store, which are the same bit for bit; the campaign records merger_population
    (zmin, mdet_range, grid, ymin, the ladder of the store), a stage setting: the campaign_key is that of the run
    without a store.  The mean of M_w over the windows is 0.9 to 1.1 on a segment without merger or glitch.  --ymin with the
    windows stage needs the store"""
    store = header_store(str(tmp_path / 'store'))
    out_s, out_n = str(tmp_path / 'with'), str(tmp_path / 'none')
    argv = ['cpu', '--stage', 'windows', '--n', '2', '--batch', '2', '--nbar', '0', '--seed', '3', '--glitch-states',
            '0.9:0.0001,0.1:0.001', '--gamma', '4', '--rho-min', '5', '--zmin', '0.5']
    assert plant.main([store] + argv + ['--out', out_s]) == 0
    assert plant.main(['none'] + argv + ['--out', out_n]) == 0
    capsys.readouterr()
    name = 'win_0000000000_0000000002.npz'
    with np.load(os.path.join(out_s, name)) as a, np.load(os.path.join(out_n, name)) as b:
        ma, mb = json.loads(str(a['meta'])), json.loads(str(b['meta']))
        assert set(a.files) == set(b.files) | {'ln_merger_H', 'ln_merger_L', 'wnet'}       # wnet: the lags of the store
        for k in b.files:
            if k != 'meta':
                assert np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), k
        M = np.exp(a['ln_merger_H'])
        quiet = ~a['merger'] & (a['n_glitches_H'] == 0)                      # no merger planted, glitches rare
        assert a['ln_merger_H'].shape == (2, 64) and a['ln_merger_L'].dtype == np.float64 and quiet.any()
        assert np.all(np.abs(M[quiet].mean(axis=1) - 1) < 0.1), (M.mean(axis=1), quiet)
    pop = ma['merger_population']
    assert pop['s'] == [float(x) for x in S_LADDER] and pop['zmin'] == 0.5 and pop['ymin'] == plant.default_ymin()
    assert pop['grid'] == json.loads(json.dumps(plant.GRID)) and pop['mdet_range'] == list(plant.MDET) and 'merger_population' not in mb
    assert plant.campaign_key({k: v for k, v in ma.items() if k != 'run'}) == plant.campaign_key(
        {k: v for k, v in mb.items() if k != 'run'})
    with pytest.raises(SystemExit):
        plant.main(['none'] + argv + ['--out', str(tmp_path / 'x'), '--ymin', '0.1'])
    assert 'give the store' in capsys.readouterr().err


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    """the toy campaign of test_windows_fit.py with merger evidence (random ln_merger_I, merger_population of the
    ladder), and the same files without it"""
    base = tmp_path_factory.mktemp('drivers')
    new, old = str(base / 'new'), str(base / 'old')
    cfg = tm.write_toy_folder(new, True)
    tm.write_toy_folder(old, False)
    return new, old, cfg


def test_windows_fit_nbar(toy, tmp_path, capsys):
    """windows_fit.py --nbar 0.05: every fit with mergers (fit0 the fit without them, its start, reported next to it),
    converged; the model records nbar 0.05 and the merger_population of the files and is written to --model; it differs
    from the model at nbar 0 only by those entries, f, q, pi_k, a, their errors and the fits' ln L; plant.glitch_model
    takes it for a run of the same population and refuses it for another (zmin) or an unknown one, and takes the model
    at nbar 0 whatever the population.  Refused: --nbar on files without merger evidence, and on a control (time
    slide)"""
    new, old, cfg = toy
    path = str(tmp_path / 'pass2' / 'glitch_model.json')
    assert windows_fit.main([new, '--nbar', '0.05', '--record', '1', '--model', path]) == 0
    text = capsys.readouterr().out
    assert 'with mergers at nbar 0.05' in text and 'without mergers (nbar 0, the start of the fit)' in text
    m1, sha = plant.read_model(path)
    assert m1['nbar'] == 0.05 and m1['merger_population'] == cfg['merger_population']
    assert windows_fit.main([new, '--record', '1']) == 0
    m0 = plant.read_model(os.path.join(new, 'glitch_model.json'))[0]
    assert 'nbar' not in m0 and 'merger_population' not in m0
    differ = {k for k in set(m0) | set(m1) if m0.get(k) != m1.get(k)}
    assert differ == {'nbar', 'merger_population', 'H', 'L', 'fits', 'sha256'}, differ
    run = dict(cfg)
    for k in plant.STAGE_KEYS:
        run.pop(k, None)
    pop = cfg['merger_population']
    glitch, got = plant.glitch_model(path, run, cfg['n_window'], population=pop)
    assert got == sha and glitch['model'] == 'windows'
    for other in (dict(pop, zmin=0.99119), None):
        with pytest.raises(SystemExit, match='another population'):
            plant.glitch_model(path, run, cfg['n_window'], population=other)
    plant.glitch_model(os.path.join(new, 'glitch_model.json'), run, cfg['n_window'], population=dict(pop, zmin=3.0))
    with pytest.raises(SystemExit, match='no merger evidence'):
        windows_fit.fit(old, nbar=0.05, record=())
    slid = str(tmp_path / 'slid')
    import test_windows_fit as twf
    twf.write(slid, dict(twf.config('--control', 'time-slide'), merger_population=pop), twf.simulate(300, 5),
              ((0, 300),))
    with pytest.raises(SystemExit, match='control'):
        windows_fit.fit(slid, nbar=0.05, record=())
    assert plant.with_model(run, sha, 32, 0.05)['glitch_model_nbar'] == 0.05
    assert 'glitch_model_nbar' not in plant.with_model(run, sha, 32, 0.0)
    assert 'glitch_model_nbar' in plant.MERGER_KEYS


def test_analyse_refuses_model_with_mergers_for_controls(toy, tmp_path):
    """analyse.load_models: a model fitted with mergers given for every variant is refused for rev (and the other
    controls) before anything else is checked, with the reason; the variants that may have one are zero and inj"""
    new, _, cfg = toy
    path = str(tmp_path / 'm.json')
    assert windows_fit.main([new, '--nbar', '0.02', '--record', '', '--model', path]) == 0
    a = argparse.Namespace(glitch_model=path, store=header_store(str(tmp_path / 'store')), zmin=0.5, n_window=32,
                           out=str(tmp_path))
    for v in ('rev', 'pr', 'slide+0.5'):
        with pytest.raises(SystemExit, match='not for the variant %s' % v.replace('+', r'\+')):
            analyse.load_models(a, {v: dict(cfg, ymin=plant.default_ymin())})
    assert analyse.WITH_MERGERS == ('zero', 'inj')


def test_iterate_dry_run(toy, tmp_path):
    """iterate.py --dry-run: pass p fits the model at the N of the pass before into <out>/pass<p>/glitch_model.json
    (windows_fit.py --nbar N --model ...) with the --fit options, then runs the command of stage 2 with {model} and
    {out} replaced; the first pass at --nbar0"""
    new = toy[0]
    lines = []
    out = str(tmp_path / 'iter')
    assert iterate.main([new, out, '--passes', '1', '--nbar0', '0.0184', '--fit', '--rho-min 5 --record 1',
                         '--dry-run', '--', 'python', 'plant.py', 'tables', 'gpu', '--glitch-model', '{model}', '--out',
                         '{out}'], log=lines.append) == 0
    model = os.path.join(out, 'pass1', 'glitch_model.json')
    assert '--nbar 0.0184 --model %s --rho-min 5 --record 1' % model in lines[0]
    assert lines[1].endswith('python plant.py tables gpu --glitch-model %s --out %s' % (model, os.path.join(out, 'pass1')))
    with pytest.raises(SystemExit):
        iterate.main([new, out, '--', 'python', 'plant.py'])


def test_iterate_detected_fraction(tmp_path, monkeypatch):
    """iterate.py --f-det: the detected fraction of the cuts reaches the estimate of a pass (plant_fit.report f_det,
    and so nbar_fit) and is written with it; a pass made with another fraction is refused, not reused; a fraction
    outside [0, 1) is an error"""
    seen = []

    def report(results, veto_w=None, f_det=0.0):
        seen.append((results, veto_w, f_det))
        return dict(fits=dict(windows=dict(nbar=0.02 + f_det, err=dict(nbar=0.004), edge=None)),
                    keep=np.ones(10, bool), cfg=dict(glitch_model_sha256='ab', glitch_model_nbar=0.0))
    monkeypatch.setattr(iterate.plant_fit, 'report', report)
    est = iterate.estimate('folder', 72.0, 0.005)
    assert seen == [('folder', 72.0, 0.005)] and est['f_det'] == 0.005 and abs(est['nbar'] - 0.025) < 1e-15
    assert iterate.estimate('folder', 72.0)['f_det'] == 0.0 and seen[-1][2] == 0.0
    out = tmp_path / 'iter'
    (out / 'pass1').mkdir(parents=True)
    (out / 'pass1' / 'estimate.json').write_text(json.dumps(dict(est, model_nbar=0.0)))
    cmd = ['--', 'python', 'plant.py', '--glitch-model', '{model}', '--out', '{out}']
    lines = []
    assert iterate.main(['win', str(out), '--passes', '1', '--f-det', '0.005', '--dry-run'] + cmd, log=lines.append) == 0
    assert 'pass 1 present' in lines[0]
    with pytest.raises(SystemExit, match='detected fraction 0.005, not 0.0'):
        iterate.main(['win', str(out), '--passes', '1', '--dry-run'] + cmd, log=lines.append)
    with pytest.raises(SystemExit):
        iterate.main(['win', str(out), '--f-det', '1.0', '--dry-run'] + cmd)


# ----------------------------------------------------------------------------------------- after the merge of a79e07d


class GainWinStub:
    """pe.window_evidences in place for analyse.py's windows stage, with the merger evidence: arrays of its shapes
    (zeros; wmax 1), and the gain of the segments each bank is built on (factory's seg0) recorded"""

    def __init__(self):
        self.gains = []

    def factory(self, seg0):
        self.gains.append(seg0.get('gain'))
        return self.win

    def win(self, seg, rows):
        from exmarg.search.single import RHO_EDGES
        n, nr, nm = np.size(rows), RHO_EDGES.size - 1, plant.MASSES.size
        out = dict(rho_edges=RHO_EDGES.copy(), rows=np.asarray(rows))
        for I in 'HL':
            out.update({'ln_window_' + I: np.zeros((n, plant.N_WINDOW, nr)), 'wmax_' + I: np.ones((n, nm)),
                        'ln_merger_' + I: np.zeros((n, plant.N_WINDOW))})
        return out


def test_unequal_windows_stage_takes_the_gain(tmp_path, monkeypatch):
    """analyse.py --stage windows --unequal with the store (merger evidence): the bank of each epoch is built on segments
    that carry the epoch's signal gain (strain.signal_gain), since the merger evidence of detector I is that of the
    model power kappa_I^2 u_I (pe._model_power), and merger_population records signal_gain 'epoch'; with --no-gain no
    gain and signal_gain 'none'; without the store (no merger evidence) no gain, as before the merge, and no
    merger_population.  The window evidences themselves do not depend on the gain (test_stage1_mergers).  The gains of
    the synthetic week are 1 to rounding, so strain.signal_gain is replaced by one of distinct values here"""
    from test_analyse import write_unequal_week
    week = write_unequal_week(str(tmp_path / 'segments'))
    store = header_store(str(tmp_path / 'store'))
    kappa = dict(H=0.97, L=0.95)
    monkeypatch.setattr(analyse.strain, 'signal_gain', lambda ep: dict(kappa))
    for tag, store_arg, extra, want in (('epoch', store, [], 'epoch'), ('nogain', store, ['--no-gain'], 'none'),
                                        ('none', 'none', [], None)):
        stub = GainWinStub()
        out = tmp_path / tag
        assert analyse.main([store_arg, 'cpu', week, '--out', str(out), '--seed', '7', '--stage', 'windows', '--unequal',
                             '--variants', 'zero,rev'] + extra, factory=stub.factory) == 0
        meta = plant.read_meta(sorted((out / 'zero').glob('win_*.npz'))[0])
        if want == 'epoch':
            assert stub.gains and all(g == kappa for g in stub.gains)
            assert meta['merger_population']['signal_gain'] == 'epoch'
            with np.load(sorted((out / 'zero').glob('win_*.npz'))[0]) as z:
                assert 'ln_merger_H' in z.files
        elif want == 'none':
            assert all(g is None for g in stub.gains) and meta['merger_population']['signal_gain'] == 'none'
        else:
            assert all(g is None for g in stub.gains) and 'merger_population' not in meta
        print('%s: %d banks, gains %s' % (tag, len(stub.gains), stub.gains[:2]))


def test_unequal_week_with_stores_that_record_their_cuts(tmp_path):
    """analyse.py, the two stages of the window model on own curves with two stores that record the loudness cuts
    they are answered for (build_ratio.py's stage R; what the search runs on).  The windows stage, given the stores,
    records no cuts; the merger stage (--glitch-model auto) takes the model of each variant, fitted on the window
    evidences of the same campaign, and its files record tables_cuts [72, 84] and tables_asym; plant_fit.load reads
    them and check_table_cuts accepts the selection 72 / 84 and refuses a louder veto.  Fires on: tables_cuts in the
    campaign key (plant.MERGER_KEYS without it): every model is refused as fitted on another campaign"""
    from test_analyse import UStub, write_unequal_week
    from test_plant import make_model
    from test_store import write_store
    plant_fit = _script('plant_fit')
    week = write_unequal_week(str(tmp_path / 'segments'))
    stores = '%s,%s' % (write_store(str(tmp_path / 's0'), asym=0.0, cuts=(72.0, 84.0)),
                        write_store(str(tmp_path / 's6'), asym=-0.6, cuts=(72.0, 84.0)))
    out = tmp_path / 'u'
    variants = ('rev', 'slide+0.5')
    common = ['cpu', week, '--out', str(out), '--seed', '7', '--unequal', '--zmin', '0', '--epochs-per-file', '2',
              '--variants', ','.join(variants)]
    assert analyse.main([stores] + common + ['--stage', 'windows'], factory=GainWinStub().factory) == 0
    for v in variants:
        win = plant.read_meta(sorted((out / v).glob('win_*.npz'))[0])
        assert 'tables_cuts' not in win and 'tables_asym' not in win
        make_model(out / v / 'glitch_model.json', plant.campaign_key(win))
    stub = UStub()
    assert analyse.main([stores] + common + ['--glitch-model', 'auto'], factory=stub.factory) == 0
    assert stub.calls
    for v in variants:
        cfg = plant_fit.load(str(out / v))[0]
        assert cfg['tables_cuts'] == [72.0, 84.0] and cfg['tables_asym'] == [-0.6, 0.0] and cfg['noise'] == 'own curves'
        plant_fit.check_table_cuts(cfg, 72.0, 84.0)
        with pytest.raises(SystemExit, match='not examined'):
            plant_fit.check_table_cuts(cfg, 98.0, 84.0)


def test_batched_merger_stage_reads_a_model_with_mergers(tmp_path):
    """analyse.py's merger stage made in one pe_redshift_epochs pass per group (make_analysis: pe.batch) and the same
    calls made one by one (a pe without batch) write the same files bit for bit with a glitch model fitted with mergers
    (nbar 0.02, the merger_population of the run) given for the variants zero and inj; the files record that model's
    SHA-256 and glitch_model_nbar.  The same model is refused for rev, and for a run of another population (--zmin)"""
    from exmarg.search import load_redshift_store
    from test_analyse import WinStub
    from test_pe_batch import same_files, small_week
    from test_plant import make_model
    from test_store import write_store
    week = small_week(str(tmp_path / 'segments'))
    store_dir = write_store(str(tmp_path / 'store'))
    store = load_redshift_store(store_dir)
    win = tmp_path / 'win'
    analyse.main(['none', 'cpu', week, '--out', str(win), '--seed', '7', '--epochs-per-file', '2', '--variants', 'zero,inj',
                  '--stage', 'windows'], factory=WinStub().factory)
    meta = plant.read_meta(sorted((win / 'zero').glob('win_*.npz'))[0])
    model_path = tmp_path / 'model.json'
    make_model(model_path, plant.campaign_key(meta))
    m = json.load(open(model_path))
    m.pop('sha256')
    m.update(nbar=0.02, merger_population=plant.population_settings(0.99119, plant.default_ymin(),
                                                                    plant.store_ladder(store_dir)))
    sha = windows_fit.write_model(str(model_path), json.loads(json.dumps(m)))
    batched = analyse.make_analysis(store, 'cpu', chunk=3, ymin=plant.default_ymin())

    def one_by_one(seg0):
        pe_ = batched(seg0)
        return lambda *args: pe_(*args)                                  # no batch: analyse_calls calls one by one
    outs = []
    for name, factory in (('batched', batched), ('one_by_one', one_by_one)):
        out = tmp_path / name
        assert analyse.main([store_dir, 'cpu', week, '--out', str(out), '--seed', '7', '--epochs-per-file', '2',
                             '--variants', 'zero,inj', '--chunk', '3', '--glitch-model', str(model_path)],
                            factory=factory) == 0
        outs.append(out)
    n = same_files(*outs)
    meta = plant.read_meta(sorted((outs[0] / 'zero').glob('plant_*.npz'))[0])
    assert meta['glitch_model_sha256'] == sha and meta['glitch_model_nbar'] == 0.02
    with pytest.raises(SystemExit, match='not for the variant rev'):
        analyse.main([store_dir, 'cpu', week, '--out', str(tmp_path / 'rev'), '--seed', '7', '--variants', 'rev',
                      '--glitch-model', str(model_path)], factory=batched)
    with pytest.raises(SystemExit, match='another population'):
        analyse.main([store_dir, 'cpu', week, '--out', str(tmp_path / 'z'), '--seed', '7', '--variants', 'zero',
                      '--zmin', '0.5', '--glitch-model', str(model_path)], factory=batched)
    print('\na model fitted with mergers: %d arrays bit for bit equal, batched and one by one' % n)


def run_committed_plant(tree, argv, cwd):
    """scripts/validate/plant.py of an exported tree, in a process of its own (its own src first on the path)"""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(tree / 'scripts' / 'validate' / 'plant.py')] + argv, cwd=cwd,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def same_win_files(a, b, prefix, new=('wnet',), tables=None):
    """the files of prefix in folders a and b: the same names and members, every array bit for bit, the meta but run;
    the files of b may hold the members new besides (wnet, of the network cut, after a79e07d), not compared; with
    tables, the meta of b holds tables = that and the meta of a none (the entry is recorded in every merger-stage run
    since 2026-11-17), and the rest of the meta is compared"""
    import glob
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a, prefix + '_*.npz')))
    assert names and names == sorted(os.path.basename(f) for f in glob.glob(os.path.join(b, prefix + '_*.npz')))
    n = 0
    for name in names:
        with np.load(os.path.join(a, name)) as x, np.load(os.path.join(b, name)) as y:
            assert set(y.files) - set(x.files) <= set(new) and [f for f in y.files if f in x.files] == x.files, name
            ma, mb = (json.loads(str(z['meta'])) for z in (x, y))
            ma.pop('run'), mb.pop('run')
            if tables is not None:
                assert 'tables' not in ma and mb.pop('tables') == tables, name
            assert ma == mb, name
            for key in x.files:
                if key != 'meta':
                    assert np.array_equal(x[key], y[key], equal_nan=x[key].dtype.kind == 'f'), (name, key)
                    n += 1
    return n


def test_plant_default_outputs_as_main(tmp_path):
    """plant.py's outputs are those of the main branch merged in (a79e07d, its tree in a process of its own): the
    windows stage without a store (2 segments, the plant's bank and window evidences on the CPU), and the merger stage
    with the default glitch terms on a small store (write_store, 2 segments), with --unbias, the default of a79e07d
    (which read the tables with unbias and recorded no tables entry): every array bit for bit, the meta but the run's
    record and the tables entry (unbiased here)"""
    from test_store import write_store
    tree = tm.committed('a79e07d')
    if tree is None:
        pytest.skip('no git history to compare with')
    store = write_store(str(tmp_path / 'store'))
    base = ['--n', '2', '--batch', '2', '--nbar', '0.5', '--seed', '3', '--glitch-states', '0.9:0.01,0.1:0.1',
            '--gamma', '4', '--rho-min', '5', '--zmin', '0.5']
    n = 0
    for stage, store_arg, extra, now, prefix in (('windows', 'none', ['--stage', 'windows'], [], 'win'),
                                                 ('merger', store, ['--chunk', '2'], ['--unbias'], 'plant')):
        a, b = tmp_path / ('main_' + stage), tmp_path / ('merged_' + stage)
        run_committed_plant(tree, [store_arg, 'cpu'] + base + extra + ['--out', str(a)], str(tmp_path))
        assert plant.main([store_arg, 'cpu'] + base + extra + now + ['--out', str(b)]) == 0
        n += same_win_files(str(a), str(b), prefix, tables='unbiased' if now else None)
    print('\nplant.py, windows stage without a store and merger stage: %d arrays bit for bit those of a79e07d' % n)
