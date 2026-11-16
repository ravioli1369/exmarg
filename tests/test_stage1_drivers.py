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
        assert set(a.files) == set(b.files) | {'ln_merger_H', 'ln_merger_L'}
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
