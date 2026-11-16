"""pe_redshift_epochs: several epochs (sets of segments on their own noise curves, so their own banks) in one pass over
the tables give, for every epoch, the arrays of a pe_redshift call of its own, bit for bit.

The reference is pe.py as committed before the batched path (COMMITTED, imported from git as a module of
exmarg.search, as tests/test_pe_unequal.py does): one call per epoch.  The new pe_redshift (the call of one epoch) is
compared with it too.

Stores: the incoherent stores of tests/test_pe_unequal.py on the nodes S_NODES, each node with its own beta grid, read
two nodes per launch (groups [0, 1], [2]), so that a launch holds nodes some epochs need and others do not.  Epochs
(4 s up to 128 Hz, templates of 40, 60 and 90 M_sun, three rows each: a merger with a glitch in H, noise, noise with a
glitch in L; each epoch its own noise draw):

    equal curves (w = 0, one store of asym 0): 'eq1' the design curve, 'eq4' four times it (u four times smaller),
    'tilt' the design curve times 1 + (f / 60 Hz)^2.  With zmin = 1 (trim) 'eq1' reads nodes 1, 2 and 'eq4' nodes
    0, 1, 2: the launch of nodes 0, 1 is copied for both, and 'eq1' reads node 1 with node 1's grids.
    unequal curves (the mixture of the stores MIX): 'shaped' (w = -0.24, -0.27, -0.35), 'shaped2' (another shape and
    level, w = -0.25, -0.28, -0.36), 'w-0.4' (every template at w = -0.4: the store of -0.15 is not read for it).

Each case analyses the three epochs of its kind with chunks of 2 segments (chunks that span epochs and split them)
and with one chunk for all, with per-epoch seeds, rows and (in two cases) glitch settings of their own.

The driver scripts/real_data/analyse.py, whose merger stage now makes the calls of an output group in one
pe_redshift_epochs pass, is compared with the committed driver (one pe_redshift call per epoch, or per epoch and
variant) on a small synthetic week with a small store: the files, bit for bit."""

import functools
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

from exmarg.pp import simulate
from exmarg.search import mass_bank, pe
from test_pe_unequal import FMAX, FMIN, MASSES, T4, WINDOWS, curves, design_sig2, inc_store
from test_pe_windows import add_glitch, law

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = "ee47f02"                       # pe.py before pe_redshift_epochs
NEW = {"wnet"}                              # outputs added after COMMITTED (with single or glitch)
S_NODES = (0.5, 2.0, 8.0)
MIX = (-0.45, -0.30, -0.15)
GRID = dict(nz=40, nm1=30, m1_range=(10.0, 60.0))
COMMON = dict(ymin=0.5, nodes_per_launch=2, **GRID)
SEL = dict(n_loud=256, n_random=256)        # the arrival times of each segment drawn with its epoch's seed + row
# a window model of its own for one epoch: other state probabilities and glitch rates
OWN = dict(WINDOWS, H=dict(f=[0.7, 0.3], q=[0.02, 0.1], pi_k=np.stack([law(4.0, 4.0), law(3.0, 5.0)])),
           L=dict(f=[0.95, 0.05], q=[0.005, 0.3], pi_k=np.stack([law(4.0, 4.0), law(2.0, 6.0)])))


@pytest.fixture(scope="module")
def committed(tmp_path_factory):
    """pe.py as committed at COMMITTED, imported as a module of exmarg.search"""
    try:
        src = subprocess.run(["git", "show", COMMITTED + ":src/exmarg/search/pe.py"], cwd=ROOT, capture_output=True,
                             check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        pytest.skip("no git history to compare with (%s)" % err)
    path = tmp_path_factory.mktemp("committed") / "pe_committed.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("exmarg.search._pe_committed", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "pe_redshift_epochs")
    return mod


def epoch_curves(name):
    f, s2 = design_sig2()
    if name in ("eq1", "eq4", "tilt"):
        s = dict(eq1=s2, eq4=4.0 * s2, tilt=s2 * (1.0 + (f / 60.0) ** 2))[name]
        return dict(H=s, L=s)
    if name == "shaped2":
        return dict(H=2.5 * s2 * (1.0 + 6.0 * (f / 100.0) ** 3), L=1.5 * s2 * (1.0 + 8.0 * (35.0 / f) ** 2))
    return curves("shaped" if name == "shaped" else -0.4)


@functools.lru_cache(maxsize=None)
def epoch(name, seed):
    """the three segments of an epoch on the curves of name (module docstring) and the bank built on them"""
    y = iter([12.0, 1e-8])
    raw, _, _ = simulate(2, seed, lambda r: (60.0, 1.0, 0.0), lambda r: next(y), T4, FMIN, FMAX, sig2=epoch_curves(name))
    g = lambda I, d, t, rho, ph: add_glitch(d, dict(raw, sig2=raw["sig2_" + I]), t, rho, 60.0, ph)
    dH = [g("H", raw["d_H"][0], 1.0, 15.0, 0.3), raw["d_H"][1], raw["d_H"][1]]
    dL = [raw["d_L"][0], raw["d_L"][1], g("L", raw["d_L"][1], 2.5, 12.0, 0.7)]
    segs = dict(raw, d_H=np.array(dH), d_L=np.array(dL))
    return segs, mass_bank(MASSES, segs)


def stores(kind):
    return dict(one=inc_store(0.0, S_NODES), mix=[inc_store(a, S_NODES) for a in MIX[::-1]],
                tol=[inc_store(a, S_NODES) for a in MIX[:2]])[kind]


def bitwise(a, b):
    """the same array (dtype, shape and bytes) or the same value"""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


CASES = {   # store kind, epochs (name, seed, own settings), settings of the call
    "plain, one store": ("one", [("eq1", 11, {}), ("eq4", 12, dict(rows=[2, 0])), ("tilt", 13, {})], dict(SEL)),
    "single, one store, zmin": ("one", [("eq1", 11, {}), ("eq4", 12, {}), ("tilt", 13, {})],
                                dict(single=True, zmin=1.0, n_loud=0)),
    "windows, one store, zmin": ("one", [("eq1", 11, {}), ("eq4", 12, dict(glitch=OWN)), ("tilt", 13, {})],
                                 dict(glitch=WINDOWS, zmin=1.0, **SEL)),
    "windows, mixture": ("mix", [("shaped", 21, {}), ("shaped2", 22, {}), ("w-0.4", 23, dict(glitch=OWN))],
                         dict(glitch=WINDOWS, **SEL)),
    "default glitch model, mixture, zmin": ("mix", [("shaped", 21, {}), ("shaped2", 22, {}), ("w-0.4", 23, {})],
                                            dict(glitch=True, zmin=1.0, n_loud=0)),
    "windows, mixture, asym_tol": ("tol", [("shaped", 21, {}), ("shaped2", 22, {}), ("w-0.4", 23, {})],
                                   dict(glitch=WINDOWS, asym_tol=0.1, **SEL)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_batched_equals_per_epoch(committed, case):
    """for each case of CASES: pe_redshift_epochs with chunks of 2 and with one chunk returns, for every epoch, the
    keys and every array of the committed pe_redshift called on that epoch alone (its seed, rows and glitch
    settings), bit for bit, and so does the new pe_redshift of one epoch"""
    kind, names, kw = CASES[case]
    st = stores(kind)
    eps, refs = [], []
    for name, seed, own in names:
        segs, bank = epoch(name, seed)
        ep = dict(segments=segs, bank=bank, seed=seed, **own)
        eps.append(ep)
        args = dict(kw, **{k: v for k, v in ep.items() if k not in ("segments", "bank")})
        refs.append((committed.pe_redshift(segs, bank, st, **COMMON, **args),
                     pe.pe_redshift(segs, bank, st, **COMMON, **args)))
    n = 0
    for chunk in (2, 100):
        got = pe.pe_redshift_epochs(eps, st, chunk=chunk, **COMMON, **kw)
        assert len(got) == len(eps)
        for (name, _, _), r, (old, new) in zip(names, got, refs):
            assert set(r) == set(new) and set(r) - NEW == set(old), (case, name)
            for k in old:
                assert bitwise(r[k], old[k]), (case, chunk, name, k)
                assert bitwise(new[k], old[k]), (case, name, k)
                n += 1
            for k in set(r) - set(old):                 # added after COMMITTED: batched as the call of one epoch
                assert bitwise(r[k], new[k]), (case, chunk, name, k)
    if kw.get("asym_tol"):
        assert [r["asym_outside"] for r in got] == [2, 2, 0]
    print("\n%s: %d arrays bit for bit equal to the per-epoch calls" % (case, n))


def test_union_of_nodes_and_stores():
    """the case the module docstring describes holds: with zmin the epochs 'eq1' and 'eq4' need different nodes of the
    launch [0, 1] (so a launch holds the union and 'eq1' reads with the grids of node 1, not node 0), and in the
    mixture the epoch 'w-0.4' reads no table of the store of -0.15, the others do"""
    need = {}
    for name in ("eq1", "eq4"):
        segs, bank = epoch(name, 11)
        geometry = pe._node_geometry(bank, stores("one"), 0.5, float(pe.D0 / pe._dl_of_z(1.0)[0]))
        need[name] = np.nonzero(pe._plan(bank, stores("one"), geometry, trim=True)[0]["need"])[0].tolist()
    assert need == {"eq1": [1, 2], "eq4": [0, 1, 2]}
    assert not np.array_equal(stores("one").beta[0], stores("one").beta[1])
    used = {}
    for name in ("shaped", "w-0.4"):
        segs, bank = epoch(name, 21)
        geometry = pe._node_geometry(bank, stores("mix")[0], 0.5)
        used[name] = sorted(e["asym"] for e in pe._plan(bank, stores("mix"), geometry, trim=True))
    assert used == {"shaped": sorted(MIX), "w-0.4": [-0.45, -0.30]}


def small_week(folder, n=2):
    """the synthetic week of tests/test_analyse.py (four epochs of 8 s up to 512 Hz, the second unusable) cut to the
    first n segments of each: n + 0 + n + n selected"""
    import os
    from exmarg import strain
    from test_analyse import EPOCHS, NSEG, make_epoch
    os.makedirs(folder, exist_ok=True)
    lines = ['# small week\n']
    for i, e in enumerate(EPOCHS):
        ep = make_epoch(e['gps'], 100 + i, e.get('loud_H', ()), e.get('loud_L', ()), e.get('event', ()),
                        e.get('usable', True))
        ep = {k: v[:n] if np.ndim(v) and np.shape(v)[0] == NSEG else v for k, v in ep.items()}
        strain.write_epoch(os.path.join(folder, 'epoch_%d.npz' % e['gps']), ep)
        lines.append('epoch_%d.npz %d %d 127 127 0 0 0 %d %d %d 0.8\n' % (e['gps'], e['gps'], n, n, n, ep['usable']))
    with open(os.path.join(folder, 'manifest.txt'), 'w') as fh:
        fh.writelines(lines)
    return folder


def same_files(a, b, new=('wnet',)):
    """every result file of folder a is in folder b with the same members in the same order, every array the same bit
    for bit and the same meta but for the run's own record (times, process): the number of arrays compared.  The
    members new (wnet, added after COMMITTED) are in the files of a only, and are not compared"""
    import glob
    import json
    import os
    names = sorted(os.path.relpath(f, a) for f in glob.glob(os.path.join(a, '*', 'plant_*.npz')))
    assert names and names == sorted(os.path.relpath(f, b) for f in glob.glob(os.path.join(b, '*', 'plant_*.npz')))
    n = 0
    for name in names:
        with np.load(os.path.join(a, name)) as x, np.load(os.path.join(b, name)) as y:
            assert set(new) <= set(x.files) and [f for f in x.files if f not in new] == y.files, name
            ma, mb = (json.loads(str(z['meta'])) for z in (x, y))
            ra, rb = ma.pop('run'), mb.pop('run')
            assert ma == mb and {k: ra[k] for k in ('first', 'end', 'epochs', 'chunk')} == {
                k: rb[k] for k in ('first', 'end', 'epochs', 'chunk')}, name
            for key in y.files:
                if key != 'meta':
                    assert bitwise(x[key], y[key]), (name, key)
                    n += 1
    return n


@pytest.fixture(scope="module")
def committed_driver(tmp_path_factory):
    """scripts/real_data/analyse.py as committed at COMMITTED (a pe_redshift call per epoch, or per epoch and variant),
    imported from git"""
    from test_analyse import analyse  # noqa: F401  (puts plant and the scripts on the path first)
    try:
        src = subprocess.run(["git", "show", COMMITTED + ":scripts/real_data/analyse.py"], cwd=ROOT, capture_output=True,
                             check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        pytest.skip("no git history to compare with (%s)" % err)
    path = tmp_path_factory.mktemp("committed_driver") / "analyse_committed.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("analyse_committed", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "analyse_calls")
    return mod


@pytest.mark.parametrize("model", [False, True])
def test_driver_batched_equals_committed(tmp_path, committed, committed_driver, monkeypatch, model):
    """analyse.main with the tables of a small store on the CPU, the merger stage made in one pe_redshift_epochs pass
    per group of two epochs (chunks of 3 segments, which span epochs and calls), writes the files of the committed
    driver calling the committed pe_redshift once per epoch (and variant), bit for bit: with the default glitch model
    (the slotted variants pr and rev in one call, the extra slide+2.25 in a call of its own with its own seed) and with
    a window model per variant (--glitch-model auto, distinct models, a call per variant)"""
    import exmarg.search as search
    from exmarg.search import load_redshift_store
    from test_analyse import WinStub, analyse, plant
    from test_plant import make_model
    from test_store import write_store
    week = small_week(str(tmp_path / 'segments'))
    store = load_redshift_store(write_store(str(tmp_path / 'store')), unbiased=True)
    new = analyse.make_analysis(store, 'cpu', chunk=3, ymin=plant.default_ymin())
    monkeypatch.setattr(search, 'pe_redshift', committed.pe_redshift)        # what the committed driver binds
    old = committed_driver.make_analysis(store, 'cpu', chunk=3, ymin=plant.default_ymin())
    variants = 'pr,slide+2.25' if model else 'pr,rev,slide+2.25'
    outs = []
    for name, driver, factory in (('batched', analyse, new), ('committed', committed_driver, old)):
        out = tmp_path / name
        extra = []
        if model:
            driver.main(['none', 'cpu', week, '--out', str(out), '--seed', '7', '--epochs-per-file', '2',
                         '--variants', variants, '--stage', 'windows'], factory=WinStub().factory)
            for i, v in enumerate(variants.split(',')):
                meta = plant.read_meta(sorted((out / v).glob('win_*.npz'))[0])
                make_model(out / v / 'glitch_model.json', plant.campaign_key(meta), q_scale=1.0 + i)
            extra = ['--glitch-model', 'auto']
        assert driver.main(['nostore', 'cpu', week, '--out', str(out), '--seed', '7', '--epochs-per-file', '2',
                            '--variants', variants, '--chunk', '3'] + extra, factory=factory) == 0
        outs.append(out)
    n = same_files(*outs)
    with np.load(sorted((outs[0] / 'pr').glob('plant_*.npz'))[0]) as x:
        assert np.isfinite(x['ln_lambda_bar']).all() and x['k'].size == 2
    print("\n%s: %d arrays of the driver's files bit for bit equal" % ('window model' if model else 'default model', n))


def test_epoch_settings_checked():
    """an epoch without segments or bank, or with an unknown key, is a ValueError; no epoch gives no result"""
    segs, bank = epoch("eq1", 11)
    for bad in (dict(segments=segs), dict(segments=segs, bank=bank, chunk=4)):
        with pytest.raises(ValueError, match="keys"):
            pe.pe_redshift_epochs([bad], stores("one"), **COMMON)
    assert pe.pe_redshift_epochs([], stores("one"), **COMMON) == []


def test_analysis_batch_wiring(tmp_path, monkeypatch):
    """analyse.make_analysis: pe.epoch(segments, seed, rows, glitch) is the epoch of those arguments with the epoch's bank,
    and pe.batch passes to pe_redshift_epochs the store and every setting that pe passes to pe_redshift (zmin and
    asym_tol only when given)"""
    import exmarg.search as search
    from test_analyse import analyse
    seen = {}
    monkeypatch.setattr(search, 'pe_redshift', lambda seg, bank, store, **kw: seen.update(one=(bank, store, kw)))
    monkeypatch.setattr(search, 'pe_redshift_epochs', lambda epochs, store, **kw: seen.update(many=(epochs, store, kw)))
    segs, _ = epoch("eq1", 11)
    for cut in (dict(zmin=0.5, asym_tol=0.05), {}):
        pe_ = analyse.make_analysis(['a', 'b'], 'cpu', chunk=5, ymin=0.2, **cut)(segs)
        pe_(segs, 17, [0, 1], 'g')
        pe_.batch([pe_.epoch(segs, 17, [0, 1], 'g')])
        bank, store, kw = seen['one']
        (ep,), store2, kw2 = seen['many']
        assert store == store2 == ['a', 'b'] and ep == dict(segments=segs, bank=bank, rows=[0, 1], seed=17, glitch='g')
        assert kw == dict(kw2, rows=[0, 1], seed=17, glitch='g') and kw2['chunk'] == 5 and kw2['ymin'] == 0.2
        assert {k: kw2.get(k) for k in ('zmin', 'asym_tol')} == {k: cut.get(k) for k in ('zmin', 'asym_tol')}
