"""scripts/real_data/condition.py on a synthetic GWOSC folder (no network, no real data).

Two detectors, two consecutive files of 300 s at 128 Hz (white noise).  H1 fails CBC_CAT2 at second 20 (finite strain,
a veto second) and BURST_CAT2 at second 450 (not required: only recorded); L1 has no data at seconds 420 to 430 (NaN,
DATA unset) and fails NO_STOCH_HW_INJ at second 560.  The stretches are then [21, 420), which crosses the file
boundary at 300, and [430, 560); [0, 20) and [561, 600) are shorter than two epochs.  Epochs of N + 1 = 8 segments of
4 s with 4 s of pad: 12 and 3 epochs, against 8 + 3 + 3 if the stretch were cut at the file boundary.  H1 is 30 times
larger at seconds 150 and 151, inside the last segment of epoch 3 of the first stretch: that segment is flagged loud.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from exmarg import strain

SOURCE = Path(__file__).parent.parent / 'scripts' / 'real_data' / 'condition.py'
spec = importlib.util.spec_from_file_location('exmarg_condition', SOURCE)
cond = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cond)

DQ = ['DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3', 'BURST_CAT1', 'BURST_CAT2', 'BURST_CAT3']
INJ = ['NO_CBC_HW_INJ', 'NO_BURST_HW_INJ', 'NO_DETCHAR_HW_INJ', 'NO_CW_HW_INJ', 'NO_STOCH_HW_INJ']
G, FS, DUR = 1000000000, 128, 300
VETO_H, BURST_H, GAP_L, INJ_L = 20, 450, (420, 430), 560
T, N, PAD = 4.0, 7, 4.0
LOUD_H = 150                                                   # seconds 150 and 151 of H1, amplitude times 30
STRETCHES = ((21, 420), (430, 560))
EVENT = G + 21 + PAD + 3 * (N + 1) * T + 2 * T + 1.5           # 1.5 s into segment 2 of epoch 3 of the first stretch
ARGS = ['--T', '4', '--N', '7', '--min-blocks', '3', '--pad', '4', '--fmin', '20', '--fmax', '40', '--band', '16', '48',
        '--roll', '4', '8', '--method', 'median', '--seed', '7', '--notches', '29.5:30.5']


def write_file(path, det, gps, rng):
    bit = {n: 1 << i for i, n in enumerate(DQ)}
    ibit = {n: 1 << i for i, n in enumerate(INJ)}
    x = rng.standard_normal(DUR * FS) * 1e-21
    dq = np.full(DUR, sum(bit.values()))
    inj = np.full(DUR, sum(v for k, v in ibit.items() if k != 'NO_CW_HW_INJ'))
    sec = np.arange(gps, gps + DUR) - G
    if det == 'H1':
        x.reshape(DUR, FS)[(sec >= LOUD_H) & (sec < LOUD_H + 2)] *= 30.0
        dq[sec == VETO_H] &= ~bit['CBC_CAT2']
        dq[sec == BURST_H] &= ~bit['BURST_CAT2']
    else:
        gap = (sec >= GAP_L[0]) & (sec < GAP_L[1])
        dq[gap] = 0
        x.reshape(DUR, FS)[gap] = np.nan
        inj[sec == INJ_L] &= ~ibit['NO_STOCH_HW_INJ']
    with h5py.File(path, 'w') as h:
        h['meta/GPSstart'] = gps
        s = h.create_dataset('strain/Strain', data=x)
        s.attrs['Xstart'], s.attrs['Xspacing'] = gps, 1.0 / FS
        h['quality/simple/DQmask'] = dq.astype(np.uint32)
        h['quality/simple/DQShortnames'] = np.array(DQ, dtype='S')
        h['quality/injections/Injmask'] = inj.astype(np.uint32)
        h['quality/injections/InjShortnames'] = np.array(INJ, dtype='S')


@pytest.fixture(scope='module')
def folder(tmp_path_factory):
    root = tmp_path_factory.mktemp('gwosc')
    rng = np.random.default_rng(3)
    for det in ('H1', 'L1'):
        (root / det).mkdir()
        for gps in (G, G + DUR):
            write_file(root / det / ('%s-%s_GWOSC_TEST-%d-%d.hdf5' % (det[0], det, gps, DUR)), det, gps, rng)
    return root


def run(folder, out, *extra):
    out.mkdir(exist_ok=True)
    with open(out / 'events.json', 'w') as fh:                    # the cached catalogue: no network
        json.dump(dict(events=[['GW_TEST', EVENT]]), fh)
    assert cond.main([str(folder), str(out)] + ARGS + list(extra)) == 0
    return {p.name: strain.read_epoch(p) for p in sorted(out.glob('epoch_*.npz'))}


@pytest.fixture(scope='module')
def blocks(folder, tmp_path_factory):
    out = tmp_path_factory.mktemp('blocks')
    return out, run(folder, out, '--workers', '1', '--block', '3')


def good_seconds():
    good = np.ones(2 * DUR, bool)
    good[[VETO_H, INJ_L]] = False
    good[GAP_L[0]:GAP_L[1]] = False
    return good


def test_stretches_joined_across_the_file_boundary(folder):
    tl = cond.timeline(str(folder))
    assert tl['gps0'] == G and tl['fs'] == FS and len(tl['slots']) == 2
    assert np.array_equal(tl['good'], good_seconds())
    assert tl['good_H1'][GAP_L[0]] and not tl['good_L1'][GAP_L[0]] and not tl['good_H1'][VETO_H]
    stretches, jobs = cond.plan(tl['good'], G, FS, T, N, PAD, block=0)
    assert [(s[0] - G, s[0] - G + s[1]) for s in stretches] == list(STRETCHES)
    assert [s[2] for s in stretches] == [12, 3]                    # cut at 300 s: 8 + 3 in the first
    assert len(cond.plan(tl['good'], G, FS, T, N, PAD, block=3)[1]) == 4 + 1


def test_segments_and_bookkeeping(blocks):
    _, eps = blocks
    good = good_seconds()
    K = [int((s1 - s0 - 2 * PAD) // ((N + 1) * T)) for s0, s1 in STRETCHES]
    starts = [G + s0 + PAD + e * (N + 1) * T for (s0, s1), k in zip(STRETCHES, K) for e in range(k)]
    assert sorted(ep['gps_epoch'] for ep in eps.values()) == starts
    assert any(ep['gps_epoch'] < G + DUR < ep['gps_epoch'] + (N + 1) * T for ep in eps.values())
    for name, ep in eps.items():
        assert name == 'epoch_%d.npz' % ep['gps_epoch']
        assert np.array_equal(ep['gps'], ep['gps_epoch'] + T * np.arange(N + 1))
        for t in ep['gps']:                                       # every second of every segment is good
            assert good[int(t - G):int(t - G + T)].all()
        burst = (ep['gps'] <= G + BURST_H) & (G + BURST_H < ep['gps'] + T)
        assert np.array_equal(ep['dq_H'], np.where(burst, 127 - 32, 127)) and np.all(ep['dq_L'] == 127)
        assert np.all(ep['inj_H'] == 23) and np.all(ep['inj_L'] == 23) and list(ep['dq_names']) == DQ
        flagged = (ep['gps'] - 2.0 <= EVENT) & (EVENT < ep['gps'] + T + 2.0)
        assert np.array_equal(ep['event'], flagged)
        assert ep['method'] == 'median' and ep['seed'] == 7 and ep['N'] == N and ep['d_H'].shape == (N + 1, 81)
        assert np.array_equal(ep['notches'], [[29.5, 30.5]]) and ep['notched'].sum() > 0
        loud = (ep['gps'] <= G + LOUD_H) & (G + LOUD_H + 2 <= ep['gps'] + T)
        assert np.array_equal(ep['loud_H'], loud) and not ep['loud_L'].any() and ep['usable']
        assert ep['n_used_H'] == N + 1 - loud.sum() and 0.5 < ep['u_ratio_60'] < 2.0
    assert sum(int(ep['event'].sum()) for ep in eps.values()) == 2
    assert sum(int(ep['loud_H'].sum()) for ep in eps.values()) == 1


def test_blocks_give_the_whole_stretch_and_resume(folder, blocks, tmp_path):
    out, eps = blocks
    whole = run(folder, tmp_path / 'whole', '--workers', '1', '--block', '0')
    assert list(whole) == list(eps)
    for name in eps:
        for key in ('d_H', 'd_L', 'sig2', 'sig2_ref_H', 'sig2_ref_L', 'monitor_H', 'gps_reference', 'loud_H',
                    'correction_H', 'n_reference_H', 'u_ratio_60'):
            assert np.array_equal(whole[name][key], eps[name][key], equal_nan=True), (name, key)
    mtime = {p.name: p.stat().st_mtime_ns for p in out.glob('epoch_*.npz')}
    again = run(folder, out, '--workers', '1', '--block', '3')
    assert list(again) == list(eps) and {p.name: p.stat().st_mtime_ns for p in out.glob('epoch_*.npz')} == mtime
    rows = [r.split() for r in open(out / 'manifest.txt') if not r.startswith('#')]
    assert [r[0] for r in rows] == list(eps) and all(r[2] == str(N + 1) for r in rows)
    assert sum(int(r[5]) for r in rows) == 2 and {r[4] for r in rows} == {'127'}
    assert sum(int(r[6]) for r in rows) == 1 and {r[7] for r in rows} == {'0'} and {r[10] for r in rows} == {'1'}
    assert [float(r[11]) for r in rows] == [round(eps[r[0]]['u_ratio_60'], 4) for r in rows]
    assert [r.split()[:2] for r in open(out / 'stretches.txt') if not r.startswith('#')] == [
        [str(G + a), str(b - a)] for a, b in STRETCHES]


def test_worker_pool_from_the_command_line(folder, blocks, tmp_path):
    """the script itself with two processes and blocks of two epochs: the same files"""
    _, eps = blocks
    out = tmp_path / 'pool'
    out.mkdir()
    with open(out / 'events.json', 'w') as fh:
        json.dump(dict(events=[['GW_TEST', EVENT]]), fh)
    env = dict(os.environ, OMP_NUM_THREADS='1')
    r = subprocess.run([sys.executable, str(SOURCE), str(folder), str(out), '--workers', '2', '--block', '2'] + ARGS,
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    pool = {p.name: strain.read_epoch(p) for p in sorted(out.glob('epoch_*.npz'))}
    assert list(pool) == list(eps)
    assert all(np.array_equal(pool[n]['d_L'], eps[n]['d_L']) for n in eps)


# ---------------------------------------------------------------------------------------------------------------------
# resuming into a folder: the settings that determine the output are stored in every epoch and must not change


def with_option(args, name, value):
    """args with the value after the option `name` replaced"""
    i = args.index(name)
    return args[:i + 1] + [value] + args[i + 2:]


def settings_of(path):
    with np.load(path) as z:
        return (str(z['settings_hash']) if 'settings_hash' in z.files else None), int(z['seed'])


def strip_hash(path):
    """the epoch file as conditioned before the settings hash existed"""
    with np.load(path) as z:
        keep = {k: z[k] for k in z.files if k not in ('settings_hash', 'settings')}
    np.savez(path, **keep)


def test_resume_refuses_other_settings(folder, tmp_path):
    """a second run into the same folder with another seed stops, names the seed, and leaves every file as it was;
    --overwrite recomputes the epochs with the new settings (one hash in all of them, the new seed); the same settings
    again skip every file"""
    out = tmp_path / 'resume'
    first = run(folder, out, '--workers', '1')
    hashes = {settings_of(p) for p in out.glob('epoch_*.npz')}
    assert len({h for h, _ in hashes}) == 1 and {s for _, s in hashes} == {7} and None not in {h for h, _ in hashes}
    before = {p.name: p.stat().st_mtime_ns for p in out.glob('epoch_*.npz')}
    args = with_option(ARGS, '--seed', '8')
    with pytest.raises(SystemExit, match='seed'):
        cond.main([str(folder), str(out)] + args + ['--workers', '1'])
    assert {p.name: p.stat().st_mtime_ns for p in out.glob('epoch_*.npz')} == before
    assert list(run(folder, out, '--workers', '1')) == list(first)
    assert {p.name: p.stat().st_mtime_ns for p in out.glob('epoch_*.npz')} == before
    assert cond.main([str(folder), str(out)] + args + ['--workers', '1', '--overwrite']) == 0
    after = {settings_of(p) for p in out.glob('epoch_*.npz')}
    assert {s for _, s in after} == {8} and len({h for h, _ in after}) == 1 and after != hashes
    assert len(list(out.glob('epoch_*.npz'))) == len(first)


def test_notch_change_is_a_different_setting(folder, tmp_path):
    out = tmp_path / 'notch'
    run(folder, out, '--workers', '1')
    args = with_option(ARGS, '--notches', '33.0:34.0')
    with pytest.raises(SystemExit, match='notches'):
        cond.main([str(folder), str(out)] + args + ['--workers', '1'])


def test_arrays_resume_refuses_other_settings_and_counts_written(tmp_path, capsys):
    """.npz input (the case that mixed a short run with seed 7 and a longer one with seed 8 in one folder): refused
    without --overwrite; the final message counts the epochs the run wrote, not those in the folder"""
    for i, name in enumerate(('H1', 'L1')):                      # the long input is the short one and 200 s more
        short = np.random.default_rng(1 + i).standard_normal(200 * FS) * 1e-21
        extra = np.random.default_rng(10 + i).standard_normal(200 * FS) * 1e-21
        for tag, x in (('short', short), ('long', np.concatenate([short, extra]))):
            np.savez(tmp_path / ('%s_%s.npz' % (name, tag)), strain=x, sample_rate=FS, gps_start=G,
                     dq=np.full(x.size // FS, 3))
    out = tmp_path / 'out'
    common = [a for a in ARGS if a not in ('--notches', '29.5:30.5')]
    args = lambda tag, seed: [str(tmp_path / ('H1_%s.npz' % tag)), str(tmp_path / ('L1_%s.npz' % tag)),
                              str(out)] + with_option(common, '--seed', seed)
    assert cond.main(args('short', '7')) == 0
    n_first = len(list(out.glob('epoch_*.npz')))
    capsys.readouterr()
    with pytest.raises(SystemExit, match='seed'):
        cond.main(args('long', '8'))
    assert len(list(out.glob('epoch_*.npz'))) == n_first
    assert cond.main(args('long', '7')) == 0                      # same settings: resumes, the first epochs kept
    last = capsys.readouterr().out.strip().splitlines()[-1]
    n_all = len(list(out.glob('epoch_*.npz')))
    assert n_all > n_first
    assert '%d epochs in' % n_all in last and '%d written by this run' % (n_all - n_first) in last, last
    assert cond.main(args('long', '7')) == 0
    assert '0 written by this run' in capsys.readouterr().out.strip().splitlines()[-1]


def test_manifest_needs_one_settings_hash(folder, tmp_path):
    """write_manifest: a folder in which no epoch has the hash is legacy and accepted; a mixture of hashed and
    unhashed epochs, or two hashes, is an error naming the files"""
    out = tmp_path / 'mix'
    run(folder, out, '--workers', '1')
    paths = sorted(out.glob('epoch_*.npz'))
    header = ['test']
    n = cond.write_manifest(str(out), header)[0]
    assert n == len(paths) and settings_of(paths[0])[0] in open(out / 'manifest.txt').read()
    strip_hash(paths[0])
    with pytest.raises(SystemExit, match=paths[0].name):
        cond.write_manifest(str(out), header)
    for p in paths:
        strip_hash(p)
    assert cond.write_manifest(str(out), header)[0] == len(paths)           # legacy folder
    with np.load(paths[1]) as z:
        d = {k: z[k] for k in z.files}
    d['settings_hash'] = np.array('0' * 64)
    np.savez(paths[1], **d)
    np.savez(paths[2], **dict(d, settings_hash=np.array('1' * 64)))
    with pytest.raises(SystemExit, match='settings'):
        cond.write_manifest(str(out), header)


def test_segment_and_outside_the_array():
    """a segment outside the data-quality seconds (np.bitwise_and.reduce of an empty slice is -1) stops the run and
    names the segment"""
    bits = np.full(50, 3, np.int64)
    assert list(cond._segment_and(bits, [10.0, 12.0], 10.0, 4.0)) == [3, 3]
    with pytest.raises(ValueError, match='GPS 60'):
        cond._segment_and(bits, [10.0, 60.0], 10.0, 4.0)
