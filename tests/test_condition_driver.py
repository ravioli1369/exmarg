"""scripts/real_data/condition.py on a synthetic GWOSC folder (no network, no real data).

Two detectors, two consecutive files of 300 s at 128 Hz (white noise).  H1 fails CBC_CAT2 at second 20 (finite strain,
a veto second) and BURST_CAT2 at second 450 (not required: only recorded); L1 has no data at seconds 420 to 430 (NaN,
DATA unset) and fails NO_STOCH_HW_INJ at second 560.  The stretches are then [21, 420), which crosses the file
boundary at 300, and [430, 560); [0, 20) and [561, 600) are shorter than two epochs.  Epochs of N + 1 = 6 segments of
4 s with 4 s of pad: 16 and 5 epochs, against 11 + 4 + 5 if the stretch were cut at the file boundary.
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
T, N, PAD = 4.0, 5, 4.0
STRETCHES = ((21, 420), (430, 560))
EVENT = G + 21 + PAD + 3 * (N + 1) * T + 2 * T + 1.5           # 1.5 s into segment 2 of epoch 3 of the first stretch
ARGS = ['--T', '4', '--N', '5', '--pad', '4', '--fmin', '20', '--fmax', '40', '--band', '16', '48', '--roll', '4', '8',
        '--method', 'median', '--seed', '7', '--notches', '29.5:30.5']


def write_file(path, det, gps, rng):
    bit = {n: 1 << i for i, n in enumerate(DQ)}
    ibit = {n: 1 << i for i, n in enumerate(INJ)}
    x = rng.standard_normal(DUR * FS) * 1e-21
    dq = np.full(DUR, sum(bit.values()))
    inj = np.full(DUR, sum(v for k, v in ibit.items() if k != 'NO_CW_HW_INJ'))
    sec = np.arange(gps, gps + DUR) - G
    if det == 'H1':
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
    assert [s[2] for s in stretches] == [16, 5]                    # cut at 300 s: 11 + 4 in the first
    assert len(cond.plan(tl['good'], G, FS, T, N, PAD, block=3)[1]) == 6 + 2


def test_segments_and_bookkeeping(blocks):
    _, eps = blocks
    good = good_seconds()
    starts = [G + s0 + PAD + e * (N + 1) * T for s0, s1 in STRETCHES for e in range((s1 - s0 - 8) // 24)]
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
    assert sum(int(ep['event'].sum()) for ep in eps.values()) == 2


def test_blocks_give_the_whole_stretch_and_resume(folder, blocks, tmp_path):
    out, eps = blocks
    whole = run(folder, tmp_path / 'whole', '--workers', '1', '--block', '0')
    assert list(whole) == list(eps)
    for name in eps:
        for key in ('d_H', 'd_L', 'sig2', 'sig2_ref_H', 'sig2_ref_L', 'monitor_H', 'gps_reference'):
            assert np.array_equal(whole[name][key], eps[name][key], equal_nan=True), (name, key)
    mtime = {p.name: p.stat().st_mtime_ns for p in out.glob('epoch_*.npz')}
    again = run(folder, out, '--workers', '1', '--block', '3')
    assert list(again) == list(eps) and {p.name: p.stat().st_mtime_ns for p in out.glob('epoch_*.npz')} == mtime
    rows = [r.split() for r in open(out / 'manifest.txt') if not r.startswith('#')]
    assert [r[0] for r in rows] == list(eps) and all(r[2] == str(N + 1) for r in rows)
    assert sum(int(r[5]) for r in rows) == 2 and {r[4] for r in rows} == {'127'}
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
