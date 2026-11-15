"""GWOSC reader and good-second selection, on a synthetic stand-in for a GWOSC HDF5 file (no network, no real data)"""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

SOURCE = Path(__file__).parent.parent / 'scripts' / 'real_data' / 'gwosc.py'     # by path: its name would hide gwosc
spec = importlib.util.spec_from_file_location('exmarg_gwosc', SOURCE)
gwosc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gwosc)

DQ = ['DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3', 'BURST_CAT1', 'BURST_CAT2', 'BURST_CAT3']
INJ = ['NO_CBC_HW_INJ', 'NO_BURST_HW_INJ', 'NO_DETCHAR_HW_INJ', 'NO_CW_HW_INJ', 'NO_STOCH_HW_INJ']
GPS, FS, SECONDS = 1266966528, 64, 16
GAP = (4, 6)                      # seconds 4 and 5: no data, NaN strain, DATA bit not set
CAT2_FAIL, INJ_FAIL = 9, 12       # second 9 finite but CBC_CAT2 not set; second 12 has the STOCH injection bit not set


def write_synthetic(path, dq_names=DQ, inj_names=INJ):
    """Same layout as a GWOSC O3 file (meta/, strain/Strain with Xstart and Xspacing, quality/simple/DQmask with
    DQShortnames, quality/injections/Injmask with InjShortnames) but 16 s at 64 Hz instead of 4096 s at 4096 Hz.
    The CW injection bit is never set, as in the real O3 files.  Returns the strain written"""
    bit = {n: 1 << i for i, n in enumerate(dq_names)}
    ibit = {n: 1 << i for i, n in enumerate(inj_names)}
    x = np.random.default_rng(0).standard_normal(FS * SECONDS) * 1e-20
    x[GAP[0] * FS:GAP[1] * FS] = np.nan
    dq = np.full(SECONDS, sum(bit.values()))
    dq[GAP[0]:GAP[1]] = 0
    dq[CAT2_FAIL] &= ~bit['CBC_CAT2']
    inj = np.full(SECONDS, sum(v for k, v in ibit.items() if k != 'NO_CW_HW_INJ'))
    inj[INJ_FAIL] &= ~ibit['NO_STOCH_HW_INJ']
    with h5py.File(path, 'w') as h:
        h['meta/GPSstart'] = GPS
        h['meta/Detector'] = 'H1'
        s = h.create_dataset('strain/Strain', data=x)
        s.attrs['Xstart'], s.attrs['Xspacing'] = GPS, 1.0 / FS
        h['quality/simple/DQmask'] = dq.astype(np.uint32)
        h['quality/simple/DQmask'].attrs['Xspacing'] = 1.0
        h['quality/simple/DQShortnames'] = np.array(dq_names, dtype='S')
        h['quality/injections/Injmask'] = inj.astype(np.uint32)
        h['quality/injections/Injmask'].attrs['Xspacing'] = 1.0
        h['quality/injections/InjShortnames'] = np.array(inj_names, dtype='S')
    return x


@pytest.fixture
def synthetic(tmp_path):
    path = tmp_path / ('H-H1_GWOSC_O3b_4KHZ_R1-%d-4096.hdf5' % GPS)
    return path, write_synthetic(path)


def test_read_strain(synthetic):
    path, x = synthetic
    r = gwosc.read_strain(path)
    assert r['strain'].dtype == np.float64 and r['strain'].shape == (FS * SECONDS,)
    assert np.array_equal(r['strain'], x, equal_nan=True)
    assert r['gps_start'] == GPS and r['sample_rate'] == FS
    assert r['dq'].shape == r['inj'].shape == (SECONDS,) and r['dq'].dtype == r['inj'].dtype == np.int64
    assert r['dq_names'] == DQ and r['inj_names'] == INJ
    assert np.isnan(r['strain'][GAP[0] * FS:GAP[1] * FS]).all() and np.isfinite(r['strain'][:GAP[0] * FS]).all()
    assert np.array_equal(r['dq'][GAP[0]:GAP[1]], [0, 0]) and r['dq'][0] == 127
    assert r['dq'][CAT2_FAIL] == 127 - 4 and r['inj'][0] == 31 - 8


def test_good_seconds_default(synthetic):
    r = gwosc.read_strain(synthetic[0])
    good = gwosc.good_seconds(r['dq'], r['inj'], r['dq_names'], r['inj_names'])
    expect = np.ones(SECONDS, dtype=bool)
    expect[GAP[0]:GAP[1]] = False        # no data
    expect[CAT2_FAIL] = False            # second failing CBC_CAT2 only, strain finite
    expect[INJ_FAIL] = False             # injection bit
    assert good.dtype == bool and np.array_equal(good, expect)
    assert np.isfinite(r['strain'].reshape(SECONDS, FS)[CAT2_FAIL]).all()
    nan_seconds = ~np.isfinite(r['strain'].reshape(SECONDS, FS)).all(axis=1)
    assert not (good & nan_seconds).any()


def test_good_seconds_choices(synthetic):
    r = gwosc.read_strain(synthetic[0])
    args = (r['dq'], r['inj'], r['dq_names'], r['inj_names'])
    no_cat2 = gwosc.good_seconds(*args, require_dq=('DATA', 'CBC_CAT1'))
    assert no_cat2[CAT2_FAIL] and not no_cat2[GAP[0]] and not no_cat2[INJ_FAIL]
    assert not gwosc.good_seconds(*args, require_inj=INJ).any()          # NO_CW_HW_INJ never passes
    assert gwosc.good_seconds(*args, require_inj=('NO_CBC_HW_INJ',))[INJ_FAIL]
    with pytest.raises(KeyError):
        gwosc.good_seconds(*args, require_dq=('NOT_A_BIT',))


def test_bit_order_is_read_from_the_file(tmp_path):
    dq_names = DQ[::-1]
    inj_names = [INJ[i] for i in (4, 3, 2, 1, 0)]
    path = tmp_path / 'shuffled.hdf5'
    write_synthetic(path, dq_names, inj_names)
    r = gwosc.read_strain(path)
    assert r['dq_names'] == dq_names and r['inj_names'] == inj_names
    good = gwosc.good_seconds(r['dq'], r['inj'], r['dq_names'], r['inj_names'])
    expect = np.ones(SECONDS, dtype=bool)
    expect[[*range(*GAP), CAT2_FAIL, INJ_FAIL]] = False
    assert np.array_equal(good, expect)


def test_slot_and_scan(tmp_path):
    name = 'H-H1_GWOSC_O3b_4KHZ_R1-%d-4096.hdf5' % GPS
    assert gwosc._slot(name) == GPS
    for det in ('H1', 'L1'):
        (tmp_path / det).mkdir()
        (tmp_path / det / name.replace('H1', det).replace('H-', det[0] + '-')).touch()
    (tmp_path / 'H1' / (name + '.part')).touch()
    found = gwosc._scan(str(tmp_path))
    assert list(found) == [GPS] and set(found[GPS]) == {'H1', 'L1'}


class FakeLocate:
    """gwosc.locate for a run with files in some slots only: get_urls raises ValueError, as the library does, unless
    the first and the last slot of the range both have a file"""

    def __init__(self, have):
        self.have, self.calls = have, 0

    def get_urls(self, det, a, b, dataset=None, sample_rate=None, format=None):
        self.calls += 1
        slots = range(a // gwosc.SLOT * gwosc.SLOT, b + 1, gwosc.SLOT)
        if slots[0] not in self.have[det] or slots[-1] not in self.have[det]:
            raise ValueError('Cannot find a GWOSC dataset for %s covering [%d, %d)' % (det, a, b + 1))
        return ['https://x/%s-%s_GWOSC_O3a_4KHZ_R1-%d-%d.hdf5' % (det[0], det, t, gwosc.SLOT)
                for t in slots if t in self.have[det]]


def test_list_pairs_with_empty_slots_at_the_edges(monkeypatch):
    """a range whose first or last slot has no file in one detector gives the pairs of the slots that have both"""
    t0 = 302697 * gwosc.SLOT
    slot = lambda *i: {t0 + k * gwosc.SLOT for k in i}
    fake = FakeLocate({'H1': slot(1, 2, 3, 5), 'L1': slot(0, 1, 3, 4, 5, 6)})    # H1 misses both edges, L1 the last
    monkeypatch.setattr(gwosc, '_locate', lambda: fake)
    pairs = gwosc.list_pairs(t0, t0 + 8 * gwosc.SLOT, run='O3a')
    assert [p[0] for p in pairs] == sorted(slot(1, 3, 5))
    assert all('H-H1' in p[1] and 'L-L1' in p[2] for p in pairs)
    # every edge present: one request per detector, the same answer as before the fix
    full = FakeLocate({'H1': slot(0, 1, 2), 'L1': slot(0, 1, 2)})
    monkeypatch.setattr(gwosc, '_locate', lambda: full)
    assert [p[0] for p in gwosc.list_pairs(t0, t0 + 3 * gwosc.SLOT)] == sorted(slot(0, 1, 2)) and full.calls == 2
    # no file at all in one detector: no pair, and no exception
    none = FakeLocate({'H1': set(), 'L1': slot(0, 1)})
    monkeypatch.setattr(gwosc, '_locate', lambda: none)
    assert gwosc.list_pairs(t0, t0 + 2 * gwosc.SLOT) == []
