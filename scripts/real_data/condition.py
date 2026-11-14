"""Condition the strain of H1 and L1 into epoch files of segments (exmarg.strain.condition, docs/real_data.md).

    python scripts/real_data/condition.py H1.npz L1.npz out_dir [--N 32] [--method mean|median]
                                          [--notches 59.5:60.5,119.5:120.5] [--events gps.txt] [--seed 0]
    -> out_dir/epoch_<gps>.npz, one per epoch (read_segments reads them; exmarg.strain.read_epoch with the bookkeeping)

Each input holds one detector: an .npz with strain (n,), sample_rate, gps_start and optionally dq (n_second,) int
bits at 1 Hz, or an .npy strain array with --fs and --gps.  The coincident stretches are the runs of whole seconds in
which both detectors have finite strain and every bit of --require set in dq; each is conditioned on its own (a
stretch shorter than two epochs, 2 (N + 1) T + 2 pad seconds, gives no epoch; one epoch with --reference epoch).
--events is a text file whose first column is the GPS time of catalogued events.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exmarg import strain


# ---------------------------------------------------------------------------------------------------------------------
# LOADER: the one function to replace by the GWOSC reader (scripts/real_data/gwosc.py, read_strain, which returns
# these keys; import h5py lazily there).  For now: .npz with strain, sample_rate, gps_start, [dq]; or .npy + fs, gps.
def load_strain(path, fs=None, gps=None):
    """one detector's strain and data-quality mask: dict with strain (n,) float64 (NaN where there are no data),
    sample_rate (Hz), gps_start (GPS of the first sample), dq (n / sample_rate,) int bits at 1 Hz or None"""
    if str(path).endswith('.npy'):
        if fs is None or gps is None:
            raise ValueError('an .npy strain needs --fs and --gps')
        return dict(strain=np.load(path).astype(float), sample_rate=float(fs), gps_start=float(gps), dq=None)
    with np.load(path) as z:
        return dict(strain=np.asarray(z['strain'], float), sample_rate=float(z['sample_rate']),
                    gps_start=float(z['gps_start']), dq=np.asarray(z['dq'], np.int64) if 'dq' in z.files else None)
# ---------------------------------------------------------------------------------------------------------------------


def good_seconds(x, require):
    """(n_second,) bool: the seconds with finite strain and every bit of `require` set in dq (when there is a dq)"""
    fs = int(round(x['sample_rate']))
    n = x['strain'].size // fs
    good = np.isfinite(x['strain'][: n * fs].reshape(n, fs)).all(axis=1)
    if x['dq'] is not None:
        good &= (x['dq'][:n] & require) == require
    return good


def runs(mask):
    """(start, stop) index pairs of the runs of True"""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def notch_list(text):
    return [tuple(float(v) for v in item.split(':')) for item in text.split(',') if item] if text else []


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('H1')
    p.add_argument('L1')
    p.add_argument('out_dir')
    p.add_argument('--fs', type=float, help='sample rate of .npy inputs (Hz)')
    p.add_argument('--gps', type=float, nargs=2, metavar=('GPS_H1', 'GPS_L1'), help='first sample of .npy inputs')
    p.add_argument('--require', type=int, default=3, help='dq bits that must be set (default 3: DATA and CBC_CAT1 in '
                                                          "GWOSC's order)")
    p.add_argument('--T', type=float, default=8.0)
    p.add_argument('--fmin', type=float, default=20.0)
    p.add_argument('--fmax', type=float, default=512.0)
    p.add_argument('--N', type=int, default=32, help='an epoch is N + 1 segments')
    p.add_argument('--method', default='mean', choices=('mean', 'median'))
    p.add_argument('--reference', default='neighbours', choices=('neighbours', 'epoch'),
                   help="curve an epoch is handed over on: the neighbouring epochs, or (mean only) its own blocks, "
                        "which also keeps stretches of one epoch")
    p.add_argument('--notches', default='', help='lo:hi,lo:hi,... (Hz)')
    p.add_argument('--events', help='text file, first column the GPS of catalogued events')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pad', type=float, default=None, help='seconds on each side of an epoch (default T)')
    p.add_argument('--band', type=float, nargs=2, default=(16.0, 600.0))
    p.add_argument('--roll', type=float, nargs=2, default=(4.0, 100.0))
    p.add_argument('--smooth', type=int, default=17)
    p.add_argument('--unequal', action='store_true', help='skip step 5: keep sig2_H and sig2_L')
    a = p.parse_args(argv)

    gps = a.gps or (None, None)
    x = {I: load_strain(path, a.fs, g) for I, path, g in (('H', a.H1, gps[0]), ('L', a.L1, gps[1]))}
    fs = x['H']['sample_rate']
    if x['L']['sample_rate'] != fs:
        raise SystemExit('the two detectors have different sample rates')
    t0 = max(x[I]['gps_start'] for I in 'HL')
    t1 = min(x[I]['gps_start'] + x[I]['strain'].size / fs for I in 'HL')
    if any(abs((t0 - x[I]['gps_start']) - round(t0 - x[I]['gps_start'])) > 1e-9 for I in 'HL'):
        raise SystemExit('the two inputs are not aligned on whole seconds')
    n0 = {I: int(round((t0 - x[I]['gps_start']) * fs)) for I in 'HL'}
    n_sec = int(np.floor(t1 - t0))
    for I in 'HL':
        x[I] = dict(x[I], strain=x[I]['strain'][n0[I]: n0[I] + n_sec * int(fs)],
                    dq=None if x[I]['dq'] is None else x[I]['dq'][int(round(n0[I] / fs)):][:n_sec])
    good = good_seconds(x['H'], a.require) & good_seconds(x['L'], a.require)
    events = np.loadtxt(a.events, ndmin=2)[:, 0] if a.events else ()
    os.makedirs(a.out_dir, exist_ok=True)
    total = 0
    for s0, s1 in runs(good):
        sl = slice(s0 * int(fs), s1 * int(fs))
        eps = strain.condition(x['H']['strain'][sl], x['L']['strain'][sl], fs, t0 + s0, T=a.T, fmin=a.fmin,
                               fmax=a.fmax, N=a.N, notches=notch_list(a.notches), method=a.method,
                               reference=a.reference, equal_noise=not a.unequal, seed=a.seed, pad=a.pad,
                               band=tuple(a.band), roll=tuple(a.roll), smooth=a.smooth,
                               dq_H=None if x['H']['dq'] is None else x['H']['dq'][s0:s1],
                               dq_L=None if x['L']['dq'] is None else x['L']['dq'][s0:s1], events=events)
        for ep in eps:
            strain.write_epoch(os.path.join(a.out_dir, 'epoch_%d.npz' % round(ep['gps_epoch'])), ep)
        used = len(eps) * (a.N + 1) * a.T
        total += len(eps)
        print('stretch GPS %d + %d s: %d epochs, %d s of segments, %d events flagged'
              % (t0 + s0, s1 - s0, len(eps), used, sum(int(ep['event'].sum()) for ep in eps)), flush=True)
    print('%d epochs written to %s (%d coincident good seconds of %d)' % (total, a.out_dir, good.sum(), good.size))
    return 0


if __name__ == '__main__':
    sys.exit(main())
