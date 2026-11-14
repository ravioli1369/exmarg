"""Condition the strain of H1 and L1 into epoch files of segments (exmarg.strain.condition, docs/real_data.md).

    python scripts/real_data/condition.py <gwosc_dir> <out_dir> [--notches-file F] [--workers 16] [--block 24]
                                          [--seed S] [--N 32] [--method median|mean] [--event-margin 2]
    python scripts/real_data/condition.py H1.npz L1.npz <out_dir> [the same options] [--require 3] [--events gps.txt]
    -> out_dir/epoch_<gps>.npz, one per epoch (read_segments reads them; exmarg.strain.read_epoch with the
       bookkeeping), out_dir/manifest.txt, out_dir/stretches.txt, and for GWOSC input out_dir/events.json

GWOSC input: a folder with H1/ and L1/ holding files named ...-<gps>-<seconds>.hdf5 (gwosc.py fetch).  The slots with
both files are read in time order; a second is good when both detectors pass gwosc.good_seconds (by default DATA,
CBC_CAT1/2/3 and no CBC, burst, detchar or stochastic hardware injection; --require-dq, --require-inj) and have finite
strain.  The good seconds of consecutive slots join into one stretch.  A stretch shorter than two epochs,
2 (N + 1) T + 2 pad seconds, gives no epoch (one epoch with --reference epoch).  strain.condition holds a whole
stretch and the transforms of all its epochs (about 95 MB per 1000 s at 4096 Hz), so a stretch of more than --block
epochs is conditioned in blocks: a block owns up to --block consecutive epochs and is given one more epoch on each
side, which makes its epochs identical to those of the whole stretch (an epoch depends on its own data, on the curves
of its two neighbours and on a noise seed set by --seed and its GPS time).  Blocks run in --workers processes.  An
epoch whose file exists is not recomputed: running the same command again resumes an interrupted run.

Catalogued events (gwosc.events(): GWTC-3 confident and marginal, fetched once and cached in out_dir/events.json; or
--events, a text file whose first column is the GPS) flag the segments within --event-margin seconds of them.

Each epoch file holds the dict of strain.condition (dq_H, dq_L: the DQ bits set over all of a segment) and
    dq_names, inj_names   bit names of the masks           inj_H, inj_L  injection bits set over all of a segment
    stretch               GPS start and end of its stretch  notch_file    the file of the notches ('' if none)
manifest.txt: one line per epoch: file, GPS start, number of segments, the DQ bits set over the whole epoch in H1 and
L1, the number of segments flagged with an event, the loud segments and the unflagged blocks of each detector
(strain.condition flags loud blocks and leaves them out of every estimate; --no-flag, --flag-p, --min-blocks),
whether the epoch is usable, and u_H1 / u_L1 before step 5 for an equal-mass template of 60 Msun (also u_ratio_60 in
the epoch file).  stretches.txt: GPS start, seconds and epochs of each stretch.

On a host whose memory is fragmented by other jobs, set NUMPY_MADVISE_HUGEPAGE=0: numpy asks for huge pages for its
large arrays, and their page faults then stall in memory compaction (five times slower on a shared machine).

.npz/.npy input: one detector per file, an .npz with strain (n,), sample_rate, gps_start and optionally dq
(n_second,) int bits at 1 Hz, or an .npy strain array with --fs and --gps.  Good seconds have finite strain and every
bit of --require set in dq; the stretches are conditioned whole, one after the other.
"""
import argparse
import glob
import importlib.util
import json
import multiprocessing as mp
import os
import re
import sys
import time

import h5py
import numpy as np

HERE =os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exmarg import strain


def _load_gwosc():
    """scripts/real_data/gwosc.py, loaded by path (its name would hide the gwosc package)"""
    if 'exmarg_gwosc' not in sys.modules:
        spec = importlib.util.spec_from_file_location('exmarg_gwosc', os.path.join(HERE, 'gwosc.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules['exmarg_gwosc'] = module
        spec.loader.exec_module(module)
    return sys.modules['exmarg_gwosc']


gw = _load_gwosc()
DETS = ('H1', 'L1')


# ---------------------------------------------------------------------------------------------------------------------
# .npz/.npy input
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


def good_seconds(x, require):
    """(n_second,) bool: the seconds with finite strain and every bit of `require` set in dq (when there is a dq)"""
    fs = int(round(x['sample_rate']))
    n = x['strain'].size // fs
    good = np.isfinite(x['strain'][: n * fs].reshape(n, fs)).all(axis=1)
    if x['dq'] is not None:
        good &= (x['dq'][:n] & require) == require
    return good


# ---------------------------------------------------------------------------------------------------------------------
# GWOSC input
def find_files(gwosc_dir):
    """{gps: {detector: path}} of the files <gwosc_dir>/<detector>/...-<gps>-<seconds>.hdf5"""
    found = {}
    for det in DETS:
        for path in sorted(glob.glob(os.path.join(gwosc_dir, det, '*.hdf5'))):
            m = re.search(r'-(\d+)-(\d+)\.hdf5$', path)
            if m:
                found.setdefault(int(m.group(1)), {})[det] = path
    return found


def _scan_slot(job):
    """the masks of one slot (both files read once): {detector: dict(gps, seconds, fs, good, dq, inj, dq_names,
    inj_names)}, good = the required bits set and finite strain"""
    paths, require_dq, require_inj = job
    out = {}
    for det in DETS:
        x = gw.read_strain(paths[det])
        n = x['dq'].size
        good = gw.good_seconds(x['dq'], x['inj'], x['dq_names'], x['inj_names'], require_dq, require_inj)
        good &= np.isfinite(x['strain'].reshape(n, -1)).all(axis=1)
        out[det] = dict(gps=int(x['gps_start']), seconds=n, fs=int(x['sample_rate']), good=good, dq=x['dq'],
                        inj=x['inj'], dq_names=x['dq_names'], inj_names=x['inj_names'])
    return out


def _map(fn, jobs, workers, initializer=None, initargs=()):
    """fn over jobs in input order: in this process for workers <= 1, else in a pool"""
    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [fn(j) for j in jobs]
    with mp.Pool(workers, initializer, initargs) as pool:
        return pool.map(fn, jobs, chunksize=1)


def timeline(gwosc_dir, require_dq=gw.REQUIRE_DQ, require_inj=gw.REQUIRE_INJ, workers=1):
    """the data of a GWOSC folder on one axis of seconds from the start of its first slot: dict with
    gps0, fs, slots [(gps, seconds, path_H1, path_L1)] in time order (the slots with both files),
    good (n,) bool both detectors good, good_<det>, dq_<det>, inj_<det> (n,) (False, 0 where there is no file),
    dq_names, inj_names"""
    files = find_files(gwosc_dir)
    pairs = [(t, p) for t, p in sorted(files.items()) if len(p) == 2]
    if not pairs:
        raise SystemExit('no H1/L1 file pairs in %s' % gwosc_dir)
    info = _map(_scan_slot, [(p, tuple(require_dq), tuple(require_inj)) for _, p in pairs], workers)
    first = info[0]['H1']
    for (t, _), s in zip(pairs, info):
        for det in DETS:
            if (s[det]['gps'] != s['H1']['gps'] or s[det]['seconds'] != s['H1']['seconds'] or
                    s[det]['fs'] != first['fs'] or s[det]['dq_names'] != first['dq_names'] or
                    s[det]['inj_names'] != first['inj_names'] or s[det]['gps'] != t):
                raise ValueError('slot %d: the files disagree on GPS, length, sample rate or mask names' % t)
    gps0 = info[0]['H1']['gps']
    n = max(s['H1']['gps'] + s['H1']['seconds'] for s in info) - gps0
    tl = dict(gps0=gps0, fs=first['fs'], dq_names=first['dq_names'], inj_names=first['inj_names'], slots=[])
    for det in DETS:
        tl['good_' + det] = np.zeros(n, bool)
        tl['dq_' + det] = np.zeros(n, np.int64)
        tl['inj_' + det] = np.zeros(n, np.int64)
    for (t, p), s in zip(pairs, info):
        a = t - gps0
        b = a + s['H1']['seconds']
        tl['slots'].append((t, s['H1']['seconds'], p['H1'], p['L1']))
        for det in DETS:
            tl['good_' + det][a:b] = s[det]['good']
            tl['dq_' + det][a:b] = s[det]['dq']
            tl['inj_' + det][a:b] = s[det]['inj']
    tl['good'] = tl['good_H1'] & tl['good_L1']
    return tl


def read_span(tl, det, gps, n_sample):
    """(n_sample,) strain of one detector from GPS gps (on the sample grid of the slots), NaN where no file covers it"""
    fs = tl['fs']
    i0 = int(round((gps - tl['gps0']) * fs))
    i1 = i0 + n_sample
    out = np.full(n_sample, np.nan)
    for t, seconds, path_h, path_l in tl['slots']:
        j0 = (t - tl['gps0']) * fs
        a, b = max(i0, j0), min(i1, j0 + seconds * fs)
        if a < b:
            with h5py.File(path_h if det == 'H1' else path_l, 'r') as h:
                out[a - i0:b - i0] = h['strain/Strain'][a - j0:b - j0]
    out[~np.isfinite(out)] = np.nan
    return out


def runs(mask):
    """(start, stop) index pairs of the runs of True"""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def epoch_name(gps_epoch):
    return 'epoch_%d.npz' % round(gps_epoch)


def plan(good, gps0, fs, T, N, pad, block, own=False):
    """stretches and blocks: (stretches [(gps start, seconds, epochs)], jobs [dict(stretch=(gps start, end),
    gps=start of the block's data, n_sample, keep=(i, j) the epochs of the block it writes, names=their files)])"""
    n = int(round(T * fs)) * (N + 1)
    p = int(round(pad * fs))
    if block > 0 and abs((N + 1) * T - round((N + 1) * T)) > 1e-9:
        raise ValueError('blocks need (N + 1) T to be whole seconds')
    stretches, jobs = [], []
    for s0, s1 in runs(good):
        K = ((s1 - s0) * fs - 2 * p) // n
        if K < (1 if own else 2):
            continue
        stretches.append((gps0 + s0, s1 - s0, K))
        size = K if block <= 0 else block
        groups = int(np.ceil(K / size))
        cuts = np.linspace(0, K, groups + 1).round().astype(int)
        for a, b in zip(cuts[:-1], cuts[1:]):
            lo, hi = max(a - 1, 0), min(b + 1, K)
            jobs.append(dict(stretch=(gps0 + s0, gps0 + s1), gps=gps0 + s0 + lo * n // fs,
                             n_sample=(hi - lo) * n + 2 * p, keep=(a - lo, b - lo),
                             names=[epoch_name(gps0 + s0 + (p + e * n) / fs) for e in range(a, b)]))
    return stretches, jobs


_CFG = {}


def _init(cfg):
    _CFG.clear()
    _CFG.update(cfg)


def _segment_and(bits, gps, gps0, T):
    """bits (n_second,) at 1 Hz from gps0 -> (n_segment,) the bits set over all of each segment [gps, gps + T)"""
    a = np.floor(np.asarray(gps) - gps0 + 1e-9).astype(int)
    b = np.ceil(np.asarray(gps) + T - gps0 - 1e-9).astype(int)
    return np.array([np.bitwise_and.reduce(bits[i:j]) for i, j in zip(a, b)], np.int64)


_A2 = {}


def u_ratio(ep, M=60.0):
    """u_H / u_L before step 5 for an equal-mass, nonspinning template of total mass M: the ratio of the sums of
    a^2 / sig2_ref_I over the bins that are not notched (exmarg.profiles.profile)"""
    key = (float(ep['T']), float(ep['fmin']), float(ep['fmax']), int(np.size(ep['nu'])), M)
    if key not in _A2:
        from exmarg.profiles import profile
        nu, _, a, _ = profile(M / 2.0, M / 2.0, 0.0, 0.0, ep['T'], ep['fmin'], ep['fmax'], 1.0)
        a2 = np.zeros(np.size(ep['nu']))
        a2[np.searchsorted(ep['nu'], nu)] = a ** 2
        _A2[key] = a2
    keep = ~np.asarray(ep['notched'], bool)
    a2 = _A2[key][keep]
    return float((a2 / ep['sig2_ref_H'][keep]).sum() / (a2 / ep['sig2_ref_L'][keep]).sum())


def write(out_dir, ep):
    """write an epoch to out_dir/epoch_<gps>.npz through a temporary file; returns the file name"""
    name = epoch_name(ep['gps_epoch'])
    tmp = os.path.join(out_dir, name[:-4] + '.tmp.npz')
    strain.write_epoch(tmp, ep)
    os.replace(tmp, os.path.join(out_dir, name))
    return name


def _condition_block(job):
    """condition one block of a stretch and write the epochs it owns (those not yet on disk)"""
    c = _CFG
    t_start = time.time()
    tl, fs = c['tl'], c['tl']['fs']
    a = int(round(job['gps'] - tl['gps0']))
    m = int(np.ceil(job['n_sample'] / fs))
    x = {det: read_span(tl, det, job['gps'], job['n_sample']) for det in DETS}
    t_read = time.time() - t_start
    eps = strain.condition(x['H1'], x['L1'], fs, job['gps'], dq_H=tl['dq_H1'][a:a + m], dq_L=tl['dq_L1'][a:a + m],
                           events=c['events'], **c['settings'])
    written = 0
    for ep in eps[job['keep'][0]:job['keep'][1]]:
        if os.path.exists(os.path.join(c['out_dir'], epoch_name(ep['gps_epoch']))):
            continue
        ep.update(inj_H=_segment_and(tl['inj_H1'][a:a + m], ep['gps'], job['gps'], ep['T']),
                  inj_L=_segment_and(tl['inj_L1'][a:a + m], ep['gps'], job['gps'], ep['T']),
                  dq_names=np.array(tl['dq_names']), inj_names=np.array(tl['inj_names']),
                  stretch=np.array(job['stretch'], float), notch_file=c['notch_file'], u_ratio_60=u_ratio(ep))
        write(c['out_dir'], ep)
        written += 1
    return dict(written=written, events=sum(int(ep['event'].sum()) for ep in eps[job['keep'][0]:job['keep'][1]]),
                read=t_read, seconds=time.time() - t_start)


def catalogue(out_dir, events_file=None):
    """(GPS times of the catalogued events, their source): the first column of events_file, else out_dir/events.json,
    written from gwosc.events() the first time"""
    if events_file:
        return np.loadtxt(events_file, ndmin=2)[:, 0], events_file
    path = os.path.join(out_dir, 'events.json')
    if not os.path.exists(path):
        found = gw.events()
        with open(path + '.tmp', 'w') as fh:
            json.dump(dict(catalogs=list(gw.CATALOGS), fetched=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                           events=found), fh, indent=1)
        os.replace(path + '.tmp', path)
    with open(path) as fh:
        return np.array([g for _, g in json.load(fh)['events']], float), path


def write_manifest(out_dir, header):
    """out_dir/manifest.txt from the epoch files in out_dir; returns (epochs, segments, flagged segments)"""
    rows, names = [], None
    for path in sorted(glob.glob(os.path.join(out_dir, 'epoch_*.npz'))):
        if path.endswith('.tmp.npz'):
            continue
        with np.load(path) as z:
            dq = [int(np.bitwise_and.reduce(z['dq_' + I])) if 'dq_' + I in z.files else -1 for I in 'HL']
            flags = [int(z['loud_' + I].sum()) if 'loud_' + I in z.files else -1 for I in 'HL']
            used = [int(z['n_used_' + I]) if 'n_used_' + I in z.files else -1 for I in 'HL']
            ratio = float(z['u_ratio_60']) if 'u_ratio_60' in z.files else u_ratio(
                {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in ('T', 'fmin', 'fmax', 'nu', 'notched',
                                                                        'sig2_ref_H', 'sig2_ref_L')})
            rows.append((os.path.basename(path), float(z['gps_epoch']), int(z['gps'].size), dq[0], dq[1],
                         int(z['event'].sum()), flags[0], flags[1], used[0], used[1],
                         int(bool(z['usable'])) if 'usable' in z.files else -1, ratio))
            if names is None and 'dq_names' in z.files:
                names = [str(v) for v in z['dq_names']]
    rows.sort(key=lambda r: r[1])
    with open(os.path.join(out_dir, 'manifest.txt.tmp'), 'w') as fh:
        for line in header:
            fh.write('# %s\n' % line)
        if names:
            fh.write('# dq bit k set: %s passes\n' % ', '.join('%d %s' % kv for kv in enumerate(names)))
        fh.write('# loud: segments flagged loud (strain.loud_blocks); used: unflagged blocks of the epoch; usable: 1 '
                 'or 0 (-1: not recorded); uH_uL_60: u_H1 / u_L1 before step 5, equal-mass template of 60 Msun\n')
        fh.write('# file gps_epoch n_segment dq_H1 dq_L1 n_event loud_H1 loud_L1 used_H1 used_L1 usable uH_uL_60\n')
        for r in rows:
            fh.write('%s %d %d %d %d %d %d %d %d %d %d %.4f\n' % r)
    os.replace(os.path.join(out_dir, 'manifest.txt.tmp'), os.path.join(out_dir, 'manifest.txt'))
    return len(rows), sum(r[2] for r in rows), sum(r[5] for r in rows)


def read_notches(path):
    """[(lo, hi)] from a text file of 'lo hi' lines (Hz), # comments"""
    return [tuple(r) for r in np.loadtxt(path, ndmin=2, comments='#')] if path else []


def notch_list(text):
    return [tuple(float(v) for v in item.split(':')) for item in text.split(',') if item] if text else []


def settings(a):
    """keyword arguments of strain.condition from the command line"""
    return dict(T=a.T, fmin=a.fmin, fmax=a.fmax, N=a.N, notches=read_notches(a.notches_file) + notch_list(a.notches),
                method=a.method, reference=a.reference, equal_noise=not a.unequal, seed=a.seed, pad=a.pad,
                band=tuple(a.band), roll=tuple(a.roll), smooth=a.smooth, event_margin=a.event_margin,
                flag=not a.no_flag, flag_p=a.flag_p, flag_iterations=a.flag_iterations, min_blocks=a.min_blocks)


def header(a, kw, source, events_source):
    return ['exmarg conditioned segments, scripts/real_data/condition.py, %s' % time.strftime('%Y-%m-%d %H:%M'),
            'input %s' % source, 'events %s (margin %g s)' % (events_source, a.event_margin),
            'settings ' + ', '.join('%s=%s' % (k, v) for k, v in kw.items() if k != 'notches'),
            'notches %d from %s %s' % (len(kw['notches']), a.notches_file or '', a.notches or '')]


def run_gwosc(a):
    os.makedirs(a.out_dir, exist_ok=True)
    for path in glob.glob(os.path.join(a.out_dir, '*.tmp.npz')):
        os.remove(path)
    t0 = time.time()
    events, events_source = catalogue(a.out_dir, a.events)
    tl = timeline(a.in_dir, a.require_dq.split(','), a.require_inj.split(','), a.workers)
    kw = settings(a)
    pad = a.T if a.pad is None else a.pad
    stretches, jobs = plan(tl['good'], tl['gps0'], tl['fs'], a.T, a.N, pad, a.block, a.reference == 'epoch')
    span = (tl['gps0'], tl['gps0'] + tl['good'].size)
    print('%d slots, GPS %d to %d: %d s both good (%.2f days) in %d runs; %d stretches of at least %s epochs: %d '
          'epochs, %d segments, in %d blocks; %d catalogued events in the span; scan %.1f s'
          % (len(tl['slots']), span[0], span[1], tl['good'].sum(), tl['good'].sum() / 86400.0, len(runs(tl['good'])),
             len(stretches), 'one' if a.reference == 'epoch' else 'two', sum(s[2] for s in stretches),
             sum(s[2] for s in stretches) * (a.N + 1), len(jobs), np.sum((events >= span[0]) & (events < span[1])),
             time.time() - t0), flush=True)
    with open(os.path.join(a.out_dir, 'stretches.txt'), 'w') as fh:
        fh.write('# gps_start seconds epochs\n')
        for s in stretches:
            fh.write('%d %d %d\n' % s)
    todo = [j for j in jobs if not all(os.path.exists(os.path.join(a.out_dir, n)) for n in j['names'])]
    print('%d blocks to run (%d done before), %d workers' % (len(todo), len(jobs) - len(todo), a.workers), flush=True)
    cfg = dict(tl=tl, settings=kw, events=events, out_dir=a.out_dir, notch_file=a.notches_file or '')
    t1, done, written = time.time(), 0, 0
    if a.workers <= 1:
        _init(cfg)
        results = map(_condition_block, todo)
    else:
        pool = mp.Pool(a.workers, _init, (cfg,))
        results = pool.imap_unordered(_condition_block, todo)
    for r in results:
        done += 1
        written += r['written']
        if done % max(1, len(todo) // 20) == 0 or done == len(todo):
            dt = time.time() - t1
            print('%d/%d blocks, %d epochs written, last block %.1f s (read %.1f s), elapsed %.0f s, eta %.0f s'
                  % (done, len(todo), written, r['seconds'], r['read'], dt, dt / done * (len(todo) - done)), flush=True)
    if a.workers > 1:
        pool.close()
        pool.join()
    n_ep, n_seg, n_ev = write_manifest(a.out_dir, header(a, kw, a.in_dir, events_source))
    print('%d epochs, %d segments (%d flagged with an event) in %s; %d written by this run in %.0f s'
          % (n_ep, n_seg, n_ev, a.out_dir, written, time.time() - t0), flush=True)
    return 0


def run_arrays(a):
    gps = a.gps or (None, None)
    x = {I: load_strain(path, a.fs, g) for I, path, g in (('H', a.inputs[0], gps[0]), ('L', a.inputs[1], gps[1]))}
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
    kw = settings(a)
    total = 0
    for s0, s1 in runs(good):
        sl = slice(s0 * int(fs), s1 * int(fs))
        eps = strain.condition(x['H']['strain'][sl], x['L']['strain'][sl], fs, t0 + s0, events=events,
                               dq_H=None if x['H']['dq'] is None else x['H']['dq'][s0:s1],
                               dq_L=None if x['L']['dq'] is None else x['L']['dq'][s0:s1], **kw)
        for ep in eps:
            if not os.path.exists(os.path.join(a.out_dir, epoch_name(ep['gps_epoch']))):
                ep.update(stretch=np.array([t0 + s0, t0 + s1], float), notch_file=a.notches_file or '',
                          u_ratio_60=u_ratio(ep))
                write(a.out_dir, ep)
        total += len(eps)
        print('stretch GPS %d + %d s: %d epochs, %d s of segments, %d events flagged'
              % (t0 + s0, s1 - s0, len(eps), len(eps) * (a.N + 1) * a.T, sum(int(ep['event'].sum()) for ep in eps)),
              flush=True)
    write_manifest(a.out_dir, header(a, kw, ' '.join(a.inputs[:2]), a.events or 'none'))
    print('%d epochs written to %s (%d coincident good seconds of %d)' % (total, a.out_dir, good.sum(), good.size))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', help='<gwosc_dir> <out_dir>, or H1.npz L1.npz <out_dir>')
    p.add_argument('--fs', type=float, help='sample rate of .npy inputs (Hz)')
    p.add_argument('--gps', type=float, nargs=2, metavar=('GPS_H1', 'GPS_L1'), help='first sample of .npy inputs')
    p.add_argument('--require', type=int, default=3, help='.npz input: dq bits that must be set (default 3: DATA and '
                                                          "CBC_CAT1 in GWOSC's order)")
    p.add_argument('--require-dq', default=','.join(gw.REQUIRE_DQ), help='GWOSC input: DQ bits that must pass')
    p.add_argument('--require-inj', default=','.join(gw.REQUIRE_INJ), help='GWOSC input: injection bits that must pass')
    p.add_argument('--T', type=float, default=8.0)
    p.add_argument('--fmin', type=float, default=20.0)
    p.add_argument('--fmax', type=float, default=512.0)
    p.add_argument('--N', type=int, default=32, help='an epoch is N + 1 segments')
    p.add_argument('--method', default='median', choices=('mean', 'median'))
    p.add_argument('--reference', default='neighbours', choices=('neighbours', 'epoch'),
                   help="curve an epoch is handed over on: the neighbouring epochs, or (mean only) its own blocks, "
                        "which also keeps stretches of one epoch")
    p.add_argument('--notches', default='', help='lo:hi,lo:hi,... (Hz), added to those of --notches-file')
    p.add_argument('--notches-file', default='', help="text file of 'lo hi' lines (Hz)")
    p.add_argument('--events', help='text file, first column the GPS of catalogued events (default for GWOSC input: '
                                    'gwosc.events(), cached in out_dir/events.json)')
    p.add_argument('--event-margin', type=float, default=2.0, help='s around an event that flag a segment')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pad', type=float, default=None, help='seconds on each side of an epoch (default T)')
    p.add_argument('--band', type=float, nargs=2, default=(16.0, 600.0))
    p.add_argument('--roll', type=float, nargs=2, default=(4.0, 100.0))
    p.add_argument('--smooth', type=int, default=17)
    p.add_argument('--unequal', action='store_true', help='skip step 5: keep sig2_H and sig2_L')
    p.add_argument('--no-flag', action='store_true', help='keep loud blocks in the estimates (strain.loud_blocks)')
    p.add_argument('--flag-p', type=float, default=1e-6, help='false-flag probability per band of a block on Gaussian '
                                                               'noise')
    p.add_argument('--flag-iterations', type=int, default=10)
    p.add_argument('--min-blocks', type=int, default=16, help='an epoch is usable when its estimates rest on at least '
                                                               'this many unflagged blocks')
    p.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1), help='GWOSC input: processes')
    p.add_argument('--block', type=int, default=24, help='GWOSC input: epochs per block (0: whole stretches)')
    a = p.parse_args(argv)
    if len(a.inputs) == 2 and os.path.isdir(a.inputs[0]):
        a.in_dir, a.out_dir = a.inputs
        return run_gwosc(a)
    if len(a.inputs) != 3:
        p.error('give <gwosc_dir> <out_dir>, or H1 L1 <out_dir>')
    a.out_dir = a.inputs[2]
    return run_arrays(a)


if __name__ == '__main__':
    sys.exit(main())
