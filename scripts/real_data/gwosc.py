"""Download and read public GWOSC strain (4096 s files at 4 kHz, HDF5, H1 and L1) with their data-quality and
injection masks.  The two detectors are always handled as a pair: only the file slots for which both exist are used.

    python gwosc.py fetch <gps_start> <gps_end> <out_dir> [--streams N] [--run O3b]
                                                    files go to <out_dir>/H1 and <out_dir>/L1
    python gwosc.py summary <out_dir>             good seconds per slot, 8 s windows, catalogued events in the range

A download that was interrupted is resumed by running the same command again: files with the size the server reports
are skipped, a partial file is fetched again.  The command exits with status 1 if a file is still missing.
In Python:  pairs = list_pairs(a, b); download(pairs, out_dir); x = read_strain(path); good_seconds(...)
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import os
import re
import sys
import threading
import time
import urllib.request

import h5py
import numpy as np

API = 'https://gwosc.org/api/v2'
SLOT = 4096
CHUNK = 16 << 20
DETECTORS = ('H1', 'L1')
REQUIRE_DQ = ('DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3')
REQUIRE_INJ = ('NO_CBC_HW_INJ', 'NO_BURST_HW_INJ', 'NO_DETCHAR_HW_INJ', 'NO_STOCH_HW_INJ')
CATALOGS = ('GWTC-3-confident', 'GWTC-3-marginal')


def _locate():
    """gwosc.locate; this file's name hides the package when its folder is on sys.path (as for a script run here)"""
    here = os.path.dirname(os.path.abspath(__file__))
    own = sys.modules.get('gwosc')
    if own is not None and os.path.abspath(getattr(own, '__file__', '') or '') == os.path.abspath(__file__):
        del sys.modules['gwosc']
    path = sys.path
    sys.path = [p for p in path if os.path.abspath(p or '.') != here]
    try:
        from gwosc import locate
    finally:
        sys.path = path
    return locate


def _slot(name):
    return int(re.search(r'-(\d+)-%d\.hdf5$' % SLOT, name).group(1))


def list_pairs(gps_start, gps_end, run='O3b'):
    """File slots overlapping [gps_start, gps_end) for which both an H1 and an L1 file exist.
    Returns a list of (slot_gps, url_H1, url_L1), sorted by slot start GPS; the slots are 4096 s long"""
    locate = _locate()
    urls = {}
    for det in DETECTORS:
        urls[det] = {_slot(u): u for u in _urls(locate, det, int(gps_start), int(gps_end), run)}
    return [(t, urls['H1'][t], urls['L1'][t]) for t in sorted(urls['H1'].keys() & urls['L1'].keys())
            if t < gps_end and t + SLOT > gps_start]


def _urls(locate, det, gps_start, gps_end, run):
    """the file urls of one detector overlapping [gps_start, gps_end).  gwosc.locate.get_urls raises ValueError unless
    the first and the last slot of the range both have a file, so the range is narrowed from each end to the first
    slot that has one; [] when none has"""
    def get(a, b):
        return locate.get_urls(det, a, b - 1, dataset=run, sample_rate=4096, format='hdf5')

    try:
        return get(gps_start, gps_end)
    except ValueError:
        pass
    slots = range(gps_start // SLOT * SLOT, gps_end, SLOT)

    def first(order):
        for t in order:
            try:
                get(max(t, gps_start), min(t + SLOT, gps_end))
                return t
            except ValueError:
                continue
        return None

    lo = first(slots)
    if lo is None:
        return []
    hi = first(reversed(slots))
    return get(max(lo, gps_start), min(hi + SLOT, gps_end))


def _open(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60)


def _size(url):
    """(size in bytes the server reports, whether it honours range requests)"""
    with _open(url, {'Range': 'bytes=0-0'}) as r:
        if r.headers.get('Content-Range'):
            return int(r.headers['Content-Range'].rsplit('/', 1)[1]), True
        return int(r.headers['Content-Length']), False


def _retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except (OSError, http.client.HTTPException):
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def _fetch_range(url, fd, a, b, ranged):
    """bytes a..b of the url into the file descriptor at offset a"""
    n = 0
    with _open(url, {'Range': 'bytes=%d-%d' % (a, b)} if ranged else {}) as r:
        if ranged and r.status != 206:
            raise OSError('range request answered with status %d' % r.status)
        for block in iter(lambda: r.read(1 << 20), b''):
            os.pwrite(fd, block, a + n)
            n += len(block)
    if n != b - a + 1:
        raise OSError('short read: %d of %d bytes' % (n, b - a + 1))


def _hms(s):
    s = int(max(s, 0))
    return '%d:%02d:%02d' % (s // 3600, s % 3600 // 60, s % 60)


def download(pairs, out_dir, streams=6):
    """Download the files of list_pairs() into out_dir/H1 and out_dir/L1 with `streams` parallel range requests,
    each file to <name>.part and renamed when complete.  A file with the size the server reports is skipped.
    Prints one line per finished file.  Returns (number of files present, number expected)"""
    jobs = []
    for _, uh, ul in pairs:
        for det, url in zip(DETECTORS, (uh, ul)):
            os.makedirs(os.path.join(out_dir, det), exist_ok=True)
            jobs.append((det, url, os.path.join(out_dir, det, os.path.basename(url))))

    def size_of(job):
        try:
            return _retry(lambda: _size(job[1]))
        except (OSError, http.client.HTTPException) as e:
            print('no size for %s: %s' % (job[1], e), flush=True)
            return None, False

    def present(path, size):
        return size is not None and os.path.isfile(path) and os.path.getsize(path) == size

    with cf.ThreadPoolExecutor(streams) as ex:
        info = list(ex.map(size_of, jobs))
        todo = [(j, i) for j, i in zip(jobs, info) if i[0] is not None and not present(j[2], i[0])]
        total = sum(i[0] for _, i in todo)
        print('%d files expected, %d present, %d to download (%.2f GB), %d streams'
              % (len(jobs), len(jobs) - len(todo), len(todo), total / 1e9, streams), flush=True)
        t0, lock = time.time(), threading.Lock()
        count = dict(files=0, bytes=0)

        def finish(st):
            if st['fd'] is not None:
                os.close(st['fd'])
            part = st['path'] + '.part'
            ok = not st['failed'] and os.path.getsize(part) == st['size']
            with lock:
                if not ok:
                    print('FAILED %s %s: %s' % (st['det'], os.path.basename(st['path']), st['error']), flush=True)
                    if os.path.exists(part):
                        os.remove(part)
                    return
                os.replace(part, st['path'])
                count['files'] += 1
                count['bytes'] += st['size']
                dt = time.time() - t0
                rate = count['bytes'] / dt
                print('%s %s %.1f MB | %d/%d files, %.2f/%.2f GB | %.1f MB/s | elapsed %s, eta %s'
                      % (st['det'], os.path.basename(st['path']), st['size'] / 1e6, count['files'], len(todo),
                         count['bytes'] / 1e9, total / 1e9, rate / 1e6, _hms(dt),
                         _hms((total - count['bytes']) / rate)), flush=True)

        def run(st, a, b):
            try:
                with st['lock']:
                    if st['fd'] is None and not st['failed']:
                        st['fd'] = os.open(st['path'] + '.part', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
                if not st['failed']:
                    _retry(lambda: _fetch_range(st['url'], st['fd'], a, b, st['ranged']))
            except (OSError, http.client.HTTPException) as e:
                st['failed'], st['error'] = True, e
            finally:
                with st['lock']:
                    st['left'] -= 1
                    last = st['left'] == 0
                if last:
                    finish(st)

        futures = []
        for (det, url, path), (size, ranged) in todo:
            edges = list(range(0, size, CHUNK)) if ranged else [0]
            st = dict(det=det, url=url, path=path, size=size, ranged=ranged, fd=None, failed=False, error=None,
                      left=len(edges), lock=threading.Lock())
            for a in edges:
                futures.append(ex.submit(run, st, a, min(a + CHUNK, size) - 1))
        cf.wait(futures)
    n = sum(present(j[2], i[0]) for j, i in zip(jobs, info))
    print('%d of %d files present in %s' % (n, len(jobs), out_dir), flush=True)
    return n, len(jobs)


def _names(h, group, key, mask):
    """bit names of a mask: the dataset (or attribute) `key` next to the mask, decoded to str"""
    g = h[group]
    v = g[key][()] if key in g else g[mask].attrs[key]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.atleast_1d(v)]


def read_strain(path):
    """Read one GWOSC HDF5 file.  Returns a dict with
    strain       (n,) float64, NaN where there is no data, n = sample_rate * duration
    gps_start    GPS of the first sample, int
    sample_rate  Hz, int
    dq           (duration,) int64, data-quality mask at 1 Hz, bit k set means the k-th name of dq_names passes
    inj          (duration,) int64, injection mask at 1 Hz, bit k set means the k-th name of inj_names passes
    dq_names, inj_names   lists of str, the bit names as written in the file (DQShortnames, InjShortnames)"""
    with h5py.File(path, 'r') as h:
        s = h['strain/Strain']
        x = np.array(s[:], dtype=np.float64)
        rate = int(round(1.0 / float(s.attrs['Xspacing'])))
        gps = int(h['meta/GPSstart'][()]) if 'GPSstart' in h['meta'] else int(round(float(s.attrs['Xstart'])))
        dq = np.array(h['quality/simple/DQmask'][:], dtype=np.int64)
        inj = np.array(h['quality/injections/Injmask'][:], dtype=np.int64)
        dq_names = _names(h, 'quality/simple', 'DQShortnames', 'DQmask')
        inj_names = _names(h, 'quality/injections', 'InjShortnames', 'Injmask')
    x[~np.isfinite(x)] = np.nan
    if x.size != rate * dq.size or inj.size != dq.size:
        raise ValueError('%s: %d samples at %d Hz but %d DQ and %d injection seconds'
                         % (path, x.size, rate, dq.size, inj.size))
    return dict(strain=x, gps_start=gps, sample_rate=rate, dq=dq, inj=inj, dq_names=dq_names, inj_names=inj_names)


def _bits(names, wanted):
    missing = [w for w in wanted if w not in names]
    if missing:
        raise KeyError('mask has no bit named %s; it has %s' % (missing, names))
    return sum(1 << names.index(w) for w in wanted)


def good_seconds(dq, inj, dq_names, inj_names, require_dq=REQUIRE_DQ, require_inj=REQUIRE_INJ):
    """(duration,) bool, True for the seconds in which every required bit of both masks is set.
    NO_CW_HW_INJ is left out of the default injection bits: it is 0 all the time in O3 (continuous CW injections)"""
    dq, inj = np.asarray(dq, dtype=np.int64), np.asarray(inj, dtype=np.int64)
    m, k = _bits(list(dq_names), require_dq), _bits(list(inj_names), require_inj)
    return ((dq & m) == m) & ((inj & k) == k)


def events(catalogs=CATALOGS):
    """Events of the GWOSC catalogs.  Returns a list of (name, gps) sorted by GPS"""
    found = {}
    for cat in catalogs:
        url = '%s/catalogs/%s/events?page_size=100' % (API, cat)
        while url:
            with _open(url) as r:
                page = json.load(r)
            found.update({e['name']: float(e['gps']) for e in page['results']})
            url = page['next']
    return sorted(found.items(), key=lambda e: e[1])


def _scan(out_dir):
    """{slot gps: {detector: path}} of the files in out_dir/H1 and out_dir/L1"""
    slots = {}
    for det in DETECTORS:
        d = os.path.join(out_dir, det)
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if re.search(r'-\d+-%d\.hdf5$' % SLOT, name):
                slots.setdefault(_slot(name), {})[det] = os.path.join(d, name)
    return slots


def _good_file(path):
    """(duration,) bool: seconds that pass the default masks and hold no NaN"""
    x = read_strain(path)
    finite = np.isfinite(x['strain'].reshape(x['dq'].size, -1)).all(axis=1)
    return good_seconds(x['dq'], x['inj'], x['dq_names'], x['inj_names']) & finite


def summary(out_dir, window=8):
    """Print the good seconds of each slot that has both files (H1, L1, both), the totals, the number of
    `window` s windows that tile the runs of both-good seconds (runs may span slots; each run is cut from its start,
    so a window lies fully inside both-good time) and the catalogued events in the span of the files"""
    slots = {t: p for t, p in _scan(out_dir).items() if len(p) == 2}
    if not slots:
        print('no H1/L1 file pairs in', out_dir)
        return
    t0, t1 = min(slots), max(slots) + SLOT
    both = np.zeros(t1 - t0, dtype=bool)
    tot = np.zeros(3, dtype=int)
    print('%-12s %8s %8s %9s' % ('slot_gps', 'H1_good', 'L1_good', 'both_good'))
    for t in sorted(slots):
        g = []
        for det in DETECTORS:
            try:
                g.append(_good_file(slots[t][det]))
            except (OSError, KeyError, ValueError) as e:
                print('cannot read %s: %s' % (slots[t][det], e))
                g.append(np.zeros(SLOT, dtype=bool))
        n = len(g[0]) if len(g[0]) == len(g[1]) else 0
        b = g[0] & g[1] if n else np.zeros(SLOT, dtype=bool)
        both[t - t0:t - t0 + len(b)] = b
        row = (int(g[0].sum()), int(g[1].sum()), int(b.sum()))
        tot += row
        print('%-12d %8d %8d %9d' % ((t,) + row))
    print('%-12s %8d %8d %9d   (%d slots, %.2f days both good)'
          % (('total',) + tuple(tot) + (len(slots), tot[2] / 86400)))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], both.astype(np.int8), [0]))))
    runs = edges[1::2] - edges[0::2]
    print('8 s windows fully inside both-good time: %d (%d runs, longest %d s)'
          % ((runs // window).sum() if runs.size else 0, runs.size, runs.max() if runs.size else 0))
    try:
        ev = [e for e in events() if t0 <= e[1] < t1]
    except (OSError, http.client.HTTPException, ValueError, KeyError) as e:
        print('catalogued events: not available (%s)' % e)
        return
    print('catalogued events in GPS %d to %d: %d' % (t0, t1, len(ev)))
    for name, gps in ev:
        print('  %-18s %.1f  both-good second: %s' % (name, gps, 'yes' if both[int(gps) - t0] else 'no'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch', help='download the H1 and L1 files of a GPS range')
    f.add_argument('gps_start', type=int)
    f.add_argument('gps_end', type=int)
    f.add_argument('out_dir')
    f.add_argument('--streams', type=int, default=6)
    f.add_argument('--run', default='O3b')
    s = sub.add_parser('summary', help='good seconds and events of the downloaded files')
    s.add_argument('out_dir')
    a = p.parse_args(argv)
    if a.cmd == 'summary':
        summary(a.out_dir)
        return 0
    pairs = list_pairs(a.gps_start, a.gps_end, a.run)
    print('%d H1/L1 file pairs of %s overlap GPS %d to %d' % (len(pairs), a.run, a.gps_start, a.gps_end), flush=True)
    if not pairs:
        return 1
    n, expected = download(pairs, a.out_dir, a.streams)
    return int(n < expected)


if __name__ == '__main__':
    sys.exit(main())
