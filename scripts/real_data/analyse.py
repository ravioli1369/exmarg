"""The conditioned week through the glitch-robust analysis, with its controls (docs/real_data.md, "Order of the analysis
and its criteria"), into files that scripts/validate/plant_fit.py fits.  Needs the redshift tables; meant for a GPU.

    python scripts/real_data/analyse.py tables/redshift gpu data/segments/O3b_v2 --out runs/week [--part 3/9]
                                        [--variants zero,pr,rev,slide+0.5,slide-1.75,inj] [--nbar-inj 0.3] [--seed 1]
                                        [--epochs-per-file 8] [--chunk 4096] [--resident]
    python scripts/real_data/summarize.py runs/week                   # the controls first; --recorded for the week

Segments.  The epochs of the manifest (strain.write_epoch files, one noise curve each), in time order.  An epoch is
analysed when it is usable; its segments when they are analysable (loud in neither detector) and not flagged as holding
a catalogued event.  The segment index k is the rank of a segment among all those of the week, in time order: it does
not depend on --part, and <out>/selection.txt records it (epoch, GPS, segments, selected, first k).

Variants, built from the selected segments of an epoch and analysed together in one pe_redshift(glitch=True) call per
epoch (they share the noise curve; the bank of plant.MASSES is built on it):
    zero         as recorded
    pr           controls.phase_randomize, seed [seed, GPS of the epoch, 1]
    rev          controls.time_reverse
    slide+0.5    controls.time_slide by +0.5 s, slide-1.75 by -1.75 s (2048 and -7168 matched-filter samples)
    inj          slide+0.5 plus mergers injected with probability --nbar-inj per segment, from the population of the
                 plant (plant.population(0.99119), plant.Sampler, plant.mergers, equal masses, no spin), the signal of
                 pp.simulate as the difference of two calls with the same seed, with y and with y = 0.  The injection
                 is added to the conditioned data, after the whitening and the equalization: unlike a real merger it
                 does not pass through the spectrum estimate (no gain error) and is not cut at a segment edge.  Its
                 optimal network SNR is computed with the epoch's sig2.
The random numbers of a segment are keyed by its GPS second (plant.stream(seed, gps, use)), those of pe_redshift by the
epoch's GPS and a fixed row per variant and segment, so every variant of every segment is the same whatever --part and
--variants are.  The chunk passed to pe_redshift is capped there by the free GPU memory (pe._fit_chunk).

Output: <out>/<variant>/plant_<first k>_<end k>.npz, one per group of --epochs-per-file epochs (groups cut at multiples
of it in the manifest; --part i/n takes the groups i n_group / n to (i + 1) n_group / n, contiguous in time), written
under a temporary name and renamed; a group is skipped when the files of every requested variant exist.  The files
follow plant.py: the reduction is plant.reduce on the same population (z >= 0.99119, detector-frame total mass 20 to
400) and ymin (plant.default_ymin), and the meta holds control = the variant, nbar = 0 for the four controls,
--nbar-inj for inj and null (unknown) for zero, glitch_truth false (no glitch is known in real data: glitch_I False,
rho_I and t0_I NaN), merger_truth 'injections' for inj and 'none' otherwise (merger False, m1 and z NaN, y and snr 0).
Per segment also: gps, gps_epoch, the flags event, analysable, loud_H, loud_L, dq_H, dq_L, inj_H, inj_L, the band
monitors monitor_H, monitor_L (n, 5) and the epoch's u_ratio_60.
"""
import argparse
import glob
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'validate'))
import plant
from exmarg import strain
from exmarg.pp import simulate
from exmarg.search import controls

VARIANTS = ('zero', 'pr', 'rev', 'slide+0.5', 'slide-1.75', 'inj')
CONTROLS = ('pr', 'rev', 'slide+0.5', 'slide-1.75')
SHIFTS = {'slide+0.5': 0.5, 'slide-1.75': -1.75}
ZMIN = 0.99119
CHUNK = 4096
VERSION = 1
FLAGS = ('event', 'analysable', 'loud_H', 'loud_L', 'dq_H', 'dq_L', 'inj_H', 'inj_L', 'monitor_H', 'monitor_L')


def manifest(seg_dir):
    """the epoch files of the manifest in time order: [(file name, GPS of the epoch)]"""
    rows = []
    with open(os.path.join(seg_dir, 'manifest.txt')) as fh:
        for line in fh:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                rows.append((parts[0], float(parts[1])))
    rows.sort(key=lambda r: r[1])
    return rows


def selected(z):
    """(n_segment,) bool: the segments of an epoch file that are analysed (usable epoch, analysable, no event)"""
    return np.asarray(z['analysable'], bool) & ~np.asarray(z['event'], bool) & bool(z['usable'])


def selection(seg_dir):
    """every epoch of the manifest: dicts name, gps_epoch, n_segment, n_selected, k (the first k of its selected
    segments)"""
    out, k = [], 0
    for name, gps in manifest(seg_dir):
        with np.load(os.path.join(seg_dir, name)) as z:
            sel = selected(z)
        out.append(dict(name=name, gps_epoch=gps, n_segment=int(sel.size), n_selected=int(sel.sum()), k=k))
        k += int(sel.sum())
    return out


def selection_text(sel):
    return ''.join('%s %d %d %d %d\n' % (e['name'], round(e['gps_epoch']), e['n_segment'], e['n_selected'], e['k'])
                   for e in sel)


def record_selection(out, sel):
    """<out>/selection.txt, written once; a different selection in an existing one stops the run"""
    path, text = os.path.join(out, 'selection.txt'), selection_text(sel)
    if os.path.exists(path):
        with open(path) as fh:
            if fh.read() != text:
                raise SystemExit('%s differs from the selection of these segments' % path)
        return
    tmp = '%s.%s.%d.tmp' % (path, socket.gethostname(), os.getpid())
    with open(tmp, 'w') as fh:
        fh.write(text)
    os.replace(tmp, path)


def groups(n_epoch, per, part=(0, 1)):
    """[(a, b)] the groups of epochs of a part: manifest indices a .. b - 1, cut at multiples of per; part (i, n) takes
    the groups i G / n to (i + 1) G / n of the G groups"""
    cuts = list(range(0, n_epoch, per)) + [n_epoch]
    allg = list(zip(cuts[:-1], cuts[1:]))
    i, n = part
    return allg[i * len(allg) // n:(i + 1) * len(allg) // n]


def whole_samples(seg, shift):
    """the number of matched-filter samples of a time slide (ValueError when it is not whole)"""
    m = shift / controls.time_step(seg)
    if abs(m - round(m)) > 1e-9:
        raise ValueError('a slide of %g s is %.6f matched-filter samples, not a whole number' % (shift, m))
    return int(round(m))


def injections(cfg, ep, rows, sampler):
    """the mergers injected into the segments rows of the epoch ep: (signal {I: (n, n_freq) complex}, truth dict of
    (n,) arrays merger, m1, z, y, snr), presence and (m1, z) from plant.mergers keyed by the GPS second, the signal of
    pp.simulate (seed plant.stream(seed, gps, 'noise')) with y less that with y = 0, snr the optimal network SNR with
    the epoch's sig2"""
    gps = [int(round(float(ep['gps'][i]))) for i in rows]
    present, m1, z, y = plant.mergers(dict(seed=cfg['seed'], nbar=cfg['nbar_inj']), gps, sampler)
    n, nu = len(rows), np.asarray(ep['nu'])
    h = {I: np.zeros((n, nu.size), complex) for I in 'HL'}
    snr = np.zeros(n)
    for j in np.flatnonzero(present):
        mdet = 2.0 * m1[j] * (1.0 + z[j])

        def one(yj):
            seg, _, _ = simulate(1, plant.stream(cfg['seed'], gps[j], 'noise'), lambda rng: (mdet, 1.0, 0.0),
                                 lambda rng: yj, float(ep['T']), float(ep['fmin']), float(ep['fmax']))
            if not np.array_equal(seg['nu'], nu):
                raise ValueError('the bins of pp.simulate are not those of the epoch')
            return seg
        a, b = one(y[j]), one(0.0)
        for I in 'HL':
            h[I][j] = a['d_' + I][0] - b['d_' + I][0]
        snr[j] = np.sqrt(sum(2.0 * np.sum(np.abs(h[I][j]) ** 2 / ep['sig2']) for I in 'HL'))
    truth = dict(merger=present, m1=np.where(present, m1, np.nan), z=np.where(present, z, np.nan), y=y, snr=snr)
    return h, truth


def variants(cfg, ep, names, sampler):
    """the selected segments of the epoch ep in each variant of names: ({variant: segments dict}, rows (n,) their
    indices in the epoch, injection truth or None)"""
    if 'sig2' not in ep:
        raise ValueError('the epoch has sig2_H and sig2_L: the tables need one noise curve (equal_noise)')
    rows = np.flatnonzero(selected(ep))
    base = dict(d_H=np.asarray(ep['d_H'])[rows].astype(complex), d_L=np.asarray(ep['d_L'])[rows].astype(complex),
                f=ep['f'], nu=ep['nu'], sig2=ep['sig2'], T=ep['T'], fmin=ep['fmin'], fmax=ep['fmax'])
    out, inj = {}, None
    for v in names:
        if v == 'zero':
            out[v] = base
        elif v == 'pr':
            out[v] = controls.phase_randomize(base, [int(cfg['seed']), int(round(float(ep['gps_epoch']))), 1])
        elif v == 'rev':
            out[v] = controls.time_reverse(base)
        elif v in SHIFTS:
            whole_samples(base, SHIFTS[v])
            out[v] = controls.time_slide(base, SHIFTS[v])
        elif v == 'inj':
            whole_samples(base, SHIFTS['slide+0.5'])
            slid = controls.time_slide(base, SHIFTS['slide+0.5'])
            h, inj = injections(cfg, ep, rows, sampler)
            out[v] = dict(slid, d_H=slid['d_H'] + h['H'], d_L=slid['d_L'] + h['L'])
        else:
            raise ValueError('unknown variant %r' % v)
    return out, rows, inj


def make_analysis(store, device, resident=False, chunk=CHUNK, ymin=None):
    """factory(segments) builds the bank of plant.MASSES on the epoch's noise curve and returns pe(segments, seed, rows)
    -> the pe_redshift dict (glitch=True) of those rows of the segments, on plant.GRID"""
    from exmarg.search import mass_bank, pe_redshift

    def factory(seg0):
        bank = mass_bank(plant.MASSES, seg0, device)

        def pe(seg, seed, rows):
            return pe_redshift(seg, bank, store, rows=rows, chunk=chunk, resident=resident, seed=seed, ymin=ymin,
                               mdet_range=plant.MDET, glitch=True, **plant.GRID)
        return pe
    return factory


PER_SEGMENT = plant.KEYS + ('ln_glitch_H', 'ln_glitch_L', 'wmax_H', 'wmax_L')


def campaign(a, seg_dir, variant):
    """the settings of one variant's files (the same in every file of it)"""
    with open(os.path.join(seg_dir, 'manifest.txt'), 'rb') as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    nbar = None if variant == 'zero' else a.nbar_inj if variant == 'inj' else 0.0
    cfg = dict(version=VERSION, source='strain', segments=os.path.basename(os.path.normpath(seg_dir)),
               manifest_sha256=digest, selection='usable epoch, analysable, no catalogued event', variant=variant,
               control=variant, seed=a.seed, nbar=nbar, nbar_inj=a.nbar_inj, shift=SHIFTS.get(
                   variant, SHIFTS['slide+0.5'] if variant == 'inj' else None),
               glitch_truth=False, merger_truth='injections' if variant == 'inj' else 'none',
               truth_note='real data: no merger and no glitch is known; merger, glitch_I False, rho_I, t0_I NaN'
                          + (' except the injected mergers' if variant == 'inj' else ''),
               eps_H=None, eps_L=None, glitch_kind=None, gamma=None, rho_min=None, rho_max=None,
               epochs_per_file=a.epochs_per_file, zmin=ZMIN, mdet_range=plant.MDET,
               bank=(float(plant.MASSES[0]), float(plant.MASSES[-1]), plant.MASSES.size), grid=plant.GRID,
               ymin=plant.default_ymin() if a.ymin is None else float(a.ymin), n_loud=1024, n_random=1024,
               glitch_settings='pe default')
    return json.loads(json.dumps(cfg))


def path_of(out, variant, k0, k1):
    return os.path.join(out, variant, 'plant_%010d_%010d.npz' % (k0, k1))


def epoch_arrays(ep, rows, n, inj):
    """the truth and bookkeeping arrays of one variant's rows of an epoch"""
    if inj is None:
        truth = dict(merger=np.zeros(n, bool), m1=np.full(n, np.nan), z=np.full(n, np.nan), y=np.zeros(n),
                     snr=np.zeros(n))
    else:
        truth = dict(inj)
    truth.update(merger_H=truth['merger'], merger_L=truth['merger'], coherent=truth['merger'])
    for I in 'HL':
        truth.update({'glitch_' + I: np.zeros(n, bool), 'kind_' + I: np.full(n, '', '<U16'),
                      'rho_' + I: np.full(n, np.nan), 't0_' + I: np.full(n, np.nan)})
    truth.update(gps=np.asarray(ep['gps'], float)[rows], gps_epoch=np.full(n, float(ep['gps_epoch'])),
                 u_ratio_60=np.full(n, float(ep.get('u_ratio_60', np.nan))))
    for key in FLAGS:
        if key in ep:
            truth[key] = np.asarray(ep[key])[rows]
    return truth


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def run(a, factory, log=print):
    """the groups of epochs of this part, every requested variant; returns the paths written"""
    names = [v for v in VARIANTS if v in a.variants]
    sel = selection(a.segments)
    os.makedirs(a.out, exist_ok=True)
    record_selection(a.out, sel)
    cfgs = {v: campaign(a, a.segments, v) for v in names}
    for v in names:
        os.makedirs(os.path.join(a.out, v), exist_ok=True)
        for path in sorted(glob.glob(os.path.join(a.out, v, 'plant_*.npz'))):
            plant.check_campaign(path, cfgs[v])
    pop = plant.population(ZMIN)
    sampler = plant.Sampler(pop, plant.MDET)
    params = dict(seed=a.seed, nbar_inj=a.nbar_inj)
    written, began, rows_done = [], time.time(), 0
    todo = groups(len(sel), a.epochs_per_file, a.part)
    for g, (e0, e1) in enumerate(todo):
        k0, k1 = sel[e0]['k'], sel[e1 - 1]['k'] + sel[e1 - 1]['n_selected']
        need = [v for v in names if k1 > k0 and not os.path.exists(path_of(a.out, v, k0, k1))]
        if not need:
            log('%s  epochs [%d, %d) k [%d, %d): %s' % (stamp(), e0, e1, k0, k1, 'present, skipped' if k1 > k0 else
                                                          'no segment selected'), flush=True)
            continue
        acc = {v: [] for v in need}
        t_build = t_pe = 0.0
        n_rows = 0
        for e in range(e0, e1):
            if sel[e]['n_selected'] == 0:
                continue
            t0 = time.time()
            ep = strain.read_epoch(os.path.join(a.segments, sel[e]['name']))
            var, rows, inj = variants(params, ep, need, sampler)
            n, ns = rows.size, int(np.asarray(ep['gps']).size)
            slots = {v: VARIANTS.index(v) * ns + rows for v in need}
            stack = {I: np.zeros((len(VARIANTS) * ns, np.size(ep['nu'])), complex) for I in 'HL'}
            for v in need:
                for I in 'HL':
                    stack[I][slots[v]] = var[v]['d_' + I]
            seg = dict(var[need[0]], d_H=stack['H'], d_L=stack['L'])
            take = np.concatenate([slots[v] for v in need])
            t1 = time.time()
            pe = factory(seg)
            res = pe(seg, plant.stream(a.seed, int(round(float(ep['gps_epoch']))), 'pe'), take)
            t2 = time.time()
            for j, v in enumerate(need):
                part = dict(m1=res['m1'], z=res['z'], **{key: np.asarray(res[key])[j * n:(j + 1) * n]
                                                         for key in PER_SEGMENT})
                arrays = plant.reduce(part, pop, plant.MDET)
                arrays.update(epoch_arrays(ep, rows, n, inj if v == 'inj' else None))
                arrays['k'] = sel[e]['k'] + np.arange(n)
                acc[v].append(arrays)
            shared = dict(rho_edges=np.asarray(res['rho_edges'], float),
                          glitch_pi_k=np.asarray(res.get('glitch_pi_k', np.nan), float),
                          glitch_window=np.asarray(res.get('glitch_window', np.nan), float))
            del res
            t_build += t1 - t0
            t_pe += t2 - t1
            n_rows += n * len(need)
        for v in need:
            parts = acc[v]
            arrays = {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if key != 'w_inf_cells'}
            arrays['w_inf_cells'] = np.unique(np.concatenate([p['w_inf_cells'] for p in parts]).reshape(-1, 2),
                                              axis=0).astype(np.int32)
            arrays.update(shared)
            info = dict(device=a.device, chunk=a.chunk, resident=a.resident, pid=os.getpid(), host=socket.gethostname(),
                        first=int(k0), end=int(k1), epochs=[int(e0), int(e1)], written=stamp(),
                        t_sim=t_build / max(n_rows, 1), t_pe=t_pe / max(n_rows, 1), t_plain=None)
            path = path_of(a.out, v, k0, k1)
            plant.write_batch(path, cfgs[v], info, arrays)
            written.append(path)
        rows_done += n_rows
        log('%s  epochs [%d, %d) k [%d, %d), %d of %d groups: %d rows (%s); building %.4f s, analysis %.4f s per '
            'row; %.0f s' % (stamp(), e0, e1, k0, k1, g + 1, len(todo), n_rows, ','.join(need),
                             t_build / max(n_rows, 1), t_pe / max(n_rows, 1), time.time() - began), flush=True)
    return written


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('store', help='folder of the redshift tables (bd_*.npz)')
    p.add_argument('device', choices=('gpu', 'cpu'))
    p.add_argument('segments', help='folder of the conditioned epochs, with manifest.txt')
    p.add_argument('--out', required=True, help='output folder: <out>/<variant>/plant_*.npz')
    p.add_argument('--variants', default=','.join(VARIANTS), help='comma-separated subset of %s' % ','.join(VARIANTS))
    p.add_argument('--nbar-inj', dest='nbar_inj', type=float, default=0.3, help='injection probability per segment')
    p.add_argument('--seed', type=int, default=1, help='0 <= seed < 2^20')
    p.add_argument('--part', default='0/1', help='i/n: the i-th of n contiguous ranges of epoch groups')
    p.add_argument('--epochs-per-file', dest='epochs_per_file', type=int, default=8)
    p.add_argument('--chunk', type=int, default=CHUNK, help='rows filtered at once, capped by the free GPU memory')
    p.add_argument('--ymin', type=float, default=None, help='ymin of pe_redshift (default plant.default_ymin())')
    p.add_argument('--resident', action='store_true', help='keep every table on the GPU (needs ~48 GB)')
    return p


def parse(argv=None):
    p = parser()
    a = p.parse_args(argv)
    a.variants = [v.strip() for v in a.variants.split(',') if v.strip()]
    if not a.variants or set(a.variants) - set(VARIANTS):
        p.error('--variants: a subset of %s' % ','.join(VARIANTS))
    try:
        i, n = (int(x) for x in a.part.split('/'))
    except ValueError:
        p.error('--part must be i/n')
    if not 0 <= i < n:
        p.error('--part i/n needs 0 <= i < n')
    a.part = (i, n)
    if not (0 <= a.nbar_inj <= 1 and 0 <= a.seed < 2 ** 20 and a.epochs_per_file >= 1):
        p.error('need 0 <= nbar-inj <= 1, 0 <= seed < 2^20, epochs-per-file >= 1')
    return p, a


def main(argv=None, factory=None):
    """factory: replaces the analysis with the tables (tests)"""
    p, a = parse(argv)
    if factory is None:
        if not glob.glob(os.path.join(a.store, 'bd_*.npz')):
            p.error('no bd_*.npz in %s' % a.store)
        loaded = {}

        def factory(seg):
            if 'store' not in loaded:
                from exmarg.search import load_redshift_store
                t = time.time()
                loaded['store'] = load_redshift_store(a.store, pinned=(a.device == 'gpu'), unbiased=True)
                print('%s  store loaded, %d nodes (%.0f s)' % (stamp(), loaded['store'].s.size, time.time() - t),
                      flush=True)
            return make_analysis(loaded['store'], a.device, a.resident, a.chunk,
                                 plant.default_ymin() if a.ymin is None else a.ymin)(seg)
    print('%s  %s, part %d/%d, variants %s, into %s' % (stamp(), a.segments, a.part[0], a.part[1], ','.join(a.variants),
                                                         a.out), flush=True)
    run(a, factory)
    return 0


if __name__ == '__main__':
    sys.exit(main())
