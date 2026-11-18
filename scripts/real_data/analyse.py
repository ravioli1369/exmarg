"""The conditioned week through the glitch-robust analysis, with its controls (docs/real_data.md, "Order of the analysis
and its criteria"), into files that scripts/validate/plant_fit.py fits.  Needs the redshift tables; meant for a GPU.

    python scripts/real_data/analyse.py tables/redshift gpu data/segments/O3b_v2 --out runs/week [--part 3/9]
                                        [--variants zero,pr,rev,slide+0.5,slide-1.75,inj[,slide+2.25,..]]
                                        [--nbar-inj 0.3] [--seed 1]
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
    slide<+-s>   any other time slide, named by its signed shift in seconds as typed (slide+2.25, slide-3.0): more
                 null data of the same week (summarize.py --extra-slides, slides_mean.py), not in the default
                 --variants.  The name must be slide, a sign and digits[.digits]; the shift a whole number of
                 matched-filter samples at least controls.MIN_SHIFT from a multiple of T (the error names the variant),
                 and no two slides, these two included, the same shift (as written or modulo T).  Merger stage: its own
                 pe_redshift call per epoch on the epoch's rows with the seed stream(seed, GPS of the epoch + 2^24 c,
                 'pe'), c = 2 |m| - [m < 0] for m samples of the shift modulo T (c = 0 for the variants above), so
                 its files do not depend on the other variants asked for; windows stage: in the one call
    inj          slide+0.5 plus mergers injected with probability --nbar-inj per segment, from the population of the
                 plant (plant.population(0.99119), plant.Sampler, plant.mergers, equal masses, no spin), the signal of
                 pp.simulate as the difference of two calls with the same seed, with y and with y = 0.  The injection
                 is added to the conditioned data, after the whitening and the equalization: unlike a real merger it
                 does not pass through the spectrum estimate (no gain error) and is not cut at a segment edge.  Its
                 optimal network SNR is computed with the epoch's sig2.
The random numbers of a segment are keyed by its GPS second (plant.stream(seed, gps, use)), those of pe_redshift by the
epoch's GPS and a fixed row per variant and segment, so every variant of every segment is the same whatever --part and
--variants are.  Merger stage: the pe_redshift calls of all the epochs of an output group (those described here and
under --glitch-model) are made at once, as the epochs of one pe_redshift_epochs call, each with its own bank, rows, seed
and glitch settings: the tables are copied to the GPU once per chunk of the group's segments instead of once per call,
and every array is that of the calls made one by one, bit for bit.  The chunk (--chunk, segments of the whole group) is
capped there by the free GPU memory (pe._fit_chunk).

Output: <out>/<variant>/plant_<first k>_<end k>.npz, one per group of --epochs-per-file epochs (groups cut at multiples
of it in the manifest; --part i/n takes the groups i n_group / n to (i + 1) n_group / n, contiguous in time), written
under a temporary name and renamed; a group is skipped when the files of every requested variant exist.  The files
follow plant.py: the reduction is plant.reduce on the same population (z >= 0.99119, detector-frame total mass 20 to
400) and ymin (plant.default_ymin), and the meta holds control = the variant, nbar = 0 for the four controls,
--nbar-inj for inj and null (unknown) for zero, glitch_truth false (no glitch is known in real data: glitch_I False,
rho_I and t0_I NaN), merger_truth 'injections' for inj and 'none' otherwise (merger False, m1 and z NaN, y and snr 0).
Per segment also: gps, gps_epoch, the flags event, analysable, loud_H, loud_L, dq_H, dq_L, inj_H, inj_L, the band
monitors monitor_H, monitor_L (n, 5) and the epoch's u_ratio_60.

Tables.  The merger stage reads the tables with the correction of store.unbias and takes Lambda linear between their
nodes and between the templates, as plant.py does (its docstring, Tables).  For comparisons only, --as-built reads the
tables as built and --interp log takes ln Lambda linear (the two together: the authors' method; with ln Lambda linear
unbias lowers the merger rate: 0.2543 against 0.2985 as built for mergers planted at nbar 0.30 in 16384 Gaussian
segments); --unbias names the default and changes nothing.  The campaign of every merger-stage run records the kind,
tables 'unbiased' or 'as built', and interp 'linear' (settings of the merger stage, plant.MERGER_KEYS), so that files
of two readings are never one campaign; a run with --interp log records no interp entry, as the files written before
2026-11-18, all of them read that way, whose campaign it continues.  Files written before 2026-11-17 record no tables
entry and were read with unbias: a run of today into their folders is refused as another campaign.  --mmap maps caches
of the kind the run reads (store_cache.py, as built with its --as-built) or, with --as-built, store folders.

The window model (docs/glitch_likelihood.md, Sec. 4a), in two stages:
    python scripts/real_data/analyse.py none gpu data/segments/O3b_v2 --out runs/week_w --stage windows
    python scripts/validate/windows_fit.py runs/week_w/<variant>                 # for every variant
    python scripts/real_data/analyse.py tables/redshift gpu data/segments/O3b_v2 --out runs/week_w --glitch-model auto
--stage windows (no tables; of the store argument only its header: the lags, for wnet, and the ladder of the merger
evidence below; 'none' writes neither, and the fits then apply the single-detector veto alone): the same selection,
variants, groups and k, through
pe.window_evidences(n_window=--n-window) of all the variants of an epoch in one call, into
<out>/<variant>/win_<first k>_<end k>.npz, the members of plant.py's win files (ln_window_H, ln_window_L float32 on the
bins that overlap [--window-rho-min, --window-rho-max], window_rho_edges, window_bin0, wmax_H, wmax_L, wnet) with the
per-segment arrays above; the meta is the campaign without the settings of the merger stage, with stage 'windows',
n_window and window_rho.  --glitch-model auto: each variant through pe_redshift with the model of
<out>/<variant>/glitch_model.json, fitted on its own single-detector data (the campaign_key of its files that of the
variant); --glitch-model <file>: that model for every variant (fitted on any variant of the same campaign: the keys
VARIANT_KEYS aside).  A model that was modified or has another n_window than --n-window is refused.  Each variant is
then one pe_redshift call per epoch (same seed and rows as above, so the same arrival times), and its files hold
plant.py's members of the window model; the meta records glitch_model 'windows', glitch_model_sha256 and n_window.

Unequal noise curves (docs/2026-11-15_unequal_noise_design.md):
    python scripts/real_data/analyse.py tables/w-0.60,tables/redshift_cache,tables/w-0.30 gpu data/segments/O3b_v2
        --out runs/week_u --unequal --glitch-model auto [--mmap] [--zmin-cut] [--zmin 0] [--asym-tol 0.05]
--unequal: every variant from the data of each detector on its own curve (strain.own_curves: the noise of step 5
regenerated and subtracted, the notched bins of the file), with sig2_H, sig2_L; the injections' SNR on those curves.
Both stages; recorded as noise 'own curves', a setting of the data (a glitch model fitted on equalized data is refused).
The store argument may list several folders (stores of distinct asym, or caches of them with --mmap,
scripts/real_data/store_cache.py): pe_redshift mixes the two that bracket each template's asymmetry; recorded as
tables_asym.  --asym-tol (default plant.ASYM_TOL = 0.05, with --unequal or several stores): a template outside the
stores' range by at most that is read from the end store (recorded as asym_tol; the number of such templates of each
segment's epoch in asym_outside).  --zmin (default 0.99119; 0: the whole population): the population of the reduction
and of the injections; for every variant but inj (whose injections are drawn from it) a setting of the merger stage
only, so the window models of a windows stage serve merger stages of any --zmin.  --zmin-cut: pe_redshift(zmin=--zmin),
the nodes that z >= zmin needs alone (the same results there).  Without these options nothing changes.

Signal gain (strain.signal_gain; src/exmarg/strain.py).  With estimated spectra the conditioning scales a signal in the
data of detector I, on its own curve, by kappa_I (0.97283 for the median of N = 32 and 66 neighbour blocks), a number of
the epoch and detector.  With --unequal the bank of each epoch takes these gains (the model's template of detector I is
kappa_I a; bank.MassBank, pe), and the injections of inj are multiplied by them, so that they are what a real signal is
in these data.  --no-gain: the model without them (a control; recorded as signal_gain none, the default as epoch; a
setting of the merger stage, plant.MERGER_KEYS: the window evidences do not depend on it; the merger evidence of a
windows stage run with the store does, being that of the model power kappa_I^2 u_I, so that stage's bank takes the
gains too and its merger_population records signal_gain).

Mergers in stage 1 (docs/2026-11-15_mergers_in_stage1.md).  --stage windows with the store folder in place of none also
writes ln_merger_H, ln_merger_L, the merger evidence of each window for the population of --zmin, GRID, MDET, ymin and
the ladder of the store (plant.py, "Mergers in stage 1"), and records merger_population.  A glitch model fitted with
mergers (windows_fit.py --nbar N > 0, on the variants zero and inj only) is refused for the variants pr, rev and the
slides, whose model is that without mergers (Sec. 3 of the note), and for a run of another population; the iteration of
the two stages on one variant:
    python scripts/real_data/analyse.py tables/redshift gpu data/segments/O3b_v2 --out runs/week_w --stage windows
    python scripts/validate/iterate.py runs/week_w/zero runs/week_iter --passes 2 --results zero -- python \
        scripts/real_data/analyse.py tables/redshift gpu data/segments/O3b_v2 --variants zero --glitch-model {model} \
        --out {out}
"""
import argparse
import hashlib
import json
import os
import re
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
SLIDE = re.compile(r'^slide([+-])([0-9]+(?:\.[0-9]+)?)$')
ZMIN = 0.99119
CHUNK = 4096
VERSION = 1
FLAGS = ('event', 'analysable', 'loud_H', 'loud_L', 'dq_H', 'dq_L', 'inj_H', 'inj_L', 'monitor_H', 'monitor_L')
# the meta entries that a variant defines (summarize.VARIANT_KEYS without the model's SHA-256)
VARIANT_KEYS = ('variant', 'control', 'nbar', 'shift', 'merger_truth', 'truth_note')
MODEL = 'glitch_model.json'


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


def manifest_digest(seg_dir):
    """SHA-256 of the epoch rows of manifest.txt (the lines that are not comments): the header carries the time of the
    run of condition.py and would change the hash of an unchanged week.  The manifests of weeks analysed before this
    (whole file hashed) have another hash for the same rows; summarize.py only needs them to agree with each other"""
    with open(os.path.join(seg_dir, 'manifest.txt'), 'rb') as fh:
        rows = [line for line in fh.read().splitlines(keepends=True) if line.strip() and not line.startswith(b'#')]
    return hashlib.sha256(b''.join(rows)).hexdigest()


def selection(seg_dir):
    """every epoch of the manifest: dicts name, gps_epoch, n_segment, n_selected, k (the first k of its selected
    segments).  The epochs must carry one settings hash (condition.py), or none of them (conditioned before it was
    stored); a mixture stops the run"""
    out, k, hashes = [], 0, {}
    for name, gps in manifest(seg_dir):
        with np.load(os.path.join(seg_dir, name)) as z:
            sel = selected(z)
            hashes[name] = str(z['settings_hash']) if 'settings_hash' in z.files else None
        out.append(dict(name=name, gps_epoch=gps, n_segment=int(sel.size), n_selected=int(sel.sum()), k=k))
        k += int(sel.sum())
    try:
        strain.one_settings_hash(hashes)
    except ValueError as err:
        raise SystemExit('%s: %s' % (seg_dir, err))
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


def slide_shift(name):
    """the shift in s of a variant named slide<+ or ->digits[.digits] (ValueError otherwise); the built-in names too"""
    m = SLIDE.match(name)
    shift = float(m.group(1) + m.group(2)) if m else np.nan
    if not np.isfinite(shift):
        raise ValueError("%r is not a time slide: the form is slide<+ or -><seconds>, as slide+2.25 or slide-3.0" % name)
    return shift


def is_slide(name):
    return name in SHIFTS or SLIDE.match(name) is not None


def slide(seg, name):
    """the time slide of the variant `name` of seg; the error of a shift that is not whole samples or too close to a
    multiple of T names the variant"""
    shift = slide_shift(name)
    try:
        whole_samples(seg, shift)
        return controls.time_slide(seg, shift)
    except ValueError as err:
        raise ValueError('%s: %s' % (name, err)) from None


def slide_samples(seg, names):
    """{name: the shift in matched-filter samples modulo the segment} of the slides names (and the built-in ones); a
    shift equal to another's modulo T (the same data under two names) is a ValueError naming both"""
    n = int(round(float(seg['T']) / controls.time_step(seg)))
    out = {}
    for v in list(SHIFTS) + [v for v in names if v not in SHIFTS and is_slide(v)]:
        try:
            m = whole_samples(seg, slide_shift(v)) % n
        except ValueError as err:
            raise ValueError('%s: %s' % (v, err)) from None
        twin = next((u for u, x in out.items() if x == m), None)
        if twin is not None:
            raise ValueError('%s and %s are the same shift modulo the %g s segment' % (twin, v, float(seg['T'])))
        out[v] = m
    return out


def slide_samples_names(names):
    """ValueError when two slides of names (or one and a built-in one) have the same shift, as written (the segment
    length is not known yet: slide_samples checks the shifts modulo T)"""
    seen = {SHIFTS[v]: v for v in SHIFTS}
    for v in names:
        if is_slide(v) and seen.setdefault(slide_shift(v), v) != v:
            raise ValueError('%s and %s are the same shift' % (seen[slide_shift(v)], v))


def extra_seed(seed, gps_epoch, seg, name):
    """the seed of pe_redshift for a slide that is not a built-in one: stream(seed, GPS + 2^24 c, 'pe'), c = 2 |m| - [m < 0]
    for its m matched-filter samples modulo T, in -T / 2 .. T / 2 (never the built-ins' seeds, c = 0, nor that of another
    shift, for epochs less than 2^24 s apart); the row of a segment is added by pe_redshift"""
    n = int(round(float(seg['T']) / controls.time_step(seg)))
    m = (whole_samples(seg, slide_shift(name)) + n // 2) % n - n // 2
    return plant.stream(seed, int(round(gps_epoch)) + (2 * abs(m) - (m < 0)) * 2 ** 24, 'pe')


def injections(cfg, ep, rows, sampler, sig2=None, gain=None):
    """the mergers injected into the segments rows of the epoch ep: (signal {I: (n, n_freq) complex}, truth dict of
    (n,) arrays merger, m1, z, y, snr), presence and (m1, z) from plant.mergers keyed by the GPS second, the signal of
    pp.simulate (seed plant.stream(seed, gps, 'noise')) with y less that with y = 0, times gain[I] when given (the
    signal gain of the epoch, strain.signal_gain: a signal added to conditioned data must carry it to be what a real
    signal is in them; simulated segments drawn in the frequency domain have gain 1), snr the optimal network SNR of
    that signal with the noise curve of each detector, sig2 {I: (n_freq,)} (default the epoch's sig2 for both)"""
    sig2 = {I: ep['sig2'] for I in 'HL'} if sig2 is None else sig2
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
            if gain is not None and gain[I] != 1.0:
                h[I][j] = gain[I] * h[I][j]
        snr[j] = np.sqrt(sum(2.0 * np.sum(np.abs(h[I][j]) ** 2 / sig2[I]) for I in 'HL'))
    truth = dict(merger=present, m1=np.where(present, m1, np.nan), z=np.where(present, z, np.nan), y=y, snr=snr)
    return h, truth


def variants(cfg, ep, names, sampler, own=False, gain=False):
    """the selected segments of the epoch ep in each variant of names: ({variant: segments dict}, rows (n,) their
    indices in the epoch, injection truth or None).  own: the data of each detector on its own curve (strain.own_curves,
    sig2_H and sig2_L), else the equalized data on the epoch's one curve sig2.  With own, the injections of inj carry
    the epoch's signal gain kappa_I (strain.signal_gain: the amplitude a real signal has in these data), and with gain
    also every variant's dict holds it under 'gain' for the bank of the analysis (make_analysis)"""
    rows = np.flatnonzero(selected(ep))
    base = dict(f=ep['f'], nu=ep['nu'], T=ep['T'], fmin=ep['fmin'], fmax=ep['fmax'])
    kappa = None
    if own:
        o = strain.own_curves(ep)
        base.update(d_H=o['H'][0][rows], d_L=o['L'][0][rows], sig2_H=o['H'][1], sig2_L=o['L'][1])
        kappa = strain.signal_gain(ep) if (gain or 'inj' in names) else None
        if gain:
            base['gain'] = dict(kappa)
    elif 'sig2' not in ep:
        raise ValueError('the epoch has sig2_H and sig2_L: the published tables need one noise curve (equal_noise), '
                         'or analyse it with --unequal')
    else:
        base.update(d_H=np.asarray(ep['d_H'])[rows].astype(complex), d_L=np.asarray(ep['d_L'])[rows].astype(complex),
                    sig2=ep['sig2'])
    out, inj = {}, None
    if any(v not in VARIANTS and is_slide(v) for v in names):
        slide_samples(base, names)
    for v in names:
        if v == 'zero':
            out[v] = base
        elif v == 'pr':
            out[v] = controls.phase_randomize(base, [int(cfg['seed']), int(round(float(ep['gps_epoch']))), 1])
        elif v == 'rev':
            out[v] = controls.time_reverse(base)
        elif is_slide(v):
            out[v] = slide(base, v)
        elif v == 'inj':
            whole_samples(base, SHIFTS['slide+0.5'])
            slid = controls.time_slide(base, SHIFTS['slide+0.5'])
            h, inj = injections(cfg, ep, rows, sampler, {I: base.get('sig2_' + I, base.get('sig2')) for I in 'HL'},
                                kappa)
            out[v] = dict(slid, d_H=slid['d_H'] + h['H'], d_L=slid['d_L'] + h['L'])
        else:
            raise ValueError('unknown variant %r' % v)
    return out, rows, inj


def make_analysis(store, device, resident=False, chunk=CHUNK, ymin=None, zmin=None, asym_tol=None,
                  interp=plant.INTERP[0]):
    """factory(segments) builds the bank of plant.MASSES on the epoch's noise curves, with the signal gain the segments
    carry (segments['gain'], set by variants with gain; gains 1 without it), and returns pe(segments, seed, rows,
    glitch=True) -> the pe_redshift dict (glitch=True, or the dict of the window model) of those rows of the segments,
    on plant.GRID; store one store or a list (the mixture), zmin and asym_tol those of pe_redshift (each given only when
    not None), interp that of pe_redshift.  pe.epoch(segments, seed, rows, glitch=True) is the same call as an epoch of
    pe_redshift_epochs, and pe.batch(epochs) analyses such epochs (of any epoch's bank) in one pass over the tables:
    [the dict of each]"""
    from exmarg.search import mass_bank, pe_redshift, pe_redshift_epochs
    cut = {k: v for k, v in (('zmin', zmin), ('asym_tol', asym_tol)) if v is not None}
    settings = dict(chunk=chunk, resident=resident, ymin=ymin, mdet_range=plant.MDET, interp=interp, **cut,
                    **plant.GRID)

    def factory(seg0):
        bank = mass_bank(plant.MASSES, seg0, device, gain=seg0.get('gain'))

        def pe(seg, seed, rows, glitch=True):
            return pe_redshift(seg, bank, store, rows=rows, seed=seed, glitch=glitch, **settings)
        pe.epoch = lambda seg, seed, rows, glitch=True: dict(segments=seg, bank=bank, rows=rows, seed=seed, glitch=glitch)
        pe.batch = lambda epochs: pe_redshift_epochs(epochs, store, **settings)
        return pe
    return factory


def analyse_calls(calls):
    """the pe dicts of the calls [(pe, args)] of a group of epochs, in order: in one pass over the tables (pe.batch of
    every call's pe.epoch(*args)) when every pe offers it, else call by call, pe(*args)"""
    if calls and all(hasattr(pe, 'batch') and hasattr(pe, 'epoch') for pe, _ in calls):
        return calls[0][0].batch([pe.epoch(*args) for pe, args in calls])
    return [pe(*args) for pe, args in calls]


def make_windows(device, n_window=plant.N_WINDOW, chunk=plant.WINDOW_CHUNK, population=None, lags=None):
    """stage 1 of the window model: factory(segments) builds the bank of plant.MASSES on the epoch's noise curve and
    returns win(segments, rows) -> the dict of pe.window_evidences of those rows (no tables), with the merger evidence
    of the population (plant.merger_population(settings)) when given; the bank then takes the signal gain the segments
    carry (segments['gain'], variants with gain), which the merger evidence depends on and the window evidences do not;
    with lags (those of the tables) also wnet, which depends on neither"""
    from exmarg.search import mass_bank
    from exmarg.search import pe as pe_module
    extra = dict({} if population is None else dict(population=population), **({} if lags is None else
                                                                                dict(lags=lags)))

    def factory(seg0):
        bank = mass_bank(plant.MASSES, seg0, device, gain=seg0.get('gain') if population is not None else None)

        def win(seg, rows):
            return pe_module.window_evidences(seg, bank, rows=rows, chunk=chunk, n_window=n_window, **extra)
        return win
    return factory


PER_SEGMENT = plant.KEYS + ('ln_glitch_H', 'ln_glitch_L', 'wmax_H', 'wmax_L')
WINDOW_KEYS = ('ln_window_H', 'ln_window_L', 'wmax_H', 'wmax_L')
MERGER_KEYS = ('ln_merger_H', 'ln_merger_L')          # the merger evidence of the windows stage (with the store)
WITH_MERGERS = ('zero', 'inj')                          # the variants whose glitch model may be fitted with mergers
NET = ('wnet',)                 # sliced with the keys above when the result has it (pe_redshift, window_evidences(lags))


def _with_net(keys, res):
    return keys + tuple(k for k in NET if k in res)


def merger_only(variant):
    """the campaign keys of a variant that are settings of the merger stage although both stages record them: zmin, the
    population of the analysis (the data of every variant but inj, whose injections are drawn from it, do not depend
    on it)"""
    return () if variant == 'inj' else ('zmin',)


def load_models(a, cfgs):
    """{variant: (the glitch settings of pe_redshift, the model's SHA-256, the nbar of its stage-1 fit)} of
    --glitch-model for the variants of cfgs (their campaigns without the model): 'auto', each variant's own
    <out>/<variant>/glitch_model.json fitted on its campaign (merger_only aside); a file, that model for every variant,
    fitted on a variant of the same campaign (VARIANT_KEYS and zmin aside).  plant.glitch_model refuses a modified
    file, another n_window, another campaign, or a model fitted with mergers (nbar > 0) of another population than
    that of this run; and a model fitted with mergers is refused for the time-slid, time-reversed and phase-randomized
    variants, whose glitch model is that without mergers (docs/2026-11-15_mergers_in_stage1.md, Sec. 3)"""
    s = plant.store_ladder(a.store)
    out = {}
    for v, cfg in cfgs.items():
        path = os.path.join(a.out, v, MODEL) if a.glitch_model == 'auto' else a.glitch_model
        nbar = plant.model_nbar(plant.read_model(path)[0]) if os.path.exists(path) else 0.0
        if nbar > 0 and v not in WITH_MERGERS:
            raise SystemExit('%s: a model fitted with mergers (nbar %g) is not for the variant %s: the time-slid, '
                             'time-reversed and phase-randomized data keep the model without mergers' % (path, nbar, v))
        pop = None if s is None else plant.population_settings(a.zmin, cfg['ymin'], s, cfg.get('signal_gain'))
        ignore = merger_only(v) if a.glitch_model == 'auto' else VARIANT_KEYS + ('zmin',)
        out[v] = plant.glitch_model(path, cfg, a.n_window, ignore=ignore, population=pop) + (nbar,)
    return out


def campaign(a, seg_dir, variant, sha=None, model_nbar=0.0):
    """the settings of one variant's files (the same in every file of it): of the merger stage, with the window model
    of SHA-256 sha when given (fitted at the merger rate model_nbar), or of the windows stage (a.stage)"""
    digest = manifest_digest(seg_dir)
    nbar = None if variant == 'zero' else a.nbar_inj if variant == 'inj' else 0.0
    cfg = dict(version=VERSION, source='strain', segments=os.path.basename(os.path.normpath(seg_dir)),
               manifest_sha256=digest, selection='usable epoch, analysable, no catalogued event', variant=variant,
               control=variant, seed=a.seed, nbar=nbar, nbar_inj=a.nbar_inj, shift=SHIFTS['slide+0.5'] if variant ==
               'inj' else slide_shift(variant) if is_slide(variant) else None,
               glitch_truth=False, merger_truth='injections' if variant == 'inj' else 'none',
               truth_note='real data: no merger and no glitch is known; merger, glitch_I False, rho_I, t0_I NaN'
                          + (' except the injected mergers' if variant == 'inj' else ''),
               eps_H=None, eps_L=None, glitch_kind=None, gamma=None, rho_min=None, rho_max=None,
               epochs_per_file=a.epochs_per_file, zmin=a.zmin, mdet_range=plant.MDET,
               bank=(float(plant.MASSES[0]), float(plant.MASSES[-1]), plant.MASSES.size), grid=plant.GRID,
               ymin=plant.default_ymin() if a.ymin is None else float(a.ymin), n_loud=1024, n_random=1024,
               glitch_settings='pe default')
    if a.unequal:
        cfg['noise'] = 'own curves'
        cfg['signal_gain'] = 'none' if a.no_gain else 'epoch'
    cfg.update(plant.store_settings(a))
    cfg = plant.stage_settings(cfg, a)
    if sha is not None:
        cfg = plant.with_model(cfg, sha, a.n_window, model_nbar)
    return json.loads(json.dumps(cfg))


def path_of(out, variant, k0, k1, prefix='plant'):
    return os.path.join(out, variant, '%s_%010d_%010d.npz' % (prefix, k0, k1))


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


def merger_shared(res):
    """the members of a plant file that are not per segment"""
    return dict(rho_edges=np.asarray(res['rho_edges'], float),
                glitch_pi_k=np.asarray(res.get('glitch_pi_k', np.nan), float),
                glitch_window=np.asarray(res.get('glitch_window', np.nan), float))


def merger_calls(seg, seed, rows, slots, n, need, models, alone):
    """the pe calls of one epoch in the merger stage: [(args of pe, targets)], targets [(variant, the slice of the
    dict's segments that is the variant's, or None: all of them)].  With a model per variant (models) one call per
    variant, at its slots or (alone) on its own epoch-sized stack; else one call for the variants with slots, each the
    slice of its slots, and one per variant without slots (alone), with the default glitch settings"""
    if models:
        return [((seg, seed, slots[v], models[v][0]) if v in slots else alone(v) + (rows, models[v][0]), [(v, None)])
                for v in need]
    out = []
    if slots:
        take = np.concatenate([slots[v] for v in slots])
        out.append(((seg, seed, take), [(v, slice(j * n, (j + 1) * n)) for j, v in enumerate(slots)]))
    return out + [(alone(v) + (rows,), [(v, None)]) for v in need if v not in slots]


def merger_arrays(res, targets, pop):
    """{variant: the per-segment arrays of plant.reduce} of one pe dict for its targets (merger_calls)"""
    out = {}
    for v, part in targets:
        if part is None:
            out[v] = plant.reduce(res, pop, plant.MDET)
            continue
        per_call = {k: res[k] for k in ('asym_outside',) if k in res}
        out[v] = plant.reduce(dict(m1=res['m1'], z=res['z'], **per_call,
                                   **{key: np.asarray(res[key])[part] for key in _with_net(PER_SEGMENT, res)}), pop,
                             plant.MDET)
    return out


def run(a, factory, log=print):
    """the groups of epochs of this part, every requested variant; returns the paths written"""
    names = [v for v in VARIANTS if v in a.variants] + [v for v in a.variants if v not in VARIANTS]
    sel = selection(a.segments)
    os.makedirs(a.out, exist_ok=True)
    record_selection(a.out, sel)
    windows = a.stage == 'windows'
    merger_evidence = windows and plant.store_ladder(a.store) is not None     # the windows stage with the store
    prefix = plant.PREFIX[a.stage]
    cfgs = {v: campaign(a, a.segments, v) for v in names}
    models = load_models(a, cfgs) if a.glitch_model else None
    if models:
        cfgs = {v: campaign(a, a.segments, v, models[v][1], models[v][2]) for v in names}
    for v in names:
        os.makedirs(os.path.join(a.out, v), exist_ok=True)
        plant.check_folder(os.path.join(a.out, v), cfgs[v], ignore=merger_only(v))
    bins = plant.window_bins(a.window_rho_min, a.window_rho_max) if windows else None
    pop = plant.population(a.zmin)
    sampler = plant.Sampler(pop, plant.MDET)
    params = dict(seed=a.seed, nbar_inj=a.nbar_inj)
    written, began, rows_done = [], time.time(), 0
    todo = groups(len(sel), a.epochs_per_file, a.part)
    for g, (e0, e1) in enumerate(todo):
        k0, k1 = sel[e0]['k'], sel[e1 - 1]['k'] + sel[e1 - 1]['n_selected']
        need = [v for v in names if k1 > k0 and not os.path.exists(path_of(a.out, v, k0, k1, prefix))]
        if not need:
            log('%s  epochs [%d, %d) k [%d, %d): %s' % (stamp(), e0, e1, k0, k1, 'present, skipped' if k1 > k0 else
                                                          'no segment selected'), flush=True)
            continue
        acc = {v: [] for v in need}
        t_build = t_pe = 0.0
        n_rows = 0
        calls, books = [], []           # merger stage: the group's pe calls; per epoch its bookkeeping and the targets
        for e in range(e0, e1):
            if sel[e]['n_selected'] == 0:
                continue
            t0 = time.time()
            ep = strain.read_epoch(os.path.join(a.segments, sel[e]['name']))
            var, rows, inj = variants(params, ep, need, sampler, a.unequal,
                                      a.unequal and not a.no_gain and (not windows or merger_evidence))
            n, ns = rows.size, int(np.asarray(ep['gps']).size)
            extra = [v for v in need if v not in VARIANTS]
            # the variants of VARIANTS at the fixed slots VARIANTS.index(v) ns + row of one stack; the windows stage has
            # no random numbers and puts the others after them; the merger stage gives each its own call (alone)
            slots = {v: (VARIANTS.index(v) if v in VARIANTS else len(VARIANTS) + extra.index(v)) * ns + rows
                     for v in need if windows or v in VARIANTS}
            stack = {I: np.zeros(((len(VARIANTS) + (len(extra) if windows else 0)) * ns, np.size(ep['nu'])), complex)
                     for I in 'HL'}
            for v in slots:
                for I in 'HL':
                    stack[I][slots[v]] = var[v]['d_' + I]
            seg = dict(var[need[0]], d_H=stack['H'], d_L=stack['L'])
            take = np.concatenate([slots[v] for v in slots]) if slots else rows
            seed = plant.stream(a.seed, int(round(float(ep['gps_epoch']))), 'pe')
            t1 = time.time()
            pe = factory(seg)

            def alone(v):
                """(segments, seed) of a variant that has no slot: its rows of an epoch-sized stack"""
                pad = {I: np.zeros((ns, np.size(ep['nu'])), complex) for I in 'HL'}
                for I in 'HL':
                    pad[I][rows] = var[v]['d_' + I]
                return dict(var[v], d_H=pad['H'], d_L=pad['L']), extra_seed(a.seed, float(ep['gps_epoch']), var[v], v)
            book = {v: dict(epoch_arrays(ep, rows, n, inj if v == 'inj' else None), k=sel[e]['k'] + np.arange(n))
                    for v in need}
            if windows:                                 # every variant in one call, sliced
                res = pe(seg, take)
                arrays = {v: plant.reduce_windows(dict(rho_edges=res['rho_edges'], **{
                    key: np.asarray(res[key])[j * n:(j + 1) * n]
                    for key in _with_net(WINDOW_KEYS, res) + MERGER_KEYS if key in res}), bins)
                    for j, v in enumerate(need)}
                shared = plant.window_shared(bins)
                for v in need:
                    acc[v].append(dict(arrays[v], **book[v]))
                del res
            else:                                       # the calls of every epoch of the group, made below at once
                c = merger_calls(seg, seed, rows, slots, n, need, models, alone)
                calls += [(pe, args) for args, _ in c]
                books.append((book, [targets for _, targets in c]))
            t_build += t1 - t0
            t_pe += time.time() - t1
            n_rows += n * len(need)
        if calls:                                       # merger stage: one pass over the tables for the whole group
            t1 = time.time()
            results = iter(analyse_calls(calls))
            for book, targets in books:
                arrays = {}
                for tg in targets:
                    res = next(results)
                    arrays.update(merger_arrays(res, tg, pop))
                for v in need:
                    acc[v].append(dict(arrays[v], **book[v]))
            shared = merger_shared(res)
            del res, results, calls
            t_pe += time.time() - t1
        for v in need:
            parts = acc[v]
            arrays = {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if key != 'w_inf_cells'}
            if not windows:
                arrays['w_inf_cells'] = np.unique(np.concatenate([p['w_inf_cells'] for p in parts]).reshape(-1, 2),
                                                  axis=0).astype(np.int32)
            arrays.update(shared)
            info = dict(device=a.device, chunk=a.chunk, resident=a.resident, pid=os.getpid(), host=socket.gethostname(),
                        first=int(k0), end=int(k1), epochs=[int(e0), int(e1)], written=stamp(),
                        t_sim=t_build / max(n_rows, 1), t_pe=t_pe / max(n_rows, 1), t_plain=None)
            path = path_of(a.out, v, k0, k1, prefix)
            plant.write_batch(path, cfgs[v], info, arrays)
            written.append(path)
        rows_done += n_rows
        log('%s  epochs [%d, %d) k [%d, %d), %d of %d groups: %d rows (%s); building %.4f s, analysis %.4f s per '
            'row; %.0f s' % (stamp(), e0, e1, k0, k1, g + 1, len(todo), n_rows, ','.join(need),
                             t_build / max(n_rows, 1), t_pe / max(n_rows, 1), time.time() - began), flush=True)
    return written


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('store', help='folder of the redshift tables (bd_*.npz) or a cache of them (store_cache.py); or '
                                 'several, comma separated, of distinct asym: the mixture of the two that bracket each '
                                 'template')
    p.add_argument('device', choices=('gpu', 'cpu'))
    p.add_argument('segments', help='folder of the conditioned epochs, with manifest.txt')
    p.add_argument('--out', required=True, help='output folder: <out>/<variant>/plant_*.npz')
    p.add_argument('--variants', default=','.join(VARIANTS),
                   help='comma-separated subset of %s, and any other slide<+ or ->seconds (slide+2.25, slide-3.0)'
                        % ','.join(VARIANTS))
    p.add_argument('--nbar-inj', dest='nbar_inj', type=float, default=0.3, help='injection probability per segment')
    p.add_argument('--seed', type=int, default=1, help='0 <= seed < 2^20')
    p.add_argument('--part', default='0/1', help='i/n: the i-th of n contiguous ranges of epoch groups')
    p.add_argument('--epochs-per-file', dest='epochs_per_file', type=int, default=8)
    p.add_argument('--chunk', type=int, default=None,
                   help='rows filtered at once, capped by the free GPU memory (default %d); windows stage: by '
                        'window_evidences (default %d)' % (CHUNK, plant.WINDOW_CHUNK))
    p.add_argument('--ymin', type=float, default=None, help='ymin of pe_redshift (default plant.default_ymin())')
    p.add_argument('--resident', action='store_true', help='keep every table on the GPU (needs ~48 GB)')
    p.add_argument('--stage', choices=plant.STAGES, default='merger',
                   help='merger: pe_redshift (default); windows: stage 1 of the window model, win_*.npz (no tables)')
    p.add_argument('--n-window', type=int, default=plant.N_WINDOW, help='windows of a segment (window model)')
    p.add_argument('--window-rho-min', type=float, default=plant.WINDOW_RHO[0],
                   help='windows stage: keep the amplitude bins that overlap [window-rho-min, window-rho-max]')
    p.add_argument('--window-rho-max', type=float, default=plant.WINDOW_RHO[1])
    p.add_argument('--glitch-model', default=None, metavar='auto|JSON',
                   help='merger stage with the window model: auto, <out>/<variant>/glitch_model.json for each variant; '
                        'or one file for every variant')
    p.add_argument('--zmin', type=float, default=ZMIN,
                   help='smallest redshift of the population of the analysis and of the injections (default %g; 0: the '
                        'whole population); a setting of the merger stage except for inj' % ZMIN)
    p.add_argument('--zmin-cut', action='store_true',
                   help='pe_redshift reads only the nodes the redshifts z >= --zmin need (the same results there)')
    p.add_argument('--unequal', action='store_true',
                   help='each detector on its own noise curve (strain.own_curves: no step 5); needs stores that cover '
                        'the u_H/u_L of the templates.  The model carries the signal gain of each epoch '
                        '(strain.signal_gain), and so do the injections of inj')
    p.add_argument('--no-gain', action='store_true',
                   help='with --unequal: the model without the signal gain (gains 1; a control: the injections of inj '
                        'still carry it); recorded as signal_gain none')
    p.add_argument('--asym-tol', type=float, default=plant.ASYM_TOL,
                   help='with --unequal or several stores: a template whose asymmetry lies outside the stores\' range by at '
                        'most this is read from the end store (default %g; recorded, and the number of such templates '
                        'per segment in asym_outside)' % plant.ASYM_TOL)
    p.add_argument('--as-built', dest='as_built', action='store_true',
                   help='merger stage: read the tables as built, without the unbias correction (store.unbias), for '
                        'comparisons; recorded in the campaign as tables as built (default: unbiased)')
    p.add_argument('--unbias', action='store_true',
                   help='merger stage: the tables with the unbias correction, the default: changes nothing (kept so '
                        'that older command lines run; not with --as-built)')
    p.add_argument('--interp', choices=plant.INTERP, default=plant.INTERP[0],
                   help='merger stage, between the nodes of the tables and between the templates: linear, Lambda '
                        'linear (default), or log, ln Lambda linear (the authors\' code, for comparisons: it lowers '
                        'the noise mean of B); the campaign records interp linear, and no entry with log')
    p.add_argument('--mmap', action='store_true',
                   help='map the tables from disk instead of reading them: caches of the kind the run reads '
                        '(store_cache.py, unbiased by default, as built with --as-built) or, with --as-built, store '
                        'folders; the jobs of a machine share them')
    return p


def parse(argv=None):
    p = parser()
    a = p.parse_args(argv)
    a.variants = list(dict.fromkeys(v.strip() for v in a.variants.split(',') if v.strip()))
    bad = [v for v in a.variants if v not in VARIANTS and SLIDE.match(v) is None]
    if not a.variants or bad:
        p.error('--variants: a subset of %s, and slides named slide<+ or ->seconds (slide+2.25); not %s'
                % (','.join(VARIANTS), ','.join(bad)))
    try:
        slide_samples_names(a.variants)
    except ValueError as err:
        p.error('--variants: %s' % err)
    if a.chunk is None:
        a.chunk = plant.WINDOW_CHUNK if a.stage == 'windows' else CHUNK
    if a.stage == 'windows' and (a.glitch_model or a.zmin_cut or a.mmap or a.unbias or a.as_built
                                 or a.interp != plant.INTERP[0]):
        p.error('--glitch-model, --zmin-cut, --mmap, --unbias, --as-built and --interp are of the merger stage')
    if a.unbias and a.as_built:
        p.error('--as-built reads the tables as built: not with --unbias')
    if a.stage == 'windows' and a.ymin is not None and plant.store_ladder(a.store) is None:
        p.error('--ymin with the windows stage sets the population of its merger evidence: give the store')
    if not a.zmin >= 0:
        p.error('need zmin >= 0')
    if a.no_gain and not a.unequal:
        p.error('--no-gain is an option of --unequal (the equalized data have no gain model)')
    if not a.asym_tol >= 0:
        p.error('need asym-tol >= 0')
    if not (0 <= a.window_rho_min < a.window_rho_max):
        p.error('need 0 <= window-rho-min < window-rho-max')
    if a.n_window < 1:
        p.error('need n-window >= 1')
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
    if factory is None and a.stage == 'windows':
        s = plant.store_ladder(a.store)
        pop = None if s is None else plant.merger_population(plant.population_settings(
            a.zmin, plant.default_ymin() if a.ymin is None else a.ymin, s))
        factory = make_windows(a.device, a.n_window, a.chunk, pop, plant.store_lags(a.store))
    elif factory is None:
        for d in a.store.split(','):
            problem = plant.store_problem(d, not a.as_built, a.mmap)
            if problem:
                p.error(problem)
        loaded = {}

        def factory(seg):
            if 'store' not in loaded:
                loaded['store'] = plant.load_stores(a)
            return make_analysis(loaded['store'], a.device, a.resident, a.chunk,
                                 plant.default_ymin() if a.ymin is None else a.ymin,
                                 a.zmin if a.zmin_cut else None, plant.asym_tol(a), a.interp)(seg)
    print('%s  %s, part %d/%d, variants %s, into %s; stage %s%s%s, zmin %g%s' % (
        stamp(), a.segments, a.part[0], a.part[1], ','.join(a.variants), a.out, a.stage,
        ', glitch model %s' % a.glitch_model if a.glitch_model else '', ', own noise curves' if a.unequal else '',
        a.zmin, ' (cut)' if a.zmin_cut else ''), flush=True)
    run(a, factory)
    return 0


if __name__ == '__main__':
    sys.exit(main())
