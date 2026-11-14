"""Noise diagnostics of real strain for exmarg: which lines to notch, and how Gaussian the conditioned segments are.

    python scripts/real_data/diagnose.py lines <gwosc_dir> <notches.txt> [--chunks 8] [--seconds 3600] [--ratio 100]
                                         [--published O3H1lines.csv O3L1lines.csv] [--save spectra.npz] [--workers 16]
    python scripts/real_data/diagnose.py lines --load spectra.npz <notches.txt> [the same options]
    python scripts/real_data/diagnose.py segments <segments_dir> [--subset 4000] [--seed 1] [--workers 16]

lines.  From --chunks chunks of --seconds of coincident good time spread evenly over the GWOSC folder
(condition.timeline), the median of the Hann periodograms of --block s (high-passed at 12 Hz first) of each chunk and
detector.  r is its ratio to a running median of the log spectrum over --smooth Hz, R the median of r over the
chunks, Rmax its largest value.  A feature is a group of bins in --band where R > --detect or r > --detect in at
least --persist chunks; it is notched when its Rmax reaches --ratio.  The notch covers the bins around it with
Rmax >= --extent, widened on each side by the larger of (the 0.25 Hz notch taper of strain.response, inside which
R is not 0, plus 1/T, the main lobe of a T s rectangular block) and the distance beyond which the line, seen through
the epoch window of strain.whiten ((N + 1) T + 2 pad s, sin^2 ramps of pad / 2 s), leaks less than --leak of the
floor.  Notches of H1 and L1 closer than --merge Hz are merged: one list for both.  The file written has one 'lo hi'
line per notch with the feature it covers, and a header with the rule, the cost (fraction of the bins of fmin..fmax
removed; fraction of the template power u = sum a^2 / sig2 lost for equal-mass templates of total mass 20, 60 and
200, with the measured spectra of each detector and their maximum) and the features left in.  --published matches
the features to the O3 line lists of LIGO-T2100200 (CSV, one per detector).

segments.  Over the epoch files of condition.py (segments flagged with a catalogued event left out), per detector and
for three sets of segments: all; those over which BURST_CAT2 passes (the dq bits of the files); and the quiet ones,
whose band monitors are all below 1 + --quiet standard deviations of Gaussian noise (a few loud glitches dominate
every mean over all segments):
  - the whitened power per bin, mean over segments, against 1 with its sampling error sqrt(V / n), V the variance
    of |e|^2 for the estimator (2 E[1/(c g)^2] - 1); its scatter over the bins of each band, the bins beyond 4
    sigma; the correlation of neighbouring bins and of H with L
  - the whitened variance monitors of each segment: median, 16-84 percent half-width, mean, rms, fractions outside
    1 +- 5 and 10 percent, rank correlation of neighbouring segments (drift)
  - the loss of template power u by step 5 (equalize) per epoch: 1 - u_common / u_own for each template
  - on a random subset of --subset segments, the matched filter of exmarg.search.mass_bank (CPU) for equal-mass
    templates of total mass 20, 60, 200: w = |z|^2 / u at every time (mean, survival against exp(-w)), the fraction
    of segments whose largest w exceeds 12.5, 18, 32, 50 (single-detector SNR sqrt(2 w) = 5, 6, 8, 10) and how many
    of those fail BURST_CAT2, and how often the loudest times of H1 and L1 fall within 15 ms of each other.
The data are analysed before step 5: the noise of equalize is regenerated from the seed and GPS time of each epoch
and subtracted (checked against the stored monitors), so that each detector is seen on its own curve; the matched
filter is also run on the equalized data.  Every number has a reference from stationary Gaussian noise passed
through the same estimator (N + 1 exponential periodograms per bin, leave-one-out median, the same correction), on
the same bins and templates.  Prints a report and writes <segments_dir>/diagnostics.npz (--out).
"""
import argparse
import csv
import glob
import importlib.util
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
from scipy import ndimage, special

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exmarg import strain

if 'exmarg_condition' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('exmarg_condition', os.path.join(HERE, 'condition.py'))
    sys.modules['exmarg_condition'] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules['exmarg_condition'])
cond = sys.modules['exmarg_condition']

MASSES = (20.0, 60.0, 200.0)
W_THRESHOLDS = (8.0, 12.5, 18.0, 32.0, 50.0)
BANDS = ((20.0, 22.0), (22.0, 40.0), (40.0, 80.0), (80.0, 160.0), (160.0, 320.0), (320.0, 512.0))
COINCIDENCE = 0.015
TD_WINDOW = 0.25                  # s, windows of the time-domain power of a segment
_G = {}


def _init(cfg):
    _G.clear()
    _G.update(cfg)


def _pool_map(fn, jobs, workers, cfg):
    if workers <= 1:
        _init(cfg)
        return [fn(j) for j in jobs]
    with mp.Pool(workers, _init, (cfg,)) as pool:
        return pool.map(fn, jobs, chunksize=1)


def template_power(nu, T, fmin, fmax, masses=MASSES):
    """a^2 (n_mass, nu.size) of equal-mass, nonspinning templates (exmarg.profiles.profile at 1 Mpc) on the bins nu"""
    from exmarg.profiles import profile
    a2 = np.zeros((len(masses), nu.size))
    for i, M in enumerate(masses):
        n, _, a, _ = profile(M / 2.0, M / 2.0, 0.0, 0.0, T, fmin, fmax, 1.0)
        a2[i, np.searchsorted(nu, n)] = a ** 2
    return a2


# ---------------------------------------------------------------------------------------------------------------------
# lines
def chunk_starts(good, gps0, n, seconds, pad):
    """GPS starts of n chunks of seconds + 2 pad of good time, the nearest to n evenly spaced times"""
    slots = np.array([gps0 + a + k * (seconds + 2 * pad) for a, b in cond.runs(good)
                      for k in range((b - a) // (seconds + 2 * pad))])
    if slots.size < n:
        raise SystemExit('only %d chunks of %d s of good time' % (slots.size, seconds))
    chosen = []
    for t in gps0 + (np.arange(n) + 0.5) / n * good.size:
        free = slots[~np.isin(slots, chosen)]
        chosen.append(int(free[np.argmin(np.abs(free - t))]))
    return sorted(chosen)


def _chunk_spectrum(job):
    """one-sided median spectrum (1/Hz) of one detector over one chunk on f = k / block up to fmax"""
    det, gps = job
    c = _G
    tl, fs, pad = c['tl'], c['tl']['fs'], c['pad']
    x = cond.read_span(tl, det, gps, (c['seconds'] + 2 * pad) * fs)
    y = strain.whiten(x, fs, lambda f: np.full(np.shape(f), 2.0 / fs), pad, band=(12.0, np.inf), roll=(4.0, 1.0))
    P = strain.periodograms(y, fs, c['block'], 'hann')[:, :int(c['fmax'] * c['block']) + 1]
    return strain.spectrum(P, 'median') * 2.0 / c['block']


def leak_envelope(T, N, pad, fs=64.0):
    """(df, env): the decreasing envelope of the power response, relative to its peak, of the window strain.whiten
    puts on an epoch ((N + 1) T + 2 pad s with sin^2 ramps of pad / 2 s at both ends) at frequency offsets df"""
    n = int(round(((N + 1) * T + 2 * pad) * fs))
    h = int(round(pad * fs)) // 2
    w = np.ones(n)
    r = strain._ramp((np.arange(h) + 0.5) / h)
    w[:h] *= r
    w[n - h:] *= r[::-1]
    P = np.abs(np.fft.rfft(w, 64 * n)) ** 2
    return np.fft.rfftfreq(64 * n, 1.0 / fs), np.maximum.accumulate((P / P[0])[::-1])[::-1]


def read_published(path, lo, hi):
    """[(f, comment)] of an O3 line list of LIGO-T2100200 (lines, and combs expanded to their visible teeth)"""
    out = []
    with open(path) as fh:
        rows = list(csv.reader(line for line in fh if not line.startswith('#')))
    for r in rows[1:]:
        if len(r) < 7:
            continue
        f, kind, offset, first, last = float(r[0]), int(r[1]), float(r[2]), int(r[3]), int(r[4])
        for k in (range(first, last + 1) if kind in (1, 2) else [1]):
            fk = k * f + offset if kind in (1, 2) else f
            if lo <= fk <= hi:
                out.append((fk, re.sub(r';.*', '', ','.join(r[7:])).strip()))
    return sorted(out)


def find_features(f, spectra, a):
    """features of the spectra (n_chunk, n_f) of one detector: (R, Rmax, [dict(lo, hi, peak, R, Rmax, chunks,
    extent=(lo, hi) of the bins with Rmax >= a.extent around it)])"""
    width = int(round(a.smooth * f[1] ** -1)) | 1
    r = spectra / np.exp(ndimage.median_filter(np.log(spectra), size=(1, width), mode='nearest'))
    R, Rmax, above = np.median(r, axis=0), r.max(axis=0), (r > a.detect).sum(axis=0)
    flag = (f >= a.band[0]) & (f <= a.band[1]) & ((R > a.detect) | (above >= a.persist))
    lab, n = ndimage.label(ndimage.binary_closing(flag, np.ones(9)))
    ext = ndimage.label(Rmax >= a.extent)[0]
    out = []
    for k in range(1, n + 1):
        idx = np.flatnonzero(lab == k)
        j = idx[np.argmax(Rmax[idx])]
        ids = np.unique(ext[idx])
        sel = np.isin(ext, ids[ids > 0]) | (lab == k)
        out.append(dict(lo=f[idx[0]], hi=f[idx[-1]], peak=f[j], R=R[idx].max(), Rmax=Rmax[j],
                        chunks=int(above[idx].max()), extent=(f[sel].min(), f[sel].max())))
    return R, Rmax, out


def notch_intervals(features, a, block):
    """the merged notches [[lo, hi, [labels]]] of the features with Rmax >= a.ratio"""
    df, env = leak_envelope(a.T, a.N, a.pad)
    span = (a.N + 1) * a.T + 2 * a.pad
    raw = []
    for det, feats in features.items():
        for ft in feats:
            if ft['Rmax'] < a.ratio:
                continue
            A = 1.0 + (ft['Rmax'] - 1.0) * 1.5 * span / block          # Hann ENBW 1.5 / block -> 1 / span
            margin = max(a.taper + 1.0 / a.T, df[np.argmax(A * env < a.leak)])
            raw.append([np.floor((ft['extent'][0] - margin) * 16) / 16, np.ceil((ft['extent'][1] + margin) * 16) / 16,
                        [(ft['Rmax'], '%s %.3f (R %.0f, Rmax %.3g, %d/%d)'
                          % (det, ft['peak'], ft['R'], ft['Rmax'], ft['chunks'], a.chunks))]])
    raw.sort(key=lambda v: v[0])
    out = []
    for lo, hi, lab in raw:
        if out and lo <= out[-1][1] + a.merge:
            out[-1][1] = max(out[-1][1], hi)
            out[-1][2] += lab
        else:
            out.append([lo, hi, lab])
    return out


def notch_cost(notches, f64, spectra, T, fmin, fmax, masses=MASSES):
    """(bins removed, fraction of the bins of fmin..fmax, {name: (n_mass,) fraction of u lost}) for the week median
    spectra (64 s resolution) averaged over the bins of T s: H1, L1, their maximum, and the maximum of their floors"""
    nu = np.arange(int(np.ceil(fmin * T - 1e-9)), int(np.floor(fmax * T + 1e-9)) + 1)
    fb = nu / T
    cut = strain.response(fb, notches=notches) < 1.0
    a2 = template_power(nu, T, fmin, fmax, masses)
    k = np.round(fb / f64[1]).astype(int)
    h = int(round(0.5 / (T * f64[1])))

    def on_bins(s):
        return np.array([s[i - h:i + h].mean() for i in k])

    med = {d: np.median(s, axis=0) for d, s in spectra.items()}
    width = int(round(4.0 / f64[1])) | 1
    S = {d: on_bins(s) for d, s in med.items()}
    floor = {d: on_bins(np.exp(ndimage.median_filter(np.log(s), size=width, mode='nearest'))) for d, s in med.items()}
    curves = dict(H1=S['H1'], L1=S['L1'], common=np.maximum(S['H1'], S['L1']),
                  common_floor=np.maximum(floor['H1'], floor['L1']))
    loss = {n: (a2[:, cut] / s[cut]).sum(axis=1) / (a2 / s).sum(axis=1) for n, s in curves.items()}
    return int(cut.sum()), float(cut.mean()), loss


def run_lines(a):
    t0 = time.time()
    if a.load:
        z = np.load(a.load)
        f, gps, block, spectra = z['f'], z['gps'], float(z['block']), {d: z['spectrum_' + d] for d in ('H1', 'L1')}
        a.chunks, a.seconds, source = len(gps), int(z['seconds']), str(z['source'])
    else:
        tl = cond.timeline(a.inputs[0], workers=a.workers)
        gps = chunk_starts(tl['good'], tl['gps0'], a.chunks, a.seconds, 8)
        cfg = dict(tl=tl, pad=8, seconds=a.seconds, block=a.block, fmax=a.fmax_spectrum)
        res = _pool_map(_chunk_spectrum, [(d, g) for d in ('H1', 'L1') for g in gps], a.workers, cfg)
        spectra = {d: np.array(res[k * len(gps):(k + 1) * len(gps)]) for k, d in enumerate(('H1', 'L1'))}
        f, block = np.arange(spectra['H1'].shape[1]) / a.block, a.block
        source = 'GWOSC files %s/{H1,L1}, GPS %d to %d' % (os.path.basename(os.path.normpath(a.inputs[0])),
                                                            tl['gps0'], tl['gps0'] + tl['good'].size)
        if a.save:
            np.savez(a.save, f=f, gps=np.array(gps), block=block, seconds=a.seconds, source=source,
                     **{'spectrum_' + d: s for d, s in spectra.items()})
    feats, ratios = {}, {}
    for d in ('H1', 'L1'):
        R, Rmax, feats[d] = find_features(f, spectra[d], a)
        ratios[d] = (R, Rmax)
    pub = {d: read_published(p, a.band[0] - 1, a.band[1] + 1) for d, p in zip(('H1', 'L1'), a.published or ())}

    def idents(d, ft):
        m = sorted({c for fp, c in pub.get(d, ()) if ft['lo'] - 0.05 <= fp <= ft['hi'] + 0.05})
        return m if m else (['not in T2100200'] if pub else [])

    def ident(d, ft):
        return '; '.join(idents(d, ft))

    notches = notch_intervals(feats, a, block)
    nb, frac, loss = notch_cost([n[:2] for n in notches], f, spectra, a.T, a.fmin, a.fmax)
    head = ['notches for exmarg.strain.condition, H1 and L1 (one list for both): lo hi (Hz)',
            'written by: python scripts/real_data/diagnose.py lines ... --ratio %g (%s)' % (a.ratio,
                                                                                         time.strftime('%Y-%m-%d')),
            'data: %s;' % source, '  %d chunks of %d s of coincident good time (GWOSC DATA, CBC_CAT1-3, no hardware '
            'injection) from GPS' % (len(gps), a.seconds), '  %s' % ' '.join(str(int(g)) for g in gps),
            'spectrum: median of the %g s Hann periodograms of each chunk; r = ratio to a running median of the log'
            % block, '  spectrum over %g Hz; R = median of r over the chunks, Rmax = largest r' % a.smooth,
            'feature: bins in %g-%g Hz with R > %g, or r > %g in at least %d chunks' % (a.band[0], a.band[1],
                                                                                      a.detect, a.detect, a.persist),
            'notched: features with Rmax >= %g (the line exceeds the noise of its %g s bin by about %.0f times or more'
            % (a.ratio, a.T, 1.5 * (a.ratio - 1) / (block / a.T)),
            '  at its worst); a notch covers the bins with Rmax >= %g around the feature, widened on each side by'
            % a.extent, '  max(%g Hz taper + 1/T = %g Hz, the offset beyond which the line leaks less than %g of the'
            % (a.taper, a.taper + 1.0 / a.T, a.leak),
            '  floor through the %g s epoch window of strain.whiten); notches closer than %g Hz are merged'
            % ((a.N + 1) * a.T + 2 * a.pad, a.merge),
            'cost: %d of the bins of %g-%g Hz at 1/%g Hz (%.2f %%); template power u lost, total mass %s:'
            % (nb, a.fmin, a.fmax, a.T, 100 * frac, '/'.join('%g' % m for m in MASSES))]
    for n, v in loss.items():
        head.append('  %-12s %s %%' % (n, ' / '.join('%.2f' % (100 * x) for x in v)))
    for r in a.compare or ():
        other = notch_intervals(feats, argparse.Namespace(**dict(vars(a), ratio=r)), block)
        nb_r, frac_r, loss_r = notch_cost([n[:2] for n in other], f, spectra, a.T, a.fmin, a.fmax)
        head.append('  with Rmax >= %g instead: %d notches, %d bins (%.2f %%), u lost (common curve) %s %%'
                    % (r, len(other), nb_r, 100 * frac_r, ' / '.join('%.2f' % (100 * x) for x in loss_r['common'])))
    head += ['note: ' + n if k == 0 else '  ' + n for k, n in enumerate(a.note or ())]
    head.append('features left in (R = week median ratio, Rmax = largest; T2100200 identification):')
    for d in ('H1', 'L1'):
        for ft in feats[d]:
            if ft['Rmax'] < a.ratio and not any(lo <= ft['lo'] and ft['hi'] <= hi for lo, hi, _ in notches):
                row = '  %s %8.3f-%8.3f Hz  R %6.1f  Rmax %7.1f  %d/%d chunks  %s' % (
                    d, ft['lo'], ft['hi'], ft['R'], ft['Rmax'], ft['chunks'], len(gps), ident(d, ft))
                head.append(row[:116])
    lines = ['# ' + h for h in head] + ['# lo hi  # strongest feature covered (detector, peak Hz, R, Rmax, chunks '
                                        'above --detect) | T2100200 identification']
    for lo, hi, lab in notches:
        lab = [t for _, t in sorted(lab, reverse=True)]
        text = '%.4f %.4f  # %s%s' % (lo, hi, lab[0], ' + %d more' % (len(lab) - 1) if len(lab) > 1 else '')
        names = sorted({c for d in ('H1', 'L1') for ft in feats[d] if ft['Rmax'] >= a.ratio and lo <= ft['peak'] <= hi
                        for c in idents(d, ft)})
        sep = ' | '
        for c in names:
            if len(text) + len(sep) + len(c) <= 120:
                text, sep = text + sep + c, '; '
        lines.append(text)
    with open(a.inputs[-1], 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\nlines: %d features in H1, %d in L1, %d notches written to %s (%.0f s)'
          % (len(feats['H1']), len(feats['L1']), len(notches), a.inputs[-1], time.time() - t0))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# segments
def whitened(ep, pre):
    """e_I = d_I / sqrt(sig2_ref_I) before step 5, 0 in the notched bins"""
    keep = ~ep['notched']
    return {I: pre[I] / np.sqrt(np.where(keep, ep['sig2_ref_' + I], 1.0)) for I in 'HL'}


def window_power(e, keep, window=TD_WINDOW, T=8.0):
    """largest power of the whitened series of each segment (the kept bins, inverse transformed) over windows of
    `window` s, over its mean for unit-variance bins: (n_segment, n_bin) -> (n_segment,)"""
    n = 8 * 1024
    full = np.zeros((e.shape[0], n // 2 + 1), complex)
    full[:, int(round(_G['f'][0] * T)):int(round(_G['f'][0] * T)) + e.shape[1]] = np.where(keep, e, 0.0)
    y = np.fft.irfft(full, n, axis=1)
    m = int(round(window / T * n))
    p = (y[:, : n // m * m] ** 2).reshape(e.shape[0], -1, m).mean(axis=2)
    return p.max(axis=1) / (2.0 * keep.sum() / n ** 2)


def _sums(e, masks):
    """sums over the segments of each set of one epoch's whitened data e {I: (n_segment, n_freq)}; masks
    {set: {I: (n_segment,) bool}} -> {(set, quantity, detector): sum}"""
    out = {}
    for name, m in masks.items():
        for I in 'HL':
            x = e[I][m[I]]
            p = np.abs(x) ** 2
            out[name, 's1', I], out[name, 's2', I] = p.sum(axis=0), (p * p).sum(axis=0)
            out[name, 'nb', I], out[name, 'n', I] = (x[:, :-1] * np.conj(x[:, 1:])).sum(axis=0), int(m[I].sum())
        both = m['H'] & m['L']
        out[name, 'hl', 'HL'] = (e['H'][both] * np.conj(e['L'][both])).sum(axis=0)
        out[name, 'n', 'HL'] = int(both.sum())
    return out


def _masks(ep, mon):
    """the sets of segments of an epoch, {set: {I: (n_segment,) bool}}, all without a catalogued event: all; burst
    (BURST_CAT2 set over the segment, when the file has the bit); quiet (every band monitor below the threshold);
    analysable (loud in neither detector, epoch usable: files with flags); selected (the segments of --select)"""
    ok = ~ep['event']
    out = {'all': {I: ok for I in 'HL'}, 'quiet': {I: ok & np.all(mon[I] < _G['thresholds'], axis=1) for I in 'HL'}}
    if _G['burst_bit'] is not None:
        out['burst'] = {I: ok & (ep['dq_' + I] & _G['burst_bit'] > 0) for I in 'HL'}
    if 'analysable' in ep:
        out['analysable'] = {I: ok & ep['analysable'] for I in 'HL'}
    if _G.get('select') is not None:
        out['selected'] = {I: ok & np.isin(np.round(ep['gps']), _G['select']) for I in 'HL'}
    return out


def _epoch_stats(path):
    ep = strain.read_epoch(path)
    keep = ~ep['notched']
    e = whitened(ep, strain.unequalized(ep))
    mon = {I: strain.monitors(e[I], ep['f'], ep['monitor_edges'], keep) for I in 'HL'}
    masks = _masks(ep, mon)
    out = _sums(e, masks)
    out['check'] = max(float(np.nanmax(np.abs(mon[I] / ep['monitor_' + I] - 1.0))) for I in 'HL')
    n = ep['gps'].size
    for I in 'HL':
        ok = ~ep['event']
        out['all', 'q1', I] = ((np.abs(ep['d_' + I][ok].astype(complex)) ** 2 / ep['sig2']).sum(axis=0)
                               if ep['equal_noise'] else np.zeros(keep.size))
        out['monitor', I], out['dq', I] = ep['monitor_' + I], ep.get('dq_' + I, np.zeros(n, np.int64))
        out['td', I] = window_power(e[I], keep, T=ep['T'])
        out['loud', I] = ep.get('loud_' + I, np.zeros(n, bool))
    for name in ('analysable', 'selected'):
        out['set', name] = masks[name]['H'] if name in masks else np.zeros(n, bool)
    out['usable'] = bool(ep.get('usable', True))
    a2 = _G['a2']
    for name, s in (('H', ep['sig2_ref_H']), ('L', ep['sig2_ref_L']), ('c', np.maximum(ep['sig2_ref_H'],
                                                                                     ep['sig2_ref_L']))):
        out['u', name] = (a2[:, keep] / s[keep]).sum(axis=1)
    out.update(gps=ep['gps'], event=ep['event'], gps_epoch=ep['gps_epoch'])
    return out


def _simulated_epoch(seed):
    """whitened data of one epoch of stationary Gaussian noise through the estimator of the files (N + 1 blocks,
    leave-one-out spectrum, correction): {I: (N + 1, n_freq)}"""
    rng = np.random.default_rng(seed)
    nf, N, method = _G['n_freq'], _G['N'], _G['method']
    c = strain.correction(N, method)
    out = {}
    for I in 'HL':
        g = rng.standard_normal((2, N + 1, nf))
        W = np.sqrt(0.5) * (g[0] + 1j * g[1])
        out[I] = np.where(_G['keep'], W / np.sqrt(c * strain.loo_spectrum(np.abs(W) ** 2, method)), 0.0)
    return out


def _simulated_stats(seed):
    e = _simulated_epoch(seed)
    out = _sums(e, {'all': {I: np.ones(_G['N'] + 1, bool) for I in 'HL'}})
    for I in 'HL':
        out['monitor', I] = strain.monitors(e[I], _G['f'], _G['edges'], _G['keep'])
        out['td', I] = window_power(e[I], _G['keep'], T=_G['T'])
    return out


_BANK = {}


def _bank(ep):
    from exmarg.search import mass_bank
    if 'bank' not in _BANK:
        _BANK['bank'] = mass_bank(np.array(MASSES), ep, 'cpu')
        _BANK['W2'] = np.abs(_BANK['bank'].W.astype(complex)) ** 2
    return _BANK['bank'], _BANK['W2']


def _filter_rows(d, sig2, rows, bank, W2, keep):
    """matched filter of the rows of d {I: (n_segment, n_freq)} with noise sig2 {I: (n_freq,)}; per detector and
    segment: histogram of w on bins of 0.25 (n_row, n_mass, 401), counts above W_THRESHOLDS (n_row, n_mass, 5), the
    largest w, its time index and the mean w (n_row, n_mass)"""
    s = {I: np.where(keep, sig2[I], 1.0) for I in 'HL'}
    u = {I: (W2[:, keep] / s[I][keep]).sum(axis=1) for I in 'HL'}
    nm, nr = len(MASSES), len(rows)
    out = {I: dict(hist=np.zeros((nr, nm, 401), np.int32), above=np.zeros((nr, nm, len(W_THRESHOLDS)), np.int32),
                   wmax=np.empty((nr, nm)), jmax=np.empty((nr, nm), np.int64), wmean=np.empty((nr, nm)))
           for I in 'HL'}
    for k, row in enumerate(rows):
        tr = bank.trajectories({I: np.where(keep, d[I][row], 0.0) for I in 'HL'}, s)
        for I in 'HL':
            w = np.exp(2.0 * tr['lq' + I].astype(float)) / u[I][:, None]
            o = out[I]
            for m in range(nm):
                o['hist'][k, m] = np.bincount(np.minimum((w[m] * 4.0).astype(np.int64), 400), minlength=401)
            o['above'][k] = (w[:, :, None] > np.array(W_THRESHOLDS)).sum(axis=1)
            o['jmax'][k] = np.argmax(w, axis=1)
            o['wmax'][k] = w[np.arange(nm), o['jmax'][k]]
            o['wmean'][k] = w.mean(axis=1)
    return out


def _filter_file(job):
    path, rows = job
    ep = strain.read_epoch(path)
    bank, W2 = _bank(ep)
    keep = ~ep['notched']
    pre = strain.unequalized(ep)
    e = whitened(ep, pre)
    mon = {I: strain.monitors(e[I], ep['f'], ep['monitor_edges'], keep) for I in 'HL'}
    masks = _masks(ep, mon)
    out = dict(pre=_filter_rows(pre, {I: ep['sig2_ref_' + I] for I in 'HL'}, rows, bank, W2, keep))
    if ep['equal_noise']:
        out['post'] = _filter_rows({I: ep['d_' + I].astype(complex) for I in 'HL'}, {I: ep['sig2'] for I in 'HL'},
                                   rows, bank, W2, keep)
    out['sets'] = {(name, I): m[I][rows] for name, m in masks.items() for I in 'HL'}
    out['gps'] = ep['gps'][rows]
    out['dq'] = {I: ep.get('dq_' + I, np.zeros(ep['gps'].size, np.int64))[rows] for I in 'HL'}
    return out


def _filter_simulated(seed):
    e = _simulated_epoch(seed)
    ep = _G['template_epoch']
    bank, W2 = _bank(ep)
    d = {I: e[I] * np.sqrt(np.where(_G['keep'], ep['sig2_ref_' + I], 0.0)) for I in 'HL'}
    return dict(pre=_filter_rows(d, {I: ep['sig2_ref_' + I] for I in 'HL'}, np.arange(_G['N'] + 1), bank, W2,
                                 _G['keep']))


def _merge_filter(parts, version):
    """{I: {key: concatenated over the parts}} of the matched-filter outputs"""
    return {I: {k: np.concatenate([p[version][I][k] for p in parts]) for k in parts[0][version][I]}
            for I in 'HL' if all(version in p for p in parts)}


def _sum_parts(parts):
    return {k: sum(p[k] for p in parts) for k in parts[0] if isinstance(k, tuple) and len(k) == 3}


def _band(f, lo, hi):
    return (f >= lo) & ((f <= hi) if hi == BANDS[-1][1] else (f < hi))


def _power_table(S, name, f, keep, V, I, sigma=4.0):
    """per band of BANDS and the whole band: (lo, hi, bins, mean - 1, its error, rms of the per-bin means over the
    bins, expected rms, bins beyond sigma); the bins beyond sigma grouped (lo, hi, bins, mean power, largest z)"""
    n = S[name, 'n', I]
    m = S[name, 's1', I] / n
    z = (m - 1.0) / np.sqrt(V / n)
    rows = []
    for lo, hi in BANDS + ((BANDS[0][0], BANDS[-1][1]),):
        b = keep & _band(f, lo, hi)
        if b.any():
            rows.append((lo, hi, int(b.sum()), m[b].mean() - 1.0, np.sqrt(V / n / b.sum()), m[b].std(), np.sqrt(V / n),
                         int((np.abs(z[b]) > sigma).sum())))
    groups = []
    for sl in ndimage.find_objects(ndimage.label(keep & (np.abs(z) > sigma))[0]):
        k = np.arange(sl[0].start, sl[0].stop)
        groups.append((f[k[0]], f[k[-1]], k.size, float(m[k].mean()), float(z[k][np.argmax(np.abs(z[k]))])))
    return rows, groups, m


def _correlations(S, name, f, keep, I):
    """per band: |<e_k e*_k+1>| / <|e|^2> over the pairs of kept bins, and its standard error"""
    pair = keep[:-1] & keep[1:]
    n = S[name, 'n', I]
    out = []
    for lo, hi in BANDS:
        b = pair & _band(f[:-1], lo, hi)
        if b.any():
            out.append((lo, hi, abs(S[name, 'nb', I][b].sum()) / S[name, 's1', I][:-1][b].sum(),
                        1.0 / np.sqrt(n * b.sum())))
    return out


def _monitor_summary(mon, pairs, quiet):
    """per band: median, half-width of the 16-84 percent range, mean, rms, fractions outside 1 +- 0.05 and 0.10,
    fraction not quiet, rank correlation of the segment pairs (i, i + 1) in pairs, and the linear correlation of the
    pairs of quiet segments"""
    from scipy.stats import rankdata
    rows = []
    for k in range(mon.shape[1]):
        m = mon[:, k]
        q = np.percentile(m, [16, 50, 84])
        r = rankdata(m)
        both = pairs & quiet[:-1] & quiet[1:]
        mq = m[quiet] - m[quiet].mean()
        lin = (np.mean((m[:-1][both] - m[quiet].mean()) * (m[1:][both] - m[quiet].mean())) / np.mean(mq * mq)
               if both.any() else np.nan)
        rho = np.corrcoef(r[:-1][pairs], r[1:][pairs])[0, 1] if pairs.sum() > 2 else np.nan
        rows.append((q[1], 0.5 * (q[2] - q[0]), m.mean(), m.std(), np.mean(np.abs(m - 1) > 0.05),
                     np.mean(np.abs(m - 1) > 0.10), 1.0 - quiet.mean(), rho, lin))
    return np.array(rows)


def _filter_report(o, sel, name, ref, P):
    """lines of the matched-filter report of one detector over the segments sel"""
    n = int(sel.sum())
    for i, M in enumerate(MASSES):
        if not n:
            continue
        sf = o['above'][sel, i].sum(axis=0) / o['hist'][sel, i].sum()
        wm = o['wmean'][sel, i]
        seg = [np.mean(o['wmax'][sel, i] > t) for t in W_THRESHOLDS[1:]]
        P('    %s M=%3.0f  n %5d  mean w %8.4f +- %.4f  median %.4f | P(w > %s) %s | segments max w > %s: %s'
          % (name, M, n, wm.mean(), wm.std() / np.sqrt(n), np.median(wm), '/'.join('%g' % t for t in W_THRESHOLDS),
             ' '.join('%.1e' % v for v in sf), '/'.join('%g' % t for t in W_THRESHOLDS[1:]),
             ' '.join('%.4f' % v for v in seg)))
        if ref is not None:
            hr = ref['hist'][:, i].sum()
            P('    %s        Gaussian: n %5d  mean w %8.4f +- %.4f  median %.4f | P(w > ...) %s [exp(-w) %s] | '
              'segments %s' % (' ' * len(name), ref['wmean'].shape[0], ref['wmean'][:, i].mean(),
                               ref['wmean'][:, i].std() / np.sqrt(ref['wmean'].shape[0]), np.median(ref['wmean'][:, i]),
                               ' '.join('%.1e' % v for v in ref['above'][:, i].sum(axis=0) / hr),
                               ' '.join('%.1e' % np.exp(-t) for t in W_THRESHOLDS),
                               ' '.join('%.4f' % np.mean(ref['wmax'][:, i] > t) for t in W_THRESHOLDS[1:])))


def _split(mon, member, epoch_of, loud_epoch):
    """band means - 1 of the monitors of the member segments in epochs without and with a loud segment, the numbers
    of segments, and the standard error of the first"""
    a, b = member & ~loud_epoch[epoch_of], member & loud_epoch[epoch_of]
    nan = np.full(mon.shape[1], np.nan)
    return (mon[a].mean(axis=0) - 1 if a.any() else nan, mon[b].mean(axis=0) - 1 if b.any() else nan, int(a.sum()),
            int(b.sum()))


def run_segments(a):
    t0 = time.time()
    seg_dir = a.inputs[0]
    paths = sorted(p for p in glob.glob(os.path.join(seg_dir, 'epoch_*.npz')) if not p.endswith('.tmp.npz'))
    if not paths:
        raise SystemExit('no epoch files in %s' % seg_dir)
    ep0 = strain.read_epoch(paths[0])
    f, keep, N, method, T = ep0['f'], ~ep0['notched'], int(ep0['N']), str(ep0['method']), float(ep0['T'])
    names = [str(v) for v in ep0.get('dq_names', [])]
    burst_bit = 1 << names.index('BURST_CAT2') if 'BURST_CAT2' in names else None
    a2 = template_power(ep0['nu'], T, ep0['fmin'], ep0['fmax']) / 10410.0 ** 2
    select = None
    if a.select:
        z = np.load(a.select)
        select = np.round(z['gps_all'][z['analysable_all']])
    cfg = dict(a2=a2, n_freq=f.size, N=N, method=method, keep=keep, f=f, edges=ep0['monitor_edges'], T=T,
               burst_bit=burst_bit, select=select)
    V = strain.whitened_variance(N, method)
    P = print

    # Gaussian reference through the same estimator; it sets the quiet thresholds of the monitors
    n_sim = max(int(np.ceil(a.simulate / (N + 1))), 1)
    sim = _pool_map(_simulated_stats, [[a.seed, 1, k] for k in range(n_sim)], a.workers, cfg)
    S_sim = _sum_parts(sim)
    mon_sim = {I: np.concatenate([s['monitor', I] for s in sim]) for I in 'HL'}
    td_sim = np.concatenate([s['td', I] for s in sim for I in 'HL'])
    thresholds = 1.0 + a.quiet * np.concatenate([mon_sim['H'], mon_sim['L']]).std(axis=0)
    dof = 2.0 * keep.sum() * TD_WINDOW / T
    td_thr = float(special.gammainccinv(dof / 2.0, 1e-6 * TD_WINDOW / T) * 2.0 / dof)
    cfg['thresholds'] = thresholds
    pairs_sim = np.tile(np.r_[np.ones(N, bool), False], n_sim)[:-1]

    parts = _pool_map(_epoch_stats, paths, a.workers, cfg)
    order = np.argsort([p['gps_epoch'] for p in parts])
    parts, paths = [parts[i] for i in order], [paths[i] for i in order]
    S = _sum_parts(parts)
    gps_all = np.concatenate([p['gps'] for p in parts])
    event = np.concatenate([p['event'] for p in parts])
    epoch_all = np.concatenate([np.full(p['gps'].size, k) for k, p in enumerate(parts)])
    ok = ~event
    g, epoch_of = gps_all[ok], epoch_all[ok]
    per = {key: {I: np.concatenate([p[key, I] for p in parts])[ok] for I in 'HL'} for key in ('monitor', 'dq', 'td',
                                                                                          'loud')}
    sets_seg = {name: np.concatenate([p['set', name] for p in parts])[ok] for name in ('analysable', 'selected')}
    usable = np.array([p['usable'] for p in parts])
    quiet = {I: np.all(per['monitor'][I] < thresholds, axis=1) for I in 'HL'}
    contiguous = np.abs(np.diff(g) - T) < 1e-6
    within = contiguous & (epoch_of[:-1] == epoch_of[1:])
    check = max(p['check'] for p in parts)
    flagged_files = 'analysable' in strain.read_epoch(paths[0])
    t_stats = time.time() - t0

    # matched filter on a random subset, and on Gaussian noise with the curves of a real epoch
    rows = [(k, r) for k, p in enumerate(parts) for r in np.flatnonzero(~p['event'])]
    pick = sorted(np.random.default_rng(a.seed).choice(len(rows), size=min(a.subset, len(rows)), replace=False))
    jobs = {}
    for i in pick:
        jobs.setdefault(rows[i][0], []).append(rows[i][1])
    fparts = _pool_map(_filter_file, [(paths[k], np.array(r)) for k, r in sorted(jobs.items())], a.workers, cfg)
    filt = {v: _merge_filter(fparts, v) for v in ('pre', 'post')}
    fsets = {key: np.concatenate([p['sets'][key] for p in fparts]) for key in fparts[0]['sets']}
    fgps = np.concatenate([p['gps'] for p in fparts])
    fdq = {I: np.concatenate([p['dq'][I] for p in fparts]) for I in 'HL'}
    n_fsim = max(int(np.ceil(a.subset_simulated / (N + 1))), 1)
    fsim = _merge_filter(_pool_map(_filter_simulated, [[a.seed, 2, k] for k in range(n_fsim)], a.workers,
                                   dict(cfg, template_epoch=strain.read_epoch(paths[len(paths) // 2]))), 'pre')
    n_time = int(filt['pre']['H']['hist'][0, 0].sum())
    t_all = time.time() - t0

    sets = [s for s in ('all', 'burst', 'quiet', 'analysable', 'selected') if (s, 'n', 'H') in S]
    label = dict(all='all segments', burst='BURST_CAT2 passes over the segment',
                 quiet='quiet: every band monitor < 1 + %g Gaussian sigma' % a.quiet,
                 analysable='analysable: loud in neither detector, epoch usable',
                 selected='selected: the analysable segments of %s' % a.select)
    out = dict(f=f, keep=keep, V=V, masses=np.array(MASSES), w_thresholds=np.array(W_THRESHOLDS), bands=np.array(BANDS),
               gps=g, gps_all=gps_all, event_all=event, n_epochs=len(paths), n_segments=gps_all.size,
               n_event=int(event.sum()), thresholds=thresholds, monitor_edges=ep0['monitor_edges'], filter_gps=fgps,
               usable=usable, td_threshold=td_thr, analysable_all=np.concatenate([p['set', 'analysable']
                                                                                  for p in parts]))
    P('diagnostics of %s: %d epochs, %d segments, %d flagged with a catalogued event and left out; method %s, N = %d;'
      ' data before step 5 regenerated (largest relative difference of the monitors %.1e)'
      % (seg_dir, len(paths), gps_all.size, event.sum(), method, N, check))
    P('Gaussian reference: stationary Gaussian noise through the same estimator on the same bins, %d segments (%d for '
      'the matched filter); V = Var|e|^2 = %.4f; quiet thresholds of the monitors %s; time-domain threshold %.3f '
      '(largest %g s window power, 1e-6 per segment)' % (n_sim * (N + 1), n_fsim * (N + 1), V,
                                                          ' '.join('%.3f' % t for t in thresholds), td_thr, TD_WINDOW))
    if flagged_files:
        loud_seg = {I: per['loud'][I] for I in 'HL'}
        P('loud blocks (strain.loud_blocks): H1 %.3f %%, L1 %.3f %% of the segments, both %.3f %%; epochs unusable '
          '%.2f %% (%d of %d); analysable segments %.2f %% (%d)'
          % (100 * loud_seg['H'].mean(), 100 * loud_seg['L'].mean(), 100 * (loud_seg['H'] & loud_seg['L']).mean(),
             100 * (~usable).mean(), (~usable).sum(), usable.size, 100 * sets_seg['analysable'].mean(),
             sets_seg['analysable'].sum()))
    for I, name in (('H', 'H1'), ('L', 'L1')):
        P('\n=== %s' % name)
        for s in sets:
            n = S[s, 'n', I]
            if not n:
                continue
            rows_t, groups, m = _power_table(S, s, f, keep, V, I)
            ref = _power_table(S_sim, 'all', f, keep, V, I)[0]
            P('whitened power per bin, %s (%d segments, %.2f %%): expected 1 +- %.4f per bin'
              % (label[s], n, 100.0 * n / max(S['all', 'n', I], 1), np.sqrt(V / n)))
            P('   band (Hz)  bins   band mean - 1 (+- err)   rms over bins (expected; Gaussian ref)   bins > 4 sigma')
            for r, rr in zip(rows_t, ref):
                P('   %3.0f-%-4.0f  %5d   %+9.4f (+- %.4f)      %9.4f (%.4f; %.4f)              %4d'
                  % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], rr[5] * np.sqrt(S_sim['all', 'n', I] / n), r[7]))
            if s in ('quiet', 'analysable', 'selected'):
                txt = ['%.3f-%.3f %d %.3f %+.0f' % gr for gr in groups]
                P('   bins beyond 4 sigma, grouped (lo-hi Hz, bins, mean power, largest z): %d groups' % len(txt))
                for k in range(0, min(len(txt), 20), 5):
                    P('     ' + '   '.join(txt[k:k + 5]))
            P('   neighbouring bins |<e_k e*_k+1>| / <|e|^2> (standard error): ' + '  '.join(
                '%g-%g %.4f (%.4f)' % c for c in _correlations(S, s, f, keep, I)))
            out['power_%s_%s' % (s, I)] = m
            out['power_bands_%s_%s' % (s, I)] = np.array(rows_t)
        q = S['all', 'q1', I] / S['all', 'n', I]
        P('after step 5, all segments, |d|^2 / sig2 per band - 1: ' + '  '.join(
            '%g-%g %+.4f' % (lo, hi, q[keep & _band(f, lo, hi)].mean() - 1) for lo, hi in BANDS))
        mon = per['monitor'][I]
        ms_sim = _monitor_summary(mon_sim[I], pairs_sim, np.ones(mon_sim[I].shape[0], bool))
        P('whitened variance monitors: median, 16-84 % half-width, mean, rms, out 5 %, out 10 %, not quiet, rank '
          'corr. of neighbours within epoch / across epochs, linear corr. of quiet neighbours')
        groups_m = [('all', np.ones(g.size, bool))]
        if flagged_files:
            groups_m.append(('analysable', sets_seg['analysable']))
        for gname, sel in groups_m:
            idx = np.flatnonzero(sel)
            nxt = np.r_[idx[1:] - idx[:-1] == 1, False][:-1] if idx.size > 1 else np.zeros(0, bool)
            ms = _monitor_summary(mon[idx], within[idx[:-1]] & nxt, quiet[I][idx])
            ms_all = _monitor_summary(mon[idx], contiguous[idx[:-1]] & nxt, quiet[I][idx])
            P('  %s segments (%d):' % (gname, idx.size))
            for k in range(ms.shape[0]):
                row = (ep0['monitor_edges'][k], ep0['monitor_edges'][k + 1], *ms[k, :7], ms[k, 7], ms_all[k, 7],
                       ms[k, 8], *ms_sim[k, :8])
                P('   %3.0f-%-3.0f %.4f %.4f %9.3f %9.3f %6.3f %6.3f %7.4f   %+.3f / %+.3f   %+.3f   [Gaussian: %.4f '
                  '%.4f %.4f %.4f %.3f %.3f %.5f %+.3f]' % row)
            out['monitor_summary_%s_%s' % (gname, I)] = ms
        if burst_bit is not None:
            P('segments failing BURST_CAT2: %.2f %%; not quiet: %.2f %%; both: %.2f %%'
              % (100 * np.mean(per['dq'][I] & burst_bit == 0), 100 * np.mean(~quiet[I]),
                 100 * np.mean((per['dq'][I] & burst_bit == 0) & ~quiet[I])))
        P('split: band means of the monitors - 1 in epochs without | with a loud segment of this detector '
          '(standard error of a mean of n monitors: rms / sqrt(n))')
        defs = [('not quiet', quiet[I])] + ([('flagged', ~per['loud'][I])] if flagged_files else [])
        members = [('quiet', quiet[I])] + ([('analysable', sets_seg['analysable'])] if flagged_files else [])
        for dname, calm in defs:
            loud_epoch = np.zeros(len(parts), bool)
            loud_epoch[np.unique(epoch_of[~calm])] = True
            for mname, member in members:
                w0, w1, n0, n1 = _split(mon, member, epoch_of, loud_epoch)
                P('   loud = %-9s  %-10s segments: %s (%d) | %s (%d); epochs with a loud segment %.1f %%'
                  % (dname, mname, ' '.join('%+.4f' % v for v in w0), n0, ' '.join('%+.4f' % v for v in w1), n1,
                     100 * loud_epoch.mean()))
                out['split_%s_%s_%s' % (dname.replace(' ', '_'), mname, I)] = np.array([w0, w1])
        if flagged_files:
            nb = [int(np.sum(keep & _band(f, lo, hi))) for lo, hi in zip(ep0['monitor_edges'][:-1],
                                                                         ep0['monitor_edges'][1:])]
            over = mon > strain.loud_thresholds(nb, N, method, a.flag_p)
            lseg = per['loud'][I]
            P('loud segments (%d): fraction above the threshold for %d blocks in each band %s; in none (their '
              'estimate rests on fewer blocks) %.3f'
              % (lseg.sum(), N, ' '.join('%.3f' % v for v in over[lseg].mean(0)), np.mean(~over[lseg].any(axis=1))))
        td = per['td'][I]
        tdl = td > td_thr
        txt = 'time domain: largest %g s window power > %.3f in %.3f %% of all segments [Gaussian %.4f %%]' % (
            TD_WINDOW, td_thr, 100 * tdl.mean(), 100 * np.mean(td_sim > td_thr))
        if flagged_files:
            hidden = tdl & ~per['loud'][I]
            calm_epoch = np.zeros(len(parts), bool)
            calm_epoch[np.unique(epoch_of[hidden])] = True
            an = sets_seg['analysable']
            w0, w1, n0, n1 = _split(mon, an & ~hidden, epoch_of, calm_epoch)
            txt += ('; of those not flagged loud: %d (%.3f %% of the segments). Other analysable segments in epochs '
                    'without | with such a segment: %s (%d) | %s (%d)' % (hidden.sum(), 100 * hidden.mean(),
                                                                          ' '.join('%+.4f' % v for v in w0), n0,
                                                                          ' '.join('%+.4f' % v for v in w1), n1))
        P(txt)
        out.update({'monitor_' + I: mon, 'quiet_' + I: quiet[I], 'dq_' + I: per['dq'][I], 'td_' + I: td,
                    'loud_' + I: per['loud'][I], 'monitor_summary_sim_' + I: ms_sim})
    out['analysable'] = sets_seg['analysable']
    for s in sets:
        n = S[s, 'n', 'HL']
        if n:
            P('correlation of H1 with L1 (%s): |<e_H e*_L>| = %.5f (standard error %.5f)'
              % (s, np.abs(S[s, 'hl', 'HL'][keep].sum()) / np.sum(np.sqrt(S[s, 's1', 'H'][keep] *
                                                                           S[s, 's1', 'L'][keep])),
                 1.0 / np.sqrt(n * keep.sum())))

    u = {k: np.array([p['u', k] for p in parts]) for k in 'HLc'}
    P('\nstep 5: loss of template power 1 - u_common / u_own per epoch (percentiles 5 25 50 75 95, mean)')
    for i, M in enumerate(MASSES):
        txt = []
        for I in 'HL':
            loss = 100 * (1 - u['c'][:, i] / u[I][:, i])
            txt.append('%s1 %s; %.1f %%' % (I, ' '.join('%5.1f' % q for q in np.percentile(loss, [5, 25, 50, 75, 95])),
                                            loss.mean()))
        P('   M = %3.0f  %s' % (M, '   '.join(txt)))
    for i, M in enumerate(MASSES):
        r = u['H'][:, i] / u['L'][:, i]
        P('   M = %3.0f  u_H1 / u_L1 before step 5 per epoch: min %.3f, percentiles 5 25 50 75 95 %s, max %.3f; H1 the '
          'quieter in %.0f %% of epochs; rms of sqrt(u_own) over epochs %.1f / %.1f %%'
          % (M, r.min(), ' '.join('%.3f' % q for q in np.percentile(r, [5, 25, 50, 75, 95])), r.max(),
             100 * np.mean(r > 1), 100 * np.std(np.sqrt(u['H'][:, i])) / np.mean(np.sqrt(u['H'][:, i])),
             100 * np.std(np.sqrt(u['L'][:, i])) / np.mean(np.sqrt(u['L'][:, i]))))
    out.update(u_H=u['H'], u_L=u['L'], u_c=u['c'], uH_uL_60=u['H'][:, 1] / u['L'][:, 1],
               gps_epoch=np.array([p['gps_epoch'] for p in parts]))

    P('\nmatched filter (exmarg.search.mass_bank, CPU), %d random segments, equal-mass templates; w = |z|^2 / u at '
      'every time (%d per segment); Gaussian: exp(-w)' % (len(pick), n_time))
    for version, vlabel in (('pre', 'before step 5, each detector on its own curve'), ('post', 'after step 5')):
        if not filt[version]:
            continue
        for s in (sets if version == 'pre' else [s for s in ('all', 'analysable', 'quiet') if s in sets]):
            P('-- %s, %s' % (vlabel, label[s]))
            for I, name in (('H', 'H1'), ('L', 'L1')):
                _filter_report(filt[version][I], fsets[s, I], name, fsim[I] if version == 'pre' and s == 'all'
                               else None, P)
        for I in 'HL':
            o = filt[version][I]
            out.update({'%s_%s_%s' % (version, I, k): o[k] for k in ('wmax', 'jmax', 'wmean', 'above')})
            out['%s_%s_hist' % (version, I)] = o['hist'].sum(axis=0)
    for I in 'HL':
        out.update({'sim_%s_%s' % (I, k): fsim[I][k] for k in ('wmax', 'wmean', 'above')})
        out['sim_%s_hist' % I] = fsim[I]['hist'].sum(axis=0)
        for s in sets:
            out['filter_%s_%s' % (s, I)] = fsets[s, I]

    o = filt['pre']
    for s in [s for s in ('all', 'analysable', 'selected') if s in sets]:
        P('\nglitch duty cycle per %g s, %s: fraction of the subset segments whose largest w over the three templates '
          'exceeds w (before step 5), binomial error; [Gaussian]; and the fraction of those failing BURST_CAT2'
          % (T, label[s]))
        for I, name in (('H', 'H1'), ('L', 'L1')):
            sel = fsets[s, I]
            wm, ws = o[I]['wmax'][sel].max(axis=1), fsim[I]['wmax'].max(axis=1)
            txt = []
            for t in W_THRESHOLDS[1:]:
                p = np.mean(wm > t)
                fail = (np.mean(fdq[I][sel][wm > t] & burst_bit == 0) if burst_bit is not None and np.any(wm > t)
                        else np.nan)
                txt.append('w > %g (SNR %.0f): %.4f +- %.4f [%.4f], %.0f %% BURST_CAT2' % (
                    t, np.sqrt(2 * t), p, np.sqrt(p * (1 - p) / max(wm.size, 1)), np.mean(ws > t), 100 * fail))
            P('   %s (%d): %s' % (name, wm.size, ';  '.join(txt)))
    P('coincidence of the loudest times of H1 and L1 within %g ms (chance %.4f for independent times), and segments '
      'loud in both (w > 12.5) against independence (all subset segments):' % (1e3 * COINCIDENCE, 2 * COINCIDENCE / T))
    for i, M in enumerate(MASSES):
        dj = (o['H']['jmax'][:, i] - o['L']['jmax'][:, i] + n_time // 2) % n_time - n_time // 2
        co = np.abs(dj) * T / n_time <= COINCIDENCE
        txt = []
        for t in (0.0, 8.0, 12.5, 32.0):
            both = (o['H']['wmax'][:, i] > t) & (o['L']['wmax'][:, i] > t)
            txt.append('both > %g: %d, %d coincident' % (t, both.sum(), co[both].sum()))
        lH, lL = o['H']['wmax'][:, i] > 12.5, o['L']['wmax'][:, i] > 12.5
        P('   M=%3.0f  %s;  loud: H1 %d, L1 %d, both %d (independent %.1f)'
          % (M, '; '.join(txt), lH.sum(), lL.sum(), (lH & lL).sum(), lH.mean() * lL.mean() * lH.size))
        out['coincident_%d' % M] = co
    loud = []
    for I, name in (('H', 'H1'), ('L', 'L1')):
        wm = o[I]['wmax'].max(axis=1)
        for k in np.argsort(wm)[::-1][:6]:
            j = int(np.argmax(o[I]['wmax'][k]))
            loud.append('%s %.2f w %.3g M%g%s' % (name, fgps[k] + o[I]['jmax'][k, j] * T / n_time, wm[k], MASSES[j],
                                                  ' (BURST_CAT2)' if burst_bit is not None and fdq[I][k] & burst_bit
                                                  == 0 else ''))
    P('loudest of the subset (GPS of the largest w): ' + ', '.join(loud))
    path = a.out or os.path.join(seg_dir, 'diagnostics.npz')
    np.savez(path, **out)
    P('\nwritten %s (statistics %.0f s, total %.0f s)' % (path, t_stats, t_all))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('lines', 'segments'))
    p.add_argument('inputs', nargs='+', help='lines: <gwosc_dir> <notches.txt> (or --load and <notches.txt>); '
                                             'segments: <segments_dir>')
    p.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument('--seed', type=int, default=1)
    g = p.add_argument_group('lines')
    g.add_argument('--chunks', type=int, default=8)
    g.add_argument('--seconds', type=int, default=3600)
    g.add_argument('--block', type=float, default=64.0, help='periodogram length (s)')
    g.add_argument('--fmax-spectrum', type=float, default=800.0)
    g.add_argument('--smooth', type=float, default=4.0, help='running median (Hz)')
    g.add_argument('--detect', type=float, default=3.0)
    g.add_argument('--persist', type=int, default=2)
    g.add_argument('--band', type=float, nargs=2, default=(16.0, 522.0), help='where features are sought (Hz)')
    g.add_argument('--ratio', type=float, default=100.0, help='Rmax from which a feature is notched')
    g.add_argument('--extent', type=float, default=2.0)
    g.add_argument('--leak', type=float, default=0.01)
    g.add_argument('--taper', type=float, default=0.25)
    g.add_argument('--merge', type=float, default=1.0, help='notches closer than this (Hz) are merged')
    g.add_argument('--T', type=float, default=8.0)
    g.add_argument('--N', type=int, default=32)
    g.add_argument('--pad', type=float, default=8.0)
    g.add_argument('--fmin', type=float, default=20.0)
    g.add_argument('--fmax', type=float, default=512.0)
    g.add_argument('--published', nargs=2, metavar=('H1_CSV', 'L1_CSV'))
    g.add_argument('--compare', type=float, nargs='*', help='also give the cost of these values of --ratio')
    g.add_argument('--note', action='append', help='a line of text for the header (repeat for more lines)')
    g.add_argument('--save', help='write the spectra to this .npz')
    g.add_argument('--load', help='read the spectra from this .npz instead of the strain')
    s = p.add_argument_group('segments')
    s.add_argument('--subset', type=int, default=4000, help='segments for the matched filter')
    s.add_argument('--out', help='the .npz of the numbers (default <segments_dir>/diagnostics.npz)')
    s.add_argument('--select', help='diagnostics.npz of another run: also report on its analysable segments')
    s.add_argument('--flag-p', type=float, default=1e-6, help='of the thresholds that say which band flagged a block')
    s.add_argument('--simulate', type=int, default=20000, help='Gaussian segments: per-bin and monitor reference')
    s.add_argument('--subset-simulated', type=int, default=4000, help='Gaussian segments for the matched filter')
    s.add_argument('--quiet', type=float, default=6.0, help='a segment is quiet when every band monitor is below 1 + '
                                                              'this many Gaussian standard deviations')
    a = p.parse_args(argv)
    if a.mode == 'lines':
        if len(a.inputs) != (1 if a.load else 2):
            p.error('lines: <gwosc_dir> <notches.txt>, or --load spectra.npz <notches.txt>')
        return run_lines(a)
    return run_segments(a)


if __name__ == '__main__':
    sys.exit(main())
