"""The unit mean and nbar of the time slides of the analysed week taken together (analyse.py --variants slide+2.25, ...):
every slide is a null control of the same segments, so all of them together give more null data than the two of the
frozen set; summarize.py has the verdicts, this script the numbers that say how the evidence behaves in the null.

    python scripts/real_data/slides_mean.py runs/week [--veto-w 72] [--blocks 32]

Every folder slide<+ or ->seconds of <out> must be complete (summarize.check_week: each segment k = 0 .. n_selected - 1
once, one campaign) and a control of the week (meta control = the folder, nbar 0, no merger known); another folder
named slide*, a folder that is not a slide, or <out> being a variant folder, is refused.  The recorded week, pr, rev and
inj are never opened (only the slide folders are read, and only k, wmax_H, wmax_L and ln_B_bar or ln_lambda_bar).

The statistic is Bbar of the window model (ln_B_bar) in files that carry it, else, in files of the mixture, the Gaussian
Lambda_bar (ln_lambda_bar; said in the output); every slide must carry the same (a mixture of the two is refused).  A
segment is set aside when wmax_H or wmax_L exceeds --veto-w, as plant_fit.py does.  Per slide: the kept segments, the
mean of B - 1 with its standard error (std / sqrt n, the segments taken as independent), the number of segments that
carry half of sum |B - 1|, the largest ln B and its k, and nbar and its error (plant_fit.nbar_fit, maximizing sum ln(1 +
nbar (B - 1)), the error from the curvature).

All slides together: the mean of B - 1 over all (segment, slide) pairs, and nbar of the pooled sum of the log-likelihoods
(nbar_fit of every B of every slide), each with a delete-one-block jackknife error.  The slides share the single-detector
data (L is shifted against H), so the pairs of one segment are correlated and the naive error of the pairs is too
small: the blocks are --blocks groups of consecutive k (k G // n_selected), and the same block is removed from every
slide at once; the error is sqrt((G - 1) / G sum_b (t_(-b) - mean t_(-b))^2) over the blocks that hold a segment.  The
curvature error of the pooled nbar, which takes the pairs as independent, and the naive error of the mean are printed
beside them.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize

plant_fit = summarize.plant_fit
BLOCKS = 32
KEYS = ('k', 'wmax_H', 'wmax_L')


def slide_folders(out):
    """the folders slide* of out, by shift; SystemExit when out is a variant folder, a name is not slide<+ or ->number,
    or there is none"""
    if glob.glob(os.path.join(out, 'plant_*.npz')):
        raise SystemExit('%s holds plant_*.npz: give the folder of the week (with the folders slide*), not a variant' % out)
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, 'slide*')) if os.path.isdir(p))
    bad = [v for v in names if not summarize.SLIDE.match(v)]
    if bad:
        raise SystemExit('%s: %s: not a time slide folder (slide<+ or ->seconds); nothing is read' % (out, ', '.join(bad)))
    if not names:
        raise SystemExit('%s: no folder slide<+ or ->seconds' % out)
    return sorted(names, key=lambda v: float(v[5:]))


def read_slide(out, variant, veto_w):
    """the kept segments of one slide in order of k: dict k, ln (ln of the statistic), stat ('window' or 'gauss'), and
    n_all; SystemExit when the files are not those of a slide control of the week"""
    files = sorted(glob.glob(os.path.join(out, variant, 'plant_*.npz')))
    parts, stat = [], None
    for path in files:
        with np.load(path) as z:
            meta = json.loads(str(z['meta']))
            this = 'window' if 'ln_B_bar' in z.files else 'gauss' if 'ln_lambda_bar' in z.files else None
            if this is None or not set(KEYS) <= set(z.files):
                raise SystemExit('%s: needs k, wmax_H, wmax_L and ln_B_bar or ln_lambda_bar' % path)
            if stat not in (None, this):
                raise SystemExit('%s: the files of %s do not all carry the same statistic' % (path, variant))
            stat = this
            part = {key: np.asarray(z[key]) for key in KEYS}
            part['ln'] = np.asarray(z['ln_B_bar' if this == 'window' else 'ln_lambda_bar'], float)
        if meta.get('control') != variant or meta.get('nbar') != 0 or meta.get('merger_truth') != 'none':
            raise SystemExit('%s: not a time slide control of the week (meta control %r, nbar %r, merger_truth %r)'
                             % (path, meta.get('control'), meta.get('nbar'), meta.get('merger_truth')))
        parts.append(part)
    d = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    order = np.argsort(d['k'], kind='stable')
    d = {key: v[order] for key, v in d.items()}
    keep = (d['wmax_H'] <= veto_w) & (d['wmax_L'] <= veto_w)
    return dict(k=d['k'][keep], ln=d['ln'][keep], stat=stat, n_all=int(d['k'].size))


def blocks_of(k, n, G):
    """the block (0 .. G - 1) of each segment k of n: consecutive k, G blocks of about n / G"""
    return (np.asarray(k, np.int64) * G) // n


def jackknife(estimate, blocks):
    """(estimate on everything, jackknife error, replicates): estimate(mask over the pooled pairs) with each block of
    blocks (the pooled pairs' blocks) removed in turn, over the blocks that hold a pair"""
    full = estimate(np.ones(blocks.size, bool))
    present = np.unique(blocks)
    reps = np.array([estimate(blocks != b) for b in present])
    g = reps.size
    if g < 2:
        return full, np.nan, reps
    return full, float(np.sqrt((g - 1) / g * np.sum((reps - reps.mean()) ** 2))), reps


def analyse(out, veto_w=plant_fit.VETO_W, G=BLOCKS):
    """the numbers of every slide of out and of all together: dict n_selected, slides {name: dict kept, aside, mean, se,
    half, max, k_max, nbar, err}, stat, pooled dict pairs, mean, jack_mean, naive_se, nbar, jack_nbar, curv_nbar,
    blocks; lines (the text)"""
    names = slide_folders(out)
    n, _ = summarize.check_week(out, names)
    data = {v: read_slide(out, v, veto_w) for v in names}
    stats = {d['stat'] for d in data.values()}
    if len(stats) > 1:
        raise SystemExit('%s: slides of the window model (%s) and of the mixture (%s) cannot be pooled' % (
            out, ', '.join(v for v in names if data[v]['stat'] == 'window'),
            ', '.join(v for v in names if data[v]['stat'] == 'gauss')))
    stat = stats.pop()
    sym = 'Bbar' if stat == 'window' else 'Lambda_bar'
    L = ['the time slides of %s: %s; %d selected segments per slide' % (out, ', '.join(names), n),
         'statistic %s%s; segments with wmax_H or wmax_L above %g set aside' % (
             sym, ' of the window model (ln_B_bar)' if stat == 'window' else ' of the Gaussian likelihood (ln_lambda_bar): '
             'these files carry the mixture, not the window model, so only the Gaussian likelihood is used', veto_w), '',
         '%-12s %6s %-20s %-13s %8s %11s %6s   %s' % ('slide', 'kept', 'mean(%s - 1) +- se' % sym, 'mean / se',
                                                     'half-sum', 'max ln %s' % sym[0], 'at k', 'nbar +- error')]
    slides = {}
    for v in names:
        d = data[v]
        mu, se, c = plant_fit.mean_se(plant_fit._excess(d['ln']))
        fit = plant_fit.nbar_fit(d['ln'])
        i = int(np.argmax(d['ln']))
        slides[v] = dict(kept=c, aside=d['n_all'] - c, mean=mu, se=se, half=plant_fit.heavy_tail(d['ln']),
                         max=float(d['ln'][i]), k_max=int(d['k'][i]), nbar=fit['nbar'], err=fit['err']['nbar'],
                         edge=fit['edge'])
        s = slides[v]
        L.append('%-12s %6d %+8.3e +- %-8.2e %+7.1f sigma %8d %11.3f %6d   %s +- %s%s' % (
            v, s['kept'], mu, se, mu / se if se else np.nan, s['half'], s['max'], s['k_max'], plant_fit._f(s['nbar']),
            plant_fit._f(s['err']), ' (at the %s edge)' % s['edge'] if s['edge'] else ''))
    ln = np.concatenate([data[v]['ln'] for v in names])
    blk = np.concatenate([blocks_of(data[v]['k'], n, G) for v in names])
    x = plant_fit._excess(ln)
    with np.errstate(over='ignore', invalid='ignore'):
        mean = lambda m: float(x[m].mean())
        mu, jse, _ = jackknife(mean, blk)
        nb = lambda m: plant_fit.nbar_fit(ln[m])['nbar']
        nbar, jnb, _ = jackknife(nb, blk)
    naive = plant_fit.mean_se(x)[1]
    curv = plant_fit.nbar_fit(ln)
    pooled = dict(pairs=int(ln.size), mean=mu, jack_mean=jse, naive_se=naive, nbar=nbar, jack_nbar=jnb,
                  curv_nbar=curv['err']['nbar'], blocks=int(np.unique(blk).size), edge=curv['edge'])
    L += ['', 'all %d slides together, %d (segment, slide) pairs; delete-one-block jackknife over %d blocks of consecutive k '
          '(the same block removed from every slide):' % (len(names), ln.size, pooled['blocks']),
          '  mean of %s - 1: %+.3e +- %.2e (jackknife), %+.1f sigma; naive standard error of the pairs %.2e (x%.2f)'
          % (sym, mu, jse, mu / jse if jse else np.nan, naive, jse / naive if naive else np.nan),
          '  nbar of the pooled sum of the log-likelihoods: %s +- %s (jackknife)%s; curvature error, pairs taken as '
          'independent: %s' % (plant_fit._f(nbar), plant_fit._f(jnb), ' (at the %s edge)' % curv['edge'] if curv['edge']
                               else '', plant_fit._f(curv['err']['nbar']))]
    return dict(n_selected=n, slides=slides, stat=stat, pooled=pooled, lines=L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', help='the output folder of analyse.py (<out>/slide*/plant_*.npz)')
    p.add_argument('--veto-w', type=float, default=plant_fit.VETO_W, help='set aside a segment whose wmax_H or wmax_L '
                   'exceeds this (as plant_fit.py)')
    p.add_argument('--blocks', type=int, default=BLOCKS, help='blocks of consecutive k of the jackknife (default %d)'
                   % BLOCKS)
    a = p.parse_args(argv)
    if a.blocks < 2:
        p.error('need blocks >= 2')
    print('\n'.join(analyse(a.out, a.veto_w, a.blocks)['lines']), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
