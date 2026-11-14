"""Controls 4 to 9 of docs/glitch_likelihood.md (Sec. 7): segments with mergers at a known rate and glitches at known
rates and amplitude law, analysed by pe_redshift with the glitch terms of Sec. 4.  plant_fit.py fits the results on the
CPU.  Needs the redshift tables; meant for a GPU.

    python scripts/validate/plant.py tables/redshift gpu --n 2000 --nbar 0.05 --eps-H 0.1 --eps-L 0.03 --out runs/p
    CUDA_VISIBLE_DEVICES=1 python scripts/validate/plant.py tables/redshift gpu --start 204800 --n 204800 --seed 1
        --nbar 0.05 --eps-H 0.1 --eps-L 0.03 --glitch-kind template --gamma 3 --rho-min 6 --out runs/template
    python scripts/validate/plant.py ... --control time-slide --out runs/template_slide [--plain] [--resident]

The plant.  Every segment (T = 8 s, 20 to 512 Hz, the H noise curve in both detectors, exmarg.pp.simulate) holds, with
probability --nbar, one equal-mass zero-spin merger of the population of the analysis: exmarg.priors.population_m1_z
restricted to z >= --zmin (1) and to 20 <= 2 m1 (1 + z) <= 400 (the bank), (m1, z) by inverse cdf of p(m1) and p(z)
with rejection of the mass cut, y = D0 / D_L(z) with the cosmology of pe_redshift, the extrinsic parameters of
pp.simulate.
Then, independently with probability --eps-H and --eps-L, one glitch in that detector (exmarg.glitches), single-detector
SNR dN/drho ~ rho^-gamma on [rho_min, rho_max], arrival time on the matched-filter grid: 'template' (the glitch of the
model: a template of the bank), 'sine-gaussian', 'gaussian', 'ringdown', or 'mixed' (each of the four with probability
1/4).  Then the control (exmarg.search.controls): 'none', 'time-slide' (L by 0.5 s, circularly), 'segment-slide' (H of
segment k with L of segment k + 1), 'phase-randomize', 'time-reverse'.

The analysis.  pe_redshift(glitch=True, chunk=128) with its default fiducial glitch law and window, on 120 equal-mass
templates of total mass 20 to 400 (detector frame), m1 in (4, 200) on 150 points, z up to 10 on 150 points (notebook 7),
the tables loaded with unbiased=True, ymin = D0 / D_L(10) (1 - 1e-6) so that the store reaches the end of the z grid:
pe_redshift's smallest y is ymin (1 + 5e-10), and at ymin = D0 / D_L(10) the row z = 10 (weight 1.3e-4) was left out;
its default 0.1267 (z = 8) would leave out 6.7e-3 (--ymin to change it; w_inf below records what is left out).
Every ln Lambda grid is then averaged over the same restricted population (population.ln_lambda_bar).

Seeds.  Each use of random numbers by segment k has a stream of its own, numpy.random.default_rng(stream(seed, k, use))
with stream = use 2^60 + seed 2^40 + k, distinct for every (seed, k, use) (seed < 2^20, k < 2^40), below 2^63 and
equal to stream(seed, 0, use) + k.  Uses, in USES order: the noise and extrinsic parameters (pp.simulate, given
stream(seed, first, 'noise'), seeds segment k of the batch with exactly that), the merger (presence first, then m1
and z, drawn in every segment), the glitch of H and the glitch of L (one exmarg.glitches call per segment and
detector, that detector drawn first so that presence, SNR and time come first in its stream), the kinds of 'mixed', the
phases of 'phase-randomize', and the arrival times pe_redshift samples (its seed plus the row).  Segment k is
therefore the same data whatever --start, --n and --batch; campaigns with different seeds share no stream; the merger
parameters and the noise do not depend on --nbar, and presence, SNR and time of the glitches do not depend on
--glitch-kind or on the eps of the other detector.

Output: one file plant_<first>_<end>.npz per batch (segments first to end - 1, batches cut at the multiples of
--batch), written under a temporary name and renamed, skipped when present after checking that it belongs to the same
campaign; existing files of another campaign in --out stop the run.  Members: meta (a JSON string: the campaign
settings, the same in every file, and under 'run' the device, host, timings), rho_edges (n_rho + 1,), glitch_pi_k
(n_rho,) and glitch_window, the fiducial law and the window of the lambda terms, and per segment (leading axis n):
    k                                    global segment index
    ln_lambda_bar, ln_lambda_gH_bar, ln_lambda_gL_bar, ln_lambda_gHL_bar, ln_lambda_H_bar, ln_lambda_L_bar
                                         ln of the population average of the ln_lambda* grids of pe_redshift
    ln_lambda_bar_plain                  the same from pe_redshift without glitch (--plain)
    w_inf                                population weight of the cells where ln_lambda is not finite
    w_inf_cells (n_cell, 2)              (m1, z) grid indices of those cells, over the batch (not per segment)
    ln_glitch_H, ln_glitch_L (n, n_rho)  ln of the template average, uniform weights, of E_I[m, k]
    wmax_H, wmax_L                       largest w over the templates
    merger, m1, z, y, snr                the merger planted in segment k (m1, z nan, y and snr 0 when absent)
    merger_H, merger_L, coherent         a merger signal in the H, L data after the control; coherent: aligned in both
    glitch_I, kind_I, rho_I, t0_I        the glitch in the data of detector I after the control (t0 moved with it)
"""
import argparse
import glob
import json
import os
import socket
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exmarg.glitches import KINDS as BURSTS, add_burst_glitches, add_template_glitches, power_law_snr
from exmarg.pp import simulate
from exmarg.priors import population_m1_z
from exmarg.search import controls, population as popavg
from exmarg.search.pe import _dl_of_z
from exmarg.search.segments import D0

T, FMIN, FMAX = 8.0, 20.0, 512.0
MASSES = np.geomspace(20.0, 400.0, 120)
MDET = (20.0, 400.0)
GRID = dict(m1_range=(4.0, 200.0), nm1=150, zmax=10.0, nz=150)
ZMIN = 1.0
SHIFT = 0.5
CHUNK = 128
KINDS = ('template',) + tuple(BURSTS)
CONTROLS = ('none', 'time-slide', 'segment-slide', 'phase-randomize', 'time-reverse')
USES = ('noise', 'merger', 'glitch_H', 'glitch_L', 'kind', 'phase', 'pe')
KEYS = ('ln_lambda', 'ln_lambda_gH', 'ln_lambda_gL', 'ln_lambda_gHL', 'ln_lambda_H', 'ln_lambda_L')
VERSION = 1


def stream(seed, k, use):
    """the integer seed of segment k for one use (USES): distinct for every (seed, k, use), below 2^63, and equal to
    stream(seed, 0, use) + k"""
    seed, k = int(seed), int(k)
    if not (0 <= seed < 2 ** 20 and 0 <= k < 2 ** 40):
        raise ValueError('need 0 <= seed < 2^20 and 0 <= segment index < 2^40')
    return (USES.index(use) << 60) + (seed << 40) + k


def population(zmin=ZMIN):
    """((m1, p_m1), (z, p_z)): exmarg.priors.population_m1_z with p_z zero below zmin (a step at zmin), the population
    of the plant and of the analysis, with the cut MDET on 2 m1 (1 + z)"""
    (m, pm), (z, pz) = population_m1_z()
    below, above = z < zmin - 1e-9, z > zmin
    zr = np.concatenate([z[below], [zmin - 1e-9, zmin], z[above]])
    pr = np.concatenate([np.zeros(below.sum() + 1), [np.interp(zmin, z, pz)], pz[above]])
    return (np.asarray(m, float), np.asarray(pm, float)), (zr, pr)


def _inverse_cdf(x, p):
    """(cdf, x) of the tabulated density p(x) on its support (trapezoid rule), for np.interp(u, cdf, x)"""
    x, p = np.asarray(x, float), np.asarray(p, float)
    on = np.nonzero(p > 0)[0]
    a, b = max(on[0] - 1, 0), min(on[-1] + 1, p.size - 1)
    x, p = x[a:b + 1], p[a:b + 1]
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return c / c[-1], x


class Sampler:
    """(m1, z) of one merger: inverse cdf of p(m1) and of p(z) of pop, rejected outside mdet_range in 2 m1 (1 + z)"""

    def __init__(self, pop, mdet_range=MDET):
        self.m, self.z, self.mdet = _inverse_cdf(*pop[0]), _inverse_cdf(*pop[1]), tuple(mdet_range)

    def __call__(self, rng):
        while True:
            m1 = float(np.interp(rng.uniform(), *self.m))
            z = float(np.interp(rng.uniform(), *self.z))
            if self.mdet[0] <= 2.0 * m1 * (1.0 + z) <= self.mdet[1]:
                return m1, z


def mergers(cfg, ks, sampler):
    """the mergers of the segments ks: present (n,) bool, m1, z (n,) drawn in every segment, y (n,) = D0 / D_L(z) where
    present and 0 elsewhere"""
    n = len(ks)
    present, m1, z = np.zeros(n, bool), np.empty(n), np.empty(n)
    for i, k in enumerate(ks):
        rng = np.random.default_rng(stream(cfg['seed'], k, 'merger'))
        present[i] = rng.uniform() < cfg['nbar']
        m1[i], z[i] = sampler(rng)
    y = np.where(present, D0 / _dl_of_z(z), 0.0) if n else np.zeros(0)
    return present, m1, z, y


def kinds(cfg, ks):
    """(n, 2) the glitch kind of H and of L in each segment of ks ('mixed': each of KINDS with probability 1/4)"""
    if cfg['glitch_kind'] != 'mixed':
        return np.full((len(ks), 2), cfg['glitch_kind'], dtype='<U16')
    pick = [np.random.default_rng(stream(cfg['seed'], k, 'kind')).integers(len(KINDS), size=2) for k in ks]
    return np.array([[KINDS[j] for j in p] for p in pick], dtype='<U16').reshape(len(ks), 2)


def glitch(seg, i, k, I, cfg, kind, draw_rho, dt):
    """the data of detector I in row i (segment k) with its glitch, and the record (present, kind, rho, t0): drawn as
    the first detector of its own exmarg.glitches call, from the stream 'glitch_I' of segment k"""
    row = seg['d_' + I][i:i + 1]
    one = dict(d_H=row, d_L=row, nu=seg['nu'], sig2=seg['sig2'], T=seg['T'], fmin=seg['fmin'], fmax=seg['fmax'])
    eps, seed = {'H': cfg['eps_' + I], 'L': 0.0}, stream(cfg['seed'], k, 'glitch_' + I)
    if kind == 'template':
        new, rec = add_template_glitches(one, eps, draw_rho, MASSES, seed, time_step=dt)
    else:
        new, rec = add_burst_glitches(one, eps, draw_rho, kind, seed, time_step=dt)
    return new['d_H'][0], {key: rec['H'][key][0] for key in ('present', 'kind', 'rho', 't0')}


def simulate_batch(cfg, first, end, sampler):
    """segments first .. end - 1 of the campaign cfg: (the segments dict of pp.simulate with mergers, glitches and the
    control applied, the truth: dict of (end - first,) arrays, see the module docstring)"""
    extra = int(cfg['control'] == 'segment-slide')                     # the L data of segment end
    ks = np.arange(first, end + extra)
    n = ks.size
    present, m1, z, y = mergers(cfg, ks, sampler)
    mdet = 2.0 * m1 * (1.0 + z)
    rows, cur = iter(range(n)), [0]

    def draw(rng):                                                   # pp.simulate calls it once per segment, in order
        cur[0] = next(rows)
        return mdet[cur[0]], 1.0, 0.0

    seg, sim, snr = simulate(n, stream(cfg['seed'], first, 'noise'), draw, lambda rng: y[cur[0]], cfg['T'], cfg['fmin'],
                             cfg['fmax'])
    if not (np.array_equal(sim[:, 0], mdet) and np.array_equal(sim[:, 3], y)):
        raise RuntimeError('pp.simulate did not take the draws in segment order')
    dt = controls.time_step(seg)
    draw_rho = power_law_snr(cfg['gamma'], cfg['rho_min'], cfg['rho_max'])
    kind = kinds(cfg, ks)
    g = {I: dict(present=np.zeros(n, bool), kind=np.full(n, '', '<U16'), rho=np.full(n, np.nan), t0=np.full(n, np.nan))
         for I in 'HL'}
    for j, I in enumerate('HL'):
        if cfg['eps_' + I] <= 0:
            continue
        for i, k in enumerate(ks):
            d, rec = glitch(seg, i, k, I, cfg, kind[i, j], draw_rho, dt)
            if rec['present']:
                seg['d_' + I][i] = d
                for key in g[I]:
                    g[I][key][i] = rec[key]
    mL, control = present.copy(), cfg['control']
    if control == 'time-slide':
        seg = controls.time_slide(seg, cfg['shift'])
        g['L']['t0'] = (g['L']['t0'] + cfg['shift']) % cfg['T']
    elif control == 'segment-slide':
        seg = controls.segment_slide(seg, 1)
        g['L'] = {key: np.roll(v, -1) for key, v in g['L'].items()}
        mL = np.roll(mL, -1)
    elif control == 'phase-randomize':
        pr = [controls.phase_randomize(dict(seg, d_H=seg['d_H'][i:i + 1], d_L=seg['d_L'][i:i + 1]),
                                       stream(cfg['seed'], k, 'phase')) for i, k in enumerate(ks)]
        seg = dict(seg, d_H=np.concatenate([p['d_H'] for p in pr]), d_L=np.concatenate([p['d_L'] for p in pr]))
    elif control == 'time-reverse':
        seg = controls.time_reverse(seg)
        for I in 'HL':
            g[I]['t0'] = (cfg['T'] - g[I]['t0']) % cfg['T']
    elif control != 'none':
        raise ValueError('control must be one of %s' % (CONTROLS,))
    keep = slice(0, end - first)
    seg = dict(seg, d_H=seg['d_H'][keep], d_L=seg['d_L'][keep])
    truth = dict(k=ks, merger=present, m1=np.where(present, m1, np.nan), z=np.where(present, z, np.nan), y=y, snr=snr,
                 merger_H=present, merger_L=mL, coherent=present & (control == 'none'))
    for I in 'HL':
        truth.update({'glitch_' + I: g[I]['present'], 'kind_' + I: g[I]['kind'], 'rho_' + I: g[I]['rho'],
                      't0_' + I: g[I]['t0']})
    return seg, {key: np.asarray(v)[keep] for key, v in truth.items()}


def reduce(res, pop, mdet_range=MDET):
    """the per-segment arrays kept from a pe_redshift result (glitch=True): ln Lbar of the six ln_lambda grids on the
    population pop, w_inf and w_inf_cells, ln of the template average of E_I (n, n_rho) and the largest w"""
    m1, z = res['m1'], res['z']
    out = {key + '_bar': np.atleast_1d(popavg.ln_lambda_bar(res[key], m1, z, mdet_range, pop)) for key in KEYS}
    w = popavg.population_weights(m1, z, mdet_range, pop)
    bad = ~np.isfinite(np.asarray(res['ln_lambda'])) & (w > 0)[None]
    out['w_inf'] = np.array([w[b].sum() for b in bad])
    out['w_inf_cells'] = np.argwhere(bad.any(axis=0)).astype(np.int32)
    for I in 'HL':
        lnE = np.asarray(res['ln_glitch_' + I], float)
        out['ln_glitch_' + I] = logsumexp(lnE, axis=1) - np.log(lnE.shape[1])
        out['wmax_' + I] = np.asarray(res['wmax_' + I], float).max(axis=1)
    return out


def make_analysis(store, device, resident=False, chunk=CHUNK, ymin=None):
    """the analysis behind a narrow interface: factory(segments) builds the bank on these segments (every batch has the
    same frequency grid and noise curve) and returns pe(segments, seed, glitch) -> the pe_redshift dict on GRID, with
    glitch=True or None"""
    from exmarg.search import mass_bank, pe_redshift

    def factory(seg0):
        bank = mass_bank(MASSES, seg0, device)

        def pe(seg, seed, glitch):
            return pe_redshift(seg, bank, store, chunk=chunk, resident=resident, seed=seed, ymin=ymin,
                               mdet_range=MDET, glitch=True if glitch else None, **GRID)
        return pe
    return factory


def batches(start, n, batch):
    """[(first, end)] covering start .. start + n - 1, cut at the multiples of batch"""
    cuts = [start] + list(range((start // batch + 1) * batch, start + n, batch)) + [start + n]
    return list(zip(cuts[:-1], cuts[1:]))


def campaign(args):
    """the settings that fix the data and the analysis: the same in every file of a campaign"""
    ymin = float(D0 / _dl_of_z(GRID['zmax'])[0] * (1.0 - 1e-6)) if args.ymin is None else float(args.ymin)
    cfg = dict(version=VERSION, seed=args.seed, nbar=args.nbar, eps_H=args.eps_H, eps_L=args.eps_L,
               glitch_kind=args.glitch_kind, gamma=args.gamma, rho_min=args.rho_min, rho_max=args.rho_max,
               control=args.control, shift=SHIFT, T=T, fmin=FMIN, fmax=FMAX, zmin=args.zmin, mdet_range=MDET,
               bank=(float(MASSES[0]), float(MASSES[-1]), MASSES.size), grid=GRID, ymin=ymin, n_loud=1024,
               n_random=1024, glitch_settings='pe default')
    return json.loads(json.dumps(cfg))


def read_meta(path):
    with np.load(path) as f:
        return json.loads(str(f['meta']))


def check_campaign(path, cfg):
    meta = read_meta(path)
    meta.pop('run', None)
    if meta != cfg:
        diff = sorted(key for key in set(meta) | set(cfg) if meta.get(key) != cfg.get(key))
        raise SystemExit('%s belongs to another campaign (differs in %s)' % (path, ', '.join(diff)))


def write_batch(path, cfg, run, arrays):
    """one result file, written under a temporary name in the same directory and renamed"""
    tmp = os.path.join(os.path.dirname(path) or '.', '.%s.%s.%d.tmp.npz' % (os.path.basename(path)[:-4],
                                                                           socket.gethostname(), os.getpid()))
    np.savez(tmp, meta=np.array(json.dumps(dict(cfg, run=run))), **arrays)
    os.replace(tmp, path)


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def run(cfg, out, start, n, batch, factory, plain=False, info=None, log=print):
    """the campaign cfg over segments start .. start + n - 1 into the directory out; factory(segments) -> pe (see
    make_analysis) is called once, before the first batch to analyse.  returns the paths written"""
    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(out, 'plant_*.npz'))):
        check_campaign(path, cfg)
    pop = population(cfg['zmin'])
    sampler = Sampler(pop, cfg['mdet_range'])
    pe, done, written, began = None, 0, [], time.time()
    for first, end in batches(start, n, batch):
        path = os.path.join(out, 'plant_%010d_%010d.npz' % (first, end))
        if os.path.exists(path):
            check_campaign(path, cfg)
            done += end - first
            log('%s  [%d, %d) present, skipped; %d of %d segments' % (stamp(), first, end, done, n), flush=True)
            continue
        t0 = time.time()
        seg, truth = simulate_batch(cfg, first, end, sampler)
        t1 = time.time()
        if pe is None:
            pe = factory(seg)
        t2 = time.time()
        res = pe(seg, stream(cfg['seed'], first, 'pe'), True)
        arrays = reduce(res, pop, cfg['mdet_range'])
        t3 = time.time()
        if plain:
            res0 = pe(seg, stream(cfg['seed'], first, 'pe'), False)
            arrays['ln_lambda_bar_plain'] = np.atleast_1d(popavg.ln_lambda_bar(res0['ln_lambda'], res0['m1'], res0['z'],
                                                                               cfg['mdet_range'], pop))
            del res0
        t4 = time.time()
        m = end - first
        times = dict(t_sim=(t1 - t0) / m, t_pe=(t3 - t2) / m, t_plain=(t4 - t3) / m if plain else None)
        arrays.update(truth, rho_edges=np.asarray(res['rho_edges'], float),
                      glitch_pi_k=np.asarray(res.get('glitch_pi_k', np.nan), float),
                      glitch_window=np.asarray(res.get('glitch_window', np.nan), float))
        write_batch(path, cfg, dict(info or {}, host=socket.gethostname(), first=first, end=end, plain=plain,
                                    written=stamp(), **times), arrays)
        written.append(path)
        done += m
        log('%s  [%d, %d) %d of %d segments; simulation %.4f s, analysis %.4f s%s per segment; %.0f s'
            % (stamp(), first, end, done, n, times['t_sim'], times['t_pe'],
               ', plain %.4f s' % times['t_plain'] if plain else '', time.time() - began), flush=True)
    return written


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('store', help='folder of the redshift tables (bd_*.npz)')
    p.add_argument('device', choices=('gpu', 'cpu'))
    p.add_argument('--n', type=int, required=True, help='number of segments of this run')
    p.add_argument('--start', type=int, default=0, help='index of the first segment')
    p.add_argument('--seed', type=int, default=1, help='seed of the campaign, 0 <= seed < 2^20')
    p.add_argument('--nbar', type=float, required=True, help='probability of a merger (z >= zmin) in a segment')
    p.add_argument('--eps-H', dest='eps_H', type=float, default=0.0, help='probability of a glitch in H')
    p.add_argument('--eps-L', dest='eps_L', type=float, default=0.0, help='probability of a glitch in L')
    p.add_argument('--glitch-kind', default='template', choices=KINDS + ('mixed',))
    p.add_argument('--gamma', type=float, default=3.0, help='dN/drho ~ rho^-gamma')
    p.add_argument('--rho-min', type=float, default=6.0)
    p.add_argument('--rho-max', type=float, default=200.0, help='default: the last amplitude bin edge of the model')
    p.add_argument('--control', default='none', choices=CONTROLS)
    p.add_argument('--batch', type=int, default=512, help='segments simulated and analysed at once')
    p.add_argument('--chunk', type=int, default=CHUNK, help='segments filtered at once inside pe_redshift')
    p.add_argument('--ymin', type=float, default=None, help='ymin of pe_redshift (default D0 / D_L(10) (1 - 1e-6))')
    p.add_argument('--zmin', type=float, default=ZMIN, help='smallest z of the population (plant and analysis); '
                   '0.99119, the lower edge of the grid cell at z = 1.025, puts the step on a cell edge')
    p.add_argument('--resident', action='store_true', help='keep every table on the GPU (needs ~48 GB)')
    p.add_argument('--plain', action='store_true', help='also run pe_redshift without glitch, to record its cost')
    p.add_argument('--out', required=True, help='directory of the result files')
    return p


def main(argv=None, factory=None):
    """factory: replaces the analysis with the tables (tests)"""
    p = parser()
    a = p.parse_args(argv)
    if not (0 <= a.nbar <= 1 and 0 <= a.eps_H <= 1 and 0 <= a.eps_L <= 1):
        p.error('nbar and eps must lie in [0, 1]')
    if not (0 < a.rho_min < a.rho_max) or a.n < 1 or a.batch < 1 or a.start < 0 or not 0 <= a.seed < 2 ** 20:
        p.error('need 0 < rho_min < rho_max, n >= 1, batch >= 1, start >= 0, 0 <= seed < 2^20')
    cfg = campaign(a)
    if factory is None:
        if not glob.glob(os.path.join(a.store, 'bd_*.npz')):
            p.error('no bd_*.npz in %s' % a.store)

        def factory(seg):
            from exmarg.search import load_redshift_store
            t = time.time()
            store = load_redshift_store(a.store, pinned=(a.device == 'gpu'), unbiased=True)
            print('%s  store loaded, %d nodes (%.0f s)' % (stamp(), store.s.size, time.time() - t), flush=True)
            return make_analysis(store, a.device, a.resident, a.chunk, cfg['ymin'])(seg)
    print('%s  campaign %s; segments %d to %d in batches of %d into %s'
          % (stamp(), json.dumps(cfg), a.start, a.start + a.n - 1, a.batch, a.out), flush=True)
    run(cfg, a.out, a.start, a.n, a.batch, factory, a.plain,
        dict(device=a.device, chunk=a.chunk, resident=a.resident, pid=os.getpid()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
