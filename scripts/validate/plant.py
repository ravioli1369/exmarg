"""Controls 4 to 9 of docs/glitch_likelihood.md (Sec. 7): segments with mergers at a known rate and glitches at known
rates and amplitude law, analysed by pe_redshift with the glitch terms of Sec. 4.  plant_fit.py fits the results on the
CPU.  Needs the redshift tables; meant for a GPU.

    python scripts/validate/plant.py tables/redshift gpu --n 2000 --nbar 0.05 --eps-H 0.1 --eps-L 0.03 --out runs/p
    CUDA_VISIBLE_DEVICES=1 python scripts/validate/plant.py tables/redshift gpu --start 204800 --n 204800 --seed 1
        --nbar 0.05 --eps-H 0.1 --eps-L 0.03 --glitch-kind template --gamma 3 --rho-min 6 --out runs/template
    python scripts/validate/plant.py ... --control time-slide --out runs/template_slide [--plain] [--resident]
    python scripts/validate/plant.py none gpu --stage windows --n 204800 --nbar 0.3 --glitch-states 0.9:0.002,0.1:0.05
        --out runs/states                                   # stage 1, then: windows_fit.py runs/states
    python scripts/validate/plant.py tables/redshift gpu --n 204800 --nbar 0.3 --glitch-states 0.9:0.002,0.1:0.05
        --glitch-model runs/states/glitch_model.json --out runs/states    # stage 2, then: plant_fit.py runs/states

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
phases of 'phase-randomize', the arrival times pe_redshift samples (its seed plus the row), and the glitches of the
window model (below).  Segment k is
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
    wnet                                 largest w_H(t + tau) + w_L(t) over the templates (pe_redshift's wnet: every
                                         arrival time and lag of the tables; the network cut of plant_fit.py)
    merger, m1, z, y, snr                the merger planted in segment k (m1, z nan, y and snr 0 when absent)
    merger_H, merger_L, coherent         a merger signal in the H, L data after the control; coherent: aligned in both
    glitch_I, kind_I, rho_I, t0_I        the glitch in the data of detector I after the control (t0 moved with it)

Glitches of the window model (docs/glitch_likelihood.md, Sec. 4a), --glitch-states f1:q1,f2:q2 for both detectors or
--glitch-states-H, --glitch-states-L (not with --eps-H, --eps-L): each detector-segment is in state s with probability
f_s, and each of its --n-window windows (n_time / n_window matched-filter samples, n_time = 8 nu_max) holds a glitch
with probability q_s, independently, at a sample uniform in the window, SNR from the law of --gamma, --rho-min,
--rho-max, kind from --glitch-kind ('mixed': drawn per glitch), added with exmarg.glitches.glitch_signal.  Streams:
child j of numpy.random.SeedSequence(stream(seed, k, 'windows')) (j = 0 for H, 1 for L) draws the state, then for
every window its presence, SNR and sample, whatever the state, q or kind; child 2 + j the kind and shape of each
glitch in window order.  Noise and mergers are those of the campaign with the same seed without glitches or with
--eps glitches.  Per segment glitch_I, kind_I, rho_I, t0_I then describe the loudest glitch of detector I, and
    glitch_state_I, n_glitches_I         its state (index into the f:q pairs) and number of glitches
The campaign records glitch_states = {H, L: [[f, q], ...], n_window}.

Unequal noise curves (docs/2026-11-15_unequal_noise_design.md): --asym W puts the noise of detector I on sig2_I =
sig2 / (1 +- W) (the same random numbers: every template then has w = W), --curves EPOCH on the two curves of a
conditioned epoch file (strain.own_sig2); mergers and glitches keep their signals and SNR on those curves.  The noise
is drawn in the frequency domain on known curves, so the signal gain of the conditioning (strain.signal_gain) is 1:
the planted signals carry none and the bank of the analysis takes none (it emulates conditioned data only in its
curves, not in its estimation noise).  Recorded as
noise {asym} or {curves, sha256}.  The store argument may list several folders (the mixture of pe_redshift, recorded as
tables_asym), --asym-tol (default 0.05, with unequal curves or several stores; recorded, and asym_outside per segment),
--zmin-cut (pe_redshift(zmin=--zmin)), --mmap (unbiased caches mapped).

Stage 1 of the window model, --stage windows (no tables; of the store argument only the lags of its header are read,
for wnet, when it names a store folder or cache: with 'none' the files have no wnet and plant_fit.py and windows_fit.py
apply the single-detector veto alone): the same segments, batches
and controls, through pe.window_evidences(n_window=--n-window) on the bank of the analysis, into
win_<first>_<end>.npz next to the plant files of the campaign, with meta (the campaign without the settings of the
merger stage, and stage 'windows', n_window, window_rho), window_rho_edges (n_bin + 1,) the edges of the amplitude bins
of RHO_EDGES that overlap [--window-rho-min, --window-rho-max], window_bin0 the index of the first, and per segment k,
the truth arrays as above, wmax_H, wmax_L, wnet (with the lags) and
    ln_window_H, ln_window_L (n, n_window, n_bin) float32   ln E_w[k] of each window on those bins
windows_fit.py fits them and writes glitch_model.json.  Stage 2, --glitch-model glitch_model.json: pe_redshift with
glitch=dict(model='windows', ...) of that file (refused when the file was modified, its n_window is not --n-window or
it was fitted on another campaign, campaign_key); the campaign records glitch_model 'windows', glitch_model_sha256 and
n_window, and the files hold ln_B_bar, ln_lambda_bar, ln_lambda_H_bar, ln_lambda_L_bar (as above), ln_c_H, ln_c_L
(n,) and state_H, state_L (n, K) the noise ratio and posterior of the state of each detector, ln_glitch_I, wmax_I,
w_inf, and no ln_lambda_g*.  The files of the two stages of one folder must be of the same campaign_key.
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
from scipy.special import logsumexp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exmarg.glitches import KINDS as BURSTS, add_burst_glitches, add_template_glitches, glitch_signal, power_law_snr
from exmarg.pp import simulate
from exmarg.priors import population_m1_z
from exmarg.search import controls, population as popavg
from exmarg.search.pe import _dl_of_z
from exmarg.search.segments import D0
from exmarg.search.single import RHO_EDGES
from exmarg.search.store import is_cache, read_header

T, FMIN, FMAX = 8.0, 20.0, 512.0
MASSES = np.geomspace(20.0, 400.0, 120)
MDET = (20.0, 400.0)
GRID = dict(m1_range=(4.0, 200.0), nm1=150, zmax=10.0, nz=150)
ZMIN = 1.0
ASYM_TOL = 0.05                             # how far outside the stores' asym a template may be read from the end store
SHIFT = 0.5
CHUNK = 128
KINDS = ('template',) + tuple(BURSTS)
CONTROLS = ('none', 'time-slide', 'segment-slide', 'phase-randomize', 'time-reverse')
USES = ('noise', 'merger', 'glitch_H', 'glitch_L', 'kind', 'phase', 'pe', 'windows')
KEYS = ('ln_lambda', 'ln_lambda_gH', 'ln_lambda_gL', 'ln_lambda_gHL', 'ln_lambda_H', 'ln_lambda_L')
KEYS_W = ('ln_lambda', 'ln_B', 'ln_lambda_H', 'ln_lambda_L')        # the grids of pe_redshift with the window model
VERSION = 1
STAGES = ('merger', 'windows')
N_WINDOW = 64
WINDOW_RHO = (3.0, 12.2)
WINDOW_CHUNK = 8                            # segments filtered at once by window_evidences (its default)
# the campaign settings of the merger stage alone, and those of the stages: campaign_key leaves them out
MERGER_KEYS = ('grid', 'ymin', 'n_loud', 'n_random', 'glitch_settings', 'tables', 'shared_times', 'glitch_model',
               'glitch_model_sha256', 'tables_asym', 'zmin_cut', 'asym_tol', 'signal_gain')
STAGE_KEYS = ('stage', 'n_window', 'window_rho')
PREFIX = dict(merger='plant', windows='win')


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


def curve(seg, I):
    """the noise curve of detector I of a segments dict (sig2_I, or the one sig2)"""
    return seg['sig2_' + I] if 'sig2_' + I in seg else seg['sig2']


_CURVES = {}


def noise_curves(cfg):
    """{I: sig2_I} of the campaign's noise (None: the H design curve in both detectors, pp.simulate): asym W gives
    sig2_H = sig2 / (1 + W), sig2_L = sig2 / (1 - W) (every template has w = W); curves the two curves of a conditioned
    epoch file (strain.own_sig2), checked against the SHA-256 recorded"""
    noise = cfg.get('noise')
    if noise is None:
        return None
    nu = np.arange(int(np.ceil(cfg['fmin'] * cfg['T'])), int(np.floor(cfg['fmax'] * cfg['T'])) + 1)
    if 'asym' in noise:
        from exmarg.detectors import S
        sig2, W = cfg['T'] * S('H', nu / cfg['T']) / 2.0, noise['asym']
        return dict(H=sig2 / (1.0 + W), L=sig2 / (1.0 - W))
    if noise['curves'] not in _CURVES:
        _CURVES[noise['curves']] = epoch_curves(noise['curves'], nu)
    sig2, sha = _CURVES[noise['curves']]
    if sha != noise['sha256']:
        raise SystemExit('%s: the noise curves changed since the campaign recorded them' % noise['curves'])
    return sig2


def epoch_curves(path, nu):
    """({I: sig2_I}, SHA-256 of the two arrays) of a conditioned epoch file, whose bins must be nu"""
    from exmarg import strain
    ep = strain.read_epoch(path)
    if not np.array_equal(np.asarray(ep['nu']), nu):
        raise SystemExit('%s: the bins of the epoch are not those of the plant' % path)
    sig2 = strain.own_sig2(ep)
    return sig2, hashlib.sha256(np.concatenate([sig2['H'], sig2['L']]).astype('<f8').tobytes()).hexdigest()


def glitch(seg, i, k, I, cfg, kind, draw_rho, dt):
    """the data of detector I in row i (segment k) with its glitch, and the record (present, kind, rho, t0): drawn as
    the first detector of its own exmarg.glitches call, from the stream 'glitch_I' of segment k"""
    row = seg['d_' + I][i:i + 1]
    one = dict(d_H=row, d_L=row, nu=seg['nu'], sig2=curve(seg, I), T=seg['T'], fmin=seg['fmin'], fmax=seg['fmax'])
    eps, seed = {'H': cfg['eps_' + I], 'L': 0.0}, stream(cfg['seed'], k, 'glitch_' + I)
    if kind == 'template':
        new, rec = add_template_glitches(one, eps, draw_rho, MASSES, seed, time_step=dt)
    else:
        new, rec = add_burst_glitches(one, eps, draw_rho, kind, seed, time_step=dt)
    return new['d_H'][0], {key: rec['H'][key][0] for key in ('present', 'kind', 'rho', 't0')}


def parse_states(text):
    """[[f, q], ...] from 'f1:q1,f2:q2,...': the probability of each state of a detector-segment and that of a glitch
    in a window in that state (ValueError unless f >= 0 sums to 1 and 0 <= q <= 1)"""
    try:
        fq = [[float(x) for x in item.split(':')] for item in text.split(',') if item.strip()]
    except ValueError:
        raise ValueError('glitch states %r: need f1:q1,f2:q2,...' % text)
    if not fq or any(len(p) != 2 for p in fq):
        raise ValueError('glitch states %r: need f1:q1,f2:q2,...' % text)
    f, q = np.array(fq).T
    if np.any(f < 0) or abs(f.sum() - 1.0) > 1e-9 or np.any((q < 0) | (q > 1)):
        raise ValueError('glitch states %r: need f >= 0 summing to 1 and 0 <= q <= 1' % text)
    return fq


def window_glitches(seg, i, k, I, cfg, draw_rho, dt):
    """the glitches of the window model in the data of detector I, row i (segment k), see the module docstring:
    (their sum (n_freq,) complex, the record: state, n, and present, kind, rho, t0 of the loudest)"""
    st = cfg['glitch_states']
    f, q = np.asarray(st[I], float).T
    nw = st['n_window']
    n_time = int(round(cfg['T'] / dt))
    if n_time % nw:
        raise ValueError('n_time = %d is not a multiple of n_window = %d' % (n_time, nw))
    nwin, j = n_time // nw, 'HL'.index(I)
    root = stream(cfg['seed'], k, 'windows')
    rng = np.random.default_rng(np.random.SeedSequence(root, spawn_key=(j,)))
    u = rng.uniform()
    u_on = rng.uniform(size=nw)
    rho = np.array([draw_rho(rng) for _ in range(nw)])
    sample = rng.integers(nwin, size=nw)
    s = min(int(np.searchsorted(np.cumsum(f) / f.sum(), u, side='right')), f.size - 1)
    on = np.flatnonzero(u_on < q[s])
    shape = np.random.default_rng(np.random.SeedSequence(root, spawn_key=(2 + j,)))
    one = dict(nu=seg['nu'], sig2=curve(seg, I), T=seg['T'], fmin=seg['fmin'], fmax=seg['fmax'])
    h, kinds_ = np.zeros(np.size(seg['nu']), complex), []
    for w in on:
        kind = KINDS[shape.integers(len(KINDS))] if cfg['glitch_kind'] == 'mixed' else cfg['glitch_kind']
        h += glitch_signal(one, I, kind, rho[w], (w * nwin + sample[w]) * dt, shape, masses=MASSES)[0]
        kinds_.append(kind)
    rec = dict(state=s, n=on.size, present=on.size > 0, kind='', rho=np.nan, t0=np.nan)
    if on.size:
        top = int(np.argmax(rho[on]))
        rec.update(kind=kinds_[top], rho=rho[on[top]], t0=(on[top] * nwin + sample[on[top]]) * dt)
    return h, rec


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
                             cfg['fmax'], sig2=noise_curves(cfg))
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
    states = cfg.get('glitch_states')
    for I in 'HL' if states else '':
        g[I].update(state=np.zeros(n, int), n=np.zeros(n, int))
        for i, k in enumerate(ks):
            h, rec = window_glitches(seg, i, k, I, cfg, draw_rho, dt)
            if rec['n']:
                seg['d_' + I][i] = seg['d_' + I][i] + h
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
        if states:
            truth.update({'glitch_state_' + I: g[I]['state'], 'n_glitches_' + I: g[I]['n']})
    return seg, {key: np.asarray(v)[keep] for key, v in truth.items()}


def reduce(res, pop, mdet_range=MDET):
    """the per-segment arrays kept from a pe_redshift result (glitch=True): ln Lbar of the six ln_lambda grids on the
    population pop, w_inf and w_inf_cells, ln of the template average of E_I (n, n_rho), the largest w and, when res
    has it, the largest wnet over the templates.  With the
    window model (ln_B in res): ln Lbar of the grids KEYS_W, w_inf over the cells where ln_lambda or ln_B is not
    finite, and ln_c_I (n,), state_I (n, K) as given.  With asym_outside in res (pe_redshift with asym_tol) that number
    for every segment"""
    m1, z = res['m1'], res['z']
    windows = 'ln_B' in res
    out = {key + '_bar': np.atleast_1d(popavg.ln_lambda_bar(res[key], m1, z, mdet_range, pop))
           for key in (KEYS_W if windows else KEYS)}
    w = popavg.population_weights(m1, z, mdet_range, pop)
    bad = ~np.isfinite(np.asarray(res['ln_lambda'])) & (w > 0)[None]
    if windows:
        bad |= ~np.isfinite(np.asarray(res['ln_B'])) & (w > 0)[None]
    out['w_inf'] = np.array([w[b].sum() for b in bad])
    if 'asym_outside' in res:                         # templates read from an end store (pe_redshift asym_tol), per call
        out['asym_outside'] = np.full(bad.shape[0], int(res['asym_outside']))
    out['w_inf_cells'] = np.argwhere(bad.any(axis=0)).astype(np.int32)
    for I in 'HL':
        lnE = np.asarray(res['ln_glitch_' + I], float)
        out['ln_glitch_' + I] = logsumexp(lnE, axis=1) - np.log(lnE.shape[1])
        out['wmax_' + I] = np.asarray(res['wmax_' + I], float).max(axis=1)
        if windows:
            out['ln_c_' + I] = np.asarray(res['ln_c_' + I], float)
            out['state_' + I] = np.asarray(res['state_' + I], float)
    if 'wnet' in res:
        out['wnet'] = np.asarray(res['wnet'], float).max(axis=1)
    return out


def window_bins(rho_min, rho_max):
    """(n_rho,) bool: the amplitude bins of RHO_EDGES that overlap [rho_min, rho_max], contiguous"""
    bins = (RHO_EDGES[1:] > rho_min) & (RHO_EDGES[:-1] < rho_max)
    if not bins.any():
        raise ValueError('no amplitude bin overlaps [%g, %g]' % (rho_min, rho_max))
    return bins


def reduce_windows(res, bins):
    """the per-segment arrays kept from a pe.window_evidences result: ln_window_I (n, n_window, bins.sum()) float32 on
    the amplitude bins `bins` of RHO_EDGES, and the largest w over the templates wmax_I (n,); wnet (n,), the largest
    over the templates, when res has it (window_evidences with lags)"""
    if not np.array_equal(np.asarray(res['rho_edges'], float), RHO_EDGES):
        raise ValueError('window_evidences returned other amplitude bins than single.RHO_EDGES')
    out = {}
    for I in 'HL':
        out['ln_window_' + I] = np.asarray(res['ln_window_' + I])[:, :, bins].astype(np.float32)
        out['wmax_' + I] = np.asarray(res['wmax_' + I], float).max(axis=1)
    if 'wnet' in res:
        out['wnet'] = np.asarray(res['wnet'], float).max(axis=1)
    return out


def window_shared(bins):
    """the members of a win file that are not per segment: the edges of the bins kept and the index of the first"""
    b = np.flatnonzero(bins)
    return dict(window_rho_edges=RHO_EDGES[b[0]:b[-1] + 2].copy(), window_bin0=np.array(b[0]))


def make_analysis(store, device, resident=False, chunk=CHUNK, ymin=None, shared_times=False, zmin=None, asym_tol=None):
    """the analysis behind a narrow interface: factory(segments) builds the bank on these segments (every batch has the
    same frequency grid and noise curves) and returns pe(segments, seed, glitch) -> the pe_redshift dict on GRID, with
    glitch=True or None, or the dict of the window model (glitch_model); store one store or a list (the mixture), zmin
    and asym_tol those of pe_redshift (each given only when not None)"""
    from exmarg.search import mass_bank, pe_redshift
    cut = {k: v for k, v in (('zmin', zmin), ('asym_tol', asym_tol)) if v is not None}

    def factory(seg0):
        bank = mass_bank(MASSES, seg0, device)

        def pe(seg, seed, glitch):
            g = glitch if isinstance(glitch, dict) else (True if glitch else None)
            return pe_redshift(seg, bank, store, chunk=chunk, resident=resident, seed=seed, ymin=ymin,
                               mdet_range=MDET, glitch=g, shared_times=shared_times, **cut, **GRID)
        return pe
    return factory


def make_windows(device, n_window=N_WINDOW, chunk=WINDOW_CHUNK, lags=None):
    """stage 1 of the window model behind the same interface: factory(segments) builds the bank on these segments and
    returns win(segments) -> the dict of pe.window_evidences (no tables; with lags, those of the tables, also wnet)"""
    from exmarg.search import mass_bank
    from exmarg.search import pe as pe_module

    def factory(seg0):
        bank = mass_bank(MASSES, seg0, device)

        def win(seg):
            return pe_module.window_evidences(seg, bank, chunk=chunk, n_window=n_window,
                                              **({} if lags is None else dict(lags=lags)))
        return win
    return factory


def store_lags(store, log=print):
    """the lags of the tables named by the store argument (comma-separated folders or unbiased caches; the header only),
    for the wnet of the windows stage; None, said in the log, when it names none ('none' or a folder without tables);
    SystemExit when the folders have different lags"""
    dirs = [d for d in store.split(',') if d and (glob.glob(os.path.join(d, 'bd_*.npz')) or is_cache(d))]
    if not dirs:
        log('%s  %s holds no tables: the windows stage writes no wnet (the network cut of plant_fit.py and '
            'windows_fit.py needs it; give the store folder for its lags)' % (stamp(), store), flush=True)
        return None
    lags = [np.asarray(read_header(d)['tau_idx'], int).ravel() for d in dirs]
    if any(not np.array_equal(x, lags[0]) for x in lags[1:]):
        raise SystemExit('the store folders %s have different lags' % ', '.join(dirs))
    return lags[0]


def batches(start, n, batch):
    """[(first, end)] covering start .. start + n - 1, cut at the multiples of batch"""
    cuts = [start] + list(range((start // batch + 1) * batch, start + n, batch)) + [start + n]
    return list(zip(cuts[:-1], cuts[1:]))


def default_ymin():
    """ymin of pe_redshift: D0 / D_L(zmax) (1 - 1e-6), so that the row z = zmax of the grid is inside the store"""
    return float(D0 / _dl_of_z(GRID['zmax'])[0] * (1.0 - 1e-6))


def campaign(args):
    """the settings that fix the data and the analysis: the same in every file of a campaign"""
    ymin = default_ymin() if args.ymin is None else float(args.ymin)
    cfg = dict(version=VERSION, seed=args.seed, nbar=args.nbar, eps_H=args.eps_H, eps_L=args.eps_L,
               glitch_kind=args.glitch_kind, gamma=args.gamma, rho_min=args.rho_min, rho_max=args.rho_max,
               control=args.control, shift=SHIFT, T=T, fmin=FMIN, fmax=FMAX, zmin=args.zmin, mdet_range=MDET,
               bank=(float(MASSES[0]), float(MASSES[-1]), MASSES.size), grid=GRID, ymin=ymin, n_loud=1024,
               n_random=1024, glitch_settings='pe default')
    if getattr(args, 'published_tables', False):
        cfg['tables'] = 'published'
    if getattr(args, 'shared_times', False):
        cfg['shared_times'] = True
    if getattr(args, 'asym', None) is not None:
        cfg['noise'] = dict(asym=args.asym)
    elif getattr(args, 'curves', None) is not None:
        path = os.path.abspath(args.curves)
        nu = np.arange(int(np.ceil(FMIN * T)), int(np.floor(FMAX * T)) + 1)
        cfg['noise'] = dict(curves=path, sha256=epoch_curves(path, nu)[1])
    cfg.update(store_settings(args))
    states = glitch_states(args)
    if states:
        cfg['glitch_states'] = dict(states, n_window=args.n_window)
    return json.loads(json.dumps(stage_settings(cfg, args)))


def asym_tol(args):
    """the asym_tol of pe_redshift for the run: args.asym_tol when the noise curves differ (analyse.py --unequal,
    plant.py --asym or --curves) or several stores are given, None (not passed, not recorded) otherwise: with equal
    curves and one store it changes nothing"""
    several = len([d for d in args.store.split(',') if d]) > 1
    unequal = getattr(args, 'unequal', False) or getattr(args, 'asym', None) is not None or getattr(args, 'curves', None)
    return float(args.asym_tol) if several or unequal else None


def store_settings(args):
    """the campaign entries of the tables of a merger-stage run: tables_asym, the sorted asym of the store folders, when
    several are given (one store must have the asym of every template, which the noise settings fix); asym_tol when it
    is passed (asym_tol()); zmin_cut when pe_redshift reads only the nodes of z >= zmin"""
    if getattr(args, 'stage', 'merger') == 'windows':
        return {}
    out = {}
    dirs = [d for d in args.store.split(',') if d]
    if len(dirs) > 1:
        out['tables_asym'] = sorted(float(read_header(d)['asym']) for d in dirs)
    if asym_tol(args) is not None:
        out['asym_tol'] = asym_tol(args)
    if getattr(args, 'zmin_cut', False):
        out['zmin_cut'] = True
    return out


def load_stores(args):
    """the store folders of args.store (comma separated): one store, or the list of them; page-locked on the GPU unless
    mapped (args.mmap: the folders must be unbiased caches), unbiased unless args.published_tables"""
    from exmarg.search import load_redshift_store
    out = []
    for d in args.store.split(','):
        t = time.time()
        out.append(load_redshift_store(d, pinned=(args.device == 'gpu' and not args.mmap), mmap=args.mmap,
                                       unbiased=not getattr(args, 'published_tables', False)))
        print('%s  store %s loaded, %d nodes, asym %g%s (%.0f s)' % (stamp(), d, out[-1].s.size, out[-1].asym,
                                                                     ', mapped' if args.mmap else '', time.time() - t),
              flush=True)
    return out[0] if len(out) == 1 else out


def glitch_states(args):
    """{H: [[f, q], ...], L: ...} of --glitch-states, --glitch-states-H, --glitch-states-L, or None"""
    both = getattr(args, 'glitch_states', None)
    one = {I: getattr(args, 'glitch_states_' + I, None) or both for I in 'HL'}
    if not any(one.values()):
        return None
    return {I: parse_states(one[I]) if one[I] else [[1.0, 0.0]] for I in 'HL'}


def stage_settings(cfg, args):
    """the campaign of the stage of args: the windows stage leaves out the settings of the merger stage (MERGER_KEYS)
    and records stage, n_window and the range of the amplitude bins kept"""
    if getattr(args, 'stage', 'merger') != 'windows':
        return cfg
    cfg = {key: v for key, v in cfg.items() if key not in MERGER_KEYS}
    return dict(cfg, stage='windows', n_window=args.n_window, window_rho=[args.window_rho_min, args.window_rho_max])


def with_model(cfg, sha, n_window):
    """the campaign of the merger stage with the window model of the file of SHA-256 sha"""
    return dict(cfg, glitch_model='windows', glitch_model_sha256=sha, n_window=n_window)


def campaign_key(cfg):
    """the settings that fix the data and the bank: the campaign without those of a stage (MERGER_KEYS, STAGE_KEYS),
    the same for the windows stage, the merger stage and the merger stage with the window model of one campaign"""
    return {key: v for key, v in cfg.items() if key not in MERGER_KEYS + STAGE_KEYS + ('run',)}


def _differ(a, b):
    return ', '.join(sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key)))


def model_sha256(model):
    """SHA-256 of a glitch model without its sha256 entry: of its JSON with sorted keys and no spaces"""
    body = {key: v for key, v in model.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_model(path):
    """(the glitch model of windows_fit.py, its SHA-256); SystemExit when the file is missing or was modified (the
    SHA-256 recorded in it is not that of the rest)"""
    if not os.path.exists(path):
        raise SystemExit('%s: no glitch model (windows_fit.py writes it)' % path)
    with open(path) as fh:
        try:
            model = json.load(fh)
        except ValueError as err:
            raise SystemExit('%s: not a JSON glitch model (%s)' % (path, err))
    if not isinstance(model, dict) or model.get('sha256') != model_sha256(model):
        raise SystemExit('%s: the glitch model was modified after it was written (SHA-256 does not match)' % path)
    return model, model['sha256']


def glitch_model(path, cfg, n_window, ignore=()):
    """the glitch settings of pe_redshift (model 'windows') from the file of windows_fit.py, for the run of campaign
    cfg: SystemExit when the file was modified, is not a window model on RHO_EDGES, has another n_window, or was fitted
    on another campaign (campaign_key of its files against that of cfg, the keys of ignore aside).  returns (glitch
    dict, the model's SHA-256)"""
    model, sha = read_model(path)
    if model.get('model') != 'windows' or not np.array_equal(np.asarray(model.get('rho_edges'), float), RHO_EDGES):
        raise SystemExit('%s: not a window model on the amplitude bins single.RHO_EDGES' % path)
    if model['n_window'] != n_window:
        raise SystemExit('%s: the model has n_window = %s, the run %s' % (path, model['n_window'], n_window))
    mine = {key: v for key, v in campaign_key(cfg).items() if key not in ignore}
    theirs = {key: v for key, v in model['campaign'].items() if key not in ignore}
    if mine != theirs:
        raise SystemExit('%s: the model was fitted on another campaign (differs in %s)' % (path, _differ(mine, theirs)))
    det = {I: dict(f=np.asarray(model[I]['f'], float), q=np.asarray(model[I]['q'], float),
                   pi_k=np.asarray(model[I]['pi_k'], float)) for I in 'HL'}
    return dict(model='windows', n_window=model['n_window'], zone=model['zone'], pi_m=model['pi_m'], **det), sha


def read_meta(path):
    with np.load(path) as f:
        return json.loads(str(f['meta']))


def check_campaign(path, cfg):
    meta = read_meta(path)
    meta.pop('run', None)
    if meta != cfg:
        diff = sorted(key for key in set(meta) | set(cfg) if meta.get(key) != cfg.get(key))
        raise SystemExit('%s belongs to another campaign (differs in %s)' % (path, ', '.join(diff)))


def check_folder(out, cfg, ignore=()):
    """every result file of the folder out against the run of campaign cfg: the files of its stage must be of the
    same campaign, those of the other stage of the same campaign_key (the keys of ignore aside)"""
    mine = PREFIX[cfg.get('stage', 'merger')]
    key = lambda c: {k: v for k, v in campaign_key(c).items() if k not in ignore}
    for path in sorted(glob.glob(os.path.join(out, 'plant_*.npz')) + glob.glob(os.path.join(out, 'win_*.npz'))):
        if os.path.basename(path).startswith(mine + '_'):
            check_campaign(path, cfg)
            continue
        meta = read_meta(path)
        if key(meta) != key(cfg):
            raise SystemExit('%s belongs to another campaign (differs in %s)' % (path, _differ(key(meta), key(cfg))))


def write_batch(path, cfg, run, arrays):
    """one result file, written under a temporary name in the same directory and renamed"""
    tmp = os.path.join(os.path.dirname(path) or '.', '.%s.%s.%d.tmp.npz' % (os.path.basename(path)[:-4],
                                                                           socket.gethostname(), os.getpid()))
    np.savez(tmp, meta=np.array(json.dumps(dict(cfg, run=run))), **arrays)
    os.replace(tmp, path)


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def run(cfg, out, start, n, batch, factory, plain=False, info=None, log=print, glitch=True):
    """the campaign cfg over segments start .. start + n - 1 into the directory out; factory(segments) -> pe (see
    make_analysis, glitch: True or the window model of glitch_model), or win for the windows stage (make_windows), is
    called once, before the first batch to analyse.  returns the paths written"""
    os.makedirs(out, exist_ok=True)
    check_folder(out, cfg)
    windows = cfg.get('stage') == 'windows'
    if windows and plain:
        raise ValueError('plain applies to the merger stage')
    bins = window_bins(*cfg['window_rho']) if windows else None
    pop = population(cfg['zmin'])
    sampler = Sampler(pop, cfg['mdet_range'])
    pe, done, written, began = None, 0, [], time.time()
    for first, end in batches(start, n, batch):
        path = os.path.join(out, '%s_%010d_%010d.npz' % ('win' if windows else 'plant', first, end))
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
        if windows:
            arrays = reduce_windows(pe(seg), bins)
            m = end - first
            times = dict(t_sim=(t1 - t0) / m, t_pe=(time.time() - t2) / m, t_plain=None)
            arrays.update(truth, **window_shared(bins))
            write_batch(path, cfg, dict(info or {}, host=socket.gethostname(), first=first, end=end, written=stamp(),
                                        **times), arrays)
            written.append(path)
            done += m
            log('%s  [%d, %d) %d of %d segments; simulation %.4f s, window evidences %.4f s per segment; %.0f s'
                % (stamp(), first, end, done, n, times['t_sim'], times['t_pe'], time.time() - began), flush=True)
            continue
        res = pe(seg, stream(cfg['seed'], first, 'pe'), glitch)
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
    p.add_argument('store', help='folder of the redshift tables (bd_*.npz) or its unbiased cache; or several, comma '
                                 'separated, of distinct asym: the mixture of the two that bracket each template')
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
    p.add_argument('--chunk', type=int, default=None,
                   help='segments filtered at once inside pe_redshift (default %d) or window_evidences (%d)'
                        % (CHUNK, WINDOW_CHUNK))
    p.add_argument('--ymin', type=float, default=None, help='ymin of pe_redshift (default D0 / D_L(10) (1 - 1e-6))')
    p.add_argument('--zmin', type=float, default=ZMIN, help='smallest z of the population (plant and analysis); '
                   '0.99119, the lower edge of the grid cell at z = 1.025, puts the step on a cell edge')
    p.add_argument('--resident', action='store_true', help='keep every table on the GPU (needs ~48 GB)')
    p.add_argument('--plain', action='store_true', help='also run pe_redshift without glitch, to record its cost')
    p.add_argument('--shared-times', action='store_true',
                   help='the same random arrival times for every template (recorded in the campaign)')
    p.add_argument('--published-tables', action='store_true',
                   help='read the tables as published, without unbias (a comparison; recorded in the campaign)')
    p.add_argument('--glitch-states', default=None, metavar='F1:Q1,F2:Q2',
                   help='glitches of the window model in both detectors: state probabilities f and glitch probability '
                        'q per window in each state (not with --eps-H, --eps-L)')
    p.add_argument('--glitch-states-H', default=None, metavar='F1:Q1,...', help='the same for H alone')
    p.add_argument('--glitch-states-L', default=None, metavar='F1:Q1,...', help='the same for L alone')
    p.add_argument('--n-window', type=int, default=N_WINDOW,
                   help='windows of a segment: of the planted window model, of the windows stage and of the model')
    p.add_argument('--stage', choices=STAGES, default='merger',
                   help='merger: pe_redshift (default); windows: stage 1 of the window model, win_*.npz (no tables)')
    p.add_argument('--window-rho-min', type=float, default=WINDOW_RHO[0],
                   help='windows stage: keep the amplitude bins that overlap [window-rho-min, window-rho-max]')
    p.add_argument('--window-rho-max', type=float, default=WINDOW_RHO[1])
    p.add_argument('--glitch-model', default=None, metavar='JSON',
                   help='merger stage with the window model of this file of windows_fit.py (same campaign)')
    p.add_argument('--asym', type=float, default=None, metavar='W',
                   help='unequal noise: sig2_H = sig2 / (1 + W), sig2_L = sig2 / (1 - W), every template at w = W')
    p.add_argument('--curves', default=None, metavar='EPOCH',
                   help='unequal noise: the curves of each detector of this conditioned epoch file (strain.own_sig2)')
    p.add_argument('--zmin-cut', action='store_true',
                   help='pe_redshift reads only the nodes the redshifts z >= --zmin need (the same results there)')
    p.add_argument('--asym-tol', type=float, default=ASYM_TOL,
                   help='unequal curves or several stores: a template whose asymmetry lies outside the stores\' range by '
                        'at most this is read from the end store (default %g; recorded)' % ASYM_TOL)
    p.add_argument('--mmap', action='store_true',
                   help='map the tables from disk (the store folders must be unbiased caches, '
                        'store.write_unbiased_cache): the jobs of a machine share them')
    p.add_argument('--out', required=True, help='directory of the result files')
    return p


def n_time_of(T_, fmax):
    """the number of matched-filter samples of a segment, 8 nu_max (bank.MassBank)"""
    return 8 * int(np.floor(fmax * T_ + 1e-9))


def main(argv=None, factory=None):
    """factory: replaces the analysis with the tables, or window_evidences for the windows stage (tests)"""
    p = parser()
    a = p.parse_args(argv)
    if not (0 <= a.nbar <= 1 and 0 <= a.eps_H <= 1 and 0 <= a.eps_L <= 1):
        p.error('nbar and eps must lie in [0, 1]')
    if not (0 < a.rho_min < a.rho_max) or a.n < 1 or a.batch < 1 or a.start < 0 or not 0 <= a.seed < 2 ** 20:
        p.error('need 0 < rho_min < rho_max, n >= 1, batch >= 1, start >= 0, 0 <= seed < 2^20')
    if a.n_window < 1 or n_time_of(T, FMAX) % a.n_window:
        p.error('--n-window must divide the %d matched-filter samples of a segment' % n_time_of(T, FMAX))
    windows = a.stage == 'windows'
    if a.chunk is None:
        a.chunk = WINDOW_CHUNK if windows else CHUNK
    if windows and (a.glitch_model or a.plain or a.published_tables or a.shared_times or a.ymin is not None
                    or a.zmin_cut or a.mmap):
        p.error('--glitch-model, --plain, --published-tables, --shared-times, --ymin, --zmin-cut and --mmap are of the '
                'merger stage')
    if a.asym is not None and a.curves is not None:
        p.error('--asym or --curves, not both')
    if not a.asym_tol >= 0:
        p.error('need asym-tol >= 0')
    if a.asym is not None and not -1.0 < a.asym < 1.0:
        p.error('need -1 < asym < 1')
    if a.mmap and a.published_tables:
        p.error('--mmap maps unbiased caches: not with --published-tables')
    if windows and not (0 <= a.window_rho_min < a.window_rho_max):
        p.error('need 0 <= window-rho-min < window-rho-max')
    try:
        states = glitch_states(a)
    except ValueError as err:
        p.error(str(err))
    if states and (a.eps_H > 0 or a.eps_L > 0):
        p.error('--glitch-states plants the window model: not with --eps-H, --eps-L')
    cfg = campaign(a)
    glitch = True
    if a.glitch_model:
        glitch, sha = glitch_model(a.glitch_model, cfg, a.n_window)
        cfg = with_model(cfg, sha, a.n_window)
    if factory is None and windows:
        factory = make_windows(a.device, a.n_window, a.chunk, store_lags(a.store))
    elif factory is None:
        for d in a.store.split(','):
            if not glob.glob(os.path.join(d, 'bd_*.npz')) and not is_cache(d):
                p.error('no bd_*.npz (or unbiased cache) in %s' % d)

        def factory(seg):
            return make_analysis(load_stores(a), a.device, a.resident, a.chunk, cfg['ymin'], a.shared_times,
                                 cfg['zmin'] if a.zmin_cut else None, asym_tol(a))(seg)
    print('%s  campaign %s; segments %d to %d in batches of %d into %s'
          % (stamp(), json.dumps(cfg), a.start, a.start + a.n - 1, a.batch, a.out), flush=True)
    run(cfg, a.out, a.start, a.n, a.batch, factory, a.plain,
        dict(device=a.device, chunk=a.chunk, resident=a.resident, pid=os.getpid()), glitch=glitch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
