"""the network cut and the detected fraction.

The tables are accurate (1e-3 in ln, order 120) only for W = w_H + w_L <= 84 at the arrival times read.  wnet, per
segment and template, is the largest w_L(t) + w_H(t + tau) over every arrival time and every lag of the store
(gather.network_max, output by pe next to wmax_H, wmax_L); the fits set a segment aside when wmax_H > 72 or wmax_L > 72
or wnet > W_NET = 84 (plant_fit.veto).  Setting segments aside is a selection on the data, the same with and without a
merger, so the kept segments have the likelihood prod_i [1 + nbar (B_i - 1)] / (1 - nbar f)^n with the detected
fraction f = 1 - P(kept | merger) / P(kept | no merger) (plant_fit.nbar_fit(f_det=f), the paper's Eq. 35).

References, none from the code under test: a double loop over arrival times and lags in Python floats (wnet); pe.py,
plant_fit.py and windows_fit.py as committed at BASE, before the network cut, imported from git (bit for bit); planted
truth (which segments hold a loud coincidence); a two-component model with a known selection, integrated by quadrature
(the limit of each estimator and f)."""

import glob
import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from exmarg.search import gather, pe
from exmarg.search import RHO_EDGES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'a79e07d'                    # the branch before the network cut


def _script(name, sub='validate'):
    spec = importlib.util.spec_from_file_location('wnet_' + name, os.path.join(ROOT, 'scripts', sub, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plant, plant_fit, windows_fit = _script('plant'), _script('plant_fit'), _script('windows_fit')
summarize, slides_mean = _script('summarize', 'real_data'), _script('slides_mean', 'real_data')


def _git(path):
    try:
        return subprocess.run(['git', 'show', '%s:%s' % (BASE, path)], cwd=ROOT, capture_output=True, check=True,
                              text=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        pytest.skip('no git history to compare with (%s)' % err)


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    """pe.py (as a module of exmarg.search), plant_fit.py and windows_fit.py of BASE: the scripts in a copy of the
    scripts folder (windows_fit reads summarize, which reads plant_fit, from it)"""
    root = tmp_path_factory.mktemp('base')
    for path in ('scripts/validate/plant.py', 'scripts/validate/plant_fit.py', 'scripts/validate/windows_fit.py',
                 'scripts/real_data/summarize.py', 'src/exmarg/search/pe.py'):
        os.makedirs(root / os.path.dirname(path), exist_ok=True)
        (root / path).write_text(_git(path))
    out = {}
    saved = list(sys.path)
    sys.modules.setdefault('plant', plant)          # windows_fit imports plant: today's (what it uses is unchanged)
    try:
        for key, name, path in (('pe', 'exmarg.search._pe_base_wnet', 'src/exmarg/search/pe.py'),
                                ('plant_fit', 'base_plant_fit', 'scripts/validate/plant_fit.py'),
                                ('windows_fit', 'base_windows_fit', 'scripts/validate/windows_fit.py')):
            spec = importlib.util.spec_from_file_location(name, str(root / path))
            out[key] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(out[key])
    finally:
        sys.path[:] = saved                         # the copy's folders off the path again
    assert not hasattr(out['pe'], '_network') and not hasattr(out['plant_fit'], 'veto')
    return out


# ------------------------------------------------------------------------------------------ wnet against brute force


def brute_wnet(lqH, lqL, ln_uH, ln_uL, lags):
    """max over t and tau of w_L(t) + w_H((t + tau) mod n_time), a double loop in Python floats"""
    nm, nt = lqH.shape
    out = np.empty(nm)
    for m in range(nm):
        wH = [math.exp(2.0 * float(x) - float(ln_uH[m])) for x in lqH[m]]
        wL = [math.exp(2.0 * float(x) - float(ln_uL[m])) for x in lqL[m]]
        best = -math.inf
        for t in range(nt):
            for tau in lags:
                best = max(best, wL[t] + wH[(t + int(tau)) % nt])
        out[m] = best
    return out


def planted_segment(lags_lo=-4, lags_hi=4, nt=512, seed=3):
    """lqH, lqL (3, nt) float32 and ln u_H, ln u_L (3,): Exp(1) noise in w (largest about 7), and per template the
    planted pairs (w_L at t, w_H at t + tau) and the truth, the largest planted sum with tau inside [lags_lo, lags_hi]:
      0: (40, 40) at the upper edge tau = lags_hi; (45, 45) one lag beyond it, which must not count -> 80
      1: (38, 38) at tau = 3 from t = nt - 2, wrapping round the end of the segment; (44, 44) one lag below the lower
         edge -> 76
      2: (35, 35) at the lower edge tau = lags_lo; (33, 33) at tau = -3 from t = 2 (wraps backwards); 55 in H
         alone -> 70"""
    rng = np.random.default_rng(seed)
    ln_u = {I: np.log(rng.uniform(0.5, 2.0, 3)) for I in 'HL'}
    w = {I: rng.exponential(1.0, (3, nt)) for I in 'HL'}
    plants = {0: [(100, lags_hi, 40.0), (300, lags_hi + 1, 45.0)],
              1: [(nt - 2, 3, 38.0), (200, lags_lo - 1, 44.0)],
              2: [(50, lags_lo, 35.0), (2, -3, 33.0)]}
    for m, pairs in plants.items():
        for t, tau, x in pairs:
            w['L'][m, t], w['H'][m, (t + tau) % nt] = x, x
    w['H'][2, 400] = 55.0
    lq = {I: (0.5 * (np.log(w[I]) + ln_u[I][:, None])).astype(np.float32) for I in 'HL'}
    return lq['H'], lq['L'], ln_u['H'], ln_u['L'], np.array([80.0, 76.0, 70.0])


class ArrayModule:
    """numpy behind the face of another array module (cupy, to the code): records every name the code takes from it"""

    def __init__(self):
        self.used = set()

    def __getattr__(self, name):
        self.used.add(name)
        return getattr(np, name)


CUPY_NAMES = {'asarray', 'empty', 'float64', 'roll', 'maximum', 'exp'}     # in cupy, with numpy's semantics


def test_wnet_brute_force():
    """network_max equals the double loop over every arrival time and lag (relative 1e-13) for the store's lags in the
    order of the tables (0, 1, .., 4, -4, .., -1), sorted, and for lags that are not contiguous; the planted truth is
    recovered (the pair at the edge of the lag window counts, the pair one lag beyond it does not, the pair that wraps
    round the end of the segment counts), to the float32 rounding of the data; and the same code through an array
    module that is not numpy (the GPU path: cupy) uses only cupy's names and gives the same bits.
    Breaks that fire it: the running maximum one lag short at the upper edge, a lag window one wider, the shift not
    circular, the negative lags dropped (mutation runs of the report)."""
    lqH, lqL, uH, uL, truth = planted_segment()
    store_order = np.r_[np.arange(0, 5), np.arange(-4, 0)]
    lines = []
    for lags in (store_order, np.sort(store_order), np.array([-4, -3, 0, 4])):
        got = gather.network_max(lqH, lqL, uH, uL, lags)
        ref = brute_wnet(lqH, lqL, uH, uL, lags)
        assert np.allclose(got, ref, rtol=1e-13, atol=0), (lags, got, ref)
        lines.append('lags %s: wnet %s, brute force %s' % (lags.tolist(), np.round(got, 6), np.round(ref, 6)))
    got = gather.network_max(lqH, lqL, uH, uL, store_order)
    assert np.allclose(got, truth, rtol=1e-5, atol=0), got
    xp = ArrayModule()
    other = gather.network_max(lqH, lqL, uH, uL, store_order, xp=xp, block=2)
    assert np.array_equal(other, got) and xp.used <= CUPY_NAMES, (other, got, xp.used)
    lines.append('through another array module (blocks of 2 templates): same bits; names used %s' % sorted(xp.used))
    print('\n' + '\n'.join(lines))


# ------------------------------------------------------------------------------- outputs bit for bit with wnet added


def traj_wnet(c, e, lags, ubar=False):
    """the brute-force wnet of row e of a case of tests/test_pe_unequal.py: its trajectories, unscaled, with u_I (or,
    ubar, the mean model power: what wnet must NOT use)"""
    bank, seg = c['bank'], c['seg']
    t = bank.trajectories({I: seg['d_' + I][e] for I in 'HL'}, {I: seg['sig2_' + I] for I in 'HL'})
    u = {I: np.log(bank.u if ubar else getattr(bank, 'u_' + I)) for I in 'HL'}
    return brute_wnet(t['lqH'], t['lqL'], u['H'], u['L'], lags)


def test_wnet_outputs_bit_identical(base):
    """every array of pe_redshift, pe_intrinsic, pe_redshift_epochs (two epochs, their own seeds, rows and glitch
    settings) and window_evidences is that of pe.py at BASE bit for bit, for glitch None, single, True and the window
    model, equal curves (one store) and unequal curves (the mixture of three stores); the only new key is wnet
    (n_segment, n_template), with single or glitch, and in window_evidences only when given the lags; wnet is the brute
    force of the unscaled data with u_H, u_L (not ubar: that differs on unequal curves) and the store's lags, and is the
    same in pe_redshift, pe_intrinsic, the epochs and window_evidences.
    Breaks that fire it: the running maximum taken in place on the trajectories (every later output changes), the data
    scaled to a store's model or ubar in place of u_I (wnet differs from the brute force on unequal curves)."""
    import test_pe_unequal as tu
    lines, refs = [], {}
    for kind, stores in ((0.0, tu.inc_store(0.0)), ('shaped', [tu.inc_store(a) for a in tu.GRID])):
        c = tu.make_case(kind)
        lags = np.asarray(tu.TAU)
        ref = refs[kind] = np.stack([traj_wnet(c, e, lags) for e in range(3)])
        if kind == 'shaped':
            off = np.stack([traj_wnet(c, e, lags, ubar=True) for e in range(3)])
            assert np.abs(off / ref - 1).max() > 0.05             # the test can tell u_I from ubar
        for glitch in (None, 'single', True, 'windows'):
            kw = dict(single=True) if glitch == 'single' else dict(glitch=tu.WINDOWS if glitch == 'windows' else glitch)
            for name, extra in (('pe_redshift', tu.REDSHIFT), ('pe_intrinsic', {})):
                a = getattr(pe, name)(c['segs'], c['bank'], stores, ymin=0.1, nodes_per_launch=1, n_loud=0, **kw,
                                      **extra)
                b = getattr(base['pe'], name)(c['segs'], c['bank'], stores, ymin=0.1, nodes_per_launch=1, n_loud=0,
                                              **kw, **extra)
                assert set(a) - set(b) == (set() if glitch is None else {'wnet'}) and set(b) <= set(a), (name, glitch)
                for k in b:
                    assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), (kind, glitch, name, k)
                if glitch is not None:
                    assert a['wnet'].shape == (3, tu.MASSES.size)
                    assert np.allclose(a['wnet'], ref, rtol=1e-13, atol=0), (kind, glitch, name)
                    assert np.all(a['wnet'] >= np.maximum(a['wmax_H'], a['wmax_L']))
                    assert np.all(a['wnet'] <= a['wmax_H'] + a['wmax_L'])
                    lines.append('%s, glitch %s, %s: %d arrays bit for bit; wnet %s' % (
                        kind, glitch, name, len(b), np.array2string(a['wnet'][:, 1], precision=2)))
        w0, w1 = pe.window_evidences(c['segs'], c['bank']), base['pe'].window_evidences(c['segs'], c['bank'])
        assert set(w0) == set(w1) and all(np.array_equal(w0[k], w1[k]) for k in w1)
        wl = pe.window_evidences(c['segs'], c['bank'], lags=lags, chunk=2)
        assert set(wl) == set(w1) | {'wnet'} and all(np.array_equal(wl[k], w1[k]) for k in w1)
        assert np.array_equal(wl['wnet'], a['wnet'])
    c = tu.make_case(0.0)
    st = tu.inc_store(0.0)
    eps = [dict(segments=c['segs'], bank=c['bank'], seed=3, glitch=True),
           dict(segments=c['segs'], bank=c['bank'], rows=[2, 0], seed=4, glitch=tu.WINDOWS)]
    common = dict(ymin=0.1, nodes_per_launch=1, n_loud=64, n_random=64, chunk=2, **tu.REDSHIFT)
    new, old = pe.pe_redshift_epochs(eps, st, **common), base['pe'].pe_redshift_epochs(eps, st, **common)
    for a, b, rows in zip(new, old, ([0, 1, 2], [2, 0])):
        assert set(a) == set(b) | {'wnet'} and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in b)
        assert np.allclose(a['wnet'], refs[0.0][rows], rtol=1e-13, atol=0)
    lines.append('pe_redshift_epochs (two epochs, chunks of 2, arrival times sampled): bit for bit, wnet of every '
                 'arrival time')
    print('\n' + '\n'.join(lines))


# ------------------------------------------------------------------------------------------------ the cut itself


def write_files(directory, cfg, lnB, m, wmax, wnet=None, splits=((0, 2000), (2000, 6000))):
    """files of the merger stage with the window model (the members of plant.py), with wnet when given"""
    n = lnB.size
    nr = RHO_EDGES.size - 1
    per = dict(k=np.arange(n), ln_B_bar=lnB, ln_lambda_bar=lnB, ln_lambda_H_bar=0.5 * lnB, ln_lambda_L_bar=0.5 * lnB,
               w_inf=np.zeros(n), merger=m, merger_H=m, merger_L=m, coherent=m, m1=np.full(n, np.nan),
               z=np.full(n, np.nan), y=np.zeros(n), snr=np.zeros(n))
    if wnet is not None:
        per['wnet'] = wnet
    for I in 'HL':
        per.update({'ln_c_' + I: np.zeros(n), 'state_' + I: np.tile([0.9, 0.1], (n, 1)), 'ln_glitch_' + I:
                    np.zeros((n, nr)), 'wmax_' + I: wmax[I], 'glitch_' + I: np.zeros(n, bool),
                    'kind_' + I: np.full(n, ''), 'rho_' + I: np.full(n, np.nan), 't0_' + I: np.full(n, np.nan)})
    os.makedirs(directory, exist_ok=True)
    for a, b in splits:
        plant.write_batch(os.path.join(directory, 'plant_%010d_%010d.npz' % (a, b)), cfg,
                          dict(first=a, end=b, t_sim=1e-3, t_pe=0.1, t_plain=None),
                          dict({key: v[a:b] for key, v in per.items()}, rho_edges=RHO_EDGES,
                               glitch_pi_k=np.array(np.nan), glitch_window=np.array(np.nan),
                               w_inf_cells=np.zeros((0, 2), np.int32)))


def synthetic_B(n, nbar, seed, mus=(1.5, 2.5, 3.5), w=(0.5, 0.3, 0.2)):
    """ln Bbar and merger flags of a known mixture (tests/test_plant.py): x ~ N(mu, 1) with a merger, N(0, 1) without"""
    rng = np.random.default_rng(seed)
    mus, w = np.asarray(mus, float), np.asarray(w, float)
    m = rng.random(n) < nbar
    mu = mus[np.minimum(np.searchsorted(np.cumsum(w), rng.random(n), side='right'), mus.size - 1)]
    x = rng.standard_normal(n) + np.where(m, mu, 0.0)
    return logsumexp(np.log(w)[None] + mus[None] * x[:, None] - 0.5 * mus[None] ** 2, axis=1), m


def planted_cut(n=6000, seed=21):
    """wmax_H, wmax_L (Exp(5): none above 72 by chance), wnet (the background below 80), with 10 segments loud in H
    alone (wmax_H 100, wnet 110) and 25 planted loud coincidences (both wmax 45, wnet 90): returns (wmax, wnet, single,
    coinc)"""
    rng = np.random.default_rng(seed)
    wmax = {I: rng.exponential(5.0, n) for I in 'HL'}
    wnet = np.minimum(wmax['H'] + 0.5 * wmax['L'], 80.0)
    pick = rng.permutation(n)
    single, coinc = np.zeros(n, bool), np.zeros(n, bool)
    single[pick[:10]], coinc[pick[10:35]] = True, True
    wmax['H'][single], wnet[single] = 100.0, 110.0
    wmax['H'][coinc], wmax['L'][coinc], wnet[coinc] = 45.0, 45.0, 90.0
    assert wmax['H'][~single].max() < 72 and wmax['L'].max() < 72
    return wmax, wnet, single, coinc


def comparable(r):
    """the numbers of a plant_fit report"""
    F = r['fits']
    return dict(keep=r['keep'], ok=r['ok'], criteria=[c[:2] for c in r['criteria']],
                fits={k: (F[k]['nbar'], F[k]['err']['nbar'], F[k]['lnl']) for k in ('gauss', 'windows')},
                curves={k: v for k, v in r['curves'].items()},
                unit={s: {k: tuple(v) for k, v in x.items()} for s, x in r['unit'].items()})


def same(a, b):
    return json.dumps({k: v for k, v in a.items() if k not in ('keep', 'curves')}, default=str) == json.dumps(
        {k: v for k, v in b.items() if k not in ('keep', 'curves')}, default=str) and np.array_equal(
        a['keep'], b['keep']) and all(np.array_equal(a['curves'][k], b['curves'][k]) for k in b['curves'])


def test_cut_plant_fit(base, tmp_path):
    """plant_fit.report on window-model files with 10 segments loud in H alone and 25 planted loud coincidences (both
    wmax 45 < 72, wnet 90 > 84; 0 to 2 of them hold a merger): switched off (w_net inf) every number is that of
    plant_fit.py at BASE (keep, fits, time curves, unit mean, criteria); on files without wnet the default also gives
    BASE's numbers and the report says the cut is NOT APPLIED and that the 2 files lack wnet; on (default 84) the
    segments set aside are exactly the loud ones and the planted coincidences, the table of conditions counts them
    (wnet alone: the 25, with their mergers), and the window-model nbar is nbar_fit of the kept segments.
    Breaks that fire it: the network condition left out of keep, the cut applied when switched off, wnet guessed from
    wmax_H + wmax_L in files without it."""
    lnB, m = synthetic_B(6000, 0.05, 5)
    wmax, wnet, single, coinc = planted_cut()
    cfg = plant.with_model(plant.campaign(plant.parser().parse_args(['s', 'cpu', '--n', '1', '--out', 'x', '--nbar',
                                                                      '0.05'])), 'ab' * 32, 64)
    with_net, without = str(tmp_path / 'with'), str(tmp_path / 'without')
    write_files(with_net, cfg, lnB, m, wmax, wnet)
    write_files(without, cfg, lnB, m, wmax)
    old = comparable(base['plant_fit'].report(with_net))
    assert same(comparable(plant_fit.report(with_net, w_net=np.inf)), old)
    r = plant_fit.report(without)
    assert same(comparable(r), old) and not r['veto']['applied']
    text = '\n'.join(r['lines'])
    assert 'NOT APPLIED' in text and '(2 of 2 files lack it)' in text
    r = plant_fit.report(with_net)
    out = single | coinc
    assert np.array_equal(r['keep'], ~out) and np.array_equal(r['veto']['net'], out)
    assert r['fits']['windows']['nbar'] == plant_fit.nbar_fit(lnB[~out])['nbar']
    text = '\n'.join(r['lines'])
    row = [l for l in r['lines'] if l.strip().startswith('wnet > 84 alone')][0]
    assert row.split()[-4:] == [str(coinc.sum()), str((coinc & m).sum()), '0', '0'], row     # aside, merger, glitches
    assert 'set aside %d of 6000 segments' % out.sum() in text
    det = r['detected']['measured']
    assert det == plant_fit.detected_fraction(m, ~out)
    assert {'f = 0', 'f = %.4g, measured' % det['f']} == set(r['detected']['fits'])
    print('\n' + '\n'.join(l for l in r['lines'] if 'wnet' in l or 'network' in l or 'detected' in l
                           or 'f = ' in l or 'measured' in l))


def write_win(folder, cfg, sim, wnet=None, splits=((0, 700), (700, 1500))):
    """the window toy of tests/test_windows_fit.py in win files, with wnet when given"""
    import test_windows_fit as tw
    os.makedirs(folder, exist_ok=True)
    n = sim['H']['E'].shape[0]
    per = dict(k=np.arange(n), merger=np.zeros(n, bool), merger_H=np.zeros(n, bool), merger_L=np.zeros(n, bool),
               coherent=np.zeros(n, bool), m1=np.full(n, np.nan), z=np.full(n, np.nan), y=np.zeros(n), snr=np.zeros(n))
    if wnet is not None:
        per['wnet'] = wnet
    for I in 'HL':
        g = sim[I]
        per.update({'ln_window_' + I: g['E'].astype(np.float32), 'wmax_' + I: g['wmax'], 'glitch_' + I: g['n'] > 0,
                    'kind_' + I: np.where(g['n'] > 0, 'template', ''), 'rho_' + I: g['rho'],
                    't0_' + I: np.full(n, np.nan), 'glitch_state_' + I: g['state'], 'n_glitches_' + I: g['n']})
    for a, b in splits:
        plant.write_batch(os.path.join(folder, 'win_%010d_%010d.npz' % (a, b)), cfg, dict(first=a, end=b),
                          dict({key: v[a:b] for key, v in per.items()}, **plant.window_shared(tw.STORED)))


def test_cut_windows_fit(base, tmp_path):
    """windows_fit.fit (one state) on the window toy of tests/test_windows_fit.py with 8 segments loud in H (wmax 100)
    and 30 planted loud coincidences (wnet 90, their wmax as simulated): switched off, and on files without wnet, the
    model written is that of windows_fit.py at BASE to the last digit (no w_net recorded); on, the segments set aside
    are exactly those above the single-detector veto and the planted ones, and the model records w_net 84 and
    set_aside_net.
    Breaks that fire it: the network condition left out, the cut applied when off."""
    import test_windows_fit as tw
    sim = tw.simulate(1500, 31)
    rng = np.random.default_rng(4)
    sim['H']['wmax'][rng.choice(1500, 8, replace=False)] = 100.0          # loud in H alone
    loud = (sim['H']['wmax'] > 72) | (sim['L']['wmax'] > 72)
    wnet = np.minimum(sim['H']['wmax'] + sim['L']['wmax'], 83.0)
    wnet[loud] = 100.0
    coinc = np.zeros(1500, bool)
    coinc[rng.choice(np.flatnonzero(~loud), 30, replace=False)] = True
    wnet[coinc] = 90.0
    cfg = tw.config()
    folders = {name: str(tmp_path / name) for name in ('with', 'without')}
    write_win(folders['with'], cfg, sim, wnet)
    write_win(folders['without'], cfg, sim)
    args = dict(n_state=1, record=(1,))
    old = base['windows_fit'].fit(folders['with'], **args)
    for folder, w_net in ((folders['with'], np.inf), (folders['without'], 84.0)):
        r = windows_fit.fit(folder, **args, w_net=w_net)
        assert r['model'] == old['model'] and np.array_equal(r['keep'], old['keep']), (folder, w_net)
    assert 'NOT APPLIED' in '\n'.join(r['lines'])
    r = windows_fit.fit(folders['with'], **args)
    assert np.array_equal(r['keep'], ~(loud | coinc)) and loud.sum() > 0
    assert r['model']['settings']['w_net'] == 84.0 and r['model']['segments']['set_aside_net'] == (loud | coinc).sum()
    assert r['model']['segments']['used'] == (~(loud | coinc)).sum()
    print('\n' + '\n'.join(r['lines'][:6]))


def test_cut_slides_mean(tmp_path):
    """slides_mean on two slides with 12 planted loud coincidences each (wnet 90): on, they and the segments loud in H
    are set aside and nbar is nbar_fit of the rest (with --f-det, of the rest with that f); off, only those loud in H,
    as before the cut; the output says which.
    Breaks that fire it: the network condition left out, f_det not passed to nbar_fit."""
    import test_analyse as ta
    week = tmp_path / 'segments'
    os.makedirs(week)
    (week / 'manifest.txt').write_text('epoch_1266933768.npz 1266933768\n')
    n, root = 3000, tmp_path / 'week'
    os.makedirs(root)
    (root / 'selection.txt').write_text('epoch_1266933768.npz 1266933768 %d %d 0\n' % (n, n))
    args = ta.analyse.parse(['x', 'cpu', str(week), '--out', 'x', '--seed', '7'])[1]
    rng = np.random.default_rng(8)
    truth = {}
    for i, v in enumerate(('slide+0.5', 'slide-1.75')):
        lnB = rng.normal(-0.02, 0.2, n)
        wmax = {I: rng.exponential(5.0, n) for I in 'HL'}
        wmax['H'][rng.choice(n, 5, replace=False)] = 100.0
        wnet = np.minimum(wmax['H'] + wmax['L'], 83.0)
        coinc = np.zeros(n, bool)
        coinc[rng.choice(np.flatnonzero(wmax['H'] < 72), 12, replace=False)] = True
        wnet[coinc] = 90.0
        os.makedirs(root / v)
        for a, b in ((0, 1000), (1000, n)):
            per = dict(k=np.arange(a, b), wmax_H=wmax['H'][a:b], wmax_L=wmax['L'][a:b], ln_B_bar=lnB[a:b],
                       wnet=wnet[a:b])
            plant.write_batch(str(root / v / ('plant_%010d_%010d.npz' % (a, b))),
                              ta.analyse.campaign(args, str(week), v, '%064x' % (i + 1)), dict(first=a, end=b), per)
        truth[v] = (lnB, wmax['H'] > 72, coinc)
    for w_net, f in ((84.0, 0.0), (84.0, 0.03), (np.inf, 0.0)):
        r = slides_mean.analyse(str(root), w_net=w_net, f_det=f)
        for v, (lnB, loud, coinc) in truth.items():
            keep = ~(loud | coinc) if np.isfinite(w_net) else ~loud
            s = r['slides'][v]
            assert s['kept'] == keep.sum() and s['nbar'] == plant_fit.nbar_fit(lnB[keep], f)['nbar'], (v, w_net, f)
        text = '\n'.join(r['lines'])
        assert ('slide+0.5 on' in text) == bool(np.isfinite(w_net)) and 'f = %g' % f in text
    print('\n' + '\n'.join(r['lines'][:4]))


# -------------------------------------------------------------------------------------------- the detected fraction


MUS, WTS = np.array([1.5, 3.0, 4.5]), np.array([0.5, 0.3, 0.2])
NBAR, CUT, GLITCH = 0.1, 4.0, 0.05


def toy_lnB(x):
    """ln B(x) = ln sum_j w_j exp(mu_j x - mu_j^2 / 2): the exact likelihood ratio of x ~ N(mu, 1), mu ~ (MUS, WTS),
    against x ~ N(0, 1)"""
    return logsumexp(np.log(WTS)[None] + MUS[None] * np.atleast_1d(x)[:, None] - 0.5 * MUS[None] ** 2, axis=1)


def toy_truth():
    """the selection of the toy: kept when x <= CUT and no glitch (probability GLITCH, independent of x and of the
    merger: noise alone is set aside too).  k0 = (1 - GLITCH) Phi(CUT), k1 = (1 - GLITCH) sum_j w_j Phi(CUT - mu_j);
    f = 1 - k1 / k0 and the fraction of mergers set aside, 1 - k1 (the paper's f_det when noise is never set aside).
    The limit of the estimate with any f: the root in nbar of E[u_f(nbar)] over the kept data (density (1 - NBAR) phi(x)
    + NBAR sum_j w_j phi(x - mu_j) on x <= CUT), by quadrature"""
    k0 = (1 - GLITCH) * norm.cdf(CUT)
    k1 = (1 - GLITCH) * float(np.sum(WTS * norm.cdf(CUT - MUS)))
    dens = lambda x: (1 - NBAR) * norm.pdf(x) + NBAR * float(np.sum(WTS * norm.pdf(x - MUS)))
    norm_k = quad(dens, -12.0, CUT, epsabs=1e-14)[0]

    def score(nb, f):
        g = lambda x: dens(x) * math.expm1(toy_lnB(x)[0]) / (1.0 + nb * math.expm1(toy_lnB(x)[0]))
        return quad(g, -12.0, CUT, limit=200, epsabs=1e-13)[0] / norm_k + f / (1.0 - nb * f)
    f, raw = 1.0 - k1 / k0, 1.0 - k1
    return dict(f=f, raw=raw, score=score, limit=lambda f_: brentq(lambda nb: score(nb, f_), 1e-6, 0.9, xtol=1e-12))


def toy_draw(rng, n):
    m = rng.random(n) < NBAR
    mu = MUS[np.minimum(np.searchsorted(np.cumsum(WTS), rng.random(n), side='right'), MUS.size - 1)]
    x = rng.standard_normal(n) + np.where(m, mu, 0.0)
    keep = (x <= CUT) & ~(rng.random(n) < GLITCH)
    return x, m, keep


def test_nbar_fit_detected_fraction(base):
    """a two-component model with a known selection that removes mergers preferentially (x > 4 set aside) and noise
    too (a glitch in 5 percent of segments, merger or not): f = 0.189 (the fraction of mergers set aside, 0.230, is not
    f).  The quadrature confirms the derivation: the expected score with the f term vanishes at the true nbar.  Over 40
    plants of 100000 segments: nbar_fit with the true f is unbiased (mean within 3 standard errors of 0.1); with f = 0 it
    is biased to the predicted limit (0.062, 0.038 below the truth, more than 15 standard errors) and is there within 3
    standard errors; with the fraction of mergers set aside likewise (0.112); the curvature error is within 30 percent
    of the spread; lnl and err are those of sum ln(1 + nbar (B - 1)) - n ln(1 - nbar f) and its second derivative
    (finite differences); and f_det = 0 is plant_fit.py of BASE bit for bit.
    Breaks that fire it: the sign of the f term in the slope flipped, the f term left out of the curvature, f taken as
    the fraction of mergers set aside."""
    tr = toy_truth()
    assert abs(tr['f'] - 0.18897) < 1e-4 and abs(tr['score'](NBAR, tr['f'])) < 1e-9
    limits = {name: tr['limit'](f) for name, f in (('right', tr['f']), ('zero', 0.0), ('raw', tr['raw']))}
    assert abs(limits['right'] - NBAR) < 1e-8
    rng = np.random.default_rng(12)
    est = {k: [] for k in limits}
    errs, fh, fe = [], [], []
    for _ in range(40):
        x, m, keep = toy_draw(rng, 100000)
        lnB = toy_lnB(x[keep])
        for name, f in (('right', tr['f']), ('zero', 0.0), ('raw', tr['raw'])):
            r = plant_fit.nbar_fit(lnB, f)
            est[name].append(r['nbar'])
            if name == 'right':
                errs.append(r['err']['nbar'])
        d = plant_fit.detected_fraction(m, keep)
        fh.append(d['f'])
        fe.append(d['err'])
    lines = ['f %.5f (fraction of mergers set aside %.5f); limits by quadrature: %s' % (
        tr['f'], tr['raw'], ', '.join('%s %.5f' % kv for kv in limits.items()))]
    for name, e in est.items():
        e = np.array(e)
        se = e.std(ddof=1) / np.sqrt(e.size)
        lines.append('f %-5s: mean nbar %.5f +- %.5f, limit %.5f (%+.1f se), truth %.2f (%+.1f se)' % (
            name, e.mean(), se, limits[name], (e.mean() - limits[name]) / se, NBAR, (e.mean() - NBAR) / se))
        assert abs(e.mean() - limits[name]) < 3 * se, lines[-1]
        if name != 'right':
            assert abs(limits[name] - NBAR) > 15 * se, lines[-1]
    spread = np.std(est['right'], ddof=1)
    lines.append('curvature error %.5f, spread %.5f; f measured %.5f +- %.5f (binomial), spread %.5f' % (
        np.mean(errs), spread, np.mean(fh), np.mean(fe), np.std(fh, ddof=1)))
    print('\n' + '\n'.join(lines))
    assert 0.7 < np.mean(errs) / spread < 1.3 and 0.7 < np.mean(fe) / np.std(fh, ddof=1) < 1.3
    assert abs(np.mean(fh) - tr['f']) < 3 * np.std(fh, ddof=1) / np.sqrt(40)
    x, m, keep = toy_draw(np.random.default_rng(13), 100000)
    lnB = toy_lnB(x[keep])
    B1 = np.expm1(lnB)
    f = tr['f']
    lnl = lambda nb: float(np.sum(np.log1p(nb * B1)) - lnB.size * np.log1p(-nb * f))
    r = plant_fit.nbar_fit(lnB, f)
    h = 1e-4
    curv = -(lnl(r['nbar'] + h) - 2 * lnl(r['nbar']) + lnl(r['nbar'] - h)) / h ** 2
    assert abs(r['lnl'] - lnl(r['nbar'])) < 1e-6 * abs(lnl(r['nbar'])) + 1e-6
    assert abs(r['err']['nbar'] ** -2 / curv - 1) < 1e-3, (r['err']['nbar'] ** -2, curv)
    assert abs((lnl(r['nbar'] + 1e-7) - lnl(r['nbar'] - 1e-7)) / 2e-7) < 1e-3 * np.sqrt(curv)
    for lnB_ in (lnB, toy_lnB(np.random.default_rng(14).standard_normal(5000))):
        a, b = plant_fit.nbar_fit(lnB_), base['plant_fit'].nbar_fit(lnB_)
        assert a['nbar'] == b['nbar'] and a['err'] == b['err'] and a['lnl'] == b['lnl'] and a['edge'] == b['edge']
    with pytest.raises(ValueError):
        plant_fit.nbar_fit(lnB, 1.0)


def test_detected_fraction_counts():
    """detected_fraction against the counts by hand: f = 1 - (kept1 / n1) / (kept0 / n0) and the first-order binomial
    error; nan without mergers or with none kept; no merger set aside gives 0 with the rule-of-three bound 3 / n1"""
    m = np.r_[np.ones(200, bool), np.zeros(800, bool)]
    keep = np.ones(1000, bool)
    keep[:30], keep[200:240] = False, False
    d = plant_fit.detected_fraction(m, keep)
    k1, k0 = 170 / 200, 760 / 800
    assert (d['n1'], d['kept1'], d['n0'], d['kept0']) == (200, 170, 800, 760)
    assert d['f'] == pytest.approx(1 - k1 / k0, rel=1e-15)
    assert d['err'] == pytest.approx(k1 / k0 * np.sqrt((1 - k1) / (200 * k1) + (1 - k0) / (800 * k0)), rel=1e-15)
    assert np.isnan(plant_fit.detected_fraction(np.zeros(10, bool), np.ones(10, bool))['f'])
    assert np.isnan(plant_fit.detected_fraction(m, ~m)['f'])
    z = plant_fit.detected_fraction(m, np.ones(1000, bool))
    assert z['f'] == 0 and z['err'] == 0 and z['upper'] == 3 / 200


def test_summarize_f_det_and_w_net(tmp_path, monkeypatch, capsys):
    """summarize gives plant_fit.report the --w-net and --f-det it is given (default 84 and 0) and prints which were
    used; the window nbar of each row is nbar_fit of the kept segments with that f.
    Break that fires it: --f-det not passed to the report."""
    lnB, m = synthetic_B(6000, 0.05, 5)
    wmax, wnet, single, coinc = planted_cut()
    cfg = plant.with_model(plant.campaign(plant.parser().parse_args(['s', 'cpu', '--n', '1', '--out', 'x', '--nbar',
                                                                      '0.05'])), 'ab' * 32, 64)
    for v in summarize.ORDER:
        write_files(str(tmp_path / v), cfg, lnB, m, wmax, wnet)
    monkeypatch.setattr(summarize, 'check_week', lambda out, variants, recorded=(): (6000, {}))
    seen = []
    real = summarize.plant_fit.report
    monkeypatch.setattr(summarize.plant_fit, 'report', lambda path, **kw: seen.append(kw) or real(path, **kw))
    keep = ~(single | coinc)
    for argv, w_net, f in (([], 84.0, 0.0), (['--f-det', '0.012', '--w-net', '84'], 84.0, 0.012),
                           (['--w-net', 'inf'], np.inf, 0.0)):
        seen.clear()
        summarize.main([str(tmp_path)] + argv)
        text = capsys.readouterr().out
        assert all(kw['w_net'] == w_net and kw['f_det'] == f for kw in seen) and len(seen) == len(summarize.ORDER)
        assert 'network cut wnet > %g' % w_net in text and 'detected fraction f = %g' % f in text
        assert ('given with --f-det' in text) == (f != 0)
        k = keep if np.isfinite(w_net) else ~single
        W = plant_fit.nbar_fit(lnB[k], f)
        row = [l for l in text.splitlines() if l.startswith('pr ')][0]
        assert '%s+-%s' % (plant_fit._f(W['nbar'], '%.3g'), plant_fit._f(W['err']['nbar'], '%.2g')) in row, row


# ------------------------------------------------------------------- the windows stage with the mergers of stage 1


def test_wnet_window_evidences_with_population():
    """window_evidences with lags gives the same wnet with and without a population (the merger evidence of stage 1),
    the brute force of the unscaled data; every other array is that of the call without lags (with the population: of
    the call with the population and without lags), bit for bit.
    Break that fires it: wnet computed only when no population is given."""
    import test_pe_unequal as tu
    import test_stage1_mergers as tm
    c = tu.make_case(0.0)
    lags = np.asarray(tu.TAU)
    pop = tm.tiny_population(c['bank'], tm.S_LADDER)
    plain = pe.window_evidences(c['segs'], c['bank'], lags=lags)
    with_pop = pe.window_evidences(c['segs'], c['bank'], lags=lags, population=pop)
    pop_only = pe.window_evidences(c['segs'], c['bank'], population=pop)
    assert set(with_pop) == set(pop_only) | {'wnet'} and set(plain) <= set(with_pop)
    for k in pop_only:
        assert np.array_equal(np.asarray(with_pop[k]), np.asarray(pop_only[k])), k
    assert np.array_equal(plain['wnet'], with_pop['wnet'])
    ref = np.stack([traj_wnet(c, e, lags) for e in range(3)])
    assert np.allclose(with_pop['wnet'], ref, rtol=1e-13, atol=0)
    print('\nwnet with and without a population: %s' % np.array2string(with_pop['wnet'], precision=3))


def test_wnet_windows_stage_driver(tmp_path, capsys):
    """plant.py --stage windows with a store folder (headers only, lags -2 .. 2) writes wnet next to the merger evidence:
    per segment the largest over the templates of window_evidences(lags) of the same simulated segments; with 'none'
    in place of the store it writes neither, and says that the network cut will lack wnet.
    Break that fires it: the lags not passed to the windows stage."""
    import test_stage1_drivers as td
    store = td.header_store(str(tmp_path / 'store'))
    argv = ['cpu', '--stage', 'windows', '--n', '2', '--batch', '2', '--nbar', '0.5', '--seed', '3', '--glitch-states',
            '0.9:0.0001,0.1:0.001', '--gamma', '4', '--rho-min', '5', '--zmin', '0.5']
    out_s, out_n = str(tmp_path / 'with'), str(tmp_path / 'none')
    assert plant.main([store] + argv + ['--out', out_s]) == 0
    assert plant.main(['none'] + argv + ['--out', out_n]) == 0
    assert 'writes no wnet' in capsys.readouterr().out
    name = 'win_0000000000_0000000002.npz'
    with np.load(os.path.join(out_s, name)) as a, np.load(os.path.join(out_n, name)) as b:
        assert {'wnet', 'ln_merger_H'} <= set(a.files) and not {'wnet', 'ln_merger_H'} & set(b.files)
        cfg = {k: v for k, v in json.loads(str(a['meta'])).items() if k != 'run'}
        wnet = a['wnet']
    from exmarg.search import mass_bank
    seg, _ = plant.simulate_batch(cfg, 0, 2, plant.Sampler(plant.population(cfg['zmin']), cfg['mdet_range']))
    ref = pe.window_evidences(seg, mass_bank(plant.MASSES, seg, 'cpu'), lags=np.arange(-2, 3))['wnet'].max(axis=1)
    assert np.array_equal(wnet, ref), (wnet, ref)
    print('\nwindows stage with the store: wnet %s' % np.round(wnet, 2))


def test_cut_windows_fit_with_mergers(tmp_path):
    """windows_fit.fit on the toy with merger evidence of tests/test_stage1_mergers.py (2000 segments) and 25 planted
    loud coincidences (wnet 90) and 6 segments loud in H: the fit without mergers (nbar 0) and the fit with them (nbar
    0.05) set aside the same segments, the loud and the planted ones, and both models record w_net 84 and the same
    set_aside_net; switched off, both keep the single-detector selection.
    Break that fires it: the network condition applied in one of the two fits only."""
    import test_stage1_mergers as tm
    folder = str(tmp_path / 'toy')
    tm.write_toy_folder(folder, True)
    rng = np.random.default_rng(6)
    loud_H = np.zeros(2000, bool)
    loud_H[rng.choice(2000, 6, replace=False)] = True
    coinc = np.zeros(2000, bool)
    coinc[rng.choice(np.flatnonzero(~loud_H), 25, replace=False)] = True
    for path in sorted(glob.glob(os.path.join(folder, 'win_*.npz'))):
        with np.load(path) as z:
            arrays = {key: z[key] for key in z.files if key != 'meta'}
            meta = json.loads(str(z['meta']))
        k = arrays['k']
        arrays['wmax_H'] = np.where(loud_H[k], 100.0, arrays['wmax_H'])
        arrays['wnet'] = np.where(coinc[k], 90.0, np.where(loud_H[k], 110.0, np.minimum(
            arrays['wmax_H'] + arrays['wmax_L'], 83.0)))
        plant.write_batch(path, {key: v for key, v in meta.items() if key != 'run'}, meta['run'], arrays)
    loud = loud_H | (np.concatenate([np.load(p)['wmax_L'] for p in sorted(glob.glob(os.path.join(folder, 'win_*.npz')))])
                     > 72)
    fits = {nb: windows_fit.fit(folder, n_state=1, record=(1,), nbar=nb) for nb in (0.0, 0.05)}
    for nb, r in fits.items():
        assert np.array_equal(r['keep'], ~(loud | coinc)), nb
        assert r['model']['settings']['w_net'] == 84.0 and r['model']['segments']['set_aside_net'] == (loud_H | coinc).sum()
    assert fits[0.05]['model']['nbar'] == 0.05 and 'nbar' not in fits[0.0]['model']
    for nb in (0.0, 0.05):
        r = windows_fit.fit(folder, n_state=1, record=(1,), nbar=nb, w_net=np.inf)
        assert np.array_equal(r['keep'], ~loud) and 'w_net' not in r['model']['settings'], nb
    print('\n' + '\n'.join(fits[0.05]['lines'][:7]))
