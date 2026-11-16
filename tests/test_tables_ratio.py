"""scripts/build_tables/build_ratio.py, the redshift store of asymmetry w (unequal noise in H1 and L1), at small n_max
and nside on the CPU (the cupy code of w_gpu.py runs on numpy when cupy is absent).  Every reference is written here
from the formula, with float64 responses from exmarg.detectors.Fplus_Fcross at each psi and its own delay weights:
(i) stage W with s_H != s_L against the quadrature of the complex monomials, (ii) stage T against ln sum_n alpha^n K_n
summed key by key, (iii) T against ln < e^{-D} I0(2 sqrt(alpha) |X|) > with no series, (iv) at w = 0 the published
chain w_table_s_gpu -> build_w.save_node -> build_k -> build_bd, (v) the noise mean of the truncated series against
< P(Poisson(D) <= N) >, (vi) slices of the sky add up to the whole sweep, (vii) stage R: the entries it must cover (every
raw entry a read of data below the cuts depends on) against a cell-by-cell emulation of the read, the bound of the
orders left out against the sum itself, and the repaired entries against the quadrature."""
import sys
import types
from functools import lru_cache
from pathlib import Path

import healpy as hp
import numpy as np
import pytest
from scipy.special import comb, gammaincc, gammaln, i0e

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'build_tables'))
try:
    import cupy  # noqa: F401
except ImportError:
    shim = types.ModuleType('cupy')
    shim.__dict__.update({k: getattr(np, k) for k in dir(np) if not k.startswith('_')})
    shim.asnumpy = np.asarray
    sys.modules['cupy'] = shim
import build_ratio as br
from exmarg.detectors import Fplus_Fcross, c, x
from exmarg.tables import _quad

# numpy 2 with Apple Accelerate flags spurious floating-point errors in matmul; finiteness is asserted instead
pytestmark = pytest.mark.filterwarnings('ignore:.*encountered in matmul:RuntimeWarning')

NSIDE, N_TAU = 2, 257
SBAR, ASYM = np.array([1.7, 0.35]), np.array([-0.45, 0.3])        # two nodes in one sweep, both signs of w


@lru_cache(maxsize=None)
def sweep(n_max, s_H, s_L):
    tau, keys, V = br.w_sweep(np.array(s_H), np.array(s_L), n_max=n_max, nside=NSIDE, n_tau=N_TAU, chunk=48)
    assert np.isfinite(V).all()
    return tau, keys, V


def stage_w(n_max, sbar=SBAR, w=ASYM):
    return sweep(n_max, tuple(sbar * (1 + w)), tuple(sbar * (1 - w)))


def responses(th, ph, psi, cg):
    """R_I = (1 + chi^2)/2 F+_I(psi) + i chi Fx_I(psi), complex128, (pixel, psi, chi), F at each psi directly"""
    out = []
    for I in 'HL':
        Fp, Fx = Fplus_Fcross(I, th[:, None], ph[:, None], psi[None, :])
        out.append(0.5 * (1 + cg**2)[None, None, :] * Fp[:, :, None] + 1j * cg[None, None, :] * Fx[:, :, None])
    return out


def delay_weights(tau_lag):
    """(n_tau, npix): linear share of each pixel's delay on the fine grid, 1/npix per pixel; (n_lag, npix): each fine
    point added to the data lag nearest to it"""
    npix = hp.nside2npix(NSIDE)
    th, ph = hp.pix2ang(NSIDE, np.arange(npix))
    dx = x['H'] - x['L']
    t = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) @ dx / c
    tf = np.linspace(-1.0, 1.0, N_TAU) * np.linalg.norm(dx) / c
    g = (t - tf[0]) / (tf[1] - tf[0])
    j0 = np.clip(np.floor(g).astype(int), 0, N_TAU - 2)
    fine = np.zeros((N_TAU, npix))
    np.add.at(fine, (j0, np.arange(npix)), (1 - (g - j0)) / npix)
    np.add.at(fine, (j0 + 1, np.arange(npix)), (g - j0) / npix)
    lag = np.argmin(np.abs(np.asarray(tau_lag)[:, None] - tf[None, :]), axis=0)
    lagw = np.zeros((len(tau_lag), npix))
    np.add.at(lagw, lag, fine)
    return th, ph, fine, lagw


def tau_lag(gr):
    return gr['tau_idx'] / 4096.0


def check_stage_w(n_max, sbar=SBAR, w=ASYM):
    """W = e^{l_n} V from the sweep at (s_H, s_L) = (sbar (1 + w), sbar (1 - w)) against sbar^n / n! <
    e^{-sbar [(1 + w) S_H + (1 - w) S_L]} Re(R_H^a conj(R_H)^b R_L^(n-a) conj(R_L)^(n-b)) > in the data lags; the
    error is measured against the same average of the modulus"""
    tau, keys, V = stage_w(n_max, sbar, w)
    th, ph, _, lagw = delay_weights(tau_lag(br.grids(tau)))
    psi, cg, wq = _quad(n_max, None, None)
    R1, R2 = responses(th, ph, psi, cg)
    worst = 0.0
    for j, s in enumerate(sbar):
        damp = np.exp(-s * ((1 + w[j]) * np.abs(R1)**2 + (1 - w[j]) * np.abs(R2)**2)) * wq[None, None, :]
        for k, (n, a, b) in enumerate(keys):
            mono = R1**a * np.conj(R1)**b * R2**(n - a) * np.conj(R2)**(n - b)
            f = np.exp(n * np.log(s) - gammaln(n + 1.0))
            ref = f * lagw @ (damp * mono.real).sum(axis=(1, 2))
            mod = f * lagw @ (damp * np.abs(mono)).sum(axis=(1, 2))
            W = V[j, :, k] * np.exp(br.ln_scale(n, s))
            worst = max(worst, float(np.max(np.abs(W - ref) / (mod + 1e-300))))
    return worst


def test_stage_w_against_direct_quadrature():
    worst = check_stage_w(10)
    # w_gpu forms the responses in complex64 (relative rounding ~1e-7 per factor, 2n factors)
    assert worst < 1e-5, 'stage W differs from the direct quadrature by %.2e of the modulus' % worst


def lnsum_direct(W, n_max, sbar, q1, q2, phi, alpha):
    """ln sum_n alpha^n K_n, K_n = sum over every (a, b) of C(n,a) C(n,b) W_ab q1^(a+b) q2^(2n-a-b) cos[(a-b) phi]
    / (sbar^n n!), W given for b <= a and mirrored; also the same sum of the moduli of the terms"""
    tot, mod, k = 0.0, 0.0, 0
    for n in range(n_max + 1):
        Wn = np.zeros((n + 1, n + 1))
        for a in range(n + 1):
            for b in range(a + 1):
                Wn[a, b] = Wn[b, a] = W[k]
                k += 1
        a, b = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
        term = comb(n, a) * comb(n, b) * Wn * q1**(a + b) * q2**(2 * n - a - b) * np.cos((a - b) * phi)
        f = alpha**n / (sbar**n * np.exp(gammaln(n + 1.0)))
        tot += f * term.sum()
        mod += f * np.abs(term).sum()
    return tot, mod


def check_stage_t(n_max, s_ref_builder, node=0, npts=60, seed=1):
    """T from t_from_v (float64 out) at random (lag, q1, q2, phi, alpha) against lnsum_direct; returns the largest
    |e^T - L| / (1e-12 L + 1e-13 sum of moduli), which must stay below 1"""
    tau, keys, V = stage_w(n_max)
    sbar = SBAR[node]
    gr = br.grids(tau)
    Vl = V[node]
    lags = np.nonzero(Vl[:, 0] > 0)[0]
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(npts):
        lag = rng.choice(lags)
        q1, q2 = np.exp(rng.uniform(np.log(0.05), np.log(100.0), 2))
        phi, alpha = rng.uniform(0, 2 * np.pi), np.exp(rng.uniform(np.log(0.16), np.log(4e4)))
        T = br.t_from_v(Vl[lag:lag + 1], s_ref_builder, [np.log(q1 / q2)], [np.log(alpha) + 2 * np.log(q2)], [phi],
                        n_max=n_max, out_dtype=np.float64)[0, 0, 0, 0]
        W = Vl[lag] * np.exp(br.ln_scale(np.array([k[0] for k in keys]), sbar))
        L, mod = lnsum_direct(W, n_max, sbar, q1, q2, phi, alpha)
        worst = max(worst, abs(np.exp(T) - L) / (1e-12 * abs(L) + 1e-13 * mod))
    return worst


def test_stage_t_against_series_sum():
    worst = check_stage_t(10, SBAR[0])
    assert worst < 1, 'stage T differs from ln sum alpha^n K_n by %.2f times the rounding bound' % worst


def lnlam_direct(sbar, w, n_max, lagw_row, th, ph, q1, q2, phi, alpha):
    """ln < e^{-sbar [(1 + w)|R_H|^2 + (1 - w)|R_L|^2]} I0(2 sqrt(alpha) |conj(R_H) z_1 + conj(R_L) z_2|) > in one
    data lag, z_1 = q1 e^{i phi}, z_2 = q2: the quadrature of the table with no series"""
    use = lagw_row > 0
    psi, cg, wq = _quad(n_max, None, None)
    R1, R2 = responses(th[use], ph[use], psi, cg)
    D = sbar * ((1 + w) * np.abs(R1)**2 + (1 - w) * np.abs(R2)**2)
    arg = 2 * np.sqrt(alpha) * np.abs(np.conj(R1) * q1 * np.exp(1j * phi) + np.conj(R2) * q2)
    e = np.log(lagw_row[use])[:, None, None] + np.log(wq)[None, None, :] - D + arg + np.log(i0e(arg))
    top = e.max()
    return top + np.log(np.exp(e - top).sum())


def check_against_bessel(n_max, sbar, w, npts=40, seed=2, vmax=4.0):
    """T built at w against lnlam_direct; points with sqrt(alpha) q2 (e^Delta + 1) <= vmax, where the series to
    n_max has converged (|R_I| <= 1, so the Bessel argument is at most 2 vmax)"""
    tau, keys, V = stage_w(n_max, np.array([sbar]), np.array([w]))
    gr = br.grids(tau)
    th, ph, _, lagw = delay_weights(tau_lag(gr))
    Vl = V[0]
    lags = np.nonzero(lagw.sum(axis=1) > 0)[0]
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(npts):
        lag = rng.choice(lags)
        q2 = np.exp(rng.uniform(np.log(0.05), np.log(100.0)))
        dl = rng.uniform(-5.0, 5.0)
        v = np.exp(rng.uniform(np.log(0.3), np.log(vmax)))
        alpha = (v / (q2 * (np.exp(dl) + 1))) ** 2
        phi = rng.uniform(0, 2 * np.pi)
        T = br.t_from_v(Vl[lag:lag + 1], sbar, [dl], [np.log(alpha) + 2 * np.log(q2)], [phi], n_max=n_max,
                        out_dtype=np.float64)[0, 0, 0, 0]
        ref = lnlam_direct(sbar, w, n_max, lagw[lag], th, ph, q2 * np.exp(dl), q2, phi, alpha)
        worst = max(worst, abs(T - ref))
    return worst


def test_table_against_bessel_quadrature():
    for sbar, w in ((1.3, -0.3), (2.5, -0.6)):
        worst = check_against_bessel(20, sbar, w)
        # tail of the series beyond n = 20 at Bessel argument <= 8: below e^-38; complex64 responses in w_gpu
        assert worst < 1e-6, 'sbar %g, w %g: T differs from the Bessel quadrature by %.2e' % (sbar, w, worst)


def published_chain(tmp_path, s, n_max, lags):
    """w = 0 through the published scripts: w_table_s_gpu -> build_w.save_node (float32 W) -> build_k.load_node and
    compile_order (float32 lnK, on the given lags) -> build_bd.build_node.  Returns the W dict before storage, the
    stored float32 W, the lnK array and the bd file"""
    import build_bd
    import build_k
    import build_w
    import w_gpu
    tau, W = w_gpu.w_table_s_gpu(s, s, n_max, n_tau=N_TAU, nside=NSIDE, chunk=48)
    stem = tmp_path / ('sladder_n%03d_s%09.3f' % (n_max, s))
    build_w.save_node(stem, s, tau, W)
    keys, WM, tau_idx = build_k.load_node(str(stem))
    lnq = np.log(np.geomspace(build_k.QMAX / build_k.QSPAN, build_k.QMAX, build_k.NQ))
    phi_g = np.arange(build_k.NPHI) * (2 * np.pi / build_k.NPHI)
    by_n = {n: [] for n in range(n_max + 1)}
    for j, (n, a, b) in enumerate(keys):
        by_n[int(n)].append((j, int(a), int(b)))
    lnK = np.stack([build_k.compile_order(n, s, keys, WM[:, lags], lnq, by_n, phi_g, len(lags))
                    for n in range(n_max + 1)])
    kstem = tmp_path / ('kstore_n%03d_s%09.3f' % (n_max, s))
    np.save(str(kstem) + '.lnK.npy', lnK)
    np.savez(str(kstem) + '.meta.npz', u_node=s, s_node=s, lnq=lnq, tau_idx=tau_idx[lags])
    build_bd.build_node(str(kstem), str(tmp_path / 'bd'))
    bd = np.load(tmp_path / 'bd' / ('bd_n%03d_s%09.3f.npz' % (n_max, s)))
    W32 = np.load(str(stem) + '.W.npy')
    return W, W32, lnK, bd


def ln_abs_series(W, keys, s, delta, beta):
    """ln sum_n alpha^n sum_{a,b} C(n,a) C(n,b) |W_ab| e^{(a+b) Delta} / (s^n n!) at q2 = 1, (n_lag, n_beta, n_delta):
    the series with every term made positive, an upper bound of the sum of the moduli at any phi"""
    d = np.asarray(delta)[None, :]
    out = []
    for n in range(max(k[0] for k in keys) + 1):
        acc = 0.0
        for j, (m, a, b) in enumerate(keys):
            if m == n:
                acc = acc + (2.0 if a > b else 1.0) * comb(n, a) * comb(n, b) * np.abs(W[:, j:j + 1]) * \
                    np.exp((a + b) * d - 2 * n * np.maximum(d, 0))
        ln_an = np.log(np.maximum(acc, 1e-300)) + 2 * n * np.maximum(d, 0) - n * np.log(s) - gammaln(n + 1.0)
        out.append(n * np.asarray(beta)[None, :, None] + ln_an[:, None, :])
    out = np.array(out)
    top = out.max(axis=0)
    return top + np.log(np.exp(out - top).sum(axis=0))


def check_w0(tmp_path, s=1.7, n_max=8):
    """(a) the sweep at s_H = s_L = s, times e^{l_n}, against w_table_s_gpu; (b) t_from_v fed with the stored float32
    W against build_bd, element by element within the float32 rounding of lnK and T, 2^-24 (max_n |lnK_n| + 2 |T|),
    plus the float64 rounding of the sums amplified by the condition number of the series, 1e3 2^-52 e^{T_abs - T}"""
    tau, keys, V = stage_w(n_max, np.array([s]), np.array([0.0]))
    gr = br.grids(tau)
    lags = list(np.argsort(-V[0][:, 0])[:3])
    W, W32, lnK, bd = published_chain(tmp_path, s, n_max, lags)
    n_key = np.array([k[0] for k in keys])
    Wme = V[0] * np.exp(br.ln_scale(n_key, s))[None, :]
    Wpub = gr['rb'] @ np.stack([W[k] for k in keys], axis=1)
    err_w = float(np.max(np.abs(Wme - Wpub) / np.abs(Wpub).max(axis=0)[None, :]))     # rebinned before or after
    V32 = W32.T.astype(float) * np.exp(-br.ln_scale(n_key, s))[None, :]
    T = br.t_from_v((gr['rb'] @ V32)[lags], s, gr['delta'], gr['beta'], gr['phi'], n_max=n_max)
    lnk = np.abs(lnK[:, :, :, -1, :gr['nphif']]).max(axis=0)                      # Delta <= 0: q2 = q_max
    lnk = np.concatenate([lnk, np.abs(lnK[:, :, -1, :, :gr['nphif']]).max(axis=0)[:, ::-1][:, 1:]], axis=1)
    t_abs = ln_abs_series((gr['rb'] @ W32.T.astype(float))[lags], keys, s, gr['delta'], gr['beta'])
    cond = np.exp(np.minimum(t_abs[:, None, :, :] - bd['T'].astype(float), 700.0))
    bound = 2.0**-24 * (lnk.transpose(0, 2, 1)[:, :, None, :] + 2 * np.abs(bd['T'])) + 1e3 * 2.0**-52 * cond
    err_t = float(np.max(np.abs(T - bd['T']) / bound))
    grids_equal = all(np.array_equal(bd[k], v) for k, v in (('beta', gr['beta']), ('delta', gr['delta']),
                                                              ('tau_idx', gr['tau_idx'][lags]), ('lnq', gr['lnq'])))
    return err_w, err_t, grids_equal


def test_w0_equals_published_chain(tmp_path):
    err_w, err_t, grids_equal = check_w0(tmp_path)
    assert grids_equal, 'beta, delta, tau_idx or lnq differ from build_bd'
    assert err_w < 1e-12, 'stage W at w = 0 differs from w_table_s_gpu (rebinned) by %.2e' % err_w
    assert err_t < 1, 'T at w = 0 differs from build_bd by %.2f times the float32 bound' % err_t


def check_noise_mean(n_max, sbar, w):
    """noise_mean of the sweep against the quadrature of the Poisson probability P(N_D <= n_max) = Q(n_max + 1, D),
    D = sbar [(1 + w)|R_H|^2 + (1 - w)|R_L|^2]"""
    tau, keys, V = stage_w(n_max, np.array([sbar]), np.array([w]))
    E = br.noise_mean(V[0].sum(axis=0), keys, sbar, w)
    npix = hp.nside2npix(NSIDE)
    th, ph = hp.pix2ang(NSIDE, np.arange(npix))
    psi, cg, wq = _quad(n_max, None, None)
    R1, R2 = responses(th, ph, psi, cg)
    D = sbar * ((1 + w) * np.abs(R1)**2 + (1 - w) * np.abs(R2)**2)
    ref = float((gammaincc(n_max + 1, D) * wq[None, None, :]).sum() / npix)
    return E, ref


def test_noise_mean_of_the_truncated_series():
    E, ref = check_noise_mean(12, 6.0, -0.45)
    assert 1e-4 < 1 - ref < 0.5, 'the truncation must show at this node (deficit %.2e)' % (1 - ref)
    assert abs(E - ref) < 1e-8, 'noise mean %.12f from W, %.12f from the Poisson quadrature' % (E, ref)


def test_sky_slices_add_up():
    s_H, s_L = (2.0, 0.4), (0.9, 0.5)
    tau, keys, V = sweep(8, s_H, s_L)
    npix = hp.nside2npix(NSIDE)
    parts = [br.w_sweep(np.array(s_H), np.array(s_L), n_max=8, nside=NSIDE, chunk=5, pix=br.part_range(k, 3, npix))[2]
             for k in range(3)]
    err = np.abs(sum(parts) - V).max() / np.abs(V).max()
    # float64 sums in another order (chunk 5 against 48); one pixel lost or counted twice would be ~1/48
    assert err < 1e-12, 'three slices of the sky differ from the whole sweep by %.2e' % err


def test_lags_without_sky():
    """no pixel deposits into a lag: V = 0 there and T = ln 1e-300 (the minimum of the published tables) at every
    (phi, beta, Delta); in the other lags T is finite and above it"""
    tau, keys, V = stage_w(8)
    gr = br.grids(tau)
    _, _, _, lagw = delay_weights(tau_lag(gr))
    empty = lagw.sum(axis=1) == 0
    assert empty.any() and (~empty).any()
    assert np.all(V[0][empty] == 0) and np.all(V[0][~empty, 0] > 0)
    T = br.t_from_v(V[0], SBAR[0], gr['delta'][::20], gr['beta'][::20], gr['phi'][::8], n_max=8, out_dtype=np.float64)
    assert np.all(T[empty] == br.LN_FLOOR)
    assert np.isfinite(T).all() and np.all(T[~empty] > br.LN_FLOOR)


def flag_and_repair(T, V, s_H, s_L, sbar, beta, delta, phi, zcut, gap, n_max, wnet=np.inf, tol=1e-6, wseries=np.inf):
    flags, counts = br.flag_entries(T, V, s_H, s_L, sbar, beta, delta, zcut, gap, n_max=n_max, nside=NSIDE, wnet=wnet,
                                    tol=tol, wseries=wseries)
    Tr, done = br.repair(T.copy(), flags, s_H, s_L, beta, delta, phi, n_max=n_max, nside=NSIDE)
    assert done == counts['flagged'] == int(flags.sum())
    return Tr, flags, counts


def check_repair(n_max=20, sbar=2.5, w=-0.6, npts=40, seed=3):
    """stage R: with every entry flagged, the repaired T against lnlam_direct at random entries (max error); the number
    of entries the production threshold (gap 18) flags below the loudness cut in this well-conditioned case (on this
    coarse grid the needed entries beyond the cut are a factor 1000 louder, and some of them are flagged); and with a
    network cut and every needed entry flagged, the number repaired against the needed cells"""
    tau, keys, V = stage_w(n_max, np.array([sbar]), np.array([w]))
    gr = br.grids(tau)
    th, ph, _, lagw = delay_weights(tau_lag(gr))
    beta, delta, phi = gr['beta'][::25], gr['delta'][::27], gr['phi'][::12]
    T = br.t_from_v(V[0], sbar, delta, beta, phi, n_max=n_max, out_dtype=np.float64)
    s_H, s_L = sbar * (1 + w), sbar * (1 - w)
    args = (T, V[0], s_H, s_L, sbar, beta, delta, phi)
    Tr, _, c_all = flag_and_repair(*args, 1e30, -np.inf, n_max)
    _, f_prod, _ = flag_and_repair(*args, 72.0, 18.0, n_max)
    _, _, c_net = flag_and_repair(*args, 1e30, -np.inf, n_max, wnet=40.0)
    lags = np.nonzero(lagw.sum(axis=1) > 0)[0]
    need = br.need_mask(beta, delta, s_H, s_L, 1e30, 40.0)[0]
    assert 0 < need.sum() < need.size
    assert c_net['flagged'] == lags.size * phi.size * int(need.sum()), 'the network cut repaired %d' % c_net['flagged']
    # compared where production repairs, below the loudness cut 72: there the Bessel argument stays below ~70, while
    # at the far corner of the grid (beta 20, Delta 7.6) it reaches 1e8 and a 1e-16 rounding of |X| moves ln I0 by 1e-8
    below = (beta[:, None] < np.log(72 * s_L)) & (beta[:, None] + 2 * delta[None, :] < np.log(72 * s_H))
    reach = np.argwhere(below)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(npts):
        (b, d), l, k = reach[rng.integers(len(reach))], rng.choice(lags), rng.integers(phi.size)
        ref = lnlam_direct(sbar, w, n_max, lagw[l], th, ph, np.exp(delta[d]), 1.0, phi[k], np.exp(beta[b]))
        worst = max(worst, abs(Tr[l, k, b, d] - ref))
    return worst, c_all['flagged'], int(f_prod[:, :, below].sum()), lags.size * phi.size * beta.size * delta.size


def test_repair_is_the_quadrature():
    worst, n_all, n_prod, n_sky = check_repair()
    assert n_all == n_sky, 'with every entry flagged, %d of %d entries with sky were repaired' % (n_all, n_sky)
    # two codes of the same quadrature (responses rotated from psi = 0 against Fplus_Fcross at each psi); the repair takes
    # its exponentials and i0e in float32 (relative 1e-7 per term, the sum in float64)
    assert worst < 1e-6, 'repaired T differs from the direct quadrature by %.2e' % worst
    assert n_prod == 0, 'the threshold gap 18 flags %d entries below the cut in this well-conditioned case' % n_prod


def inside_cuts(b, d, s_H, s_L, zcut, wnet):
    """data below the loudness cuts: w_L = e^beta / s_L < zcut, w_H = e^(beta + 2 Delta) / s_H < zcut, w_L + w_H <= wnet"""
    wL, wH = np.exp(b) / s_L, np.exp(b + 2 * d) / s_H
    return (wL < zcut) & (wH < zcut) & (wL + wH <= wnet)


def entries_read(beta, delta, s_H, s_L, zcut, wnet, sub=(1e-7, 0.25, 0.5, 0.75, 1 - 1e-7)):
    """the raw table entries on which a read of data below the cuts depends, emulated cell by cell: a cell of the
    (beta, Delta) grid is read when one of its interior points (a lattice down to 1e-7 of a cell from its corners) is
    inside the cuts; the read takes the four corners of the cell from the unbiased table (gather.py: b0, b0 + 1, d0,
    d0 + 1), and an unbiased entry is the raw one minus its second differences along beta and along Delta
    (store.unbias: the entry and its two neighbours on each axis)"""
    nb, nd = beta.size, delta.size
    db, dd = beta[1] - beta[0], delta[1] - delta[0]
    hit = np.zeros((nb - 1, nd - 1), bool)
    for fb in sub:
        for fd in sub:
            hit |= inside_cuts((beta[:-1] + fb * db)[:, None], (delta[:-1] + fd * dd)[None, :], s_H, s_L, zcut, wnet)
    read = np.zeros((nb, nd), bool)
    for i in (0, 1):
        for j in (0, 1):
            read[i:nb - 1 + i, j:nd - 1 + j] |= hit
    raw = read.copy()
    raw[1:] |= read[:-1]
    raw[:-1] |= read[1:]
    raw[:, 1:] |= read[:, :-1]
    raw[:, :-1] |= read[:, 1:]
    return read, raw


def check_need(sbar, w, zcut=72.0, wnet=84.0):
    gr = br.grids(stage_w(8)[0])
    beta, delta = gr['beta'], gr['delta']
    s_H, s_L = sbar * (1 + w), sbar * (1 - w)
    read, raw = entries_read(beta, delta, s_H, s_L, zcut, wnet)
    need = br.need_mask(beta, delta, s_H, s_L, zcut, wnet)[0]
    grid_inside = inside_cuts(beta[:, None], delta[None, :], s_H, s_L, zcut, wnet)
    return need, raw, read, grid_inside


def test_need_mask_is_what_the_search_reads():
    """the entries stage R answers for are exactly those a read of data below the cuts depends on; the grid points
    below the cuts alone (the rule before) miss the upper corners of the last cells and the neighbours that the
    removal of the interpolation bias brings in"""
    for sbar, w, zcut, wnet in ((0.29, 0.0, 72.0, 84.0), (34.26, -0.30, 72.0, 84.0), (628.22, -0.60, 72.0, 84.0),
                                (98.66, -0.30, 72.0, np.inf), (5.38, 0.3, 50.0, 60.0)):
        need, raw, read, grid_inside = check_need(sbar, w, zcut, wnet)
        assert raw.any() and not raw.all()
        assert not (raw & ~need).any(), 'sbar %g w %g: %d entries are read and not covered' % (sbar, w, (raw & ~need).sum())
        assert np.array_equal(need, raw), 'sbar %g w %g: %d entries covered and never read' % (sbar, w, (need & ~raw).sum())
        import check_ratio
        other = check_ratio.reach_mask(br.grids(stage_w(8)[0])['beta'], br.grids(stage_w(8)[0])['delta'],
                                       sbar * (1 + w), sbar * (1 - w), zcut, wnet)
        assert np.array_equal(other, need), 'check_ratio draws from other entries than the builder answers for'
        missed = raw & ~grid_inside
        assert (read & ~grid_inside).sum() > 100 and missed.sum() > (read & ~grid_inside).sum(), \
            'the grid points below the cuts must miss entries that are read (%d)' % missed.sum()


def test_tail_bound_of_the_orders_left_out():
    """ln_tail >= ln sup_{x <= x_max} e^{-x} sum_{n > N} (x W)^n / (n!)^2 (the sum taken to 4000 orders, the supremum
    on 4000 points of x), equal to it when x_max <= N + 1 (every term largest at x_max), within a factor e^3 otherwise"""
    for n_max, x_max, W in ((12, 0.7, 5.0), (12, 12.0, 8.0), (12, 300.0, 30.0), (120, 40.0, 125.0), (120, 121.0, 60.0),
                            (120, 800.0, 100.0), (120, 8000.0, 84.0), (120, 8000.0, 190.0)):
        n = np.arange(n_max + 1.0, n_max + 4001.0)
        xs = np.linspace(x_max / 4000, x_max, 4000)
        lt = n[:, None] * np.log(xs[None, :] * W) - xs[None, :] - 2 * gammaln(n + 1.0)[:, None]
        top = lt.max(axis=0)
        brute = float((top + np.log(np.exp(lt - top).sum(axis=0))).max())
        got = float(br.ln_tail(np.array([W]), x_max, n_max)[0])
        assert got >= brute - 1e-9, 'N %d x_max %g W %g: bound %.6f below the tail %.6f' % (n_max, x_max, W, got, brute)
        assert got < brute + (1e-9 if x_max <= n_max + 1 else 3.0), \
            'N %d x_max %g W %g: bound %.3f, tail %.3f' % (n_max, x_max, W, got, brute)
    lt = br.ln_tail(np.array([10.0, 20.0, 40.0, 80.0, 1e9]), 30.0, 12)
    assert np.all(np.diff(lt) > 0) and lt[-1] == np.inf, 'the bound must grow with W, and be +inf where none is attempted'


def check_layer(sbar, w, n_max=12, zcut=8.0, wnet=10.0, tol=1e-6, npts=60, seed=5):
    """stage R outside the network cut, at an order (12) where the truncation shows, the series trusted up to
    wseries = wnet: the needed entries with W = w_H + w_L > wnet, before and after the repair, against lnlam_direct.
    Returns the largest error after the
    repair at the flagged and at the unflagged ones (those the bound proves accurate), the largest error of the
    series at the flagged ones, and the counts"""
    tau, keys, V = stage_w(n_max, np.array([6.0, 0.35]), np.array([-0.45, 0.3]))
    j = [6.0, 0.35].index(sbar)
    gr = br.grids(tau)
    th, ph, _, lagw = delay_weights(tau_lag(gr))
    beta, delta, phi = gr['beta'][::6], gr['delta'][::9], gr['phi'][::12]
    s_H, s_L = sbar * (1 + w), sbar * (1 - w)
    T = br.t_from_v(V[j], sbar, delta, beta, phi, n_max=n_max, out_dtype=np.float64)
    Tr, flags, counts = flag_and_repair(T, V[j], s_H, s_L, sbar, beta, delta, phi, zcut, 18.0, n_max, wnet=wnet, tol=tol,
                                        wseries=wnet)
    need, W = br.need_mask(beta, delta, s_H, s_L, zcut, wnet)
    lags = np.nonzero(lagw.sum(axis=1) > 0)[0]
    assert not flags[:, :, ~need].any() and not flags[lagw.sum(axis=1) == 0].any()
    layer = np.argwhere(need & (W > wnet))
    rng = np.random.default_rng(seed)
    err = {True: [0.0], False: [0.0]}
    series = [0.0]
    for _ in range(npts):
        (b, d), l, k = layer[rng.integers(len(layer))], rng.choice(lags), rng.integers(phi.size)
        ref = lnlam_direct(sbar, w, n_max, lagw[l], th, ph, np.exp(delta[d]), 1.0, phi[k], np.exp(beta[b]))
        f = bool(flags[l, k, b, d])
        err[f].append(abs(Tr[l, k, b, d] - ref))
        if f:
            series.append(abs(T[l, k, b, d] - ref))
    return max(err[True]), max(err[False]), max(series), len(err[True]) - 1, len(err[False]) - 1, counts


def test_repair_outside_the_network_cut():
    """the search reads entries with W above its network cut (the corners of the last cells, the neighbours of the
    bias removal); there the truncated series is kept only where the bound proves it, and replaced elsewhere"""
    for sbar, w in ((6.0, -0.45), (0.35, 0.3)):
        e_flag, e_keep, e_series, n_flag, n_keep, counts = check_layer(sbar, w)
        assert n_flag > 10, 'sbar %g: %d sampled entries flagged' % (sbar, n_flag)
        assert e_flag < 1e-5 and e_keep < 1e-5, \
            'sbar %g: after the repair %.2e (flagged), %.2e (kept by the bound)' % (sbar, e_flag, e_keep)
        assert e_series > 1e-2, 'sbar %g: the truncation must show before the repair (%.2e)' % (sbar, e_series)
        assert counts['truncation'] > 0
    assert n_keep > 5, 'at sbar 0.35 the bound must prove some entries above the cut (%d sampled)' % n_keep


def column_toy(sbar=6.0, w=-0.45, n_max=12, zcut=8.0, wnet=10.0, tol=1e-6):
    """the toy of check_layer for the column rule: the series table, the entries stage R replaces before the rule
    (must: cancellation, and beyond the network cut what the bound does not cover) and those left to the rule (test:
    up to the cut, not covered by the bound), with the grids"""
    tau, keys, V = stage_w(n_max, np.array([6.0, 0.35]), np.array([-0.45, 0.3]))
    j = [6.0, 0.35].index(sbar)
    gr = br.grids(tau)
    beta, delta, phi = gr['beta'][::6], gr['delta'][::9], gr['phi'][::12]
    s_H, s_L = sbar * (1 + w), sbar * (1 - w)
    T = br.t_from_v(V[j], sbar, delta, beta, phi, n_max=n_max, out_dtype=np.float64)
    kw = dict(n_max=n_max, nside=NSIDE, wnet=wnet, tol=tol)
    must = br.flag_entries(T, V[j], s_H, s_L, sbar, beta, delta, zcut, 18.0, wseries=wnet, **kw)[0]
    test = br.flag_entries(T, V[j], s_H, s_L, sbar, beta, delta, zcut, 18.0, wseries=0.0, **kw)[0] & ~must
    return dict(T=T, must=must, test=test, beta=beta, delta=delta, phi=phi, s_H=s_H, s_L=s_L, sbar=sbar, w=w, gr=gr,
                n_max=n_max, zcut=zcut, wnet=wnet)


def test_column_rule_below_the_network_cut(monkeypatch):
    """up to the network cut the truncation of the series is settled column by column (lag, phi, Delta), at an order
    (12) where it shows there, with ctol 1e-4.  (1) column_test: fewer entries are evaluated than the rule is asked
    about (most columns are accepted at their loudest entry), some are replaced, nothing else changes; against
    lnlam_direct the replaced entries were wrong by more than ctol / 2 and are right to 1e-6, and the entries that
    kept the series, most of them never evaluated, are right to ctol / 2.  (2) The lemma on numbers: along beta in a
    column, T_quad - T_series = -ln(1 - eps) never falls, to the rounding of the series (every entry of 60 columns
    by repair, 40 of them columns where an entry was replaced: there it grows by more than 100).  (3) --reuse:
    copying what the store without the rule had replaced and running the rule gives the table built in one go, bit
    for bit in float32; an earlier store that replaced entries the build keeps is refused.  Fires on: the columns
    taken from the quietest entry up, a column accepted whatever the quadrature says, the offset left out of the
    quadrature, a reuse that does not copy.  (4) An entry farther than BIG from its series is redone by repair: the
    same table"""
    t = column_toy()
    T, must, test, beta, delta, phi = (t[k] for k in ('T', 'must', 'test', 'beta', 'delta', 'phi'))
    kw = dict(n_max=t['n_max'], nside=NSIDE)
    assert must.sum() > 0 and test.sum() > 1000 and not (must & test).any()
    need, W = br.need_mask(beta, delta, t['s_H'], t['s_L'], t['zcut'], t['wnet'])
    assert not test[:, :, W > t['wnet']].any() and not test[:, :, ~need].any()
    rng = np.random.default_rng(8)
    th, ph, _, lagw = delay_weights(tau_lag(t['gr']))

    def ref(l, k, b, d):
        return lnlam_direct(t['sbar'], t['w'], t['n_max'], lagw[l], th, ph, np.exp(delta[d]), 1.0, phi[k], np.exp(beta[b]))
    # (1) the rule
    ctol = 1e-4
    Tm = br.repair(T.copy(), must, t['s_H'], t['s_L'], beta, delta, phi, **kw)[0]
    Tc, n_eval, n_rep = br.column_test(Tm.copy(), test, t['s_H'], t['s_L'], beta, delta, phi, ctol / 2, **kw)
    n_col = int(test.any(axis=2).sum())
    changed = Tc != Tm
    print('column rule: %d entries in %d columns, %d evaluated, %d replaced' % (int(test.sum()), n_col, n_eval, n_rep))
    assert 50 < n_rep < n_eval < 0.7 * int(test.sum()) and n_eval - n_rep > 0.9 * n_col
    assert int(changed.sum()) == n_rep and not changed[~test].any()
    rep = np.argwhere(changed)
    for l, k, b, d in rep[rng.choice(len(rep), 20, replace=False)]:
        r = ref(l, k, b, d)
        assert r - Tm[l, k, b, d] > ctol / 2 - 1e-7 and abs(Tc[l, k, b, d] - r) < 1e-6, (l, k, b, d)
    kept = np.argwhere(test & ~changed)
    for l, k, b, d in kept[rng.choice(len(kept), 40, replace=False)]:
        assert abs(Tc[l, k, b, d] - ref(l, k, b, d)) < ctol / 2 + 1e-7, (l, k, b, d)
    # (2) the share of the orders left out grows with beta in every column
    hit = np.argwhere(changed.any(axis=2))
    rest = np.argwhere(test.any(axis=2) & ~changed.any(axis=2))
    pick = np.concatenate([hit[rng.choice(len(hit), 40, replace=False)], rest[rng.choice(len(rest), 20, replace=False)]])
    clean = (need & (W <= t['wnet']))[None, None] & ~must        # every entry of a column the lemma speaks of
    f = np.zeros(T.shape, bool)
    for l, k, d in pick:
        f[l, k, :, d] = clean[l, k, :, d]
    Tq = br.repair(T.copy(), f, t['s_H'], t['s_L'], beta, delta, phi, **kw)[0]
    for n, (l, k, d) in enumerate(pick):
        b = np.nonzero(clean[l, k, :, d])[0]
        eps = (Tq - T)[l, k, b, d]
        # (the series carries the rounding of its cancellation, below 2.5e-8 for the entries kept: gap <= 18)
        assert eps.min() > -1e-7 and np.all(np.diff(eps) > -1e-7), (l, k, d, eps)
        assert n >= 40 or (b.size >= 2 and eps[-1] > 100 * max(abs(eps[0]), 1e-9)), (l, k, d, eps)
    # (3) reuse of the store built without the rule
    T2 = T.copy()
    todo = br.reuse_entries(T2, must, Tm.astype(np.float32), must.copy())
    assert not todo.any()
    T2 = br.column_test(T2, test, t['s_H'], t['s_L'], beta, delta, phi, ctol / 2, **kw)[0]
    assert np.array_equal(T2.astype(np.float32), Tc.astype(np.float32))
    with pytest.raises(ValueError, match='other settings'):
        br.reuse_entries(T.copy(), must & ~test, Tm.astype(np.float32), must | test)
    # an entry far from its series is redone by repair: the same table (the two quadratures agree in float64)
    assert ((Tc - Tm)[changed] > 1e-3).sum() > 0
    monkeypatch.setattr(br, 'BIG', 1e-3)
    T3 = br.column_test(Tm.copy(), test, t['s_H'], t['s_L'], beta, delta, phi, ctol / 2, **kw)[0]
    assert np.array_equal(T3 != Tm, changed) and np.abs(T3 - Tc).max() < 1e-9


def _real_cupy():
    mod = sys.modules.get('cupy')
    return mod is not None and hasattr(mod, 'cuda') and bool(mod.cuda.is_available())


@pytest.mark.skipif(not _real_cupy(), reason='needs a GPU')
def test_repair_on_the_gpu():
    """the fused reductions: the device i0e against scipy where the library one fails (88 in float32, 713 in float64),
    and the repaired entries against the numpy path"""
    import cupy as cp
    ker = br._repair_kernels()
    xs = np.array([0.5, 10.0, 29.9, 30.0, 50.0, 88.0, 89.0, 100.0, 300.0, 713.0, 720.0, 1640.0, 3000.0])
    got = np.array([float(ker['sum'](cp.zeros(1), cp.asarray([v]), 1.0, v).get()) for v in xs])
    rel = np.abs(got / i0e(xs) - 1).max()
    assert rel < 1e-6, 'device i0e off by %.2e (relative)' % rel
    n_max, sbar, w = 20, 2.5, -0.6
    tau, keys, V = stage_w(n_max, np.array([sbar]), np.array([w]))
    gr = br.grids(tau)
    beta, delta, phi = gr['beta'][::25], gr['delta'][::27], gr['phi'][::12]
    T = br.t_from_v(V[0], sbar, delta, beta, phi, n_max=n_max, out_dtype=np.float64)
    s_H, s_L = sbar * (1 + w), sbar * (1 - w)
    flags, counts = br.flag_entries(T, V[0], s_H, s_L, sbar, beta, delta, 72.0, -np.inf, n_max=n_max, nside=NSIDE)
    Tc, nc = br.repair(T.copy(), flags, s_H, s_L, beta, delta, phi, n_max=n_max, nside=NSIDE)
    Tg, ng = br.repair(T.copy(), flags, s_H, s_L, beta, delta, phi, n_max=n_max, nside=NSIDE, xp=cp)
    assert nc == ng == counts['flagged'] > 0
    worst = np.abs(Tg - Tc).max()
    assert worst < 1e-6, 'GPU repair differs from the numpy path by %.2e' % worst
    # the column rule: the one-pass kernel against the numpy path (a column decided the other way within the float32
    # error of the kernel is allowed for one entry in fifty; where the two decide alike the tables agree to 1e-6)
    t = column_toy()
    kw = dict(n_max=t['n_max'], nside=NSIDE)
    args = (t['test'], t['s_H'], t['s_L'], t['beta'], t['delta'], t['phi'], 5e-5)
    Tm = br.repair(t['T'].copy(), t['must'], *args[1:6], **kw)[0]
    Tc, ec, rc = br.column_test(Tm.copy(), *args, **kw)
    Tg, eg, rg = br.column_test(Tm.copy(), *args, xp=cp, **kw)
    alike = (Tc != Tm) == (Tg != Tm)
    assert rc > 50 and (~alike).sum() <= 0.02 * rc + 2 and abs(eg - ec) <= 0.02 * rc + 2, (ec, eg, rc, rg)
    worst = np.abs(Tg - Tc)[alike].max()
    assert worst < 1e-6, 'GPU column rule differs from the numpy path by %.2e' % worst
    Tb = br.column_test(Tm.copy(), *args, xp=cp, batch=5, **kw)[0]
    assert np.array_equal(Tb, Tg), 'the result of the column rule depends on the batch'
