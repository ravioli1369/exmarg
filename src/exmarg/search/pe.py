"""
parameter estimation of single segments with the stored tables, on the
CPU or the GPU (the same steps, device='cpu' or 'gpu').

pe_redshift   (m1_source, z) at equal mass and zero spin, the distance
              a parameter through y = D0 / D_L(z)
pe_intrinsic  (M_det, q, chi) on a template bank, the distance
              integrated inside the tables

both return ln Lambda, the likelihood ratio against noise marginalized
over the sky position, orientation, polarization, phase and arrival
time, and ln B, its prior average.  The sum over arrival times of
gather.py runs, by default, over the n_loud loudest times of each template
plus n_random times drawn from the rest (gather.select_times); n_loud = 0
sums over every arrival time.

with single=True both also return the single-detector evidences of
single.py (docs/glitch_likelihood.md, Sec. 5): ln Lambda_S^H, ln Lambda_S^L
on the nodes of ln Lambda, reduced the same way, and the glitch evidences.
With glitch, also the merger likelihood ratios given a glitch of Sec. 4,
lambda_H, lambda_L and lambda_HL, from the same table read (the gather sums
S, S kbar_H, S kbar_L and S kbar_H kbar_L in one pass) and one-detector sums
weighted by o_H, o_L and their products with kbar.
"""

import numpy as np

from . import gather as _gather, single as _single, store as _store
from .bank import array_module
from .segments import D0, read_segments
from .. import detectors as _det
from ..priors import distance_density

GLITCH = dict(pi_k=None, pi_m=None, window=0.1)    # the glitch settings and their defaults, see pe_redshift


def _dl_of_z(z):
    from astropy.cosmology import Planck18
    return Planck18.luminosity_distance(np.atleast_1d(z)).value


def _chunks(seg, chunk):
    n = seg["rows"].size
    for c0 in range(0, n, chunk):
        yield range(c0, min(c0 + chunk, n))


def _fit_chunk(bank, store, chunk, nodes_per_launch, resident):
    """GPU: the number of segments filtered at once, at most chunk, so that their trajectories
    (16 n_template n_time bytes each) fit next to the tables on the device"""
    if bank.device == "cpu":
        return chunk
    import cupy as cp
    free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
    sizes = [t.nbytes for t in store.tables]
    if resident and getattr(store, "device", None) is None:
        free -= sum(sizes)
    elif not resident:
        free -= nodes_per_launch * max(sizes)
    per_segment = 16 * bank.u.size * bank.n_time
    return max(1, min(chunk, int(0.6 * free / per_segment)))


def _filter(bank, seg, e):
    return bank.trajectories({I: seg["d_" + I][e] for I in "HL"},
                             {I: seg["sig2_" + I] for I in "HL"})


def _node_geometry(bank, st, ymin):
    """which (node, template) pairs the redshift store covers, and their ln a and y.

    a = s/u is valid for y = sqrt(a) >= ymin and alpha_lo' < a < alpha_hi,
    alpha_lo' = min(alpha_lo, ymin^2).  a template whose a crosses
    alpha_lo' between two nodes also gets a boundary point at
    a = alpha_lo' (1 + 1e-9), read from the upper node's table."""
    nn, nm = st.s.size, bank.u.size
    valid = np.zeros((nn, nm), bool)
    lna = np.zeros((nn, nm))
    y = np.zeros((nn + 1, nm))                 # row nn: the boundary point
    bnode = np.full(nm, -1)
    blna = np.zeros(nm)
    for k in range(nn):
        alo = min(st.alpha_lo[k], ymin * ymin)
        a = st.s[k] / bank.u
        o = (np.sqrt(a) >= ymin) & (a > alo) & (a < st.alpha_hi[k])
        valid[k] = o
        lna[k][o] = np.log(a[o])
        y[k] = np.where(o, np.sqrt(a), 0.0)
        s_prev = st.s[k - 1] if k > 0 else 0.0
        low = o & (s_prev / bank.u <= alo)
        a_b = float(alo) * (1.0 + 1e-9)
        bnode[low] = k
        blna[low] = np.log(a_b)
        y[nn][low] = np.sqrt(a_b)
    return valid, lna, y, bnode, blna


def _to_source_frame(lamN, yN, Mdet, yq, m1, z, cp):
    """node values (n_node + 1, n_template) -> ln Lambda on (m1, z) (cp: numpy or cupy):
    linear in y per template, then linear in M_det = 2 m1 (1 + z) per z, each
    between the neighbouring finite values (as np.interp on the finite ones)"""
    L = cp.asarray(lamN)
    Y = cp.asarray(yN)
    ok = cp.isfinite(L) & (Y > 0)
    order = cp.argsort(cp.where(ok, Y, cp.inf), axis=0)
    Ys = cp.take_along_axis(cp.where(ok, Y, cp.inf), order, axis=0)
    Ls = cp.take_along_axis(L, order, axis=0)
    n = ok.sum(axis=0)                                            # finite nodes per template
    yq_ = cp.asarray(yq)
    j = (Ys[:, None, :] <= yq_[None, :, None]).sum(axis=0) - 1    # (nz, nm): last node <= yq
    lo = cp.clip(j, 0, cp.maximum(n - 2, 0)[None, :])
    hi = cp.minimum(lo + 1, cp.maximum(n - 1, 0)[None, :])
    cols = cp.arange(Ys.shape[1])[None, :]
    y0, y1 = Ys[lo, cols], Ys[hi, cols]
    v0, v1 = Ls[lo, cols], Ls[hi, cols]
    t = cp.where(y1 > y0, (yq_[:, None] - y0) / cp.where(y1 > y0, y1 - y0, 1.0), 1.0)
    ylast = Ys[cp.maximum(n - 1, 0), cp.arange(Ys.shape[1])][None, :]
    inside = (n[None, :] >= 2) & (yq_[:, None] >= Ys[0][None, :]) & (yq_[:, None] <= ylast)
    surf = cp.where(inside, v0 + t * (v1 - v0), -cp.inf)        # (nz, nm)

    M = cp.asarray(Mdet)
    good = cp.isfinite(surf)                                      # (nz, nm)
    rank = cp.cumsum(good, axis=1)                                # good masses up to and including each index
    ngood = rank[:, -1]
    G = cp.zeros(surf.shape, cp.int64)                            # G[iz, r] = index of the r-th good mass
    iz, im = cp.nonzero(good)
    G[iz, rank[iz, im] - 1] = im
    Md = 2.0 * cp.asarray(m1)[:, None] * (1.0 + cp.asarray(z))[None, :]   # (nm1, nz)
    p = cp.searchsorted(M, Md.ravel()).reshape(Md.shape)          # masses with M_det < Md
    zi = cp.arange(z.size)[None, :]
    below = cp.where(p > 0, rank[zi, cp.maximum(p - 1, 0)], 0)    # good masses with M_det < Md
    lo = G[zi, cp.clip(below - 1, 0, None)]
    hi = G[zi, cp.minimum(below, cp.maximum(ngood - 1, 0)[None, :])]
    lo = cp.where(below >= 1, lo, hi)
    m0, m1_ = M[lo], M[hi]
    w0, w1 = surf[zi, lo], surf[zi, hi]
    t = cp.where(m1_ > m0, (Md - m0) / cp.where(m1_ > m0, m1_ - m0, 1.0), 0.0)
    first = M[G[:, 0]][None, :]
    last = M[G[zi[0], cp.maximum(ngood - 1, 0)]][None, :]
    inb = (ngood[None, :] >= 2) & (Md >= first) & (Md <= last)
    with np.errstate(invalid="ignore"):                 # -inf - (-inf) outside the bank; those points are masked by inb
        out = cp.where(inb, w0 + t * (w1 - w0), -cp.inf)
    return out if cp is np else cp.asnumpy(out)


def _log_mean(lam, w, cp=np):
    """ln sum_i w_i exp(lam_i) for normalized weights w, lam = -inf allowed"""
    lam, w = cp.asarray(lam), cp.asarray(w)
    fin = cp.isfinite(lam) & (w > 0)
    if not bool(fin.any()):
        return -np.inf
    mx = cp.where(fin, lam, -cp.inf).max()
    return float(mx + cp.log(cp.sum(cp.where(fin, w * cp.exp(cp.where(fin, lam, mx) - mx), 0.0))))



def _tables(store, ks, beta, delta, bank, resident):
    """the tables ks for a gather: host list (CPU) or device stack (GPU), and the offset of ks[0]"""
    if bank.device == "cpu":
        return dict(T=[store.tables[k] for k in ks], tau=store.tau_idx, beta=beta, delta=delta), ks[0]
    if resident:
        if getattr(store, "device", None) is None:
            store.device = _store.upload(store.tables, store.tau_idx, beta, delta)
        return dict(store.device, beta=beta, delta=delta), 0
    return _store.upload([store.tables[k] for k in ks], store.tau_idx, beta, delta), ks[0]


def _read(bank, traj, tab, ids, lna, tab_ids, block, times, time_weights=None):
    if bank.device == "cpu":
        return _gather.gather_cpu(traj, tab, ids, lna, tab_ids, times, time_weights)
    return _gather.gather_gpu(traj, tab, ids, lna, tab_ids, block, times, time_weights)


def _release(bank):
    if bank.device == "gpu":
        _store.free()


def _single_setup(bank, seg, store, geometry):
    """what single=True needs once per call: the columns of single.py (amplitude bins, then the store nodes s) and
    the signal power s = a u of the boundary point of every template"""
    if not np.array_equal(seg["sig2_H"], seg["sig2_L"]):
        raise ValueError("single=True needs sig2_H == sig2_L, as the tables (u_H = u_L, "
                         "docs/glitch_likelihood.md Sec. 6)")
    bnode, blna = geometry[3], geometry[4]
    lnT = np.concatenate([_single.glitch_columns(), _single.merger_columns(store.s)], axis=1)
    return dict(lnT=array_module(bank.device).asarray(lnT), ln_u=np.log(bank.u),
                s_b=np.where(bnode >= 0, np.exp(blna) * bank.u, 0.0), n_rho=_single.RHO_EDGES.size - 1)


def _glitch_setup(glitch, bank, seg, store):
    """the glitch settings (True: the defaults of GLITCH) with pi_k (n_rho,), the half window (samples) and the lags
    of the delay average: the lags of the tables within the light travel time between the sites"""
    g = dict(GLITCH, **({} if glitch is True else glitch))
    if set(g) != set(GLITCH):
        raise ValueError("unknown glitch settings %s" % sorted(set(g) - set(GLITCH)))
    dt = seg["T"] / bank.n_time
    lags = np.asarray(store.tau_idx, int).ravel()
    lags = lags[np.abs(lags) * dt <= np.linalg.norm(_det.x["H"] - _det.x["L"]) / _det.c]
    pi_k = _single.amplitude_weights() if g["pi_k"] is None else np.asarray(g["pi_k"], float)
    return dict(pi_k=pi_k, pi_m=g["pi_m"], window=float(g["window"]), half_window=int(round(g["window"] / dt)),
                lags=lags)


def _kbar(traj, I, gd, sd, xp):
    """ln kbar_I and ln(1 - kbar_I) (n_time,) of one segment with the fiducial glitch law of gd, on the device"""
    lg = _single.glitch_time_series(traj["lq" + I], sd["ln_u"], gd["pi_k"], gd["pi_m"], xp)
    return _single.glitch_time_weights(lg, gd["half_window"], xp)


def _glitch_weights(traj, gd, sd, xp):
    """per segment, on the device and in logs (a loud glitch takes kbar and o below 1e-308 where l_g or Phi is
    above 1e308): the time weights (ln kbar_H, ln kbar_L) of the gather, the weights per time of the one-detector
    sums on each trajectory (rows ln o, ln o + ln kbar; o_H(t) on L, o_L(t') on H) and
    ln c0 = ln (1/n_t) sum_t o_H(t) (1 - kbar_L(t))"""
    (kH, nH), (kL, nL) = _kbar(traj, "H", gd, sd, xp), _kbar(traj, "L", gd, sd, xp)
    oH = _single.ln_near_fraction(nH, gd["lags"], xp)               # of the L arrival time t, kbar_H at t + tau
    oL = _single.ln_near_fraction(nL, -gd["lags"], xp)              # of the H arrival time t', kbar_L at t' - tau
    x = oH + nL
    m = float(x.max())
    lc0 = -np.inf if m == -np.inf else m + float(xp.log(xp.exp(x - m).sum())) - np.log(x.size)
    return dict(tw=(kH, kL), H=xp.stack([oL, oL + kH]), L=xp.stack([oH, oH + kL]), lc0=lc0)


def _single_values(traj, sd, xp, gw=None):
    """per detector: ln E (n_w, n_template, n_rho + n_node), ln Lambda_S^I at the boundary point (n_w, n_template)
    and wmax, for no weights (n_w = 1) or, with the glitch weights gw, the log weight rows 0, ln o, ln o + ln kbar
    (n_w = 3; the first is the plain sum)"""
    out = {}
    for I in "HL":
        lq = traj["lq" + I]
        W = None if gw is None else xp.concatenate([xp.zeros((1, lq.shape[1])), gw[I]])
        lnE = _single._host(_single.ln_evidence(lq, sd["ln_u"], sd["lnT"], W, xp, log_weights=True))
        b = _single._host(_single.ln_merger_at(lq, sd["ln_u"], sd["s_b"], W, xp, log_weights=True))
        out[I] = (lnE.reshape(-1, *lnE.shape[-2:]), b.reshape(-1, b.shape[-1]),
                  _single._host(_single._wmax(lq, sd["ln_u"], xp)))
    return out


def _on_nodes(lnE, b, geometry):
    """one detector's values on the node geometry: lnE (n_template, n_node) at the nodes, b (n_template,) at the
    boundary point -> (n_node + 1, n_template), -inf off the valid pairs"""
    valid, bnode = geometry[0], geometry[3]
    return np.concatenate([np.where(valid, lnE.T, -np.inf), np.where(bnode >= 0, b, -np.inf)[None]])


def _single_nodes(one, lamN, geometry, n_rho, far=None, lc0=None):
    """the single-detector values on the node geometry of lamN (same valid pairs, same boundary row, -inf where
    lamN is), the glitch evidences and the largest w, stacked over the segments; with far (n_seg, 3, n_node + 1,
    n_template), the gather sums weighted by kbar_H, kbar_L and both, and ln c0 (n_seg,) also lamN_gH, lamN_gL,
    lamN_gHL"""
    nodes = lambda I, i: np.stack([_on_nodes(o[I][0][i][:, n_rho:], o[I][1][i], geometry) for o in one])
    out = {}
    for I in "HL":
        out["lamN_" + I] = nodes(I, 0)
        out["ln_glitch_" + I] = np.stack([o[I][0][0][:, :n_rho] for o in one])
        out["wmax_" + I] = np.stack([o[I][2] for o in one])
    if far is not None:
        lc0 = np.asarray(lc0, float)[:, None, None]
        out["lamN_gH"] = np.logaddexp(far[:, 0], nodes("L", 1))
        out["lamN_gL"] = np.logaddexp(far[:, 1], nodes("H", 1))
        out["lamN_gHL"] = np.logaddexp(np.logaddexp(far[:, 2], nodes("L", 2)), np.logaddexp(nodes("H", 2), lc0))
    for k in out:
        if k.startswith("lamN_"):
            out[k][~np.isfinite(lamN)] = -np.inf
    return out


def _node_values(bank, seg, ev, store, geometry, groups, block, resident, n_loud, n_random, seed, single=None,
                 glitch=None):
    """ln Lambda of every template at every node of the redshift store, for the segments ev:
    (len(ev), n_node + 1, n_template), row n_node the boundary point at the smallest amplitude.
    with single (_single_setup) also the dict of _single_nodes, with glitch (_glitch_setup) its glitch terms"""
    valid, lna, _, bnode, blna = geometry
    nn = store.s.size
    traj = [_filter(bank, seg, e) for e in ev]
    xp = array_module(bank.device)
    gw = [None] * len(ev) if glitch is None else [_glitch_weights(t, glitch, single, xp) for t in traj]
    one = None if single is None else [_single_values(t, single, xp, g) for t, g in zip(traj, gw)]
    tw = [None if g is None else g["tw"] for g in gw]
    times = [_gather.select_times(t, store.tau_idx, n_loud, n_random, seed + int(seg["rows"][e]), xp) if n_loud > 0 else None
             for t, e in zip(traj, ev)]
    lamN = np.full((len(ev), nn + 1, bank.M.size), -np.inf)
    far = None if glitch is None else np.full((len(ev), 3, nn + 1, bank.M.size), -np.inf)
    for ks in groups:
        k0 = ks[0]
        tab, base = _tables(store, ks, store.beta[k0], store.delta[k0], bank, resident)
        ok = valid[ks]
        ids = np.nonzero(ok.any(axis=0))[0]
        lnaM = np.where(ok[:, ids], lna[ks][:, ids], 0.0).T
        tid = np.tile(np.arange(len(ks)) + (k0 - base), (ids.size, 1))
        bids = np.nonzero((bnode >= k0) & (bnode <= ks[-1]))[0]
        for j in range(len(ev)):
            if ids.size:
                val = _read(bank, traj[j], tab, ids, lnaM, tid, block, times[j], tw[j])
                val, fv = (val, None) if tw[j] is None else (val[0], val[1:])
                for i, k in enumerate(ks):
                    keep = ok[i][ids]
                    lamN[j, k, ids[keep]] = val[i][keep]
                    if fv is not None:
                        far[j, :, k][:, ids[keep]] = fv[:, i][:, keep]
            if bids.size:
                val = _read(bank, traj[j], tab, bids, blna[bids][:, None], (bnode[bids] - base)[:, None], block,
                            times[j], tw[j])
                val, fv = (val, None) if tw[j] is None else (val[0], val[1:])
                lamN[j, nn, bids] = val[0]
                if fv is not None:
                    far[j, :, nn][:, bids] = fv[:, 0]
        del tab
        if not resident:
            _release(bank)
    del traj, times, tw
    _release(bank)
    if single is None:
        return lamN
    lc0 = None if glitch is None else [g["lc0"] for g in gw]
    return lamN, _single_nodes(one, lamN, geometry, single["n_rho"], far, lc0)


def _single_out(single, n_seg, shape, bank, gd=None):
    """the arrays single=True adds to the result: ln_lambda_H, ln_lambda_L of the given shape per segment, and
    ln_glitch_H, ln_glitch_L, wmax_H, wmax_L, rho_edges; with the glitch settings gd also ln_lambda_gH,
    ln_lambda_gL, ln_lambda_gHL and glitch_pi_k, glitch_window"""
    if not single:
        return {}
    nm, nr = bank.u.size, _single.RHO_EDGES.size - 1
    out = {"rho_edges": _single.RHO_EDGES.copy()}
    for I in "HL":
        out["ln_lambda_" + I] = np.full((n_seg,) + shape, -np.inf)
        out["ln_glitch_" + I] = np.full((n_seg, nm, nr), -np.inf)
        out["wmax_" + I] = np.zeros((n_seg, nm))
    if gd is not None:
        for X in ("H", "L", "HL"):
            out["ln_lambda_g" + X] = np.full((n_seg,) + shape, -np.inf)
        out["glitch_pi_k"], out["glitch_window"] = gd["pi_k"], gd["window"]
    return out


def _node_keys(gd):
    """the node arrays reduced into the result: single detector, and with the glitch terms"""
    glitch = [] if gd is None else [("ln_lambda_g" + X, "lamN_g" + X) for X in ("H", "L", "HL")]
    return [("ln_lambda_" + I, "lamN_" + I) for I in "HL"] + glitch


def pe_redshift(segments, bank, store, rows=None, chunk=8, zmax=8.0, nz=1600,
                m1_range=(3.0, 150.0), nm1=1600, ymin=0.1267, nodes_per_launch=6, block=256,
                mdet_range=(10.0, 500.0), resident=False, n_loud=1024, n_random=1024, seed=0, single=False,
                glitch=None):
    """ln Lambda(m1_source, z) of each segment.

    segments          anything read_segments takes (see there)
    bank              mass_bank of a one-column bank of detector-frame total
                      masses, ascending, built on the same segments
    store             load_redshift_store
    rows              segment indices (default all)
    chunk             segments filtered at once (16 n_template n_time bytes each; on the GPU
                      at most what fits next to the tables)
    zmax, nz          z grid linspace(0.02, zmax, nz)
    m1_range, nm1     m1 grid geomspace(*m1_range, nm1)
    ymin              smallest y = D0 / D_L the store represents
    nodes_per_launch  store nodes read in one gather
    block             GPU threads per block
    mdet_range        support of the ln B prior
    resident          GPU: keep every table on the device across calls
    n_loud, n_random  arrival times visited per template (gather.select_times);
                      n_loud = 0 visits all of them
    seed              seed of the random draw (plus the segment index)
    single            also the single-detector evidences (needs sig2_H == sig2_L)
    glitch            None, or True or a dict of settings: also the merger likelihood ratios
                      given a glitch (implies single), with the glitch-time weights kbar of the
                      fiducial law pi_k (n_rho,) on rho_edges (default dN/drho ~ rho^-4 above
                      rho = 4, single.amplitude_weights) and pi_m (n_template,) over the
                      templates (default uniform), and the window (s, default 0.1) of the
                      arrival times near a glitch; the delay average runs over the lags of the
                      tables within the light travel time between the sites

    ln B averages Lambda over the prior uniform in M_det = 2 m1 (1+z) on
    mdet_range and proportional to D_L^2 in luminosity distance, i.e.
    p(m1, z) ~ (1+z) D_L^2 dD_L/dz, on the (m1, z) cells.

    returns dict: m1 (nm1,), z (nz,), ln_lambda (n_segment, nm1, nz),
    ln_b (n_segment,), rows; with single also ln_lambda_H, ln_lambda_L
    (n_segment, nm1, nz) ln Lambda_S^I through the nodes and the reduction of
    ln_lambda, ln_glitch_H, ln_glitch_L (n_segment, n_template, n_rho) ln E_I
    of the amplitude bins rho_edges (n_rho + 1,), wmax_H, wmax_L
    (n_segment, n_template) the largest w of each template; with glitch also
    ln_lambda_gH, ln_lambda_gL, ln_lambda_gHL (n_segment, nm1, nz) ln lambda_H,
    ln lambda_L, ln lambda_HL through the nodes and the reduction of ln_lambda,
    and the settings used, glitch_pi_k and glitch_window."""
    if np.any(np.diff(bank.M) <= 0) or np.any(bank.q != 1.0) or np.any(bank.chi != 0.0):
        raise ValueError("pe_redshift needs an ascending equal-mass zero-spin bank")
    xp = array_module(bank.device)
    seg = read_segments(segments, rows)
    z = np.linspace(0.02, zmax, int(nz))
    m1 = np.geomspace(m1_range[0], m1_range[1], int(nm1))
    yq = D0 / _dl_of_z(z)
    geometry = _node_geometry(bank, store, ymin)
    nn = store.s.size
    groups = [list(range(g, min(g + nodes_per_launch, nn))) for g in range(0, nn, nodes_per_launch)]

    out = np.full((seg["rows"].size, m1.size, z.size), -np.inf)
    gd = None if glitch is None or glitch is False else _glitch_setup(glitch, bank, seg, store)
    single = single or gd is not None
    sd = _single_setup(bank, seg, store, geometry) if single else None
    extra = _single_out(single, seg["rows"].size, (m1.size, z.size), bank, gd)
    chunk = _fit_chunk(bank, store, chunk, nodes_per_launch, resident)
    for ev in _chunks(seg, chunk):
        lamN = _node_values(bank, seg, ev, store, geometry, groups, block, resident, n_loud, n_random, seed, sd, gd)
        if single:
            lamN, one = lamN
        for j, e in enumerate(ev):
            out[e] = _to_source_frame(lamN[j], geometry[2], bank.M, yq, m1, z, xp)
            for key, node in _node_keys(gd) if single else ():
                extra[key][e] = _to_source_frame(one[node][j], geometry[2], bank.M, yq, m1, z, xp)
            for I in "HL" if single else "":
                extra["ln_glitch_" + I][e] = one["ln_glitch_" + I][j]
                extra["wmax_" + I][e] = one["wmax_" + I][j]

    dl = _dl_of_z(z)
    lnp = np.log(np.maximum((1.0 + z) * dl ** 2 * np.gradient(dl, z), 1e-300))
    Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
    inside = (Md >= mdet_range[0]) & (Md <= mdet_range[1])
    w = np.where(inside, np.exp(lnp)[None, :], 0.0) * np.gradient(m1)[:, None] * np.gradient(z)[None, :]
    w /= w.sum()
    ln_b = np.array([_log_mean(L, w) for L in out])
    return dict(m1=m1, z=z, ln_lambda=out, ln_b=ln_b, rows=seg["rows"], **extra)


def distance_weights(y, prior, ymin):
    """ln of the weights that integrate the distance prior against a likelihood ratio linear in x = ln y between
    the points y (ascending): sum_j w_j Lambda(y_j) = integral dy p(y) Lambda(y) over [y_0, y_last].

    prior  'volume': p(y) = 3 ymin^3 y^-4 (uniform in volume), integrated exactly on every interval;
           'madau-dickinson': the density in ln y of exmarg.priors, integrated by the trapezoid rule
           on 401 points inside every interval"""
    density = distance_density(prior)
    x = np.log(y)
    w = np.zeros_like(x)
    for i in range(x.size - 1):
        a, b = x[i], x[i + 1]
        h = b - a
        if density is None:
            c, k = 3.0 * ymin ** 3, 3.0
            ea, eb = np.exp(-k * a), np.exp(-k * b)
            I0 = (ea - eb) / k
            I1 = (a * ea - b * eb) / k + (ea - eb) / k ** 2
            w[i] += c * (b * I0 - I1) / h
            w[i + 1] += c * (I1 - a * I0) / h
        else:
            t = np.linspace(a, b, 401)
            f = np.interp(t, density[0], density[1], left=0.0, right=0.0)
            w[i] += np.trapezoid(f * (b - t) / h, t)
            w[i + 1] += np.trapezoid(f * (t - a) / h, t)
    with np.errstate(divide="ignore"):
        return np.log(w)


def _distance_integral(lnW, lam):
    """ln sum_nodes exp(lnW + lam) per template: (n_template,)"""
    v = lnW + lam
    mx = v.max(axis=0)
    return mx + np.log(np.exp(v - mx[None, :]).sum(axis=0))


def pe_intrinsic(segments, bank, store, rows=None, prior="volume", ymin=0.1267, chunk=4,
                 nodes_per_launch=6, block=256, resident=False, n_loud=1024, n_random=1024, seed=0, single=False,
                 glitch=None):
    """ln Lambda of each template of the bank, the distance integrated over its prior.

    segments          anything read_segments takes (see there)
    bank              mass_bank built on the same segments, rows (M_det, q, chi)
    store             load_redshift_store
    rows              segment indices (default all)
    prior             'volume' (uniform in volume) or 'madau-dickinson' (the distance prior of the
                      simulated year, exmarg.priors)
    ymin              smallest y = D0 / D_L of the prior
    chunk             segments filtered at once (16 n_template n_time bytes each; on the GPU
                      at most what fits next to the tables)
    nodes_per_launch  store nodes read in one gather
    block             GPU threads per block
    resident          GPU: keep every table on the device across calls
    n_loud, n_random  arrival times visited per template (gather.select_times);
                      n_loud = 0 visits all of them
    seed              seed of the random draw (plus the segment index)
    single            also the single-detector evidences (needs sig2_H == sig2_L)
    glitch            None, or True or a dict of settings: also the merger likelihood ratios
                      given a glitch (see pe_redshift)

    each template reads the store at every node s it reaches, y = sqrt(s/u), and
    Lambda = sum_j w_j Lambda(y_j) with the weights of distance_weights.  ln B
    is the average of Lambda over the bank, each template weighted by its cell
    (the product of the spacings of the distinct values of every axis that varies).

    returns dict: M, q, chi, u (n_template,), ln_lambda (n_segment, n_template),
    ln_b (n_segment,), rows; with single also ln_lambda_H, ln_lambda_L
    (n_segment, n_template) ln Lambda_S^I with the same distance weights, and
    ln_glitch_H, ln_glitch_L, rho_edges, wmax_H, wmax_L as for pe_redshift; with
    glitch also ln_lambda_gH, ln_lambda_gL, ln_lambda_gHL (n_segment, n_template)
    with the same distance weights, glitch_pi_k and glitch_window."""
    seg = read_segments(segments, rows)
    geometry = _node_geometry(bank, store, ymin)
    yN = geometry[2]
    nn = store.s.size
    groups = [list(range(g, min(g + nodes_per_launch, nn))) for g in range(0, nn, nodes_per_launch)]
    lnW = np.full(yN.shape, -np.inf)
    for m in range(bank.M.size):
        pts = np.nonzero(yN[:, m] > 0)[0]
        if pts.size < 2:
            raise ValueError("template %d (M %.2f, u %.3f) reaches fewer than two store nodes" % (m, bank.M[m], bank.u[m]))
        order = pts[np.argsort(yN[pts, m])]
        lnW[order, m] = distance_weights(yN[order, m], prior, ymin)

    out = np.full((seg["rows"].size, bank.u.size), -np.inf)
    gd = None if glitch is None or glitch is False else _glitch_setup(glitch, bank, seg, store)
    single = single or gd is not None
    sd = _single_setup(bank, seg, store, geometry) if single else None
    extra = _single_out(single, seg["rows"].size, (bank.u.size,), bank, gd)
    chunk = _fit_chunk(bank, store, chunk, nodes_per_launch, resident)
    for ev in _chunks(seg, chunk):
        lamN = _node_values(bank, seg, ev, store, geometry, groups, block, resident, n_loud, n_random, seed, sd, gd)
        if single:
            lamN, one = lamN
        for j, e in enumerate(ev):
            out[e] = _distance_integral(lnW, lamN[j])
            for key, node in _node_keys(gd) if single else ():
                extra[key][e] = _distance_integral(lnW, one[node][j])
            for I in "HL" if single else "":
                extra["ln_glitch_" + I][e] = one["ln_glitch_" + I][j]
                extra["wmax_" + I][e] = one["wmax_" + I][j]

    cell = np.ones(bank.u.size)
    for ax in (bank.M, bank.q, bank.chi):
        vals = np.unique(ax)
        if vals.size > 1:
            cell *= np.gradient(vals)[np.searchsorted(vals, ax)]
    w = cell / cell.sum()
    ln_b = np.array([_log_mean(L, w) for L in out])
    return dict(M=bank.M, q=bank.q, chi=bank.chi, u=bank.u, ln_lambda=out, ln_b=ln_b,
                rows=seg["rows"], **extra)
