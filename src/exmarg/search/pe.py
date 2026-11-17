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

With glitch=dict(model='windows', ...) the glitch-time weights are those of
the model with many glitches per segment (Sec. 4a): the window evidences of
each detector (single.window_evidence), the posterior of its state and the
weights kbar of each window (single.window_posterior), kbar of a window
given to each of its arrival times; the rest is the same computation, and
the lambda_HL expression with these weights is the segment's B.
window_evidences gives the window evidences alone (stage 1 of Sec. 4a).

Unequal noise curves (docs/2026-11-15_unequal_noise_design.md, Secs. 2, 3).  A template has u_H, u_L,
ubar = (u_H + u_L) / 2 and w = (u_H - u_L) / (u_H + u_L) (bank.MassBank); a store has an asymmetry w_j and its
node s is sbar (s_H = sbar (1 + w_j), s_L = sbar (1 - w_j)).  Store j is read at a = sbar / ubar with the data of
the template scaled to the store's model, ln|z_I''| = ln|z_I| + c_I, c_H = ln((1 + w_j) / (1 + w)) / 2,
c_L = ln((1 - w_j) / (1 - w)) / 2, so that E_0|z_I''|^2 = ubar (1 +- w_j); its single-detector merger columns are
those at s_H and s_L, read with the data w_I = |z_I|^2 / u_I.  Each X_j is then the likelihood ratio of a model
with the same total power, E_0 X_j = 1.  store may be one store, which every template's w must match within
ASYM_TOL, or a sequence of stores with distinct asym and the same nodes, grids and lags, whose range must hold
every w: the result of a template is (1 - f) X_j + f X_{j+1} of the two stores that bracket its w,
f = (w - w_j) / (w_{j+1} - w_j), taken on the node values (the same nodes sbar and y = sqrt(sbar / ubar) in every
store) of every quantity linear in the merger likelihood: Lambda, Lambda_S^I and the lambda terms (B).  What
depends on the data alone (glitch and window evidences, the posteriors and weights kbar, ln c, wmax, the arrival
times visited) is computed once per segment from the unscaled data with the true u_I.  Per store only the
(node, template) pairs of non-zero weight are read, and a store of zero weight for every template not at all.  With
asym_tol > 0 a template whose w lies outside the stores' range by at most asym_tol is read from the end store alone
(its data scaled to that store's model, so still a likelihood ratio), and the result counts them (asym_outside).

Signal gain (bank.gain, strain.signal_gain).  The data of detector I carry a signal with amplitude kappa_I times the
true strain (the conditioning with estimated spectra).  The model's template of detector I is then kappa_I a: its
powers kappa_I^2 u_I give ubar and w (bank.u, bank.asym, so the node geometry and the single-detector merger powers are
those of the model), and the scale of the data to store j becomes c_I + ln kappa_I, so that E_0|z_I''|^2 = ubar (1 +-
w_j) still (each X_j a likelihood ratio, E_0 X_j = 1, whatever the gains).  The single-detector statistics of the data,
w_I = |z_I|^2 / u_I, and everything computed from them alone (glitch and window evidences, kbar, wmax) do not depend on
the gain and are bit for bit those of gains 1.  With gains 1 every output is bit for bit that of the code without them.

pe_redshift(zmin=...) reads only the nodes that the redshifts z >= zmin need: a template skips the nodes with y
above D0 / D_L(zmin) but the first one (the interpolation in y needs it), and ln Lambda below zmin is -inf.

pe_redshift_epochs analyses several epochs (sets of segments with their own noise curves, so their own banks) in one
pass over the tables: the segments of all the epochs are filtered in chunks, and each launch of tables (the nodes of a
group that any epoch of the chunk needs) is copied to the device once and read for every segment of the chunk with its
own epoch's arguments (node geometry, stores and weights, single-detector columns, glitch settings, seed).  Per segment
the arithmetic is that of pe_redshift on its epoch alone; only the order of the loops and the number of table copies
change, and the results are the same bit for bit.  pe_redshift is the call of one epoch; pe_intrinsic stays per call.
"""

import numpy as np

from . import gather as _gather, single as _single, store as _store
from .bank import array_module
from .segments import D0, read_segments
from .. import detectors as _det
from ..priors import distance_density

GLITCH = dict(pi_k=None, pi_m=None, window=0.1)    # the glitch settings and their defaults, see pe_redshift
WINDOWS = dict(model="windows", n_window=64, zone=1, pi_m=None, H=None, L=None)   # those of model='windows'
STATE = ("f", "q", "pi_k")                          # the settings of one detector in model='windows'
SUM_TOL = 1e-6                                      # how far from 1 the sum of a law may be
ASYM_TOL = 1e-9                                     # how far a template's w may be from the asym of the one store
# The single-detector merger term at the boundary point of each template (its own signal power): from a table of Phi
# made once per bank (single.MergerTable, True) or computed at every segment (single.ln_merger_at, False).  The two
# use the same interpolation and agree to rounding (tests/test_single.py, tests/test_pe_batch.py); the table saves
# 0.07 s per segment of the full bank on a GPU.  None: the table on the GPU, and on the CPU the computation per
# segment, whose results are bit for bit those of before the table
BOUNDARY_TABLES = None


def _dl_of_z(z):
    from astropy.cosmology import Planck18
    return Planck18.luminosity_distance(np.atleast_1d(z)).value


def _chunks(seg, chunk):
    n = seg["rows"].size
    for c0 in range(0, n, chunk):
        yield range(c0, min(c0 + chunk, n))


def _epoch_chunks(sizes, chunk):
    """the segments of several epochs (sizes: the number of each), in order, cut into chunks of at most chunk: per
    chunk [(epoch index, range of its segments)], one epoch's segments split between chunks where a chunk is full"""
    part, room = [], chunk
    for i, n in enumerate(sizes):
        e = 0
        while e < n:
            take = min(room, n - e)
            part.append((i, range(e, e + take)))
            e, room = e + take, room - take
            if room == 0:
                yield part
                part, room = [], chunk
    if part:
        yield part


def _store_list(store):
    """store, or a sequence of stores, as a list sorted by asym; several must have distinct asym and the same nodes,
    amplitude limits and grids (to 1e-12: the published store's own Delta grids differ between nodes by 4e-16), lags,
    truncation order and table shape (ValueError otherwise).  Each store is read on its own grids; the geometry is that
    of the first"""
    stores = list(store) if isinstance(store, (list, tuple)) else [store]
    if not stores:
        raise ValueError("no store given")
    stores.sort(key=lambda st: float(getattr(st, "asym", 0.0)))
    asym = [float(getattr(st, "asym", 0.0)) for st in stores]
    if len(set(asym)) < len(asym):
        raise ValueError("the stores must have distinct asym, not %s" % asym)
    a = stores[0]
    close = lambda x, y: np.shape(x) == np.shape(y) and np.allclose(x, y, rtol=1e-12, atol=1e-12)
    for st in stores[1:]:
        same = dict(s=close(st.s, a.s), alpha=close(st.alpha_lo, a.alpha_lo) and close(st.alpha_hi, a.alpha_hi),
                    lags=np.array_equal(st.tau_idx, a.tau_idx),
                    beta=len(st.beta) == len(a.beta) and all(close(x, y) for x, y in zip(st.beta, a.beta)),
                    delta=len(st.delta) == len(a.delta) and all(close(x, y) for x, y in zip(st.delta, a.delta)),
                    n_max=st.n_max == a.n_max, shape=np.shape(st.tables[0]) == np.shape(a.tables[0]))
        bad = [k for k, v in same.items() if not v]
        if bad:
            raise ValueError("the stores of asym %g and %g differ in %s: a mixture needs the same nodes, grids and lags"
                             % (a.asym, st.asym, ", ".join(bad)))
    return stores


def _store_weights(stores, bank, asym_tol=0.0):
    """(n_store, n_template) the weight of each store (sorted by asym) in the result of each template: one store, 1 if
    every template's w is its asym within ASYM_TOL + asym_tol; several, 1 - f and f on the two that bracket w,
    f = (w - w_j) / (w_{j+1} - w_j), 1 on w_j alone for a w within ASYM_TOL of it (a weight of 1e-9 is not worth
    reading a store for), and 1 on the end store for a w outside the range by at most ASYM_TOL + asym_tol (its data
    scaled to that store's model like any other: a likelihood ratio, with the model error of the nearest table).
    ValueError for a w farther outside (no extrapolation)"""
    grid = np.array([float(getattr(st, "asym", 0.0)) for st in stores])
    w = np.asarray(bank.asym, float)
    tol = ASYM_TOL + float(asym_tol)
    if np.any(w < grid[0] - tol) or np.any(w > grid[-1] + tol):
        ratio = lambda x: (1.0 + x) / (1.0 - x)
        cover = ("u_H/u_L = %.4g only (w = %g)" % (ratio(grid[0]), grid[0]) if grid.size == 1 else
                 "u_H/u_L from %.4g to %.4g (w from %g to %g, %d stores)" % (ratio(grid[0]), ratio(grid[-1]), grid[0],
                                                                             grid[-1], grid.size))
        raise ValueError("the templates have u_H/u_L from %.4g to %.4g (w = (u_H - u_L)/(u_H + u_L) from %.4g to %.4g); "
                         "the tables cover %s, asym_tol %g: give stores whose asym covers every template" % (
                             ratio(w.min()), ratio(w.max()), w.min(), w.max(), cover, asym_tol))
    if grid.size == 1:
        return np.ones((1, w.size))
    near = np.abs(w[:, None] - grid[None, :]) <= ASYM_TOL
    wc = np.clip(np.where(near.any(axis=1), grid[np.argmax(near, axis=1)], w), grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, wc, side="right") - 1, 0, grid.size - 2)
    f = (wc - grid[j]) / (grid[j + 1] - grid[j])
    W = np.zeros((grid.size, w.size))
    m = np.arange(w.size)
    W[j, m] = 1.0 - f
    W[j + 1, m] += f
    return W


def _outside(stores, bank):
    """the number of templates whose w lies outside the range of the stores by more than ASYM_TOL (read from an end
    store, within asym_tol)"""
    grid = np.array([float(getattr(st, "asym", 0.0)) for st in stores])
    w = np.asarray(bank.asym, float)
    return int(np.sum((w < grid[0] - ASYM_TOL) | (w > grid[-1] + ASYM_TOL)))


def _plan(bank, store, geometry, trim=False, asym_tol=0.0):
    """the stores a call reads (those of non-zero weight for some template), each a dict: store; use (n_template,) the
    templates it is read for; lnw (n_template,) ln of its weight; need (n_node,) the nodes read: all of them, or with
    trim only those of a valid pair or boundary point of a template in use; cH, cL (n_template,) the scales of the
    data to its model (_scaled), None where all zero; with bank.gain kappa_I the scale includes ln kappa_I (the model's
    template is kappa_I a)"""
    stores = _store_list(store)
    W = _store_weights(stores, bank, asym_tol)
    valid, bnode = geometry[0], geometry[3]
    out = []
    for st, w in zip(stores, W):
        use = w > 0
        if not use.any():
            continue
        need = np.ones(st.s.size, bool)
        if trim:
            need = valid[:, use].any(axis=1)
            need[bnode[use & (bnode >= 0)]] = True
        a = float(getattr(st, "asym", 0.0))
        cH = 0.5 * (np.log1p(a) - np.log1p(bank.asym))
        cL = 0.5 * (np.log1p(-a) - np.log1p(-bank.asym))
        gain = getattr(bank, "gain", None) or {}
        if gain.get("H", 1.0) != 1.0:                    # the model's template kappa_I a: data scaled by kappa_I too
            cH = cH + np.log(gain["H"])
        if gain.get("L", 1.0) != 1.0:
            cL = cL + np.log(gain["L"])
        with np.errstate(divide="ignore"):
            lnw = np.log(w)
        out.append(dict(store=st, asym=a, use=use, lnw=lnw, need=need, cH=cH if cH.any() else None,
                        cL=cL if cL.any() else None))
    return out


def _scaled(traj, e, xp):
    """the trajectories of a segment with the data of every template scaled to the model of the store of e (Sec. 3):
    ln|z_I''| = ln|z_I| + c_I, so that E_0|z_I''|^2 = ubar (1 +- w_j); the phases are kept"""
    if e["cH"] is None and e["cL"] is None:
        return traj
    out = dict(traj)
    for I in "HL":
        if e["c" + I] is not None:
            out["lq" + I] = (traj["lq" + I] + xp.asarray(e["c" + I])[:, None]).astype(xp.float32)
    return out


def _mix(xs, plan):
    """ln sum_j w_j exp x_j over the stores of the plan, the weights per template on the last axis (one store of
    weight 1 for every template: x itself)"""
    if len(xs) == 1 and np.all(plan[0]["lnw"] == 0.0):
        return xs[0]
    out = np.full(xs[0].shape, -np.inf)
    for x, e in zip(xs, plan):
        out = np.logaddexp(out, x + e["lnw"])
    return out


def _fit_chunk(bank, store, chunk, nodes_per_launch, resident, kept=None):
    """GPU: the number of segments of one pass over the tables, at most chunk, so that what the pass keeps of each on
    the device fits next to the tables: its trajectories (16 n_template n_time bytes each; bank may be a list of
    banks: the largest), or kept bytes when given (pe_redshift_epochs, which keeps the arrival times and the time
    weights of a segment and filters it again at every launch: the trajectories of two segments are then set aside
    for the one at work)"""
    banks = list(bank) if isinstance(bank, (list, tuple)) else [bank]
    if banks[0].device == "cpu":
        return chunk
    import cupy as cp
    stores = _store_list(store)
    free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
    sizes = [t.nbytes for t in stores[0].tables]
    if resident:
        free -= sum(sum(t.nbytes for t in st.tables) for st in stores if getattr(st, "device", None) is None)
    else:
        free -= nodes_per_launch * max(sizes)
    per_segment = max(16 * b.u.size * b.n_time for b in banks)
    if kept is not None:
        free -= 2 * per_segment
        per_segment = max(int(kept), 1)
    return max(1, min(chunk, int(0.6 * free / per_segment)))


def _filter(bank, seg, e):
    return bank.trajectories({I: seg["d_" + I][e] for I in "HL"},
                             {I: seg["sig2_" + I] for I in "HL"})


class _Trajectories:
    """the trajectories of the segments ev of one epoch, as a sequence: traj[j] filters the segment ev[j] when it is
    asked for (_filter: the same arrays every time) and only the one asked for last is kept, until drop().  A segment
    is 16 n_template n_time bytes on the device and the matched filter takes about a millisecond on a GPU, far less
    than a copy of the tables; so a pass over the tables holds as many segments as have room for their arrival times
    (_fit_chunk), not as many as have room for their trajectories"""

    def __init__(self, bank, seg, ev):
        self.bank, self.seg, self.ev = bank, seg, list(ev)
        self.j = self.last = None

    def __len__(self):
        return len(self.ev)

    def __getitem__(self, j):
        if j != self.j:
            self.last = None                               # freed before the next is filtered
            self.last = _filter(self.bank, self.seg, self.ev[j])
            self.j = j
        return self.last

    def drop(self):
        self.j = self.last = None


def _node_geometry(bank, st, ymin, ymax=None):
    """which (node, template) pairs the redshift store covers, and their ln a and y.

    a = s/u (u = ubar) is valid for y = sqrt(a) >= ymin and alpha_lo' < a < alpha_hi,
    alpha_lo' = min(alpha_lo, ymin^2).  a template whose a crosses
    alpha_lo' between two nodes also gets a boundary point at
    a = alpha_lo' (1 + 1e-9), read from the upper node's table.  With ymax a
    template keeps, of its nodes with y > ymax, only the first."""
    st = _store_list(st)[0]
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
    if ymax is not None:
        above = valid & (y[:nn] > ymax)
        drop = above & (np.cumsum(above, axis=0) > 1)
        valid &= ~drop
        lna[drop] = 0.0
        y[:nn][drop] = 0.0
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
    """the tables ks for a gather: host list (CPU) or device stack (GPU), and the slot of every node in it
    (n_node,), -1 for a node not in it"""
    slot = np.full(store.s.size, -1)
    if bank.device == "gpu" and resident:
        if getattr(store, "device", None) is None:
            store.device = _store.upload(store.tables, store.tau_idx, beta, delta)
        slot[:] = np.arange(store.s.size)
        return dict(store.device, beta=beta, delta=delta), slot
    slot[ks] = np.arange(len(ks))
    if bank.device == "cpu":
        return dict(T=[store.tables[k] for k in ks], tau=store.tau_idx, beta=beta, delta=delta), slot
    return _store.upload([store.tables[k] for k in ks], store.tau_idx, beta, delta), slot


def _read(bank, traj, tab, ids, lna, tab_ids, block, times, time_weights=None):
    if bank.device == "cpu":
        return _gather.gather_cpu(traj, tab, ids, lna, tab_ids, times, time_weights)
    return _gather.gather_gpu(traj, tab, ids, lna, tab_ids, block, times, time_weights)


def _release(bank):
    if bank.device == "gpu":
        _store.free()


def _single_setup(bank, seg, store, geometry, trim=False, asym_tol=0.0):
    """what single=True needs once per call: per detector I the columns of single.py (the amplitude bins, then for
    every store read the merger columns at s_H = s (1 + w_j), s_L = s (1 - w_j) of the nodes it reads) and ln u_I; the
    plan of the stores (_plan), each with the first column of its nodes and the signal powers s_I = a ubar (1 +- w_j)
    of the boundary point of every template.  (seg is not read: the stores' coverage of u_H / u_L is checked by
    _store_weights)"""
    bnode, blna = geometry[3], geometry[4]
    plan = _plan(bank, store, geometry, trim, asym_tol)
    xp = array_module(bank.device)
    G = _single.glitch_columns()
    lnT, col = {"H": [G], "L": [G]}, G.shape[1]
    for e in plan:
        nodes = np.nonzero(e["need"])[0]
        e.update(nodes=nodes, col=col)
        col += nodes.size
        for I, sg in (("H", 1.0), ("L", -1.0)):
            lnT[I].append(_single.merger_columns(e["store"].s[nodes] * (1.0 + sg * e["asym"])))
            e["s_b_" + I] = np.where(bnode >= 0, np.exp(blna) * bank.u * (1.0 + sg * e["asym"]), 0.0)
            if (xp is not np) if BOUNDARY_TABLES is None else BOUNDARY_TABLES:
                e["own_" + I] = _single.MergerTable(e["s_b_" + I], xp)   # Phi of the boundary point, once per bank
    out = dict(plan=plan, n_rho=_single.RHO_EDGES.size - 1)
    for I in "HL":
        out["lnT_" + I] = xp.asarray(np.concatenate(lnT[I], axis=1))
        out["ln_u_" + I] = np.log(getattr(bank, "u_" + I))
    return out


def _weights_array(x, shape, name):
    """x as a float array of the given shape, finite and >= 0 (ValueError otherwise)"""
    try:
        a = np.array(x, float)
    except (TypeError, ValueError):
        raise ValueError("glitch setting %s is not an array of numbers" % name)
    if a.shape != tuple(shape):
        raise ValueError("glitch setting %s has shape %s, not %s" % (name, a.shape, tuple(shape)))
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("glitch setting %s must be finite and >= 0" % name)
    return a


def _positive_sum(x, shape, name):
    """weights defined up to a factor: finite, >= 0, with a positive sum"""
    a = _weights_array(x, shape, name)
    if not a.sum() > 0:
        raise ValueError("glitch setting %s must have a positive sum" % name)
    return a


def _law(x, shape, name):
    """probabilities over the last axis: finite, >= 0, each row summing to 1 within SUM_TOL; returned divided by
    the sums, so that the model is normalized exactly"""
    a = _weights_array(x, shape, name)
    s = a.sum(axis=-1, keepdims=True)
    if np.any(np.abs(s - 1.0) > SUM_TOL):
        raise ValueError("glitch setting %s must sum to 1 (within %g) over its last axis; sums %s"
                         % (name, SUM_TOL, np.array2string(s.ravel(), precision=8)))
    return a / s


def _template_weights(pi_m, n_template, normalize):
    """pi_m (n_template,) checked (None stays None, uniform); divided by its sum when normalize"""
    if pi_m is None:
        return None
    a = _positive_sum(pi_m, (n_template,), "pi_m")
    return a / a.sum() if normalize else a


def _window_count(n_window, n_time):
    """n_window as an int that divides n_time (ValueError otherwise)"""
    if isinstance(n_window, (bool, np.bool_)) or not isinstance(n_window, (int, np.integer)):
        raise ValueError("glitch setting n_window must be an integer, not %r" % (n_window,))
    if n_window < 1 or n_time % n_window:
        raise ValueError("glitch setting n_window = %d must divide n_time = %d" % (n_window, n_time))
    return int(n_window)


def _windows_setup(glitch, bank, lags):
    """the settings of model='windows' (WINDOWS), checked: n_window divides n_time, 0 <= zone and 2 zone + 1 <=
    n_window, pi_m (n_template,) >= 0 with a positive sum (divided by it), and for each detector f (K,) and pi_k (K,
    n_rho) laws (summing to 1 within SUM_TOL, divided by their sums), 0 <= q < 1 (K,)"""
    g = dict(WINDOWS, **glitch)
    if set(g) != set(WINDOWS):
        raise ValueError("unknown glitch settings %s for model 'windows' (known: %s)"
                         % (sorted(set(g) - set(WINDOWS)), sorted(WINDOWS)))
    nw = _window_count(g["n_window"], bank.n_time)
    zone = g["zone"]
    if isinstance(zone, (bool, np.bool_)) or not isinstance(zone, (int, np.integer)) or zone < 0:
        raise ValueError("glitch setting zone must be an integer >= 0, not %r" % (zone,))
    if 2 * zone + 1 > nw:
        raise ValueError("a zone of %d windows on each side does not fit in n_window = %d" % (zone, nw))
    nr = _single.RHO_EDGES.size - 1
    out = dict(model="windows", n_window=nw, zone=int(zone), lags=lags,
               pi_m=_template_weights(g["pi_m"], bank.u.size, True), lnG=_window_columns(array_module(bank.device)))
    for I in "HL":
        p = g[I]
        if not isinstance(p, dict):
            raise ValueError("model 'windows' needs glitch setting %s = dict(f=(K,), q=(K,), pi_k=(K, %d))" % (I, nr))
        if set(p) != set(STATE):
            raise ValueError("glitch setting %s needs exactly the keys %s; got %s" % (I, list(STATE), sorted(p)))
        try:
            f = np.array(p["f"], float)
        except (TypeError, ValueError):
            raise ValueError("glitch setting %s.f is not an array of numbers" % I)
        if f.ndim != 1 or f.size < 1:
            raise ValueError("glitch setting %s.f must be a non-empty one-dimensional array" % I)
        K = f.size
        q = _weights_array(p["q"], (K,), I + ".q")
        if np.any(q >= 1.0):
            raise ValueError("glitch setting %s.q must lie in [0, 1); got %s" % (I, q))
        out[I] = dict(f=_law(f, (K,), I + ".f"), q=q, pi_k=_law(p["pi_k"], (K, nr), I + ".pi_k"))
    return out


def _glitch_setup(glitch, bank, seg, store):
    """the glitch settings (True: the defaults of GLITCH) with pi_k (n_rho,), the half window (samples) and the lags
    of the delay average: the lags of the tables within the light travel time between the sites; with
    model='windows' those of _windows_setup.  pi_k and pi_m are checked (shapes (n_rho,) and (n_template,), >= 0,
    positive sums); the window weights kbar of the default model do not depend on their normalization"""
    if glitch is not True and not isinstance(glitch, dict):
        raise ValueError("glitch must be None, False, True or a dict of settings, not %r" % (glitch,))
    dt = seg["T"] / bank.n_time
    lags = np.asarray(_store_list(store)[0].tau_idx, int).ravel()
    lags = lags[np.abs(lags) * dt <= np.linalg.norm(_det.x["H"] - _det.x["L"]) / _det.c]
    if glitch is not True and "model" in glitch:
        if glitch["model"] is None:                         # the default model, named
            glitch = {k: v for k, v in glitch.items() if k != "model"}
        elif glitch["model"] != "windows":
            raise ValueError("unknown glitch model %r: leave model out or None (one glitch per segment, Sec. 4) or "
                             "give 'windows' (Sec. 4a)" % (glitch["model"],))
        else:
            return _windows_setup(glitch, bank, lags)
    g = dict(GLITCH, **({} if glitch is True else glitch))
    if set(g) != set(GLITCH):
        raise ValueError("unknown glitch settings %s" % sorted(set(g) - set(GLITCH)))
    nr = _single.RHO_EDGES.size - 1
    pi_k = _single.amplitude_weights() if g["pi_k"] is None else _positive_sum(g["pi_k"], (nr,), "pi_k")
    pi_m = _template_weights(g["pi_m"], bank.u.size, False)
    window = float(g["window"])
    if not np.isfinite(window):
        raise ValueError("glitch setting window must be finite")
    return dict(model=None, pi_k=pi_k, pi_m=pi_m, window=window, half_window=int(round(g["window"] / dt)),
                lags=lags)


def _kbar(traj, I, gd, sd, xp):
    """ln kbar_I and ln(1 - kbar_I) (n_time,) of one segment with the fiducial glitch law of gd, on the device"""
    lg = _single.glitch_time_series(traj["lq" + I], sd["ln_u_" + I], gd["pi_k"], gd["pi_m"], xp)
    return _single.glitch_time_weights(lg, gd["half_window"], xp)


def _glitch_weights(traj, gd, sd, xp):
    """per segment, on the device and in logs (a loud glitch takes kbar and o below 1e-308 where l_g or Phi is
    above 1e308): the time weights (ln kbar_H, ln kbar_L) of the gather, the weights per time of the one-detector
    sums on each trajectory (rows ln o, ln o + ln kbar; o_H(t) on L, o_L(t') on H) and
    ln c0 = ln (1/n_t) sum_t o_H(t) (1 - kbar_L(t))"""
    if gd["model"] == "windows":
        return _window_weights(traj, gd, sd, xp)
    (kH, nH), (kL, nL) = _kbar(traj, "H", gd, sd, xp), _kbar(traj, "L", gd, sd, xp)
    return _time_weights(kH, nH, kL, nL, gd["lags"], xp)


def _time_weights(kH, nH, kL, nL, lags, xp):
    """the weights of _glitch_weights from ln kbar_I and ln(1 - kbar_I) (n_time,) of each detector"""
    oH = _single.ln_near_fraction(nH, lags, xp)                     # of the L arrival time t, kbar_H at t + tau
    oL = _single.ln_near_fraction(nL, -lags, xp)                    # of the H arrival time t', kbar_L at t' - tau
    x = oH + nL
    m = float(x.max())
    lc0 = -np.inf if m == -np.inf else m + float(xp.log(xp.exp(x - m).sum())) - np.log(x.size)
    return dict(tw=(kH, kL), H=xp.stack([oL, oL + kH]), L=xp.stack([oH, oH + kL]), lc0=lc0)


def _window_columns(xp):
    """the glitch columns ln L_k (NV, n_rho) on the device: the first n_rho columns of single=True"""
    return xp.asarray(_single.glitch_columns())


def _window_evidence(lq, ln_u, lnG, n_window, pi_m, xp):
    """ln E_w[k] (n_window, n_rho) of one detector; the one call of pe_redshift and of window_evidences"""
    return _single.window_evidence(lq, ln_u, lnG, n_window, pi_m, xp)


def _window_weights(traj, gd, sd, xp):
    """_glitch_weights of model='windows' (Sec. 4a): per detector the window evidences, the posterior of the state
    and ln kbar, ln(1 - kbar) of each window (single.window_posterior), given to each of its n_time / n_window
    arrival times; then the delay averages, the weight rows and ln c0 as for the default model.  Also, per detector
    I, win_I = (ln c_I, the posterior of the state (K,), ln E_w (n_window, n_rho)) on the host"""
    nw = gd["n_window"]
    n_win = traj["lqH"].shape[1] // nw
    lk, ln, win = {}, {}, {}
    for I in "HL":
        lnE = _window_evidence(traj["lq" + I], sd["ln_u_" + I], gd["lnG"], nw, gd["pi_m"], xp)
        p = gd[I]
        post = _single.window_posterior(lnE, p["f"], p["q"], p["pi_k"], gd["zone"], xp)
        lk[I], ln[I] = xp.repeat(post["ln_kbar"], n_win), xp.repeat(post["ln_near"], n_win)
        win[I] = (float(post["ln_c"]), _single._host(post["r"]), _single._host(lnE))
    out = _time_weights(lk["H"], ln["H"], lk["L"], ln["L"], gd["lags"], xp)
    out.update(win_H=win["H"], win_L=win["L"])
    return out


def _single_values(traj, sd, xp, gw=None):
    """per detector: ln E (n_w, n_template, n_rho + n_col) of the columns of _single_setup (one store, every node: n_col
    = n_node), per store read ln Lambda_S^I at the boundary point (n_w, n_template), and wmax, w = |z_I|^2 / u_I; for
    no weights (n_w = 1) or, with the glitch weights gw, the log weight rows 0, ln o, ln o + ln kbar (n_w = 3; the
    first is the plain sum)"""
    out = {}
    for I in "HL":
        lq, ln_u = traj["lq" + I], sd["ln_u_" + I]
        W = None if gw is None else xp.concatenate([xp.zeros((1, lq.shape[1])), gw[I]])
        if "own_" + I in sd["plan"][0]:                     # the boundary point of each store: a column of its own
            nc = sd["lnT_" + I].shape[1]
            lnE = _single._host(_single.ln_evidence(lq, ln_u, sd["lnT_" + I], W, xp, log_weights=True,
                                                    own=[e["own_" + I] for e in sd["plan"]]))
            lnE = lnE.reshape(-1, *lnE.shape[-2:])
            lnE, b = np.ascontiguousarray(lnE[..., :nc]), [lnE[..., nc + i].copy() for i in range(len(sd["plan"]))]
        else:
            lnE = _single._host(_single.ln_evidence(lq, ln_u, sd["lnT_" + I], W, xp, log_weights=True))
            b = [_single._host(_single.ln_merger_at(lq, ln_u, e["s_b_" + I], W, xp, log_weights=True))
                 for e in sd["plan"]]
            lnE, b = lnE.reshape(-1, *lnE.shape[-2:]), [x.reshape(-1, x.shape[-1]) for x in b]
        out[I] = (lnE, b, _single._host(_single._wmax(lq, ln_u, xp)))
    return out


def _network(traj, sd, tau, xp):
    """wnet (n_template,) on the host: the largest w_H(t + tau) + w_L(t) of the unscaled data over every arrival time
    and lag tau of the tables (gather.network_max), w_I = |z_I|^2 / u_I with the true u_I"""
    return _single._host(_gather.network_max(traj["lqH"], traj["lqL"], sd["ln_u_H"], sd["ln_u_L"], tau, xp))


def _store_single(one, i, e, n_node, n_rho):
    """the single-detector values of the store i of the plan (e) in the layout _single_nodes reads: per segment and
    detector ln E with the amplitude bins, then one column per node of the store (-inf at the nodes it does not read),
    its boundary values, and wmax"""
    def layout(lnE):
        if e["col"] == n_rho and e["nodes"].size == n_node == lnE.shape[-1] - n_rho:
            return lnE
        out = np.full(lnE.shape[:-1] + (n_rho + n_node,), -np.inf)
        out[..., :n_rho] = lnE[..., :n_rho]
        out[..., n_rho + e["nodes"]] = lnE[..., e["col"]:e["col"] + e["nodes"].size]
        return out
    return [{I: (layout(o[I][0]), o[I][1][i], o[I][2]) for I in "HL"} for o in one]


def _on_nodes(lnE, b, geometry):
    """one detector's values on the node geometry: lnE (n_template, n_node) at the nodes, b (n_template,) at the
    boundary point -> (n_node + 1, n_template), -inf off the valid pairs"""
    valid, bnode = geometry[0], geometry[3]
    return np.concatenate([np.where(valid, lnE.T, -np.inf), np.where(bnode >= 0, b, -np.inf)[None]])


def _single_nodes(one, lamN, geometry, n_rho, far=None, lc0=None, win=None):
    """the single-detector values on the node geometry of lamN (same valid pairs, same boundary row, -inf where
    lamN is), the glitch evidences and the largest w, stacked over the segments; with far (n_seg, 3, n_node + 1,
    n_template), the gather sums weighted by kbar_H, kbar_L and both, and ln c0 (n_seg,) also lamN_gH, lamN_gL,
    lamN_gHL; with win (model='windows': per segment the dict of _window_weights) only lamN_gHL of the three, and
    ln_c_I (n_seg,), state_I (n_seg, K), ln_window_I (n_seg, n_window, n_rho)"""
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
    if win is not None:
        del out["lamN_gH"], out["lamN_gL"]                  # no meaning in this model
        for I in "HL":
            w = [g["win_" + I] for g in win]
            out["ln_c_" + I] = np.array([v[0] for v in w])
            out["state_" + I] = np.stack([v[1] for v in w])
            out["ln_window_" + I] = np.stack([v[2] for v in w])
    for k in out:
        if k.startswith("lamN_"):
            out[k][~np.isfinite(lamN)] = -np.inf
    return out


def _launch(r, ks, tab, slot, lamN, far, block, xp):
    """the gathers of one launch of tables for the segments of one epoch (r, a read of _store_values): its nodes ks of
    the group, from tab (their slots in it: slot), at the valid (node, template) pairs and boundary points of the
    templates the store is read for, with the data scaled to the store's model (_scaled); into lamN and far"""
    valid, lna, _, bnode, blna = r["geometry"]
    bank, e, traj, times, tw = r["bank"], r["e"], r["traj"], r["times"], r["tw"]
    use, nn = e["use"], lamN.shape[1] - 1
    ok = valid[ks] & use[None, :]
    ids = np.nonzero(ok.any(axis=0))[0]
    lnaM = np.where(ok[:, ids], lna[ks][:, ids], 0.0).T
    tid = np.tile(slot[ks], (ids.size, 1))
    bids = np.nonzero(np.isin(bnode, ks) & use)[0]
    for j in range(len(traj)):
        t = _scaled(traj[j], e, xp) if ids.size or bids.size else None
        if ids.size:
            val = _read(bank, t, tab, ids, lnaM, tid, block, times[j], tw[j])
            val, fv = (val, None) if tw[j] is None else (val[0], val[1:])
            for i, k in enumerate(ks):
                keep = ok[i][ids]
                lamN[j, k, ids[keep]] = val[i][keep]
                if fv is not None:
                    far[j, :, k][:, ids[keep]] = fv[:, i][:, keep]
        if bids.size:
            val = _read(bank, t, tab, bids, blna[bids][:, None], slot[bnode[bids]][:, None], block, times[j], tw[j])
            val, fv = (val, None) if tw[j] is None else (val[0], val[1:])
            lamN[j, nn, bids] = val[0]
            if fv is not None:
                far[j, :, nn][:, bids] = fv[:, 0]
        del t
    traj.drop()


def _store_values(store, reads, groups, block, resident, xp):
    """the gather of one store for the segments of several epochs, each launch of tables copied once: reads, one per
    epoch that reads the store, dicts with bank, geometry, e (its plan entry of the store, _plan), traj, times, tw and
    gd (None: no glitch weights).  A launch holds the nodes of a group that any read needs; each read gathers its own
    nodes of the group, with the grids of the first of them, exactly as in a launch of its own (only the slots of the
    tables in the copy differ).  Per read lamN (n_seg, n_node + 1, n_template) and, with the glitch weights, far
    (n_seg, 3, n_node + 1, n_template), at the valid pairs and boundary points of the templates the store is read for;
    -inf elsewhere"""
    nn = store.s.size
    out = [(np.full((len(r["traj"]), nn + 1, r["bank"].M.size), -np.inf),
            None if r["gd"] is None else np.full((len(r["traj"]), 3, nn + 1, r["bank"].M.size), -np.inf)) for r in reads]
    bank = reads[0]["bank"]
    for ks in groups:
        mine = [[k for k in ks if r["e"]["need"][k]] for r in reads]
        union = [k for k in ks if any(k in m for m in mine)]
        if not union:
            continue
        tab, slot = _tables(store, union, store.beta[union[0]], store.delta[union[0]], bank, resident)
        for r, m, (lamN, far) in zip(reads, mine, out):
            if m:
                _launch(r, m, dict(tab, beta=store.beta[m[0]], delta=store.delta[m[0]]), slot, lamN, far, block, xp)
        del tab
        if not resident:
            _release(bank)
    return out


def _epochs_values(jobs, groups, block, resident, n_loud, n_random, shared_times=False):
    """_node_values of the segments of several epochs, each launch of tables copied once for all of them: jobs
    [(epoch, ev)], an epoch a dict bank, seg, geometry, plan (_plan), sd (_single_setup or None), gd (_glitch_setup
    or None) and seed, ev the indices of its segments.  Each segment is filtered, weighted, sampled and gathered with
    its own epoch's arguments; the stores are read in the order of asym.  Returns per job what _node_values returns"""
    xp = array_module(jobs[0][0]["bank"].device)
    data = []
    for ep, ev in jobs:
        bank, seg, sd, gd = ep["bank"], ep["seg"], ep["sd"], ep["gd"]
        traj = _Trajectories(bank, seg, ev)
        tau = ep["plan"][0]["store"].tau_idx
        gw, one, net, times = [], [], [], []
        for j, e in enumerate(ev):                          # what is kept of a segment; its trajectories are not
            t = traj[j]
            g = None if gd is None else _glitch_weights(t, gd, sd, xp)
            if sd is not None:
                one.append(_single_values(t, sd, xp, g))
                net.append(_network(t, sd, tau, xp))
            times.append(_gather.select_times(t, tau, n_loud, n_random, ep["seed"] + int(seg["rows"][e]), xp,
                                              shared=shared_times) if n_loud > 0 else None)
            if g is not None:                               # the rows of the one-detector sums were read above
                g = {k: v for k, v in g.items() if k not in ("H", "L")}
                g["tw"] = _gather.checked_weights(g["tw"], bank.n_time, xp)   # checked once, not at every launch
            gw.append(g)
            del t
        traj.drop()
        data.append(dict(ep, traj=traj, gw=gw, one=one if sd is not None else None, times=times,
                         tw=[None if g is None else g["tw"] for g in gw], net=net if sd is not None else None))
    read = {id(e["store"]): e["store"] for d in data for e in d["plan"]}
    parts = [[None] * len(d["plan"]) for d in data]
    for st in sorted(read.values(), key=lambda st: float(getattr(st, "asym", 0.0))):
        who = [(i, p) for i, d in enumerate(data) for p, e in enumerate(d["plan"]) if e["store"] is st]
        got = _store_values(st, [dict(data[i], e=data[i]["plan"][p]) for i, p in who], groups, block, resident, xp)
        for (i, p), g in zip(who, got):
            parts[i][p] = g
    for d in data:
        del d["traj"], d["times"], d["tw"]
    _release(jobs[0][0]["bank"])
    return [_mixed(d, part) for d, part in zip(data, parts)]


def _mixed(d, parts):
    """what _node_values returns from the gathers of one epoch's segments (d of _epochs_values) per store of its plan
    (parts, (lamN, far) each): the node values mixed over the stores (_mix), and with sd the dict of _single_nodes"""
    plan, sd, gd = d["plan"], d["sd"], d["gd"]
    lamN = _mix([p[0] for p in parts], plan)
    if sd is None:
        return lamN
    lc0 = None if gd is None else [g["lc0"] for g in d["gw"]]
    win = d["gw"] if gd is not None and gd["model"] == "windows" else None
    nn = plan[0]["store"].s.size
    nodes = [_single_nodes(_store_single(d["one"], i, e, nn, sd["n_rho"]), p[0], d["geometry"], sd["n_rho"], p[1], lc0,
                           win) for i, (e, p) in enumerate(zip(plan, parts))]
    out = dict(nodes[0])
    for k in out:
        if k.startswith("lamN_"):
            out[k] = _mix([x[k] for x in nodes], plan)
    out["wnet"] = np.stack(d["net"])
    return lamN, out


def _node_values(bank, seg, ev, store, geometry, groups, block, resident, n_loud, n_random, seed, single=None,
                 glitch=None, shared_times=False, trim=False, asym_tol=0.0):
    """ln Lambda of every template at every node of the redshift store, for the segments ev:
    (len(ev), n_node + 1, n_template), row n_node the boundary point at the smallest amplitude.
    with single (_single_setup) also the dict of _single_nodes, with glitch (_glitch_setup) its glitch terms.
    With several stores each is gathered for the templates it has weight for and the node values of the
    quantities linear in the merger likelihood are mixed (_mix); the weights of the data are computed once.
    One epoch of _epochs_values"""
    plan = single["plan"] if single is not None else _plan(bank, store, geometry, trim, asym_tol)
    ep = dict(bank=bank, seg=seg, geometry=geometry, plan=plan, sd=single, gd=glitch, seed=seed)
    return _epochs_values([(ep, ev)], groups, block, resident, n_loud, n_random, shared_times)[0]


def _single_out(single, n_seg, shape, bank, gd=None):
    """the arrays single=True adds to the result: ln_lambda_H, ln_lambda_L of the given shape per segment, and
    ln_glitch_H, ln_glitch_L, wmax_H, wmax_L, wnet, rho_edges; with the glitch settings gd also ln_lambda_gH,
    ln_lambda_gL, ln_lambda_gHL and glitch_pi_k, glitch_window, or for model='windows' ln_B of the given shape,
    ln_c_H, ln_c_L, state_H, state_L, ln_window_H, ln_window_L and glitch_model, glitch_n_window, glitch_zone"""
    if not single:
        return {}
    nm, nr = bank.u.size, _single.RHO_EDGES.size - 1
    out = {"rho_edges": _single.RHO_EDGES.copy()}
    for I in "HL":
        out["ln_lambda_" + I] = np.full((n_seg,) + shape, -np.inf)
        out["ln_glitch_" + I] = np.full((n_seg, nm, nr), -np.inf)
        out["wmax_" + I] = np.zeros((n_seg, nm))
    out["wnet"] = np.zeros((n_seg, nm))
    if gd is not None and gd["model"] == "windows":
        out["ln_B"] = np.full((n_seg,) + shape, -np.inf)
        for I in "HL":
            out["ln_c_" + I] = np.full(n_seg, np.nan)
            out["state_" + I] = np.full((n_seg, gd[I]["f"].size), np.nan)
            out["ln_window_" + I] = np.full((n_seg, gd["n_window"], nr), -np.inf)
        out.update(glitch_model="windows", glitch_n_window=gd["n_window"], glitch_zone=gd["zone"])
    elif gd is not None:
        for X in ("H", "L", "HL"):
            out["ln_lambda_g" + X] = np.full((n_seg,) + shape, -np.inf)
        out["glitch_pi_k"], out["glitch_window"] = gd["pi_k"], gd["window"]
    return out


def _node_keys(gd):
    """the node arrays reduced into the result: single detector, and with the glitch terms (model='windows':
    ln_B, the lambda_HL expression with the window weights)"""
    if gd is not None and gd["model"] == "windows":
        glitch = [("ln_B", "lamN_gHL")]
    else:
        glitch = [] if gd is None else [("ln_lambda_g" + X, "lamN_g" + X) for X in ("H", "L", "HL")]
    return [("ln_lambda_" + I, "lamN_" + I) for I in "HL"] + glitch


def _segment_keys(gd):
    """the per-segment arrays copied into the result"""
    keys = [k + I for I in "HL" for k in ("ln_glitch_", "wmax_")] + ["wnet"]
    if gd is not None and gd["model"] == "windows":
        keys += [k + I for I in "HL" for k in ("ln_c_", "state_", "ln_window_")]
    return keys


def pe_redshift(segments, bank, store, rows=None, chunk=8, zmax=8.0, nz=1600,
                m1_range=(3.0, 150.0), nm1=1600, ymin=0.1267, nodes_per_launch=6, block=256,
                mdet_range=(10.0, 500.0), resident=False, n_loud=1024, n_random=1024, seed=0, single=False,
                glitch=None, shared_times=False, zmin=None, asym_tol=0.0):
    """ln Lambda(m1_source, z) of each segment.

    segments          anything read_segments takes (see there)
    bank              mass_bank of a one-column bank of detector-frame total
                      masses, ascending, built on the same segments
    store             load_redshift_store, whose asym every template's w = (u_H - u_L)/(u_H + u_L)
                      must equal within ASYM_TOL; or a sequence of stores of distinct asym (same
                      nodes, grids and lags) whose range holds every w: the mixture of the two that
                      bracket it (module docstring).  ValueError otherwise
    rows              segment indices (default all)
    chunk             segments of one pass over the tables (each launch of tables is copied
                      once for all of them; on the GPU at most those whose arrival times, 8
                      n_template (n_loud + n_random) bytes each, fit next to the tables)
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
    shared_times      the same random arrival times for every template (gather.select_times)
    single            also the single-detector evidences (w = |z_I|^2 / u_I in each detector)
    glitch            None, or True or a dict of settings: also the merger likelihood ratios
                      given a glitch (implies single), with the glitch-time weights kbar of the
                      fiducial law pi_k (n_rho,) on rho_edges (default dN/drho ~ rho^-4 above
                      rho = 4, single.amplitude_weights) and pi_m (n_template,) over the
                      templates (default uniform), and the window (s, default 0.1) of the
                      arrival times near a glitch; the delay average runs over the lags of the
                      tables within the light travel time between the sites.
                      Or dict(model='windows', n_window=64, zone=1, pi_m=None, H=dict(f=, q=,
                      pi_k=), L=dict(...)): many glitches per segment (docs/glitch_likelihood.md
                      Sec. 4a), the segment cut into n_window windows (a divisor of n_time), the
                      zone of a window that and zone windows on each side, pi_m (n_template,)
                      the glitches over the templates (divided by its sum; default uniform), and
                      per detector K states (K may differ between H and L) with probabilities
                      f (K,), glitch probability per window q (K,) in [0, 1) and amplitude laws
                      pi_k (K, n_rho) on rho_edges (f and the rows of pi_k sum to 1 within 1e-6
                      and are divided by their sums).  Every setting is checked (ValueError)
    zmin              None, or the smallest redshift needed: a template reads, of its nodes with
                      y above D0 / D_L(zmin), only the first; ln Lambda (and every grid) is then the
                      same for z >= zmin and -inf below, and ln B averages over z >= zmin only
    asym_tol          a template whose w lies outside the range of the stores by at most asym_tol
                      (one store: differs from its asym by at most asym_tol) is read from the end
                      store alone, its data scaled to that store's model; 0: none (ValueError)

    ln B averages Lambda over the prior uniform in M_det = 2 m1 (1+z) on
    mdet_range and proportional to D_L^2 in luminosity distance, i.e.
    p(m1, z) ~ (1+z) D_L^2 dD_L/dz, on the (m1, z) cells.

    returns dict: m1 (nm1,), z (nz,), ln_lambda (n_segment, nm1, nz),
    ln_b (n_segment,), rows; with single also ln_lambda_H, ln_lambda_L
    (n_segment, nm1, nz) ln Lambda_S^I through the nodes and the reduction of
    ln_lambda, ln_glitch_H, ln_glitch_L (n_segment, n_template, n_rho) ln E_I
    of the amplitude bins rho_edges (n_rho + 1,), wmax_H, wmax_L
    (n_segment, n_template) the largest w of each template, wnet
    (n_segment, n_template) the largest w_H(t + tau) + w_L(t) over every
    arrival time and every lag of the store (gather.network_max: the largest
    W any table read of the template can have); with glitch also
    ln_lambda_gH, ln_lambda_gL, ln_lambda_gHL (n_segment, nm1, nz) ln lambda_H,
    ln lambda_L, ln lambda_HL through the nodes and the reduction of ln_lambda,
    and the settings used, glitch_pi_k and glitch_window.  With model='windows'
    instead of the three: ln_B (n_segment, nm1, nz) the lambda_HL expression
    with the window weights (the B of Sec. 4a, not averaged over the
    population; ln_b is that of ln_lambda), ln_c_H, ln_c_L (n_segment,) ln of
    the noise ratio c_I, state_H, state_L (n_segment, K) the posterior of the
    state, ln_window_H, ln_window_L (n_segment, n_window, n_rho) ln E_w[k]
    (as window_evidences), and glitch_model ('windows'), glitch_n_window,
    glitch_zone.  With asym_tol > 0 also asym_outside (int), the number of
    templates read from an end store outside its range (_outside).

    The call of one epoch of pe_redshift_epochs."""
    return pe_redshift_epochs([dict(segments=segments, bank=bank, rows=rows, seed=seed, single=single, glitch=glitch)],
                              store, chunk=chunk, zmax=zmax, nz=nz, m1_range=m1_range, nm1=nm1, ymin=ymin,
                              nodes_per_launch=nodes_per_launch, block=block, mdet_range=mdet_range, resident=resident,
                              n_loud=n_loud, n_random=n_random, shared_times=shared_times, zmin=zmin,
                              asym_tol=asym_tol)[0]


EPOCH = ("segments", "bank", "rows", "seed", "single", "glitch")    # the keys of an epoch of pe_redshift_epochs


def _check_bank(bank):
    if np.any(np.diff(bank.M) <= 0) or np.any(bank.q != 1.0) or np.any(bank.chi != 0.0):
        raise ValueError("pe_redshift needs an ascending equal-mass zero-spin bank")


def _epoch(ep, stores, ymin, ymax, trim, asym_tol, shape, setups=None):
    """what pe_redshift sets up once for one epoch (a dict of EPOCH, every key given): its segments, node geometry,
    glitch settings (gd), single-detector setup (sd) and plan of the stores read, and the arrays of its result (out,
    extra), each checked as pe_redshift checks them.  setups (a dict, filled here): the single-detector setup of a
    bank is made once for the epochs that share the bank (it depends on the bank and the stores only)"""
    bank = ep["bank"]
    store = stores[0] if len(stores) == 1 else stores
    _store_weights(stores, bank, asym_tol)
    seg = read_segments(ep["segments"], ep["rows"])
    geometry = _node_geometry(bank, stores[0], ymin, ymax)
    out = np.full((seg["rows"].size,) + shape, -np.inf)
    gd = None if ep["glitch"] is None or ep["glitch"] is False else _glitch_setup(ep["glitch"], bank, seg, store)
    single = ep["single"] or gd is not None
    sd = None
    if single:
        setups = {} if setups is None else setups
        if id(bank) not in setups:
            setups[id(bank)] = _single_setup(bank, seg, store, geometry, trim, asym_tol)
        sd = setups[id(bank)]
    extra = _single_out(single, seg["rows"].size, shape, bank, gd)
    if asym_tol:
        extra["asym_outside"] = _outside(stores, bank)
    plan = sd["plan"] if single else _plan(bank, store, geometry, trim, asym_tol)
    return dict(bank=bank, seg=seg, geometry=geometry, plan=plan, sd=sd, gd=gd, seed=ep["seed"], out=out, extra=extra)


def _source_frame(ep, ev, values, yq, m1, z, below):
    """the node values of the segments ev of one epoch (_node_values) through the source frame into its result"""
    bank, geometry, gd, out, extra = ep["bank"], ep["geometry"], ep["gd"], ep["out"], ep["extra"]
    single = ep["sd"] is not None
    xp = array_module(bank.device)
    lamN, one = values if single else (values, None)
    for j, e in enumerate(ev):
        out[e] = _to_source_frame(lamN[j], geometry[2], bank.M, yq, m1, z, xp)
        for key, node in _node_keys(gd) if single else ():
            extra[key][e] = _to_source_frame(one[node][j], geometry[2], bank.M, yq, m1, z, xp)
        for key in _segment_keys(gd) if single else ():
            extra[key][e] = one[key][j]
        if below is not None:
            out[e][:, below] = -np.inf
            for key, _ in _node_keys(gd) if single else ():
                extra[key][e][:, below] = -np.inf


def _prior_cells(m1, z, mdet_range, below):
    """the normalized weights of the ln B prior on the (m1, z) cells (pe_redshift)"""
    dl = _dl_of_z(z)
    lnp = np.log(np.maximum((1.0 + z) * dl ** 2 * np.gradient(dl, z), 1e-300))
    Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
    inside = (Md >= mdet_range[0]) & (Md <= mdet_range[1])
    if below is not None:
        inside &= ~below[None, :]
    w = np.where(inside, np.exp(lnp)[None, :], 0.0) * np.gradient(m1)[:, None] * np.gradient(z)[None, :]
    return w / w.sum()


def pe_redshift_epochs(epochs, store, chunk=8, zmax=8.0, nz=1600, m1_range=(3.0, 150.0), nm1=1600, ymin=0.1267,
                       nodes_per_launch=6, block=256, mdet_range=(10.0, 500.0), resident=False, n_loud=1024,
                       n_random=1024, seed=0, single=False, glitch=None, shared_times=False, zmin=None, asym_tol=0.0):
    """pe_redshift of several epochs, each with its own bank (its own noise curves), in one pass over the tables per
    chunk of their segments: the list of the dicts pe_redshift returns for each epoch alone, bit for bit.

    epochs   sequence of dicts, each with segments and bank as for pe_redshift and, optionally, its own rows, seed,
             single and glitch (EPOCH; default the arguments of this call: rows None, all of them)
    chunk    segments of all the epochs in one pass over the tables, in the order of the epochs (on the GPU at most
             those whose arrival times fit next to the tables, _fit_chunk): each launch of tables is copied once per
             chunk and read for every segment of it with its own epoch's arguments, the segment filtered again for
             the launch (_Trajectories)
    the other arguments as for pe_redshift, the same for every epoch.  The banks must be on one device.

    Example: res = pe_redshift_epochs([dict(segments=s1, bank=b1, seed=11), dict(segments=s2, bank=b2, seed=12,
    glitch=model2)], store, chunk=4096, glitch=model1) gives res[0] == pe_redshift(s1, b1, store, chunk=4096,
    seed=11, glitch=model1) and res[1] == pe_redshift(s2, b2, store, chunk=4096, seed=12, glitch=model2), array for
    array (a chunk never changes a result)."""
    default = dict(rows=None, seed=seed, single=single, glitch=glitch)
    epochs = [dict(default, **ep) for ep in epochs]
    for ep in epochs:
        if set(ep) != set(EPOCH):
            raise ValueError("an epoch of pe_redshift_epochs has the keys %s (segments and bank required); got %s"
                             % (list(EPOCH), sorted(ep)))
        _check_bank(ep["bank"])
    if len({ep["bank"].device for ep in epochs}) > 1:
        raise ValueError("the banks of pe_redshift_epochs must be on one device")
    stores = _store_list(store)
    z = np.linspace(0.02, zmax, int(nz))
    m1 = np.geomspace(m1_range[0], m1_range[1], int(nm1))
    yq = D0 / _dl_of_z(z)
    below = None if zmin is None else z < zmin
    ymax = None if zmin is None else float(D0 / _dl_of_z(zmin)[0])
    trim = zmin is not None or len(stores) > 1
    nn = stores[0].s.size
    groups = [list(range(g, min(g + nodes_per_launch, nn))) for g in range(0, nn, nodes_per_launch)]
    setups = {}
    eps = [_epoch(ep, stores, ymin, ymax, trim, asym_tol, (m1.size, z.size), setups) for ep in epochs]
    if not eps:
        return []
    store = stores[0] if len(stores) == 1 else stores
    kept = max((8 * ep["bank"].u.size * (n_loud + n_random) if n_loud > 0 else 0) + 8 * ep["bank"].n_time
               for ep in eps)                             # the arrival times (sel, lw) and time weights of a segment
    chunk = _fit_chunk([ep["bank"] for ep in eps], store, chunk, nodes_per_launch, resident, kept)
    for part in _epoch_chunks([ep["seg"]["rows"].size for ep in eps], chunk):
        values = _epochs_values([(eps[i], ev) for i, ev in part], groups, block, resident, n_loud, n_random,
                                shared_times)
        for (i, ev), v in zip(part, values):
            _source_frame(eps[i], ev, v, yq, m1, z, below)
    w = _prior_cells(m1, z, mdet_range, below)
    return [dict(m1=m1.copy(), z=z.copy(), ln_lambda=ep["out"], ln_b=np.array([_log_mean(L, w) for L in ep["out"]]),
                 rows=ep["seg"]["rows"], **ep["extra"]) for ep in eps]


def _model_power(bank, I):
    """kappa_I^2 u_I (n_template,): the power of the model's template of detector I (bank.MassBank), u_I itself when
    the gain is 1 or absent"""
    u, g = getattr(bank, "u_" + I), float((getattr(bank, "gain", None) or {}).get(I, 1.0))
    return u if g == 1.0 else g * g * u


def _merger_setup(population, bank, xp):
    """what the merger evidence of window_evidences needs, per detector I: the templates of non-zero weight (use_I),
    ln W of those rows on the nodes that any template of either detector uses (ln_W_I, on the device), the covered
    fraction sum W_I, and the merger columns of those nodes (lnT, on the device).

    The weights of detector I bracket the signal power of the model, s_I = kappa_I^2 u_I y^2 (_model_power): a merger
    at y arrives in the data of I with amplitude kappa_I (bank.gain, the signal gain), so its matched filter has mean
    kappa_I y u_I and w_I = |z_I|^2 / u_I (the data's statistic, which the columns Phi(w; s) read) the noncentrality
    kappa_I^2 u_I y^2 |R|^2.  The powers of stage 2's single-detector columns are the same (bank.u, bank.asym)"""
    from .population import merger_weights
    if np.any(np.diff(bank.M) <= 0):
        raise ValueError("the merger evidence needs a bank of ascending masses (a one-column bank, as pe_redshift)")
    W = {I: merger_weights(bank.M, _model_power(bank, I), population) for I in "HL"}
    cols = np.flatnonzero((W["H"] > 0).any(axis=0) | (W["L"] > 0).any(axis=0))
    if cols.size == 0:
        raise ValueError("the merger population has no weight that the ladder reaches")
    out = dict(lnT=xp.asarray(_single.merger_columns(np.asarray(population["s"], float)[cols])))
    for I in "HL":
        use = np.flatnonzero((W[I] > 0).any(axis=1))
        with np.errstate(divide="ignore"):
            out["ln_W_" + I] = xp.asarray(np.log(W[I][np.ix_(use, cols)]))
        out["use_" + I], out["covered_" + I] = use, float(W[I].sum())
    return out


def window_evidences(segments, bank, rows=None, chunk=8, n_window=64, pi_m=None, population=None, lags=None):
    """the window evidences of each segment and detector (docs/glitch_likelihood.md Sec. 4a, stage 1): the
    matched filter and single.window_evidence, no tables.

    segments   anything read_segments takes
    bank       mass_bank built on the same segments
    rows       segment indices (default all)
    chunk      segments filtered at once
    n_window   windows per segment, a divisor of bank.n_time
    pi_m       (n_template,) weights of the templates (>= 0, divided by their sum; default uniform)
    population None, or the population that stage 2 averages over (population.merger_population: the (m1, z) grid
               of pe_redshift, its cell weights population.population_weights, the ladder s of signal powers of
               the tables, store.s, and pe_redshift's ymin): also the merger evidence of each window
               (docs/2026-11-15_mergers_in_stage1.md, Sec. 4), single.window_merger with the weights
               population.merger_weights of each detector, on the power of the model kappa_I^2 u_I (bank.gain;
               _model_power), on the merger columns of the ladder.  Needs a bank of ascending masses (the
               one-column bank of pe_redshift)

    returns dict: ln_window_H, ln_window_L (n_segment, n_window, n_rho) ln E_w[k], the same arrays as pe_redshift
    with glitch=dict(model='windows') and the same n_window and pi_m; wmax_H, wmax_L (n_segment, n_template) the
    largest w of each template; rho_edges (n_rho + 1,); rows.  w = |z_I|^2 / u_I with the power u_I of detector I.
    With population also ln_merger_H, ln_merger_L (n_segment, n_window) ln M_w, and merger_covered_H,
    merger_covered_L the population weight the ladder reaches (sum W_I, the mean of M_w over Gaussian noise); the
    other arrays are those without population, bit for bit.  With lags (the lags of the tables,
    store.tau_idx) also wnet (n_segment, n_template) as pe_redshift gives it, with or without population."""
    seg = read_segments(segments, rows)
    nw = _window_count(n_window, bank.n_time)
    pm = _template_weights(pi_m, bank.u.size, True)
    xp = array_module(bank.device)
    lnG, ln_u = _window_columns(xp), {I: np.log(getattr(bank, "u_" + I)) for I in "HL"}
    n, nm, nr = seg["rows"].size, bank.u.size, _single.RHO_EDGES.size - 1
    mg = None if population is None else _merger_setup(population, bank, xp)
    out, net = {}, np.zeros((n, nm))
    for I in "HL":
        out["ln_window_" + I] = np.full((n, nw, nr), -np.inf)
        out["wmax_" + I] = np.zeros((n, nm))
        if mg is not None:
            out["ln_merger_" + I] = np.full((n, nw), -np.inf)
    for ev in _chunks(seg, chunk):
        traj = [_filter(bank, seg, e) for e in ev]
        for t, e in zip(traj, ev):
            for I in "HL":
                out["ln_window_" + I][e] = _single._host(_window_evidence(t["lq" + I], ln_u[I], lnG, nw, pm, xp))
                out["wmax_" + I][e] = _single._host(_single._wmax(t["lq" + I], ln_u[I], xp))
                if mg is not None:
                    use = mg["use_" + I]
                    out["ln_merger_" + I][e] = _single._host(_single.window_merger(
                        t["lq" + I][xp.asarray(use)], ln_u[I][use], mg["lnT"], nw, mg["ln_W_" + I], xp))
            if lags is not None:
                net[e] = _network(t, dict(ln_u_H=ln_u["H"], ln_u_L=ln_u["L"]), lags, xp)
        del traj
    if mg is not None:
        out.update(merger_covered_H=mg["covered_H"], merger_covered_L=mg["covered_L"])
    return dict(out, rho_edges=_single.RHO_EDGES.copy(), rows=seg["rows"], **({} if lags is None else dict(wnet=net)))


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
                 glitch=None, shared_times=False, asym_tol=0.0):
    """ln Lambda of each template of the bank, the distance integrated over its prior.

    segments          anything read_segments takes (see there)
    bank              mass_bank built on the same segments, rows (M_det, q, chi)
    store             load_redshift_store, or a sequence of stores of distinct asym: as for pe_redshift
                      (the mixture is taken on the node values, before the distance integral)
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
    shared_times      the same random arrival times for every template (gather.select_times)
    single            also the single-detector evidences (w = |z_I|^2 / u_I in each detector)
    glitch            None, or True or a dict of settings: also the merger likelihood ratios
                      given a glitch, or with model='windows' the B of Sec. 4a (see pe_redshift)
    asym_tol          as for pe_redshift (and asym_outside in the result when > 0)

    each template reads the store at every node s it reaches, y = sqrt(s/u), and
    Lambda = sum_j w_j Lambda(y_j) with the weights of distance_weights.  ln B
    is the average of Lambda over the bank, each template weighted by its cell
    (the product of the spacings of the distinct values of every axis that varies).

    returns dict: M, q, chi, u (n_template,), ln_lambda (n_segment, n_template),
    ln_b (n_segment,), rows; with single also ln_lambda_H, ln_lambda_L
    (n_segment, n_template) ln Lambda_S^I with the same distance weights, and
    ln_glitch_H, ln_glitch_L, rho_edges, wmax_H, wmax_L, wnet as for pe_redshift; with
    glitch also ln_lambda_gH, ln_lambda_gL, ln_lambda_gHL (n_segment, n_template)
    with the same distance weights, glitch_pi_k and glitch_window; with
    model='windows' ln_B (n_segment, n_template) with the same distance weights
    in their place and the other arrays and settings of pe_redshift."""
    stores = _store_list(store)
    _store_weights(stores, bank, asym_tol)
    store = stores[0] if len(stores) == 1 else stores
    seg = read_segments(segments, rows)
    geometry = _node_geometry(bank, stores[0], ymin)
    yN = geometry[2]
    nn = stores[0].s.size
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
    trim = len(stores) > 1
    sd = _single_setup(bank, seg, store, geometry, trim, asym_tol) if single else None
    extra = _single_out(single, seg["rows"].size, (bank.u.size,), bank, gd)
    if asym_tol:
        extra["asym_outside"] = _outside(stores, bank)
    chunk = _fit_chunk(bank, store, chunk, nodes_per_launch, resident)
    for ev in _chunks(seg, chunk):
        lamN = _node_values(bank, seg, ev, store, geometry, groups, block, resident, n_loud, n_random, seed, sd, gd,
                            shared_times, trim, asym_tol)
        if single:
            lamN, one = lamN
        for j, e in enumerate(ev):
            out[e] = _distance_integral(lnW, lamN[j])
            for key, node in _node_keys(gd) if single else ():
                extra[key][e] = _distance_integral(lnW, one[node][j])
            for key in _segment_keys(gd) if single else ():
                extra[key][e] = one[key][j]

    cell = np.ones(bank.u.size)
    for ax in (bank.M, bank.q, bank.chi):
        vals = np.unique(ax)
        if vals.size > 1:
            cell *= np.gradient(vals)[np.searchsorted(vals, ax)]
    w = cell / cell.sum()
    ln_b = np.array([_log_mean(L, w) for L in out])
    return dict(M=bank.M, q=bank.q, chi=bank.chi, u=bank.u, ln_lambda=out, ln_b=ln_b,
                rows=seg["rows"], **extra)
