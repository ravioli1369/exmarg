"""
the average of the likelihood ratio of a segment over the population, Eq. (32) of the paper:

    Lbar_i = int dz int dM_det pi(M_det, z) Lambda_i(M_det, z) = sum_{m1, z} w(m1, z) Lambda_i(m1, z)

on the (m1, z) grid of pe_redshift, with the cell weights of notebook 07 (and of pe_redshift's ln B):

    w(m1, z) ~ p(m1) p(z) dm1 dz      inside mdet_range of M_det = 2 m1 (1 + z), zero outside, sum 1

dm1, dz the np.gradient spacings of the grid, p(m1), p(z) the population of exmarg.priors.population_m1_z()
interpolated on the grid (zero outside its tables).  A cell where ln Lambda is -inf (outside the bank) adds
nothing; a nan anywhere is a ValueError, not a Lambda of 0.  The same average applies to ln_lambda_gH,
ln_lambda_gL, ln_lambda_H, ln_lambda_L of pe_redshift.

merger_population, merger_weights: the weight W[m, k] that this average puts on template m and node k of the ladder of
the tables for one detector (docs/2026-11-15_mergers_in_stage1.md, Sec. 4), for the merger evidence of a window in
stage 1 (single.window_merger, pe.window_evidences).
"""

import numpy as np

from ..priors import population_m1_z


def population_weights(m1, z, mdet_range=(20.0, 400.0), population=None):
    """w (nm1, nz): the normalized weights of the (m1, z) cells.

    m1 (nm1,), z (nz,)  the grid of pe_redshift (source-frame primary mass, redshift)
    mdet_range          support in detector-frame total mass 2 m1 (1 + z) (notebook 07: the bank, 20 to 400)
    population          ((m1, p_m1), (z, p_z)), default exmarg.priors.population_m1_z()"""
    (pm, ppm), (pz, ppz) = population_m1_z() if population is None else population
    m1, z = np.asarray(m1, float), np.asarray(z, float)
    p_m1 = np.interp(m1, pm, ppm, left=0.0, right=0.0)
    p_z = np.interp(z, pz, ppz, left=0.0, right=0.0)
    Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
    inside = (Md >= mdet_range[0]) & (Md <= mdet_range[1])
    w = np.where(inside, p_m1[:, None] * p_z[None, :], 0.0) * np.gradient(m1)[:, None] * np.gradient(z)[None, :]
    total = w.sum()
    if not total > 0:
        raise ValueError('the population has no weight on the grid inside mdet_range')
    return w / total


def ln_lambda_bar(ln_lambda, m1, z, mdet_range=(20.0, 400.0), population=None, chunk=16):
    """ln Lbar_i (n_segment,), or a float for one segment.

    ln_lambda  (n_segment, nm1, nz) or (nm1, nz): ln Lambda on the grid (m1, z) of pe_redshift
    m1, z, mdet_range, population  see population_weights
    chunk      segments averaged at once

    -inf is Lambda = 0; nan (a likelihood ratio that failed) raises ValueError naming the segments."""
    w = population_weights(m1, z, mdet_range, population)
    L = np.asarray(ln_lambda, float)
    one = L.ndim == 2
    L = L[None] if one else L
    if L.shape[1:] != w.shape:
        raise ValueError('ln_lambda has cells %s, the grid %s' % (L.shape[1:], w.shape))
    nan = np.nonzero(np.isnan(L).any(axis=(1, 2)))[0]
    if nan.size:
        raise ValueError('ln_lambda has NaN in %d of %d segments, segments %s'
                         % (nan.size, L.shape[0], nan[:20].tolist()))
    pos = w > 0
    out = np.empty(L.shape[0])
    for c in range(0, L.shape[0], chunk):
        V = L[c:c + chunk]
        ok = (V > -np.inf) & pos[None]                     # -inf counts as Lambda = 0
        V = np.where(ok, V, -np.inf)
        mx = V.max(axis=(1, 2))
        sh = np.where(np.isfinite(mx), mx, 0.0)
        s = np.sum(np.where(ok, w[None] * np.exp(V - sh[:, None, None]), 0.0), axis=(1, 2))
        with np.errstate(divide='ignore'):
            out[c:c + chunk] = sh + np.log(s)
    return float(out[0]) if one else out


POPULATION_KEYS = ('m1', 'z', 'weights', 's', 'ymin')


def merger_population(m1, z, weights, s, ymin):
    """the population of the merger evidence of stage 1 (docs/2026-11-15_mergers_in_stage1.md, Sec. 4), checked:
    dict m1 (nm1,), z (nz,) the grid of pe_redshift; weights (nm1, nz) the cell weights of stage 2
    (population_weights: finite, >= 0, sum 1 over the cells of the population); s (n_node,) the ladder of signal
    powers of the tables (store.s, ascending, > 0); ymin the smallest y = D0 / D_L the tables represent (pe_redshift's
    ymin).  ValueError naming what is wrong"""
    m1, z, s = (np.asarray(a, float) for a in (m1, z, s))
    w = np.asarray(weights, float)
    if m1.ndim != 1 or z.ndim != 1 or w.shape != (m1.size, z.size):
        raise ValueError('the merger population needs m1 (nm1,), z (nz,) and weights (nm1, nz); got %s, %s, %s'
                         % (m1.shape, z.shape, w.shape))
    if not np.all(np.isfinite(w)) or np.any(w < 0) or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError('the merger population weights must be finite, >= 0 and sum to 1 (sum %r)' % float(w.sum()))
    if s.ndim != 1 or s.size < 2 or np.any(np.diff(s) <= 0) or s[0] <= 0 or not np.all(np.isfinite(s)):
        raise ValueError('the ladder s must hold at least two finite, positive, ascending signal powers')
    if np.any(z < 0) or np.any(m1 <= 0) or not np.isfinite(float(ymin)) or float(ymin) < 0:
        raise ValueError('the merger population needs z >= 0, m1 > 0 and a finite ymin >= 0')
    return dict(m1=m1, z=z, weights=w, s=s, ymin=float(ymin))


def checked_population(population):
    """merger_population(**population) for a dict with exactly the keys POPULATION_KEYS (ValueError otherwise)"""
    if not isinstance(population, dict) or set(population) != set(POPULATION_KEYS):
        raise ValueError('the merger population must be a dict with the keys %s, not %r'
                         % (list(POPULATION_KEYS), sorted(population) if isinstance(population, dict) else population))
    return merger_population(**population)


def merger_weights(M, u, population):
    """W (n_template, n_node): the weight that the population of stage 2 puts on template m and node k of the ladder
    for one detector (docs/2026-11-15_mergers_in_stage1.md, Sec. 4), so that the merger evidence of a window is
    M_w = sum_{m, k} W[m, k] (1/n_win) sum_{t in w} Phi(w_m(t); s_k).

    M           (n_template,) ascending detector-frame total masses of the bank
    u           (n_template,) the power u_I of each template in this detector (bank.u_H or bank.u_L)
    population  merger_population(...)

    Template m reaches the redshift z when ymin <= y(z) <= y_m,last and y(z) >= y_m,0, with y(z) = D0 / D_L(z) and
    y_m,k = sqrt(s_k / u_m) its nodes.  Every cell (m1, z) of weight w gives w to the two templates that bracket its
    M_det = 2 m1 (1 + z) among those that reach z (linearly in M_det, as _to_source_frame interpolates among the
    templates with a finite value), and for each of them to the two nodes that bracket y(z) (linearly in y); a cell
    whose M_det lies outside the templates that reach z, or z where fewer than two do, is left out, as in stage 2.
    sum W is then the covered fraction of the population (1 - w_inf of plant.reduce for one store and equal noise
    curves).  float64"""
    from .pe import _dl_of_z
    from .segments import D0
    p = checked_population(population)
    M, u = np.asarray(M, float), np.asarray(u, float)
    if M.ndim != 1 or u.shape != M.shape or np.any(np.diff(M) <= 0) or np.any(u <= 0):
        raise ValueError('merger_weights needs ascending masses M and positive powers u of the same length')
    s, w = p['s'], p['weights']
    y = D0 / _dl_of_z(p['z'])                                         # (nz,)
    Md = 2.0 * p['m1'][:, None] * (1.0 + p['z'])[None, :]             # (nm1, nz)
    yk = np.sqrt(s[None, :] / u[:, None])                             # (n_template, n_node), ascending in k
    W = np.zeros(yk.shape)
    for j in np.flatnonzero(w.sum(axis=0) > 0):
        good = np.flatnonzero((y[j] >= p['ymin']) & (y[j] >= yk[:, 0]) & (y[j] <= yk[:, -1]))
        if good.size < 2:
            continue
        Mg = M[good]
        on = np.flatnonzero((w[:, j] > 0) & (Md[:, j] >= Mg[0]) & (Md[:, j] <= Mg[-1]))
        if not on.size:
            continue
        md, wc = Md[on, j], w[on, j]
        i = np.clip(np.searchsorted(Mg, md, side='right') - 1, 0, good.size - 2)
        t = (md - Mg[i]) / (Mg[i + 1] - Mg[i])
        for m, wm in ((good[i], wc * (1.0 - t)), (good[i + 1], wc * t)):
            ym = yk[m]                                                # (n_on, n_node)
            k = np.clip((ym <= y[j]).sum(axis=1) - 1, 0, s.size - 2)
            r = np.arange(m.size)
            tau = (y[j] - ym[r, k]) / (ym[r, k + 1] - ym[r, k])
            np.add.at(W, (m, k), wm * (1.0 - tau))
            np.add.at(W, (m, k + 1), wm * tau)
    return W
