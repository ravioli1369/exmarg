"""
the average of the likelihood ratio of a segment over the population, Eq. (32) of the paper:

    Lbar_i = int dz int dM_det pi(M_det, z) Lambda_i(M_det, z) = sum_{m1, z} w(m1, z) Lambda_i(m1, z)

on the (m1, z) grid of pe_redshift, with the cell weights of notebook 07 (and of pe_redshift's ln B):

    w(m1, z) ~ p(m1) p(z) dm1 dz      inside mdet_range of M_det = 2 m1 (1 + z), zero outside, sum 1

dm1, dz the np.gradient spacings of the grid, p(m1), p(z) the population of exmarg.priors.population_m1_z()
interpolated on the grid (zero outside its tables).  A cell where ln Lambda is -inf or nan (outside the bank)
adds nothing.  The same average applies to ln_lambda_gH, ln_lambda_gL, ln_lambda_H, ln_lambda_L of pe_redshift.
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
    chunk      segments averaged at once"""
    w = population_weights(m1, z, mdet_range, population)
    L = np.asarray(ln_lambda, float)
    one = L.ndim == 2
    L = L[None] if one else L
    if L.shape[1:] != w.shape:
        raise ValueError('ln_lambda has cells %s, the grid %s' % (L.shape[1:], w.shape))
    pos = w > 0
    out = np.empty(L.shape[0])
    for c in range(0, L.shape[0], chunk):
        V = L[c:c + chunk]
        ok = (V > -np.inf) & pos[None]                     # nan and -inf count as Lambda = 0
        V = np.where(ok, V, -np.inf)
        mx = V.max(axis=(1, 2))
        sh = np.where(np.isfinite(mx), mx, 0.0)
        s = np.sum(np.where(ok, w[None] * np.exp(V - sh[:, None, None]), 0.0), axis=(1, 2))
        with np.errstate(divide='ignore'):
            out[c:c + chunk] = sh + np.log(s)
    return float(out[0]) if one else out
