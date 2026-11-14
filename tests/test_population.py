"""the population average of Eq. (32) (exmarg.search.population) against a grid done by hand and against the lines of
notebook 07 that build its population weights."""
import numpy as np
from scipy.special import logsumexp

from exmarg.priors import population_m1_z
from exmarg.search import population

# p(m1) = m1 and p(z) = 2 - z, linear so that the interpolation is exact
POP = ((np.array([0.0, 100.0]), np.array([0.0, 100.0])), (np.array([0.0, 1.0]), np.array([2.0, 1.0])))
M1, Z = np.array([10.0, 20.0, 40.0]), np.array([0.1, 0.3, 0.7])


def test_hand_grid():
    """3 x 3 grid by hand: gradients dm1 = (10, 15, 20), dz = (0.2, 0.3, 0.4); p(m1) dm1 = (100, 300, 800), p(z) dz =
    (0.38, 0.51, 0.52); M_det = 2 m1 (1 + z) = (22, 26, 34 | 44, 52, 68 | 88, 104, 136), the last two outside
    [20, 100].  Cell weights (38, 51, 52 | 114, 153, 156 | 304, 0, 0), sum 868.  Lambda = (1, nan, 3 | 4, -inf, 6 |
    7, 8, 9), nan and -inf adding nothing: Lbar = (38 + 156 + 456 + 936 + 2128) / 868 = 3714 / 868.  Agreement to
    1e-13."""
    w = population.population_weights(M1, Z, (20.0, 100.0), POP)
    hand = np.array([[38.0, 51.0, 52.0], [114.0, 153.0, 156.0], [304.0, 0.0, 0.0]]) / 868.0
    assert np.allclose(w, hand, rtol=1e-13, atol=0), w * 868
    with np.errstate(divide='ignore'):
        L = np.log(np.array([[1.0, np.nan, 3.0], [4.0, 0.0, 6.0], [7.0, 8.0, 9.0]]))
    assert abs(population.ln_lambda_bar(L, M1, Z, (20.0, 100.0), POP) - np.log(3714.0 / 868.0)) < 1e-13
    stack = np.stack([L, L + 1000.0, np.full((3, 3), -np.inf), np.where(w > 0, -np.inf, 50.0)])
    out = population.ln_lambda_bar(stack, M1, Z, (20.0, 100.0), POP, chunk=3)
    assert np.allclose(out[:2], np.log(3714.0 / 868.0) + np.array([0.0, 1000.0]), rtol=0, atol=1e-12), out
    assert out[2] == -np.inf and out[3] == -np.inf, out


def test_matches_notebook_07():
    """the default population and support reproduce the weights of notebook 07 (its code cell with
    demo_population.npz, grid 150 x 150, M_det in [20, 400]) to 1e-13, and its lambda_bar to 1e-12"""
    (pm, ppm), (pz, ppz) = population_m1_z()
    m1 = np.geomspace(4.0, 200.0, 150)
    z = np.linspace(0.02, 10.0, 150)
    p_m1 = np.interp(m1, pm, ppm, left=0.0, right=0.0)
    p_z = np.interp(z, pz, ppz, left=0.0, right=0.0)
    M_det = 2.0 * m1[:, None] * (1.0 + z[None, :])
    w = p_m1[:, None] * p_z[None, :] * np.gradient(m1)[:, None] * np.gradient(z)[None, :] * (
        (M_det >= 20.0) & (M_det <= 400.0))
    ln_w = np.log(np.maximum(w / w.sum(), 1e-300))
    mine = population.population_weights(m1, z)
    assert np.allclose(mine, w / w.sum(), rtol=1e-13, atol=0)
    rng = np.random.default_rng(3)
    L = rng.normal(0.0, 3.0, (4, 150, 150))
    L[0, :40] = -np.inf
    L[1] += 400.0
    nb = logsumexp(np.where(np.isfinite(L), L, -np.inf) + ln_w[None], axis=(1, 2))
    assert np.allclose(population.ln_lambda_bar(L, m1, z), nb, rtol=0, atol=1e-12)
