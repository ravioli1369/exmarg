"""the unbiased tables: linear interpolation of a convex function overshoots by h^2 T'' / 12 on average, and unbias()
removes exactly that (constructed functions, no store needed)."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from exmarg.search.store import unbias


def test_unbias_subtracts_the_second_differences():
    phi, b, d = np.linspace(0.0, np.pi, 49), np.linspace(-2.0, 3.0, 40), np.linspace(-1.0, 1.0, 31)
    T = (0.7 * np.cos(2 * phi)[None, :, None, None] + 1.3 * b[None, None, :, None] ** 2
         + 0.4 * d[None, None, None, :] ** 2 + 0.2 * b[None, None, :, None] * d[None, None, None, :]) * np.ones((3, 1, 1, 1))
    hp, hb, hd = phi[1] - phi[0], b[1] - b[0], d[1] - d[0]
    second = (2 * 1.3 * hb ** 2 + 2 * 0.4 * hd ** 2
              + 0.7 * (np.cos(2 * (phi + hp)) - 2 * np.cos(2 * phi) + np.cos(2 * (phi - hp)))[None, :, None, None])
    got = (T - unbias(T.astype(np.float32)))[:, :, 1:-1, 1:-1]
    err = np.abs(got - second / 12.0).max()
    assert err < 5e-6, 'interior of T - unbias(T) differs from (second differences)/12 by %.1e' % err
    edge = np.abs((T - unbias(T.astype(np.float32)))[:, :, 0, 1:-1] - (second - 2 * 1.3 * hb ** 2) / 12.0).max()
    assert edge < 5e-6, 'the end point along beta must keep its value along beta (error %.1e)' % edge


def test_unbiased_interpolation_has_no_mean_error():
    f = lambda b, d: np.log(1.0 + 0.3 * np.exp(b) + 0.2 * np.exp(b + 2.0 * d))      # a log-sum-exp, convex like lnS
    bg, dg = np.arange(-3.0, 3.0, 0.139), np.arange(-2.0, 2.0, 0.0697)              # the spacings of the tables
    G = f(bg[:, None], dg[None, :])[None, None].repeat(3, 1)
    Gu = unbias(G.astype(np.float32))[0, 1].astype(float)
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-2.5, 2.5, 200000), rng.uniform(-1.5, 1.5, 200000)])
    truth = f(pts[:, 0], pts[:, 1])
    raw = RegularGridInterpolator((bg, dg), G[0, 1])(pts) - truth
    unb = RegularGridInterpolator((bg, dg), Gu)(pts) - truth
    se = raw.std() / np.sqrt(raw.size)
    assert raw.mean() > 100 * se, 'the control must fail: plain interpolation overshoots (mean %.2e)' % raw.mean()
    assert abs(unb.mean()) < 0.01 * raw.mean(), 'mean error %.2e after unbias, %.2e before' % (unb.mean(), raw.mean())
