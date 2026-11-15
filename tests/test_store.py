"""the unbiased tables: linear interpolation of a convex function overshoots by h^2 T'' / 12 on average, and unbias()
removes exactly that (constructed functions, no store needed); and the loading of a store folder of small random
tables: asym, nodes, the unbiased cache."""
import numpy as np
import pytest
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
    edge = np.abs((T - unbias(T.astype(np.float32)))[:, :, 0, 1:-1] - ((second - 2 * 1.3 * hb ** 2) / 12.0)[:, :, 0, :]).max()
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


# ----------------------------------------------------------------------------------------------- loading


def write_store(folder, s_nodes=(0.5, 2.0, 8.0), asym=None, seed=0):
    """a store folder of bd_*.npz files with the members load_redshift_store reads (asym only when given: a list, one
    per file, or one value)"""
    import os
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, s in enumerate(s_nodes):
        extra = {} if asym is None else dict(asym=np.array(asym[i] if np.ndim(asym) else asym))
        np.savez(os.path.join(folder, 'bd_s%07.3f_n120_x.npz' % s), T=rng.normal(size=(3, 49, 6, 5)).astype(np.float32),
                 beta=np.linspace(-2.0, 3.0, 6) + np.log(s), delta=np.linspace(-1.0, 1.0, 5), tau_idx=np.arange(-1, 2),
                 alpha=np.geomspace(1e-3 * (i + 1), 1e3, 7), s_node=np.array(s), **extra)
    return folder


def test_store_asym_nodes_and_cache(tmp_path):
    """asym: absent in the files 0, the member when present, the override when given, files that disagree a ValueError;
    nodes= loads those nodes only; the unbiased cache holds the tables of unbias bit for bit, mapped (mmap=True) or read
    (pinned or not), with the same header; it needs unbiased=True, is not rewritten when complete, refuses a folder
    holding the cache of other files, and refuses to load once its source files changed"""
    import json
    import os
    from exmarg.search.store import is_cache, load_redshift_store, write_unbiased_cache
    src = write_store(str(tmp_path / 'store'))
    a = load_redshift_store(src, unbiased=True)
    assert a.asym == 0.0 and a.n_max == 120 and np.array_equal(a.s, [0.5, 2.0, 8.0])
    assert load_redshift_store(src, asym=-0.3).asym == -0.3
    assert load_redshift_store(write_store(str(tmp_path / 'w'), asym=-0.6)).asym == -0.6
    with pytest.raises(ValueError, match='different asym'):
        load_redshift_store(write_store(str(tmp_path / 'mixed'), asym=[0.0, -0.3, 0.0]))
    sub = load_redshift_store(src, unbiased=True, nodes=[2, 0])
    assert np.array_equal(sub.s, [0.5, 8.0]) and np.array_equal(sub.tables[1], a.tables[2])
    assert np.array_equal(sub.beta[1], a.beta[2]) and np.array_equal(sub.alpha_lo, a.alpha_lo[[0, 2]])
    with pytest.raises(ValueError, match='nodes must be'):
        load_redshift_store(src, nodes=[3])
    cache = str(tmp_path / 'cache')
    write_unbiased_cache(src, cache, readers=2)
    assert is_cache(cache) and json.load(open(os.path.join(cache, 'cache.json')))['source'] == os.path.abspath(src)
    for kw in (dict(mmap=True), dict(), dict(nodes=[1])):
        b = load_redshift_store(cache, unbiased=True, **kw)
        keep = kw.get('nodes', [0, 1, 2])
        assert np.array_equal(b.s, a.s[keep]) and b.n_max == a.n_max and b.asym == a.asym
        assert np.array_equal(b.alpha_lo, a.alpha_lo[keep]) and np.array_equal(b.alpha_hi, a.alpha_hi[keep])
        assert all(np.array_equal(x, a.beta[k]) and np.array_equal(y, a.delta[k]) for x, y, k in zip(b.beta, b.delta, keep))
        assert all(t.dtype == np.float32 and np.array_equal(t, a.tables[k]) for t, k in zip(b.tables, keep))
        assert np.array_equal(b.tau_idx, a.tau_idx)
        if kw.get('mmap'):
            assert all(isinstance(t.base, np.memmap) for t in b.tables)
    assert load_redshift_store(cache, unbiased=True, asym=-0.3).asym == -0.3
    with pytest.raises(ValueError, match='unbiased cache: load it with unbiased=True'):
        load_redshift_store(cache)
    with pytest.raises(ValueError, match='computed in memory'):
        load_redshift_store(src, mmap=True, unbiased=True)
    stamp = os.path.getmtime(os.path.join(cache, 'T_000.npy'))
    write_unbiased_cache(src, cache)
    assert os.path.getmtime(os.path.join(cache, 'T_000.npy')) == stamp
    with pytest.raises(ValueError, match='cache of other files'):
        write_unbiased_cache(write_store(str(tmp_path / 'other'), s_nodes=(1.0, 3.0)), cache)
    write_store(src, s_nodes=(32.0,))                                  # a node added to the source
    with pytest.raises(ValueError, match='written from other files'):
        load_redshift_store(cache, unbiased=True, mmap=True)
