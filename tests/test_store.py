"""the unbiased tables: linear interpolation of a convex function overshoots by h^2 T'' / 12 on average, and unbias()
removes exactly that (constructed functions, no store needed), while the average of exp(lnS) over a cell, which the
likelihood takes, is closer to the truth without it where lnS is steep (why the search reads the tables as built); and
the loading of a store folder of small random tables: asym, nodes, the caches of both kinds."""
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
from scipy.special import i0e, logsumexp

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


def cell_errors(f, grid, table, n_point=1000):
    """per interior cell [grid[k], grid[k + 1]] (k = 1 .. n - 3, so that both nodes are interior along the axis):
    ln of the average of exp of the linear interpolation of table over n_point points uniform in the cell (the
    midpoints of n_point equal parts) minus ln of the average of exp(f) over the same points"""
    u = (np.arange(n_point) + 0.5) / n_point
    out = []
    for k in range(1, grid.size - 2):
        exact = logsumexp(f(grid[k] + u * (grid[k + 1] - grid[k])))
        out.append(logsumexp((1.0 - u) * table[k] + u * table[k + 1]) - exact)
    return np.array(out)


def test_cell_average_of_exp_is_closer_without_unbias_where_lnS_is_steep():
    """why the search reads the tables as built.  unbias() gives the interpolated lnS no mean error over a cell, but
    the likelihood averages S = exp(lnS).  The table: lnS(beta) = ln sum_n e^(n beta) / (n!)^2 = ln I_0(2 e^(beta / 2)),
    the form of a lag sum at fixed lag, phi and Delta (a power series in e^beta with coefficients >= 0, convex), on 100
    nodes beta = -4 + 0.139 j (the beta step of the tables), lnS from 0.02 to 260, as a table (1, 3, 100, 3) constant
    along phi and Delta and passed through store.unbias.  Per interior cell, the average of exp of the linear
    interpolation over 1000 points uniform in the cell against the average of exp(lnS) over the same points, in ln
    (97 interior cells):
      steep cells, where lnS rises by 12 or more across the cell (lnS from 170; 5 cells): as built +0.029 to +0.031,
      with unbias -0.045 to -0.068: as built is closer in every one;
      shallow cells, where lnS rises by less than 1 (lnS below 12; 57 cells): as built +4e-5 to +0.006, with unbias
      1.3e-4 at most in size: with unbias closer in every one, the regime unbias was derived for (the control).
    As built is closer in exactly the cells where lnS rises by more than 9 (the cell from lnS 119, rise 8.8, is the
    last where unbias is closer; the next, from lnS 128, rise 9.4, the first where it is not).  Fires on: unbias()
    returning its table unchanged (both errors then equal)"""
    h = 0.139
    lnS = lambda b: np.log(i0e(2.0 * np.exp(0.5 * b))) + 2.0 * np.exp(0.5 * b)
    beta = -4.0 + h * np.arange(100)
    T = np.ascontiguousarray(np.broadcast_to(lnS(beta)[None, None, :, None], (1, 3, beta.size, 3)), np.float32)
    built = cell_errors(lnS, beta, T[0, 1, :, 1].astype(float))
    unb = cell_errors(lnS, beta, unbias(T)[0, 1, :, 1].astype(float))
    rise = np.diff(lnS(beta))[1:-1]
    steep, shallow = rise >= 12.0, rise < 1.0
    assert built.size == 97 and steep.sum() == 5 and shallow.sum() == 57
    assert np.all(np.abs(built[steep]) < np.abs(unb[steep])), (built[steep], unb[steep])
    assert 0.029 < built[steep].min() and built[steep].max() < 0.031, built[steep]
    assert -0.069 < unb[steep].min() and unb[steep].max() < -0.045, unb[steep]
    assert np.all(np.abs(unb[shallow]) < np.abs(built[shallow])), (built[shallow], unb[shallow])
    assert np.abs(unb[shallow]).max() < 2e-4 and 0.0 < built[shallow].min() and built[shallow].max() < 0.006
    assert np.array_equal(np.abs(built) < np.abs(unb), rise > 9.0)


# ----------------------------------------------------------------------------------------------- loading


def write_store(folder, s_nodes=(0.5, 2.0, 8.0), asym=None, seed=0, cuts=None):
    """a store folder of bd_*.npz files with the members load_redshift_store reads (asym only when given: a list, one
    per file, or one value; cuts, (zcut, wnet) or a list of them, one per file: the members repair_zcut, repair_wnet
    of build_ratio.py)"""
    import os
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, s in enumerate(s_nodes):
        extra = {} if asym is None else dict(asym=np.array(asym[i] if np.ndim(asym) else asym))
        if cuts is not None:
            zc, wn = cuts[i] if np.ndim(cuts) == 2 else cuts
            extra.update(repair_zcut=np.array(float(zc)), repair_wnet=np.array(float(wn)))
        np.savez(os.path.join(folder, 'bd_s%07.3f_n120_x.npz' % s), T=rng.normal(size=(3, 49, 6, 5)).astype(np.float32),
                 beta=np.linspace(-2.0, 3.0, 6) + np.log(s), delta=np.linspace(-1.0, 1.0, 5), tau_idx=np.arange(-1, 2),
                 alpha=np.geomspace(1e-3 * (i + 1), 1e3, 7), s_node=np.array(s), **extra)
    return folder


def test_store_asym_nodes_and_cache(tmp_path):
    """asym: absent in the files 0, the member when present, the override when given, files that disagree a ValueError;
    nodes= loads those nodes only; the unbiased cache (write_cache(unbiased=True)) holds the tables of unbias bit for
    bit, mapped (mmap=True) or read (pinned or not), with the same header; it loads as unbiased without a request and
    with unbiased=True, refuses unbiased=False, is not rewritten when complete, refuses a folder holding the cache of
    other files, and refuses to load once its source files changed"""
    import json
    import os
    from exmarg.search.store import is_cache, load_redshift_store, write_cache
    src = write_store(str(tmp_path / 'store'))
    a = load_redshift_store(src, unbiased=True)
    assert a.unbiased and not load_redshift_store(src).unbiased
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
    write_cache(src, cache, readers=2, unbiased=True)
    rec = json.load(open(os.path.join(cache, 'cache.json')))
    assert is_cache(cache) and rec['source'] == os.path.abspath(src) and rec['unbiased'] is True
    for kw in (dict(mmap=True), dict(), dict(nodes=[1]), dict(unbiased=True), dict(mmap=True, unbiased=True)):
        b = load_redshift_store(cache, **kw)
        keep = kw.get('nodes', [0, 1, 2])
        assert b.unbiased and np.array_equal(b.s, a.s[keep]) and b.n_max == a.n_max and b.asym == a.asym
        assert np.array_equal(b.alpha_lo, a.alpha_lo[keep]) and np.array_equal(b.alpha_hi, a.alpha_hi[keep])
        assert all(np.array_equal(x, a.beta[k]) and np.array_equal(y, a.delta[k]) for x, y, k in zip(b.beta, b.delta, keep))
        assert all(t.dtype == np.float32 and np.array_equal(t, a.tables[k]) for t, k in zip(b.tables, keep))
        assert np.array_equal(b.tau_idx, a.tau_idx)
        if kw.get('mmap'):
            assert all(isinstance(t.base, np.memmap) for t in b.tables)
    assert load_redshift_store(cache, unbiased=True, asym=-0.3).asym == -0.3
    with pytest.raises(ValueError, match='cache of the tables unbiased'):
        load_redshift_store(cache, unbiased=False)
    with pytest.raises(ValueError, match='computed in memory'):
        load_redshift_store(src, mmap=True, unbiased=True)
    stamp = os.path.getmtime(os.path.join(cache, 'T_000.npy'))
    write_cache(src, cache, unbiased=True)
    assert os.path.getmtime(os.path.join(cache, 'T_000.npy')) == stamp
    with pytest.raises(ValueError, match='cache of other files'):
        write_cache(write_store(str(tmp_path / 'other'), s_nodes=(1.0, 3.0)), cache, unbiased=True)
    write_store(src, s_nodes=(32.0,))                                  # a node added to the source
    with pytest.raises(ValueError, match='written from other files'):
        load_redshift_store(cache, unbiased=True, mmap=True)


def test_store_records_the_cuts_it_was_repaired_for(tmp_path):
    """read_header: cuts None for files without the members (the published store) and for a store built without stage R
    (repair_zcut 0), (zcut, wnet) otherwise, a ValueError when the files disagree; caches of both kinds carry them"""
    from exmarg.search.store import read_header, write_cache
    assert read_header(write_store(str(tmp_path / 'plain')))['cuts'] is None
    assert read_header(write_store(str(tmp_path / 'norepair'), cuts=(0.0, 0.0)))['cuts'] is None
    src = write_store(str(tmp_path / 'r'), cuts=(72.0, 84.0))
    assert read_header(src)['cuts'] == (72.0, 84.0)
    assert read_header(write_store(str(tmp_path / 'nonet'), cuts=(72.0, np.inf)))['cuts'] == (72.0, np.inf)
    with pytest.raises(ValueError, match='different loudness cuts'):
        read_header(write_store(str(tmp_path / 'mixed'), cuts=[(72.0, 84.0), (72.0, 84.0), (50.0, 84.0)]))
    assert read_header(write_cache(src, str(tmp_path / 'cache'), readers=2))['cuts'] == (72.0, 84.0)
    assert read_header(write_cache(src, str(tmp_path / 'cache_u'), readers=2, unbiased=True))['cuts'] == (72.0, 84.0)
    plain = write_cache(str(tmp_path / 'plain'), str(tmp_path / 'cache_plain'), readers=2)
    assert read_header(plain)['cuts'] is None


def test_cache_as_built_is_the_store_bit_for_bit(tmp_path):
    """write_cache with its default writes the tables as built: every T_<k>.npy holds the member T of the store file
    of node k byte for byte (float32, the same shape), cache.json records unbiased false, and the cache loads back,
    read or mapped, with or without the request unbiased=False, as a store of unbiased False whose tables are, byte for
    byte, those of the store folder loaded by default (as built), with the same header.  A complete cache is not
    rewritten, and the unbiased cache of the same files is refused in its folder.  Fires on: write_cache passing the
    tables through unbias() whatever its unbiased (the bytes differ, and the cache then loads as unbiased False all
    the same: the flag alone does not show it)"""
    import json
    import os
    from exmarg.search.store import is_cache, load_redshift_store, read_header, read_npz_member, write_cache
    src = write_store(str(tmp_path / 'store'), cuts=(72.0, 84.0))
    cache = write_cache(src, str(tmp_path / 'cache'), readers=2)
    rec = json.load(open(os.path.join(cache, 'cache.json')))
    assert is_cache(cache) and rec['unbiased'] is False and rec['source'] == os.path.abspath(src)
    for k, path in enumerate(read_header(src)['files']):
        T = read_npz_member(path, 'T.npy')
        C = np.load(os.path.join(cache, 'T_%03d.npy' % k))
        assert T.dtype == C.dtype == np.float32 and T.shape == C.shape and T.tobytes() == C.tobytes(), k
    a = load_redshift_store(src)
    assert a.unbiased is False
    for kw in (dict(), dict(mmap=True), dict(unbiased=False), dict(mmap=True, unbiased=False), dict(nodes=[2, 0])):
        b = load_redshift_store(cache, **kw)
        keep = sorted(kw.get('nodes', [0, 1, 2]))
        assert b.unbiased is False and np.array_equal(b.s, a.s[keep]) and b.n_max == a.n_max and b.asym == a.asym
        assert all(t.tobytes() == a.tables[k].tobytes() for t, k in zip(b.tables, keep))
        assert all(np.array_equal(x, a.beta[k]) and np.array_equal(y, a.delta[k])
                   for x, y, k in zip(b.beta, b.delta, keep))
        assert np.array_equal(b.alpha_lo, a.alpha_lo[keep]) and np.array_equal(b.tau_idx, a.tau_idx)
        if kw.get('mmap'):
            assert all(isinstance(t.base, np.memmap) for t in b.tables)
    assert read_header(cache)['cuts'] == (72.0, 84.0)
    stamp = os.path.getmtime(os.path.join(cache, 'T_000.npy'))
    assert write_cache(src, cache) == cache and os.path.getmtime(os.path.join(cache, 'T_000.npy')) == stamp
    with pytest.raises(ValueError, match='same files with the tables as built, not unbiased'):
        write_cache(src, cache, unbiased=True)


def test_cache_of_the_other_kind_is_refused(tmp_path):
    """a request for the other kind of a cache is a ValueError, read or mapped, that names both kinds: unbiased=True of
    a cache as built, unbiased=False of an unbiased cache, and unbiased=False of an older unbiased cache whose
    cache.json has no unbiased entry (written before both kinds existed), which loads as unbiased without a request
    and with unbiased=True.  Fires on: load_redshift_store taking a cache as its own kind whatever was asked"""
    import json
    import os
    from exmarg.search.store import load_redshift_store, write_cache
    src = write_store(str(tmp_path / 'store'))
    built = write_cache(src, str(tmp_path / 'built'), readers=2)
    unb = write_cache(src, str(tmp_path / 'unb'), readers=2, unbiased=True)
    old = write_cache(src, str(tmp_path / 'old'), readers=2, unbiased=True)
    rec = json.load(open(os.path.join(old, 'cache.json')))
    rec.pop('unbiased')
    json.dump(rec, open(os.path.join(old, 'cache.json'), 'w'))
    assert load_redshift_store(old).unbiased is True and load_redshift_store(old, unbiased=True).unbiased is True
    for mmap in (False, True):
        with pytest.raises(ValueError, match='cache of the tables as built .*, not unbiased as asked'):
            load_redshift_store(built, mmap=mmap, unbiased=True)
        for cache in (unb, old):
            with pytest.raises(ValueError, match='cache of the tables unbiased .*, not as built as asked'):
                load_redshift_store(cache, mmap=mmap, unbiased=False)
