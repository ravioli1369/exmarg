"""the two readings of the node values in pe._to_source_frame, between the nodes of the tables in y and between the
templates in M_det.  interp 'log', ln Lambda linear between two neighbours, is the function as it was up to 2026-11-17
(copied here), bit for bit.  interp 'linear', Lambda linear, is ln((1 - t) e^v0 + t e^v1) in both steps, the node value
at a node, and takes the same nodes and the same range as before.  Why it is the default: two likelihood ratios
combined with the weights 1 - t and t are a likelihood ratio, of mean 1 under noise; their geometric mean is not, and
for exp(A x - A^2 / 2), x ~ N(0, 1), at two amplitudes dA apart its mean is exp(-t (1 - t) dA^2 / 2).  Every grid of
pe_redshift takes the interp it is given (the small store of test_pe_glitch.py)."""
import numpy as np
import pytest

from exmarg.search import pe
from exmarg.search.segments import D0
from test_pe_glitch import REDSHIFT, case, node_values  # noqa: F401  (case is a fixture)
from test_pe_windows import WINDOWS


def before(lamN, yN, Mdet, yq, m1, z, cp):
    """pe._to_source_frame of commit 9571a6c, before interp"""
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


def cases(n=120, seed=3):
    """n sets of arguments of _to_source_frame: 1 to 8 nodes and the boundary row, 1 to 11 templates, values -inf and
    nodes left out (y = 0) at random, so that templates have two finite nodes, one or none, in every seventh case a
    template with the same y at every node; 40 redshifts with y between, at and outside the nodes, 32 masses m1 with
    M_det between, at and outside the templates"""
    rng = np.random.default_rng(seed)
    for i in range(n):
        nn, nm = int(rng.integers(1, 9)), int(rng.integers(1, 12))
        M = np.sort(rng.uniform(20.0, 400.0, nm))
        Y = rng.uniform(0.05, 3.0, (nn + 1, nm)) * (rng.uniform(size=(nn + 1, nm)) > 0.15)
        L = rng.normal(0.0, 30.0, (nn + 1, nm))
        L[rng.uniform(size=L.shape) < 0.15] = -np.inf
        if i % 7 == 0:
            Y[:, 0] = Y[0, 0]
        z = np.linspace(0.02, 10.0, 40)
        yq = np.concatenate([rng.uniform(0.01, 3.2, 36), Y[0, :1], Y.max(axis=0)[:1], np.resize(Y[Y > 0], 2)])
        m1 = np.concatenate([np.geomspace(4.0, 200.0, 30), M[:1] / 2.0 / (1.0 + z[0]), M[-1:] / 2.0 / (1.0 + z[3])])
        yield L, Y, M, yq, m1, z


def test_log_is_the_function_of_before_bit_for_bit():
    """interp 'log' against the copy of the function before the keyword, on the random cases: the same bytes; and
    another interp is a ValueError, of _to_source_frame and, before any work, of pe_redshift_epochs.  Fires on: the
    'log' expression written in another order (v0 (1 - t) + t v1 differs in the last bit)"""
    n = 0
    with np.errstate(invalid="ignore"):                 # -inf - (-inf) in a template without a node, masked in both
        for args in cases():
            got, want = pe._to_source_frame(*args, np, interp="log"), before(*args, np)
            assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes()
            n += int(np.isfinite(got).sum())
    assert n > 20000
    for bad in ("geometric", "lin", None):
        with pytest.raises(ValueError, match="interp"):
            pe._to_source_frame(*args, np, interp=bad)
        with pytest.raises(ValueError, match="interp"):
            pe.pe_redshift_epochs([], None, interp=bad)


def blend(v0, v1, t):
    return np.log((1.0 - t) * np.exp(v0) + t * np.exp(v1))


def test_linear_is_the_closed_form_in_both_steps():
    """two nodes (and the boundary row, unused) and two templates, by hand.  Template 0 (M_det 50) has the nodes y =
    0.5, 0.57 with ln Lambda = 1, 3.5, template 1 (M_det 100) the nodes y = 0.4, 0.8 with -2, 0.5.  At y = 0.52, 0.5
    and 0.57, which both reach, the surface of template m is S_m = ln((1 - t_m) e^v0 + t_m e^v1), t_0 = 2/7, 0, 1 and
    t_1 = 0.3, 0.25, 0.425; at M_det = 60 and 90 the result is ln((1 - t) e^S0 + t e^S1), t = 0.2 and 0.8, at M_det =
    50 and 100 S0 and S1 themselves, and at the nodes y = 0.5, 0.57 of template 0 with M_det = 50 the node values 1
    and 3.5 exactly.  M_det = 40 and 120 (outside the templates) and y = 0.45 (template 1 alone reaches it), 0.3 and
    0.9 (neither does) give -inf, where 'log' does.  The default is 'linear'.  Fires on: Lambda linear in one of the
    two steps only"""
    Y = np.array([[0.5, 0.4], [0.57, 0.8], [0.0, 0.0]])
    L = np.array([[1.0, -2.0], [3.5, 0.5], [-np.inf, -np.inf]])
    M = np.array([50.0, 100.0])
    yq = np.array([0.52, 0.5, 0.57, 0.45, 0.3, 0.9])
    m1 = np.array([20.0, 25.0, 30.0, 45.0, 50.0, 60.0])            # M_det = 40, 50, 60, 90, 100, 120 at z = 0
    args = (L, Y, M, yq, m1, np.zeros(yq.size), np)
    got = pe._to_source_frame(*args, interp="linear")              # (m1, z)
    assert got.tobytes() == pe._to_source_frame(*args).tobytes()
    assert np.array_equal(np.isneginf(got), np.isneginf(pe._to_source_frame(*args, interp="log")))
    assert np.all(np.isneginf(got[:, 3:])) and np.all(np.isneginf(got[[0, 5]])) and np.all(np.isfinite(got[1:5, :3]))
    t0, t1 = np.array([(0.52 - 0.5) / (0.57 - 0.5), 0.0, 1.0]), (yq[:3] - 0.4) / (0.8 - 0.4)
    S0, S1 = blend(1.0, 3.5, t0), blend(-2.0, 0.5, t1)
    want = np.stack([S0, blend(S0, S1, 0.2), blend(S0, S1, 0.8), S1])
    assert np.abs(got[1:5, :3] - want).max() < 1e-13
    assert got[1, 1] == 1.0 and got[1, 2] == 3.5


def test_linear_takes_the_nodes_and_the_range_of_before():
    """a node of value -inf is left out, not weighted: with ln Lambda = 2, -inf, 5 at y = 0.2, 0.3, 0.6 of template 0
    the surface at y = 0.4 is ln(e^2 / 2 + e^5 / 2), between the two finite neighbours (template 2, with 2, 3, 5:
    ln(2 e^3 / 3 + e^5 / 3)); template 1, with one finite node, is left out of the interpolation in M_det, which runs
    between its neighbours (M_det 50 and 100: t = 0.5 at 75), and with fewer than two templates left the result is
    -inf.  On the random cases the result is finite exactly where that of 'log' is, and not below it (the arithmetic
    mean of two numbers against the geometric one).  At a node of a template, read at the mass of the template, it is
    the node value bit for bit (t = 0 or 1 in both steps)"""
    Y = np.array([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.0, 0.0, 0.0]])
    L = np.array([[2.0, -np.inf, 2.0], [-np.inf, -np.inf, 3.0], [5.0, 1.0, 5.0], [-np.inf, -np.inf, -np.inf]])
    M = np.array([50.0, 60.0, 100.0])
    y, z = np.array([0.4]), np.zeros(1)
    got = pe._to_source_frame(L, Y, M, y, np.array([25.0, 37.5, 50.0]), z, np)
    s0, s2 = blend(2.0, 5.0, 0.5), blend(3.0, 5.0, 1.0 / 3.0)
    assert np.abs(got[:, 0] - np.array([s0, blend(s0, s2, 0.5), s2])).max() < 1e-13
    assert np.all(np.isneginf(pe._to_source_frame(L[:, :2], Y[:, :2], M[:2], y, np.array([25.0, 27.0, 30.0]), z, np)))
    n = 0
    with np.errstate(invalid="ignore"):
        for args in cases():
            lin, log = pe._to_source_frame(*args, np), pe._to_source_frame(*args, np, interp="log")
            fin = np.isfinite(log)
            assert np.array_equal(np.isfinite(lin), fin) and np.array_equal(np.isneginf(lin), ~fin)
            assert np.all(lin[fin] >= log[fin] - 1e-9 * np.maximum(1.0, np.abs(log[fin])))
            n += int(fin.sum())
    assert n > 20000
    Y = np.array([[0.5, 0.4], [0.57, 0.8], [0.9, 1.0], [0.0, 0.0]])
    L = np.array([[1.25, -7.0], [3.5, 0.5], [-0.75, 2.0], [-np.inf, -np.inf]])
    node = pe._to_source_frame(L, Y, M[:2], Y[:3, 0], M[:1] / 2.0, np.zeros(3), np)
    assert np.array_equal(node[0], L[:3, 0])


def toy(a0, a1, n, seed):
    """ln Lambda(x; A) = A x - A^2 / 2 of n draws x ~ N(0, 1) at the amplitudes a0 and a1: (n,), (n,)"""
    x = np.random.default_rng(seed).standard_normal(n)
    return a0 * x - 0.5 * a0 * a0, a1 * x - 0.5 * a1 * a1


def check_means(lin, log, t, dA):
    """the means over the draws of Lambda read with 'linear' and with 'log' at the fraction t between two amplitudes
    dA apart: 1 and exp(-t (1 - t) dA^2 / 2), each within 4 standard errors; their difference, whose error is far
    smaller (the two readings of one draw differ little), is that deficit within 4 standard errors and is resolved at
    more than 10"""
    lin, log = np.exp(lin), np.exp(log)
    se = lambda v: v.std() / np.sqrt(v.size)
    low = np.exp(-0.5 * t * (1.0 - t) * dA * dA)
    assert abs(lin.mean() - 1.0) < 4.0 * se(lin), (lin.mean(), se(lin))
    assert abs(log.mean() - low) < 4.0 * se(log), (log.mean(), low, se(log))
    d = lin - log
    assert abs(d.mean() - (1.0 - low)) < 4.0 * se(d) and 1.0 - low > 10.0 * se(d), (d.mean(), 1.0 - low, se(d))
    return lin.mean(), log.mean(), low, se(lin)


def test_linear_keeps_the_mean_under_noise_and_log_lowers_it():
    """why Lambda and not ln Lambda is linear between the nodes.  The toy family Lambda(x; A) = exp(A x - A^2 / 2), x ~
    N(0, 1) under noise, has mean 1 at every A.  Two nodes at A = 1 and 1.14 (the amplitudes of neighbouring nodes of
    the tables are a factor 1.14 apart), read at the fraction t = 0.4 between them for 10^6 draws of x:
      interp 'linear'  (1 - t) Lambda_0 + t Lambda_1, the likelihood ratio of a population on the two nodes: mean 1;
      interp 'log'     Lambda_0^(1 - t) Lambda_1^t = exp(Abar x - ((1 - t) A0^2 + t A1^2) / 2) with Abar = (1 - t) A0 +
                       t A1: mean exp(-t (1 - t) dA^2 / 2) = 1 - 2.35e-3 here, and smaller as dA^2 grows.
    Once in each step of _to_source_frame, with the draws in place of the templates: between the nodes in y (template
    m holds draw m at the nodes y = A0, A1, read at its own M_det), and between the templates in M_det (the templates
    2 m and 2 m + 1 hold draw m at A0 and at A1, the same at both nodes in y, read at M_det a fraction t between them).
    Fires on: the two forms exchanged, or 'linear' in one step only"""
    a0, t, n = 1.0, 0.4, 1000000
    a1 = 1.14 * a0
    z = np.zeros(1)
    v0, v1 = toy(a0, a1, n, 11)
    Y = np.stack([np.full(n, a0), np.full(n, a1), np.zeros(n)])
    L = np.stack([v0, v1, np.full(n, -np.inf)])
    M = 1.0 + np.arange(n)
    args = (L, Y, M, np.array([a0 + t * (a1 - a0)]), M / 2.0, z, np)
    in_y = check_means(pe._to_source_frame(*args)[:, 0], pe._to_source_frame(*args, interp="log")[:, 0], t, a1 - a0)
    v0, v1 = toy(a0, a1, n // 2, 12)
    L = np.empty((3, n))
    L[:2, 0::2], L[:2, 1::2], L[2] = v0, v1, -np.inf
    args = (L, Y, M, np.array([0.5 * (a0 + a1)]), (M[0::2] + t) / 2.0, z, np)
    in_m = check_means(pe._to_source_frame(*args)[:, 0], pe._to_source_frame(*args, interp="log")[:, 0], t, a1 - a0)
    print("\nmean of Lambda under noise at t = %g between A = %g and %g: in y linear %.5f, log %.5f; in M_det linear "
          "%.5f, log %.5f; exp(-t (1 - t) dA^2 / 2) = %.5f (standard error of a mean %.1e)"
          % (t, a0, a1, in_y[0], in_y[1], in_m[0], in_m[1], in_y[2], in_y[3]))


@pytest.mark.parametrize("glitch", [True, WINDOWS], ids=["glitch", "windows"])
def test_every_grid_of_pe_redshift_takes_the_interp(case, glitch):
    """pe_redshift on the store of test_pe_glitch.py with the default glitch terms and with the window model: every
    grid of the result (ln_lambda, ln_lambda_H, ln_lambda_L and ln_lambda_gH, ln_lambda_gL, ln_lambda_gHL, or ln_B of
    the window model) is _to_source_frame of its node values with the interp given, bit for bit, 'linear' when none is
    given; the two interp differ in every grid; and pe_redshift_epochs gives the arrays of pe_redshift.  Fires on: a
    grid reduced without the keyword (it would stay 'linear' in a run with interp 'log')"""
    c = case
    lamN, one = node_values(c, glitch)
    gd = pe._glitch_setup(glitch, c["bank"], c["seg"], c["store"])
    nodes = [("ln_lambda", lamN)] + [(key, one[node]) for key, node in pe._node_keys(gd)]
    assert len(nodes) == (4 if glitch is WINDOWS else 6)
    kw = dict(n_loud=0, glitch=glitch, **REDSHIFT)
    res = {}
    for interp, extra in (("linear", {}), ("log", dict(interp="log"))):
        res[interp] = r = pe.pe_redshift(c["segs"], c["bank"], c["store"], **kw, **extra)
        yq = D0 / pe._dl_of_z(r["z"])
        for key, node in nodes:
            for e in range(3):
                want = pe._to_source_frame(node[e], c["geometry"][2], c["bank"].M, yq, r["m1"], r["z"], np, interp)
                assert np.array_equal(r[key][e], want), (interp, key, e)
    for key, _ in nodes:
        fin = np.isfinite(res["log"][key])
        assert fin.any() and np.array_equal(fin, np.isfinite(res["linear"][key]))
        assert np.abs(res["linear"][key][fin] - res["log"][key][fin]).max() > 1e-3, key
    both = pe.pe_redshift_epochs([dict(segments=c["segs"], bank=c["bank"])], c["store"], interp="log", **kw)[0]
    assert all(np.array_equal(both[key], res["log"][key]) for key in [key for key, _ in nodes] + ["ln_b"])
    assert not np.array_equal(res["log"]["ln_b"], res["linear"]["ln_b"])
