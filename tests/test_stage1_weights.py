"""
The weighted stage 1 (mixture.fit_windows with weights): the fit maximizes sum_i w_i ln c_i.  With w_i = 1 - p_i, p_i the
probability that segment i holds a merger (stage 2), sum_i (1 - p_i) d ln c_i / d theta has the mean of the score of the
no-merger density under itself, zero: the glitch model is fitted on the part of the data that holds no merger.

1. weights of one change nothing
2. whole-number weights are repeats of the segments
3. the objective is sum_i w_i ln c_i, with its gradient and Hessian (finite differences)
4. a common factor of the weights changes neither the model nor its errors' ratio: ln L and the curvature scale with it
5. what is refused
"""
import numpy as np
import pytest

from exmarg.search import mixture

N, NW, NK, K = 300, 8, 4, 2


def evidences(seed=5, n=N):
    """ln E_w[k] (n, NW, NK) of a toy: noise windows (ln E about N(-0.5, 1), mean of E one), and in a tenth of the segments
    a few windows with a large evidence in one bin"""
    rng = np.random.default_rng(seed)
    lnE = rng.normal(-0.5, 1.0, (n, NW, NK))
    for i in np.flatnonzero(rng.uniform(size=n) < 0.1):
        for w in rng.choice(NW, 2, replace=False):
            lnE[i, w, rng.integers(NK)] += rng.uniform(4.0, 9.0)
    return lnE


def brute(lnE, f, Om, w):
    """sum_i w_i ln sum_s f_s prod_w (Om[s, 0] + Om[s, 1:] . E_iw)"""
    E = np.exp(lnE)
    D = Om[:, 0][None, None, :] + np.einsum('iwk,sk->iws', E, Om[:, 1:])
    return float(np.sum(w * np.log(np.prod(D, axis=1) @ f)))


def point(seed=1):
    rng = np.random.default_rng(seed)
    f = rng.dirichlet(np.ones(K))
    Om = rng.dirichlet(np.ones(NK + 1), size=K)
    Om[:, 0] += 3.0
    Om /= Om.sum(axis=1, keepdims=True)
    return np.concatenate([f, Om.ravel()])


def test_weights_of_one_change_nothing():
    lnE = evidences()
    a, b = mixture.fit_windows(lnE), mixture.fit_windows(lnE, weights=np.ones(N))
    assert a['converged'] and b['converged'] and a['n_iter'] == b['n_iter']
    for key in ('f', 'q', 'pi', 'a', 'lnl', 'cov', 'err_f', 'err_q', 'err_a', 'trace'):
        np.testing.assert_allclose(b[key], a[key], rtol=1e-12, atol=1e-14, err_msg=key)


def test_whole_number_weights_are_repeats():
    lnE = evidences()
    w = np.random.default_rng(2).integers(0, 4, N)
    a = mixture.fit_windows(lnE, weights=w.astype(float))
    b = mixture.fit_windows(np.repeat(lnE, w, axis=0))
    assert a['converged'] and b['converged']
    np.testing.assert_allclose(a['lnl'], b['lnl'], rtol=1e-10)
    for key in ('f', 'q', 'pi'):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-9, err_msg=key)
    for key in ('err_f', 'err_q'):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-4, atol=1e-9, err_msg=key)


def test_weighted_objective_gradient_and_hessian():
    lnE = evidences(n=60)
    w = np.random.default_rng(3).uniform(0.0, 1.5, 60)
    w[:5] = 0.0
    model = mixture._Windows(lnE, np.eye(NK), True, 16, weights=w)
    x = point()
    ev = model(x, K, hess=True)
    f, Om = x[:K], x[K:].reshape(K, NK + 1)
    np.testing.assert_allclose(ev['lnl'], brute(lnE, f, Om, w), rtol=1e-12)
    g, mu = model.gradient(ev)
    H = model.hessian(ev)
    lnl = lambda y: brute(lnE, y[:K], y[K:].reshape(K, NK + 1), w)
    grad = lambda y: model.gradient(model(y, K))[0]
    h = 1e-6
    for j in range(x.size):
        e = np.zeros(x.size)
        e[j] = h
        np.testing.assert_allclose(g[j], (lnl(x + e) - lnl(x - e)) / (2 * h), rtol=2e-6, atol=1e-6, err_msg='gradient %d' % j)
        np.testing.assert_allclose(H[:, j], (grad(x + e) - grad(x - e)) / (2 * h), rtol=2e-5, atol=2e-4, err_msg='Hessian %d' % j)
    np.testing.assert_allclose(mu[:K], w.sum())                       # f . g = sum of the weights: the EM step keeps f on its simplex
    np.testing.assert_allclose(f @ g[:K], w.sum(), rtol=1e-12)
    for s in range(K):                                                # Om[s] . g = n_w sum_i w_i r_is
        blk = slice(K + s * (NK + 1), K + (s + 1) * (NK + 1))
        np.testing.assert_allclose(x[blk] @ g[blk], mu[blk][0], rtol=1e-12)


def test_common_factor_of_the_weights():
    lnE = evidences()
    w = np.random.default_rng(4).uniform(0.2, 1.0, N)
    a, b = mixture.fit_windows(lnE, weights=w), mixture.fit_windows(lnE, weights=3.0 * w)
    assert a['converged'] and b['converged']
    np.testing.assert_allclose(b['lnl'], 3.0 * a['lnl'], rtol=1e-9)
    for key in ('f', 'q', 'pi'):
        np.testing.assert_allclose(b[key], a[key], rtol=1e-5, atol=1e-8, err_msg=key)
    np.testing.assert_allclose(b['err_f'], a['err_f'] / np.sqrt(3.0), rtol=1e-3)


def test_weights_refused():
    lnE = evidences(n=40)
    for bad in (np.ones(39), -np.ones(40), np.full(40, np.nan), np.zeros(40), np.ones((40, 1))):
        with pytest.raises(ValueError):
            mixture.fit_windows(lnE, weights=bad)
    with pytest.raises(ValueError):                                   # the two accounts of the mergers do not combine
        mixture.fit_windows(lnE, weights=np.ones(40), ln_M=np.zeros((40, NW)), nbar=0.1)


# ---------------------------------------------------------------------------------------------------------------------
# windows_fit.py --weights: the weights from the merger-stage files of a pass, and the model that records them

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts', 'validate'))
_spec = importlib.util.spec_from_file_location('twf', os.path.join(HERE, 'test_windows_fit.py'))
twf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(twf)
plant, windows_fit = twf.plant, twf.windows_fit
NS, NB = 1500, 0.1


def stage2(folder, cfg, lnB, splits, sha='ab' * 32, first=0):
    """merger-stage files of the campaign cfg holding k and ln_B_bar (what the weights read)"""
    os.makedirs(folder, exist_ok=True)
    for a, b in splits:
        plant.write_batch(os.path.join(folder, 'plant_%010d_%010d.npz' % (a, b)), dict(cfg, glitch_model_sha256=sha),
                          dict(first=a, end=b), dict(k=np.arange(a, b), ln_B_bar=lnB[a - first:b - first]))


@pytest.fixture(scope='module')
def passes(tmp_path_factory):
    """a toy campaign of the windows stage (test_windows_fit) and ln B of its segments: most near 1, some large, and
    the special values B = 1, B infinite and B = 0"""
    root = tmp_path_factory.mktemp('weights')
    folder = str(root / 'states')
    cfg, sim = twf.config(), twf.simulate(NS, 11)
    twf.write(folder, cfg, sim, ((0, 700), (700, NS)))
    rng = np.random.default_rng(12)
    lnB = rng.normal(0.0, 0.3, NS)
    lnB[rng.choice(NS, 40, replace=False)] = rng.uniform(3.0, 40.0, 40)
    lnB[:3] = 0.0, np.inf, -np.inf
    return root, folder, cfg, lnB


def test_no_merger_weights(passes):
    root, folder, cfg, lnB = passes
    st2 = str(root / 'pass1')
    stage2(st2, cfg, lnB, ((0, 400), (400, NS)))
    w, info = windows_fit.no_merger_weights(st2, np.arange(NS), NB, cfg)
    B = np.exp(lnB[3:])
    np.testing.assert_allclose(w[3:], 1.0 - NB * B / (1.0 + NB * (B - 1.0)), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(w[:3], [1.0 - NB, 0.0, 1.0])
    assert info == dict(nbar=NB, stage2_model_sha256='ab' * 32, sum=float(w.sum()), below_half=int((w < 0.5).sum()))
    assert info['below_half'] >= 40
    sub = np.array([5, 900, 17])                                       # any order, any subset of the segments
    np.testing.assert_array_equal(windows_fit.no_merger_weights(st2, sub, NB, cfg)[0], w[sub])
    np.testing.assert_array_equal(windows_fit.no_merger_weights(st2, np.arange(NS), 0.0, cfg)[0], np.ones(NS))


def test_windows_fit_with_weights(passes):
    root, folder, cfg, lnB = passes
    st2 = str(root / 'pass1b')
    stage2(st2, cfg, lnB, ((0, NS),))
    plain = windows_fit.fit(folder, record=())
    r = windows_fit.fit(folder, record=(), weights=st2, weights_nbar=NB)
    assert r['ok'] and 'nbar' not in r['model'] and 'merger_population' not in r['model']
    w, info = windows_fit.no_merger_weights(st2, r['data']['k'][r['keep']], NB, cfg)
    assert r['model']['weights'] == info
    assert any(line.startswith('weights 1 - p') for line in r['lines'])
    for I in 'HL':
        E = windows_fit.evidences(sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.startswith('win_')),
                                  I, r['keep'], r['cols'])
        want = twf.mixture.fit_windows(E, weights=w)
        np.testing.assert_allclose(r['model'][I]['f'], want['f'], rtol=1e-12)
        np.testing.assert_allclose(r['model'][I]['q'], want['q'], rtol=1e-12)
        assert not np.allclose(r['model'][I]['q'], plain['model'][I]['q'], rtol=1e-6)        # the weights act
    one = windows_fit.fit(folder, record=(), weights=st2, weights_nbar=0.0)              # weights of one: the plain model
    for I in 'HL':
        for key in ('f', 'q', 'pi_k'):
            np.testing.assert_allclose(one['model'][I][key], plain['model'][I][key], rtol=1e-10, atol=1e-14)
    assert plant.model_sha256(r['model']) != plant.model_sha256(plain['model'])


def test_windows_fit_weights_refused(passes):
    root, folder, cfg, lnB = passes
    with pytest.raises(ValueError):
        windows_fit.fit(folder, record=(), weights=str(root / 'pass1'), weights_nbar=None)
    with pytest.raises(ValueError):
        windows_fit.fit(folder, record=(), nbar=0.05, weights=str(root / 'pass1'), weights_nbar=NB)
    with pytest.raises(SystemExit, match='no merger-stage files'):
        windows_fit.fit(folder, record=(), weights=str(root / 'nothing'), weights_nbar=NB)
    part = str(root / 'part')
    stage2(part, cfg, lnB[:800], ((0, 800),))
    with pytest.raises(SystemExit, match='no merger-stage result for'):
        windows_fit.fit(folder, record=(), weights=part, weights_nbar=NB)
    other = str(root / 'other')
    stage2(other, dict(cfg, seed=cfg['seed'] + 1), lnB, ((0, NS),))
    with pytest.raises(SystemExit, match='another campaign'):
        windows_fit.fit(folder, record=(), weights=other, weights_nbar=NB)
    two = str(root / 'two')
    stage2(two, cfg, lnB, ((0, 800),))
    stage2(two, cfg, lnB[800:], ((800, NS),), sha='cd' * 32, first=800)
    with pytest.raises(SystemExit, match='glitch models'):
        windows_fit.fit(folder, record=(), weights=two, weights_nbar=NB)
    for argv in (['--weights', part], ['--weights-nbar', '0.1'], ['--weights', part, '--weights-nbar', '0.1', '--nbar', '0.05']):
        with pytest.raises(SystemExit):
            windows_fit.main([folder] + argv)
