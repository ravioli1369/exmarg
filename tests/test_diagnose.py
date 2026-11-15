"""scripts/real_data/diagnose.py segments: the two statistics of the recorded H1-L1 pair (cross-correlation of the
whitened bins, coincidence of the loudest times) are computed with L shifted against H unless --recorded is given"""
import importlib.util
import os
import re

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('exmarg_diagnose', os.path.join(ROOT, 'scripts', 'real_data', 'diagnose.py'))
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

T, NSEG = 8.0, 40
F = np.arange(160, 4097) / T


def common_noise(seed=1):
    """the same whitened bins in H and L: a recorded pair that is perfectly correlated (a real coincidence)"""
    rng = np.random.default_rng(seed)
    e = (rng.standard_normal((NSEG, F.size)) + 1j * rng.standard_normal((NSEG, F.size))) / np.sqrt(2)
    return {'H': e, 'L': e.copy()}


def correlation(S, keep):
    return np.abs(S['all', 'hl', 'HL'][keep].sum()) / np.sum(np.sqrt(S['all', 's1', 'H'][keep] * S['all', 's1', 'L'][keep]))


def sums(e, shift):
    diag._G.clear()
    diag._G.update(f=F, shift=shift)
    return diag._sums(e, {'all': {I: np.ones(NSEG, bool) for I in 'HL'}})


def test_cross_correlation_is_of_the_shifted_pair_by_default():
    """L = H: unshifted the correlation is 1; shifted by the default (circular, as controls.time_slide) it is what
    independent noise gives (about 1 / sqrt(n_segment n_bin) = 0.0025), whichever sign"""
    e, keep = common_noise(), np.ones(F.size, bool)
    assert abs(correlation(sums(e, 0.0), keep) - 1.0) < 1e-12
    for shift in (diag.DEFAULT_SLIDE, -diag.DEFAULT_SLIDE, 0.5):
        assert correlation(sums(e, shift), keep) < 0.02, shift


def test_default_shift_cannot_keep_a_coincidence():
    assert abs((diag.DEFAULT_SLIDE + T / 2) % T - T / 2) >= 0.5


def test_coincidence_of_the_loudest_times_is_of_the_shifted_pair():
    """loudest times equal in H and L (a glitch or event seen at the same time by both): coincident unshifted, none
    once L is shifted, at any sampling; independent random times keep their chance rate with the shift"""
    n_time = 8 * 4096
    rng = np.random.default_rng(3)
    j = rng.integers(0, n_time, 500)
    assert diag.coincident(j, j, n_time, T).all()
    for shift in (diag.DEFAULT_SLIDE, -diag.DEFAULT_SLIDE, 0.5, 4.0):
        assert not diag.coincident(j, j, n_time, T, shift).any(), shift
    k = rng.integers(0, n_time, 20000)
    chance = 2 * diag.COINCIDENCE / T
    assert abs(diag.coincident(j.repeat(40), k, n_time, T, diag.DEFAULT_SLIDE).mean() - chance) < 0.003
    # the shifted L of a pair whose loudest L time lies shift earlier is back on H: the shift is circular
    assert diag.coincident(j, (j - int(round(diag.DEFAULT_SLIDE / T * n_time))) % n_time, n_time, T,
                           diag.DEFAULT_SLIDE).all()


def test_options():
    """--recorded is opt-in and its help says that it opens the recorded data; a slide closer than 0.5 s to a multiple
    of the segment length is refused (SystemExit, naming the minimum), also when it is a whole number of segments"""
    a = diag.parser().parse_args(['segments', 'x'])
    assert a.recorded is False and diag.pair_shift(a, T) == diag.DEFAULT_SLIDE
    r = diag.parser().parse_args(['segments', 'x', '--recorded'])
    assert r.recorded and diag.pair_shift(r, T) == 0.0
    assert 'opens the recorded data' in re.sub(r'\s+', ' ', diag.parser().format_help())
    for bad in ('0.1', '-0.4', '8.0', '16.2'):
        with pytest.raises(SystemExit, match='at least 0.5'):
            diag.pair_shift(diag.parser().parse_args(['segments', 'x', '--slide', bad]), T)
    assert diag.pair_shift(diag.parser().parse_args(['segments', 'x', '--slide', '-1.75']), T) == -1.75
