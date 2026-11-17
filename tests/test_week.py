"""scripts/real_data/week.py: the commands of one week in the final configuration.

The script only builds and runs commands, so the test reads them (--dry-run and a stand-in for subprocess.call): the
settings of the final configuration are in every command that needs them, the week as recorded is refused without
--recorded, the detected fraction reaches the summary, and a step stops at its first command that fails."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'real_data'))
import week

STORES = 'c/w-0.60,c/w-0.30,c/w0'


def dry(*argv):
    lines = []
    assert week.main(list(argv) + ['--dry-run', '--python', 'py'], log=lines.append) == 0
    return lines


def test_commands_of_the_final_configuration():
    v = ','.join(week.CONTROLS)
    a = os.path.join(ROOT, 'scripts', 'real_data', 'analyse.py')
    w, = dry('windows', STORES, 'segs', '--out', 'out', '--part', '1/2')
    assert w == 'py %s c/w-0.60 gpu segs --out out --stage windows --variants %s --unequal --zmin 0.02 --part 1/2' % (a, v)
    m, = dry('merger', STORES, 'segs', '--out', 'out')
    assert m == ('py %s %s gpu segs --out out --variants %s --glitch-model auto --unequal --mmap --zmin 0.02 --part 0/1'
                 % (a, STORES, v))
    fits = dry('fit', STORES, 'segs', '--out', 'out', '--variants', 'pr,slide+0.5,pr')
    f = os.path.join(ROOT, 'scripts', 'validate', 'windows_fit.py')
    assert fits == ['py %s %s --rho-min 5 --record 1' % (f, os.path.join('out', x)) for x in ('pr', 'slide+0.5')]
    s = os.path.join(ROOT, 'scripts', 'real_data', 'summarize.py')
    assert dry('summary', STORES, 'segs', '--out', 'out') == ['py %s out --extra-slides' % s]
    assert dry('summary', STORES, 'segs', '--out', 'out', '--f-det', '0.007') == [
        'py %s out --extra-slides --f-det 0.007' % s]
    assert week.RECORDED not in week.CONTROLS and 'inj' in week.CONTROLS


def test_the_recorded_week_needs_its_flag(capsys):
    for step in ('windows', 'fit', 'merger'):
        with pytest.raises(SystemExit):
            week.main([step, STORES, 'segs', '--out', 'out', '--variants', 'pr,zero', '--dry-run'])
    assert 'only with --recorded' in capsys.readouterr().err
    lines = dry('merger', STORES, 'segs', '--out', 'out', '--variants', 'zero', '--recorded')
    assert '--variants zero ' in lines[0]
    assert dry('summary', STORES, 'segs', '--out', 'out', '--recorded')[0].endswith('--extra-slides --recorded')
    assert not dry('summary', STORES, 'segs', '--out', 'out')[0].endswith('--recorded')
    with pytest.raises(SystemExit):
        week.main(['summary', STORES, 'segs', '--out', 'out', '--f-det', '1.0', '--dry-run'])


def test_stops_at_the_first_failure():
    ran = []

    def run(cmd):
        ran.append(cmd[2])
        return 3 if cmd[2].endswith('rev') else 0
    status = week.main(['fit', STORES, 'segs', '--out', 'out', '--variants', 'pr,rev,inj', '--python', 'py'],
                       log=lambda text: None, run=run)
    assert status == 3 and ran == [os.path.join('out', 'pr'), os.path.join('out', 'rev')]
