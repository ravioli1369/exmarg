"""One conditioned week through the search in its final configuration, step by step.  The settings are fixed here, in
one place, so that every week of a run is analysed with the same commands:

    each detector on its own noise curve, with the signal gain of the conditioning       analyse.py --unequal
    the whole population of the paper, z from 0.02                                       --zmin 0.02
    the stores of the asymmetries w = 0, -0.30, -0.60, mapped from their unbiased caches <stores>, --mmap
    the tables with unbias and Lambda linear between their nodes and between the         the defaults of analyse.py
    templates                                                                            (its docstring, Tables)
    the window model: stage 1, the fit of each variant on its own data, stage 2          --stage windows, windows_fit.py
                                                                                         --rho-min 5 --record 1,
                                                                                         --glitch-model auto
    loudness cuts 72 and 84 in the fits (the defaults of windows_fit.py and plant_fit.py; the stores record the cuts
    they answer for and the fits refuse a selection beyond them)

    python scripts/real_data/week.py <step> <stores> <segments> --out <out> [--variants a,b,...] [--part i/n]
                                     [--device gpu] [--f-det f] [--recorded] [--dry-run]

    <stores>    the three caches, comma separated (store_cache.py); the step windows reads the header of the first
    <segments>  the folder of the conditioned epochs of the week (condition.py), with manifest.txt
    <step>      windows   stage 1 of the variants in one call (no table is read)
                fit       the glitch model of each variant at N = 0, <out>/<variant>/glitch_model.json
                merger    stage 2 of the variants, each with the model of its own data
                summary   summarize.py <out> --extra-slides [--f-det f]: the controls in their fixed order and their
                          criteria; the week as recorded only with --recorded
    --part i/n  of analyse.py (windows, merger): the i-th of n ranges of the week, one per GPU or machine
    --f-det f   the detected fraction of the loudness cuts (summary): measured with a merger injected into every
                segment of the slid week (analyse.py --stage windows --variants inj --nbar-inj 1) against the stage-1
                files of the slide itself

The default variants are the controls: the phase-randomized, time-reversed and time-slid weeks and the injections
(CONTROLS).  The week as recorded (the variant zero) is refused unless --recorded is given: it is opened last, by the
person who decides it, after the controls of the week have passed (docs/real_data.md).  The variants zero and inj hold
mergers; their glitch model is fitted again with the mergers at the rate stage 2 found (scripts/validate/iterate.py,
with --f-det), which this script does not do.

--dry-run prints the commands.  Exit status: that of the first command that fails.
"""
import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATE = os.path.join(os.path.dirname(HERE), 'validate')
CONTROLS = ('pr', 'rev', 'slide+0.5', 'slide-1.75', 'inj', 'slide+2.25', 'slide-3.0', 'slide+3.75')
RECORDED = 'zero'
STEPS = ('windows', 'fit', 'merger', 'summary')
ZMIN = '0.02'                                           # the first point of the redshift grid: the whole population
FIT = ['--rho-min', '5', '--record', '1']               # the fitted amplitude bins start at SNR 5; the record is kept


def commands(a):
    """the commands of the step, in order: lists of arguments"""
    py = [a.python]
    analyse = py + [os.path.join(HERE, 'analyse.py')]
    variants = ','.join(a.variants)
    if a.step == 'windows':
        return [analyse + [a.stores.split(',')[0], a.device, a.segments, '--out', a.out, '--stage', 'windows',
                           '--variants', variants, '--unequal', '--zmin', ZMIN, '--part', a.part]]
    if a.step == 'fit':
        return [py + [os.path.join(VALIDATE, 'windows_fit.py'), os.path.join(a.out, v)] + FIT for v in a.variants]
    if a.step == 'merger':
        return [analyse + [a.stores, a.device, a.segments, '--out', a.out, '--variants', variants, '--glitch-model',
                           'auto', '--unequal', '--mmap', '--zmin', ZMIN, '--part', a.part]]
    cmd = py + [os.path.join(HERE, 'summarize.py'), a.out, '--extra-slides']
    if a.f_det:
        cmd += ['--f-det', repr(float(a.f_det))]
    return [cmd + (['--recorded'] if a.recorded else [])]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=STEPS)
    p.add_argument('stores', help='the three unbiased caches (store_cache.py), comma separated')
    p.add_argument('segments', help='the folder of the conditioned epochs of the week')
    p.add_argument('--out', required=True, help='the folder of the week\'s results')
    p.add_argument('--variants', default=','.join(CONTROLS), help='default: the controls, %s' % ','.join(CONTROLS))
    p.add_argument('--part', default='0/1', help='i/n of analyse.py (windows, merger)')
    p.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    p.add_argument('--f-det', type=float, default=0.0, help='the detected fraction of the cuts (summary)')
    p.add_argument('--recorded', action='store_true', help='allow the week as recorded (the variant zero)')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    a.variants = list(dict.fromkeys(v.strip() for v in a.variants.split(',') if v.strip()))
    if not a.variants:
        p.error('no variant')
    if RECORDED in a.variants and not a.recorded:
        p.error('the week as recorded (zero) is analysed only with --recorded, after the controls have passed')
    if not 0 <= a.f_det < 1:
        p.error('need 0 <= f-det < 1')
    return a


def main(argv=None, log=print, run=subprocess.call):
    a = parse(argv)
    for cmd in commands(a):
        log(' '.join(shlex.quote(c) for c in cmd))
        sys.stdout.flush()
        if not a.dry_run:
            status = run(cmd)
            if status:
                return status
    return 0


if __name__ == '__main__':
    sys.exit(main())
