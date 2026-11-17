"""The two stages of the window model iterated (docs/2026-11-15_mergers_in_stage1.md, Sec. 3) on one campaign of plant.py
or one variant of analyse.py whose windows stage was run with the store (its win files hold the merger evidence):

    theta^(0) = stage 1 at N = 0;  N^(1) = stage 2 with theta^(0);  theta^(1) = stage 1 at N^(1);  N^(2) = ...

    python scripts/validate/iterate.py <win folder> <out> --passes 2 [--fit '--rho-min 5 --record 1'] \\
        -- python scripts/validate/plant.py tables/redshift gpu <the campaign options> --glitch-model {model} --out {out}
    python scripts/validate/iterate.py runs/week_w/zero runs/week_iter --passes 2 --results zero \\
        -- python scripts/real_data/analyse.py tables/redshift gpu data/segments/O3b_v2 --variants zero \\
           --glitch-model {model} --out {out}

Pass p = first, first + 1, .., passes, each in its own folder <out>/pass<p>:
  1. the glitch model of the win files fitted at N^(p-1) (N^(0) = --nbar0, default 0; then the estimate of the pass
     before): windows_fit.py <win folder> --nbar N^(p-1) --model <out>/pass<p>/glitch_model.json <--fit options>;
  2. stage 2 with that model: the command after --, {model} and {out} replaced by the model's path and <out>/pass<p>
     (plant.py and analyse.py skip the files present, so an interrupted pass is resumed by running the same command);
  3. N^(p): plant_fit.py's estimate of the window model on <out>/pass<p> (or its sub-folder --results, the variant of
     analyse.py), nbar maximizing sum ln(1 + N (Bbar - 1)) over the kept segments, with its error from the curvature.
--f-det f: the detected fraction of the loudness cuts enters every estimate (plant_fit.nbar_fit, the paper's Eq. 35).
Each pass writes <out>/pass<p>/estimate.json (nbar, err, the model's SHA-256 and nbar, f, the segments) when it is
complete, and a pass with that file is not run again: --first p with --nbar0 N^(p-1) starts at pass p (a pass 1 made
before, e.g. the analysis with the model at N = 0).  <out>/iteration.json records every pass.  --dry-run prints the
commands.  The fit and stage 2 are the drivers' own, with their checks: windows_fit.py refuses N > 0 for a control of
the data, plant.py and analyse.py a model fitted with the mergers of another population.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import plant
import plant_fit

ESTIMATE = 'estimate.json'
RECORD = 'iteration.json'


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def run(cmd, dry, log):
    log('%s  %s' % (stamp(), ' '.join(shlex.quote(c) for c in cmd)))
    if not dry:
        subprocess.run(cmd, check=True)


def estimate(results, veto_w, f_det=0.0):
    """nbar of the window model on the plant files of results (plant_fit.report, with the detected fraction f_det of
    the loudness cuts in the estimate, plant_fit.nbar_fit): dict nbar, err, segments, kept, model_sha256, model_nbar,
    f_det"""
    r = plant_fit.report(results, veto_w=veto_w, f_det=f_det)
    w = r['fits']['windows']
    return dict(nbar=float(w['nbar']), err=float(w['err']['nbar']), edge=w.get('edge'), segments=int(r['keep'].size),
                kept=int(r['keep'].sum()), model_sha256=r['cfg'].get('glitch_model_sha256'),
                model_nbar=float(r['cfg'].get('glitch_model_nbar', 0.0)), f_det=float(f_det))


def main(argv=None, log=print):
    argv = sys.argv[1:] if argv is None else list(argv)
    if '--' not in argv:
        raise SystemExit('give the command of stage 2 after --, with {model} and {out}')
    k = argv.index('--')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('win', help='the folder of the win files (windows stage with the store)')
    p.add_argument('out', help='the folder of the passes, <out>/pass<p>')
    p.add_argument('--passes', type=int, default=2, help='the last pass')
    p.add_argument('--first', type=int, default=1, help='the first pass run here')
    p.add_argument('--nbar0', type=float, default=0.0, help='N of the pass before the first (default 0)')
    p.add_argument('--fit', default='', help='options of windows_fit.py, as one string')
    p.add_argument('--results', default='', help='sub-folder of a pass that holds the plant files (analyse.py: the '
                                                 'variant)')
    p.add_argument('--veto-w', type=float, default=plant_fit.VETO_W, help='the loudness veto of the estimate')
    p.add_argument('--f-det', type=float, default=0.0,
                   help='the detected fraction of the loudness cuts in the estimate of every pass (plant_fit.nbar_fit; '
                        'measured on a plant with the same cuts).  Without it the N handed to the next pass is too '
                        'low by about n f sigma^2')
    p.add_argument('--python', default=sys.executable, help='the interpreter of windows_fit.py')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv[:k])
    stage2 = argv[k + 1:]
    if not stage2 or not any('{model}' in c for c in stage2) or not any('{out}' in c for c in stage2):
        p.error('the command of stage 2 must hold {model} and {out}')
    if not 1 <= a.first <= a.passes or not 0 <= a.nbar0 <= 1 or not 0 <= a.f_det < 1:
        p.error('need 1 <= first <= passes, 0 <= nbar0 <= 1 and 0 <= f-det < 1')
    os.makedirs(a.out, exist_ok=True)
    record_path = os.path.join(a.out, RECORD)
    record = json.load(open(record_path)) if os.path.exists(record_path) else dict(win=os.path.abspath(a.win), passes={})
    nbar = a.nbar0
    for n in range(a.first, a.passes + 1):
        folder = os.path.join(a.out, 'pass%d' % n)
        done = os.path.join(folder, ESTIMATE)
        if os.path.exists(done):
            est = json.load(open(done))
            if abs(est['model_nbar'] - nbar) > 1e-12 * max(1.0, nbar):
                raise SystemExit('%s was made with a model at nbar %r, not %r' % (done, est['model_nbar'], nbar))
            if abs(est.get('f_det', 0.0) - a.f_det) > 1e-15:
                raise SystemExit('%s was made with the detected fraction %r, not %r' % (done, est.get('f_det', 0.0),
                                                                                       a.f_det))
            log('%s  pass %d present: nbar %.6g +- %.2g (model at nbar %.6g)' % (stamp(), n, est['nbar'], est['err'],
                                                                                  nbar))
            record['passes'][str(n)] = est
            nbar = est['nbar']
            continue
        os.makedirs(folder, exist_ok=True)
        model = os.path.join(folder, 'glitch_model.json')
        if not os.path.exists(model):
            run([a.python, os.path.join(HERE, 'windows_fit.py'), a.win, '--nbar', repr(float(nbar)), '--model', model]
                + shlex.split(a.fit), a.dry_run, log)
        if not a.dry_run:
            m = plant.read_model(model)[0]
            if abs(plant.model_nbar(m) - nbar) > 1e-12 * max(1.0, nbar):
                raise SystemExit('%s is a model at nbar %g, not %g: remove it to fit again' % (model, plant.model_nbar(m),
                                                                                             nbar))
        run([c.replace('{model}', model).replace('{out}', folder) for c in stage2], a.dry_run, log)
        if a.dry_run:
            nbar = float('nan')
            continue
        est = dict(estimate(os.path.join(folder, a.results) if a.results else folder, a.veto_w, a.f_det),
                   model_path=model, finished=stamp())
        with open(done, 'w') as fh:
            json.dump(est, fh, indent=1)
        record['passes'][str(n)] = est
        with open(record_path, 'w') as fh:
            json.dump(record, fh, indent=1)
        log('%s  pass %d: model at nbar %.6g, estimate nbar %.6g +- %.2g on %d kept segments' % (
            stamp(), n, nbar, est['nbar'], est['err'], est['kept']))
        nbar = est['nbar']
    return 0


if __name__ == '__main__':
    sys.exit(main())
