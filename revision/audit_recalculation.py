"""Read-only integrity and coverage audit of completed jobs in an ongoing revision.

Reports counts and selection provenance; test scores never enter audit decisions.
"""
import argparse
import ast
import collections
import contextlib
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import scipy


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def count_terms(memory, quintic):
    result = math.comb(memory+1, 2)*memory
    if quintic:
        result += math.comb(memory+2, 3)*memory**2 + math.comb(memory+1, 2)**2*memory
    return result


def audit(run):
    manifest = read(run/'manifest.json')
    cfg = manifest['config']
    jobs = [dict(repeat=r,case=c,pnr=n,method=m) for r in range(cfg['repeats'])
            for c in cfg['cases'] for n in cfg['pnr_db'] for m in cfg['methods']]
    progress = read(run/'progress.json')
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    for name, digest in manifest['source_sha256'].items():
        check(hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==digest,
              'Frozen source mismatch: '+name)
    check(cfg == read(run/'source/config.json'), 'Frozen config mismatch')
    seen = set()
    coverage = collections.Counter()
    candidates = collections.defaultdict(collections.Counter)
    boundary = collections.defaultdict(collections.Counter)
    memory_counts = collections.defaultdict(collections.Counter)
    grid_rows = {}
    trajectory_count = 0
    n_candidate_expected = 0
    missing_result_files = []
    sample_counts = set()
    successful = 0
    for outcome in progress:
        index = outcome['index']
        check(index not in seen, f'Duplicate index {index}')
        seen.add(index)
        check(0 <= index < len(jobs), f'Index out of range: {index}')
        if not 0 <= index < len(jobs):
            continue
        job = jobs[index]
        check(outcome['job']==job, f'Unexpected job definition {index}')
        if outcome['status'] != 'ok':
            continue
        folder = run/f'job_{index:04d}'
        required = ['job.json','selection.json','model.npz','validation.json','result.json']
        missing = [name for name in required if not (folder/name).exists()]
        if missing:
            missing_result_files.append(dict(index=index,missing=missing))
            errors.append(f'Missing job files {index}')
            continue
        successful += 1
        result = read(folder/'result.json')
        selection = read(folder/'selection.json')
        rows = read(folder/'validation.json')
        selected = result['selected']
        check(result['job']==job and selection['job']==job and read(folder/'job.json')==job,
              f'Job identity mismatch {index}')
        check(result['status']=='ok' and selected['converged'], f'Invalid selected status {index}')
        check(selection['chosen']==selected, f'Selection/result mismatch {index}')
        label = job['method']
        method = label.removeprefix('screen_')
        key = (job['case']['name'], str(job['pnr']), label)
        coverage[key] += 1
        sample_counts.add(tuple(selection[k] for k in ('fit_samples','validation_samples','test_samples')))
        check(all(selection[k]==cfg['samples_per_stage'] for k in ('fit_samples','validation_samples','test_samples')),
              f'Stage counts mismatch {index}')
        check(selection['stage_absolute_slice']==[4112,4112+cfg['samples_per_stage']], f'Split mismatch {index}')
        expected = {(m,e,p) for m in cfg['memories']
                    for e in (cfg['eta'] if label.startswith('screen_') else [None])
                    for p in cfg['grids'][method]}
        n_candidate_expected += len(expected)
        actual = {(r['memory'],r['eta'],r['parameter']) for r in rows if r['status']=='ok'}
        check(expected==actual and len(rows)==len(expected), f'Incomplete candidate grid {index}')
        eligible = [r for r in rows if r['status']=='ok' and r['converged'] and math.isfinite(r['validation_db'])]
        if eligible:
            best = max(eligible, key=lambda r:(r['validation_db'],-r['n_active'],-r['memory']))
            check(best==selected, f'Winner does not match fixed validation rule {index}')
        else:
            errors.append(f'No admissible candidate {index}')
        for row in rows:
            if row['status']!='ok':
                continue
            group = (job['case']['name'], label, row['memory'])
            candidates[group]['total'] += 1
            candidates[group]['converged' if row['converged'] else 'nonconverged'] += 1
            candidates[group]['screen_reduced'] += row['n_screen'] < row['n_full']
            check(row['n_full']==count_terms(row['memory'],job['case']['quintic']),f'Wrong dictionary size {index}')
            check(0 <= row['n_active'] <= row['n_screen'] <= row['n_full'],f'Wrong stage counts {index}')
            if row['eta'] is None:
                check(row['n_screen']==row['n_full'],f'Unscreened columns lost {index}')
            candidates[group]['eta_001_reduced'] += row['eta']==.01 and row['n_screen']<row['n_full']
        family = (job['case']['name'],label)
        boundary[family]['selected_jobs'] += 1
        boundary[family]['parameter_lower_edge'] += selected['parameter']==min(cfg['grids'][method])
        boundary[family]['parameter_upper_edge'] += selected['parameter']==max(cfg['grids'][method])
        boundary[family]['memory_upper_edge'] += selected['memory']==max(cfg['memories'])
        memory_counts[family][selected['memory']] += 1
        with np.load(folder/'model.npz',allow_pickle=False) as model:
            weights = model['weights']
            terms = model['terms']
            support = [list(ast.literal_eval(str(terms[i]))) for i in np.flatnonzero(weights)]
            check(support==result['support'],f'Support/model mismatch {index}')
            check(len(support)==selected['n_active'],f'Active-count mismatch {index}')
        if method=='sbl':
            check((folder/'trajectory.json').exists(),f'Selected SBL trajectory missing {index}')
            if (folder/'trajectory.json').exists():
                history = read(folder/'trajectory.json')
                trajectory_count += 1
                check(len(history)==selected['iterations'],f'Trajectory length mismatch {index}')
                check(history[-1]['mu_relative_change']==selected['last_mu_relative_change'],
                      f'Last relative change mismatch {index}')
        grid_rows[(job['repeat'],job['case']['name'],job['pnr'],label)] = rows
    pairs = 0
    maximum_difference = 0.0
    paired_count_mismatches = 0
    for key, rows in grid_rows.items():
        if key[-1] not in ('ridge','sbl'):
            continue
        screened = grid_rows.get(key[:-1]+('screen_'+key[-1],))
        if screened is None:
            continue
        mapping = {(r['memory'],r['parameter']):r for r in screened if r.get('eta')==.01}
        for row in rows:
            if row['status']!='ok':
                continue
            other = mapping[(row['memory'],row['parameter'])]
            pairs += 1
            maximum_difference = max(maximum_difference,abs(row['validation_db']-other['validation_db']))
            paired_count_mismatches += any(row[k]!=other[k] for k in ('n_full','n_screen','n_active','converged'))
    tx = [cfg['seed_base']+1000*r+10*s for r in range(cfg['repeats']) for s in range(3)]
    noise = [s+500000 for s in tx]
    check(len(set(tx+noise))==len(tx+noise),'Seed collision across stages/repetitions')
    physical = {}
    try:
        import psutil
        physical.update(logical_cpus=psutil.cpu_count(),physical_cores=psutil.cpu_count(logical=False),
                        ram_bytes=psutil.virtual_memory().total,psutil_version=psutil.__version__)
        proc = Path('/proc/cpuinfo')
        if proc.exists():
            physical['cpu_model']=next(line.split(':',1)[1].strip() for line in proc.read_text().splitlines()
                                       if line.startswith('model name'))
    except (ImportError, StopIteration):
        pass
    for label, module in [('numpy',np),('scipy',scipy)]:
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            module.show_config()
        physical[label+'_build_config']=out.getvalue()
    return dict(audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_run=str(run),
        expected_jobs=len(jobs),successful_jobs=successful,failed_jobs=[r for r in progress if r['status']!='ok'],
        live_status=read(run/'parallel_status.json') if (run/'parallel_status.json').exists() else None,
        complete_marker=read(run/'completion.json') if (run/'completion.json').exists() else None,
        integrity_errors=errors,missing_result_files=missing_result_files,stage_counts=sorted(sample_counts),
        independent_stage_seed_count=len(tx),noise_seed_count=len(noise),
        candidate_fits_in_completed_jobs=n_candidate_expected,
        coverage=[dict(case=c,pnr=p,method=m,repeats_done=coverage[(c,p,m)],repeats_required=cfg['repeats'])
                  for c in [case['name'] for case in cfg['cases']] for p in map(str,cfg['pnr_db']) for m in cfg['methods']],
        candidate_status=[dict(case=k[0],method=k[1],memory=k[2],**v) for k,v in sorted(candidates.items())],
        selection_boundaries=[dict(case=k[0],method=k[1],selected_memories=dict(memory_counts[k]),**v)
                              for k,v in sorted(boundary.items())],
        selected_sbl_trajectories=trajectory_count,
        eta_001_paired_controls=dict(candidate_pairs=pairs,max_validation_db_difference=maximum_difference,
                                    stage_count_or_convergence_mismatches=paired_count_mismatches),
        environment_supplement=physical,
        test_scores_used_for_audit_decisions=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.run.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    summary={key:report[key] for key in ('audit_utc','expected_jobs','successful_jobs','candidate_fits_in_completed_jobs',
             'integrity_errors','selected_sbl_trajectories','eta_001_paired_controls')}
    summary['nonconverged_candidates']=sum(r.get('nonconverged',0) for r in report['candidate_status'])
    print(json.dumps(summary))
    return 1 if report['integrity_errors'] else 0


if __name__=='__main__':
    raise SystemExit(main())
