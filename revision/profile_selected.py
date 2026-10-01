"""Serial resource measurements of frozen validation-selected models.

The representative slice is declared before profiling: repetition 0, PNR 35 dB,
all three source/dictionary cases and all six methods. No model is reselected.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def worker(run, out, index):
    import resource
    import numpy as np
    import psutil
    psutil.Process().cpu_affinity([0])
    spec = importlib.util.spec_from_file_location('profile_frozen_experiment', run/'source/experiment.py')
    ex = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ex
    spec.loader.exec_module(ex)
    cfg = read(run/'source/config.json')
    folder = run/f'job_{index:04d}'
    selection = read(folder/'selection.json')
    job, selected = selection['job'], selection['chosen']
    assert job['repeat'] == 0 and job['pnr'] == 35
    seed = cfg['seed_base']
    count = cfg['samples_per_stage']
    fit_record = ex.generate(job['case']['source'], seed, seed+500000, 35)
    reference = ex.generate(job['case']['source'], seed+10, seed+500010, 35)
    keys = ex.terms(selected['memory'], job['case']['quintic'])
    y = fit_record.observed[ex.DISCARD:ex.DISCARD+count]
    begin = time.perf_counter()
    x = ex.design(fit_record, keys, count)
    dictionary_sec = time.perf_counter()-begin
    begin = time.perf_counter()
    weights, details = ex.fit(x, y, job['method'].removeprefix('screen_'),
                             selected['parameter'], selected['eta'], max_iter=cfg['sbl_max_iter'])
    fit_sec = time.perf_counter()-begin
    with np.load(folder/'model.npz', allow_pickle=False) as saved:
        original = saved['weights']
        assert list(map(tuple, map(ast.literal_eval, saved['terms']))) == keys
        relative_error = float(np.linalg.norm(weights-original)/max(np.linalg.norm(original), 1e-12))
        support_equal = np.array_equal(np.flatnonzero(weights), np.flatnonzero(original))
    if not (details['converged'] and support_equal and relative_error < 1e-9):
        raise RuntimeError(f'Frozen model reproduction failed: {relative_error=}, {support_equal=}')
    active = np.flatnonzero(weights)
    active_keys = [keys[i] for i in active]
    # The validation references provide the timing input. No test target is read.
    _ = ex.design(reference, active_keys, count) @ weights[active]
    durations = []
    for _ in range(7):
        begin = time.perf_counter()
        _ = ex.design(reference, active_keys, count) @ weights[active]
        durations.append(time.perf_counter()-begin)
    result = dict(index=index, job=job, memory=selected['memory'], parameter=selected['parameter'],
                  eta=selected['eta'], dictionary_sec=dictionary_sec, fit_sec=fit_sec,
                  prepare_screen_sec=details['screen_sec'],
                  train_total_sec=dictionary_sec+fit_sec,
                  n_full=details['n_full'], n_screen=details['n_screen'], n_active=len(active),
                  iterations=details['iterations'], converged=details['converged'],
                  inference_seconds=durations,
                  inference_median_us_per_sample=float(np.median(durations))*1e6/count,
                  process_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  peak_scope='Linux process high-water RSS; imports, records, one dictionary/fit, and inference; excludes hyperparameter search',
                  coefficient_relative_error=relative_error, support_equal=bool(support_equal),
                  cpu_slot=0, threads=1, test_target_read=False, status='ok')
    dump(out/f'profile_{index:04d}.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-index', type=int)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    if args.worker_index is not None:
        worker(run, out, args.worker_index)
        return
    assert read(run/'completion.json')['status'] == 'complete'
    manifest = read(run/'manifest.json')
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest() == digest
    jobs = [p for p in read(run/'progress.json') if p['job']['repeat'] == 0 and p['job']['pnr'] == 35]
    assert len(jobs) == 18 and all(p['status'] == 'ok' for p in jobs)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'profile_selected.py')
    dump(out/'protocol.json', dict(source_run=str(run), selection='repeat 0, PNR 35 dB, every case/method; determined without reading test scores',
         jobs=[p['index'] for p in jobs], refits_per_job=1, inference_warmups=1, inference_repeats=7,
         cpu_slot=0, blas_threads=1, timeout_per_job_sec=3600, python=sys.version, platform=platform.platform(),
         source_sha256=manifest['source_sha256'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         interpretation='Representative single-refit timings, not population timing estimates. Full tuning is excluded.'))
    outcomes = []
    for job in jobs:
        index = job['index']
        begin = time.time()
        dump(out/'status.json', dict(completed=len(outcomes), total=len(jobs), active=index, started_unix=begin))
        with (out/f'profile_{index:04d}.log').open('w') as log:
            proc = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--run', str(run),
                '--output', str(out), '--worker-index', str(index)], stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        outcomes.append(dict(index=index, returncode=proc.returncode, seconds=time.time()-begin))
        dump(out/'progress.json', outcomes)
        print(index, proc.returncode, round(outcomes[-1]['seconds'], 2), flush=True)
        if proc.returncode:
            dump(out/'completion.json', dict(status='failed', jobs=len(outcomes), failed_index=index))
            raise SystemExit(proc.returncode)
    dump(out/'status.json', dict(completed=len(outcomes), total=len(jobs), active=None))
    dump(out/'completion.json', dict(status='complete', jobs=len(outcomes), failed=0))


if __name__ == '__main__':
    main()
