"""Resume a frozen benchmark with bounded concurrency, without changing its workers.

Linux host only. A handoff records the old coordinator and any adopted worker.
Scientific outputs stay untouched; scheduling/timing provenance is kept separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GIB = 1024 ** 3


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def jobs_for(cfg):
    return [dict(repeat=rep, case=case, pnr=pnr, method=method)
            for rep in range(cfg['repeats']) for case in cfg['cases']
            for pnr in cfg['pnr_db'] for method in cfg['methods']]


def verify_sources(run):
    manifest = read(run / 'manifest.json')
    cfg = read(run / 'source/config.json')
    if manifest['config'] != cfg:
        raise ValueError('Frozen configuration does not match manifest')
    for name, expected in manifest['source_sha256'].items():
        actual = hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Frozen source hash mismatch: {name}')
    return cfg


def reservation(job):
    # M<=6: SBL's conservative bound is unchanged. Other methods have smaller
    # workspaces; observed peaks were ridge 1.67, LASSO 1.56 and OMP 1.21 GiB.
    if not job['case']['quintic']:
        return 0.5 * GIB
    return {'sbl': 4.5, 'screen_sbl': 4.5, 'ridge': 2.25,
            'screen_ridge': 2.25, 'lasso': 2.0, 'omp': 1.75}[job['method']] * GIB


def can_launch(job, active, available, headroom):
    # Available RAM already excludes current RSS; reserve only remaining growth.
    growth = sum(max(0, item['reserve'] - item['rss']) for item in active)
    return available - growth >= reservation(job) + headroom


def next_fitting(pending, jobs, active, available, headroom):
    # Do not leave a core idle just because the first pending job needs more RAM.
    return next((i for i in pending if can_launch(jobs[i], active, available, headroom)), None)


def physical_slots(cpus, root=Path('/sys/devices/system/cpu')):
    slots = {}
    for cpu in sorted(cpus):
        topology = root / f'cpu{cpu}/topology'
        key = ((topology / 'physical_package_id').read_text().strip(),
               (topology / 'core_id').read_text().strip())
        slots.setdefault(key, cpu)
    return list(slots.values())


def load_completed(run, jobs):
    outcomes = {}
    for row in read(run / 'progress.json'):
        index = row['index']
        if index in outcomes or row['job'] != jobs[index]:
            raise ValueError('Invalid progress index/job')
        outcomes[index] = row
    for index, job in enumerate(jobs):
        folder = run / f'job_{index:04d}'
        if not folder.exists():
            if index in outcomes:
                raise ValueError(f'Missing recorded job directory: {folder}')
            continue
        if read(folder / 'job.json') != job:
            raise ValueError(f'Job definition mismatch: {folder}')
        result = folder / 'result.json'
        if result.exists():
            value = read(result)
            if value['status'] != 'ok' or value['job'] != job:
                raise ValueError(f'Invalid result: {result}')
            # Recover a result committed just before the old progress journal.
            outcomes.setdefault(index, dict(index=index, job=job, status='ok',
                seconds=max(0, result.stat().st_mtime - (folder / 'job.json').stat().st_mtime),
                timing_phase='recovered_at_handoff'))
        elif index in outcomes and outcomes[index]['status'] == 'ok':
            raise ValueError(f'Missing successful result: {result}')
    return outcomes


def alive(process, created):
    import psutil
    try:
        return (process.create_time() == created and process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def execute(args):
    import fcntl
    import psutil
    run = args.run.resolve()
    lock = (run / 'parallel.lock').open('a')
    cfg = verify_sources(run)
    if max(cfg['memories']) > 6:
        raise ValueError('RAM reservations are validated only through M=6')
    jobs = jobs_for(cfg)
    handoff = read(args.handoff)
    if handoff['run'] != str(run):
        raise ValueError('Wrong handoff directory')
    # Handoff launcher must retire the old coordinator before we write progress.
    old = handoff['coordinator']
    for _ in range(100):
        try:
            process = psutil.Process(old['pid'])
            running = alive(process, old['created'])
        except psutil.NoSuchProcess:
            running = False
        if not running:
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Old coordinator still exists; refusing duplicate scheduling')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    slots = physical_slots(psutil.Process().cpu_affinity())
    if args.workers > len(slots):
        raise ValueError('Requested workers exceed available physical cores')
    outcomes = load_completed(run, jobs)
    initial_indices = sorted(outcomes)
    active = {}
    for entry in handoff['adopted']:
        index = entry['index']
        if index in outcomes:
            continue
        try:
            process = psutil.Process(entry['pid'])
            running = alive(process, entry['created'])
        except psutil.NoSuchProcess:
            running = False
        if not running:
            recovered = load_completed(run, jobs)
            if index not in recovered:
                raise RuntimeError('Adopted worker vanished without a result')
            outcomes[index] = recovered[index]
            continue
        if process.cmdline() != entry['cmdline']:
            raise RuntimeError('Adopted worker identity changed')
        slot = slots[len(active)]
        process.cpu_affinity([slot])
        active[index] = dict(process=process, created=entry['created'], popen=None,
            started=entry['created'], reserve=reservation(jobs[index]), adopted=True, slot=slot)
    for index in range(len(jobs)):
        folder = run / f'job_{index:04d}'
        if folder.exists() and index not in outcomes and index not in active:
            raise ValueError(f'Untracked partial job: {folder}; refusing overwrite')
    pending = [i for i in range(len(jobs)) if i not in outcomes and i not in active]
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    start = time.time()
    deadline = handoff['original_started'] + cfg['max_run_hours'] * 3600
    provenance_path = run / 'timing_provenance.json'
    previous_provenance = read(provenance_path) if provenance_path.exists() else None
    dump(provenance_path, dict(
        previous_phase=previous_provenance,
        parallel_started_unix=start,
        initial_serial_indices=(previous_provenance['initial_serial_indices']
                                if previous_provenance else initial_indices),
        adopted_indices=[e['index'] for e in handoff['adopted']],
        max_workers=args.workers, blas_threads_per_worker=1,
        completed_indices_at_transition=initial_indices,
        physical_core_cpu_slots=slots[:args.workers], headroom_gib=args.headroom_gib,
        performance_comparison_valid=False,
        note='Concurrent workloads affect wall-clock timing. Use separate serial profiling for efficiency claims. '
             'Workers, seeds, splits, grids and scientific results are unchanged. '
             'RSS is still per workflow; timing values are operational measurements.'))
    reason = None
    while pending or active:
        for index, item in list(active.items()):
            process = item['process']
            running = alive(process, item['created'])
            elapsed = time.time() - item['started']
            timed_out = running and elapsed > cfg['task_timeout_sec']
            if timed_out:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
                running = False
            if running:
                continue
            code = item['popen'].poll() if item['popen'] is not None else None
            result = run / f'job_{index:04d}/result.json'
            ok = result.exists() and read(result).get('job') == jobs[index]
            ok = ok and read(result).get('status') == 'ok' and code in (None, 0)
            status = 'timeout' if timed_out else ('ok' if ok else 'failed')
            outcomes[index] = dict(index=index, job=jobs[index], status=status,
                seconds=elapsed, timing_phase='parallel_adopted' if item['adopted'] else 'parallel')
            del active[index]
            dump(run / 'progress.json', [outcomes[i] for i in sorted(outcomes)])
            print(index, jobs[index]['case']['name'], jobs[index]['method'], status,
                  round(elapsed, 2), flush=True)
        failed = sum(r['status'] != 'ok' for r in outcomes.values())
        if time.time() >= deadline:
            reason = 'Original overall time budget reached'
        if failed >= cfg['max_failed_jobs']:
            reason = 'Original failure budget reached'
        while pending and len(active) < args.workers and reason is None:
            samples = []
            for item in active.values():
                try:
                    rss = item['process'].memory_info().rss
                except psutil.NoSuchProcess:
                    rss = 0
                samples.append(dict(reserve=item['reserve'], rss=rss))
            index = next_fitting(pending, jobs, samples, psutil.virtual_memory().available,
                                 args.headroom_gib * GIB)
            if index is None:
                break
            folder = run / f'job_{index:04d}'
            folder.mkdir()
            dump(folder / 'job.json', jobs[index])
            with (folder / 'worker.log').open('w') as log:
                child = subprocess.Popen([sys.executable, str(run / 'source/run_experiments.py'),
                    '--config', str(run / 'source/config.json'), '--output', str(folder),
                    '--worker', str(folder / 'job.json')], env=env, stdout=log, stderr=subprocess.STDOUT)
            process = psutil.Process(child.pid)
            slot = next(cpu for cpu in slots[:args.workers]
                        if cpu not in {item['slot'] for item in active.values()})
            process.cpu_affinity([slot])
            active[index] = dict(process=process, created=process.create_time(), popen=child,
                started=time.time(), reserve=reservation(jobs[index]), adopted=False, slot=slot)
            pending.remove(index)
        dump(run / 'parallel_status.json', dict(pid=os.getpid(), updated_unix=time.time(),
            elapsed_since_handoff_sec=time.time()-start, completed=len(outcomes), failed=failed,
            total=len(jobs), pending=len(pending), max_workers=args.workers,
            scheduler_version=2, physical_core_cpu_slots=slots[:args.workers],
            available_gib=psutil.virtual_memory().available/GIB, stop_reason=reason,
            active=[dict(index=i, pid=v['process'].pid, elapsed_sec=time.time()-v['started'],
                         adopted=v['adopted'], cpu_slot=v['slot']) for i, v in active.items()]))
        if not active and (not pending or reason is not None):
            break
        time.sleep(2)
    complete = len(outcomes) == len(jobs) and all(r['status'] == 'ok' for r in outcomes.values())
    dump(run / 'completion.json', dict(status='complete' if complete else 'incomplete',
         jobs=len(outcomes), failed=sum(r['status'] != 'ok' for r in outcomes.values()), reason=reason))
    if complete:
        # Keep frozen aggregation code; separate provenance disallows timing claims.
        subprocess.run([sys.executable, str(run / 'source/summarize.py'), str(run)], env=env, check=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--headroom-gib', type=float, default=2.5)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.headroom_gib < 2.5:
        parser.error('Validated envelope: 1..4 workers and at least 2.5 GiB headroom')
    try:
        return execute(args)
    except Exception as exc:
        dump(args.run / 'parallel_error.json', dict(error=repr(exc), time_unix=time.time()))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
