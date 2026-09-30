"""Retire only the specified serial coordinator; retain its numerical worker."""
from pathlib import Path
import argparse
import hashlib
import os
import subprocess
import sys
import time
import psutil
import parallel_continue as pc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--coordinator', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = pc.verify_sources(run)
    jobs = pc.jobs_for(cfg)
    coordinator = psutil.Process(args.coordinator)
    command = coordinator.cmdline()
    if ('--worker' in command or str(run.parent / 'run_experiments.py') not in command
            or '--output' not in command or command[command.index('--output') + 1] != str(run)):
        raise RuntimeError('Specified PID is not this run\'s serial coordinator')
    manifest = pc.read(run / 'manifest.json')
    for name in ('run_experiments.py', 'experiment.py'):
        if hashlib.sha256((run.parent / name).read_bytes()).hexdigest() != manifest['source_sha256'][name]:
            raise RuntimeError('Live serial sources do not match frozen sources')
    if pc.read(run.parent / 'benchmark.json') != cfg:
        raise RuntimeError('Live serial config does not match frozen config')
    if (run / 'handoff.json').exists():
        raise RuntimeError('Handoff already exists; inspect status before continuing')
    if not args.execute:
        print('Preflight OK:', len(jobs), 'jobs; coordinator', coordinator.pid,
              'children', [p.pid for p in coordinator.children()], flush=True)
        return
    # Stop scheduling atomically while the already-running worker continues.
    coordinator.suspend()
    committed = False
    manager = None
    try:
        adopted = []
        for process in coordinator.children():
            child_command = process.cmdline()
            if '--worker' not in child_command:
                raise RuntimeError('Unexpected coordinator child')
            jobfile = Path(child_command[child_command.index('--worker') + 1]).resolve()
            if jobfile.parent.parent != run or jobfile.name != 'job.json':
                raise RuntimeError('Worker is outside the intended run')
            index = int(jobfile.parent.name.removeprefix('job_'))
            if pc.read(jobfile) != jobs[index]:
                raise RuntimeError('Worker job does not match frozen grid')
            adopted.append(dict(index=index, pid=process.pid, created=process.create_time(), cmdline=child_command))
        outcomes = pc.load_completed(run, jobs)
        saved_hashes = {}
        for index, row in outcomes.items():
            if row['status'] == 'ok':
                for path in (run / f'job_{index:04d}').iterdir():
                    if path.is_file():
                        saved_hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        pc.dump(run / 'progress_before_parallel.json', pc.read(run / 'progress.json'))
        handoff = dict(run=str(run), time_unix=time.time(),
            original_started=coordinator.create_time(),
            coordinator=dict(pid=coordinator.pid, created=coordinator.create_time(), cmdline=command),
            adopted=adopted, completed_file_sha256=saved_hashes,
            scheduler_sha256=hashlib.sha256((run.parent / 'parallel_continue.py').read_bytes()).hexdigest())
        pc.dump(run / 'handoff.json', handoff)
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with (run.parent / (run.name + '_parallel.log')).open('x') as log:
            manager = subprocess.Popen([sys.executable, '-u', str(run.parent / 'parallel_continue.py'),
                '--run', str(run), '--handoff', str(run / 'handoff.json'), '--workers', '3'],
                env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        # The new manager waits for this PID to exit before touching progress.
        coordinator.terminate()
        coordinator.resume()
        coordinator.wait(timeout=5)
        committed = True
        pc.dump(run / 'parallel_launch.json', dict(pid=manager.pid, workers=3,
            directory=str(run), adopted=adopted, serial_completed=len(outcomes),
            log=str(run.parent / (run.name + '_parallel.log'))))
        print('Parallel manager launched:', manager.pid, 'adopted:', [r['pid'] for r in adopted],
              'completed:', len(outcomes), flush=True)
    finally:
        if not committed:
            if manager is not None and manager.poll() is None:
                manager.terminate()
            try:
                if coordinator.is_running():
                    coordinator.resume()
            except psutil.NoSuchProcess:
                pass


if __name__ == '__main__':
    main()
