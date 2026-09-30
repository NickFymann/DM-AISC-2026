"""Transfer this benchmark from its three-worker scheduler to four physical cores."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
import parallel_continue_v2 as pc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = pc.verify_sources(run)
    jobs = pc.jobs_for(cfg)
    status = pc.read(run / 'parallel_status.json')
    coordinator = psutil.Process(status['pid'])
    command = coordinator.cmdline()
    if (str(run.parent / 'parallel_continue.py') not in command or '--run' not in command
            or command[command.index('--run') + 1] != str(run)):
        raise RuntimeError('PID does not match the expected three-worker scheduler')
    if status['max_workers'] != 3:
        raise RuntimeError('Expected original three-worker status')
    if (run / 'handoff_4workers.json').exists():
        raise RuntimeError('Four-worker handoff already exists; inspect status')
    manifest = pc.read(run / 'manifest.json')
    for name in ('run_experiments.py', 'experiment.py'):
        if hashlib.sha256((run.parent / name).read_bytes()).hexdigest() != manifest['source_sha256'][name]:
            raise RuntimeError('Root scientific source does not match frozen version')
    slots = pc.physical_slots(coordinator.cpu_affinity())
    if len(slots) < 4:
        raise RuntimeError('Four physical cores are not available')
    print('Preflight OK: physical core slots', slots, 'coordinator', coordinator.pid, flush=True)
    if not args.execute:
        return
    coordinator.suspend()
    committed = False
    manager = None
    try:
        status = pc.read(run / 'parallel_status.json')
        # One worker was adopted from the serial scheduler and has PPID 1.
        # Thus both active journal entries and newly spawned children are needed.
        candidates = {entry['pid'] for entry in status['active']}
        candidates.update(process.pid for process in coordinator.children())
        adopted = []
        for pid in sorted(candidates):
            try:
                process = psutil.Process(pid)
                if not pc.alive(process, process.create_time()):
                    continue
                worker_command = process.cmdline()
            except psutil.NoSuchProcess:
                continue
            if '--worker' not in worker_command:
                raise RuntimeError('Unexpected coordinator child')
            jobfile = Path(worker_command[worker_command.index('--worker') + 1]).resolve()
            if jobfile.parent.parent != run or jobfile.name != 'job.json':
                raise RuntimeError('Worker outside intended run')
            index = int(jobfile.parent.name.removeprefix('job_'))
            if pc.read(jobfile) != jobs[index]:
                raise RuntimeError('Worker configuration does not match grid')
            adopted.append(dict(index=index, pid=pid, created=process.create_time(), cmdline=worker_command))
        outcomes = pc.load_completed(run, jobs)
        saved_hashes = {}
        for index, row in outcomes.items():
            if row['status'] == 'ok':
                for path in (run / f'job_{index:04d}').iterdir():
                    if path.is_file():
                        saved_hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        pc.dump(run / 'progress_before_4workers.json', pc.read(run / 'progress.json'))
        for name in ('parallel_launch', 'timing_provenance'):
            shutil.copy2(run / f'{name}.json', run / f'{name}_3workers.json')
        original = pc.read(run / 'handoff.json')
        handoff = dict(run=str(run), time_unix=time.time(),
            original_started=original['original_started'],
            coordinator=dict(pid=coordinator.pid, created=coordinator.create_time(), cmdline=command),
            adopted=adopted, completed_file_sha256=saved_hashes,
            scheduler_sha256=hashlib.sha256((run.parent / 'parallel_continue_v2.py').read_bytes()).hexdigest(),
            physical_core_cpu_slots=slots[:4])
        pc.dump(run / 'handoff_4workers.json', handoff)
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        logpath = run.parent / (run.name + '_parallel_v2.log')
        with logpath.open('x') as log:
            manager = subprocess.Popen([sys.executable, '-u', str(run.parent / 'parallel_continue_v2.py'),
                '--run', str(run), '--handoff', str(run / 'handoff_4workers.json'),
                '--workers', '4', '--headroom-gib', '2.5'], env=env,
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        coordinator.terminate()
        coordinator.resume()
        coordinator.wait(timeout=5)
        committed = True
        pc.dump(run / 'parallel_launch.json', dict(pid=manager.pid, workers=4,
            directory=str(run), adopted=adopted, completed_at_transition=len(outcomes),
            log=str(logpath), scheduler='parallel_continue_v2.py',
            physical_core_cpu_slots=slots[:4], headroom_gib=2.5))
        print('Launched', manager.pid, 'adopted worker PIDs', [r['pid'] for r in adopted],
              'preserved completed jobs', len(outcomes), flush=True)
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
