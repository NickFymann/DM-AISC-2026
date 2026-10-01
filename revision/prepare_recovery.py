"""Copy successful jobs into a new run and retry only timeouts with frozen workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
import parallel_continue_v2 as pc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    source=args.source.resolve();output=args.output.resolve()
    cfg=pc.verify_sources(source);jobs=pc.jobs_for(cfg)
    progress=pc.read(source/'progress.json');completion=pc.read(source/'completion.json')
    status=pc.read(source/'parallel_status.json')
    if completion['status']!='incomplete' or len(progress)!=len(jobs):
        raise RuntimeError('Expected fully attempted but incomplete primary run')
    if psutil.pid_exists(status['pid']) or status['active']:
        raise RuntimeError('Primary scheduler or workers still active')
    if len({r['index'] for r in progress})!=len(jobs):
        raise RuntimeError('Missing or duplicate source indices')
    for row in progress:
        if row['job']!=jobs[row['index']]:
            raise RuntimeError('Unexpected source job')
    success=pc.load_completed(source,jobs)
    retry=[row for row in progress if row['status']!='ok']
    if not retry or any(row['status']!='timeout' for row in retry):
        raise RuntimeError('This continuation accepts only timeouts, not unexplained failures')
    for row in retry:
        folder=source/f"job_{row['index']:04d}"
        if any((folder/name).exists() for name in ('result.json','selection.json','model.npz')):
            raise RuntimeError('Timed-out job has committed model/test artifacts; inspect first')
    if output.exists():
        raise RuntimeError('Recovery destination already exists; refusing overwrite')
    original_started=pc.read(source/'handoff.json')['original_started']
    if time.time()+6*3600 > original_started+cfg['max_run_hours']*3600:
        raise RuntimeError('Less than six hours remain in the original overall budget')
    print(json.dumps(dict(preflight='ok',successful=len(progress)-len(retry),
        retry_indices=[r['index'] for r in retry],task_timeout_sec=10800,
        original_overall_deadline_unix=original_started+cfg['max_run_hours']*3600)),flush=True)
    if not args.execute:return
    output.mkdir(parents=True)
    shutil.copytree(source/'source',output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('manifest.json','timing_provenance.json'):
        shutil.copy2(source/name,output/name)
    retained=[];hashes={}
    for row in progress:
        if row['status']!='ok':continue
        name=f"job_{row['index']:04d}"
        shutil.copytree(source/name,output/name,ignore=shutil.ignore_patterns('__pycache__'))
        for path in (source/name).iterdir():
            if path.is_file():
                relative=path.relative_to(source)
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if hashlib.sha256((output/relative).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Copied completed file hash mismatch')
                hashes[relative.as_posix()]=digest
        retained.append(row)
    pc.dump(output/'progress.json',retained)
    pc.dump(output/'primary_completion.json',completion)
    pc.dump(output/'primary_progress.json',progress)
    provenance=dict(primary_run=str(source),recovery_run=str(output),created_unix=time.time(),
        retained_successful_jobs=len(retained),original_timeout_attempts=retry,
        retained_file_sha256=hashes,scientific_config_unchanged=True,
        scheduling_override=dict(task_timeout_sec=10800,workers=3,headroom_gib=2.5,
            original_overall_deadline_preserved=True,skip_summary=True),
        scheduler_sha256=hashlib.sha256((source.parent/'parallel_continue_v2.py').read_bytes()).hexdigest())
    pc.dump(output/'recovery_provenance.json',provenance)
    pc.dump(output/'handoff.json',dict(run=str(output),original_started=original_started,
        coordinator=dict(pid=status['pid'],created=0),adopted=[],
        source_run=str(source),retry_indices=[r['index'] for r in retry]))
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    logpath=source.parent/(output.name+'.log')
    with logpath.open('x') as log:
        process=subprocess.Popen([sys.executable,'-u',str(source.parent/'parallel_continue_v2.py'),
            '--run',str(output),'--handoff',str(output/'handoff.json'),'--workers','3',
            '--headroom-gib','2.5','--task-timeout-sec','10800','--skip-summary'],
            env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    launch=dict(pid=process.pid,output=str(output),primary_run=str(source),log=str(logpath),
        retained_jobs=len(retained),retry_indices=[r['index'] for r in retry],workers=3,
        task_timeout_sec=10800,retained_files_verified=len(hashes),
        scientific_config_unchanged=True,automatic_summary=False)
    pc.dump(output/'recovery_launch.json',launch)
    print(json.dumps(launch),flush=True)


if __name__=='__main__':main()
