"""Run each method/repetition/scenario in an isolated process; journal every outcome."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')

import numpy as np
import scipy
import experiment as ex

METHODS={'ridge':('ridge',False),'sbl':('sbl',False),
         'screen_ridge':('ridge',True),'screen_sbl':('sbl',True),
         'lasso':('lasso',False),'omp':('omp',False)}


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


class Memory:
    def __init__(self):
        import psutil
        self.process=psutil.Process();self.peak=0;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.sample,daemon=True)
    def sample(self):
        while not self.stop.wait(.01):self.peak=max(self.peak,self.process.memory_info().rss)
    def __enter__(self):self.peak=self.process.memory_info().rss;self.thread.start();return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss)
        self.stop.set();self.thread.join()


def record(config,case,rep,stage,pnr):
    seed=config['seed_base']+1000*rep+10*stage
    return ex.generate(case['source'],seed,seed+500000,pnr)


def run_worker(config,job,out):
    label=job['method'];method,screened=METHODS[label];case=job['case'];rep=job['repeat'];pnr=job['pnr']
    count=config['samples_per_stage'];rows=[];winner=None
    start=time.perf_counter()
    with Memory() as rss:
        fit_record=record(config,case,rep,0,pnr)
        validation_record=record(config,case,rep,1,pnr)
        y=fit_record.observed[ex.DISCARD:ex.DISCARD+count]
        yv=validation_record.observed[ex.DISCARD:ex.DISCARD+count]
        for memory in config['memories']:
            keys=ex.terms(memory,case['quintic']);p=len(keys)
            # Deterministic preflight applied to all methods, including inexpensive baselines.
            # This caps the common dictionary grid, not individual competitors differently.
            estimate=16*(8*p*p+6*count*p)
            if estimate>config['memory_budget_gib']*1024**3:
                rows.append({'memory':memory,'status':'resource_excluded_common_grid','estimate_bytes':estimate})
                continue
            t=time.perf_counter();x=ex.design(fit_record,keys,count);xv=ex.design(validation_record,keys,count)
            build_sec=time.perf_counter()-t
            for eta in (config['eta'] if screened else [None]):
                for param in config['grids'][method]:
                    h,d=ex.fit(x,y,method,param,eta,max_iter=config['sbl_max_iter'])
                    trajectory=d.pop('history',None)
                    score=ex.cancellation(yv,xv@h)
                    row=dict(memory=memory,parameter=param,eta=eta,validation_db=score,
                             build_fit_validation_sec=build_sec,status='ok',**d)
                    rows.append(row)
                    # Test is neither generated nor evaluated while selecting settings.
                    admissible=d['converged'] and np.isfinite(score)
                    rank=(score,-d['n_active'],-memory)
                    if admissible and (winner is None or rank>winner['rank']):
                        winner={'rank':rank,'row':row.copy(),'weights':h.copy(),'keys':keys,'history':trajectory}
            del x,xv
        dump(out/'validation.json',rows)
        if winner is None:
            raise RuntimeError('No converged admissible candidate; no test result reported')
        chosen=winner['row'];h=winner['weights'];keys=winner['keys']
        # Freeze selection and its coefficients before revealing test data.
        dump(out/'selection.json',{'job':job,'chosen':chosen,'fit_samples':count,
                                   'validation_samples':count,'test_samples':count,
                                   'stage_absolute_slice':[ex.DISCARD,ex.DISCARD+count]})
        np.savez(out/'model.npz',weights=h,terms=np.asarray([str(k) for k in keys]))
        if winner['history'] is not None:dump(out/'trajectory.json',winner['history'])
        tune_sec=time.perf_counter()-start
        test_record=record(config,case,rep,2,pnr)
        yt=test_record.observed[ex.DISCARD:ex.DISCARD+count]
        clean=test_record.clean[ex.DISCARD:ex.DISCARD+count]
        active=np.flatnonzero(h);active_keys=[keys[i] for i in active]
        durations=[]
        for _ in range(config['inference_repeats']):
            t=time.perf_counter()
            pred=ex.design(test_record,active_keys,count)@h[active]
            durations.append(time.perf_counter()-t)
        result={'job':job,'selected':chosen,'test_observed_db':ex.cancellation(yt,pred),
                'test_pim_only_db':ex.cancellation(clean,pred),'support':[list(k) for k in active_keys],
                'tuning_sec':tune_sec,'inference_median_sec':float(np.median(durations)),
                'inference_us_per_sample':float(np.median(durations))*1e6/count,
                'inference_repeats':len(durations),'status':'ok'}
    result['workflow_peak_rss_bytes']=rss.peak
    result['peak_scope']='isolated complete method workflow, including validation search and inference; RSS sampled every 10 ms'
    dump(out/'result.json',result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if args.worker:
        run_worker(cfg,json.loads(args.worker.read_text()),args.output);return
    args.output.mkdir(parents=True,exist_ok=False)
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (args.output/'source').mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p,args.output/'source'/p.name)
    shutil.copy2(args.config,args.output/'source/config.json')
    dump(args.output/'manifest.json',{'config':cfg,'source_sha256':source_hashes,
         'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
         'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
         'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
         'status':'running','seed_rule':'tx=seed_base+1000*repeat+10*stage; noise=tx+500000; stage fit=0,val=1,test=2',
         'protocol':'Independent records; first samples_per_stage observations after discard=4112; no refit after validation.'})
    outcomes=[];index=0;started=time.monotonic()
    for rep in range(cfg['repeats']):
        for case in cfg['cases']:
            for pnr in cfg['pnr_db']:
                for method in cfg['methods']:
                    if (time.monotonic()-started>cfg.get('max_run_hours',24)*3600
                            or sum(r['status']!='ok' for r in outcomes)>=cfg.get('max_failed_jobs',6)):
                        dump(args.output/'completion.json',{'status':'incomplete','jobs':len(outcomes),
                            'failed':sum(r['status']!='ok' for r in outcomes),
                            'reason':'overall time or failure budget reached; remaining jobs not evaluated'})
                        raise SystemExit(1)
                    job={'repeat':rep,'case':case,'pnr':pnr,'method':method}
                    folder=args.output/f'job_{index:04d}';folder.mkdir();dump(folder/'job.json',job)
                    t=time.perf_counter()
                    try:
                        with (folder/'worker.log').open('w') as log:
                            done=subprocess.run([sys.executable,str(Path(__file__).resolve()),
                                 '--config',str(args.config.resolve()),'--output',str(folder.resolve()),
                                 '--worker',str((folder/'job.json').resolve())],stdout=log,stderr=subprocess.STDOUT,
                                 timeout=cfg['task_timeout_sec'])
                        status='ok' if done.returncode==0 else 'failed'
                    except subprocess.TimeoutExpired:status='timeout'
                    outcomes.append({'index':index,'job':job,'status':status,'seconds':time.perf_counter()-t})
                    dump(args.output/'progress.json',outcomes)
                    print(index,method,case['name'],pnr,status,round(outcomes[-1]['seconds'],2),flush=True)
                    index+=1
    status='complete' if all(r['status']=='ok' for r in outcomes) else 'incomplete'
    dump(args.output/'completion.json',{'status':status,'jobs':len(outcomes),
                                       'failed':sum(r['status']!='ok' for r in outcomes)})
    if status!='complete':raise SystemExit(1)


if __name__=='__main__':main()
