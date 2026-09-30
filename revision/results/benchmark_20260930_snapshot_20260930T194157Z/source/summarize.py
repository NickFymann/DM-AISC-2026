"""Summarize completed jobs without treating a partial benchmark as final evidence."""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def statistics(values):
    a=np.asarray(values,float);n=len(a);mean=float(a.mean())
    if n<2:return {'n':n,'mean':mean,'sd':None,'ci95':None}
    sd=float(a.std(ddof=1));half=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
    return {'n':n,'mean':mean,'sd':sd,'ci95':[mean-half,mean+half]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--pilot-tex',type=Path)
    args=parser.parse_args();root=args.run
    manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config']
    expected=cfg['repeats']*len(cfg['cases'])*len(cfg['pnr_db'])*len(cfg['methods'])
    results=[json.loads(p.read_text()) for p in sorted(root.glob('job_*/result.json'))]
    complete=(root/'completion.json').exists() and json.loads((root/'completion.json').read_text())['status']=='complete'
    if len(results)!=expected or not complete:
        raise SystemExit(f'Incomplete run: {len(results)}/{expected}; final summary refused')
    groups=defaultdict(list);lookup={};flat=[]
    for r in results:
        j=r['job'];s=r['selected'];key=(j['case']['name'],str(j['pnr']),j['method'])
        groups[key].append(r);lookup[key+(j['repeat'],)]=r
        flat.append({'case':key[0],'pnr':key[1],'method':key[2],'repeat':j['repeat'],
                     'memory':s['memory'],'parameter':s['parameter'],'eta':s['eta'],
                     'validation_db':s['validation_db'],'test_db':r['test_observed_db'],
                     'pim_only_db':r['test_pim_only_db'],'n_full':s['n_full'],
                     'n_screen':s['n_screen'],'active':s['n_active'],'fit_sec':s['fit_sec'],
                     'tuning_sec':r['tuning_sec'],'inference_us_per_sample':r['inference_us_per_sample'],
                     'workflow_peak_mib':r['workflow_peak_rss_bytes']/1024**2})
    summary=[]
    for key,rs in groups.items():
        pairs=[]
        for a,b in itertools.combinations(rs,2):
            if a['selected']['memory']==b['selected']['memory']:
                aa=set(map(tuple,a['support']));bb=set(map(tuple,b['support']))
                pairs.append(len(aa&bb)/max(len(aa|bb),1))
        diff=[]
        for r in rs:
            base=lookup.get(key[:2]+('ridge',r['job']['repeat']))
            if base:diff.append(r['test_observed_db']-base['test_observed_db'])
        summary.append({'case':key[0],'pnr':key[1],'method':key[2],
            'observed_db':statistics([r['test_observed_db'] for r in rs]),
            'paired_difference_to_ridge_db':statistics(diff) if diff else None,
            'support_jaccard_same_memory':statistics(pairs) if pairs else None,
            'active':statistics([r['selected']['n_active'] for r in rs]),
            'workflow_peak_mib':statistics([r['workflow_peak_rss_bytes']/1024**2 for r in rs]),
            'inference_us_per_sample':statistics([r['inference_us_per_sample'] for r in rs])})
    (root/'summary.json').write_text(json.dumps({'scope':cfg['purpose'],'rows':summary},indent=2)+'\n')
    with (root/'selected_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    if args.pilot_tex:
        if cfg['repeats']!=1 or 'pilot' not in cfg['purpose'].lower():
            raise SystemExit('Pilot renderer requires the explicitly labeled one-repetition pilot')
        args.pilot_tex.parent.mkdir(parents=True,exist_ok=True)
        lines=[r'An infrastructure pilot completed all 24 method/scenario jobs using independent fitting, validation and test records. It covers only $M\in\{2,4\}$, one repetition, the cubic/cubic and tanh/cubic--quintic cases, and clean or 35 dB PIM-to-noise observations. Table~\ref{tab:pilot} reports its untouched-test scores; these pilot results do not provide means, variability or the final benchmark comparison.',
               r'\begin{table*}[!t]',r'\caption{Pipeline pilot only: validation-selected test cancellation (dB), one repetition and $M\in\{2,4\}$.}',
               r'\label{tab:pilot}\centering\small',r'\begin{tabular}{lrrrr}\toprule',
               r'Method & Cubic clean & Cubic 35 dB & Tanh 3+5 clean & Tanh 3+5 35 dB\\\midrule']
        names={'ridge':'Ridge-LS','sbl':'SBL','screen_ridge':'Screen + ridge-LS','screen_sbl':'Screen + SBL','lasso':'Complex LASSO','omp':'Complex OMP'}
        for method in cfg['methods']:
            values=[lookup[(case,str(pnr),method,0)]['test_observed_db'] for case in ('cubic_c3','tanh_c35') for pnr in (None,35)]
            lines.append(names[method]+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
        lines.extend([r'\bottomrule\end{tabular}\end{table*}',
            r'The full candidate logs, selected settings, screening counts, process peak memory and inference timings are retained with the pilot. Hyperparameter search grids differ between estimator families but all use the same validation records. A screening branch may choose a higher threshold and therefore need not equal its unscreened counterpart. The fixed-$\eta=0.01$ control remains separate from this tuned comparison.'])
        args.pilot_tex.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Validated {len(results)} completed jobs; wrote summary and selected_results.csv.')


if __name__=='__main__':main()
