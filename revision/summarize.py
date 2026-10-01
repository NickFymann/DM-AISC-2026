"""Summarize completed jobs without treating a partial benchmark as final evidence."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def statistics(values):
    a=np.asarray(values,float);n=len(a)
    if not n:return {'n':0,'mean':None,'sd':None,'ci95':None}
    if not np.isfinite(a).all():raise ValueError('Non-finite observation in summary')
    mean=float(a.mean())
    if n<2:return {'n':n,'mean':mean,'sd':None,'ci95':None}
    sd=float(a.std(ddof=1));half=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
    return {'n':n,'mean':mean,'sd':sd,'ci95':[mean-half,mean+half]}


def support_statistics(results):
    """Pairs sharing a run are dependent: report descriptive overlap, no pair CI."""
    strata=defaultdict(list)
    for result in results:strata[result['selected']['memory']].append(result)
    summaries=[];all_pairs=[];frequencies=[]
    for memory,rs in sorted(strata.items()):
        supports=[set(map(tuple,r['support'])) for r in rs]
        pairs=[len(a&b)/len(a|b) if a|b else 1.
               for a,b in itertools.combinations(supports,2)]
        all_pairs.extend(pairs)
        summaries.append({'memory':memory,'n_runs':len(rs),'n_pairs':len(pairs),
            'mean':float(np.mean(pairs)) if pairs else None,
            'min':min(pairs) if pairs else None,'max':max(pairs) if pairs else None})
        for term,count in sorted(Counter(term for support in supports for term in support).items()):
            frequencies.append({'memory':memory,'term':str(term),'included_runs':count,
                                'eligible_runs':len(rs),'frequency':count/len(rs)})
    return {'n_runs':len(results),'n_pairs':len(all_pairs),
        'mean':float(np.mean(all_pairs)) if all_pairs else None,'ci95':None,
        'interpretation':'Descriptive pairwise Jaccard within equal selected memory; pairs are dependent; no confidence interval.',
        'strata':summaries},frequencies


def validate_jobs(results,cfg):
    expected={(c['name'],str(p),m,r) for c in cfg['cases'] for p in cfg['pnr_db']
              for m in cfg['methods'] for r in range(cfg['repeats'])}
    identities=[(a['job']['case']['name'],str(a['job']['pnr']),a['job']['method'],a['job']['repeat']) for a in results]
    if set(identities)!=expected or len(identities)!=len(set(identities)):
        raise ValueError('Missing, unexpected or duplicated job identity')
    if any(a['status']!='ok' or not a['selected']['converged'] for a in results):
        raise ValueError('A selected model is not successful and converged')


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--pilot-tex',type=Path)
    parser.add_argument('--output',type=Path,help='Separate derived output directory; raw run is unchanged')
    args=parser.parse_args();root=args.run;out=args.output or root
    manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config']
    expected=cfg['repeats']*len(cfg['cases'])*len(cfg['pnr_db'])*len(cfg['methods'])
    results=[json.loads(p.read_text()) for p in sorted(root.glob('job_*/result.json'))]
    complete=(root/'completion.json').exists() and json.loads((root/'completion.json').read_text())['status']=='complete'
    if len(results)!=expected or not complete:
        raise SystemExit(f'Incomplete run: {len(results)}/{expected}; final summary refused')
    validate_jobs(results,cfg)
    out.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list);lookup={};flat=[];inclusion=[]
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
    summary=[];paired=[]
    for key,rs in groups.items():
        support,freqs=support_statistics(rs)
        inclusion.extend(dict(case=key[0],pnr=key[1],method=key[2],**row) for row in freqs)
        diff=[]
        for r in rs:
            base=lookup.get(key[:2]+('ridge',r['job']['repeat']))
            if base:diff.append(r['test_observed_db']-base['test_observed_db'])
        summary.append({'case':key[0],'pnr':key[1],'method':key[2],
            'observed_db':statistics([r['test_observed_db'] for r in rs]),
            'pim_only_db':statistics([r['test_pim_only_db'] for r in rs]),
            'paired_difference_to_ridge_db':statistics(diff) if diff else None,
            'support_jaccard_same_memory':support,
            'n_full':statistics([r['selected']['n_full'] for r in rs]),
            'n_screen':statistics([r['selected']['n_screen'] for r in rs]),
            'active':statistics([r['selected']['n_active'] for r in rs]),
            'selected_memory_counts':dict(Counter(str(r['selected']['memory']) for r in rs)),
            'selected_parameter_counts':dict(Counter(str(r['selected']['parameter']) for r in rs)),
            'selected_eta_counts':dict(Counter(str(r['selected']['eta']) for r in rs)),
            'workflow_peak_mib':statistics([r['workflow_peak_rss_bytes']/1024**2 for r in rs]),
            'inference_us_per_sample':statistics([r['inference_us_per_sample'] for r in rs])})
        for baseline in ('ridge','sbl'):
            differences=[r['test_observed_db']-lookup[key[:2]+(baseline,r['job']['repeat'])]['test_observed_db'] for r in rs]
            paired.append(dict(case=key[0],pnr=key[1],method=key[2],baseline=baseline,**statistics(differences)))
    candidate_rows=[];candidate_lookup={};failures=defaultdict(Counter)
    fields=('memory','parameter','eta','status','converged','iterations','n_full','n_screen','n_active',
            'validation_db','last_mu_relative_change','kkt_relative','score_min','score_max',
            'fit_sec','screen_sec','build_fit_validation_sec')
    for folder in sorted(root.glob('job_*')):
        j=json.loads((folder/'job.json').read_text());key=(j['case']['name'],str(j['pnr']),j['method'],j['repeat'])
        for row in json.loads((folder/'validation.json').read_text()):
            candidate_rows.append(dict(case=key[0],pnr=key[1],method=key[2],repeat=key[3],
                                       **{field:row.get(field) for field in fields}))
            failures[key[:3]]['total']+=1
            failures[key[:3]]['converged' if row.get('converged') else 'nonconverged']+=1
            candidate_lookup[key+(row['memory'],row['parameter'],row['eta'])]=row
    controlled=[]
    for key,row in candidate_lookup.items():
        case,pnr,method,rep,memory,parameter,eta=key
        if method not in ('ridge','sbl'):continue
        for threshold in cfg['eta']:
            other=candidate_lookup[(case,pnr,'screen_'+method,rep,memory,parameter,threshold)]
            both=row['converged'] and other['converged']
            controlled.append(dict(case=case,pnr=pnr,method=method,repeat=rep,memory=memory,
                parameter=parameter,eta=threshold,both_converged=both,
                n_full=row['n_full'],n_screen=other['n_screen'],
                active_without_screen=row['n_active'],active_with_screen=other['n_active'],
                validation_difference_db=other['validation_db']-row['validation_db'] if both else None))
    report={'scope':cfg['purpose'],'rows':summary,'paired_differences':paired,
        'candidate_convergence':[dict(case=k[0],pnr=k[1],method=k[2],**v) for k,v in sorted(failures.items())],
        'confidence_intervals':'Two-sided Student-t intervals over independent repetitions, not samples or support pairs; descriptive and unadjusted for multiple comparisons.',
        'parallel_timings_valid_for_method_comparison':False,
        'source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'aggregation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv(out/'selected_results.csv',flat)
    write_csv(out/'candidate_results.csv',candidate_rows)
    write_csv(out/'controlled_validation_ablation.csv',controlled)
    write_csv(out/'support_inclusion_frequencies.csv',inclusion)
    write_csv(out/'paired_test_differences.csv',paired)
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
    print(f'Validated {len(results)} completed jobs; {len(candidate_rows)} candidates; {len(controlled)} matched screening controls; wrote {out}.')


if __name__=='__main__':main()
