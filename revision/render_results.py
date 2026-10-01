"""Generate manuscript tables directly from audited benchmark summaries."""
import argparse
import csv
import json
from pathlib import Path

NAMES={'ridge':'Ridge-LS','sbl':'SBL','screen_ridge':'Screen + ridge',
       'screen_sbl':'Screen + SBL','lasso':'Complex LASSO','omp':'Complex OMP'}
CASES={'cubic_c3':'Cubic / cubic','tanh_c3':'Tanh / cubic','tanh_c35':'Tanh / cubic--quintic'}


def mean_sd(s,places=2):
    return f"${s['mean']:.{places}f}\\pm{s['sd']:.{places}f}$"


def render(analysis,generated,profile=None):
    report=json.loads((analysis/'summary.json').read_text())
    rows={(r['case'],r['pnr'],r['method']):r for r in report['rows']}
    assert len(rows)==72 and all(r['observed_db']['n']==10 for r in rows.values())
    generated.mkdir(parents=True,exist_ok=True)
    lines=[r'All 720 independent selection/test tasks completed, covering 15,840 candidate fits. Each cell in Table~\ref{tab:benchmark} summarizes ten independently generated fitting/validation/test triplets. The reported dispersion is the sample standard deviation across repetitions; means are computed in dB. The supplement also provides PIM-only scores, unadjusted 95\% Student-$t$ intervals and paired method differences.',
           r'\begin{table*}[!t]',
           r'\caption{Untouched-test observed cancellation (dB), mean $\pm$ SD over ten repetitions. All six methods use the same fitting/validation/test records and the frozen grids in Section~\ref{sec:selection}.}',
           r'\label{tab:benchmark}\centering\small\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{llrrrr}\toprule',
           r'Source / dictionary & Method & Clean & PNR 20 dB & PNR 35 dB & PNR 50 dB\\\midrule']
    for case,label in CASES.items():
        for number,method in enumerate(NAMES):
            values=[mean_sd(rows[case,str(pnr),method]['observed_db']) for pnr in (None,20,35,50)]
            lines.append((label if number==0 else '')+' & '+NAMES[method]+' & '+' & '.join(values)+r'\\')
        if case!='tanh_c35':lines.append(r'\midrule')
    lines += [r'\bottomrule\end{tabular}\end{table*}',
        r'\begin{table}[!t]',r'\caption{Mean dictionary sizes at PNR 35 dB over ten independently selected models. Fractional counts reflect different selections across repetitions. Per-configuration counts for all 15,840 fits are supplied in \texttt{candidate\_results.csv}.}',
        r'\label{tab:stage-counts}\centering\scriptsize',
        r'\begin{tabular}{lrrr}\toprule',r'Method & $N_{\rm full}$ & $N_{\rm screen}$ & $N_{\rm active}$\\\midrule']
    for case,label in CASES.items():
        lines.append(r'\multicolumn{4}{l}{\emph{'+label+r'}}\\')
        for method in NAMES:
            r=rows[case,'35',method]
            lines.append(NAMES[method]+' & '+' & '.join(f"{r[k]['mean']:.1f}" for k in ('n_full','n_screen','active'))+r'\\')
        if case!='tanh_c35':lines.append(r'\midrule')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    def value(case,pnr,method,key='observed_db',field='mean'):
        return rows[case,str(pnr),method][key][field]
    lines += [
        f"For the clean cubic source, tuned ridge reaches {value('cubic_c3',None,'ridge'):.2f} dB, compared with {value('cubic_c3',None,'sbl'):.2f} dB for SBL and {value('cubic_c3',None,'omp'):.2f} dB for OMP. Thus the revised comparison does not support an unconditional cancellation advantage for SBL. At PNR 35 dB, ridge and SBL give {value('cubic_c3',35,'ridge'):.2f} and {value('cubic_c3',35,'sbl'):.2f} dB, while SBL uses {value('cubic_c3',35,'sbl','active'):.1f} active coefficients on average instead of 126.",
        f"For tanh at PNR 35 dB, increasing the dictionary from cubic to cubic--quintic raises ridge cancellation from {value('tanh_c3',35,'ridge'):.2f} to {value('tanh_c35',35,'ridge'):.2f} dB. In the combined dictionary, screen-plus-SBL gives {value('tanh_c35',35,'screen_sbl'):.2f} dB with {value('tanh_c35',35,'screen_sbl','active'):.1f} active coefficients on average, compared with {value('tanh_c35',35,'ridge'):.2f} dB and 4788 coefficients for ridge. The selected memories and thresholds can differ between methods; these are comparisons of validation-tuned pipelines."
    ]
    (generated/'benchmark_results.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    support=[]
    for case in CASES:
        for method in ('sbl','screen_sbl','lasso','omp'):
            r=rows[case,'35',method]['support_jaccard_same_memory']
            support.append(f"{case}: {method}: Jaccard={r['mean']:.6f}, pairs={r['n_pairs']}, strata={r['strata']}")
    (analysis/'support_at_35db.txt').write_text('\n'.join(support)+'\n',encoding='utf-8')
    if profile:
        completion=json.loads((profile/'completion.json').read_text())
        assert completion['status']=='complete' and completion['jobs']==18
        profiles=[json.loads(p.read_text()) for p in sorted(profile.glob('profile_*.json'))]
        assert len(profiles)==18 and all(p['support_equal'] and p['converged'] for p in profiles)
        lines=[r'\begin{table*}[!t]',
            r'\caption{Sequential resource measurements: one refit of the frozen selected model per method/case, repetition 0 at PNR 35 dB. Times exclude hyperparameter search. Inference is the median of seven warmed block evaluations, including active-feature construction and receiver filtering. Peak is the process high-water RSS.}',
            r'\label{tab:resources}\centering\scriptsize\setlength{\tabcolsep}{3pt}',
            r'\begin{tabular}{llrrrrrr}\toprule',
            r'Source / dictionary & Method & $M$ & Active & Dictionary (s) & Fit (s) & Peak (MiB) & Inference ($\mu$s/sample)\\\midrule']
        for i,p in enumerate(profiles):
            job=p['job'];case=job['case']['name']
            lines.append((CASES[case] if i%6==0 else '')+' & '+NAMES[job['method']]+f" & {p['memory']} & {p['n_active']} & {p['dictionary_sec']:.3f} & {p['fit_sec']:.3f} & {p['process_peak_mib']:.1f} & {p['inference_median_us_per_sample']:.2f}"+r'\\')
            if i in (5,11):lines.append(r'\midrule')
        lines += [r'\bottomrule\end{tabular}\end{table*}']
        (generated/'resource_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        cols=['dictionary_sec','fit_sec','train_total_sec','process_peak_mib','inference_median_us_per_sample','n_full','n_screen','n_active','coefficient_relative_error']
        with (analysis/'serial_resources.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['case','method','memory']+cols);w.writeheader()
            w.writerows(dict(case=p['job']['case']['name'],method=p['job']['method'],memory=p['memory'],**{k:p[k] for k in cols}) for p in profiles)
    print('Rendered completed benchmark tables'+(' and serial resources' if profile else ''))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--generated',type=Path,required=True)
    parser.add_argument('--profile',type=Path)
    args=parser.parse_args();render(args.analysis,args.generated,args.profile)
