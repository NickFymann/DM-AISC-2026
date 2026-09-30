"""Instrument the original cubic solver without changing its update order."""
from pathlib import Path
import ast
import hashlib
import json
import __future__
import numpy as np
from test_revision import original_cubic


def main():
    root=Path(__file__).resolve().parent
    out=root/'results'/'historical_diagnostics';out.mkdir(parents=True,exist_ok=False)
    nb_path=next(root.parent.glob('01_*.ipynb'))
    nb=json.loads(nb_path.read_text(encoding='utf-8'));old=original_cubic()
    source=''.join(nb['cells'][7]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='solve_sbl_pruned')
    function=ast.get_source_segment(source,node)
    function=function.replace('        gamma = 1.0 - alpha * np.real(np.diag(sigma))',
        '        _trace.append({"iteration": iteration, "active": len(active), '
        '"residual_power": float(np.mean(np.abs(y_normed-x_active@mu)**2)), '
        '"neg_log_evidence": float(-len(y_normed)*np.log(cfg.beta)-np.log(alpha).sum()'
        '+np.linalg.slogdet(system)[1]+cfg.beta*np.vdot(y_normed,y_normed).real'
        '-cfg.beta*np.vdot(rhs,mu).real), '
        '"mu_relative_change": None if prev_mu is None else float(np.linalg.norm(mu-prev_mu)/max(np.linalg.norm(prev_mu),cfg.eps))})\n'
        '        gamma = 1.0 - alpha * np.real(np.diag(sigma))')
    exec(compile(function,'instrumented_original','exec',flags=__future__.annotations.compiler_flag),old.__dict__)
    data=old.generate_standalone_data();x,y,n=old.build_problem(data,list(range(6)));k=len(y)//2
    summaries=[]
    for init in [.1,1.,10.]:
        for cap in [500,2000,5000]:
            trace=[];old.__dict__['_trace']=trace
            fit=old.solve_sbl_pruned(x[:k],y[:k],old.SblConfig(beta=1e6,alpha_init=init,max_iter=cap))
            row={'alpha_init':init,'cap':cap,'iterations':fit.iterations,'converged':bool(fit.converged),
                 'active':len(fit.active),'support':fit.active.tolist(),
                 'test_db':float(old.cancellation_db(y[k:],y[k:]-x[k:]@fit.h)),
                 'last_mu_relative_change':trace[-1]['mu_relative_change']}
            name=f'init_{init:g}_cap_{cap}.json';(out/name).write_text(json.dumps(trace,indent=2)+'\n')
            row['trajectory']=name;summaries.append(row)
            print(json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps({'source_sha256':hashlib.sha256(nb_path.read_bytes()).hexdigest(),
        'scope':'Historical split and exact update order; diagnostic only, not a new independent test estimate.',
        'rows':summaries},indent=2)+'\n')


if __name__=='__main__':main()
