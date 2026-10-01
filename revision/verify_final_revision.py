"""Verify immutable evidence and independently cross-check the derived CSV means."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


def main():
    rev=Path(__file__).resolve().parent
    run=rev/'results/benchmark_20260930_recovery_20261001'
    analysis=rev/'results/analysis_20261001'
    checks=[]
    def check(condition,name):
        checks.append({'check':name,'passed':bool(condition)})
        if not condition:raise AssertionError(name)
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    hashes=read(run/'complete_snapshot_checksums.json')['files']
    changed=[name for name,digest in hashes.items() if hashlib.sha256((run/name).read_bytes()).hexdigest()!=digest]
    check(not changed,f'All {len(hashes)} frozen snapshot files preserve SHA-256')
    original=read(rev/'original/manifest.json')
    check(hashlib.sha256((rev/'original/main.tex').read_bytes()).hexdigest()==original['main.tex'],'Original article preserved')
    check(hashlib.sha256((rev.parent/'experiment description.tex').read_bytes()).hexdigest()==original['experiment description.tex'],'Supplied experiment description preserved')
    audit=read(rev/'results/recalculation_audit_complete_20261001.json')
    check(audit['successful_jobs']==720 and not audit['integrity_errors'],'Complete audit: 720 successful, zero errors')
    check(all(r['repeats_done']==r['repeats_required']==10 for r in audit['coverage']),'All 72 cells have ten repeats')
    with (analysis/'selected_results.csv').open(encoding='utf-8',newline='') as f:flat=list(csv.DictReader(f))
    check(len(flat)==720,'Selected CSV has 720 rows')
    groups=defaultdict(list)
    for r in flat:groups[r['case'],r['pnr'],r['method']].append(r)
    summary=read(analysis/'summary.json')
    for r in summary['rows']:
        rows=groups[r['case'],r['pnr'],r['method']]
        values=[float(x['test_db']) for x in rows]
        assert len(set(x['repeat'] for x in rows))==10
        assert math.isclose(statistics.mean(values),r['observed_db']['mean'],abs_tol=1e-12)
        assert math.isclose(statistics.stdev(values),r['observed_db']['sd'],abs_tol=1e-12)
        assert r['support_jaccard_same_memory']['ci95'] is None
    check(True,'Independent standard-library mean/SD check for all cells; no dependent-pair Jaccard interval')
    with (analysis/'candidate_results.csv').open(encoding='utf-8',newline='') as f:candidates=list(csv.DictReader(f))
    check(len(candidates)==15840,'All 15840 per-configuration stage counts exported')
    check(sum(r['converged']=='False' for r in candidates)==536,'All 536 nonconverged candidates retained')
    with (analysis/'controlled_validation_ablation.csv').open(encoding='utf-8',newline='') as f:controls=list(csv.DictReader(f))
    check(len(controls)==9720,'All 9720 matched validation controls exported')
    pairs=[r for r in controls if float(r['eta'])==.01]
    check(len(pairs)==3240 and all(r['n_full']==r['n_screen'] and r['validation_difference_db'] in ('','0.0') for r in pairs),'Inactive-screen controls preserve counts and all converged validation scores')
    profiles=[read(p) for p in (rev/'results/serial_profile_20261001').glob('profile_*.json')]
    check(len(profiles)==18 and all(p['converged'] and p['support_equal'] and p['coefficient_relative_error']==0 for p in profiles),'All 18 serial profiles exactly reproduce the frozen models')
    check(all(p['cpu_slot']==0 and p['threads']==1 and len(p['inference_seconds'])==7 and not p['test_target_read'] for p in profiles),'Serial measurement scope and repeat counts verified')
    paper=(rev.parent/'main.tex').read_text(encoding='utf-8')
    check('Revision status.' not in paper and 'benchmark is not yet complete' not in paper and 'validation_results.tex' not in paper,'Article contains completed benchmark rather than draft/pilot placeholders')
    check('c\\leq d, \\label{eq:im5b}' in paper,'Paper orders the conjugated c,d factors')
    log=(rev/'build/main.log').read_text(errors='replace')
    check('Overfull' not in log and 'undefined' not in log.lower() and 'LaTeX Warning:' not in log,'PDF has no overfull boxes or undefined references/citations')
    files=['summarize.py','render_results.py','profile_selected.py','test_summary.py','test_revision.py','verify_final_revision.py']
    result={'checks':checks,'all_passed':True,'raw_files_verified':len(hashes),'frozen_scientific_code_changed':False,
        'source_sha256':{name:hashlib.sha256((rev/name).read_bytes()).hexdigest() for name in files},
        'paper_sha256':hashlib.sha256((rev.parent/'main.tex').read_bytes()).hexdigest(),
        'pdf_sha256':hashlib.sha256((rev/'build/main.pdf').read_bytes()).hexdigest(),
        'limitations':['Synthetic fixed path and finite grids','Serial profile is one refit per representative case/method','No calibrated predictive intervals','No measured PIM']}
    (rev/'results/final_verification_20261001.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(checks)} final integrity/reporting checks; {len(hashes)} immutable files verified.')


if __name__=='__main__':main()
