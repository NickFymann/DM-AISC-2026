"""Copy completed jobs from a running benchmark into an immutable partial snapshot.

Only the atomic progress journal defines the cutoff. Active jobs keep running and
are excluded, even if they finish while the snapshot is being copied.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def snapshot(run, output):
    progress_bytes = (run/'progress.json').read_bytes()
    progress = json.loads(progress_bytes)
    manifest = read(run/'manifest.json')
    cfg = manifest['config']
    total = cfg['repeats'] * len(cfg['cases']) * len(cfg['pnr_db']) * len(cfg['methods'])
    if len({row['index'] for row in progress}) != len(progress):
        raise ValueError('Duplicate job indices in progress')
    completed = [row for row in progress if row['status']=='ok']
    failed = [row for row in progress if row['status']!='ok']
    for row in completed:
        result = read(run/f"job_{row['index']:04d}/result.json")
        if result['status'] != 'ok' or result['job'] != row['job']:
            raise ValueError('Progress and result mismatch')
    for name, digest in manifest['source_sha256'].items():
        if hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen source hash mismatch: '+name)
    output.mkdir(parents=True, exist_ok=False)
    (output/'progress.json').write_bytes(progress_bytes)
    for name in ('manifest.json', 'handoff.json', 'handoff_4workers.json',
                 'parallel_launch.json', 'parallel_status.json', 'timing_provenance.json'):
        if (run/name).exists():
            shutil.copy2(run/name, output/name)
    (output/'source').mkdir()
    for name in list(manifest['source_sha256'])+['config.json']:
        shutil.copy2(run/'source'/name, output/'source'/name)
    results = []
    for row in completed:
        folder = f"job_{row['index']:04d}"
        shutil.copytree(run/folder, output/folder, ignore=shutil.ignore_patterns('__pycache__'))
        result = read(output/folder/'result.json')
        job, chosen = result['job'], result['selected']
        results.append(dict(index=row['index'], repeat=job['repeat'], case=job['case']['name'],
            pnr_db=job['pnr'], method=job['method'], memory=chosen['memory'],
            parameter=chosen['parameter'], eta=chosen['eta'],
            validation_db=chosen['validation_db'], test_observed_db=result['test_observed_db'],
            test_pim_only_db=result['test_pim_only_db'],
            n_full=chosen.get('n_full'), n_screen=chosen.get('n_screen'), n_active=chosen['n_active'],
            converged=chosen['converged'], tuning_sec=result['tuning_sec'],
            inference_us_per_sample=result['inference_us_per_sample'],
            workflow_peak_rss_bytes=result['workflow_peak_rss_bytes']))
    if results:
        with (output/'interim_results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    info = dict(status='partial' if len(completed)<total else 'complete',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_run=str(run), expected_jobs=total, completed_jobs=len(completed),
        failed_jobs=len(failed), completed_indices=[row['index'] for row in completed],
        cutoff='One atomic progress.json read; only its successful jobs are included.',
        benchmark_keeps_running=True, performance_comparison_valid=False,
        statistical_conclusions_ready=len(completed)==total,
        warning='Partial and uneven scenario/repetition coverage. No final means, confidence intervals, '
                'method rankings or efficiency claims. Timing spans different concurrency phases.')
    dump(output/'snapshot.json',info)
    groups = collections.Counter((r['case'],r['method']) for r in results)
    lines=['# Промежуточный снимок расчёта', '',
        f"Сохранено {len(completed)} из {total} задач; неудачных к моменту среза: {len(failed)}.", '',
        'Расчёт на сервере продолжается. Включены только завершённые задачи из одного чтения журнала.',
        'Работающие задачи не скопированы. Набор повторов пока неполон; итоговые статистические выводы делать рано.',
        'Время получено при разных режимах параллелизма и не является окончательным сравнением эффективности.', '',
        '| Сценарий | Метод | Завершено | Всего запланировано |', '|---|---|---:|---:|']
    for case in cfg['cases']:
        for method in cfg['methods']:
            lines.append(f"| {case['name']} | {method} | {groups[(case['name'],method)]} | {cfg['repeats']*len(cfg['pnr_db'])} |")
    lines += ['', '`interim_results.csv` содержит результаты отдельных завершённых задач.',
        '`source/` содержит точные исходники и конфигурацию серии; `checksums.json` проверяет все файлы снимка.',
        'Для живого состояния использовать серверный `parallel_status.json`; его локальная копия — только снимок.', '']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    checksums={path.relative_to(output).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(output.rglob('*')) if path.is_file()}
    dump(output/'checksums.json',checksums)
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    info=snapshot(args.run.resolve(),args.output.resolve())
    if args.archive:
        archive=Path(shutil.make_archive(str(args.output.resolve()),'gztar',
                     root_dir=args.output.resolve().parent,base_dir=args.output.name))
        info['archive']=str(archive);info['archive_bytes']=archive.stat().st_size
        info['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    print(json.dumps(info))


if __name__=='__main__':
    main()
