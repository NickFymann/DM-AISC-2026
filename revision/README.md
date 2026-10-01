# Ревизия DM-AISC-2026

На 01.10.2026 основная серия и отдельное профилирование завершены. Статья обновлена по восьми замечаниям: 720 успешных задач, 15 840 подгонок, по 10 независимых повторов в 72 сочетаниях условий. Все выбранные модели сошлись; 536 несошедшихся кандидатов исключены из выбора. Подготовлен английский ответ рецензентам.

## Статья и выводы

- [Текст](../main.tex), [PDF](build/main.pdf), [ответ рецензентам](reviewer_response_en.md).
- [Состояние восьми пунктов](reviewer_response.md), [решения по границам эксперимента](final_revision_decisions.md).
- [Исходный план](dm_aisc_2026_review_plan.md), [предыдущий аудит 72/720](recalculation_audit_20260930.md) — исторические документы, не текущий статус.

SBL даёт компактные модели, но общего преимущества по подавлению нет. На чистом cubic ridge: 58.02±0.87 дБ, SBL: 51.96±0.65 дБ. При PNR 35 дБ — около 35 дБ, со средним числом коэффициентов 126 и 7.8. Для tanh/cubic–quintic при 35 дБ screen+SBL: 28.33±0.46 дБ и 24 коэффициента, ridge: 28.31±0.50 дБ и 4788. Пары Jaccard зависимы; для них сообщаются описательные средние без ложного доверительного интервала.

## Данные и проверки

- [Полный сырой запуск](results/benchmark_20260930_recovery_20261001/): 4578 файлов сверены с `complete_snapshot_checksums.json`; сохранены точные исходники, конфигурация, validation, selection, model, test и 240 выбранных SBL-траекторий. Статус завершения — `completion.json`; исходный manifest не переписывается.
- [Итоговый аудит](results/recalculation_audit_complete_20261001.json): все 720 идентификаторов, сетки и правила выбора корректны; каждый из 72 наборов имеет 10 повторов; ошибок целостности нет.
- [Обработка](results/analysis_20261001/README.md): 15 840 строк кандидатов с N_full/N_screen/N_active, 9720 согласованных validation-контролей, mean/SD/CI, парные тестовые разности и частоты включения мономов.
- [Последовательное профилирование](results/serial_profile_20261001/): по одному повтору обучения в 18 заранее определённых сочетаниях (repeat=0, PNR=35, три сценария и шесть методов), CPU 0, один BLAS-поток. Все коэффициенты совпали точно. Семь замеров inference после прогрева; Linux high-water RSS всего процесса, без поиска гиперпараметров. Это репрезентативные замеры, не статистика времени по десяти реализациям.
- [Историческая сходимость](results/historical_diagnostics/summary.json): caps 500/2000/5000, alpha_init 0.1/1/10, исходный порядок обновления. Итог 51.265265 дБ, одинаковые семь мономов.
- [Сверка с сохранённым MATLAB-эталоном](results/reference_verification.json): сам MATLAB-скрипт не объявляется выполненным.

Научные исходники основного запуска не изменялись. `experiment.py` независимо реализует генератор и словари; архив не импортируется. В новом SBL кешируется Gram и после последнего alpha-update пересчитывается posterior. Исторические пробы сохраняют старый порядок. В текущем этапе изменены статистическая агрегация, обёртка профилирования и текст.

## Воспроизведение из корня Huawei

```powershell
$env:PYTHONUTF8='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv\Scripts\python.exe -B DM-AISC-2026\revision\test_revision.py
.venv\Scripts\python.exe -B DM-AISC-2026\revision\test_summary.py
.venv\Scripts\python.exe -B DM-AISC-2026\revision\summarize.py DM-AISC-2026\revision\results\benchmark_20260930_recovery_20261001 --output DM-AISC-2026\revision\results\analysis_reproduced
.venv\Scripts\python.exe -B DM-AISC-2026\revision\render_results.py --analysis DM-AISC-2026\revision\results\analysis_reproduced --generated DM-AISC-2026\revision\generated --profile DM-AISC-2026\revision\results\serial_profile_20261001
```

Новый численный эксперимент: `run_experiments.py --config benchmark.json --output <новая папка>`. Runner не перезаписывает существующую папку. Исходный config содержит часовой runtime-лимит; шесть тяжёлых задач потребовали восстановления с лимитом три часа (`prepare_recovery.py`, `parallel_continue_v2.py`). Это изменение диспетчера, не научных параметров.

Сборка PDF из `DM-AISC-2026`: `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=revision/build main.tex`.

## История и сохранность

Исходные `original/main.tex` и пользовательский `experiment description.tex` сохраняют исходные SHA-256. Промежуточная редакция и прежние описания сохранены в [history/pre_finalization_20261001](history/pre_finalization_20261001/manifest.json). Пилот и снимок 67/720 являются отдельными историческими версиями, их данные не входят в итоговые средние.

Удалённый корень: `/home/agent/4lchem1st/dm-aisc-2026-revision` на `4l-ru-a`. Основной запуск завершил 714 задач и шесть часовых тайм-аутов; recovery сохранил 4518 файлов успешных задач побайтово и досчитал только шесть пропусков. Полная серия завершилась 01.10.2026 в 15:47:53 МСК. Диспетчеры и профилирование завершились естественно. История последовательного и 3/4-ядерного режимов сохранена в handoff/timing provenance; эти смешанные времена не используются для сравнения скорости методов.
