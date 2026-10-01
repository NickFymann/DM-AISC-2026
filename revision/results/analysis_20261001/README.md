# Completed benchmark analysis

Derived from the immutable `../benchmark_20260930_recovery_20261001/` snapshot using `../../summarize.py` and `../../render_results.py`. All 720 jobs completed, with ten repetitions in every one of 72 case/noise/method cells. All selected models converged. The original pilot and partial snapshot are excluded.

| File | Contents and interpretation |
|---|---|
| `summary.json` | All 72 cells: observed/PIM-only cancellation, means, SD, Student-t 95% intervals, counts, selected settings, convergence totals and support-overlap strata. |
| `selected_results.csv` | One row per frozen selected/tested model (720). Original concurrent timing fields are provenance, not a controlled speed comparison. |
| `candidate_results.csv` | Every candidate fit (15,840), including N_full, N_screen, N_active immediately after each stage, M/eta/parameter, convergence and validation score. Nonconverged candidates remain present. |
| `controlled_validation_ablation.csv` | 9720 screened/unscreened comparisons with matching repeat, case, noise, M and regularization. Validation difference is absent unless both converge. These are not additional test evaluations. |
| `paired_test_differences.csv` | Method-minus-ridge and method-minus-SBL test differences over matched repetitions, including mean, SD and unadjusted 95% interval. |
| `support_inclusion_frequencies.csv` | Monomial frequency within each selected-memory stratum. `eligible_runs` is the stratum denominator. Omitted monomials have zero inclusions in that stratum. Stored q5b indices put the non-conjugated high-band delay before the conjugated pair. |
| `support_at_35db.txt` | Readable sparse-method support overlaps and memory strata for the representative noise condition. |
| `serial_resources.csv` | Separate sequential profiles: repeat 0, PNR 35, all cases/methods. One fixed-model refit; inference median over seven warmed block evaluations. |

Cancellation means and SD are computed in dB over independent repetitions. Student-t intervals are descriptive and unadjusted for multiple comparisons. Jaccard pair means are descriptive: pairs sharing a repetition are dependent, and no pair-based confidence interval is reported. Comparisons use equal selected M; the number of eligible pairs can therefore be below 45. PIM-only cancellation is a simulation diagnostic, not a tuning target.

Both finite-grid selection and the common M=2/4/6 restriction are material limitations. Boundary frequencies are disclosed in `../../final_revision_decisions.md`; no post-test retuning was performed. Resource values measure this implementation on the declared representative slice, not average runtime over ten independent fits.

Full provenance is in the raw snapshot manifests, the completed integrity audit and `../serial_profile_20261001/protocol.json`. `summary.json` records the exact aggregator SHA-256 and raw manifest SHA-256.
