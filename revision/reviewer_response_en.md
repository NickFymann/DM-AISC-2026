# Response to the reviewers — DM-AISC-2026

The revised manuscript reports a completed synthetic benchmark: 720 successful selection/test tasks, 15,840 candidate fits and 18 separate serial resource profiles. The contribution is a controlled study of screening, sparsity and cost. We do not claim a new Bayesian inference principle, universal superiority or calibrated predictive uncertainty.

## 1. Reproducibility

The Experimental Rationale and Setup section now specifies the generator, 10240-sample records, 256 MHz sampling, carriers at −30/+30 MHz, 60 MHz spacing, transmit filter cutoff 6.4 MHz and nominal two-sided bandwidth 12.8 MHz, the causal 11-tap complex path with every coefficient, receiver cutoff 25.6 MHz, and discard of 4112 samples. Both transmit components are filtered versions of one white sequence, not independently generated OFDM transmissions. The supplied experiment description is incorporated concisely and reconciled with the code.

We specify G=1.2+3.7j, gamma=2, epsilon=10⁻¹², fitting-only normalization, screening thresholds, SBL initialization/pruning/floors, stopping ratio/tolerance, iteration limits, LASSO KKT tolerance, OMP budgets, every tuning grid and the exact cancellation metric. Each independent fitting/validation/test record contributes 3064 scored samples. Seed formulas, hardware/software versions, BLAS settings and measurement scope are explicit. Frozen source/configuration manifests and per-job artifacts accompany the manuscript.

## 2. Independent selection and test evaluation

Each repetition has separate fitting, validation and test records with independent filter histories and stage seeds. Normalization and screening use fitting samples only. Selection maximizes observed validation cancellation among converged finite candidates; exact ties prefer fewer active coefficients, then smaller memory. Selection and coefficients are frozen before the test record is generated. No fitting-plus-validation refit is performed.

The completed audit verifies all 720 task identities, candidate grids, stage counts and selected winners against this rule. The main comparison now gives untouched-test means and sample standard deviations over ten repetitions. Original test-selected values are explicitly exploratory diagnostics and are not pooled with the new estimates. Later inspection of grid-boundary frequencies does not change the models, test data or rankings.

## 3. Screening and ablation

We report historical full/screened counts and mean full/screened/active counts for the independently selected models. `candidate_results.csv` supplies these three counts for every one of the 15,840 configurations, including all memories, thresholds, regularization settings, noise conditions and repetitions.

The main test table compares ridge-LS, SBL without screening, screening plus ridge-LS, and screening plus SBL, alongside LASSO and OMP. `controlled_validation_ablation.csv` additionally contains 9720 comparisons matched on memory and regularization. At eta=0.01, all 3240 matched ridge/SBL pairs have identical validation scores, stage counts and convergence flags: no column is removed. Compactness at that threshold is therefore attributable to relevance pruning. Higher thresholds can reduce the dictionary, as documented by the per-configuration counts.

For tanh/cubic–quintic at 35 dB, screen-plus-SBL minus SBL has a mean paired test difference of 0.039 dB with descriptive 95% interval [−0.027, 0.106] dB. This does not establish a cancellation gain. The matched fixed-parameter controls use existing validation observations; the tuned test branches evaluate complete pipelines. We do not conflate these two ablations or add unplanned test evaluations.

## 4. Baselines and computational cost

All six methods use the same fitting and validation data. Ridge is tuned over six values from 10⁻⁸ to 10⁻³, replacing the fixed 10⁻⁴ baseline. Conventional fixed-noise SBL, complex LASSO and complex OMP use explicitly reported grids and stopping criteria. On the clean cubic source, ridge obtains 58.02±0.87 dB versus 51.96±0.65 dB for SBL; the previous general cancellation-advantage claim is removed.

The grid-boundary audit is reported candidly: LASSO chooses its smallest penalty in 118/120 tasks, SBL its largest beta in 102/120, and ridge the lower/upper regularization boundary in 23/120 and 46/120 tasks. Finite-grid tuning does not prove globally optimal tuning. We preserve the predeclared comparison and restrict the conclusions instead of enlarging grids after examining test scores. M=2/4/6 is common to every method and case; combined M=8 exceeds the stated dense-workspace budget.

The resource table uses a separate serial protocol declared before measurement: repetition 0, PNR 35 dB, all three cases and six methods, one refit each, fresh processes, CPU 0 and one BLAS thread. It reports dictionary time, fit time, process high-water RSS and inference cost including active-feature construction and receiver filtering. Inference uses seven measurements after one warm-up. All 18 refits reproduce the frozen coefficients exactly. These are representative implementation measurements; full hyperparameter search is excluded and concurrent-search timings are not used for speed comparisons.

In the combined representative case, screening reduces SBL fit time from 243.41 to 23.47 s and peak RSS from 3439 to 1125 MiB, while ridge fits in 15.65 s. Sparse inference is cheaper in this implementation. We therefore separate coefficient count, fitting cost and deployment cost, without claiming universal efficiency.

## 5. Repetitions, noise and support stability

All 72 combinations of source/dictionary, noise condition and method have ten independent signal/noise realizations. The conditions are clean and PNR 20/35/50 dB. Main tables give mean±sample SD in dB; supplementary files include PIM-only scores and paired differences with unadjusted Student-t intervals over repetitions. These descriptive intervals are not simultaneous multiple-comparison bounds.

Support stability uses monomial identifiers, conditional on equal selected memory. We report descriptive Jaccard means, eligible pair counts, memory strata and inclusion frequencies. Pairs sharing a repetition are dependent: no confidence interval treats the 45 possible pairs as 45 independent runs. At 35 dB, mean Jaccard is 0.364 for cubic SBL and 0.095 for combined screen-plus-SBL. Similar cancellation thus does not establish unique physical support recovery. Measured PIM remains future work; claims are limited to a fixed-path synthetic study.

## 6. Iteration cap and stability

We reproduced the original solver without altering its historical alpha/mean update order. For cubic M=6, beta=10⁶ and alpha initialization 1, cap 500 gives 51.264645 dB with final relative mean change 2.4041×10⁻⁵. Cap 2000 converges at iteration 1623 to 51.265265 dB with final change 9.9881×10⁻⁷. Initializations 0.1, 1 and 10 converge at iterations 1603, 1623 and 1612 to the same seven-term support. Cap 5000 gives the same stopping points. The article reports objective/residual checkpoints and final changes; all nine complete diagnostic trajectories are supplied.

These are stability diagnostics on the historical record, not unbiased test estimates. All 720 final benchmark models converged. There were 536 nonconverged candidates excluded from selection: 238 LASSO, 86 SBL and 212 screened SBL. No repetition was dropped. Full trajectories accompany all 240 selected SBL models; unselected candidates retain final stopping diagnostics, not complete trajectories.

## 7. Conjugated indices

Equation (6) now uses conjugated high-band indices c,d, non-conjugated index e, and restrictions a≤b and c≤d, matching the reviewer's notation and Equation (7)'s count. The stored `q5b` tuple puts the non-conjugated high-band index before the ordered conjugated pair: paper `(a,b,c,d,e)` maps to stored `(a,b,e,c,d)`. The implementation already orders the conjugated pair. A direct monomial/filtering test verifies the mapping, in addition to count and permutation-symmetry tests. No scientific result changed through this relabeling.

## 8. Uncertainty and prior work

We removed predictive uncertainty-quantification claims. Posterior inference estimates sparse coefficients; experimental confidence intervals summarize repeated-run variability and are not posterior predictive intervals.

The related-work discussion distinguishes frequency-domain OFDM amplifier-regressor selection (Crespo-Cadenas et al., 2025), SBL with BIC/PESR refinement for nonlinear system identification (Wei, 2025), multi-band Wiener–Hammerstein PIM reconstruction (Liu et al., 2024), and measured PIM/cascade work by Lampu et al. SBL and correlation screening are explicitly identified as established tools. Our contribution is their controlled comparison in the specified receiver-filtered PIM dictionary, including negative results, resource costs and limitations. Published cancellation values from different targets are not presented as direct baselines.

## Evidence package

- `main.tex` and `revision/build/main.pdf`: revised article.
- `revision/results/benchmark_20260930_recovery_20261001/`: complete raw benchmark; 4578 source/result files verified against snapshot hashes.
- `revision/results/analysis_20261001/`: complete candidate, selection, ablation, support-frequency and paired-difference tables.
- `revision/results/serial_profile_20261001/`: declared profiling protocol and 18 measurements.
- `revision/results/historical_diagnostics/`: cap/initialization trajectories.
- `revision/results/recalculation_audit_complete_20261001.json`: complete integrity/coverage audit.

The original article, supplied experiment description, pilot and earlier partial snapshot remain preserved. The preceding intermediate draft is retained with hashes in `revision/history/pre_finalization_20261001/`.
