"""Deterministic first revision from the preserved user-supplied manuscript."""
from pathlib import Path
root=Path(__file__).resolve().parent.parent
original=(root/'revision/original/main.tex').read_text(encoding='utf-8')
prefix=original[:original.index(r'\begin{abstract}')]
prefix=prefix.replace('Correlation-Pruned Fixed-Noise Sparse Bayesian Learning for Passive Intermodulation Cancellation',
                      'Fixed-Noise Sparse Bayesian Learning for Passive Intermodulation Cancellation')
prefix=prefix.replace(r'\usepackage{balance}', '')
model=original[original.index(r'\section{PIM Regression Model}'):original.index(r'\section{Experimental Rationale and Setup}')]
model=model.replace('Normalization makes both the screen and the Bayesian updates invariant to individual column scale and prevents the SBL matrix operations from being spent on negligible candidates.',
    'Normalization makes the screen and Bayesian updates invariant to individual column scale. Whether screening actually removes candidates must be measured; it is not implied by the presence of the screening step.')
model=model.replace(r'\rho=10^{-4},',r'\rho>0,')
model=model.replace('Thus, $10^{-4}$ is a relative ridge factor, whereas the actual penalty $\\lambda_X$ is scaled by the mean diagonal level of the sample covariance.',
    'The historical sweeps used $\\rho=10^{-4}$; the revised protocol selects $\\rho$ on validation data. The actual penalty $\\lambda_X$ is scaled by the mean diagonal level of the fitting covariance.')
model=model.replace('the experiments repeat the same solver at $\\beta\\in\\{10^2,10^4,10^6\\}$',
                    'the historical sweeps repeat the same solver at $\\beta\\in\\{10^2,10^4,10^6\\}$')
model=model.replace('clips the resulting precision to a finite range.',
                    'clips the resulting precision to $[10^{-12},10^{16}]$, with $\\varepsilon=10^{-12}$.')
model=model[:model.index('Algorithm~\\ref{fig:algorithm} has two distinct sparsification stages.')]+r'''
Algorithm~\ref{fig:algorithm} separates one-shot screening from iterative relevance pruning. In the historical experiments the first stage retains every candidate at $\eta=0.01$; sparsity is therefore attributable to the second stage. The revised solver caches the Gram matrix on each active set, uses Hermitian solves, and recomputes the final posterior after the last precision update. This last operation is an explicit numerical variant; the historical convergence diagnostic below preserves the original update order.

The two conjugated high-band factors in \eqref{eq:im5b} carry indices $d,e$, hence $d\leq e$. Under the alternative naming $\overline{x_{h,c}}\overline{x_{h,d}}x_{h,e}$, the same symmetry restriction would be $c\leq d$. The implementation orders the conjugated pair and removes permutation duplicates; it does not assume that the resulting numerical columns are linearly independent.
'''
intro=r'''\begin{abstract}
We study fixed-noise sparse Bayesian learning (SBL) for complex Volterra-type passive intermodulation (PIM) cancellation. Cubic and phase-equivariant tanh sources separate nonlinear-order matching from deliberate model mismatch. A direct audit of the historical experiments shows that one-shot correlation screening at $\eta=0.01$ retains every candidate: the observed sparsity is produced by Bayesian relevance pruning. The historical cubic $M=6$ result is stable under a higher iteration cap, converging to 51.2653 dB with seven terms for three initial precision values. These values are diagnostic results on the original split, whose operating points were selected with access to test performance; they are not unbiased estimates of a tuned method. We specify a separate fitting, validation and test protocol, sparse alternatives, screening ablations and repeated noise conditions to support a revised comparison. Posterior inference is used for sparse coefficient estimation; predictive uncertainty calibration is not claimed.
\end{abstract}
\begin{IEEEkeywords}
passive intermodulation, sparse Bayesian learning, Volterra model, correlation screening, nonlinear system identification
\end{IEEEkeywords}

\section{Introduction}\label{sec:intro}
Passive intermodulation (PIM) arises when strong transmitted signals excite weak nonlinearities in nominally passive structures. In frequency-division-duplex radios, products generated in connectors, antennas or external objects can overlap the receiver band \cite{Lui1990PIM,Lampu2021AirInduced,Lampu2023Algorithms,Ahmmed2023Overview}. Digital cancellation reconstructs these products from known transmit samples and subtracts the replica from the received waveform.

Volterra and memory-polynomial models express nonlinear interactions through a regression linear in its coefficients, but their size grows rapidly with nonlinear order and memory \cite{Schetzen1980Volterra,Morgan2006GMP,Waheed2020PIM}. Physical and cascaded structures reduce this cost \cite{Lampu2023Cascade}. Wiener--Hammerstein models also address multi-band PIM \cite{Liu2024MultiBand}. Sparse Bayesian learning (SBL) is an established basis-selection method \cite{Tipping2001SBL,Wipf2004SBL}, including recent frequency-domain nonlinear RF regression \cite{Crespo2025SparseRF} and system identification from limited data \cite{Wei2025Identification}. Those studies select regressors or system models. Here we isolate the contribution of target-correlation screening in a PIM dictionary with receiver filtering. Applying SBL or placing a correlation test before it is not, by itself, a new inference principle.

Our question is narrower: for a frequency-selected complex PIM dictionary, does an inexpensive target-correlation screen contribute to compactness or cost beyond conventional relevance pruning? We examine symmetry-reduced cubic and cubic--quintic dictionaries, distinguish source-order mismatch from finite-memory truncation, and explicitly separate the sizes before screening, after screening and after pruning. Historical single-realization results provide reproduction controls. Independent validation-based selection and repeated comparisons are required before cancellation or efficiency advantages can be claimed.

\noindent\textbf{Revision status.} The reproduction and convergence diagnostics reported here are complete. The new multi-realization benchmark is being implemented and validated; its results must replace the historical comparison before submission. The protocol below describes the revised experiment, not completed evidence of superiority.

'''
setup=r'''
\section{Experimental Rationale and Setup}\label{sec:setup}
\subsection{Synthetic Signal and Receiver Chain}
Each record contains $N_s=10240$ complex samples at $f_s=256$ MHz. A single sequence $u[n]=a[n]+jb[n]$ is drawn using NumPy RandomState, with independent $a,b\sim U[0,1)$. It is neither centered nor normalized in power. A 2049-tap Hamming low-pass prototype $h_0$ is generated by \texttt{firwin(2049,0.05)} with unit DC gain. The transmit components are
\begin{equation}
 z_q[n]=\sum_{k=0}^{2048}h_0[k]e^{j2\pi f_qk/f_s}u[n-k].
 \label{eq:tx-generation}
\end{equation}
Here $f_\ell=-30$ MHz and $f_h=30$ MHz. Thus the carrier separation is 60 MHz, the prototype cutoff is 6.4 MHz, and its nominal two-sided bandwidth is 12.8 MHz; these are filter specifications, not measured occupied bandwidth. Both components are obtained from one white sequence by modulating the filter coefficients, rather than by modulating one already filtered waveform. They are not independently generated OFDM transmissions.

The composite $z_\ell+z_h$ passes through a causal 11-tap complex FIR,
\begin{equation}
 s[n]=\sum_{m=0}^{10}h[m](z_\ell[n-m]+z_h[n-m]).
 \label{eq:linear-path}
\end{equation}
Table~\ref{tab:path} specifies the fixed synthetic path. There are no negative delays, centered convolution or group-delay compensation. The available regression references are $x_q[n]=z_q[n]e^{-j2\pi f_qn/f_s}$, before this path; the path therefore contributes effective memory to the target.
\begin{table}[!t]
\caption{Coefficients of the causal synthetic linear path.}\label{tab:path}
\centering\scriptsize
\begin{tabular}{rr@{\qquad}rr}\toprule
$m$ & $h[m]$ & $m$ & $h[m]$\\\midrule
0 & $0.186-0.023j$ & 6 & $1.633-1.725j$\\
1 & $-0.098+0.117j$ & 7 & $-1.415+1.365j$\\
2 & $0.291-0.338j$ & 8 & $0.760-0.668j$\\
3 & $-0.607+0.628j$ & 9 & $-0.256+0.211j$\\
4 & $1.131-1.348j$ & 10 & $0.045-0.037j$\\
5 & $-1.090+0.993j$ & & \\\bottomrule
\end{tabular}
\end{table}

The nonlinear output is shifted by $+90$ MHz and passed through a causal 2049-tap Hamming receiver low-pass, \texttt{firwin(2049,0.2)}, with cutoff 25.6 MHz. This selects the product centered at $2f_\ell-f_h=-90$ MHz. Every dictionary monomial undergoes the same receiver filter. The first $D=2\cdot2048+16=4112$ samples are discarded, leaving 6128 post-transient samples per generated record.

\subsection{Nonlinear Sources and Added Noise}
The order-matched control and bounded surrogate are, respectively,
\begin{align}
 g_3(s)&=G s|s|^2,\label{eq:pure-cubic}\\
 g_{\tanh}(s)&=G\tanh(\gamma|s|)\frac{s}{|s|+\epsilon},\label{eq:tanh}
\end{align}
with $G=1.2+3.7j$, $\gamma=2$ and $\epsilon=10^{-12}$. Apart from the common phase rotation in $G$, the second mapping preserves input phase. For $|s|\gg\epsilon$, its local series is
\begin{equation}
 g_{\tanh}(s)=G\left(\gamma s-\frac{\gamma^3}{3}s|s|^2
 +\frac{2\gamma^5}{15}s|s|^4+\cdots\right).
 \label{eq:tanh-series}
\end{equation}
The leading small-signal response is linear, not cubic; receiver-band selection suppresses the linear component. Higher-order measurements and connector models motivate studying departures from a cubic law \cite{Christianson2008HigherOrder,Henrie2008Prediction}. However, this radial complex-baseband tanh mapping is a controlled surrogate, not a demonstrated exact baseband equivalent of a measured real-valued connector I--V law. Even the cubic source need not lie exactly in a dictionary with truncated memory $M<11$.

For noisy records, circular complex AWGN is added after the receiver filter: $y=p+v$, with $v=\sqrt{\sigma_v^2/2}(a_v+jb_v)$ and independent standard-normal components. Its variance is $\sigma_v^2=P_p10^{-L/10}$, where $P_p$ is clean PIM power over the 6128 post-transient samples and $L$ is the configured PIM-to-noise ratio. This simulation calibration does not enter model selection as a clean target. The historical condition is $L=35$ dB; the revised grid adds 20 and 50 dB and a clean condition.

\subsection{Cancellation Metric and Historical Split}
For an evaluation index set $\mathcal I$, cancellation depth is
\begin{equation}
 C_{\mathcal I}=10\log_{10}\frac{\operatorname{mean}_{n\in\mathcal I}|y[n]|^2}
 {\operatorname{mean}_{n\in\mathcal I}|y[n]-\widehat p[n]|^2+10^{-30}}.
 \label{eq:metric}
\end{equation}
Observed cancellation uses $y=p+v$; PIM-only cancellation replaces $y$ by $p$ and is a simulation diagnostic. Selection always uses observed validation cancellation. Neither $R^2$ nor an amplitude-spectrum ordinate substitutes for \eqref{eq:metric}.

The historical notebooks use TX seed 12345 and noise seed 54321. Their fitting and test intervals are $[4112,7176)$ and $[7176,10240)$, with 3064 samples each and no validation set. Several operating points were selected by maximizing the displayed test score. These values are exploratory and may be optimistically biased; relabeling that segment cannot repair the selection protocol.

\subsection{Independent Selection and Repetition Protocol}\label{sec:selection}
The revision generates three independent records for fitting, validation and test. To retain the original fitting count, each stage uses the first 3064 post-transient samples, $[4112,7176)$, of its own 10240-sample record; the remaining samples are not scored. Each record starts with its own filter transient. The counts are therefore $N_f=N_v=N_t=3064$, with no shared waveform history between stages. These are sample counts, not counts of statistically independent observations within a filtered record.

For repetition $r$ and stage $s=0,1,2$, TX and noise seeds are $b+1000r+10s$ and $b+1000r+10s+500000$. The fixed benchmark base is $b=30261001$; the infrastructure pilot uses a disjoint base $20261001$. All methods share the records in a repetition. Ten repetitions are specified for each source/dictionary and noise condition. Normalization and screening use fitting samples only. Parameters are chosen by the largest observed validation cancellation among converged finite solutions, with exact ties resolved by fewer active coefficients and then smaller $M$. Coefficients and selection metadata are frozen before generating the test record; no fitting-plus-validation retraining is performed. Nonconverged configurations and resource failures remain in the log.

The initial common benchmark grid is $M\in\{2,4,6\}$. The historical $M=8$ results are retained only as diagnostics; the combined $M=8$ dense solver requires a separate memory-feasibility study before adding that memory to all methods. This restricted grid must not be described as a completed replacement of the original four-memory sweep.

\subsection{Ablations, Baselines and Numerical Settings}
The four ablations are full ridge-LS, conventional fixed-noise SBL without screening, screening followed by ridge-LS, and screening followed by SBL. Complex LASSO and complex orthogonal matching pursuit (OMP) provide additional sparse baselines. LASSO penalizes $\sum_i|w_i|$ in fitting-normalized coordinates, keeping the real and imaginary parts of one complex coefficient together; OMP also selects whole complex monomials. Both use the same candidate dictionary and receiver filter.

The ridge grid is $\rho=10^k$ for $k=-8,\ldots,-3$. Other grids are $\beta\in\{10^2,10^4,10^6\}$, and $\eta\in\{0.01,0.65,0.8\}$. LASSO uses penalties $\{10^{-4},10^{-3},10^{-2},10^{-1}\}\lambda_{\max}$, where $\lambda_{\max}=\|\bar X_f^H\bar y_f\|_\infty$; OMP budgets are 5, 10, 20 and 40 terms. Controlled fixed-parameter ablations additionally retain $\eta=0.01$ to isolate the original screening stage from hyperparameter tuning.

For SBL, $\alpha_0=1$, $\alpha_{\max}=10^6$, denominator floor $10^{-12}$ and relative-mean tolerance $10^{-6}$ are fixed. The cap is 2000 for the revision, versus 500 for the historical main sweep (200 in its preliminary runs). Pruning resets the previous-mean comparison; at least one coefficient remains. The benchmark LASSO uses complex coordinate descent, global relative KKT tolerance $10^{-5}$ and a 100000-sweep cap. The earlier infrastructure pilot used a 10000-step proximal solver with tolerance $10^{-6}$; its weak-penalty convergence was inadequate and its LASSO scores are not used as final baseline evidence. Both source versions and failures are retained. Solver trajectories, numerical failures and seed/configuration manifests are saved.

Timing distinguishes dictionary construction, screening, fitting and the complete validation search. Inference includes construction of the active monomials and receiver filtering, with repeated timings per scored block. Each method/repetition/scenario runs in a separate process; sampled peak RSS includes native numerical arrays and is labeled as the complete workflow peak, not isolated solver allocation. Comparisons require identical hardware and BLAS thread counts. Means, standard deviations and paired confidence intervals are computed over independent repetitions; support stability uses monomial identifiers and Jaccard overlap, conditioned on a common memory where appropriate. For a truncated tanh model this is stability, not recovery of a known exact finite support.
'''
results=r'''
\section{Verified Diagnostics and Revision Results}\label{sec:results}
\subsection{What the Original Screen Retains}
Table~\ref{tab:screen-counts} reports the directly recomputed counts for all five historical scenarios. Screening is independent of $\beta$ and retains every column for every original configuration. Scores over all tested columns range from approximately 0.587 to 0.991. The previously quoted lower bound 0.61 omitted the extreme at combined $M=8$. These observations concern the supplied realizations; frequency-cell restriction alone does not prove that every possible target-correlation screen is ineffective.
\begin{table}[!t]
\caption{Historical candidate counts before/after screening at $\eta=0.01$. Counts apply to all original $\beta$ values and both clean/noisy versions where present.}\label{tab:screen-counts}
\centering\small
\begin{tabular}{rrr}\toprule
$M$ & Cubic: full/screened & Cubic--quintic: full/screened\\\midrule
2 & 6/6 & 40/40\\4 & 40/40 & 760/760\\
6 & 126/126 & 4788/4788\\8 & 288/288 & 18336/18336\\\bottomrule
\end{tabular}
\end{table}
Consequently, screen-plus-SBL and no-screen SBL have the same input dictionary at this threshold; the same holds for the two ridge branches. Final historical supports of seven cubic terms or 17 combined terms must be attributed to relevance pruning, not to a reduction by the first stage. Tuned higher thresholds require separate validation and may discard useful correlated columns.

\subsection{Iteration-Cap and Initialization Diagnostic}
For the original cubic source at $M=6$, $\beta=10^6$, raising the cap from 500 to 2000 changes cancellation from 51.264645 to 51.265265 dB. With $\alpha_0=1$, convergence occurs at iteration 1623; the last relative change in the posterior mean is $9.9881\times10^{-7}$, compared with $2.4041\times10^{-5}$ at iteration 500. Table~\ref{tab:convergence} reports initialization sensitivity. A cap of 5000 yields the same stopping points as 2000. All converged runs select the same seven monomials, and their cancellation differs by less than $4\times10^{-8}$ dB. Per-iteration residual power, negative log marginal likelihood and relative-mean changes are provided as machine-readable trajectories. This supports numerical stability on the historical record, not unbiased model selection or global optimality.
\begin{table}[!t]
\caption{Historical cubic $M=6$, $\beta=10^6$, cap 2000.}\label{tab:convergence}
\centering\small
\begin{tabular}{rrrr}\toprule
$\alpha_0$ & Iterations & Cancellation (dB) & Active\\\midrule
0.1 & 1603 & 51.265265 & 7\\
1 & 1623 & 51.265265 & 7\\
10 & 1612 & 51.265265 & 7\\\bottomrule
\end{tabular}
\end{table}

\subsection{Why the Historical LS Margin Is Not a Final Comparison}
At the same historical cubic $M=6$, the saved ridge sensitivity study gives 50.682 dB for $\rho=10^{-4}$ but 57.891 dB for $\rho=10^{-8}$, against about 51.265 dB for SBL. The advertised margin over fixed ridge is therefore not evidence of superiority over a competitive tuned baseline. These alternative ridge values were also evaluated on the original test segment; they motivate validation tuning rather than supplying a replacement unbiased estimate. The historical tanh results, approximately 24 dB with a cubic dictionary and 30 dB with a combined dictionary, similarly remain exploratory controls.

\subsection{Independent Benchmark}
\IfFileExists{revision/generated/validation_results.tex}{\input{revision/generated/validation_results.tex}}{
The independent benchmark is not yet complete. No new mean cancellation, confidence interval, support-recovery claim or computational speedup is reported in this draft. The historical numbers above are not substituted for its missing results.}

\section{Discussion and Limitations}\label{sec:discussion}
The data distinguish order matching from order mismatch: a cubic source has a matching interaction family but still suffers finite-memory truncation, whereas tanh introduces higher odd orders. Sparse supports can approximate a dense fit without identifying unique physical coefficients because narrowband Volterra columns are strongly correlated. Stability of cancellation and stability of selected monomials must therefore be evaluated separately.

Compactness alone does not establish computational efficiency. Even a final seven-term model can require building a full dictionary and repeatedly solving a large system. At combined $M=8$, one complex128 $18336\times18336$ Gram matrix occupies about 5.01 GiB, before other arrays. The present implementation and resource-limited benchmark must not inherit speedup claims from an untested incremental variant. No residual-ranked incremental method is evaluated in this revision.

The posterior equations describe regularized coefficient inference. Since the likelihood precision is externally tuned, the tanh source is misspecified by a finite-order dictionary and no predictive-coverage analysis is reported, calibrated predictive uncertainty is not claimed. Real measured PIM, changing paths, multiple nonlinear sources and MIMO remain outside this synthetic study \cite{Lampu2023Algorithms,Lampu2023Cascade}. Testing measured data is future work.

\section{Conclusion}\label{sec:concl}
Fixed-noise SBL provides sparse coefficient estimates for symmetry-reduced complex PIM dictionaries. The historical audit establishes two limited results: the original one-shot screen at $\eta=0.01$ is inactive, and the cubic 51.3 dB point is stable under higher iteration caps and three initializations. Neither result establishes a cancellation or efficiency advantage over validation-tuned alternatives. Independent fitting, validation and test records, screening ablations, sparse baselines and repeated noise conditions define the revised evaluation. Its completed results are required before broader performance claims can be made.

\bibliographystyle{IEEEtran}
\IEEEtriggeratref{9}
\bibliography{references}
\end{document}
'''
(root/'main.tex').write_text(prefix+intro+model+setup+results,encoding='utf-8')
print('Revised manuscript written; original and supplied experiment description preserved.')
