# Methodology

This document explains every statistical step in the pipeline, the formulas used, and why each choice was made. Code references point to `src/secom_spc/`.

## 0. Framing

SECOM is a record of 1567 production runs from a semiconductor fab between July and October 2008. Each run has 590 anonymised sensor / process-measurement readings and a pass/fail result from in-house line testing. The donors' stated goal was feature selection: find the few signals that explain yield excursions.

Two questions drive this project:

1. **Monitoring.** If an engineer had frozen a monitoring plan at some point in time, which runs and which sensors would it have flagged afterwards, and how would the line yield KPI have looked? This is classic statistical process control (SPC).
2. **Diagnosis.** Which sensors separate failing runs from passing runs, and does that separation survive into a later time period?

## 1. Time is the primary axis

Fab data is a time series. Every split in this project respects time order (`data.phase_split`):

| Split | Runs | Purpose |
|---|---|---|
| Phase I | first 50 % (784 runs, to 2008-09-11) | baseline: control limits, imputation medians, sensor screening, PCA model |
| Phase I reference | passing runs within Phase I (717) | "known good" data the limits are fitted on |
| Phase II | last 50 % (783 runs) | monitoring with frozen limits |
| Model train | first 70 % (1097 runs, 78 fails) | feature ranking, model fitting, threshold choice |
| Model test | last 30 % (470 runs, 26 fails) | untouched hold-out |

Random shuffled splits are also reported, only to show how much they flatter a model (section 8).

## 2. Sensor screening (`preprocess.SensorScreen`)

Fitted on Phase I only, then applied unchanged to all runs:

1. Drop sensors missing in more than 50 % of Phase I runs.
2. Drop constant or near-constant sensors (fewer than 3 distinct values, or relative SD < 1e-8). SECOM has well over a hundred of these.
3. Collapse near-duplicates: walking the sensor list in order, a sensor whose |Pearson r| with an already-kept sensor exceeds 0.98 is dropped and recorded in `reports/tables/duplicate_sensors.csv`.
4. Median imputation using Phase I medians. SPC charts use the **un-imputed** values (missing readings are simply not plotted and never trigger a rule); imputed values are used only where an algorithm needs a complete matrix (PCA, mutual information, tree models).

## 3. Univariate control charts (`spc.py`, `rules.py`)

### Individuals and moving range (I-MR)

Wafer runs are individual measurements, so the I-MR chart is the natural default.

* Centre line: mean of the Phase I reference values.
* Within-run sigma: $\hat\sigma = \overline{MR} / d_2$, with $d_2 = 1.128$ for spans of two. The moving-range estimate is used instead of the overall SD because it reflects short-term noise and is less inflated by slow drifts.
* Limits: $\bar x \pm 3\hat\sigma$. MR chart UCL $= D_4 \overline{MR}$, $D_4 = 3.267$.
* **Phase I cleanup:** points beyond 3 sigma are removed and limits re-estimated, up to three times, the standard procedure for building a trial baseline.

### X-bar / S

Consecutive runs are grouped into rational subgroups of five. Constants are computed exactly: $c_4(n) = \sqrt{2/(n-1)}\,\Gamma(n/2)/\Gamma((n-1)/2)$, $A_3 = 3/(c_4\sqrt n)$, $B_{3,4} = 1 \mp 3\sqrt{1-c_4^2}/c_4$. Unit tests check these against Montgomery's published tables.

### Nelson rules

All eight Nelson (1984) rules are implemented with sliding windows on the standardised series $z = (x - CL)/\hat\sigma$. A rule flags the point that completes the pattern. Any window containing a missing reading cannot fire a run rule.

| Rule | Pattern |
|---|---|
| 1 | 1 point beyond 3 sigma |
| 2 | 9 in a row on one side of the centre |
| 3 | 6 in a row increasing or decreasing |
| 4 | 14 in a row alternating |
| 5 | 2 of 3 beyond 2 sigma, same side |
| 6 | 4 of 5 beyond 1 sigma, same side |
| 7 | 15 in a row within 1 sigma |
| 8 | 8 in a row beyond 1 sigma, either side |

### EWMA and CUSUM

For the top-ranked sensors the export also carries:

* EWMA with $\lambda = 0.2$, $L = 2.86$ (in-control ARL about 370) and exact time-varying limits $\mu \pm L\sigma\sqrt{\tfrac{\lambda}{2-\lambda}(1-(1-\lambda)^{2t})}$.
* Tabular CUSUM with $k = 0.5\sigma$, $h = 5\sigma$.

Both detect sustained small shifts (about 1 sigma) faster than a Shewhart chart; a unit test verifies this on synthetic data.

## 4. Yield KPI (`pipeline.daily_kpis`)

* Daily yield = passing runs / runs that day; a 7-day rolling yield weights by run count.
* **p chart** of the daily fail fraction with variable limits $\bar p \pm 3\sqrt{\bar p(1-\bar p)/n_i}$, $\bar p$ from Phase I days.
* **Laney p' chart** as a robustness check. Day-to-day fail rates in fabs are usually over-dispersed relative to the binomial model; Laney rescales the limits by $\sigma_z$, the moving-range sigma of the Phase I z scores. $\sigma_z > 1$ indicates over-dispersion.

## 5. Process capability (`capability.py`)

SECOM has **no specification limits**, so true Cpk cannot be computed. The pipeline uses *proxy* limits:

$$ LSL, USL = \text{median}_{I} \mp 4\,\hat\sigma_{\text{robust}, I} $$

where $\hat\sigma_{\text{robust}} = 1.4826 \times \text{MAD}$. A normally distributed sensor that keeps behaving like its baseline would score Cpk about 1.33. The quantities reported:

* Cp, Cpk with within sigma (MR-bar / d2), Pp, Ppk with overall SD
* Percentile (Clements) Cpk, which uses the 0.135th, 50th and 99.865th percentiles and does not assume normality
* D'Agostino-Pearson normality test per sensor
* $\Delta$Ppk = Ppk(II) minus Ppk(I), mean shift in Phase I SDs, SD ratio

Read the result as **capability drift** relative to each sensor's own baseline, not as product capability. Because most sensors are strongly non-normal and heavy tailed, the median Phase I Ppk sits near 1.0 rather than 1.33: the overall SD is inflated by tails that the MAD ignores. That gap is itself a finding about the data.

## 6. Multivariate SPC (`multivariate.py`)

Charting 324 sensors one by one at 3 sigma would give, even for perfectly normal independent data, $1-(1-0.0027)^{324} \approx 58\%$ of in-control runs at least one false alarm. The standard remedy is PCA-based monitoring:

1. Standardise with Phase I reference means and SDs; winsorise z at $\pm 10$.
2. Fit PCA on the reference runs; keep the smallest number of components explaining 80 % of variance.
3. **Hotelling T²** $= \sum_a t_a^2/\lambda_a$ (distance inside the model plane). Phase II limit: $\frac{p(m+1)(m-1)}{m(m-p)}F_{\alpha}(p, m-p)$.
4. **SPE / Q** $= \lVert z - \hat z\rVert^2$ (distance off the plane). Limit from Box's scaled chi-square approximation $g\chi^2_h$ with $g = \text{var}/(2\,\text{mean})$, $h = 2\,\text{mean}^2/\text{var}$ of reference SPE.
5. Because the parametric limits assume multivariate normality, an **empirical** limit (99th percentile of the reference statistic) is reported alongside and used for the exported alarm flag.
6. **Contribution plots** decompose SPE into per-sensor squared residuals, pointing an engineer at the sensors behind an alarm.

## 7. Drift diagnostics (`drift.py`)

* **Kolmogorov-Smirnov** two-sample test per sensor, Phase I vs Phase II.
* **Adversarial validation:** a random forest is trained to predict whether a run came from Phase I or Phase II. Cross-validated AUC near 0.5 means the periods look alike; near 1.0 means the process moved enough that a model fitted on one period should not be trusted on the other.
* **Frozen-then-live detection:** sensors that report one repeated value for at least two whole weeks and then vary. That usually indicates a stale or held tag rather than a perfectly stable process, and it breaks any control limit fitted on the frozen stretch.
* **MSPC sensitivity:** refit the PCA monitor after removing the drifting sensors to see how much of the Phase II alarm load they explain.

## 8. Ranking failure drivers (`features.py`)

Fitted on the training period only. Five views, each converted to a rank:

| View | What it captures |
|---|---|
| Mann-Whitney U with Benjamini-Hochberg FDR | any distribution shift between pass and fail, robust to skew |
| Univariate AUC, abs(AUC - 0.5) | effect size, directly interpretable |
| Mutual information (k-NN estimator, 5 seeds) | non-monotonic association |
| L1-logistic stability selection (200 balanced half-samples) | survives in a sparse multivariate model |
| Extra-Trees importance (balanced, 5 seeds) | interactions, non-linearity |

The consensus rank is the ordinal position of the mean of the five ranks.

Two out-of-sample checks guard against false discoveries:

* **Test-period AUC**: does the sensor still separate pass and fail in the final 30 % of runs, in the same direction?
* **Within-week AUC**: the AUC computed only between fails and passes from the same calendar week, averaged with weights equal to the number of fails. If both the fail rate and a sensor trend over time, the pooled AUC will look strong even when the sensor has nothing to do with failure. Within-week AUC removes that shared trend. A unit test plants exactly this confound and checks it is exposed.

The **alarm lift** column links SPC to failure: the fail rate among runs where this sensor's I-chart fired, divided by the overall fail rate, with a one-sided Fisher exact test.

## 9. Predicting failure (`modeling.py`)

Pipelines (imputation, scaling, clipping, the top-k filter) are refit inside every fold to avoid leakage.

| Model | Notes |
|---|---|
| L1 logistic, all sensors | C = 0.05, balanced class weights |
| Logistic, top 20 by AUC | the top-20 filter is learned inside each fold |
| Random forest | 500 trees, balanced subsample weights |
| Histogram gradient boosting | handles missing values natively, balanced weights |
| SPC baselines (no learning) | count of sensors beyond 3 sigma; Hotelling T²; SPE |

Metrics are imbalance-aware: ROC AUC with a 1000-sample bootstrap CI, PR AUC (compare with the base fail rate), balanced error rate (BER, the metric in the donors' baseline), recall on fails, specificity. The decision threshold is chosen to minimise BER on out-of-fold training predictions and then frozen for the test period.

Three evaluation designs are reported:

1. **Temporal hold-out** (train first 70 %, test last 30 %): the headline number.
2. **Walk-forward**: starting at the Phase I boundary, score each block of 100 runs with a model retrained on everything before it (and, separately, on a sliding 500-run window), versus a model trained once. This is how a yield model would actually be operated.
3. **Shuffled stratified 10-fold CV** on all data: comparable to most published SECOM results, and optimistic because it trains on the future.

## 10. Limitations

* Sensor meaning, units and spec limits are unknown, so physical interpretation of any sensor is impossible and Cpk is a proxy.
* 104 fails in total, only 26 in the hold-out and 37 in the walk-forward period. AUC confidence intervals are wide; differences of a few points between models are not meaningful.
* Fail labels come from a downstream line test. Some failures will be caused by steps not represented in these 590 signals.
* Control limits, run rules and multivariate statistics all assume independent runs. Autocorrelation in fab data makes some alarms more frequent than nominal; the empirical MSPC limits partly compensate.

## References

* Montgomery, D. C. *Introduction to Statistical Quality Control*, 8th ed. Wiley, 2019.
* Nelson, L. S. "The Shewhart Control Chart: Tests for Special Causes." *Journal of Quality Technology* 16(4), 1984.
* Laney, D. B. "Improved Control Charts for Attributes." *Quality Engineering* 14(4), 2002.
* Clements, J. A. "Process Capability Calculations for Non-Normal Distributions." *Quality Progress* 22(9), 1989.
* Jackson, J. E. & Mudholkar, G. S. "Control Procedures for Residuals Associated with Principal Component Analysis." *Technometrics* 21(3), 1979.
* MacGregor, J. F. & Kourti, T. "Statistical Process Control of Multivariate Processes." *Control Engineering Practice* 3(3), 1995.
* Meinshausen, N. & Buehlmann, P. "Stability Selection." *JRSS B* 72(4), 2010.
* Benjamini, Y. & Hochberg, Y. "Controlling the False Discovery Rate." *JRSS B* 57(1), 1995.
* McCann, M. & Johnston, A. *SECOM* [Dataset]. UCI Machine Learning Repository, 2008. https://doi.org/10.24432/C54305
