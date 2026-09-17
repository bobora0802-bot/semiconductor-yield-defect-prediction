# Model Card: SECOM Failure-Risk Screening and Excursion Triage

## Intended use

This model is a portfolio-scale analysis of rare semiconductor manufacturing
failures in the public UCI SECOM dataset. It demonstrates leakage-aware model
comparison, temporal robustness checks, review-capacity analysis, uncertainty
reporting, calibration diagnostics, and cautious feature prioritisation.

## Not intended use

It is not validated for production deployment, automatic pass/fail decisions,
fab process control, safety-critical decisions, or physical root-cause
diagnosis. No result has been reviewed by process engineers from the source fab.

## Data

- 1,567 runs and 590 anonymous process measurements
- 104 failures (6.64%)
- 41,951 missing cells
- timestamps from 19 July to 17 October 2008
- source: UCI SECOM, CC BY 4.0, DOI 10.24432/C54305

## Selected workflow

The selected model is a class-weighted random forest preceded by a training-only
missingness filter, median imputation, constant-feature removal, and univariate
feature selection. Candidate models within 0.005 mean CV PR-AUC are treated as
practical ties; lower variability and then higher ROC-AUC decide the tie.

Adding missingness indicators changed mean CV PR-AUC from 0.1776 to 0.1784 but
increased its standard deviation and reduced holdout PR-AUC. It was therefore
not promoted.

## Evaluation

| Validation scheme | ROC-AUC | PR-AUC | Failure prevalence |
| --- | ---: | ---: | ---: |
| Random stratified holdout | 0.739 | 0.244 | 6.69% |
| Chronological holdout | 0.616 | 0.087 | 5.41% |

Random-holdout 95% stratified-bootstrap intervals were 0.620-0.846 for ROC-AUC
and 0.130-0.433 for PR-AUC. Chronological intervals were 0.480-0.760 and
0.061-0.170. These are uncertainty estimates for fixed holdouts, not guarantees
of performance on another fabrication process.

## Decision policy

An illustrative policy assigns a missed failure 10 times the cost of a false
alarm. On the random holdout its training-derived threshold detected 11 of 21
failures while flagging 57 passing runs. The analogous temporally selected
threshold detected 0 of 17 failures in the latest holdout, so a universal
threshold is not supported.

Capacity-based ranking is more interpretable for this dataset. In the
chronological holdout, reviewing the top 10% highest-risk runs captured 3 of 17
failures (17.6%, 1.73x lift); reviewing the top 20% captured 6 of 17 (35.3%,
1.76x lift).

## Calibration

Isotonic calibration had the lowest random-holdout Brier score (0.0567) and
five-bin calibration error, while sigmoid calibration had the lowest log loss
(0.2186). Because the methods disagree on a small holdout, neither is promoted
as a probability scale for operational decisions.

## Feature interpretation

`sensor_103` and `sensor_059` appeared in the random forest's top 20 in all 10
repeated-CV folds. Held-out permutation importance ranked `sensor_059` first.
All variables are anonymous; rankings identify candidates for investigation,
not physical causes.

## Primary limitations

- only 104 failures are available;
- measurements and process context are anonymous;
- lot, tool, recipe, chamber, and maintenance identifiers are unavailable;
- failure prevalence and several feature distributions change over time;
- random splitting materially overstates chronological performance;
- cost ratios are illustrative and not supplied by a fabrication facility;
- no external fab dataset is available for validation.

## Required work before operational consideration

Obtain process identities and grouped lot/tool data; define engineering costs;
use rolling retraining and calibration windows; validate drift alarms; test on
an external fab period; and require process-engineering review of every proposed
action.

## Phase I/II monitoring extension

The earliest 80% of timestamp-ordered runs remains the development partition;
the latest 20% (314 runs, 17 failures) remains the final holdout. Within
development data, known pass runs from the earliest 60% window form a
label-assisted historical reference, not a validated in-control production
period.

Fifteen monitoring variables are selected from training data only. Median
imputation, standardisation, PCA, and empirical 99th-percentile control limits
are fitted only on 684 Phase I reference rows. PCA retains 9 components, with a
T² limit of 79.473 and Q/SPE limit of 11.279. The latest holdout is never used
to choose variables, PCA dimension, limits, thresholds, or combination weights.

## Monitoring evidence

I-MR, EWMA, and CUSUM each flagged 100% of the final holdout. Hotelling's T²
flagged 83.1% of runs, captured 13 of 17 failures, and also flagged 248 passes.
Q/SPE flagged one passing run and captured no failures. This indicates severe
reference-to-later distribution shift and poor actionability, not successful
failure prediction.

At 10% and 20% review capacity, the existing ML ranking captured 3/17 and 6/17
failures; SPC severity captured 0/17 and 1/17. A validation-selected blend chose
zero ML weight and reduced to SPC, so it did not improve the final result.

## Candidate-variable and run triage

The consensus combines training-only temporal stability, permutation
importance, standardised association, missingness and KS drift, control-chart
violations, and Q/SPE contribution. `sensor_021`, `sensor_247`, and
`sensor_059` are the top three candidate process variables. Run-level triage
combines model risk, T², Q/SPE, EWMA/CUSUM state, and anonymous-variable
contributions. Actual labels are included only for offline evaluation.

These fields establish association, triage signal, and investigation priority
only. No physical mechanism, causal conclusion, wafer/lot/equipment effect,
true warning lead time, or operational readiness is claimed. Cp/Cpk is not
calculated because specification limits are unavailable.
