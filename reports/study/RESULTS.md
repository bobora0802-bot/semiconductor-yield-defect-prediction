# Results and interpretation

Generated CSV/JSON files are the numerical source of truth. This narrative describes
the checked-in configuration; all data periods were previously inspected.

## Time robustness

| Window | Evaluation runs / failures | Selected model | Average precision | ROC-AUC |
| --- | --- | --- | --- | --- |
| 1 | 157 / 9 | Logistic regression | 0.087 | 0.528 |
| 2 | 156 / 2 | Logistic regression | 0.013 | 0.250 |
| 3 | 157 / 9 | Random forest | 0.092 | 0.561 |
| 4 | 314 / 17 | Logistic regression | 0.185 | 0.647 |

Earlier-only selection does not guarantee the best later-period model. For example,
the fixed random forest scored AP 0.371 in window 1 but was not selected from its
earlier inner results. This is retained as evidence of selection instability.
The second window has only two failures; interpreting its metric as a reliable
estimate of future manufacturing performance would be inappropriate.

Latest-period AP is 0.185 against prevalence 0.054, but Brier score is 0.182 versus
0.051 for constant historical prevalence. These class-weighted scores are useful
only as exploratory ranks, not calibrated probabilities. Conditional whole-day
bootstrap intervals are AP 0.070–0.418 and ROC-AUC 0.452–0.815; 499/500 samples
contained both classes. Sixteen observed days cannot establish external robustness.

## Review trade-off

| Target capacity | Reviewed | Captured / 17 | Passing runs reviewed | Random expected captured |
| --- | --- | --- | --- | --- |
| 5% | 16 | 3 | 13 | 0.87 |
| 10% | 32 | 5 | 27 | 1.73 |
| 20% | 63 | 6 | 57 | 3.41 |
| 30% | 95 | 8 | 87 | 5.14 |

These are completed-batch rankings. Random-null tail probabilities are exploratory,
not multiple-comparison-adjusted significance claims. At the illustrative 10:1
miss/false-alert cost, the earlier-selected threshold yields cost 163 (16 misses,
3 false alerts), compared with 170 for review-none. Batch top-20% costs 167. The
assumed costs are not observed fab economics; no monetary or yield benefit is claimed.

## Why the alarms fail

Reference-only feature selection changes the monitoring set from v2. The latest
I-MR union flags 35.7% of records and captures 8/17 failures, with 104 false alerts.
EWMA flags 67.5%, captures 14/17 and generates 198 false alerts. CUSUM still flags
every record, capturing all failures by also flagging all 297 passing records.

Post-hoc v2 examples sensor_542 and sensor_543 are no longer in the reference-selected
15-variable set. On their original early pass reference, later passing records
shift by about +7.10 and +8.37 reference standard deviations respectively. This
supports a measurement distribution change affecting passing runs too. It does
not identify physical cause; each historical variable had only 7 or 6 observed
values and very small variance, so standardized shifts also depend on resolution
and reference scale. Marginal lag-one correlations were small in these examples;
autocorrelation alone is not established as their explanation.

In the revised monitoring set, sensor_527 alone triggers CUSUM throughout the latest
period. Removing it post-hoc still leaves an 85.4% union alarm rate. This removal is
a diagnosis, not a selected improved policy. A recent-pass reference reduces EWMA
alarm rate in the reset-state comparison, but also reduces failure capture. Lower
alarm frequency is not automatically better screening. No reference variant is promoted.

## Separate known-change simulation

Across 200 repeats with a +1-sigma step, EWMA and CUSUM detect within the 100-observation
horizon in all simulated trials, with median delay 8 observations. I-MR detects 88%,
with conditional median delay 24. Pre-change probability of at least one alarm is
17.5% for EWMA, 22% for CUSUM and 64.5% for I-MR (which combines I and MR rules).
This demonstrates sensitivity/false-alarm trade-offs under the stated synthetic
assumptions, not production early-warning capability. State is not reset after alarms.

## Case interpretation

Fixed-rule examples at 20% batch review are original run IDs 1254 (captured failure),
1302 (missed failure), and 1253 (false alert). The displayed top variables explain
Q/SPE residuals only. To investigate a physical failure one would additionally need
sensor identities/units, tool/recipe/lot and maintenance records, and independent
measurement or failure-analysis evidence. SECOM supplies none of those.

The project supports an auditable historical analysis and several falsifiable
questions. It does not establish reliable production screening, root-cause
identification, real yield improvement, or the author's independent mastery.
