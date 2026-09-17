# Retrospective study protocol

Scope: public SECOM data, 1,567 runs, 590 anonymous variables, 104 failures.
All periods were examined in earlier work. No split is represented as newly unseen data.
The protocol configuration was written before the new run, but after inspecting historical
results. This is not prospective preregistration or external validation.

## Time evaluation

Sort timestamps stably, keep original row IDs, and move boundaries left to avoid splitting
equal timestamps. Outer training endpoints are 50%, 60%, 70%, 80%; evaluation ends are 60%,
70%, 80%, 100%. Fractions specify approximate row counts, not equal calendar durations.
Actual dates, rows and failures are published. Unknown lot or equipment dependence remains.
Earlier labels are assumed available at each refit; SECOM does not establish actual label latency.

Inside each past-only outer training period, use three expanding inner windows with training
endpoints 50%, 65%, 80%. Compare logistic regression with 40 selected variables against a
random forest with 80 selected variables. All preprocessing fits inside each training window.
Select the highest unweighted mean inner average precision (exact ties use config order).
Single-class validation metrics are marked undefined and excluded from that mean, never
silently reported as successful discrimination; a single-class training period stops execution.

Fit a 10:1 illustrative cost threshold on earlier inner validation predictions. Scores come
from different expanding fits; threshold transfer can fail. Retrain the selected model on
all available past rows and score the next window. No previous random-split winner is loaded.

Fixed random-forest ablations remove feature selection or add missingness indicators. These
are reported comparisons, not a second route for choosing the policy. Fixed-model random
splits are a separate diagnostic, not part of chronological model selection.

## Capacity and uncertainty

Show every configured capacity, with ceil(n*capacity) records selected. Ties use seeded,
label-blind random ordering. This is retrospective batch review, not arrival-time monitoring.
Frozen percentiles use only earlier validation risks; refitting changes score scales, so they
are descriptive and not calibrated probabilities or a validated online policy.

The hypergeometric random baseline conditions on evaluation size, failure count and review
budget. Its range is NOT a confidence interval on model performance. Tail probabilities are
descriptive, with no adjustment for the many models/windows/capacities examined. Do not claim
significance or pick a capacity after looking at the best tail probability.

For the latest selected policy, 500 whole-day bootstrap samples provide conditional intervals.
The fitted model remains fixed; single-class samples are counted and excluded. Only 16 days
are available. Dependence across days, model selection variability, and repeated historical
inspection are not captured. Report valid resamples and these limitations alongside intervals.

## Monitoring and diagnosis

Choose 15 variables using labels within the earliest 60% of the final development period.
Fit imputation, scaling and PCA/control limits on known pass records in that early period.
This is label-assisted historical reference, not certified in-control operation. Median
imputation and filtering of failed runs can alter the moving-range distribution.

EWMA and CUSUM start after reference fitting and carry state forward without reset after an
alarm. This represents a fixed-reference excursion diagnostic, not a production alarm-response
policy. Persistent CUSUM alarms can reflect accumulated history rather than new incidents.
Sensitivity experiments use early-pass, early-all and recent-pass references with the same
selected variables; they reset state at evaluation start. All variants are reported, none promoted.

Sensors 542 and 543 are explicitly post-hoc examples from the v2 audit. Their physical meaning
is unknown. Q/SPE variable contributions explain PCA residual error only. Permutation importance
is measured on an earlier validation block and is an exploratory, noisy association diagnostic.
The old equal-weight consensus and ML/SPC blend are retired from the primary workflow.

## Simulation and cases

Separate synthetic experiment: 200 repeats, 200 Gaussian reference observations, 100 pre-change
and 100 post-change observations. Scenarios: stable, +1 sigma step, gradual increase to +2 sigma.
Use shared noise across scenarios within each repeat. Report false-alarm frequency and chance
of any pre-change alarm, detection probability, conditional median delay, and restricted mean
delay with non-detection censored at 101. State is not reset at the change or at alarms; early
false alarms may affect later detection. This is not a theoretical ARL or real-fab lead time.

At fixed 20% batch review, select the earliest captured failure, missed failure and false alert
in the latest period. Missing categories stay missing. This avoids manual selection of the most
impressive examples but does not make three examples representative of the entire population.

## Sources

- [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom), DOI 10.24432/C54305, CC BY 4.0.
- [scikit-learn: leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html).
- [NIST: assessing process stability](https://www.itl.nist.gov/div898/handbook/ppc/section4/ppc45.htm).
- [NIST: process monitoring](https://www.itl.nist.gov/div898/handbook/pmc/pmc.htm).

These references motivate methods, not a claim of reproducing a published benchmark.
