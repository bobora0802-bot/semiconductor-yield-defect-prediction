# Revision decisions — September 17, 2026

These notes were written during the current revision. They are not a reconstruction
of an earlier independent learning diary. Historical claims remain visible in v2.

| Observation | Decision | Evidence / trade-off |
| --- | --- | --- |
| 243 of 314 later evaluation records, including 16 failures, were in the old random selection pool | Replace imported winner with past-only inner selection at every outer window | New selected model may differ; previous inspection still prevents a fresh-test claim |
| Old monitoring selection used downstream blend-validation labels | Select monitoring variables inside early reference only; retire blend | Removes circular validation; reference is still label-assisted |
| Same-length contribution tables could be mismatched | Require unique matching IDs and align by ID | Shuffled input must preserve output; duplicate/missing IDs must fail |
| EWMA severity divided all earlier values by the last row's limit | Normalize each row by its own time-dependent limit | Prefix invariance test verifies future rows cannot change earlier EWMA scores |
| Old plot used fixed +/-3 despite moving-range limits | Plot actual fitted I limits | I-MR additionally uses moving-range alerts; plotted lines alone do not explain every flag |
| Old priority percentile used all historical/future rows | New descriptive percentiles use earlier validation reference | Batch top-k is explicitly retrospective and kept separate |
| Q residual contributions were ambiguously named | Name as Q/SPE residual contributions only | No implication that they explain classifier, T2 or physical cause |
| Composite consensus mixed unequal evidence and assumed zero for unmeasured signals | Retire from primary outputs | Separate importance, drift and residual evidence remain inspectable |
| A single later period gives fragile estimates | Publish four time windows and conditional day bootstrap | Still few failures and previously inspected data; no external guarantee |
| Early/late reference and chart state can change alarms | Publish reference sensitivity with explicit state-reset policy | No variant promoted after inspecting latest-period outcomes |
| Model performance is not the same as ownership/mastery | Record actual AI assistance and defer teaching to a separate phase | No claim that the owner independently wrote or mastered every component |

## Findings kept even when inconvenient

- Earlier-only selection does not pick the model that later happens to perform best in every window.
- Latest-period ranking improved relative to the fixed RF comparison, but probability quality is poor.
- Latest 20% review captures the same six failures as the previous RF result, despite changed ranking.
- CUSUM remains saturated; reference changes do not consistently improve failure capture.
- No score target was used as an acceptance criterion.

## Historical reproduction

Use commit `b0362e640c1a08b39752f0f36077db3cdde5079a` in a separate checkout
for the original scripts and paths. Archived copies are disabled as entry points
to prevent them from silently regenerating superseded reports.
