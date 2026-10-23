# Captured-data example

The experiment used one roughly 20-second BTC-USD recording on September 25, 2026.
It contained 423 messages and produced 193 sampled observations with 99 features.
Chronological splitting yielded 136 TRAIN, 29 validation, and 28 TEST feature rows.
Endpoint purging reduces usable target counts separately for each horizon.

Raw and processed data are not included in this compact repository. The tables and
three figures retain selected results from the original experiment; a new capture
will not reproduce the same numbers.

## Horizon comparison

Cells show validation accuracy and eligible target count.

| Model | 100 ms | 500 ms | 1 second | 5 seconds |
| --- | ---: | ---: | ---: | --- |
| Majority | 0.929 (28) | 0.583 (24) | 0.158 (19) | unavailable |
| Imbalance sign | 0.071 (28) | 0.417 (24) | 0.842 (19) | unavailable |
| Logistic regression | 0.893 (28) | 0.583 (24) | 0.789 (19) | unavailable |
| XGBoost | 0.893 (28) | 0.500 (24) | 0.474 (19) | unavailable |
| MLP | 0.929 (28) | 0.583 (24) | 0.000 (19) | unavailable |
| GRU | 0.667 (9) | 0.600 (5) | unavailable | unavailable |
| Matched MLP | 0.889 (9) | 0.400 (5) | unavailable | unavailable |

Snapshot models share target rows within a horizon. The GRU and matched MLP share
their own sequence targets. Their accuracies should not be compared directly with
the larger snapshot cohort.

![Performance versus horizon](figures/signal_decay.png)

Changing class balance explains why raw accuracy alone is misleading. At 100 ms,
26 of 28 validation targets are STABLE; at one second, 16 of 19 are UP. The
imbalance rule's one-second accuracy is high while its balanced accuracy is 0.5.
The MLP's 0/19 result is retained rather than tuning settings to hide a bad interval.

![Target counts by class](figures/class_distribution.png)

There are no 5-second validation targets after purging. The short sequence validation
partition also cannot support a one-second target after constructing 20-row histories.
Missing scores are not zeros.

## Feature ablations at 500 ms

All rows below use the same 131 TRAIN and 24 validation targets.

| Inputs | Features | XGBoost accuracy | MLP accuracy |
| --- | ---: | ---: | ---: |
| Book | 5 | 0.4583 | 0.5833 |
| + imbalance | 9 | 0.2500 | 0.5833 |
| + flow | 29 | 0.5833 | 0.5833 |
| + trades | 61 | 0.5000 | 0.5833 |
| All groups | 99 | 0.5000 | 0.5833 |

There is no consistent accuracy improvement as groups are added. MLP log loss changes
across the ablations even though its hard predictions remain STABLE. The results do
not establish that any group is unhelpful in a larger dataset.

## What the experiment supports

The experiment verifies that the pipeline can train, save, reload, and compare
models on checked chronological targets. It completed 27 fits, with nine unavailable
model/horizon combinations, and produced regime, confidence, calibration, and timing
diagnostics.

It does not establish a profitable signal or a reliable ordering of models:

- Validation is reused for checkpoint selection and these comparisons.
- Most horizons have fewer than 30 validation targets and no observed DOWN class.
- The GRU comparison has just five validation targets at 500 ms.
- One recording session cannot support the configured session-bootstrap intervals.
- Horizons differ in observation coverage and class balance, confounding a pure
  signal-decay interpretation.
- Signed future mid moves do not include execution costs or fill assumptions.

A meaningful next experiment would collect longer, separated periods, predeclare
the comparisons, and keep a final period out of model and feature selection.

## Provenance

The original split manifest SHA256 was
38429cd7631aadd055af2766e112876c2420e3145a8d1762a7ff5e9e32df2773.
The software used seed 42 and the architecture settings retained in configs.

The numerical results were checked against saved predictions and independently
audited for row identity, confusion counts, log loss, regime membership, and
confidence-conditioned mid moves. This example is historical evidence of the
implementation, not a replacement for collecting an adequate research sample.
