# Same-day research results

This experiment replaces the original 20-second software demonstration. Six new
BTC-USD sessions were recorded on October 22, 2026 in Chicago time (October 23 UTC).
Each lasted three minutes, with one-minute gaps: 18 recorded minutes in a 23-minute
window. The feed delivered 23,608 messages and 10,756 sampled observations.

Assignments were declared before collection: four TRAIN, one validation, one TEST.
Feature rows were 7,170 / 1,793 / 1,793. Label availability reduces target counts by
horizon; windows cannot span a gap. The one-tick threshold was fixed in advance.
TRAIN alone supplied preprocessing and regime cutoffs.

The neural budget was 40 epochs, patience 8, batch size 128, seed 42. Validation
selected checkpoints. After 36 classification and 12 regression runs, all models,
thresholds, source hashes, and data hashes were frozen before TEST access. No models
were refitted or selected using the following results. The exact
[classification configuration](../configs/experiments/completion.yaml),
[regression settings](../configs/regression.yaml), and
[numerical record](experiment.json) accompany the tables.

## Classification across horizons

Cells show final TEST balanced accuracy. All three classes are present at every
horizon, so these average DOWN, STABLE, and UP recall equally.

| Model | 100 ms | 500 ms | 1 second | 5 seconds |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| Imbalance sign | 0.4338 | 0.4715 | 0.4414 | 0.3501 |
| Logistic regression | 0.4289 | 0.4186 | 0.3901 | 0.3566 |
| XGBoost | 0.5135 | 0.4519 | 0.4271 | 0.3059 |
| MLP | 0.4258 | 0.3822 | 0.3736 | 0.3177 |
| GRU | 0.4220 | 0.3855 | 0.3571 | 0.2887 |
| Matched MLP | 0.4008 | 0.4049 | 0.3720 | 0.3124 |

Snapshot target counts are 1,792 / 1,788 / 1,783 / 1,743. GRU and matched MLP use
1,773 / 1,769 / 1,764 / 1,724 targets; compare those two to each other. Their
20-observation histories stay inside a split and uninterrupted segment.

| Horizon | DOWN | STABLE | UP |
| --- | ---: | ---: | ---: |
| 100 ms | 106 | 1,591 | 95 |
| 500 ms | 293 | 1,268 | 227 |
| 1 second | 459 | 980 | 344 |
| 5 seconds | 936 | 175 | 632 |

![TRAIN and TEST class distributions](figures/class_distribution.png)

Raw accuracy can conceal weak directional recall. At 100 ms, always predicting
STABLE achieves 88.78% accuracy but 33.33% balanced accuracy. At 500 ms, logistic
regression reaches 73.55% accuracy versus the majority rule's 70.92%; its balanced
accuracy is 41.86%, below the imbalance rule's 47.15%. Metrics answer different
questions, so no model is declared best from one score.

XGBoost's balanced accuracy declines from 51.35% at 100 ms to 30.59% at 5 seconds.
Other models also tend to lose their advantage at longer horizons. This describes
this interval; differing label availability and class balance prevent interpreting
the curves as a clean estimate of a signal's decay rate.

![Final TEST horizon comparison](figures/signal_decay.png)

The full-feature MLP does not beat logistic regression on balanced accuracy at any
horizon. GRU beats its matched MLP control at 100 ms, but trails it at the other
three horizons. At 500 ms the GRU has 12,867 parameters versus 8,579 for the matched
MLP and takes about 7.16 seconds to fit versus 2.39 seconds on this machine. Its
better log loss (0.7815 versus 1.0310) does not translate into better balanced
accuracy. These observations do not establish a consistent benefit from temporal
complexity. Timing scopes are separate from trading latency.

## Feature ablations at 500 ms

All ablations within a model use the same 7,145 TRAIN, 1,788 validation, and 1,788
TEST targets, with the same optimization budget. Checkpoints are selected separately.
Values below are final TEST balanced accuracy.

| Inputs | Features | XGBoost | MLP |
| --- | ---: | ---: | ---: |
| Book state | 5 | 0.4517 | 0.3917 |
| + imbalance | 9 | 0.4794 | 0.3844 |
| + sampled depth flow | 29 | 0.4594 | 0.3954 |
| + trades | 61 | 0.4578 | 0.3862 |
| All groups | 99 | 0.4519 | 0.3822 |

Adding imbalance helps the tree in this interval. Adding flow and trades does not
improve its balanced accuracy further. MLP results fluctuate without a consistent
gain from more inputs. Tree log loss improves from 1.0443 to 1.0160 when trades are
added, even though balanced accuracy falls slightly. The full tree has log loss
0.7964, so its probability estimates improve more than its equal-class recall.

L1/3/5/10 imbalance features are added together. This does not isolate the benefit
of deeper imbalance beyond L1. Likewise, the final step adds microprice, depth, and
dynamics jointly. Those narrower questions need separately declared ablations and
a new holdout, rather than additional tuning on this TEST period.

![Feature ablations on final TEST](figures/ablations.png)

## Return regression

Final TEST RMSE, in basis points:

| Model | 100 ms | 500 ms | 1 second | 5 seconds |
| --- | ---: | ---: | ---: | ---: |
| Zero return | 0.14889 | 0.36664 | 0.55102 | 1.37260 |
| Ridge | 0.14833 | 0.36113 | 0.54629 | 1.53416 |
| XGBoost | 0.14547 | 0.36165 | 0.55008 | 1.40118 |

Ridge and XGBoost reduce 500 ms MSE by about 2.98% and 2.70% relative to predicting
zero. Both are worse at 5 seconds. Predicting zero has lower MAE than either learned
regressor at every horizon, reflecting the large mass of unchanged prices and the
different penalties of MAE and RMSE. The numerical record retains both metrics,
R-squared, correlation, and directional coverage; negative skill is retained.

![Return regression errors](figures/regression.png)

## Regimes and economic interpretation

Using TRAIN median cutoffs, the 500 ms tree's balanced accuracy is 0.3664 in the
lower trailing-volatility bin (975 targets) and 0.4862 in the upper bin (803 targets).
It is 0.4201 versus 0.4809 in lower/higher trade-intensity bins (868 / 907 targets).
This suggests condition dependence within the reserved interval, not a stable
regime effect. Wide-spread support is only 124 targets. Unknown warm-up bins contain
10-13 targets and should not be interpreted from their high scores.

Confidence thresholds were fixed before evaluation. At 500 ms and a 0.7 threshold,
logistic regression produces 42 directional signals with a mean signed future mid
move of 0.1081 bps; XGBoost produces 11 with 0.1854 bps. These small, overlapping
subsets have no independent-session confidence interval. They are price-movement
diagnostics, not executable trades. Fees, spread crossing, queue position, fills,
latency, adverse selection during execution, and inventory risk are not modeled.

## Checks, provenance, and remaining limits

- 420 offline tests passed; Ruff lint/format checks and a wheel build passed.
- All 48 declared runs completed; no horizon/model combinations were unavailable.
- Ten final PNG/SVG figure pairs were rendered and visually checked; four are kept
  here. Full local artifacts retain predictions, checkpoints, metrics, and figures.
- An independent audit matched 63,894 classification and 21,318 regression prediction
  rows to labels and checked confusion counts, accuracy, log loss, confidence moves,
  regression errors, and artifact hashes. Source remained unchanged through TEST.
- The split manifest SHA256 is
  fa1e1ffa257dd6ddfa20a697024f2cdad5bb44644620cf2f10dc86002e37a242.
  Source commit and further hashes are in the numerical record.

This completes an end-to-end same-day prediction experiment, not a generalization
study across market periods. TEST is one three-minute session; overlapping labels
are dependent, and the sessions are not independent market days. The configured
ten-session bootstrap therefore remains unavailable. No significance or profitability
claim is made. Crypto spot, public WebSocket delivery, and local receive-time sampling
also limit transfer to colocated futures or options data.

After using these outcomes to change features or models, reserve new separated
periods for the next final evaluation. Broader data collection and specifically
isolated depth ablations remain empirical follow-up work. The optional fill-aware
market-making simulator has not been implemented.
