# Regression and final evaluation

Validation selects learned checkpoints and supports exploratory comparisons. Final
TEST evaluates a frozen collection of models. The two outputs are kept separate.

## Future-return regression

The secondary task predicts relative mid-price returns at 100 ms, 500 ms, 1 second,
and 5 seconds. It uses the same eligible rows, purged chronological splits, and
TRAIN-fitted preprocessing as classification. Three fixed models are compared:

- Zero: predict no price change. This is the error baseline.
- Ridge: linear regression with an L2 penalty of 1, fitted on TRAIN only.
- XGBoost: depth-3 histogram trees, up to 200 rounds, learning rate 0.05,
  and validation RMSE early stopping with patience 15.

Targets are scaled to basis points inside fitting. Predictions saved to Parquet
remain relative returns. Settings and feature schemas are saved with each run;
models reload from JSON or XGBoost's native format, without Python pickle.

Report MAE, RMSE, mean error, R-squared, Pearson correlation, and MSE improvement
over predicting zero. Undefined statistics are null. Negative R-squared or skill
is a valid result. Directional accuracy uses the sign of nonzero predictions;
an actual zero move counts as a miss. It differs from thresholded three-class
classification accuracy. These scores do not establish trading profitability.

## Freeze before TEST

The final plan records every completed classification run, including predeclared
feature ablations, and explicitly supplied regression runs. It binds the dataset
manifest, chronological ranges, saved model manifests, validation research outputs,
confidence thresholds, and hashes of the Python package source.

Freezing verifies training artifacts without opening TEST files. Execution checks
the hashes again before TEST access. It reloads saved models without fitting,
checkpoint selection, or threshold selection. Every declared model is reported;
TEST is not used to choose a winner.

The explicit --allow-test flag is required. A plan is single-use: an access marker
is created exclusively before the first TEST read, and an interrupted attempt also
consumes the plan. This does not stop deliberate creation of another plan or use of
the individual-model evaluation command. A new holdout is still needed after making
research decisions from TEST outcomes.

## Artifacts and limits

Classification outputs include predictions, horizon comparisons, feature ablations,
TRAIN-defined regimes, calibration, fixed-confidence diagnostics, and paired model
comparisons. Regression outputs include predictions and error metrics. Figures are
saved as PNG and SVG. Timing and complexity metadata in the final report come from
the validation experiments; TEST is not re-benchmarked. Economic diagnostics remain
signed future mid-price movement, without fills or trading costs.

The bootstrap still requires ten sessions per evaluated cohort. Short sessions in
one market interval do not establish independence. Unavailable confidence intervals
are reported rather than replaced with an iid row bootstrap. A same-day experiment
is evidence about its reserved interval, not about performance across trading days.

The core implementation covers collection, reconstruction, features, classification,
regression, validation research, and frozen TEST evaluation. The market-making
simulator remains optional. Stronger conclusions require separated market periods
and a new untouched holdout after changes motivated by the current results.
