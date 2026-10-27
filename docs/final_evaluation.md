# Regression and final evaluation

Validation selects checkpoints. Final TEST evaluates the saved models under a
separately frozen plan.

## Future-return regression

The secondary task predicts relative mid-price returns at 100 ms, 500 ms, 1 second,
and 5 seconds. It uses the same eligible rows, purged chronological splits, and
TRAIN-fitted preprocessing as classification:

- Zero: predict no price change.
- Ridge: linear regression with an L2 penalty of 1, fitted on TRAIN only.
- XGBoost: depth-3 histogram trees, up to 200 rounds, learning rate 0.05,
  and validation RMSE early stopping with patience 15.

Targets are scaled to basis points inside fitting. Predictions saved to Parquet
remain relative returns. Runs save their settings, feature schema, and model in
JSON or XGBoost's native format.

Metrics include MAE, RMSE, mean error, R-squared, Pearson correlation, and MSE
improvement over predicting zero. Undefined statistics are null. Directional accuracy uses the
sign of nonzero predictions; an actual zero move counts as a miss. Unlike the
classification target, this measure has no tick-size deadband.

## Freeze before TEST

The plan includes every completed classification run and the supplied regression
runs. It records the dataset manifest, chronological ranges, saved model manifests,
validation research outputs,
confidence thresholds, and hashes of the Python package source.

Freezing verifies the training artifacts without opening TEST files. Evaluation
checks the hashes again, reloads the models, and reports every declared run without
refitting or selecting thresholds.

Evaluation requires `--allow-test`. An access marker makes each plan single-use,
including interrupted attempts. This is a workflow guard, not access control: a
user could create another plan or call individual-model evaluation. Changes based
on TEST results still require a new holdout.

## Outputs

Classification outputs include predictions, horizon comparisons, feature ablations,
TRAIN-defined regimes, calibration, fixed-confidence diagnostics, and paired model
comparisons. Regression outputs include predictions and error metrics. Figures are
saved as PNG and SVG. Timing and complexity metadata come from the validation
experiments; TEST is not re-benchmarked.

See [Results](results.md) for the experiment's findings and limitations.
