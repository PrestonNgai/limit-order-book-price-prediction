# Model tuning

The [forward results](tuning_results.md) report how the selected model performed
against the original baselines.

The next experiment keeps the original 500 ms target and one-tick threshold. It
tests whether regularization, class weighting, and removing absolute price levels
improve transfer to a later recording date.

## Development search

Only the first four October 22 training sessions are used for model selection.
The three expanding folds train on sessions 1, 1-2, and 1-3, then validate on
sessions 2, 3, and 4. Label endpoints crossing a fold boundary are purged. Each
fold fits its own imputer and scaler; the original validation and TEST files are
not read by the search.

The fixed grid has 40 configurations:

- Logistic regression: C = 0.01, 0.1, or 1.
- XGBoost: depth 2 or 3, with minimum child weight 10, L2 penalty 10,
  learning rate 0.05, up to 200 rounds, and early-stopping patience 15.
- Both families: uniform or balanced class weights, and four feature sets.

Feature sets contain 3 L1 features; 12 L1/deeper-book features; 20 book, flow,
trade-imbalance, and lagged features; or 93 features excluding midprice, microprice,
and the four absolute VWAP levels. The last set is called `stationary` in the
configuration, but removing price levels does not guarantee statistical stationarity.

Selection maximizes mean fold balanced accuracy. Ties use mean log loss, then the
configuration ID. All candidates and fold scores are saved, including the losers.

Balanced weights are derived from each training fold. Predictions use the weighted
classifier's argmax; probabilities receive an inverse-cost correction using TRAIN
class priors. This correction is not a fitted calibrator. Hard predictions can
therefore differ from the largest reported probability, as with the imbalance rule.

## Run it

~~~bash
python scripts/tune.py --dataset data/processed/splits --config configs/tuning.yaml --output results/tuning
python scripts/train.py --dataset data/processed/splits --config results/tuning/selected_config.json --output results/selected-model
~~~

The search requires at least four TRAIN sessions with all three classes represented
in each fold. It rejects grids above 64 candidates and enforces row/memory limits.
Saved tuned models use JSON coefficients or native XGBoost files. Inference reads
only the selected feature columns and their frozen preprocessing parameters.

## Forward evaluation

The selected configuration is fitted on all four development sessions, with the
original fifth session retained for validation. The old TEST session is excluded.
Three five-minute recordings on October 29 form the new TEST period. Their roles
were fixed before collection, and their outcomes are not used for tuning.

The primary comparison is balanced accuracy at 500 ms against the original XGBoost
configuration. Majority, imbalance, and original logistic regression are also
reported. The same selected hyperparameters are applied at 100 ms, 1 s, and 5 s as
secondary analyses; those horizons do not influence selection.

This is a test on another date, not a multi-day performance study. All three TEST
sessions are from one short market interval. Improvements must be reported with
that scope, alongside the stronger baselines and any losses in other metrics.
