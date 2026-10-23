# Limit Order Book Price Prediction

A Python research project on short-horizon BTC-USD mid-price prediction using
Coinbase order-book updates and trades.

The question is whether book imbalance, recent order flow, and trade activity add
useful information beyond simple baselines, and whether that information changes
with the prediction horizon. The project implements collection, reconstruction,
feature engineering, chronological evaluation, and model comparisons.

## What is implemented

- Public WebSocket recorder with raw message storage, reconnects, and file rotation.
- L2 book with exact decimal prices, atomic updates, and snapshot recovery.
- Receive-time sampling at 100 ms and 99 book, flow, trade, and lagged features.
- DOWN / STABLE / UP targets at 100 ms, 500 ms, 1 second, and 5 seconds.
- Chronological splits, endpoint purging, TRAIN-only preprocessing, and guarded TEST access.
- Majority, imbalance-sign, logistic regression, XGBoost, MLP, and GRU models.
- Future-return regression with zero-return, ridge, and XGBoost models.
- Horizon comparisons, feature ablations, calibration, regime analysis, and signed
  future mid-price moves at fixed confidence thresholds.
- A frozen-plan final TEST workflow with model/source hashes and explicit access.

The implementation emphasizes traceable data and fair comparisons. It does not
submit orders or model fills.

## Start here

Python 3.11 or later is required; the test suite was run on Python 3.13.

~~~bash
python -m venv .venv
~~~

Activate the environment with **.venv\Scripts\Activate.ps1** on PowerShell or
**source .venv/bin/activate** on macOS/Linux, then:

~~~bash
python -m pip install -e ".[dev]"
python scripts/book_demo.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
~~~

The book demo and 420 tests run offline. Tests build synthetic data in temporary
directories; they do not need the original capture or trained checkpoints.

For a full collection-to-evaluation example, see [Running experiments](docs/usage.md).
The training dependencies include PyTorch and XGBoost, so the environment is larger
than a recorder-only installation.

## Design

~~~text
WebSocket -> raw JSONL -> L2 reconstruction -> 100 ms observations
                                              |
                                     features + future labels
                                              |
                                   chronological split + purge
                                              |
                              TRAIN fit -> validation comparisons
~~~

The source adapter owns exchange message semantics. The book, processing code, and
models consume normalized events or tables, so another source can be added without
rewriting the research pipeline. Data collection runs separately from model training.

| Directory | Purpose |
| --- | --- |
| src/market_microstructure/book | Book state and atomic updates |
| src/market_microstructure/data | Coinbase adapter, recorder, and raw storage |
| src/market_microstructure/processing | Sampling, features, labels, splits, and preprocessing |
| src/market_microstructure/models | Baselines, neural models, and saved-run handling |
| src/market_microstructure/evaluation | Metrics, comparisons, and plots |
| scripts | Command-line entry points |
| configs | Experiment and processing settings |
| tests | Reconstruction, causality, alignment, model, and corruption checks |
| docs | Methods, usage, and a small captured-data example |

See [Methods](docs/methods.md) for label definitions, leakage controls, model settings,
and evaluation conventions. [Final evaluation](docs/final_evaluation.md) describes
regression and the separate frozen-model TEST workflow.

## Same-day experiment

Six three-minute BTC-USD sessions, separated by one-minute gaps, produced 23,608
messages and 10,756 sampled observations. Sessions 1-4 were assigned to TRAIN,
session 5 to validation, and session 6 to TEST before collection. Recording took
place on October 22, 2026 in Chicago time (October 23 UTC).

The frozen grid completed 36 classification and 12 regression runs. At 500 ms,
the snapshot models use the same 1,788 final TEST targets:

| Model | TEST accuracy | Balanced accuracy | Log loss |
| --- | ---: | ---: | ---: |
| Majority | 0.7092 | 0.3333 | 0.8176 |
| Imbalance sign | 0.2092 | 0.4715 | 0.7886 |
| Logistic regression | 0.7355 | 0.4186 | 0.7904 |
| XGBoost | 0.6924 | 0.4519 | 0.7964 |
| MLP | 0.7232 | 0.3822 | 1.0145 |

Metrics favor different models. Neural models and larger feature sets do not show
a consistent advantage. Regression improves on the zero-return baseline's RMSE
slightly at short horizons, while both learned regressors are worse at 5 seconds.

![Final TEST performance versus horizon](docs/figures/signal_decay.png)

[Results](docs/results.md) discusses class imbalance, sequence controls, ablations,
regimes, and economic diagnostics. A [numerical experiment record](docs/experiment.json)
retains metrics and hashes. This is one reserved three-minute TEST interval, with
overlapping targets, not evidence of performance across trading days or profitability.

## Repository scope

This repository contains the implementation, tests, configurations, and selected
results. Raw captures, processed datasets, checkpoints, virtual environments, and
run archives are omitted. New data and results are ignored by Git.

Further empirical work needs separated market periods and a fresh holdout after
changes motivated by these results. A fill-aware execution simulator remains optional
and is not implemented. Python timings measure local inference work, not
exchange-to-order latency. Crypto spot and a public WebSocket feed also differ from
colocated futures/options market data; receive timestamps include network jitter.
