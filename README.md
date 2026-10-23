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
- Horizon comparisons, feature ablations, calibration, regime analysis, and signed
  future mid-price moves at fixed confidence thresholds.

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

The book demo and tests are offline. Tests build synthetic data in temporary
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
and evaluation conventions.

## Example results

The saved experiment used one roughly 20-second BTC-USD capture from September 25,
2026. At 500 ms it had 131 TRAIN targets and only 24 validation targets:

| Model | Validation accuracy | Log loss |
| --- | ---: | ---: |
| Majority | 0.5833 | 0.9452 |
| Imbalance sign | 0.4167 | 0.7957 |
| Logistic regression | 0.5833 | 1.9542 |
| XGBoost | 0.5000 | 1.0690 |
| MLP | 0.5833 | 1.0901 |

Adding features did not produce a consistent accuracy improvement in this sample.
The GRU comparison uses a smaller, separately matched set of sequence targets.

![Feature ablations at 500 ms](docs/figures/ablations.png)

These results demonstrate the pipeline, not a reliable trading signal. Validation
also selected learned checkpoints, and there are too few observations for a useful
generalization claim. [Results and limitations](docs/results.md) explains the horizon
comparisons, class imbalance, sequence comparison, and missing 5-second evaluation.

## Repository scope

This repository contains the implementation, tests, configurations, and selected
results. Raw captures, processed datasets, checkpoints, virtual environments, and
run archives are omitted. New data and results are ignored by Git.

The documentation is maintained separately from experiment code. Experiments write
JSON, Parquet, model checkpoints, and figures; they do not write Markdown documents.

Further work would need longer, separated recording periods and a frozen final
evaluation. A fill-aware execution simulator is not implemented. Python timings
measure local inference work, not exchange-to-order latency.
