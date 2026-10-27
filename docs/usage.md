# Running experiments

Run these commands from the repository root after installing the package. Each
processing or model run needs a new output directory; existing runs are not overwritten.

## Collect data

~~~bash
python scripts/collect.py --config configs/collector.yaml --duration 600 --output data/raw
~~~

This records BTC-USD from Coinbase Exchange. Ctrl+C closes a longer recording
cleanly. Check the capture summary and warnings before processing.

The adapter uses level2_batch, matches, and heartbeat messages. It records product
metadata when available; see the
[Coinbase channel documentation](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels).

## Build features and splits

~~~bash
python scripts/features.py --raw-dir data/raw --output data/processed/features
python scripts/split.py --input data/processed/features/research --output data/processed/splits
~~~

Feature construction writes snapshots to `observations/` and features, labels, and
their manifest to `research/`. The split command consumes `research/`.

To inspect snapshots without building features:

~~~bash
python scripts/process.py --raw-dir data/raw --output data/processed/observations
~~~

Raw discovery accepts the recorder's closed JSONL files, or compressed JSONL.zst
files with the optional compression dependency installed. Partial files are rejected.
Use **--input** to select explicit closed files if a directory also contains an
unfinished capture.

The default class threshold is one quoted tick. Feature construction reads the
captured quote increment; if metadata is absent, provide **--tick-size** using a
verified value for that product.

## Train one model

~~~bash
python scripts/train.py --dataset data/processed/splits --model logistic --horizon-ms 500 --output results/logistic-500
python scripts/train.py --dataset data/processed/splits --model xgboost --horizon-ms 500 --output results/tree-500
python scripts/train.py --dataset data/processed/splits --model mlp --horizon-ms 500 --output results/mlp-500
python scripts/train.py --dataset data/processed/splits --model gru --horizon-ms 500 --output results/gru-500
~~~

Other model names are majority, imbalance, and sequence_mlp. The latter trains a
snapshot MLP on the same eligible target rows as the GRU.

The sequence length must match the split configuration. If it changes, rebuild
splits and change both sequence model configurations. Training saves its resolved
configuration, preprocessing state, model, validation predictions, and metrics.

Re-evaluate a saved model without fitting:

~~~bash
python scripts/evaluate.py --run results/tree-500 --dataset data/processed/splits --output results/tree-500-check
~~~

Evaluation defaults to validation. Individual TEST evaluation requires
**--split test --allow-test**. For the full comparison, use the frozen-plan workflow
below after finishing model selection.

## Run the comparisons

~~~bash
python scripts/research.py --dataset data/processed/splits --config configs/research.yaml --output results/comparison
~~~

The configuration specifies horizons, models, cumulative feature subsets, confidence
thresholds, seed, resource limits, and bootstrap settings. A null dataset hash binds
the run to the supplied split manifest and records that hash in resolved_config.json.
To require exactly the same dataset on a later run, put that SHA256 digest in the YAML.

The output includes:

- research.json: all metrics, class counts, confidence diagnostics, regime breakdowns,
  and paired comparisons.
- runs/: frozen models, per-run settings, predictions, and fit metadata.
- figures/: PNG and SVG plots.
- manifest.json: hashes, source identity, environment, and completion status.
- resolved_config.json: settings used for this dataset.

Empty combinations are marked unavailable. Failed runs retain `_INCOMPLETE` and
`failure.json`; retry in a new directory after fixing the cause. Statistical results
are reproducible in the tested environment; timings and cross-version results can vary.

## Secondary return regression

Train one model, or compare zero-return, ridge, and XGBoost regressors at all horizons:

~~~bash
python scripts/regression.py train --dataset data/processed/splits --model ridge --horizon-ms 500 --output results/ridge-500
python scripts/regression.py compare --dataset data/processed/splits --output results/return-comparison
~~~

Use **--horizons-ms 100 500 1000 5000** to declare horizons explicitly. Settings are in
configs/regression.yaml. Outputs include validation metrics, saved runs, and error figures.

## Freeze and evaluate final TEST

Finish validation comparisons and freeze all choices first. The example includes
one regression run; supply multiple paths to include the full regression grid:

~~~bash
python scripts/final_test.py freeze --research results/comparison --dataset data/processed/splits --regression-runs results/return-comparison/runs/ridge_500ms --output results/final-plan
python scripts/final_test.py evaluate --plan results/final-plan --output results/final-test --allow-test
~~~

Omit **--regression-runs** for classification alone. Each plan allows one attempt,
including a failed attempt. Final outputs include `research.json`, TEST predictions
and metrics, and PNG/SVG figures. See [Final evaluation](final_evaluation.md) for the
hash checks and access policy.

## Configuration notes

Collector and processing YAML paths are relative to the configuration file.
Explicit CLI paths are relative to the current directory.

The default research layer loads bounded partitions in memory, with a 250,000-row
cap. Training has separate row and working-memory limits.

For compressed capture files:

~~~bash
python -m pip install -e ".[compression]"
~~~

Each command's **--help** lists its configuration overrides.
