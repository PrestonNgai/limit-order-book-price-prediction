# Running experiments

Run these commands from the repository root after installing the package. Each
processing or model run needs a new output directory; existing runs are not overwritten.

## Collect data

~~~bash
python scripts/collect.py --config configs/collector.yaml --duration 600 --output data/raw
~~~

This connects to the public Coinbase Exchange feed and records BTC-USD. Stop a longer
run with Ctrl+C. The recorder closes its journal and prints message counts. Inspect
warnings and capture summaries before treating a recording as usable data.

The adapter uses level2_batch, matches, and heartbeat messages. It records product
metadata when available. The protocol was implemented against the
[Coinbase channel documentation](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels);
feed availability and permissions may change.

Use separate capture runs across longer periods for research. Ten minutes is a
convenient collection example, not a claim of adequate sample size.

## Build features and splits

~~~bash
python scripts/features.py --raw-dir data/raw --output data/processed/features
python scripts/split.py --input data/processed/features/research --output data/processed/splits
~~~

The feature command includes raw replay and snapshot construction. Its output has
an observations directory and a research directory containing features, labels,
and their manifest. The split command consumes the latter.

To inspect snapshots without building features:

~~~bash
python scripts/process.py --raw-dir data/raw --output data/processed/observations
~~~

Raw discovery accepts the recorder's closed JSONL files, or compressed JSONL.zst
files with the optional compression dependency installed. Partial files are rejected.
Use **--input** to select explicit closed files if a directory also contains an
unfinished capture. Do not rename a partial file to bypass that check.

The default class threshold is one quoted tick. Feature construction reads the
captured quote increment; if metadata is absent, provide **--tick-size** using a
verified value for that product. Do not silently reuse another product's tick size.

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

Evaluation defaults to validation. TEST access requires both **--split test** and
**--allow-test**. Use that only after freezing model, feature, threshold, and
evaluation choices. The multi-experiment research runner has no TEST option.

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

Empty validation combinations are recorded as unavailable. A failed run retains an
_INCOMPLETE marker and failure.json; use a fresh destination after fixing the cause.
Statistics are deterministic within the tested environment, but timing and timestamps
will differ. Cross-version model results may also change.

## Configuration notes

Collector and processing YAML paths are relative to the configuration file.
Explicit CLI paths are relative to the current directory.

The default research layer loads bounded partitions in memory, with a 250,000-row
cap. Model training has separate row and working-memory limits. These are guards,
not a promise that every configuration fits available RAM.

For compressed capture files:

~~~bash
python -m pip install -e ".[compression]"
~~~

Use **--help** on an entry point for its available overrides. Configurations reject
unknown model fields rather than silently ignoring misspelled settings.
