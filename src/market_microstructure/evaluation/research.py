"""Run validation comparisons and save metrics, predictions and figures."""

import shutil
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from market_microstructure.models.experiments import (
    artifact,
    begin,
    check_dataset,
    commit,
    fail,
    load_run,
    read_json,
    software,
    train_run,
)
from market_microstructure.models.learned_config import LearnedConfig, config_from_mapping
from market_microstructure.models.learned_experiments import benchmark, train_learned_run
from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.regime_cutoffs import apply_regimes
from market_microstructure.processing.research_dataset import _write_json
from market_microstructure.processing.split_dataset import iter_model_batches

from .metrics import classification_metrics
from .research_analysis import align_predictions, arrays, cluster_interval, confidence_analysis

MODELS = ("majority", "imbalance", "logistic", "xgboost", "mlp", "gru", "sequence_mlp")
FIELDS = {
    "schema_version",
    "dataset_manifest_sha256",
    "horizons_ms",
    "models",
    "ablation_horizons_ms",
    "feature_sets",
    "confidence_thresholds",
    "minimum_sessions",
    "bootstrap_repeats",
    "seed",
    "max_analysis_rows",
    "model_parameters",
}


def validate_config(config, manifest):
    if not isinstance(config, dict) or set(config) != FIELDS or config["schema_version"] != 1:
        raise ValueError("unknown/missing research configuration fields")
    expected_hash = config["dataset_manifest_sha256"]
    if expected_hash is not None and (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(c not in "0123456789abcdef" for c in expected_hash)
    ):
        raise ValueError("dataset_manifest_sha256 must be null or a lowercase SHA256 digest")
    for key in ("horizons_ms", "models", "ablation_horizons_ms"):
        values = config[key]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError("configuration lists must be unique")
    if not config["horizons_ms"] or set(config["horizons_ms"]) - set(manifest["horizons_ms"]):
        raise ValueError("unknown or empty horizons")
    if (
        not config["models"]
        or set(config["models"]) - set(MODELS)
        or set(config["ablation_horizons_ms"]) - set(config["horizons_ms"])
    ):
        raise ValueError("unknown models or ablation horizons")
    for key, low in (
        ("minimum_sessions", 2),
        ("bootstrap_repeats", 100),
        ("max_analysis_rows", 1),
        ("seed", 0),
    ):
        if type(config[key]) is not int or config[key] < low:
            raise ValueError("invalid research resource/uncertainty setting")
    thresholds = config["confidence_thresholds"]
    if (
        not thresholds
        or any(
            type(t) not in (int, float) or not np.isfinite(t) or not 0 <= t <= 1 for t in thresholds
        )
        or thresholds != sorted(set(thresholds))
    ):
        raise ValueError("confidence thresholds must be ordered unique values in [0,1]")
    for name, columns in config["feature_sets"].items():
        if (
            not name
            or not all(c.isalnum() or c == "_" for c in name)
            or name == "all"
            or not columns
            or len(set(columns)) != len(columns)
            or set(columns) - set(manifest["feature_columns"])
        ):
            raise ValueError("invalid ablation name or feature subset")
    if set(config["model_parameters"]) - set(MODELS):
        raise ValueError("unknown model parameter section")
    for h in config["horizons_ms"]:
        for model in config["models"]:
            make_model_config(config, model, h)


def make_model_config(config, model, horizon):
    parameters = dict(config["model_parameters"].get(model, {}))
    parameters.update(model=model, horizon_ms=horizon, seed=config["seed"])
    return config_from_mapping(parameters)


def checked_table(dataset, manifest, name, limit):
    path = artifact(dataset, name, manifest["outputs"][name]["sha256"])
    if pq.read_metadata(path).num_rows > limit:
        raise ValueError("research table exceeds max_analysis_rows")
    return pq.read_table(path)


def analyze_run(run, joined, cutoffs, config, *, split="validation", predictions_path=None):
    frozen, manifest = load_run(run)
    del frozen
    table = pq.read_table(predictions_path or run / f"{split}/predictions.parquet")
    raw = align_predictions(table, joined, manifest["horizon_ms"], split=split)
    y, pred, probabilities = arrays(table)
    metrics = classification_metrics(y, pred, probabilities)
    sessions = np.array(
        [
            "|".join(map(str, x))
            for x in zip(
                *[raw[n].to_pylist() for n in ("source", "symbol", "run_id", "session_id")],
                strict=True,
            )
        ]
    )
    regimes = apply_regimes(raw, cutoffs)
    breakdown = {}
    for column in regimes.column_names:
        labels = regimes[column].to_pylist()
        bins = range(len(cutoffs["metrics"][column.removeprefix("regime_")]["thresholds"]) + 1)
        breakdown[column] = {}
        for value in [*bins, None]:
            mask = np.array([x == value for x in labels], dtype=bool)
            breakdown[column]["unknown" if value is None else str(value)] = classification_metrics(
                y[mask], pred[mask], probabilities[mask]
            )
    uncertainty = cluster_interval(
        (y == pred).astype(float),
        sessions,
        minimum_sessions=config["minimum_sessions"],
        repeats=config["bootstrap_repeats"],
        seed=config["seed"],
    )
    return (
        {
            "metrics": metrics,
            "confidence": confidence_analysis(table, config["confidence_thresholds"]),
            "regimes": breakdown,
            "accuracy_uncertainty": uncertainty,
        },
        table,
        sessions,
    )


def run_research(dataset, destination, config_path):
    dataset, config_path = Path(dataset), Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest, digest = check_dataset(dataset)
    validate_config(config, manifest)
    check_dataset(dataset, config["dataset_manifest_sha256"])
    config["dataset_manifest_sha256"] = digest
    limit = config["max_analysis_rows"]
    paths = {
        split: checked_table(dataset, manifest, f"{split}/features.parquet", limit)
        for split in ("train", "validation")
    }
    destination = begin(destination)
    try:
        shutil.copyfile(config_path, destination / "research.yaml")
        shutil.copyfile(dataset / "manifest.json", destination / "dataset_manifest.json")
        _write_json(destination / "resolved_config.json", config)
        records, predictions, sessions_by_run = [], {}, {}
        distributions, cutoffs_by_horizon = {}, {}
        (destination / "runs").mkdir()
        for horizon in config["horizons_ms"]:
            suffix = f"{horizon}ms"
            counts = manifest["target_counts"][suffix]
            if any(counts[s]["rows"] > limit for s in ("train", "validation")):
                raise ValueError("research targets exceed max_analysis_rows")
            joined_batches = list(
                iter_model_batches(dataset, horizon, "validation", transformed=False)
            )
            joined = pa.concat_tables(joined_batches) if joined_batches else None
            cutoff_name = manifest["regime_cutoffs"][suffix]
            cutoffs = read_json(
                artifact(dataset, cutoff_name, manifest["outputs"][cutoff_name]["sha256"])
            )
            train_range = manifest["ranges"]["train"]
            if (
                cutoffs["fit_split"] != "train"
                or cutoffs["horizon_ms"] != horizon
                or cutoffs["fit_rows"] != counts["train"]["rows"]
                or cutoffs["train_start_ns"] != train_range["start_ns"]
                or cutoffs["train_end_exclusive_ns"] != train_range["end_exclusive_ns"]
            ):
                raise ValueError("regime cutoffs do not match TRAIN")
            cutoffs_by_horizon[suffix] = cutoffs
            distributions[suffix] = {}
            for split in ("train", "validation"):
                balance = np.zeros(3, dtype=np.int64)
                for batch in iter_model_batches(dataset, horizon, split, transformed=False):
                    balance += np.bincount(batch[f"y_class_{suffix}"].to_numpy() + 1, minlength=3)
                distributions[suffix][split] = balance.tolist()
            jobs = [(m, "all", None) for m in config["models"]]
            if horizon in config["ablation_horizons_ms"]:
                jobs += [
                    (m, name, names)
                    for m in ("xgboost", "mlp")
                    if m in config["models"]
                    for name, names in config["feature_sets"].items()
                ]
            for model, feature_set, names in jobs:
                key = f"{model}_{suffix}_{feature_set}"
                sequence = model in ("gru", "sequence_mlp")
                count_key = "sequence_rows" if sequence else "rows"
                record = {
                    "id": key,
                    "model": model,
                    "horizon_ms": horizon,
                    "feature_set": feature_set,
                    "features": len(names or manifest["feature_columns"]),
                    "cohort": "sequence" if sequence else "snapshot",
                    "train_rows": counts["train"][count_key],
                    "validation_rows": counts["validation"][count_key],
                }
                if not counts["train"][count_key] or not counts["validation"][count_key]:
                    record.update(
                        status="unavailable", reason="no eligible TRAIN or validation targets"
                    )
                    records.append(record)
                    continue
                cfg = make_model_config(config, model, horizon)
                run = destination / "runs" / key
                print(f"Fitting {key}", flush=True)
                if names is None:
                    fitted = train_run(dataset, run, cfg)
                else:
                    fitted = train_learned_run(dataset, run, cfg, feature_columns=names)
                analysis, table, sessions = analyze_run(run, joined, cutoffs, config)
                record.update(status="complete", **analysis)
                record["fit"] = fitted["fit"]
                record["features"] = len(
                    fitted.get("active_feature_columns", fitted["feature_columns"])
                )

                timing_path = run / "inference_timing.json"
                if timing_path.exists():
                    record["timing"] = read_json(timing_path)
                else:
                    frozen, _ = load_run(run)
                    record["timing"] = benchmark(
                        frozen, joined, LearnedConfig(model="mlp"), "validation"
                    )
                    record["timing"]["hardware"] = software()
                predictions[key], sessions_by_run[key] = table, sessions
                records.append(record)
        comparisons = compare_records(records, predictions, sessions_by_run, config)
        summary = {
            "scope": "validation; checkpoints selected on this same partition",
            "test_accessed": False,
            "records": records,
            "comparisons": comparisons,
            "class_distribution": distributions,
            "regime_cutoffs": cutoffs_by_horizon,
        }
        _write_json(destination / "research.json", summary)
        from .research_figures import render_research

        render_research(destination, summary, paths, config)
        check_dataset(dataset, digest)
        for split in ("train", "validation"):
            artifact(
                dataset,
                f"{split}/features.parquet",
                manifest["outputs"][f"{split}/features.parquet"]["sha256"],
            )
        result = {
            "schema_version": 1,
            "kind": "research_evaluation",
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "test_accessed": False,
            "dataset_manifest_sha256": digest,
            "config_sha256": file_hash(destination / "research.yaml"),
            "source_sha256": {
                p.relative_to(Path(__file__).parents[1]).as_posix(): file_hash(p)
                for p in sorted(Path(__file__).parents[1].rglob("*.py"))
            },
            "software": {**software(), "matplotlib": version("matplotlib")},
            "completed_runs": sum(r["status"] == "complete" for r in records),
            "unavailable_runs": sum(r["status"] != "complete" for r in records),
            "run_manifest_hashes": {
                p.parent.name: file_hash(p)
                for p in sorted((destination / "runs").glob("*/manifest.json"))
            },
        }
        return commit(destination, result)
    except Exception as exc:
        fail(destination, exc)
        raise


def compare_records(records, predictions, sessions, config):
    comparisons = []
    for record in records:
        if record["status"] != "complete":
            continue
        reference_model = "sequence_mlp" if record["cohort"] == "sequence" else "majority"
        reference = f"{reference_model}_{record['horizon_ms']}ms_all"
        if reference not in predictions:
            continue
        key = record["id"]
        a, b = predictions[key], predictions[reference]
        for column in ("row_id", "timestamp_ns", "label_end_ns", "target", "future_return"):
            if a[column].to_pylist() != b[column].to_pylist():
                raise ValueError("comparison cohorts differ")
        ya, pa_, _ = arrays(a)
        yb, pb, _ = arrays(b)
        values = (ya == pa_).astype(float) - (yb == pb).astype(float)
        comparisons.append(
            {
                "run": key,
                "reference": reference,
                "rows": len(values),
                "accuracy_difference": float(values.mean()),
                "paired_uncertainty": cluster_interval(
                    values,
                    sessions[key],
                    minimum_sessions=config["minimum_sessions"],
                    repeats=config["bootstrap_repeats"],
                    seed=config["seed"],
                ),
            }
        )
    return comparisons
