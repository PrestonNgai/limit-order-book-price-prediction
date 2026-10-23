"""Freeze completed experiments before a single, explicit final TEST evaluation."""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from market_microstructure.models.experiments import (
    artifact,
    begin,
    check_dataset,
    commit,
    evaluate_run,
    fail,
    load_run,
    read_json,
    software,
)
from market_microstructure.models.regression import evaluate_regression_run, load_regression_run
from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.research_dataset import _write_json
from market_microstructure.processing.split_dataset import iter_model_batches

from .regression import render_regression
from .research import analyze_run, checked_table, compare_records
from .research_figures import render_research


def verify_committed(root, kind):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if (
        (root / "_INCOMPLETE").exists()
        or manifest.get("kind") != kind
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"require committed {kind}")
    for name, info in manifest["outputs"].items():
        artifact(root, name, info["sha256"])
    return manifest


def source_hashes():
    root = Path(__file__).parents[1]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*.py"))}


def freeze_final_plan(research, dataset, destination, *, regression_runs=()):
    research, dataset = Path(research).resolve(), Path(dataset).resolve()
    research_manifest = verify_committed(research, "research_evaluation")
    manifest, digest = check_dataset(dataset, research_manifest["dataset_manifest_sha256"])
    summary = read_json(research / "research.json")
    config = read_json(research / "resolved_config.json")
    if summary["test_accessed"] or config["dataset_manifest_sha256"] != digest:
        raise ValueError("final plan requires TEST-sealed validation research")
    runs = []
    for record in summary["records"]:
        if record["status"] != "complete":
            continue
        path = research / "runs" / record["id"]
        _, trained = load_run(path)
        if (
            trained["test_accessed"]
            or trained["dataset"]["manifest_sha256"] != digest
            or trained["model"] != record["model"]
            or trained["horizon_ms"] != record["horizon_ms"]
        ):
            raise ValueError("classification run differs from declared validation research")
        runs.append(
            {
                "id": record["id"],
                "path": str(path),
                "manifest_sha256": file_hash(path / "manifest.json"),
            }
        )
    if not runs:
        raise ValueError("no completed classification runs to freeze")
    regression = []
    for item in regression_runs:
        path = Path(item).resolve()
        _, trained = load_regression_run(path)
        if trained["test_accessed"] or trained["dataset"]["manifest_sha256"] != digest:
            raise ValueError("regression run does not match the sealed dataset")
        key = f"{trained['model']}_{trained['horizon_ms']}ms"
        if any(r["id"] == key for r in regression):
            raise ValueError("duplicate regression model and horizon")
        regression.append(
            {"id": key, "path": str(path), "manifest_sha256": file_hash(path / "manifest.json")}
        )
    destination = begin(destination)
    try:
        _write_json(
            destination / "plan.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "research": str(research),
                "research_manifest_sha256": file_hash(research / "manifest.json"),
                "dataset": str(dataset),
                "dataset_manifest_sha256": digest,
                "classification_runs": runs,
                "regression_runs": regression,
                "evaluation_config": config,
                "source_sha256": source_hashes(),
                "policy": "evaluate every declared model without retraining "
                "or TEST-based selection",
                "ranges": manifest["ranges"],
            },
        )
        return commit(
            destination,
            {
                "kind": "final_test_plan",
                "status": "complete",
                "schema_version": 1,
                "test_accessed": False,
                "created_utc": datetime.now(UTC).isoformat(),
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise


def run_final_test(plan_directory, destination, *, allow_test=False):
    if not allow_test:
        raise PermissionError("final evaluation requires explicit --allow-test")
    plan_directory = Path(plan_directory)
    verify_committed(plan_directory, "final_test_plan")
    plan = read_json(plan_directory / "plan.json")
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("source changed after freezing the final plan")
    dataset, research = Path(plan["dataset"]), Path(plan["research"])
    if file_hash(research / "manifest.json") != plan["research_manifest_sha256"]:
        raise ValueError("research changed after freezing")
    verify_committed(research, "research_evaluation")
    manifest, digest = check_dataset(dataset, plan["dataset_manifest_sha256"])
    for item in plan["classification_runs"] + plan["regression_runs"]:
        if file_hash(Path(item["path"]) / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("frozen run manifest changed")
    for item in plan["classification_runs"]:
        load_run(item["path"])
    for item in plan["regression_runs"]:
        load_regression_run(item["path"])
    config = plan["evaluation_config"]
    for horizon in config["horizons_ms"]:
        if manifest["target_counts"][f"{horizon}ms"]["test"]["rows"] > config["max_analysis_rows"]:
            raise ValueError("TEST exceeds the frozen analysis row limit")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("final output directory must be new")
    # An interrupted attempt still counts as TEST access; it must never be silently repeated.
    with (plan_directory / "test_access.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "started_utc": datetime.now(UTC).isoformat(),
                "output": str(destination.resolve()),
                "status": "started",
            },
            stream,
            indent=2,
        )
    destination = begin(destination)
    try:
        summary = read_json(research / "research.json")
        prior = {r["id"]: r for r in summary["records"]}
        records, predictions, sessions, joined_by_horizon = [], {}, {}, {}
        distributions, cutoffs = {}, {}
        paths = {
            "test": checked_table(
                dataset, manifest, "test/features.parquet", config["max_analysis_rows"]
            )
        }
        (destination / "runs").mkdir()
        for horizon in config["horizons_ms"]:
            suffix = f"{horizon}ms"
            parts = list(
                iter_model_batches(dataset, horizon, "test", transformed=False, allow_test=True)
            )
            joined = pa.concat_tables(parts) if parts else None
            joined_by_horizon[horizon] = joined
            balance = (
                np.bincount(joined[f"y_class_{suffix}"].to_numpy() + 1, minlength=3)
                if joined is not None
                else np.zeros(3, dtype=int)
            )
            distributions[suffix] = {
                "train": summary["class_distribution"][suffix]["train"],
                "test": balance.tolist(),
            }
            name = manifest["regime_cutoffs"][suffix]
            cutoffs[suffix] = read_json(
                artifact(dataset, name, manifest["outputs"][name]["sha256"])
            )
        for item in plan["classification_runs"]:
            record = copy.deepcopy(prior[item["id"]])
            horizon, path = record["horizon_ms"], Path(item["path"])
            count_key = "sequence_rows" if record["cohort"] == "sequence" else "rows"
            count = manifest["target_counts"][f"{horizon}ms"]["test"][count_key]
            record["test_rows"] = count
            if not count:
                record.update(status="unavailable", reason="no eligible final TEST targets")
                for key in ("metrics", "confidence", "regimes", "accuracy_uncertainty"):
                    record.pop(key, None)
            else:
                output = destination / "runs" / item["id"]
                evaluate_run(path, dataset, output, split="test", allow_test=True)
                analysis, table, ids = analyze_run(
                    path,
                    joined_by_horizon[horizon],
                    cutoffs[f"{horizon}ms"],
                    config,
                    split="test",
                    predictions_path=output / "test/predictions.parquet",
                )
                record.update(**analysis)
                predictions[item["id"]], sessions[item["id"]] = table, ids
            records.append(record)
        regression_records = []
        if plan["regression_runs"]:
            (destination / "regression").mkdir()
        for item in plan["regression_runs"]:
            _, trained = load_regression_run(item["path"])
            horizon = trained["horizon_ms"]
            if not manifest["target_counts"][f"{horizon}ms"]["test"]["rows"]:
                regression_records.append({"id": item["id"], "status": "unavailable"})
                continue
            output = destination / "regression" / item["id"]
            evaluate_regression_run(item["path"], dataset, output, split="test", allow_test=True)
            regression_records.append(
                {
                    "id": item["id"],
                    "model": trained["model"],
                    "horizon_ms": horizon,
                    **read_json(output / "test/metrics.json"),
                }
            )
        final = {
            "scope": "final TEST; all models and thresholds frozen before access",
            "test_accessed": True,
            "records": records,
            "regression": regression_records,
            "comparisons": compare_records(records, predictions, sessions, config),
            "class_distribution": distributions,
            "regime_cutoffs": cutoffs,
            "timing_scope": "timing and fit metadata retained from validation experiments",
        }
        _write_json(destination / "research.json", final)
        _write_json(destination / "frozen_plan.json", plan)
        render_research(destination, final, paths, config, split="test", importance_root=research)
        if regression_records:
            render_regression(destination, regression_records, split="test")
        check_dataset(dataset, digest)
        result = commit(
            destination,
            {
                "kind": "final_test_evaluation",
                "status": "complete",
                "schema_version": 1,
                "created_utc": datetime.now(UTC).isoformat(),
                "test_accessed": True,
                "dataset_manifest_sha256": digest,
                "plan_manifest_sha256": file_hash(plan_directory / "manifest.json"),
                "completed_runs": sum(r["status"] == "complete" for r in records),
                "regression_runs": len(regression_records),
                "software": software(),
            },
        )
        _write_json(
            plan_directory / "test_access.json",
            {
                "status": "complete",
                "output": str(destination.resolve()),
                "output_manifest_sha256": file_hash(destination / "manifest.json"),
            },
        )
        return result
    except Exception as exc:
        fail(destination, exc)
        raise
