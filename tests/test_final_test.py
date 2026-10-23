from pathlib import Path

import pytest
import test_baselines
import yaml

from market_microstructure.evaluation.final_test import freeze_final_plan, run_final_test
from market_microstructure.evaluation.research import run_research
from market_microstructure.models.experiments import read_json
from market_microstructure.models.regression import RegressionConfig, train_regression_run
from market_microstructure.processing.build_dataset import file_hash

dataset = test_baselines.dataset


def research_run(dataset, tmp_path):
    config = yaml.safe_load(Path("configs/research.yaml").read_text(encoding="utf-8"))
    config.update(
        dataset_manifest_sha256=file_hash(dataset / "manifest.json"),
        horizons_ms=[500],
        models=["majority", "logistic"],
        ablation_horizons_ms=[],
        feature_sets={},
        bootstrap_repeats=100,
    )
    config["model_parameters"] = {}
    path = tmp_path / "research.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "comparison"
    run_research(dataset, output, path)
    return output


def test_final_freeze_does_not_read_test_and_prevents_repeat(dataset, tmp_path):
    regression = tmp_path / "ridge"
    train_regression_run(dataset, regression, RegressionConfig())
    research = research_run(dataset, tmp_path)
    test = dataset / "test/features.parquet"
    original = test.read_bytes()
    test.write_bytes(b"TEST forbidden before freeze")
    plan = tmp_path / "plan"
    frozen = freeze_final_plan(research, dataset, plan, regression_runs=[regression])
    assert frozen["test_accessed"] is False
    with pytest.raises(PermissionError):
        run_final_test(plan, tmp_path / "without-opt-in")
    assert not (plan / "test_access.json").exists()
    test.write_bytes(original)
    result = run_final_test(plan, tmp_path / "final", allow_test=True)
    assert result["test_accessed"] and result["completed_runs"] == 2
    summary = read_json(tmp_path / "final/research.json")
    assert len(summary["regression"]) == 1
    assert summary["regression"][0]["rows"] > 0
    assert all(r["test_rows"] == r["metrics"]["rows"] for r in summary["records"])
    assert (tmp_path / "final/figures/signal_decay.png").exists()
    assert all(r["accuracy_uncertainty"]["status"] == "unavailable" for r in summary["records"])
    with pytest.raises(FileExistsError):
        run_final_test(plan, tmp_path / "repeat", allow_test=True)


def test_changed_checkpoint_rejected_before_test_access(dataset, tmp_path):
    research = research_run(dataset, tmp_path)
    plan = tmp_path / "plan"
    freeze_final_plan(research, dataset, plan)
    model = research / "runs/majority_500ms_all/model.json"
    model.write_text(model.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_final_test(plan, tmp_path / "final", allow_test=True)
    assert not (plan / "test_access.json").exists()


def test_changed_source_rejected_before_test_access(dataset, tmp_path, monkeypatch):
    research = research_run(dataset, tmp_path)
    plan = tmp_path / "plan"
    freeze_final_plan(research, dataset, plan)
    monkeypatch.setattr("market_microstructure.evaluation.final_test.source_hashes", lambda: {})
    with pytest.raises(ValueError, match="source changed"):
        run_final_test(plan, tmp_path / "final", allow_test=True)
    assert not (plan / "test_access.json").exists()


def test_changed_regression_rejected_before_test_access(dataset, tmp_path):
    research = research_run(dataset, tmp_path)
    regression = tmp_path / "ridge"
    train_regression_run(dataset, regression, RegressionConfig())
    plan = tmp_path / "plan"
    freeze_final_plan(research, dataset, plan, regression_runs=[regression])
    model = regression / "model.json"
    model.write_text(model.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_final_test(plan, tmp_path / "final", allow_test=True)
    assert not (plan / "test_access.json").exists()
