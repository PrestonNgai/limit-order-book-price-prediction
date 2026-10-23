"""Compare fixed regression models across declared prediction horizons."""

from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from market_microstructure.models.experiments import begin, check_dataset, commit, fail, read_json
from market_microstructure.models.regression import train_regression_run
from market_microstructure.processing.research_dataset import _write_json


def render_regression(destination, records, *, split):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(destination) / "figures"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    complete = [r for r in records if r["status"] == "complete"]
    for model in sorted({r["model"] for r in complete}):
        rows = sorted([r for r in complete if r["model"] == model], key=lambda r: r["horizon_ms"])
        for ax, field in zip(axes, ("mae_bps", "rmse_bps"), strict=True):
            ax.plot([r["horizon_ms"] for r in rows], [r[field] for r in rows], "o-", label=model)
            ax.set(xlabel="Horizon (ms)", ylabel=field.replace("_", " ").upper(), xscale="log")
            ax.grid(alpha=0.2)
    for ax in axes:
        if ax.lines:
            ax.legend()
    scope = (
        "Exploratory validation | TEST sealed"
        if split == "validation"
        else "Frozen models | Final TEST"
    )
    fig.suptitle("Future-return regression\n" + scope)
    for extension in ("png", "svg"):
        fig.savefig(output / f"regression.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_regression_comparison(dataset, destination, config, *, horizons=(100, 500, 1000, 5000)):
    manifest, digest = check_dataset(dataset)
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or set(horizons) - set(manifest["horizons_ms"])
    ):
        raise ValueError("regression horizons must be unique and present in the dataset")
    destination = begin(destination)
    try:
        _write_json(
            destination / "config.json",
            {
                "base": asdict(config),
                "horizons_ms": list(horizons),
                "models": ["zero", "ridge", "xgboost"],
            },
        )
        (destination / "runs").mkdir()
        records = []
        for horizon in horizons:
            counts = manifest["target_counts"][f"{horizon}ms"]
            for model in ("zero", "ridge", "xgboost"):
                key = f"{model}_{horizon}ms"
                record = {"id": key, "model": model, "horizon_ms": horizon}
                if not counts["train"]["rows"] or not counts["validation"]["rows"]:
                    records.append(
                        {
                            **record,
                            "status": "unavailable",
                            "reason": "no TRAIN or validation targets",
                        }
                    )
                    continue
                output = destination / "runs" / key
                print(f"Fitting return regression {key}", flush=True)
                train_regression_run(
                    dataset, output, replace(config, model=model, horizon_ms=horizon)
                )
                records.append({**record, **read_json(output / "validation/metrics.json")})
        _write_json(
            destination / "research.json",
            {"scope": "validation", "test_accessed": False, "records": records},
        )
        render_regression(destination, records, split="validation")
        check_dataset(dataset, digest)
        return commit(
            destination,
            {
                "kind": "regression_research",
                "schema_version": 1,
                "status": "complete",
                "created_utc": datetime.now(UTC).isoformat(),
                "test_accessed": False,
                "dataset_manifest_sha256": digest,
                "completed_runs": sum(r["status"] == "complete" for r in records),
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise
