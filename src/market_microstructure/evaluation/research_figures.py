"""Plot validation diagnostics with matplotlib."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from market_microstructure.models.experiments import read_json


def render_research(
    destination, summary, paths, config, *, split="validation", importance_root=None
):
    output = destination / "figures"
    output.mkdir()
    records = summary["records"]
    complete = [r for r in records if r["status"] == "complete"]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "microstructure-research-v1",
        }
    )

    def save(fig, name, title):
        scope = (
            "Exploratory validation | TEST sealed"
            if split == "validation"
            else "Frozen models | Final TEST evaluation"
        )
        fig.suptitle(title + "\n" + scope, fontsize=13)
        for extension in ("png", "svg"):
            fig.savefig(
                output / f"{name}.{extension}",
                dpi=160,
                bbox_inches="tight",
                metadata={"Creator": "market-microstructure-research"}
                if extension == "svg"
                else None,
            )
        plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), layout="constrained", sharex=True)
    origin = min(t["timestamp_ns"][0].as_py() for t in paths.values() if t.num_rows)
    for partition, table in paths.items():
        keys = list(
            zip(
                *[
                    table[n].to_pylist()
                    for n in ("source", "symbol", "run_id", "session_id", "segment_id")
                ],
                strict=True,
            )
        )
        times = (table["timestamp_ns"].to_numpy() - origin) / 1e9
        for i, (field, label) in enumerate(
            (
                ("f_midprice", "Mid (USD)"),
                ("f_spread", "Spread (USD)"),
                ("f_obi_1", "L1 imbalance"),
                ("f_microprice_minus_mid", "Microprice - mid (USD)"),
            )
        ):
            # A line must not bridge a gap or split boundary.
            start = 0
            for stop in range(1, len(keys) + 1):
                if stop == len(keys) or keys[stop] != keys[start]:
                    if field in table.column_names:
                        axes[i].plot(
                            times[start:stop],
                            table[field].to_numpy()[start:stop],
                            label=partition if start == 0 else None,
                        )
                    start = stop
            axes[i].set_ylabel(label)
            axes[i].grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("Seconds since first plotted observation")
    save(fig, "market_paths", "Book state and microprice displacement")

    horizons = config["horizons_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, partition in zip(axes, ("train", split), strict=True):
        data = np.array([summary["class_distribution"][f"{h}ms"][partition] for h in horizons])
        bottom = np.zeros(len(horizons))
        for i, name in enumerate(("DOWN", "STABLE", "UP")):
            ax.bar(np.arange(len(horizons)), data[:, i], bottom=bottom, label=name)
            bottom += data[:, i]
        ax.set_xticks(np.arange(len(horizons)), [str(h) for h in horizons])
        ax.set(title=partition.upper(), xlabel="Horizon (ms)", ylabel="Purged target count")
        ax.legend()
    save(fig, "class_distribution", "Target availability and class balance")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, cohort in enumerate(("snapshot", "sequence")):
        for model in config["models"]:
            selected = sorted(
                [
                    r
                    for r in complete
                    if r["model"] == model and r["cohort"] == cohort and r["feature_set"] == "all"
                ],
                key=lambda r: r["horizon_ms"],
            )
            if selected:
                x = [r["horizon_ms"] for r in selected]
                for col, metric in enumerate(("balanced_accuracy_present_classes", "log_loss")):
                    axes[row, col].plot(
                        x, [r["metrics"][metric] for r in selected], "o-", label=model
                    )
        for col, label in enumerate(("Balanced accuracy (observed classes)", "Log loss")):
            ax = axes[row, col]
            ax.set(title=f"{cohort} cohort", xlabel="Horizon (ms)", ylabel=label, xscale="log")
            ax.set_xticks(horizons, [str(h) for h in horizons])
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=8)
    save(fig, "signal_decay", "Performance versus horizon (missing values omitted; cohorts differ)")

    focus = next(
        (h for h in config["ablation_horizons_ms"] if any(r["horizon_ms"] == h for r in complete)),
        next((r["horizon_ms"] for r in complete), horizons[0]),
    )
    selected = [r for r in complete if r["horizon_ms"] == focus and r["feature_set"] == "all"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, r in zip(axes.flat, selected, strict=False):
        ax.set_visible(True)
        matrix = np.array(r["metrics"]["confusion_matrix_actual_rows_predicted_columns"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                )
        ax.set_xticks(range(3), ["D", "S", "U"])
        ax.set_yticks(range(3), ["D", "S", "U"])
        ax.set(title=f"{r['model']} (n={matrix.sum()})", xlabel="Predicted", ylabel="Actual")
    save(
        fig,
        "confusion_matrices",
        f"Confusion counts at {focus} ms; sequence rows are a separate cohort",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, model in zip(axes, ("xgboost", "mlp"), strict=True):
        order = [*config["feature_sets"], "all"]
        rows = [
            next(
                (
                    r
                    for r in complete
                    if r["model"] == model and r["horizon_ms"] == focus and r["feature_set"] == name
                ),
                None,
            )
            for name in order
        ]
        for metric, label in (
            ("accuracy", "Accuracy"),
            ("balanced_accuracy_present_classes", "Balanced accuracy"),
        ):
            ax.plot(
                range(len(rows)),
                [r["metrics"][metric] if r else np.nan for r in rows],
                "o-",
                label=label,
            )
        ax.set_xticks(range(len(rows)), [str(r["features"]) if r else "NA" for r in rows])
        ax.set(
            title=model,
            xlabel="Cumulative features: book + OBI + flow + trades + remainder",
            ylabel=f"{split.capitalize()} score",
            ylim=(0, 1),
        )
        ax.legend()
    save(fig, "ablations", f"Predeclared feature ablations at {focus} ms")

    importance = (
        (importance_root or destination)
        / "runs"
        / f"xgboost_{focus}ms_all"
        / "feature_importance.json"
    )
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    if importance.exists():
        rows = sorted(read_json(importance)["rows"], key=lambda x: x["normalized_total_gain"])[-12:]
        ax.barh([r["feature"] for r in rows], [r["normalized_total_gain"] for r in rows])
        ax.set_xlabel("Normalized TRAIN split total gain (not a causal effect)")
    else:
        ax.text(0.5, 0.5, "Tree importance unavailable", ha="center", transform=ax.transAxes)
    save(fig, "feature_importance", f"XGBoost feature importance at {focus} ms")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    metrics = list(next(iter(summary["regime_cutoffs"].values()))["metrics"])
    for ax, metric in zip(axes.flat, metrics, strict=True):
        labels = []
        for r in selected:
            if r["cohort"] != "snapshot":
                continue
            bins = r["regimes"]["regime_" + metric]
            labels = list(bins)
            values = [bins[k].get("balanced_accuracy_present_classes", np.nan) for k in labels]
            ax.plot(range(len(labels)), values, "o-", label=r["model"])
        ax.set_xticks(range(len(labels)), labels)
        ax.set(
            title=metric,
            xlabel="TRAIN quantile bin (ties lower)",
            ylabel="Balanced accuracy",
            ylim=(0, 1.05),
        )
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=8)
    save(fig, "regimes", f"TRAIN-defined regimes at {focus} ms (empty bins omitted)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    for r in selected:
        if r["cohort"] != "snapshot":
            continue
        bins = [b for b in r["metrics"]["calibration"] if b["rows"]]
        axes[0].plot(
            [b["mean_confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            "o-",
            label=r["model"],
        )
        c = r["confidence"]
        axes[1].plot(
            [v["threshold"] for v in c],
            [v["signal_frequency_per_eligible_observation"] for v in c],
            "o-",
        )
        axes[2].plot(
            [v["threshold"] for v in c],
            [
                v["mean_signed_mid_move_bps"]
                if v["mean_signed_mid_move_bps"] is not None
                else np.nan
                for v in c
            ],
            "o-",
        )
    axes[0].set(
        xlabel="Probability assigned to hard prediction",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8)
    axes[1].set(
        xlabel="Frozen confidence threshold", ylabel="Directional signal fraction", ylim=(0, 1)
    )
    axes[2].set(xlabel="Frozen confidence threshold", ylabel="Mean signed future mid move (bps)")
    save(
        fig, "confidence_and_moves", f"Calibration, coverage and cost-free mid moves at {focus} ms"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    timed = [r for r in selected if r["timing"]["status"] == "complete"]
    for ax, cohort in zip(axes, ("snapshot", "sequence"), strict=True):
        rows = [r for r in timed if r["cohort"] == cohort]
        ax.bar(
            [r["model"] for r in rows],
            [r["timing"]["single"]["median_call_ns"] / 1e6 for r in rows],
        )
        ax.set(title=cohort, ylabel="Warmed single-call median (ms)")
    save(
        fig,
        "complexity_timing",
        "Python timing: separate scopes; no exchange-to-order latency claim",
    )
