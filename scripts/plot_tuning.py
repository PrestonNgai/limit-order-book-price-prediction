"""Plot the primary forward comparison from saved experiment metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix")
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    horizon = result["primary_horizon_ms"]
    models = ["majority", "imbalance", "logistic", "xgboost", "tuned"]
    labels = ["Majority", "Imbalance", "Logistic", "XGBoost", "Tuned logistic"]
    pooled = {r["model"]: r["metrics"] for r in result["forward"] if r["horizon_ms"] == horizon}
    colors = ["#7a7a7a", "#e69f00", "#009e73", "#0072b2", "#cc79a7"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["balanced_accuracy_present_classes", "log_loss"],
        ["Balanced accuracy", "Log loss"],
        strict=True,
    ):
        values = [pooled[m][metric] for m in models]
        ax.bar(range(len(models)), values, color=colors, alpha=0.75, label="Pooled TEST")
        for i, model in enumerate(models):
            sessions = [r[metric] for r in result["primary_per_session"] if r["model"] == model]
            ax.scatter(
                i + np.linspace(-0.12, 0.12, len(sessions)),
                sessions,
                color="black",
                s=18,
                label="Individual sessions" if i == 0 else None,
                zorder=3,
            )
        ax.set_xticks(range(len(models)), labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    axes[0].axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="Constant-class recall")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{horizon} ms price direction | three later-date TEST sessions\n"
        f"Pooled n={pooled['tuned']['rows']:,}; models fixed before evaluation"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
