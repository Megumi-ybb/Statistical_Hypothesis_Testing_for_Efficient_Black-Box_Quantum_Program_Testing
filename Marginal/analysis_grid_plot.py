
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "combined_results_long.csv"
FIG_DIR = SCRIPT_DIR / "figure"

POWER_METRICS = ["power_bitflip", "power_asymmetric", "power_depolarize"]
TYPE_I_METRIC = "type_i_error"
ALL_METRICS = POWER_METRICS + [TYPE_I_METRIC]

TEST_METHODS = ["tolerance", "bootstrap", "sequential", "auto"]

METHOD_COLORS = {
    "tolerance": "#1f77b4",
    "bootstrap": "#ff7f0e",
    "sequential": "#2ca02c",
    "auto": "#d62728",
}

METRIC_LINESTYLES = {
    "power_bitflip": "-",
    "power_asymmetric": "--",
    "power_depolarize": ":",
    "type_i_error": "-.",
}

METRIC_MARKERS = {
    "power_bitflip": "o",
    "power_asymmetric": "s",
    "power_depolarize": "^",
    "type_i_error": "x",
}

EXPERIMENTS = ["grovertest", "BV", "QFT", "QPE", "QWALK", "add"]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = ["n_qubits", "m_samples", "alpha", "value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pivot_for_plotting(df: pd.DataFrame, experiment: str = None) -> pd.DataFrame:
    df_filtered = df[df["metric"].isin(ALL_METRICS)].copy()

    if experiment is not None:
        df_filtered = df_filtered[df_filtered["experiment"] == experiment]

    pivot = df_filtered.pivot_table(
        index=["n_qubits", "m_samples", "test_method"],
        columns="metric",
        values="value",
        aggfunc="mean"
    ).reset_index()

    return pivot


def create_grid_plot(df: pd.DataFrame, output_path: Path, experiment_name: str = "") -> None:

    n_qubits_values = sorted(df["n_qubits"].dropna().unique().astype(int))
    n_qubits_values = [n for n in n_qubits_values if 2 <= n <= 12]

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    if experiment_name:
        fig.suptitle(f"Experiment: {experiment_name}", fontsize=14, fontweight="bold", y=1.02)

    axes_flat = axes.flatten()

    axes_right = []
    for idx, ax in enumerate(axes_flat):
        if idx < len(n_qubits_values):
            axes_right.append(ax.twinx())
        else:
            axes_right.append(None)

    for idx, ax in enumerate(axes_flat):
        ax_right = axes_right[idx]
        row = idx // 4
        col = idx % 4

        if idx < len(n_qubits_values):
            n_q = n_qubits_values[idx]
            df_nq = df[df["n_qubits"] == n_q].copy()

            if df_nq.empty:
                ax.set_visible(False)
                if ax_right:
                    ax_right.set_visible(False)
                continue

            for method in TEST_METHODS:
                df_method = df_nq[df_nq["test_method"] == method].sort_values("m_samples")
                if df_method.empty:
                    continue

                color = METHOD_COLORS.get(method, "gray")

                for metric in POWER_METRICS:
                    if metric in df_method.columns:
                        ls = METRIC_LINESTYLES.get(metric, "-")
                        marker = METRIC_MARKERS.get(metric, "o")
                        ax.plot(
                            df_method["m_samples"],
                            df_method[metric],
                            color=color,
                            linestyle=ls,
                            marker=marker,
                            markersize=3,
                            linewidth=1.2,
                            alpha=0.8
                        )

            for method in TEST_METHODS:
                df_method = df_nq[df_nq["test_method"] == method].sort_values("m_samples")
                if df_method.empty:
                    continue

                color = METHOD_COLORS.get(method, "gray")

                if TYPE_I_METRIC in df_method.columns:
                    ls = METRIC_LINESTYLES.get(TYPE_I_METRIC, "-.")
                    marker = METRIC_MARKERS.get(TYPE_I_METRIC, "x")
                    ax_right.plot(
                        df_method["m_samples"],
                        df_method[TYPE_I_METRIC],
                        color=color,
                        linestyle=ls,
                        marker=marker,
                        markersize=4,
                        linewidth=1.2,
                        alpha=0.8
                    )

            ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1, alpha=0.7)

            ax.set_title(f"n_qubits = {n_q}", fontsize=10, fontweight="bold")

            if row == 2:
                ax.set_xlabel("m_samples", fontsize=9)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

            if col == 0:
                ax.set_ylabel("Power\n(bitflip/depolarize/asymmetric)", fontsize=9, color="black")
            else:
                ax.set_ylabel("")

            show_right_label = False
            if row < 2 and col == 3:
                show_right_label = True
            elif row == 2 and col == 2:
                show_right_label = True

            if show_right_label:
                ax_right.set_ylabel("Type I Error\n(False Positive Rate)", fontsize=9, color="gray")
            else:
                ax_right.set_ylabel("")

            ax.set_ylim(-0.05, 1.05)
            ax_right.set_ylim(-0.05, 1.05)

            yticks = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
            ax.set_yticks(yticks)
            ax_right.set_yticks(yticks)

            if col != 0:
                ax.tick_params(axis="y", labelleft=False)

            if not show_right_label:
                ax_right.tick_params(axis="y", labelright=False)

            ax.tick_params(axis="both", labelsize=7)
            ax_right.tick_params(axis="y", labelsize=7, colors="gray")

            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_xscale("log")

            x_ticks = [50, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(x) for x in x_ticks], rotation=45, ha="right")

        else:
            ax.axis("off")

            legend_handles = []

            legend_handles.append(mlines.Line2D([], [], color="none", label="Test Methods:"))
            for method in TEST_METHODS:
                color = METHOD_COLORS.get(method, "gray")
                handle = mlines.Line2D([], [], color=color, linestyle="-", linewidth=2.5, label=f"  {method}")
                legend_handles.append(handle)

            legend_handles.append(mlines.Line2D([], [], color="none", label=""))

            legend_handles.append(mlines.Line2D([], [], color="none", label="Metrics:"))
            for metric in ALL_METRICS:
                ls = METRIC_LINESTYLES.get(metric, "-")
                marker = METRIC_MARKERS.get(metric, "o")
                label_name = metric.replace("_", " ").replace("power ", "")
                handle = mlines.Line2D(
                    [], [], color="black", linestyle=ls, marker=marker,
                    markersize=6, linewidth=1.5, label=f"  {label_name}"
                )
                legend_handles.append(handle)

            legend_handles.append(mlines.Line2D([], [], color="none", label=""))

            legend_handles.append(mlines.Line2D([], [], color="none", label="Thresholds:"))
            legend_handles.append(mlines.Line2D(
                [], [], color="red", linestyle="--", linewidth=1.5,
                label="  y = 0.05, 0.95"
            ))

            ax.legend(
                handles=legend_handles,
                loc="center",
                fontsize=9,
                frameon=True,
                title="Legend",
                title_fontsize=11,
                handlelength=2.5,
                labelspacing=0.8
            )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {DATA_PATH}")
    df_long = load_data(DATA_PATH)
    print(f"Loaded {len(df_long):,} rows")

    print(f"Processing experiments: {EXPERIMENTS}")

    for experiment in EXPERIMENTS:
        print(f"\nProcessing experiment: {experiment}")

        df_wide = pivot_for_plotting(df_long, experiment=experiment)

        if df_wide.empty:
            print(f"  No data for experiment '{experiment}', skipping...")
            continue

        print(f"  Pivoted to {len(df_wide):,} rows")
        print(f"  n_qubits values: {sorted(df_wide['n_qubits'].unique())}")
        print(f"  test_methods: {df_wide['test_method'].unique().tolist()}")

        output_path = FIG_DIR / f"power_typei_grid_{experiment}.png"
        create_grid_plot(df_wide, output_path, experiment_name=experiment)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
