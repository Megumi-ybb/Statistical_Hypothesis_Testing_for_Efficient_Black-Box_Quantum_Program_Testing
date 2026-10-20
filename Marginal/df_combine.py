
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WIDE_NAME = "combined_results.csv"
DEFAULT_LONG_NAME = "combined_results_long.csv"

ID_COLUMNS = {
    "experiment",
    "case",
    "source_dir",
    "source_file",
    "n_qubits",
    "m_samples",
    "alpha",
    "test_method",
    "measurement_reduction_ratio",
}

CONFUSION_COLS = {"TP", "TN", "FP", "FN"}

EXCLUDE_FROM_OUTPUT = {"TP", "TN", "FP", "FN", "precision"}


def parse_experiment_case(dir_path: Path) -> tuple[str, str]:
    name = dir_path.name

    if "flase_" in name or "false_" in name:
        case = "falsecase"
        if "flase_" in name:
            start_idx = name.find("flase_") + 6
        else:
            start_idx = name.find("false_") + 6
        end_idx = name.find("_nested_parallel_results")
        if end_idx > start_idx:
            experiment = name[start_idx:end_idx]
        else:
            experiment = name[start_idx:]
    else:
        case = "baseline"

        base_pattern = "marginal_experiment_nested_parallel_results"
        if name == base_pattern:
            experiment = "grovertest"
        elif name.startswith(base_pattern + "_"):
            experiment = name[len(base_pattern) + 1:]
        else:
            experiment = "grovertest"

    return experiment, case


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num.astype(float) / den.astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if not CONFUSION_COLS.issubset(df.columns):
        return df

    df = df.copy()
    for col in CONFUSION_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    tp, tn, fp, fn = df["TP"], df["TN"], df["FP"], df["FN"]
    total = tp + tn + fp + fn

    type_i = safe_divide(fp, fp + tn)
    type_ii = safe_divide(fn, tp + fn)
    recall = safe_divide(tp, tp + fn)
    precision = safe_divide(tp, tp + fp)
    accuracy = safe_divide(tp + tn, total)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    power = 1.0 - type_ii

    for name, series in [
        ("type_i_error", type_i),
        ("type_ii_error", type_ii),
        ("power", power),
        ("accuracy", accuracy),
        ("precision", precision),
        ("f1_score", f1),
        ("recall", recall),
    ]:
        if name in df.columns:
            df[name] = df[name].fillna(series)
        else:
            df[name] = series

    return df


def collect_frames() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    experiment_dir = BASE_DIR / "Experiment"

    if not experiment_dir.exists():
        print(f"Warning: Experiment directory not found: {experiment_dir}")
        return frames

    result_dirs = []
    for d in experiment_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("marginal_experiment_flase_") and name.endswith("_nested_parallel_results"):
            result_dirs.append(d)
        elif name.startswith("marginal_experiment_nested_parallel_results"):
            result_dirs.append(d)

    print(f"Found {len(result_dirs)} result directories:")
    for d in sorted(result_dirs):
        print(f"  - {d.name}")

    for result_dir in sorted(result_dirs):
        experiment, case = parse_experiment_case(result_dir)
        csv_files = list(result_dir.glob("*.csv"))
        print(f"\nProcessing: {result_dir.name}")
        print(f"  Experiment: {experiment}, Case: {case}")
        print(f"  CSV files: {len(csv_files)}")

        for csv_file in sorted(csv_files):
            df = pd.read_csv(csv_file)
            print(f"    - {csv_file.name}: {len(df)} rows, columns: {list(df.columns)}")
            df = add_derived_metrics(df)
            df["experiment"] = experiment
            df["case"] = case
            df["source_dir"] = result_dir.name
            df["source_file"] = csv_file.name
            frames.append(df)

    return frames


def build_long_form(df: pd.DataFrame) -> pd.DataFrame:
    id_columns = [c for c in ID_COLUMNS if c in df.columns]
    metric_columns = [c for c in df.columns if c not in id_columns]
    long_df = df.melt(id_vars=id_columns, value_vars=metric_columns, var_name="metric", value_name="value")
    return long_df.dropna(subset=["value"])


def resolve_output(path_arg: Path | None, default_name: str) -> Path:
    if path_arg is None:
        return BASE_DIR / default_name
    if path_arg.is_absolute():
        return path_arg
    return BASE_DIR / path_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate marginal experiment result CSV files.")
    parser.add_argument("--output-wide", type=Path, default=None, help=f"Path for the wide combined CSV (default: {DEFAULT_WIDE_NAME}).")
    parser.add_argument("--output-long", type=Path, default=None, help=f"Path for the long (melted) CSV (default: {DEFAULT_LONG_NAME}).")
    args = parser.parse_args()

    frames = collect_frames()
    if not frames:
        raise SystemExit("No result CSV files found.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    cols_to_drop = [c for c in EXCLUDE_FROM_OUTPUT if c in combined.columns]
    combined = combined.drop(columns=cols_to_drop)

    wide_path = resolve_output(args.output_wide, DEFAULT_WIDE_NAME)
    long_path = resolve_output(args.output_long, DEFAULT_LONG_NAME)

    combined.to_csv(wide_path, index=False)
    long_df = build_long_form(combined)
    long_df.to_csv(long_path, index=False)

    print(f"\n{'='*60}")
    print(f"Wrote wide table to: {wide_path}")
    print(f"Wrote long table to: {long_path}")
    print(f"Rows: wide={len(combined)}, long={len(long_df)}")

    print(f"\nSummary by experiment and case:")
    summary = combined.groupby(["experiment", "case"]).size().reset_index(name="rows")
    print(summary.to_string(index=False))

    metric_cols = ["power_bitflip", "power_asymmetric", "power_depolarize", "type_i_error"]
    print(f"\nMetrics coverage:")
    for col in metric_cols:
        if col in combined.columns:
            non_null = combined[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")
        else:
            print(f"  {col}: NOT PRESENT")


if __name__ == "__main__":
    main()
