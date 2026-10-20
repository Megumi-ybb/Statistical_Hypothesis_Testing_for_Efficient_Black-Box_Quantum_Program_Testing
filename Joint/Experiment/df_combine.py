from pathlib import Path
import argparse
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WIDE_NAME = "combined_results.csv"
DEFAULT_LONG_NAME = "combined_results_long.csv"
RESULT_DIR_PREFIX = "joint_experiment_"
RESULT_DIR_SUFFIX = "_results"


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


def _strip_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def _strip_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text


def parse_experiment_case(dir_path: Path) -> tuple[str, str]:
    name = dir_path.name
    name = _strip_prefix(name, RESULT_DIR_PREFIX)
    name = _strip_suffix(name, RESULT_DIR_SUFFIX)

    tokens = name.split("_")
    case = "baseline"

    if tokens and tokens[-1].lower() == "falsecase":
        case = "falsecase"
        tokens = tokens[:-1]
    elif tokens and tokens[0].lower() == "error":
        case = "error"
        tokens = tokens[1:]

    experiment = "_".join(tokens) if tokens else name
    return experiment, case


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num.astype(float) / den.astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required = {"TP", "TN", "FP", "FN"}
    if not required.issubset(df.columns):
        return df

    df = df.copy()
    for col in required:
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

    df["type_i_error"] = df.get("type_i_error", pd.Series(dtype=float)).fillna(type_i)
    if "type_ii_error" in df.columns:
        df["type_ii_error"] = df["type_ii_error"].fillna(type_ii)
    else:
        df["type_ii_error"] = type_ii
    if "power" in df.columns:
        df["power"] = df["power"].fillna(power)
    else:
        df["power"] = power
    if "accuracy" in df.columns:
        df["accuracy"] = df["accuracy"].fillna(accuracy)
    else:
        df["accuracy"] = accuracy
    if "precision" in df.columns:
        df["precision"] = df["precision"].fillna(precision)
    else:
        df["precision"] = precision
    if "f1_score" in df.columns:
        df["f1_score"] = df["f1_score"].fillna(f1)
    else:
        df["f1_score"] = f1

    if "recall" not in df.columns:
        df["recall"] = recall
    else:
        df["recall"] = df["recall"].fillna(recall)

    return df


def collect_frames() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    result_dirs = [
        d
        for d in BASE_DIR.iterdir()
        if d.is_dir()
        and d.name.startswith(RESULT_DIR_PREFIX)
        and d.name.endswith(RESULT_DIR_SUFFIX)
    ]

    for result_dir in sorted(result_dirs):
        experiment, case = parse_experiment_case(result_dir)
        for csv_file in sorted(result_dir.glob("*.csv")):
            df = pd.read_csv(csv_file)
            df = add_derived_metrics(df)
            df["experiment"] = experiment
            df["case"] = case
            df["source_dir"] = result_dir.name
            df["source_file"] = csv_file.name
            frames.append(df)
    return frames


def build_long_form(df: pd.DataFrame) -> pd.DataFrame:
    id_columns = [col for col in ID_COLUMNS if col in df.columns]
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
    parser = argparse.ArgumentParser(description="Concatenate joint experiment result CSV files.")
    parser.add_argument("--output-wide", type=Path, default=None, help=f"Path for the wide combined CSV (default: {DEFAULT_WIDE_NAME}).")
    parser.add_argument("--output-long", type=Path, default=None, help=f"Path for the long (melted) CSV (default: {DEFAULT_LONG_NAME}).")
    args = parser.parse_args()

    frames = collect_frames()
    if not frames:
        raise SystemExit("No result CSV files found.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    wide_path = resolve_output(args.output_wide, DEFAULT_WIDE_NAME)
    long_path = resolve_output(args.output_long, DEFAULT_LONG_NAME)

    combined.to_csv(wide_path, index=False)
    long_df = build_long_form(combined)
    long_df.to_csv(long_path, index=False)

    print(f"Wrote wide table to: {wide_path}")
    print(f"Wrote long table to: {long_path}")
    print(f"Rows: wide={len(combined)}, long={len(long_df)}")


if __name__ == "__main__":
    main()
