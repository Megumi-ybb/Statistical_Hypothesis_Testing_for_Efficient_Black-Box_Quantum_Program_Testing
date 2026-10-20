
from __future__ import annotations

import datetime as _dt
import glob as _glob
import os
import subprocess
import sys

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

HARNESS_VERSION = "0.1.0"

COUNT_COLUMNS: tuple[str, ...] = ("TP", "FP", "TN", "FN")

CELL_COLUMNS: tuple[str, ...] = (
    "campaign",
    "circuit",
    "d",
    "m",
    "paradigm",
    "operator",
    "strength",
    "case",
    "methods",
    "n_reps",
)
RESULT_COLUMNS: tuple[str, ...] = (
    "method",
    *COUNT_COLUMNS,
    "n_evaluated",
    "mean_p",
    "n_used_mean",
    "n_used_median",
    "method_wall_s",
    "cell_wall_s",
)
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "seed_master",
    "harness_version",
    "git_describe",
    "python_version",
    "numpy_version",
    "scipy_version",
    "pandas_version",
    "pyarrow_version",
    "joblib_version",
    "qiskit_version",
    "qiskit_aer_version",
    "created_utc",
)
COLUMNS: tuple[str, ...] = CELL_COLUMNS + RESULT_COLUMNS + PROVENANCE_COLUMNS

_DTYPES = {
    "d": "int64",
    "m": "int64",
    "n_reps": "int64",
    "TP": "int64",
    "FP": "int64",
    "TN": "int64",
    "FN": "int64",
    "n_evaluated": "int64",
    "strength": "float64",
    "mean_p": "float64",
    "n_used_mean": "float64",
    "n_used_median": "float64",
    "method_wall_s": "float64",
    "cell_wall_s": "float64",
    "seed_master": "uint64",
}

_PACKAGES = {
    "numpy_version": "numpy",
    "scipy_version": "scipy",
    "pandas_version": "pandas",
    "pyarrow_version": "pyarrow",
    "joblib_version": "joblib",
    "qiskit_version": "qiskit",
    "qiskit_aer_version": "qiskit-aer",
}


@lru_cache(maxsize=1)
def git_describe() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def provenance() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {
        "harness_version": HARNESS_VERSION,
        "git_describe": git_describe(),
        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
    }
    for column, package in _PACKAGES.items():
        try:
            out[column] = version(package)
        except PackageNotFoundError:
            out[column] = ""
    return out


def rows_from_cell_result(cell_result) -> list[dict]:
    cell = cell_result.cell
    prov = provenance()
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for method, res in cell_result.methods.items():
        rows.append(
            {
                "campaign": cell.campaign,
                "circuit": cell.circuit,
                "d": int(cell.d),
                "m": int(cell.m),
                "paradigm": cell.paradigm,
                "operator": cell.operator,
                "strength": float("nan") if cell.strength is None else float(cell.strength),
                "case": cell.case,
                "methods": ",".join(cell.methods),
                "n_reps": int(cell.n_reps),
                "method": method,
                "TP": int(res.TP),
                "FP": int(res.FP),
                "TN": int(res.TN),
                "FN": int(res.FN),
                "n_evaluated": int(res.n_evaluated),
                "mean_p": float(res.mean_p),
                "error": getattr(res, "error", None) or "",
                "n_used_mean": float("nan") if res.n_used_mean is None else float(res.n_used_mean),
                "n_used_median": (
                    float("nan") if res.n_used_median is None else float(res.n_used_median)
                ),
                "method_wall_s": float(res.wall_s),
                "cell_wall_s": float(cell_result.wall_s),
                "seed_master": int(cell_result.seed_master),
                "created_utc": stamp,
                **prov,
            }
        )
    return rows


def to_frame(cell_results: Iterable):
    import pandas as pd

    rows: list[dict] = []
    for item in cell_results:
        if isinstance(item, dict):
            rows.append(item)
        else:
            rows.extend(rows_from_cell_result(item))
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    missing = [c for c in COUNT_COLUMNS if c not in df.columns or df[c].isna().any()]
    if missing:
        raise ValueError(f"per-cell confusion counts missing/NA: {missing}")
    if len(df):
        df = df.astype(_DTYPES)
    return df


def write_job(path: str | Path, cell_results: Iterable) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = to_frame(cell_results)
    if df.empty:
        raise ValueError(f"refusing to write an empty job file: {path}")
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    return path


def read_grid(pattern: str | Path | Sequence[str | Path]):
    import pandas as pd

    if isinstance(pattern, (str, Path)):
        p = Path(pattern)
        if p.is_dir():
            paths = sorted(p.glob("*.parquet"))
        else:
            paths = [Path(x) for x in sorted(_glob.glob(str(pattern)))]
    else:
        paths = [Path(x) for x in pattern]
    if not paths:
        raise FileNotFoundError(f"no parquet files matched {pattern!r}")

    frames = [pd.read_parquet(p, engine="pyarrow") for p in paths]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in COUNT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"grid is missing confusion-count columns {missing}")
    return df
