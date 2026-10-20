
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from harness.ci_reanalysis import CI_ALPHA, NOMINAL_ALPHA, REPO_ROOT, wilson_ci

POWER_TARGET = 0.80

METHOD_COLORS = {
    "mmdagg": "#2ca02c",
    "chi_square": "#ff7f0e",
    "g_test": "#d62728",
    "multinomial_mc": "#9467bd",
    "energy_distance": "#8c564b",
    "tv_bound": "#e377c2",
    "tolerance": "#1f77b4",
    "exact_binomial": "#17becf",
    "adaptive": "#bcbd22",
    "auto": "#bcbd22",
    "bootstrap": "#7f7f7f",
    "sequential": "#393b79",
    "sprt": "#637939",
    "g_test_marginal": "#d62728",
}

_FALLBACK_COLORS = (
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
    "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd",
)

_LINESTYLES = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)))

_PNG_METADATA = {"Software": "harness.figures"}

_DPI = 300




def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.autolayout": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
        "savefig.bbox": "tight",
    })
    return plt


def method_color(method: str, known: tuple[str, ...] = ()) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    pool = sorted(set(known) - set(METHOD_COLORS)) or [method]
    return _FALLBACK_COLORS[pool.index(method) % len(_FALLBACK_COLORS)]


def _style_for(values: tuple, table: tuple) -> dict:
    ordered = sorted(values)
    return {v: table[i % len(table)] for i, v in enumerate(ordered)}


def load_campaign(pattern, *, ci_alpha: float = CI_ALPHA):
    from harness import io_store

    df = io_store.read_grid(pattern)
    n = df[["TP", "FP", "TN", "FN"]].sum(axis=1)
    k = df["TP"] + df["FP"]
    lo, hi = wilson_ci(k.to_numpy(dtype=float), n.to_numpy(dtype=float), alpha=ci_alpha)
    out = df.copy()
    out["n_cell"] = n
    out["k_reject"] = k
    out["rate"] = k / n
    out["wilson_lo"] = lo
    out["wilson_hi"] = hi
    out["metric"] = np.where(out["case"].to_numpy() == "baseline", "type_i_error", "power")
    return out


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, metadata=_PNG_METADATA)
    _plt().close(fig)
    return path


def _panel_grid(plt, n_panels: int, *, per_row: int = 3, size=(4.2, 3.2)):
    ncols = min(per_row, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat[:n_panels]


def _errorbar(ax, x, sub, *, color, linestyle="-", marker="o", label=None):
    yerr = np.vstack([
        (sub["rate"] - sub["wilson_lo"]).clip(lower=0).to_numpy(dtype=float),
        (sub["wilson_hi"] - sub["rate"]).clip(lower=0).to_numpy(dtype=float),
    ])
    ax.errorbar(x, sub["rate"].to_numpy(dtype=float), yerr=yerr, color=color,
                linestyle=linestyle, marker=marker, markersize=3.5, linewidth=1.2,
                capsize=1.5, elinewidth=0.7, label=label)


def _log_x(ax, values):
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

    ticks = sorted({float(v) for v in np.asarray(values, dtype=float) if np.isfinite(v)})
    ax.set_xscale("log")
    if ticks:
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{int(t)}" for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=7, rotation=45)


def _finish(fig, handles, labels, title: str):
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))




def power_typei_grid(df, out_dir: Path, *, per_row: int = 3) -> list[Path]:
    plt = _plt()
    out: list[Path] = []
    if df.empty:
        return out

    for (paradigm, circuit), block in df.groupby(["paradigm", "circuit"], sort=True):
        methods = tuple(sorted(block["method"].unique()))
        operators = tuple(sorted(block.loc[block["case"] == "falsecase",
                                           "operator"].unique()))
        op_style = _style_for(operators, _LINESTYLES)
        widths = sorted(block["d"].unique())
        fig, axes = _panel_grid(plt, len(widths), per_row=per_row)
        for ax, d in zip(axes, widths):
            panel = block[block["d"] == d]
            twin = ax.twinx()
            twin.grid(False)
            for method in methods:
                colour = method_color(method, methods)
                power = panel[(panel["method"] == method) & (panel["case"] == "falsecase")]
                for (operator, _), curve in power.groupby(["operator", "strength"],
                                                          dropna=False, sort=True):
                    curve = curve.sort_values("m")
                    _errorbar(ax, curve["m"].to_numpy(dtype=float), curve, color=colour,
                              linestyle=op_style.get(operator, "-"), marker="o")
                typei = panel[(panel["method"] == method) & (panel["case"] == "baseline")]
                if not typei.empty:
                    typei = typei.sort_values("m")
                    twin.plot(typei["m"].to_numpy(dtype=float),
                              typei["rate"].to_numpy(dtype=float), color=colour,
                              linestyle="-.", marker="x", markersize=3.5, linewidth=1.0)
            ax.axhline(POWER_TARGET, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            twin.axhline(NOMINAL_ALPHA, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
            _log_x(ax, panel["m"])
            ax.set_ylim(-0.03, 1.03)
            twin.set_ylim(0.0, max(0.12, float(np.nanmax(
                np.append(block.loc[block["case"] == "baseline", "rate"].to_numpy(), 0.1)))))
            ax.set_title(f"d = {int(d)}", fontsize=9)
            ax.set_xlabel("shots m")
            ax.set_ylabel("power")
            twin.set_ylabel("Type I", fontsize=8)

        handles = [plt.Line2D([], [], color=method_color(m, methods), label=m)
                   for m in methods]
        labels = list(methods)
        if len(operators) > 1:
            handles.append(plt.Line2D([], [], color="none", label=""))
            labels.append("")
            for operator in sorted(operators):
                handles.append(plt.Line2D([], [], color="black",
                                          linestyle=op_style[operator], label=operator))
                labels.append(operator)
        handles.append(plt.Line2D([], [], color="black", linestyle="-.", marker="x",
                                  label="Type I (right axis)"))
        labels.append("Type I (right axis)")
        _finish(fig, handles, labels,
                f"{paradigm} -- {circuit}: power and Type I vs shots (95% Wilson)")
        out.append(_save(fig, Path(out_dir) / f"{paradigm}_power_typei_grid_{circuit}.png"))
    return out




def frontier_curves(df, out_dir: Path, *, per_row: int = 5) -> list[Path]:
    plt = _plt()
    out: list[Path] = []
    frontier = df[(df["case"] == "falsecase") & df["strength"].notna()]
    if frontier.empty:
        return out
    for (paradigm, circuit, operator), block in frontier.groupby(
            ["paradigm", "circuit", "operator"], sort=True):
        methods = tuple(sorted(block["method"].unique()))
        panels = sorted({(int(d), int(m)) for d, m in
                         zip(block["d"], block["m"])})
        fig, axes = _panel_grid(plt, len(panels), per_row=per_row, size=(3.4, 2.8))
        for ax, (d, m) in zip(axes, panels):
            panel = block[(block["d"] == d) & (block["m"] == m)]
            for method in methods:
                curve = panel[panel["method"] == method].sort_values("strength")
                if curve.empty:
                    continue
                _errorbar(ax, curve["strength"].to_numpy(dtype=float), curve,
                          color=method_color(method, methods), marker="o")
            ax.axhline(POWER_TARGET, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axhline(NOMINAL_ALPHA, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"d = {d}, m = {m}", fontsize=9)
            ax.set_xlabel(r"fault strength $\delta$")
            ax.set_ylabel("power")
        handles = [plt.Line2D([], [], color=method_color(m, methods), label=m)
                   for m in methods]
        _finish(fig, handles, list(methods),
                f"{paradigm} -- {circuit} / {operator}: power vs fault strength")
        out.append(_save(fig, Path(out_dir)
                         / f"{paradigm}_frontier_{circuit}_{operator}.png"))
    return out




def c3_separation(df, out_dir: Path, *, per_row: int = 3) -> list[Path]:
    plt = _plt()
    out: list[Path] = []
    block = df[df["strength"].notna() & (df["case"] == "falsecase")
               & (df["method"] != "_error")].copy()
    if block.empty:
        return out
    from harness.runner import MARGINAL_METHODS

    marginal_set = set(MARGINAL_METHODS) | {"sprt"}
    block["family"] = [
        "marginal" if m in marginal_set else "joint" for m in block["method"]
    ]
    if block["family"].nunique() < 2:
        return out
    x_col = "strength" if block["strength"].nunique() > 1 else "m"
    eps_note = ("" if x_col == "strength"
                else rf" at $\varepsilon$ = {block['strength'].iloc[0]:g}")
    for circuit, panel_block in block.groupby("circuit", sort=True):
        methods = tuple(sorted(panel_block["method"].unique()))
        widths = sorted(panel_block["d"].unique())
        fig, axes = _panel_grid(plt, len(widths), per_row=per_row)
        for ax, d in zip(axes, widths):
            panel = panel_block[panel_block["d"] == d]
            for (method, family), curve in panel.groupby(["method", "family"],
                                                         sort=True):
                curve = curve.sort_values(x_col)
                _errorbar(ax, curve[x_col].to_numpy(dtype=float), curve,
                          color=method_color(method, methods),
                          linestyle="-" if family == "joint" else "--",
                          marker="o" if family == "joint" else "s")
            ax.axhline(NOMINAL_ALPHA, color="red", linestyle=":", linewidth=1.0)
            ax.set_ylim(-0.03, 1.03)
            if x_col == "m":
                ax.set_xscale("log")
            ax.set_title(f"d = {int(d)}", fontsize=9)
            ax.set_xlabel("shots m" + eps_note if x_col == "m"
                          else r"dependence-only TVD $\varepsilon$")
            ax.set_ylabel("rejection rate")
        handles = [plt.Line2D([], [], color=method_color(m, methods), label=m)
                   for m in methods]
        handles += [plt.Line2D([], [], color="black", linestyle="-", label="joint"),
                    plt.Line2D([], [], color="black", linestyle="--", label="marginal"),
                    plt.Line2D([], [], color="red", linestyle=":", label=r"$\alpha=0.05$")]
        labels = list(methods) + ["joint", "marginal", r"$\alpha=0.05$"]
        _finish(fig, handles, labels,
                f"C3 -- {circuit}: marginal-preserving fault, joint vs marginal tests")
        out.append(_save(fig, Path(out_dir) / f"c3_separation_{circuit}.png"))
    return out




def noisy_two_regime(df, out_dir: Path, *, per_row: int = 3) -> list[Path]:
    plt = _plt()
    out: list[Path] = []
    if "arm" not in df.columns or "noise_label" not in df.columns:
        return out
    block = df[df["arm"].notna() & (df["arm"] != "")]
    if block.empty:
        return out
    arm_style = {"A": "-", "B": "--", "P": ":"}

    for noise_label, panel_block in block.groupby("noise_label", sort=True):
        methods = tuple(sorted(panel_block["method"].unique()))
        arms = sorted(panel_block["arm"].unique())
        fig, axes = _panel_grid(plt, len(arms), per_row=per_row)
        for ax, arm in zip(axes, arms):
            panel = panel_block[panel_block["arm"] == arm]
            for method in methods:
                curve = panel[panel["method"] == method].sort_values("m")
                if curve.empty:
                    continue
                _errorbar(ax, curve["m"].to_numpy(dtype=float), curve,
                          color=method_color(method, methods),
                          linestyle=arm_style.get(arm, "-"), marker="o")
            is_power = arm == "P"
            ax.axhline(POWER_TARGET if is_power else NOMINAL_ALPHA, color="red",
                       linestyle="--" if is_power else ":", linewidth=1.0)
            _log_x(ax, panel["m"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(
                {"A": "Regime A: ideal null (rejections are false alarms)",
                 "B": "Regime B: noise-aware two-sample null",
                 "P": "Power arm: structural bug under the same noise"}.get(
                     arm, f"arm {arm}"), fontsize=8)
            ax.set_xlabel("shots m")
            ax.set_ylabel("power" if is_power else "Type I error")
        handles = [plt.Line2D([], [], color=method_color(m, methods), label=m)
                   for m in methods]
        _finish(fig, handles, list(methods),
                f"noisy d=5 Grover, noise = {noise_label}: two regimes (95% Wilson)")
        out.append(_save(fig, Path(out_dir) / f"noisy_two_regime_{noise_label}.png"))
    return out




def en_comparison(df, out_dir: Path, *, per_row: int = 3) -> list[Path]:
    plt = _plt()
    out: list[Path] = []
    if "n_used_mean" not in df.columns:
        return out
    block = df[df["n_used_mean"].notna()]
    if block.empty:
        return out
    for (paradigm, case), panel_block in block.groupby(["paradigm", "case"], sort=True):
        methods = tuple(sorted(panel_block["method"].unique()))
        circuits = sorted(panel_block["circuit"].unique())
        fig, axes = _panel_grid(plt, len(circuits), per_row=per_row)
        for ax, circuit in zip(axes, circuits):
            panel = panel_block[panel_block["circuit"] == circuit]
            for method in methods:
                curve = panel[panel["method"] == method].sort_values("m")
                if curve.empty:
                    continue
                ax.plot(curve["m"].to_numpy(dtype=float),
                        curve["n_used_mean"].to_numpy(dtype=float),
                        color=method_color(method, methods), marker="o", markersize=3.5,
                        linewidth=1.2)
            ms = np.array(sorted(panel["m"].unique()), dtype=float)
            ax.plot(ms, ms, color="black", linestyle="--", linewidth=0.9)
            _log_x(ax, panel["m"])
            ax.set_yscale("log")
            ax.set_title(f"{circuit}", fontsize=9)
            ax.set_xlabel("budget m")
            ax.set_ylabel("E[N] (shots used)")
        handles = [plt.Line2D([], [], color=method_color(m, methods), label=m)
                   for m in methods]
        handles.append(plt.Line2D([], [], color="black", linestyle="--", label="fixed-m cost"))
        _finish(fig, handles, list(methods) + ["fixed-m cost"],
                f"{paradigm} ({case}): expected sample size vs fixed-sample budget")
        out.append(_save(fig, Path(out_dir) / f"en_comparison_{paradigm}_{case}.png"))
    return out



FAMILIES = {
    "power_typei": power_typei_grid,
    "frontier": frontier_curves,
    "c3": c3_separation,
    "noisy": noisy_two_regime,
    "en": en_comparison,
}


def render(df, out_dir, families=tuple(FAMILIES)) -> dict[str, list[Path]]:
    out_dir = Path(out_dir)
    written: dict[str, list[Path]] = {}
    for name in families:
        try:
            fn = FAMILIES[name]
        except KeyError:
            raise ValueError(f"unknown figure family {name!r}; have {sorted(FAMILIES)}") from None
        written[name] = fn(df, out_dir)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Figures of the revised campaign, rendered from the campaign Parquet files.')
    parser.add_argument("--campaign", required=True,
                        help="parquet glob / directory of campaign results")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "revision" / "analysis_out" / "figures")
    parser.add_argument("--families", default="all",
                        help=f"comma-separated subset of {sorted(FAMILIES)}, or 'all'")
    args = parser.parse_args(argv)

    families = tuple(FAMILIES) if args.families == "all" else tuple(
        f.strip() for f in args.families.split(",") if f.strip())
    df = load_campaign(args.campaign)
    print(f"grid: {len(df):,d} rows, {df['campaign'].nunique()} campaign(s), "
          f"{df['circuit'].nunique()} circuit(s)")
    written = render(df, args.out_dir, families)
    total = 0
    for name in families:
        paths = written.get(name, [])
        total += len(paths)
        if paths:
            print(f"  {name:<12} {len(paths):>3d} figure(s)")
        else:
            print(f"  {name:<12}   0 figures (grid has no rows/columns for this family)")
    print(f"{total} figure(s) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
