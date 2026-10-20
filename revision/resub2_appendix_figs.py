from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "analysis_out" / "resub2" / "appendix"
RESUB = ROOT / "Quantum_testing_access_resub" / "figures"
TEXTWIDTH_IN = 7.16

SPECS = {
    "marginal": dict(csv=ROOT / "Marginal" / "combined_results_long.csv", ds=range(2, 13), rows=4, cols=3,
                     methods={"auto": "Adaptive exact/chi-square", "bootstrap": "Bootstrap",
                              "sequential": "Sequential", "tolerance": "Tolerance"},
                     groups=["add", "BV", "QFT", "QPE", "QWALK", "grovertest"]),
    "joint": dict(csv=ROOT / "Joint" / "combined_results_long.csv", ds=range(2, 11), rows=3, cols=3,
                  methods={"chi_square": "Chi-square, lumped", "mmdagg": "MMDAgg", "tolerance": "Tolerance"},
                  groups=["add", "BV", "QFT", "QPE", "QWALK", "GROVER"]),
}
COLORS = {"auto": "#d62728", "bootstrap": "#ff7f0e", "sequential": "#2ca02c", "tolerance": "#1f77b4",
          "chi_square": "#ff7f0e", "mmdagg": "#2ca02c"}
METRICS = {"power_bitflip": ("-", "o", "Power, bit-flip"), "power_asymmetric": ("--", "s", "Power, asymmetric"),
           "power_depolarize": (":", "^", "Power, depolarizing"), "type_i_error": ("-.", "x", "Type I error")}
M_TICKS = [50, 100, 200, 500, 1000, 2000]


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[df.metric.isin(METRICS)]
    return df.pivot_table(index=["experiment", "n_qubits", "m_samples", "test_method"],
                          columns="metric", values="value", aggfunc="mean").reset_index()


def grid(df: pd.DataFrame, spec: dict, out: Path) -> int:
    plt.rcParams.update({k: 8 for k in ("font.size", "axes.titlesize", "axes.labelsize",
                                        "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    rows, cols, ds = spec["rows"], spec["cols"], list(spec["ds"])
    fig, axes = plt.subplots(rows, cols, figsize=(TEXTWIDTH_IN, 1.75 * rows + 1.0),
                             sharex=True, sharey=True, dpi=300)
    for ax, d in zip(axes.flat, ds):
        sub = df[df.n_qubits == d]
        for meth in spec["methods"]:
            g = sub[sub.test_method == meth].sort_values("m_samples")
            for metric, (ls, mk, _) in METRICS.items():
                if metric in g and not g.empty:
                    ax.plot(g.m_samples, g[metric], color=COLORS[meth], ls=ls, marker=mk,
                            ms=2.5, lw=0.9, alpha=0.85)
        for y in (0.05, 0.95):
            ax.axhline(y, color="red", ls="--", lw=0.6, alpha=0.6)
        ax.set_title(f"$d = {d}$", pad=2)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xticks(M_TICKS)
        ax.set_xticklabels([str(x) for x in M_TICKS])
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        ax.set_yticklabels(["0", ".05", ".25", ".5", ".75", ".95", "1"])
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)
        ax.tick_params(length=2, pad=1)
    for ax in list(axes.flat)[len(ds):]:
        ax.set_visible(False)
    for c in range(cols):
        for r in range(rows - 1, -1, -1):
            if axes[r, c].get_visible():
                axes[r, c].tick_params(labelbottom=True)
                axes[r, c].set_xlabel("Measurements $m$")
                break
    for ax in axes[:, 0]:
        ax.set_ylabel("Rejection rate")
    handles = [mlines.Line2D([], [], color=COLORS[m], lw=2, label=lab) for m, lab in spec["methods"].items()]
    handles += [mlines.Line2D([], [], color="black", ls=ls, marker=mk, ms=3, lw=1, label=lab)
                for ls, mk, lab in METRICS.values()]
    handles += [mlines.Line2D([], [], color="red", ls="--", lw=1, label=r"$\alpha = 0.05$ and power $0.95$")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, handlelength=2.4,
               columnspacing=1.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.75 / (1.75 * rows + 1.0), 1, 1], h_pad=0.6, w_pad=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, metadata={"Software": "resub2_appendix_figs"})
    plt.close(fig)
    print("wrote", out)
    return sum(1 for d in ds if not df[df.n_qubits == d].empty)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    args = ap.parse_args()
    for paradigm, spec in SPECS.items():
        df = load(spec["csv"])
        for grp in spec["groups"]:
            name = f"{paradigm}_power_typei_grid_{grp}.png"
            n = grid(df[df.experiment == grp], spec, OUT / name)
            assert n == len(list(spec["ds"])), f"{name}: {n} panels drawn, expected {len(list(spec['ds']))}"
            if args.install:
                shutil.copy(OUT / name, RESUB / name)
    if args.install:
        print("installed into", RESUB)


if __name__ == "__main__":
    main()
