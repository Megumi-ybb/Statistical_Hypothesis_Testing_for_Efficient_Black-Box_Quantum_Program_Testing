from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "analysis_out" / "resub2"
FIG = HERE / "figures" / "en_by_width.png"
FRONTIER_FIG = HERE / "figures" / "frontier_QWALK_O1.png"
RESUB = HERE.parent / "Quantum_testing_access_final"

OPS = {"O1": "O1 gate-angle perturbation (0.1 rad)", "O2": "O2 gate deletion", "O3": "O3 gate insertion",
       "O4": "O4 control--target swap", "O5": "O5 phase error", "O6": "O6 wrong oracle encoding",
       "O7": "O7 routing defect"}
ORDER = ["O2", "O4", "O7", "O3", "O1", "O5", "O6"]
LAB = {"chi_square": "Chi-square, lumped", "g_test": "G-test, lumped", "multinomial_mc": "MC multinomial",
       "energy_distance": "Energy distance", "mmdagg": "MMDAgg", "tv_bound": "TV bound",
       "tolerance": "Tolerance", "adaptive": "Adaptive", "exact_binomial": "Exact binomial",
       "g_test_marginal": "Per-qubit G-test", "bootstrap": "Bootstrap (QFT)", "sequential": "Sequential, adaptive (QFT)",
       "sequential_exact": "Sequential, exact (QFT)"}
VACUOUS_METHODS = ("chi_square", "g_test")


def load(campaign: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RESULTS / campaign / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {RESULTS / campaign}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["n"] = df[["TP", "FP", "TN", "FN"]].sum(axis=1)
    df["rate"] = (df["TP"] + df["FP"]) / df["n"]
    return df


def f2(x): return "--" if pd.isna(x) else f"{x:.2f}"
def f3(x): return "--" if pd.isna(x) else f"{x:.3f}"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)
    print("wrote", OUT / name)


def struct_tables(e1: pd.DataFrame, e3: pd.DataFrame) -> None:
    ms = [100, 300, 1000]
    j = e1[e1.paradigm == "joint"]
    napp = j.groupby("operator").apply(lambda x: x[["circuit", "d"]].drop_duplicates().shape[0])
    null = e3.groupby("operator").rate.apply(lambda s: (s <= 0.08).mean())
    rows = []
    for op in ORDER:
        cells = [OPS[op], str(napp.get(op, 0)), f2(null.get(op, np.nan))]
        for mk in ("chi_square", "multinomial_mc", "mmdagg"):
            for m in ms:
                cells.append(f2(j[(j.operator == op) & (j.method == mk) & (j.m == m)].rate.median()))
        rows.append(" & ".join(cells) + r" \\")
    write("tab_struct_joint.tex", r"""\begin{table*}[ht]
\centering
\caption{\rev{Structural fault study, joint paradigm: median power by mutation operator across the six benchmarks and $d = 2$ to $10$. ``Cells'' is the number of applicable (circuit, $d$) pairs. ``Null-rate'' is the fraction of those pairs whose rejection rate at $m = 2{,}000$ does not exceed $0.08$, the operational sign of an equivalent mutant.}}
\label{tab:struct_joint}
\small
\begin{tabular}{l r r ccc ccc ccc}
\toprule
& & & \multicolumn{3}{c}{Chi-square} & \multicolumn{3}{c}{MC multinomial} & \multicolumn{3}{c}{MMDAgg} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}
Operator & Cells & Null-rate & $m{=}100$ & $300$ & $1000$ & $m{=}100$ & $300$ & $1000$ & $m{=}100$ & $300$ & $1000$ \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    mg = e1[e1.paradigm == "marginal"]
    nappm = mg.groupby("operator").apply(lambda x: x[["circuit", "d"]].drop_duplicates().shape[0])
    rows = []
    for op in ORDER:
        cells = [OPS[op], str(nappm.get(op, 0))]
        for mk in ("adaptive", "exact_binomial", "g_test_marginal"):
            for m in ms:
                cells.append(f2(mg[(mg.operator == op) & (mg.method == mk) & (mg.m == m)].rate.median()))
        rows.append(" & ".join(cells) + r" \\")
    write("tab_struct_marginal.tex", r"""\begin{table*}[ht]
\centering
\caption{\rev{Structural fault study, marginal paradigm: median power by mutation operator across the six benchmarks and $d = 2$ to $12$. The bootstrap and sequential methods apply only to QFT, whose marginals are all non-degenerate, and are reported in the text.}}
\label{tab:struct_marginal}
\small
\begin{tabular}{l r ccc ccc ccc}
\toprule
& & \multicolumn{3}{c}{Adaptive} & \multicolumn{3}{c}{Exact binomial} & \multicolumn{3}{c}{Per-qubit G-test} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
Operator & Cells & $m{=}100$ & $300$ & $1000$ & $m{=}100$ & $300$ & $1000$ & $m{=}100$ & $300$ & $1000$ \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    q = mg[(mg.circuit == "QFT") & mg.method.isin(["sequential", "sequential_exact"]) & mg.m.isin(ms)]
    if "sequential_exact" in set(q.method):
        print("QFT marginal, sequential adaptive vs exact branch, median power by operator:")
        print(q.pivot_table(index="operator", columns=["method", "m"], values="rate", aggfunc="median").round(2).to_string())

    det, inv = ["O2", "O4", "O7"], ["O1", "O5", "O6"]
    for mk in ("chi_square", "mmdagg", "multinomial_mc"):
        a = j[j.method == mk]
        print(f"{mk}: mean all={a.rate.mean():.3f} detectable={a[a.operator.isin(det)].rate.mean():.3f} "
              f"invisible={a[a.operator.isin(inv)].rate.mean():.3f} O3={a[a.operator == 'O3'].rate.mean():.3f}")


def baseline_table(e1b: pd.DataFrame) -> None:
    rows = []
    for par, order in (("joint", ["chi_square", "g_test", "multinomial_mc", "energy_distance", "mmdagg", "tv_bound", "tolerance"]),
                       ("marginal", ["adaptive", "exact_binomial", "g_test_marginal", "bootstrap", "sequential", "sequential_exact"])):
        rows.append(r"\multicolumn{6}{l}{\emph{%s paradigm}} \\" % par.capitalize())
        for mk in order:
            a = e1b[(e1b.paradigm == par) & (e1b.method == mk)]
            if a.empty:
                continue
            vac = (a.mean_p >= 0.999999) if mk in VACUOUS_METHODS else pd.Series(False, index=a.index)
            b = a[~vac]
            rows.append(" & ".join([LAB[mk], str(len(a)), f"{vac.mean():.2f}" if mk in VACUOUS_METHODS else "--",
                                    f3(b.rate.median()), f"[{b.rate.quantile(.25):.3f}, {b.rate.quantile(.75):.3f}]",
                                    f3(b.rate.max())]) + r" \\")
    write("tab_baseline_typei.tex", r"""\begin{table}[ht]
\centering
\caption{\rev{Type~I error of the proposed tests and the classical baselines on the six named benchmarks under the null (campaign I4E1B, $1{,}000$ replications per cell, $\alpha = 0.05$). ``Vacuous'' is the fraction of cells in which pooling leaves a single cell and the lumped tests return no decision. The statistics in the last three columns exclude those cells. Brackets are interquartile ranges across cells. The bootstrap and sequential procedures run on QFT only.}}
\label{tab:baseline_typei}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}l r c c c c@{}}
\toprule
Test & Cells & Vacuous & Median & IQR & Max \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    c = e1b[e1b.method == "chi_square"]
    vac = c.mean_p >= 0.999999
    print("vacuous chi-square null cells:", f"{vac.mean():.1%}", c.assign(v=vac).groupby("circuit").v.mean().round(2).to_dict())


def en_table_and_figure(s: pd.DataFrame) -> None:
    ap = pd.read_csv(HERE / "analysis_out" / "a_priori_sizes.csv")
    bon = ap[ap.design == "Bonferroni marginal"].copy()
    bon["d"] = bon.cell.str.extract(r"d=(\d+)").astype(int)
    bon = bon.set_index("d").normal
    seqs = [mk for mk in ("sequential", "sequential_exact") if mk in set(s.method)]
    piv = s.pivot_table(index="d", columns=["method", "case"], values="n_used_mean")
    pw = s[s.case == "falsecase"].pivot_table(index="d", columns="method", values="rate")
    t1 = s[s.case == "baseline"].pivot_table(index="d", columns="method", values="rate")
    horizon = int(s.m.max())
    rows = []
    for d in sorted(piv.index):
        cells = [str(d)]
        for mk in seqs + ["sprt"]:
            cells += [f"{piv.loc[d, (mk, 'falsecase')]:,.0f}", f"{piv.loc[d, (mk, 'baseline')]:,.0f}", f2(pw.loc[d, mk])]
        cells.append(f"{bon[d]:,.0f}")
        rows.append(" & ".join(cells) + r" \\")
    blocks = {"sequential": "Sequential, adaptive branch", "sequential_exact": "Sequential, exact branch", "sprt": "SPRT"}
    head = " & ".join([""] + [r"\multicolumn{3}{c}{%s}" % blocks[mk] for mk in seqs + ["sprt"]] + ["Fixed"]) + r" \\"
    mids = " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(seqs) + 1))
    sub = " & ".join(["$d$"] + ["Shift & Null & Power"] * (len(seqs) + 1) + ["a priori"]) + r" \\"
    env = "table*" if len(seqs) == 2 else "table"
    hz = f"{horizon:,}".replace(",", "{,}")
    write("tab_en_width.tex", r"""\begin{%s}[ht]
\centering
\caption{\rev{Expected number of measurements on QFT under the matched alternative $|\Delta p| = 0.02$ (shift) and under the null, for the sequential design with horizon $%s$ on its adaptive branch and on its exact branch, and for the per-qubit two-sided SPRT, together with the rejection rate under the shift and the a priori fixed-sample requirement of the Bonferroni-corrected exact test at power $0.80$. The sequential design stops at the horizon when it does not reject; the SPRT runs to its own boundary and may exceed it. $1{,}000$ replications per cell.}}
\label{tab:en_width}
\small
\begin{tabular}{r %s r}
\toprule
%s
%s
%s
\midrule
""" % (env, hz, " ".join(["rrr"] * (len(seqs) + 1)), head, mids, sub) + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{%s}\n" % env)
    for mk in seqs + ["sprt"]:
        sav = {d: round(1 - piv.loc[d, (mk, "falsecase")] / bon[d], 3) for d in sorted(piv.index)}
        print(f"{mk}: saving under the shift by d: {sav}")
        print(f"{mk}: power under the shift by d: {pw[mk].round(3).to_dict()}")
        print(f"{mk}: Type I by d: {t1[mk].round(3).to_dict()}  E[N] shift: {piv[(mk, 'falsecase')].round(0).to_dict()}  null: {piv[(mk, 'baseline')].round(0).to_dict()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = np.array(sorted(piv.index))
    fig, ax = plt.subplots(figsize=(3.6, 3.5), dpi=300)
    ax.plot(ds, [piv.loc[d, ("sequential", "falsecase")] for d in ds], "-o", ms=3, color="#393b79", label="Sequential, adaptive branch, shift")
    if "sequential_exact" in seqs:
        ax.plot(ds, [piv.loc[d, ("sequential_exact", "falsecase")] for d in ds], "-D", ms=3, color="#ad494a", label="Sequential, exact branch, shift")
    ax.plot(ds, [piv.loc[d, ("sequential", "baseline")] for d in ds], "--o", ms=3, color="#393b79", alpha=.55, label=f"Sequential, null (horizon {horizon:,})")
    ax.plot(ds, [piv.loc[d, ("sprt", "falsecase")] for d in ds], "-s", ms=3, color="#637939", label="SPRT, shift")
    ax.plot(ds, [piv.loc[d, ("sprt", "baseline")] for d in ds], "--s", ms=3, color="#637939", alpha=.55, label="SPRT, null")
    ax.plot(ds, [bon[d] for d in ds], "-", color="black", lw=1.2, label="Fixed sample, power 0.80")
    ax.set_xlabel("Number of qubits $d$"); ax.set_ylabel("Expected measurements"); ax.set_xticks(ds)
    ax.set_ylim(0, max(10000, float(piv.max().max()) * 1.05)); ax.grid(alpha=.25, lw=.5)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handlelength=2.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, metadata={"Software": "resub2_tables"}, bbox_inches="tight")
    fig.savefig(OUT / "en_by_width.png", metadata={"Software": "resub2_tables"}, bbox_inches="tight")
    print("wrote", FIG)
    graphical_abstract(piv, bon, horizon, seqs)


def graphical_abstract(piv: pd.DataFrame, bon: pd.Series, horizon: int, seqs: list[str]) -> None:
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from PIL import Image
    ds = np.array(sorted(piv.index))
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 9})
    fig, ax = plt.subplots(figsize=(6.6, 2.95))
    kw = {"lw": 2, "ms": 4.5}
    ax.plot(ds, [piv.loc[d, ("sequential", "falsecase")] for d in ds], "-o", color="#393b79", label="Sequential, adaptive branch, shift", **kw)
    if "sequential_exact" in seqs:
        ax.plot(ds, [piv.loc[d, ("sequential_exact", "falsecase")] for d in ds], "-D", color="#ad494a", label="Sequential, exact branch, shift", **kw)
    ax.plot(ds, [piv.loc[d, ("sequential", "baseline")] for d in ds], "--o", color="#393b79", alpha=.55, label=f"Sequential, null (horizon {horizon:,})", **kw)
    ax.plot(ds, [piv.loc[d, ("sprt", "falsecase")] for d in ds], "-s", color="#637939", label="SPRT, shift", **kw)
    ax.plot(ds, [piv.loc[d, ("sprt", "baseline")] for d in ds], "--s", color="#637939", alpha=.55, label="SPRT, null", **kw)
    ax.plot(ds, [bon[d] for d in ds], "-", color="black", lw=2.4, label="Fixed sample, power 0.80")
    ax.set_xlabel("Number of qubits $d$")
    ax.set_ylabel("Expected measurements")
    ax.set_xticks(ds)
    ax.set_ylim(0, max(10000, float(piv.max().max()) * 1.05))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(alpha=.25, lw=.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=2.2)
    fig.subplots_adjust(left=0.135, right=0.585, bottom=0.17, top=0.95)
    master = HERE / "figures" / "graphical_abstract_fig4.png"
    fig.savefig(master, dpi=400, metadata={"Software": "resub2_tables"})
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, format="png")
    plt.close(fig)
    img = Image.open(buf).convert("RGB")
    assert img.size == (660, 295), img.size
    out = HERE / "figures" / "graphical_abstract_fig4.jpg"
    for quality in range(95, 49, -5):
        img.save(out, "JPEG", quality=quality, optimize=True)
        if out.stat().st_size < 45_000:
            break
    print(f"wrote {out} (660x295, quality {quality}, {out.stat().st_size:,} bytes) and {master}")


def wilson(k, n, z: float = 1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def frontier_figure(e2: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    q = e2[(e2.circuit == "QWALK") & (e2.operator == "O1") & e2.method.isin(["chi_square", "mmdagg"])]
    ds, ms, ss = sorted(q.d.unique()), sorted(q.m.unique()), sorted(q.strength.unique())
    plt.rcParams.update({k: 8 for k in ("font.size", "axes.titlesize", "axes.labelsize",
                                        "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    fig, axes = plt.subplots(2, len(ds), figsize=(7.16, 4.0), sharex=True, sharey=True, dpi=300)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(ms)))
    for r, mk in enumerate(("chi_square", "mmdagg")):
        for c, d in enumerate(ds):
            ax = axes[r, c]
            for m, col in zip(ms, colors):
                a = q[(q.method == mk) & (q.d == d) & (q.m == m)].sort_values("strength")
                lo, hi = wilson(a.TP + a.FP, a.n)
                ax.errorbar(a.strength, a.rate, yerr=[a.rate - lo, hi - a.rate], fmt="-o", ms=2.5, lw=1,
                            capsize=1.5, color=col, label=f"$m$ = {m:,}")
            ax.axhline(0.80, ls="--", lw=0.7, color="0.4")
            ax.axhline(0.05, ls=":", lw=0.7, color="0.4")
            ax.set_xscale("log")
            ax.set_xticks(ss)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=.25, lw=.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(f"$d = {d}$")
            if c == 0:
                ax.set_ylabel(f"{LAB[mk]}\nPower")
            if r == 1:
                ax.set_xlabel("Fault strength (rad)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ms), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FRONTIER_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for out in (FRONTIER_FIG, OUT / FRONTIER_FIG.name):
        fig.savefig(out, metadata={"Software": "resub2_tables"}, bbox_inches="tight")
    print("wrote", FRONTIER_FIG)


def bsweep_table(i8: pd.DataFrame) -> None:
    sh = {"shapeA_dense": "Dense", "shapeB_sparse": "Sparse", "shapeC_ghz": "GHZ-type"}
    rows = []
    for (c, d), g in i8.groupby(["circuit", "d"]):
        p = g.pivot_table(index="m", columns="method", values="rate")
        rows.append(" & ".join([sh.get(c, c), str(d), f3(p.loc[200, "mmdagg_B500"]), f3(p.loc[200, "mmdagg_B2000"]),
                                f3(p.loc[1000, "mmdagg_B500"]), f3(p.loc[1000, "mmdagg_B2000"])]) + r" \\")
    k = i8.groupby("method").apply(lambda x: (x.TP + x.FP).sum()); n = i8.groupby("method").n.sum()
    rows += [r"\midrule", " & ".join(["Pooled, all cells", "", f3(k["mmdagg_B500"] / n["mmdagg_B500"]),
                                      f3(k["mmdagg_B2000"] / n["mmdagg_B2000"]), "", ""]) + r" \\"]
    write("tab_bsweep.tex", r"""\begin{table}[ht]
\centering
\caption{\rev{MMDAgg Type~I error under the null as a function of the permutation budget $B$, for three distribution shapes, $d \in \{3, 6, 9\}$ and $m \in \{200, 1000\}$, $1{,}000$ replications per cell (campaign I8). The pooled rates aggregate $18{,}000$ replications per budget.}}
\label{tab:bsweep}
\small
\begin{tabular}{l r cc cc}
\toprule
& & \multicolumn{2}{c}{$m = 200$} & \multicolumn{2}{c}{$m = 1000$} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
Shape & $d$ & $B{=}500$ & $B{=}2000$ & $B{=}500$ & $B{=}2000$ \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def noisy_table(i5: pd.DataFrame) -> None:
    nl = {"aer0": "0.25", "aer1": "0.5", "aer2": "1 (base)", "aer3": "2", "aer4": "4", "fake": "FakeManilaV2"}
    rows = []
    for lab in ["aer0", "aer1", "aer2", "aer3", "aer4", "fake"]:
        g = i5[i5.noise_label == lab]
        def cell(arm, meth, m):
            v = g[(g.arm == arm) & (g.method == meth) & (g.m == m)].rate
            return f3(v.iloc[0]) if len(v) else "--"
        def rng(arm, meth):
            v = g[(g.arm == arm) & (g.method == meth)].rate
            return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]"
        rows.append(" & ".join([nl[lab], cell("A", "regimeA_chi2_1s", 50), cell("A", "regimeA_chi2_1s", 300),
                                cell("A", "regimeA_mmdagg_1s", 50), cell("A", "regimeA_mmdagg_1s", 300),
                                rng("B", "regimeB_chi2_2s"), rng("B", "regimeB_mmdagg_2s"),
                                f3(g[(g.arm == "P") & (g.method == "regimeB_chi2_2s")].rate.min()),
                                f3(g[(g.arm == "P") & (g.method == "regimeB_mmdagg_2s")].rate.min())]) + r" \\")
    write("tab_noisy.tex", r"""\begin{table*}[ht]
\centering
\caption{\rev{Noisy simulator validation on the five-qubit textbook Grover circuit, $1{,}000$ replications per cell (campaign I5). Regime A tests noisy shots from the correct circuit against the ideal null and every rejection is a false alarm. Regime B tests them against a reference sample of size $r = 1{,}000$ from the characterized noisy backend; the entry is the mean over $m \in \{50, \ldots, 300\}$ with the range in brackets. The power columns give the minimum over $m$ against the wrong-oracle mutant under the same noise. Noise scale multiplies depolarizing rates of $0.001$ (one-qubit) and $0.01$ (two-qubit) and a readout error of $0.02$.}}
\label{tab:noisy}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l cc cc cc cc}
\toprule
& \multicolumn{4}{c}{Regime A, ideal null (false alarm rate)} & \multicolumn{2}{c}{Regime B, noise-aware null (Type~I)} & \multicolumn{2}{c}{Regime B power, min over $m$} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
Noise scale & $\chi^2$, $m{=}50$ & $\chi^2$, $300$ & MMDAgg, $50$ & MMDAgg, $300$ & Homogeneity $\chi^2$ & MMDAgg & $\chi^2$ & MMDAgg \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def audit(facts: dict[str, str]) -> None:
    tex = (RESUB / "Main Manuscript.tex").read_text()
    missing = [k for k, v in facts.items() if v not in tex]
    for k, v in facts.items():
        print(f"  {'ok  ' if v in tex else 'MISS'} {k}: {v!r}")
    if missing:
        raise SystemExit(f"numbers audit: {len(missing)} fact(s) not found in the manuscript: {missing}")
    print(f"numbers audit: {len(facts)} facts found in Main Manuscript.tex")


def collect_facts(e1, e1b, s, i8, i5, e2) -> dict[str, str]:
    fr = e2[(e2.circuit == "QWALK") & (e2.operator == "O1") & (e2.d == 3)].set_index(["method", "m", "strength"]).rate
    e1b = e1b[e1b.paradigm == "marginal"] if False else e1b
    j = e1[e1.paradigm == "joint"]
    def med(df, method, op, m): return df[(df.method == method) & (df.operator == op) & (df.m == m)].rate.median()
    seq = s.pivot_table(index="d", columns=["method", "case"], values="n_used_mean")
    pw = s[s.case == "falsecase"].pivot_table(index="d", columns="method", values="rate")
    ap = pd.read_csv(HERE / "analysis_out" / "a_priori_sizes.csv")
    bon = ap[ap.design == "Bonferroni marginal"].copy(); bon["d"] = bon.cell.str.extract(r"d=(\d+)").astype(int); bon = bon.set_index("d").normal
    sav = lambda mk, d: round(100 * (1 - seq.loc[d, (mk, "falsecase")] / bon[d]))
    c = e1b[e1b.method == "chi_square"]; vac = (c.mean_p >= 0.999999)
    def tex_int(x): return f"{int(round(x)):,}".replace(",", "{,}")
    return {
        "O3 chi-square m=1000": f"{med(j, 'chi_square', 'O3', 1000):.2f}",
        "O3 MMDAgg m=1000": f"MMDAgg at {med(j, 'mmdagg', 'O3', 1000):.2f}",
        "vacuous chi-square fraction": f"{round(100 * vac.mean())} percent of the null cells",
        "MC multinomial median Type I": f"median Type~I error of {e1b[e1b.method == 'multinomial_mc'].rate.median():.3f}",
        "MMDAgg median Type I (benchmarks)": f"{e1b[e1b.method == 'mmdagg'].rate.median():.3f} for MMDAgg",
        "adaptive median Type I (benchmarks)": f"median Type~I error of {e1b[e1b.method == 'adaptive'].rate.median():.3f}",
        "sequential E[N] d=2 shift": tex_int(seq.loc[2, ('sequential', 'falsecase')]),
        "sequential E[N] d=12 shift": tex_int(seq.loc[12, ('sequential', 'falsecase')]),
        "exact E[N] d=2 shift": tex_int(seq.loc[2, ('sequential_exact', 'falsecase')]),
        "exact E[N] d=12 shift": tex_int(seq.loc[12, ('sequential_exact', 'falsecase')]),
        "SPRT null d=2": tex_int(seq.loc[2, ('sprt', 'baseline')]),
        "SPRT null d=12": tex_int(seq.loc[12, ('sprt', 'baseline')]),
        "sequential saving range": f"{sav('sequential', 2)} to {sav('sequential', 12)} percent",
        "exact saving range": f"{sav('sequential_exact', 2)} to {max(sav('sequential_exact', d) for d in seq.index)} percent",
        "sequential power d=2": f"{pw.loc[2, 'sequential']:.2f} at $d = 2$",
        "sequential power d=3": f"{pw.loc[3, 'sequential']:.2f} at $d = 3$",
        "exact power d=2": f"{pw.loc[2, 'sequential_exact']:.2f} at $d = 2$",
        "exact power d=4": f"{pw.loc[4, 'sequential_exact']:.2f} at $d = 4$",
        "horizon": tex_int(s.m.max()),
        "B-sweep pooled Type I": f"{(i8.TP + i8.FP).sum() / i8.n.sum():.3f} at both budgets",
        "noisy regime A chi2 m=50 base": f"{i5[(i5.noise_label == 'aer2') & (i5.arm == 'A') & (i5.method == 'regimeA_chi2_1s') & (i5.m == 50)].rate.iloc[0]:.3f} at $m = 50$",
        "frontier chi-square 0.4 rad m=50": f"{fr['chi_square', 50, 0.4]:.2f} at $m = 50$ for 0.4 rad",
        "frontier chi-square 0.2 rad m=200": f"{fr['chi_square', 200, 0.2]:.2f} at $m = 200$ for 0.2 rad",
        "frontier chi-square 0.1 rad m=1000": f"{fr['chi_square', 1000, 0.1]:.2f} at $m = 1{{,}}000$ for 0.1 rad",
        "frontier chi-square 0.05 rad m=2000": f"{fr['chi_square', 2000, 0.05]:.2f} at $m = 2{{,}}000$",
        "frontier MMDAgg 0.2 rad m=200": f"reaching {fr['mmdagg', 200, 0.2]:.2f} at $m = 200$ for 0.2 rad",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="copy outputs into the resubmission folder")
    ap.add_argument("--audit", action="store_true", help="check the numbers quoted in the manuscript against the data")
    args = ap.parse_args()
    e1, e1b, s, i8, i5 = load("I4E1"), load("I4E1B"), load("I7seq"), load("I8"), load("I5")
    e2 = load("I4E2")
    struct_tables(e1, load("I4E3"))
    baseline_table(e1b)
    en_table_and_figure(s)
    frontier_figure(e2)
    bsweep_table(i8)
    noisy_table(i5)
    if args.audit:
        audit(collect_facts(e1, e1b, s, i8, i5, e2))
    if args.install:
        (RESUB / "tables").mkdir(exist_ok=True)
        for f in OUT.glob("tab_*.tex"):
            shutil.copy(f, RESUB / "tables" / f.name)
        shutil.copy(FIG, RESUB / "figures" / FIG.name)
        shutil.copy(FRONTIER_FIG, RESUB / "figures" / FRONTIER_FIG.name)
        print("installed into", RESUB)


if __name__ == "__main__":
    main()
