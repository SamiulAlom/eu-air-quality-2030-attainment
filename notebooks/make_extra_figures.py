"""Extra manuscript figures drawn from the stored result files in outputs/ (nothing is recomputed or refitted).

Usage: python make_extra_figures.py <project folder>
Reads   <project>/outputs/*.csv and <project>/outputs/v7_preregistered/*.csv
Writes  <project>/figures/<name>.pdf and .png (600 dpi), and copies each PDF to <project>/drafts/latex/figures/

Every plotted value is read from an outputs/ file; the only numbers typed here are the study design
(forecast origins 2017-2023, horizons 1-6, 20-year weather pool, data from 2013), which come from the analysis plan.

The three method diagrams (workflow, inclusion flow, hindcast design) are not drawn here: their draw.io
sources are in figures/.
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLS = ["NO2", "PM10", "PM2.5"]
POL = {"NO2": "NO$_2$", "PM10": "PM$_{10}$", "PM2.5": "PM$_{2.5}$"}
COL = {"NO2": "#1f77b4", "PM10": "#d95f02", "PM2.5": "#7570b3"}
SCEN = {"frozen": "level frozen", "trend": "trend continuing"}
RUN = {"v7_preregistered": "original run (v7)", "v8_corrected": "corrected run (v8)"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "axes.titlesize": 8, "legend.fontsize": 7,
                     "legend.frameon": False})


def save(fig, P: Path, name: str):
    out = P / "figures"
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight")
    latex = P / "drafts" / "latex" / "figures"
    if latex.exists():  # only inside the authors' manuscript folder
        shutil.copyfile(out / f"{name}.pdf", latex / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def e1_distribution(P: Path):
    """Distribution of E1 (both metrics) by pollutant; E1 is identical in the original and corrected runs."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=False)
    bins = np.linspace(0, 1, 21)
    for ax, p in zip(axes, POLS):
        d = pd.read_csv(P / "outputs" / f"main_{p}_station_results.csv")
        e = d.loc[d["status"].eq("ok"), "E1_att_joint"].dropna()
        ax.axvspan(0.1, 0.9, color="0.92", zorder=0)
        ax.hist(e, bins=bins, color=COL[p], edgecolor="white", lw=0.4)
        ax.set_title(f"{POL[p]} (n = {len(e):,})")
        ax.set_xlabel("E1, both metrics")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Series")
    axes[1].text(0.5, 0.97, "weather-marginal\n0.1–0.9", transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=6.5, color="0.35")
    fig.tight_layout()
    save(fig, P, "fig_e1_distribution")


def coverage_by_horizon(P: Path):
    """90% interval coverage of the annual mean by horizon, both scenarios, both runs (hindcast set)."""
    v7 = pd.read_csv(P / "outputs" / "v7_preregistered" / "hindcast_coverage.csv").assign(run="v7_preregistered")
    v8 = pd.read_csv(P / "outputs" / "hindcast_coverage.csv").assign(run="v8_corrected")
    d = pd.concat([v7, v8])
    d = d[d["set"].eq("hindcast")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, p in zip(axes, POLS):
        ax.axhline(0.9, color="0.5", lw=0.6, ls="--")
        for sc, mk in [("frozen", "o"), ("trend", "s")]:
            for run, ls in [("v7_preregistered", "-"), ("v8_corrected", ":")]:
                g = d[(d["pollutant"] == p) & (d["scenario"] == sc) & (d["run"] == run)].sort_values("h")
                ax.plot(g["h"], g["cov90"], ls=ls, marker=mk, ms=3, lw=0.9, color=COL[p] if sc == "trend" else "0.45",
                        label=f"{SCEN[sc]}, {RUN[run]}")
        ax.set_title(POL[p]); ax.set_xlabel("Horizon (years)"); ax.set_xticks(range(1, 7))
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Coverage of 90% interval")
    axes[0].text(6.0, 0.915, "nominal 0.9", fontsize=6, color="0.4", ha="right", va="bottom")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="0.45", marker="o", ms=3, label="level frozen"),
                        Line2D([], [], color="k", marker="s", ms=3, label="trend continuing (pollutant colour)"),
                        Line2D([], [], color="0.3", ls="-", label="original run (v7)"),
                        Line2D([], [], color="0.3", ls=":", label="corrected run (v8)")],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, P, "fig_coverage_horizon")


def by_target_year(P: Path):
    """One-year-ahead coverage and mean log error by target year (post-hoc), original run."""
    d = pd.read_csv(P / "outputs" / "posthoc_by_target_year.csv")
    d = d[d["rows_version"].eq("v7_preregistered")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True, sharey="row")
    for j, p in enumerate(POLS):
        for sc, mk, c in [("frozen", "o", "0.45"), ("trend", "s", COL[p])]:
            g = d[(d["pollutant"] == p) & (d["scenario"] == sc)].sort_values("target")
            axes[0, j].plot(g["target"], g["coverage90"], marker=mk, ms=3, lw=0.9, color=c, label=SCEN[sc])
            axes[1, j].plot(g["target"], g["mean_log_error"], marker=mk, ms=3, lw=0.9, color=c, label=SCEN[sc])
        for i in range(2):
            axes[i, j].axvspan(2019.5, 2021.5, color="0.92", zorder=0)
        axes[0, j].axhline(0.9, color="0.5", lw=0.6, ls="--")
        axes[1, j].axhline(0, color="0.5", lw=0.6)
        axes[0, j].set_title(POL[p])
        axes[1, j].set_xlabel("Target year")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Coverage of 90% interval")
    axes[1, 0].set_ylabel("Mean log error\n(observed − forecast median)")
    from matplotlib.lines import Line2D
    axes[0, 1].legend(handles=[Line2D([], [], color="0.45", marker="o", ms=3, label="level frozen"),
                               Line2D([], [], color="k", marker="s", ms=3, label="trend continuing\n(pollutant colour)")],
                      loc="lower center")
    axes[0, 1].text(2020.5, 0.97, "2020–2021", ha="center", va="top", fontsize=6.5, color="0.35")
    fig.tight_layout()
    save(fig, P, "fig_target_year")


def spread(P: Path):
    """Spread ratio and coverage before and after removing each cell's mean error (post-hoc), original run."""
    d = pd.read_csv(P / "outputs" / "posthoc_diag_errors.csv")
    d = d[d["rows_version"].eq("v7_preregistered")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for p in POLS:
        for sc, mk, ls in [("frozen", "o", "-"), ("trend", "s", "--")]:
            g = d[(d["pollutant"] == p) & (d["scenario"] == sc)].sort_values("h")
            axes[0].plot(g["h"], g["spread_ratio"], marker=mk, ms=3, lw=0.9, ls=ls, color=COL[p],
                         label=f"{POL[p]}, {SCEN[sc]}")
            axes[1].plot(g["h"], g["coverage90"], marker=mk, ms=3, lw=0.9, ls=ls, color=COL[p])
            axes[1].plot(g["h"], g["coverage90_bias_removed"], marker=mk, ms=3, lw=0.5, ls=ls, color=COL[p],
                         alpha=0.45, mfc="white")
    axes[0].axhline(1, color="0.5", lw=0.6, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_yticks([1, 1.5, 2, 3, 4, 5]); axes[0].set_yticklabels(["1", "1.5", "2", "3", "4", "5"])
    axes[0].set_ylabel("Spread ratio\n(sd of log error / sd implied by interval)")
    axes[1].axhline(0.9, color="0.5", lw=0.6, ls="--")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Coverage of 90% interval")
    axes[1].text(0.03, 0.03, "filled: as forecast\nopen: mean error removed", transform=axes[1].transAxes,
                 ha="left", va="bottom", fontsize=6.5, color="0.35")
    for ax in axes:
        ax.set_xlabel("Horizon (years)"); ax.set_xticks(range(1, 7))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, P, "fig_spread")


def bands(P: Path):
    """One-year Brier score of the model and of the naive trend on the same station-years, by probability band."""
    d = pd.read_csv(P / "outputs" / "posthoc_forecast_bands.csv")
    d = d[d["rows_version"].eq("v7_preregistered")]
    order = ["p <= 0.1", "0.1 < p < 0.9", "p >= 0.9"]
    lab = {"p <= 0.1": "p ≤ 0.1", "0.1 < p < 0.9": "0.1 < p < 0.9", "p >= 0.9": "p ≥ 0.9"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharey=True)
    for i, sc in enumerate(["frozen", "trend"]):
        for j, p in enumerate(POLS):
            ax = axes[i, j]
            g = d[(d["pollutant"] == p) & (d["scenario"] == sc)].set_index("band").loc[order]
            x = np.arange(3)
            ax.bar(x - 0.18, g["brier"], 0.36, color=COL[p], label="model")
            ax.bar(x + 0.18, g["brier_naive_same_rows"], 0.36, color="0.75", label="naive trend")
            for k, b in enumerate(order):
                ax.text(k, max(g.loc[b, "brier"], g.loc[b, "brier_naive_same_rows"]) + 0.01,
                        f"{g.loc[b, 'share_of_forecasts']:.0%}", ha="center", fontsize=6, color="0.3")
            ax.set_xticks(x); ax.set_xticklabels([lab[b] for b in order], fontsize=6.5)
            ax.set_title(f"{POL[p]}, {SCEN[sc]}")
        axes[i, 0].set_ylabel("Brier score")
        axes[i, 0].set_ylim(0, 0.3)
    from matplotlib.patches import Patch
    axes[0, 1].legend(handles=[Patch(color="0.25", label="model (pollutant colour)"), Patch(color="0.75", label="naive trend")],
                      loc="upper right")
    fig.tight_layout()
    save(fig, P, "fig_bands")


def recal(P: Path):
    """Exploratory: coverage of stored and empirically scaled 90% intervals, corrected run."""
    d = pd.read_csv(P / "outputs" / "exploratory_recal_summary.csv")
    d = d[d["rows_version"].eq("v8_corrected")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, s, title in zip(axes, ["hindcast", "check2025"], ["Hindcast, origins 2020–2023, h = 1–4", "2025 check"]):
        k = 0
        ticks = []
        for p in POLS:
            for sc in ["frozen", "trend"]:
                g = d[(d["pollutant"] == p) & (d["scenario"] == sc) & (d["set"] == s)].iloc[0]
                ax.plot([k, k], [g["coverage90_raw"], g["coverage90_recal"]], color="0.8", lw=0.8, zorder=1)
                ax.scatter(k, g["coverage90_raw"], marker="o", facecolor="white", edgecolor=COL[p], s=18, zorder=2)
                ax.errorbar(k, g["coverage90_recal"], yerr=[[g["coverage90_recal"] - g["cov90_lo"]], [g["cov90_hi"] - g["coverage90_recal"]]],
                            fmt="o", color=COL[p], ms=4, lw=0.8, zorder=3)
                ticks.append(f"{POL[p]}\n{'frozen' if sc == 'frozen' else 'trend'}")
                k += 1
        ax.axhline(0.9, color="0.5", lw=0.6, ls="--")
        ax.set_xticks(range(k)); ax.set_xticklabels(ticks, fontsize=6.5)
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Coverage of 90% interval")
    axes[1].text(0.98, 0.04, "open: stored interval\nfilled: scaled (90% station bootstrap)", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=6.5, color="0.35")
    fig.tight_layout()
    save(fig, P, "fig_recal")


def sensitivity(P: Path):
    """Pre-specified E1 sensitivity analyses: mean absolute change in E1 against the main result on the same series."""
    d = pd.read_csv(P / "outputs" / "v7_preregistered" / "sensitivity_summary.csv")
    d = d[d["estimand"].eq("E1 joint") & ~d["variant"].eq("main")].copy()
    lab = {"E1 pool1995": "weather pool 1995–2024", "E1 pool2015": "weather pool 2015–2024",
           "E1 recency": "recency-weighted pool", "E1 block3": "residual blocks of 3 days",
           "E1 block14": "residual blocks of 14 days", "E1 unstrat": "unstratified residuals",
           "E1 dustdrop5": "five highest dust days dropped", "GAM (E1_att_joint)": "additive model (GAM) instead of trees",
           "meteorology within 25 km": "weather station within 25 km*", "training share >=85%": "training share ≥ 85%*"}
    d = d[d["variant"].isin(lab)]
    variants = [v for v in lab if v in set(d["variant"])]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for k, p in enumerate(POLS):
        g = d[d["pollutant"] == p].set_index("variant")
        y = [len(variants) - 1 - variants.index(v) + (k - 1) * 0.22 for v in g.index]
        ax.scatter(g["mean_abs_diff_vs_main"], y, color=COL[p], s=14, label=POL[p], zorder=3)
    ax.set_yticks(range(len(variants))); ax.set_yticklabels([lab[v] for v in variants][::-1])
    ax.set_xlabel("Mean absolute difference in E1 from the main analysis, same series")
    ax.grid(axis="x", lw=0.3, color="0.85")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, P, "fig_sensitivity")


def by_origin(P: Path):
    """One-year Brier skill of joint attainment against climatology and the naive trend, by forecast origin (post-hoc)."""
    d = pd.read_csv(P / "outputs" / "posthoc_by_origin.csv")
    d = d[d["rows_version"].eq("v7_preregistered")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True, sharey="row")
    for i, (ref, name) in enumerate([("bss_clim", "climatology"), ("bss_naive", "naive trend")]):
        for j, p in enumerate(POLS):
            ax = axes[i, j]
            for sc, mk, c in [("frozen", "o", "0.45"), ("trend", "s", COL[p])]:
                g = d[(d["pollutant"] == p) & (d["scenario"] == sc)].sort_values("origin")
                ax.plot(g["origin"], g[ref], marker=mk, ms=3, lw=0.9, color=c)
            ax.axhline(0, color="0.5", lw=0.6)
            if i == 0:
                ax.set_title(POL[p])
            else:
                ax.set_xlabel("Forecast origin")
        axes[i, 0].set_ylabel(f"Brier skill score\nagainst {name}")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="0.45", marker="o", ms=3, label="level frozen"),
                        Line2D([], [], color="k", marker="s", ms=3, label="trend continuing (pollutant colour)")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    save(fig, P, "fig_origin")


def own_weather(P: Path):
    """Brier score of joint attainment with resampled weather and with the target year's own weather (post-hoc)."""
    d = pd.read_csv(P / "outputs" / "posthoc_diag_perfect_weather.csv")
    d = d[d["rows_version"].eq("v7_preregistered")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True, sharey=True)
    for i, sc in enumerate(["frozen", "trend"]):
        for j, p in enumerate(POLS):
            ax = axes[i, j]
            g = d[(d["pollutant"] == p) & (d["scenario"] == sc)].sort_values("h")
            ax.plot(g["h"], g["brier_resampled_weather"], marker="o", ms=3, lw=0.9, color=COL[p], label="resampled weather")
            ax.plot(g["h"], g["brier_perfect_weather"], marker="o", ms=3, lw=0.9, ls="--", color=COL[p], mfc="white",
                    label="target year's own weather")
            ax.set_title(f"{POL[p]}, {SCEN[sc]}")
            ax.set_xticks(range(1, 7))
            if i == 1:
                ax.set_xlabel("Horizon (years)")
        axes[i, 0].set_ylabel("Brier score, both metrics")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, P, "fig_own_weather")


def e1_vs_e2(P: Path):
    """E1 (2024 level, weather resampled) against E2 (2030, trend scenario, corrected run) for each series."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, p in zip(axes, POLS):
        d = pd.read_csv(P / "outputs" / f"main_{p}_station_results.csv")
        d = d[d["status"].eq("ok")].dropna(subset=["E1_att_joint", "E2_att_joint"])
        ax.scatter(d["E1_att_joint"], d["E2_att_joint"], s=3, color=COL[p], alpha=0.35, linewidths=0)
        ax.plot([0, 1], [0, 1], color="0.5", lw=0.6, ls="--")
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{POL[p]} (n = {len(d):,})")
        ax.set_xlabel("E1: 2024 level")
        ax.set_aspect("equal")
    axes[0].set_ylabel("E2: 2030, trend continuing")
    fig.tight_layout()
    save(fig, P, "fig_e1_e2")


def member_states(P: Path):
    """Mean E1 (both metrics) and share of weather-marginal series by member state and pollutant (original run)."""
    d = pd.read_csv(P / "outputs" / "v7_preregistered" / "summary_by_group.csv")
    d = d[d["cc"].notna()]
    ccs = sorted(set(d["cc"]))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
    x = np.arange(len(ccs))
    for k, p in enumerate(POLS):
        g = d[d["pollutant"] == p].set_index("cc")
        for ax, col in zip(axes, ["mean_E1_joint", "share_weather_marginal_joint"]):
            vals = [g.loc[c, col] if c in g.index else np.nan for c in ccs]
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=COL[p], label=POL[p])
            for xi, c in zip(x, ccs):
                if c not in g.index:          # no series of this pollutant in this member state
                    ax.text(xi + (k - 1) * 0.27, 0.01, "×", ha="center", va="bottom", fontsize=5, color=COL[p])
    axes[0].set_ylabel("Mean E1, both metrics")
    axes[1].set_ylabel("Share weather-marginal\n(0.1 < E1 < 0.9)")
    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", lw=0.3, color="0.88")
    axes[1].set_xticks(x); axes[1].set_xticklabels(ccs, fontsize=6.5)
    axes[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout()
    save(fig, P, "fig_member_states")


if __name__ == "__main__":
    P = Path(sys.argv[1])
    for f in (e1_distribution, coverage_by_horizon, by_target_year, spread, bands,
              recal, sensitivity, by_origin, own_weather, e1_vs_e2, member_states):
        f(P)
