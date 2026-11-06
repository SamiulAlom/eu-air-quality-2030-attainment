"""
p1fig — results figures for the paper. Each figure is saved as PDF (vector) and 600-dpi PNG in
outputs/figures/ and listed in outputs/MANIFEST.md. Method/workflow figures are drawn separately.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import p1lib as L

POL_LABEL = {"NO2": "NO$_2$", "PM10": "PM$_{10}$", "PM2.5": "PM$_{2.5}$"}
METRIC_LABEL = {"att_annual": "annual mean", "att_daily": "daily (max 18 exceedances)", "att_joint": "both metrics"}


def _save(fig, P: L.Paths, name: str, description: str):
    folder = P.outputs / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{name}.png", dpi=600, bbox_inches="tight")
    L.manifest_add(P, f"figures/{name}.pdf", "07_results_CPU", name, description)
    L.manifest_add(P, f"figures/{name}.png", "07_results_CPU", name, description + " (PNG copy)")


def make_all(P: L.Paths, main: pd.DataFrame, scores: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})

    ok = main[main["status"] == "ok"].copy()
    coords = pd.read_parquet(P.work / "manifest.parquet")[["station", "pollutant", "lat", "lon"]]
    ok = ok.merge(coords, on=["station", "pollutant"], how="left")
    pollutants = [p for p in L.POLLUTANTS if p in set(ok["pollutant"])]

    # 1. Maps of E1 by pollutant and metric
    fig, axes = plt.subplots(len(pollutants), 3, figsize=(7.2, 2.3 * len(pollutants)), squeeze=False)
    points = None
    for i, pol in enumerate(pollutants):
        g = ok[ok["pollutant"] == pol]
        for j, metric in enumerate(["att_annual", "att_daily", "att_joint"]):
            ax = axes[i, j]
            points = ax.scatter(g["lon"], g["lat"], c=g[f"E1_{metric}"], s=2.5, cmap="RdYlBu", vmin=0, vmax=1,
                                linewidths=0)
            ax.set_xlim(-11, 35)
            ax.set_ylim(34, 70)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{POL_LABEL[pol]}: {METRIC_LABEL[metric]}", fontsize=7)
    fig.colorbar(points, ax=axes, shrink=0.6, label="Probability of attaining the 2030 limit at the 2024 level")
    _save(fig, P, "fig_e1_maps_all", "Station maps of E1 by pollutant and metric (supplementary)")
    plt.close(fig)

    # 1b. main-text map: joint attainment only, larger points, longitude/latitude axes; low probabilities drawn last
    fig, axes = plt.subplots(1, len(pollutants), figsize=(7.2, 3.0), squeeze=False, sharey=True)
    for j, pol in enumerate(pollutants):
        ax = axes[0, j]
        g = ok[ok["pollutant"] == pol].sort_values("E1_att_joint", ascending=False)
        points = ax.scatter(g["lon"], g["lat"], c=g["E1_att_joint"], s=4, cmap="RdYlBu", vmin=0, vmax=1,
                            linewidths=0.1, edgecolors="0.4")
        ax.set_xlim(-11, 35)
        ax.set_ylim(34, 70)
        ax.set_aspect(1.5)
        ax.set_axisbelow(True)
        ax.grid(lw=0.3, color="0.85")
        ax.set_xlabel("Longitude (°E)")
        ax.set_title(f"{POL_LABEL[pol]} (n = {len(g)})", fontsize=8)
    axes[0, 0].set_ylabel("Latitude (°N)")
    fig.colorbar(points, ax=axes, shrink=0.8, label="E1, both metrics")
    _save(fig, P, "fig_e1_maps", "Map of E1 for joint attainment (both metrics) at each station's 2024 level")
    plt.close(fig)

    # 2. Hindcast skill by horizon (joint attainment), both scenarios, three baselines
    fig, axes = plt.subplots(1, len(pollutants), figsize=(7.2, 2.4), squeeze=False, sharey=True)
    for j, pol in enumerate(pollutants):
        ax = axes[0, j]
        for scenario, marker, dx in [("frozen", "o", -0.12), ("trend", "s", 0.12)]:
            for ref, colour in [("clim", "C0"), ("pers", "C1"), ("naive", "C2")]:
                s = scores[(scores.pollutant == pol) & (scores.scenario == scenario) & (scores["set"] == "hindcast") &
                           (scores.metric == "att_joint")].sort_values("h")
                if s.empty:
                    continue
                low = (s[f"bss_{ref}"] - s[f"bss_{ref}_lo90"]).clip(lower=0)
                high = (s[f"bss_{ref}_hi90"] - s[f"bss_{ref}"]).clip(lower=0)
                ax.errorbar(s["h"] + dx, s[f"bss_{ref}"], yerr=[low, high], fmt=marker, color=colour, ms=2.5,
                            lw=0.7, label=f"{scenario} vs {ref}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Forecast horizon (years)")
        ax.set_title(POL_LABEL[pol])
    axes[0, 0].set_ylabel("Brier skill score")
    axes[0, -1].legend(fontsize=5, frameon=False, loc="lower left")
    _save(fig, P, "fig_hindcast_skill", "Brier skill score for joint attainment by horizon, scenario and baseline, with 90% two-way cluster-bootstrap intervals")
    plt.close(fig)

    # 3. Reliability diagrams
    rel = pd.read_csv(P.outputs / "reliability_bins.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (scenario, h) in zip(axes, [("frozen", 1), ("trend", 1), ("trend", 6)]):
        for k, pol in enumerate(pollutants):
            r = rel[(rel.pollutant == pol) & (rel.scenario == scenario) & (rel.h == h)]
            ax.plot(r["mean_p"], r["obs_freq"], "-", lw=0.7, color=f"C{k}", label=POL_LABEL[pol])
            ax.scatter(r["mean_p"], r["obs_freq"], s=np.sqrt(r["n"]) * 1.2, color=f"C{k}", alpha=0.7, linewidths=0)
        ax.plot([0, 1], [0, 1], "k--", lw=0.5)
        ax.set_xlabel("Forecast probability")
        ax.set_title(f"{'Level frozen' if scenario == 'frozen' else 'Trend'}, {h} yr ahead", fontsize=8)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(frameon=False, fontsize=6)
    _save(fig, P, "fig_reliability", "Reliability diagrams for joint attainment (level frozen 1 year ahead; trend 1 and 6 years ahead); marker area proportional to the square root of the number of forecasts in the bin")
    plt.close(fig)

    # 4. Weather-marginal share by member state
    grp = pd.read_csv(P.outputs / "summary_by_group.csv")
    by_cc = grp.dropna(subset=["cc"])
    countries = sorted(by_cc["cc"].unique())
    width = 0.8 / max(len(pollutants), 1)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for i, pol in enumerate(pollutants):
        g = by_cc[by_cc.pollutant == pol].set_index("cc").reindex(countries)
        ax.bar(np.arange(len(countries)) + i * width, g["share_weather_marginal_joint"], width=width, label=POL_LABEL[pol])
    ax.set_xticks(np.arange(len(countries)) + width)
    ax.set_xticklabels(countries, fontsize=6)
    ax.set_ylabel("Share weather-marginal")
    ax.legend(frameon=False, fontsize=6)
    _save(fig, P, "fig_marginal_by_country", "Share of stations with E1 (both metrics) between 0.1 and 0.9, by member state and pollutant")
    plt.close(fig)

    # 5. Uncertainty decomposition (E5)
    e5 = pd.read_csv(P.outputs / "summary_e5.csv")
    labels = [f"{POL_LABEL[r.pollutant]}\n{'annual mean' if r.metric == 'mean' else 'exceedance days'}" for r in e5.itertuples()]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    bottom = np.zeros(len(e5))
    for part, colour in [("share_trend", "C0"), ("share_weather", "C1"), ("share_resid", "C2")]:
        lab = {"share_trend": "trend + year-to-year level", "share_weather": "weather", "share_resid": "residual"}[part]
        ax.bar(labels, e5[part], bottom=bottom, color=colour, label=lab)
        bottom += e5[part].fillna(0).values
    ax.set_ylabel("Median share of 2030 variance")
    ax.tick_params(axis="x", labelsize=6)
    ax.legend(frameon=False, fontsize=6)
    _save(fig, P, "fig_e5_variance", "Median shares of the 2030 predictive variance from trend, weather and residual noise")
    plt.close(fig)

    # 6. Headroom (E3)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    for pol in pollutants:
        v = ok.loc[ok.pollutant == pol, "E3_att_joint_90"].dropna() * 100
        ax.hist(v.clip(-90, 100), bins=40, histtype="step", label=POL_LABEL[pol])
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Change in underlying level for 90% probability (%)")
    ax.set_ylabel("Stations")
    ax.legend(frameon=False, fontsize=6)
    _save(fig, P, "fig_e3_headroom", "Change in the weather-adjusted level needed for a 90% probability of attaining both metrics (negative = reduction needed)")
    plt.close(fig)

    # 7. Worked examples, chosen by the pre-specified median rule (§3.5)
    examples = []
    for pol, stype, area in [("NO2", "traffic", "urban"), ("PM2.5", "background", "urban")]:
        g = ok[(ok.pollutant == pol) & (ok.stype == stype) & (ok.area == area)].sort_values(["E1_att_joint", "station"])
        if len(g):
            examples.append(g.iloc[(len(g) - 1) // 2])
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(3.6 * len(examples), 2.4), squeeze=False)
        for ax, r in zip(axes[0], examples):
            a = json.loads(r["a_series"])
            years = sorted(int(k) for k in a)
            level = np.exp([a[str(y)] for y in years]) - L.OFFSET
            ax.plot(years, level, "o-", ms=2.5, lw=0.8)
            ax.axhline(L.LIMITS[r["pollutant"]]["annual"], color="r", lw=0.6, ls="--", label="2030 annual limit")
            ax.set_title(f"{r['station']}: {POL_LABEL[r['pollutant']]}, {r['area']} {r['stype']}", fontsize=7)
            ax.set_ylabel("Weather-adjusted level (µg/m³)")
            ax.legend(frameon=False, fontsize=6)
        _save(fig, P, "fig_worked_examples", "Weather-adjusted yearly levels for the two pre-specified example stations")
        plt.close(fig)
        pd.DataFrame(examples).drop(columns=["a_series", "se_series"]).to_csv(P.outputs / "worked_examples.csv", index=False)
        L.manifest_add(P, "worked_examples.csv", "07_figures_CPU", "examples", "The two example stations chosen by the pre-specified median rule")
