"""Write notebooks/08_posthoc_diagnostics_CPU.ipynb.

POST-HOC diagnostics of the failed pre-specified hindcast validation (see the paper, Section 3.6). They read the stored hindcast rows only: nothing is refitted and no pre-specified number changes. Every
output file is prefixed "posthoc_" and labelled as post-hoc in MANIFEST.md. Each table has a `rows_version` column:
`v7_preregistered` (the original run, frozen in outputs/v7_preregistered/) and `v8_corrected` (the corrected run,
deviations 15 and 17), whichever are supplied.

Run on Colab with:
  node scripts/colab-run.mjs projects/project-1/j1-env-modeling-assessment notebooks/08_posthoc_diagnostics_CPU.ipynb \
       --input outputs/hindcast_NO2_rows.csv.gz --input outputs/hindcast_PM10_rows.csv.gz --input outputs/hindcast_PM2.5_rows.csv.gz \
       --input outputs/v7_preregistered/hindcast_NO2_rows.csv.gz --input outputs/v7_preregistered/hindcast_PM10_rows.csv.gz \
       --input outputs/v7_preregistered/hindcast_PM2.5_rows.csv.gz
(The first three are the corrected run's rows; leave them out to analyse the original run only.)
"""
import json
from pathlib import Path

CELLS = [
    # 0. runtime and inputs
    '''# RUNTIME: CPU (standard) - do NOT pick a GPU
# POST-HOC diagnostics (not pre-specified). Reads stored hindcast rows only; nothing is refitted.
import os, platform
import numpy as np, pandas as pd
ROOT = os.environ.get("PAPER_ROOT", ".")
IN = os.path.join(ROOT, "prior_outputs")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
print("python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__)
POLS = ["NO2", "PM10", "PM2.5"]
# the corrected run's rows are in IN itself; the original run's in IN/v7_preregistered. A file in IN that is
# byte-identical to the v7 copy is the original run (corrected rows not supplied) and is not counted twice.
VERSIONS = {"v7_preregistered": os.path.join(IN, "v7_preregistered"), "v8_corrected": IN}
frames = []
for ver, folder in VERSIONS.items():
    files = [os.path.join(folder, f"hindcast_{p}_rows.csv.gz") for p in POLS]
    if not all(os.path.exists(f) for f in files):
        print(ver, "not supplied"); continue
    if ver == "v8_corrected" and all(os.path.exists(os.path.join(VERSIONS["v7_preregistered"], os.path.basename(f))) and
                                     open(f, "rb").read() == open(os.path.join(VERSIONS["v7_preregistered"], os.path.basename(f)), "rb").read() for f in files):
        print("v8_corrected rows are identical to v7: skipped"); continue
    frames.append(pd.concat([pd.read_csv(f) for f in files], ignore_index=True).assign(rows_version=ver))
allrows = pd.concat(frames, ignore_index=True)
rows = allrows[allrows["set"] == "hindcast"].copy()          # the 2025 check is scored separately, as in the plan
print(rows.groupby("rows_version").size())
G = ["rows_version", "pollutant", "scenario", "h"]''',

    # 1. D1 + D2: bias and dispersion of the annual-mean prediction, on the log scale used by the model
    '''# D1 signed error and D2 dispersion of the annual-mean prediction (log(C + 0.1), the model's scale);
# only forecasts with an interval (deviation 17)
r = rows.dropna(subset=["obs_mean", "pred_mean_med", "pi90_lo", "pi90_hi"]).copy()
lg = lambda x: np.log(np.clip(x, 0, None) + 0.1)
r["err"] = lg(r["obs_mean"]) - lg(r["pred_mean_med"])                       # > 0: observed higher than predicted
r["sd_implied"] = (lg(r["pi90_hi"]) - lg(r["pi90_lo"])) / (2 * 1.6449)      # spread the 90% interval implies
r["inside90"] = (r["obs_mean"] >= r["pi90_lo"]) & (r["obs_mean"] <= r["pi90_hi"])
out = []
for keys, g in r.groupby(G):
    bias = g["err"].mean()
    centred = (g["err"] - bias).abs() <= 1.6449 * g["sd_implied"]          # coverage if the cell's bias were removed
    out.append({**dict(zip(G, keys)), "n": len(g),
                "mean_log_error": bias, "median_log_error": g["err"].median(),
                "share_obs_below_pred": float((g["err"] < 0).mean()),
                "sd_log_error": g["err"].std(ddof=1), "mean_sd_implied": g["sd_implied"].mean(),
                "spread_ratio": g["err"].std(ddof=1) / g["sd_implied"].mean(),
                "coverage90": float(g["inside90"].mean()), "coverage90_bias_removed": float(centred.mean())})
d12 = pd.DataFrame(out)
d12.to_csv(os.path.join(OUT, "posthoc_diag_errors.csv"), index=False)
print(d12.round(3).to_string(index=False))''',

    # 2. D3: perfect weather vs resampled weather
    '''# D3 Brier score of joint attainment with resampled vs the target year's own weather (within the fitted model)
p = rows.dropna(subset=["obs_att_joint", "p_att_joint", "pperfect_att_joint"]).copy()
out = []
for keys, g in p.groupby(G):
    b_res = ((g["p_att_joint"] - g["obs_att_joint"]) ** 2).mean()
    b_per = ((g["pperfect_att_joint"] - g["obs_att_joint"]) ** 2).mean()
    out.append({**dict(zip(G, keys)), "n": len(g), "brier_resampled_weather": b_res,
                "brier_perfect_weather": b_per, "share_of_brier_removed_by_perfect_weather": (b_res - b_per) / b_res if b_res > 0 else np.nan})
d3 = pd.DataFrame(out)
d3.to_csv(os.path.join(OUT, "posthoc_diag_perfect_weather.csv"), index=False)
print(d3.round(4).to_string(index=False))''',

    # 3. D4: CRPS blow-up
    '''# D4 the CRPS blow-up: how many rows, where, and the robust (median) CRPS
c = rows.dropna(subset=["crps_mean"]).copy()
out = []
for keys, g in c.groupby(G):
    out.append({**dict(zip(G, keys)), "n": len(g), "crps_mean": g["crps_mean"].mean(),
                "crps_median": g["crps_mean"].median(), "n_rows_crps_gt_1e3": int((g["crps_mean"] > 1e3).sum()),
                "max_crps": g["crps_mean"].max()})
d4 = pd.DataFrame(out)
d4.to_csv(os.path.join(OUT, "posthoc_diag_crps.csv"), index=False)
print(d4.to_string(index=False))''',

    # 4. D5: coverage and Brier by target year; D6: without target years 2020-2021
    '''# D5 90% coverage and Brier score of joint attainment by target year (h = 1);
# D6 the same summaries pooled with and without target years 2020-2021 (pandemic years)
def brier(pcol, g):
    ok = g.dropna(subset=[pcol, "obs_att_joint"])
    return float(((ok[pcol] - ok["obs_att_joint"]) ** 2).mean()) if len(ok) else np.nan

def paired(g, cols):
    """Brier scores of several forecast columns on the SAME rows: every column and the outcome observed
    (a skill score must compare model and baseline on identical forecasts)."""
    ok = g.dropna(subset=list(cols) + ["obs_att_joint"])
    return [float(((ok[c] - ok["obs_att_joint"]) ** 2).mean()) if len(ok) else np.nan for c in cols], len(ok)

out = []
for keys, g in r[r["h"] == 1].groupby(["rows_version", "pollutant", "scenario", "target"]):
    gr = rows[(rows["rows_version"] == keys[0]) & (rows["pollutant"] == keys[1]) & (rows["scenario"] == keys[2]) &
              (rows["target"] == keys[3]) & (rows["h"] == 1)]
    out.append({"rows_version": keys[0], "pollutant": keys[1], "scenario": keys[2], "target": keys[3], "n": len(g),
                "coverage90": float(g["inside90"].mean()), "mean_log_error": float(g["err"].mean()),
                "brier": paired(gr, ["p_att_joint", "naive_att_joint"])[0][0],
                "brier_naive": paired(gr, ["p_att_joint", "naive_att_joint"])[0][1]})
d5 = pd.DataFrame(out)
d5.to_csv(os.path.join(OUT, "posthoc_by_target_year.csv"), index=False)
print(d5.round(3).to_string(index=False))

out = []
for excl, lab in [((), "all target years"), ((2020, 2021), "without 2020-2021")]:
    for keys, g in r[~r["target"].isin(excl)].groupby(G):
        gr = rows[~rows["target"].isin(excl)]
        gr = gr[(gr["rows_version"] == keys[0]) & (gr["pollutant"] == keys[1]) & (gr["scenario"] == keys[2]) & (gr["h"] == keys[3])]
        (b, bc, bp, bn), n_scored = paired(gr, ["p_att_joint", "clim_att_joint", "pers_att_joint", "naive_att_joint"])
        out.append({**dict(zip(G, keys)), "target_years": lab, "n": len(g), "n_brier": n_scored, "coverage90": float(g["inside90"].mean()),
                    "brier": b, "bss_clim": 1 - b / bc if bc else np.nan, "bss_pers": 1 - b / bp if bp else np.nan,
                    "bss_naive": 1 - b / bn if bn else np.nan})
d6 = pd.DataFrame(out)
d6.to_csv(os.path.join(OUT, "posthoc_without_2020_2021.csv"), index=False)
print(d6[d6["h"] == 1].round(3).to_string(index=False))''',

    # 5. D7: forecast bands, discrimination; D8: skill by origin
    '''# D7 joint-attainment forecasts by probability band (h = 1): share, observed frequency, Brier score, and the naive
# trend's Brier score on the same rows; AUC (rank-based) over all forecasts.  D8 Brier skill by forecast origin (h = 1)
from scipy.stats import rankdata
j = rows.dropna(subset=["p_att_joint", "obs_att_joint", "naive_att_joint"])
j = j[j["h"] == 1].copy()
# explicit band edges: p <= 0.1, 0.1 < p < 0.9, p >= 0.9 (pd.cut is right-inclusive and would put p = 0.9 in the middle band)
j["band"] = np.select([j["p_att_joint"] <= 0.1, j["p_att_joint"] >= 0.9], ["p <= 0.1", "p >= 0.9"], default="0.1 < p < 0.9")
out = []
for keys, g in j.groupby(["rows_version", "pollutant", "scenario"]):
    y, s_ = g["obs_att_joint"].to_numpy(), g["p_att_joint"].to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    auc = float((rankdata(s_)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan
    for band, gb in g.groupby("band", observed=True):
        out.append({"rows_version": keys[0], "pollutant": keys[1], "scenario": keys[2], "band": band, "n": len(gb),
                    "share_of_forecasts": len(gb) / len(g), "mean_p": gb["p_att_joint"].mean(),
                    "observed_frequency": gb["obs_att_joint"].mean(),
                    "brier": float(((gb["p_att_joint"] - gb["obs_att_joint"]) ** 2).mean()),
                    "brier_naive_same_rows": float(((gb["naive_att_joint"] - gb["obs_att_joint"]) ** 2).mean()),
                    "share_p_exactly_0_or_1_in_cell": float(((g["p_att_joint"] == 0) | (g["p_att_joint"] == 1)).mean()),
                    "auc_all_forecasts": auc})
d7 = pd.DataFrame(out)
d7.to_csv(os.path.join(OUT, "posthoc_forecast_bands.csv"), index=False)
print(d7.round(3).to_string(index=False))

out = []
for keys, g in rows[rows["h"] == 1].groupby(["rows_version", "pollutant", "scenario", "origin"]):
    (b, bc, bn), n_scored = paired(g, ["p_att_joint", "clim_att_joint", "naive_att_joint"])
    out.append({"rows_version": keys[0], "pollutant": keys[1], "scenario": keys[2], "origin": keys[3],
                "n": n_scored, "brier": b,
                "bss_clim": 1 - b / bc if bc else np.nan, "bss_naive": 1 - b / bn if bn else np.nan})
d8 = pd.DataFrame(out)
d8.to_csv(os.path.join(OUT, "posthoc_by_origin.csv"), index=False)
print(d8.round(3).to_string(index=False))''',

    # 6. manifest
    '''# MANIFEST rows (exact file names; every file labelled POST-HOC)
nb = "08_posthoc_diagnostics_CPU"
rows_md = ["| file | notebook | cell | description |", "|---|---|---|---|",
    f"| posthoc_diag_errors.csv | {nb} | 1 | POST-HOC D1/D2: signed log error, spread ratio and bias-removed 90% coverage of the annual-mean prediction, by run x pollutant x scenario x horizon |",
    f"| posthoc_diag_perfect_weather.csv | {nb} | 2 | POST-HOC D3: Brier score of joint attainment with resampled vs the target year's own weather |",
    f"| posthoc_diag_crps.csv | {nb} | 3 | POST-HOC D4: CRPS blow-up rows and robust median CRPS |",
    f"| posthoc_by_target_year.csv | {nb} | 4 | POST-HOC D5: one-year coverage, log error and Brier score (model and naive trend) by target year |",
    f"| posthoc_without_2020_2021.csv | {nb} | 4 | POST-HOC D6: coverage and Brier skill (point estimates) with and without target years 2020-2021 |",
    f"| posthoc_forecast_bands.csv | {nb} | 5 | POST-HOC D7: one-year joint-attainment forecasts by probability band, with the naive trend on the same rows, and AUC |",
    f"| posthoc_by_origin.csv | {nb} | 5 | POST-HOC D8: one-year Brier score and skill by forecast origin |"]
open(os.path.join(OUT, "MANIFEST.md"), "w").write("\\n".join(rows_md) + "\\n")
print("manifest written")''',
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "cells": [{"cell_type": "code", "id": f"c{i}", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": src} for i, src in enumerate(CELLS)]}
    out = Path(__file__).parent / "08_posthoc_diagnostics_CPU.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
