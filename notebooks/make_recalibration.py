"""Write notebooks/09_exploratory_recalibration_CPU.ipynb.

EXPLORATORY, POST-HOC interval recalibration (deviation 16 in Supplementary Section S5).
It reads stored hindcast rows only: nothing is refitted. Every output file is prefixed "exploratory_" and
labelled "exploratory recalibration, not validated" in MANIFEST.md.

Method (split conformal on the model's log(C + 0.1) scale):
  * standardised error z = |log(obs + 0.1) - log(pred + 0.1)| / implied sd, where the implied sd is the
    half-width of the stored 90% interval divided by 1.6449;
  * calibration rows: target year <= 2020 (all known at the end of 2020);
    evaluation rows: origin >= 2020 (target years 2021-2024) and the 2025 check;
  * k_h = q_0.90(z) / 1.6449 for the 90% interval (q_0.80(z) / 1.2816 for 80%), per pollutant x scenario x h;
  * horizons without calibration rows (h >= 4) take log k_h extrapolated linearly in log h from h = 1-3;
  * widened interval: exp(centre +/- z_level * k_h * sd) - 0.1, with centre = log(pred + 0.1). Bias is not removed.

Run on Colab with (after the deviation-15 rerun; the v7 rows are recalibrated for comparison only):
  node scripts/colab-run.mjs projects/project-1/j1-env-modeling-assessment notebooks/09_exploratory_recalibration_CPU.ipynb \
       --input outputs/hindcast_NO2_rows.csv.gz --input outputs/hindcast_PM10_rows.csv.gz --input outputs/hindcast_PM2.5_rows.csv.gz \
       --input outputs/v7_preregistered/hindcast_NO2_rows.csv.gz --input outputs/v7_preregistered/hindcast_PM10_rows.csv.gz \
       --input outputs/v7_preregistered/hindcast_PM2.5_rows.csv.gz
"""
import json
from pathlib import Path

CELLS = [
    # 0. runtime and inputs
    '''# RUNTIME: CPU (standard) - do NOT pick a GPU
# EXPLORATORY, POST-HOC recalibration (deviation 16). Reads stored hindcast rows only; nothing is refitted.
import os, platform
import numpy as np, pandas as pd
ROOT = os.environ.get("PAPER_ROOT", ".")
IN = os.path.join(ROOT, "prior_outputs")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
print("python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__)
POLS = ["NO2", "PM10", "PM2.5"]
VERSIONS = {"v8_corrected": IN, "v7_preregistered": os.path.join(IN, "v7_preregistered")}   # IN holds the corrected run's rows
rows = {}
for ver, folder in VERSIONS.items():
    files = [os.path.join(folder, f"hindcast_{p}_rows.csv.gz") for p in POLS]
    if all(os.path.exists(f) for f in files):
        rows[ver] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(ver, len(rows[ver]), "rows")
    else:
        print(ver, "not supplied; skipped")
assert "v8_corrected" in rows, "the corrected run's hindcast rows are required"''',

    # 1. functions
    '''# split-conformal factors and their evaluation
Z90, Z80 = 1.6449, 1.2816
CAL_LAST_TARGET, EVAL_FIRST_ORIGIN = 2020, 2020
lg = lambda x: np.log(np.clip(x, 0, None) + 0.1)

def prepare(r):
    r = r.dropna(subset=["obs_mean", "pred_mean_med", "pi90_lo", "pi90_hi"]).copy()
    r["centre"] = lg(r["pred_mean_med"])
    r["sd"] = (lg(r["pi90_hi"]) - lg(r["pi90_lo"])) / (2 * Z90)
    r = r[r["sd"] > 0].copy()
    r["z"] = (lg(r["obs_mean"]) - r["centre"]).abs() / r["sd"]
    return r

def factors(r):
    cal = r[(r["set"] == "hindcast") & (r["target"] <= CAL_LAST_TARGET)]
    out = []
    for (pol, sc), g in cal.groupby(["pollutant", "scenario"]):
        est = {h: (len(gh), np.quantile(gh["z"], 0.90) / Z90, np.quantile(gh["z"], 0.80) / Z80)
               for h, gh in g.groupby("h")}
        hs = np.array(sorted(est))
        fit90 = np.polyfit(np.log(hs), np.log([est[h][1] for h in hs]), 1) if len(hs) >= 2 else None
        fit80 = np.polyfit(np.log(hs), np.log([est[h][2] for h in hs]), 1) if len(hs) >= 2 else None
        for h in range(1, 7):
            if h in est:
                out.append({"pollutant": pol, "scenario": sc, "h": h, "n_cal": est[h][0],
                            "k90": est[h][1], "k80": est[h][2], "source": "estimated"})
            elif fit90 is not None:
                out.append({"pollutant": pol, "scenario": sc, "h": h, "n_cal": 0,
                            "k90": float(np.exp(np.polyval(fit90, np.log(h)))),
                            "k80": float(np.exp(np.polyval(fit80, np.log(h)))), "source": "extrapolated in log h"})
    return pd.DataFrame(out)

def evaluate(r, k):
    ev = r[r["origin"] >= EVAL_FIRST_ORIGIN].merge(k, on=["pollutant", "scenario", "h"], how="inner")
    obs = lg(ev["obs_mean"])
    res = []
    for lvl, z, kc, lo, hi in [(90, Z90, "k90", "pi90_lo", "pi90_hi"), (80, Z80, "k80", "pi80_lo", "pi80_hi")]:
        ev[f"in{lvl}_raw"] = (ev["obs_mean"] >= ev[lo]) & (ev["obs_mean"] <= ev[hi])
        ev[f"in{lvl}_cal"] = (obs - ev["centre"]).abs() <= z * ev[kc] * ev["sd"]
    for (pol, sc, h, st), g in ev.groupby(["pollutant", "scenario", "h", "set"]):
        res.append({"pollutant": pol, "scenario": sc, "h": h, "set": st, "n_eval": len(g),
                    "k90": g["k90"].iloc[0], "k_source": g["source"].iloc[0],
                    "coverage90_raw": g["in90_raw"].mean(), "coverage90_recal": g["in90_cal"].mean(),
                    "coverage80_raw": g["in80_raw"].mean(), "coverage80_recal": g["in80_cal"].mean(),
                    "median_width90_raw": (g["pi90_hi"] - g["pi90_lo"]).median(),
                    "median_width90_recal": ((np.exp(g["centre"] + Z90 * g["k90"] * g["sd"])
                                              - np.exp(g["centre"] - Z90 * g["k90"] * g["sd"]))).median()})
    return pd.DataFrame(res), ev[["pollutant", "scenario", "set", "station", "h", "in90_cal"]]''',

    # 2. run
    '''# factors and evaluation for each supplied version
kt, et, rowlevel = [], [], []
for ver, r in rows.items():
    pr = prepare(r)
    k = factors(pr)
    e, ev_rows = evaluate(pr, k)
    rowlevel.append(ev_rows.assign(rows_version=ver))
    k.insert(0, "rows_version", ver); e.insert(0, "rows_version", ver)
    kt.append(k); et.append(e)
kt, et = pd.concat(kt, ignore_index=True), pd.concat(et, ignore_index=True)
kt.to_csv(os.path.join(OUT, "exploratory_recal_factors.csv"), index=False)
et.to_csv(os.path.join(OUT, "exploratory_recal_coverage.csv"), index=False)
print(kt.round(3).to_string(index=False))
print(et[et["set"] == "hindcast"][["rows_version", "pollutant", "scenario", "h", "n_eval", "coverage90_raw", "coverage90_recal"]].round(3).to_string(index=False))''',

    # 3. pooled summary
    '''# pooled over horizons (station-origin rows weighted equally), hindcast and 2025 check separately
summ = (et.assign(w90r=et["coverage90_raw"] * et["n_eval"], w90c=et["coverage90_recal"] * et["n_eval"],
                  w80r=et["coverage80_raw"] * et["n_eval"], w80c=et["coverage80_recal"] * et["n_eval"])
          .groupby(["rows_version", "pollutant", "scenario", "set"], as_index=False)
          [["n_eval", "w90r", "w90c", "w80r", "w80c"]].sum())
for a, b in [("w90r", "coverage90_raw"), ("w90c", "coverage90_recal"), ("w80r", "coverage80_raw"), ("w80c", "coverage80_recal")]:
    summ[b] = summ[a] / summ["n_eval"]
summ = summ.drop(columns=["w90r", "w90c", "w80r", "w80c"])
# uncertainty of the widened 90% coverage: bootstrap over stations (999 replicates). Only four evaluation years
# exist, so dependence between target years is not captured (stated in the paper).
rl = pd.concat(rowlevel, ignore_index=True)
rng = np.random.default_rng(20261105)
ci = []
for keys, g in rl.groupby(["rows_version", "pollutant", "scenario", "set"]):
    per = g.groupby("station")["in90_cal"].agg(["sum", "count"])
    s_, c_ = per["sum"].to_numpy(), per["count"].to_numpy()
    idx = rng.integers(0, len(per), size=(999, len(per)))
    boot = s_[idx].sum(axis=1) / c_[idx].sum(axis=1)
    ci.append(dict(zip(["rows_version", "pollutant", "scenario", "set"], keys),
                   cov90_lo=float(np.quantile(boot, 0.05)), cov90_hi=float(np.quantile(boot, 0.95))))
summ = summ.merge(pd.DataFrame(ci), on=["rows_version", "pollutant", "scenario", "set"], how="left")
summ.to_csv(os.path.join(OUT, "exploratory_recal_summary.csv"), index=False)
print(summ.round(3).to_string(index=False))''',

    # 4. manifest
    '''# MANIFEST rows (exact file names; every file labelled EXPLORATORY)
nb = "09_exploratory_recalibration_CPU"
lab = "EXPLORATORY recalibration, not validated (post-hoc, deviation 16)"
rows_md = ["| file | notebook | cell | description |", "|---|---|---|---|",
    f"| exploratory_recal_factors.csv | {nb} | 2 | {lab}: split-conformal widening factors k80/k90 per rows version x pollutant x scenario x horizon, from targets <= 2020; h >= 4 extrapolated in log h |",
    f"| exploratory_recal_coverage.csv | {nb} | 2 | {lab}: 80%/90% coverage and median 90% width, stored vs widened intervals, on origins >= 2020 and the 2025 check |",
    f"| exploratory_recal_summary.csv | {nb} | 3 | {lab}: the same coverage pooled over horizons, with a 90% station-bootstrap interval for the widened 90% coverage |"]
open(os.path.join(OUT, "MANIFEST.md"), "w").write("\\n".join(rows_md) + "\\n")
print("manifest written")''',
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "cells": [{"cell_type": "code", "id": f"c{i}", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": src} for i, src in enumerate(CELLS)]}
    out = Path(__file__).parent / "09_exploratory_recalibration_CPU.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
