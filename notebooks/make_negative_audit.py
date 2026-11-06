"""Write notebooks/12_negative_values_audit_CPU.ipynb.

POST-HOC audit (deviation 18a in Supplementary Section S5): how many daily means in the
analysis set are negative, how large they are, and how setting them to zero changes the observed annual means and the
observed attainment of the annual limit value. Daily exceedances are unaffected, because a negative value can never
exceed a limit. Reads the EEA daily means (as downloaded, unclipped) and the station manifest; no model is fitted.

Inputs are copied from the analysis folder on Drive into data/derived/ (public EEA data), then:
  node scripts/colab-run.mjs projects/project-1/j1-env-modeling-assessment notebooks/12_negative_values_audit_CPU.ipynb \
       --input data/derived/aq_daily_analysis_set.parquet --input data/derived/manifest.parquet
"""
import json
from pathlib import Path

CELLS = [
    '''# RUNTIME: CPU (standard) - do NOT pick a GPU
# POST-HOC audit of negative daily concentrations (deviation 18a). No model is fitted.
import os, sys, glob
import numpy as np, pandas as pd
ROOT = os.environ.get("PAPER_ROOT", ".")
sys.path.insert(0, ROOT)
import p1lib as L
IN = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
man = pd.read_parquet(os.path.join(IN, "manifest.parquet"))
inc = man[man["excluded_reason"] == ""][["sp", "pollutant"]]
code = {v: k for k, v in L.POLLUTANTS.items()}
# daily means of the analysis-set series, 2013-2025, as downloaded (unclipped); subset made from data/aq_daily/??.parquet
d = pd.read_parquet(os.path.join(IN, "aq_daily_analysis_set.parquet"))
d["pollutant"] = d["Pollutant"].map(code)
d = d.merge(inc, on=["sp", "pollutant"], how="inner")          # the analysis set only
d["date"] = pd.to_datetime(d["date"])
d = d[(d["date"].dt.year >= 2013) & (d["date"].dt.year <= 2025)].dropna(subset=["value"])
print(len(d), "daily means in the analysis set, 2013-2025")''',

    '''# counts and sizes of negative daily means, by pollutant
rows = []
for pol, g in d.groupby("pollutant"):
    neg = g.loc[g["value"] < 0, "value"]
    rows.append({"pollutant": pol, "n_daily": len(g), "n_negative": len(neg), "share_negative": len(neg) / len(g),
                 "n_series_with_negative": int(g.loc[g["value"] < 0, "sp"].nunique()), "n_series": int(g["sp"].nunique()),
                 "median_negative": float(neg.median()) if len(neg) else np.nan,
                 "min_negative": float(neg.min()) if len(neg) else np.nan})
neg_summary = pd.DataFrame(rows)
print(neg_summary.to_string(index=False))''',

    '''# effect on observed annual means and annual attainment (station-years with at least 85% valid days)
d["year"] = d["date"].dt.year
d["clipped"] = d["value"].clip(lower=0)
sy = d.groupby(["pollutant", "sp", "year"]).agg(n=("value", "size"), raw=("value", "mean"), clipped=("clipped", "mean")).reset_index()
sy["days"] = np.where(sy["year"] % 4 == 0, 366, 365)
sy = sy[sy["n"] >= 0.85 * sy["days"]]
rows = []
for pol, g in sy.groupby("pollutant"):
    lim = L.LIMITS[pol]["annual"]
    diff = g["clipped"] - g["raw"]
    changed = ((g["raw"] <= lim) != (g["clipped"] <= lim)).sum()
    rows.append({"pollutant": pol, "n_station_years": len(g), "n_station_years_changed": int((diff > 0).sum()),
                 "max_increase_annual_mean": float(diff.max()), "median_increase_where_changed": float(diff[diff > 0].median()) if (diff > 0).any() else 0.0,
                 "n_annual_attainment_changed": int(changed)})
eff = pd.DataFrame(rows)
audit = neg_summary.merge(eff, on="pollutant")
audit.to_csv(os.path.join(OUT, "posthoc_negative_values.csv"), index=False)
print(audit.to_string(index=False))''',

    '''# MANIFEST rows
nb = "12_negative_values_audit_CPU"
rows_md = ["| file | notebook | cell | description |", "|---|---|---|---|",
    f"| posthoc_negative_values.csv | {nb} | 2 | POST-HOC audit (deviation 18a): negative daily means in the analysis set by pollutant, and the effect of setting them to zero on observed annual means and annual attainment |"]
open(os.path.join(OUT, "MANIFEST.md"), "w").write("\\n".join(rows_md) + "\\n")
print("manifest written")''',
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "cells": [{"cell_type": "code", "id": f"c{i}", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": src} for i, src in enumerate(CELLS)]}
    out = Path(__file__).parent / "12_negative_values_audit_CPU.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
