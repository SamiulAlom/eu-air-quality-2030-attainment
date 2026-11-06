"""Write notebooks/11_mc_convergence_CPU.ipynb.

POST-HOC Monte Carlo convergence check of the hindcast predictive intervals (deviation 19 in Supplementary Section S5). Uses the pre-set 2% subsample of the Monte Carlo-error stage (73 series) and p1mc.mc_check.

The subsample's inputs are written by the snippet in the docstring of p1mc.py's caller (data/derived/mc_*.parquet,
public EEA and NOAA data). Run on Colab with:
  node scripts/colab-run.mjs projects/project-1/j1-env-modeling-assessment notebooks/11_mc_convergence_CPU.ipynb \
       --input data/derived/mc_manifest.parquet --input data/derived/mc_daily.parquet --input data/derived/mc_met.parquet \
       --input outputs/hyperparams.json
"""
import json
from pathlib import Path

CELLS = [
    '''# RUNTIME: CPU (standard) - do NOT pick a GPU
# POST-HOC Monte Carlo convergence check of the hindcast intervals (deviation 19). Fixed fits; no new data.
import os, sys, json, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = os.environ.get("PAPER_ROOT", ".")
sys.path.insert(0, ROOT)                     # p1lib.py and p1mc.py are uploaded next to the notebook
import p1lib as L, p1mc
IN = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
man = pd.read_parquet(os.path.join(IN, "mc_manifest.parquet"))
daily = pd.read_parquet(os.path.join(IN, "mc_daily.parquet"))
met = pd.read_parquet(os.path.join(IN, "mc_met.parquet"))
hp = json.load(open(os.path.join(ROOT, "prior_outputs", "hyperparams.json")))
by_sp = {k: g[["date", "value"]].sort_values("date") for k, g in daily.groupby("sp")}
by_isd = {k: g.drop(columns=["isd"]) for k, g in met.groupby("isd")}
items = [({"station": r.station, "pollutant": r.pollutant, "cc": r.cc, "stype": r.stype, "area": r.area},
          by_sp[r.sp], by_isd[r.isd]) for r in man.itertuples() if r.sp in by_sp and r.isd in by_isd]
print(len(items), "series;", os.cpu_count(), "cores")''',

    '''# run (all cores); each series is independent
def one(k, s, m):
    try:
        return p1mc.mc_check(k, s, m, hp=hp)
    except Exception as e:
        return [{**k, "status": f"error: {type(e).__name__}: {e}"}]
t = time.time()
res = Parallel(n_jobs=os.cpu_count(), backend="loky")(delayed(one)(k, s, m) for k, s, m in items)
rows = pd.DataFrame([r for rr in res for r in rr])
print(f"{len(rows)} rows in {(time.time() - t) / 60:.1f} min")
rows.to_csv(os.path.join(OUT, "posthoc_mc_convergence_rows.csv"), index=False)''',

    '''# summary: coverage and median width of the 90% interval at default and 5x outer draws
ok = rows[rows["status"].isna()] if "status" in rows else rows
summ = p1mc.summarise(ok)
summ.to_csv(os.path.join(OUT, "posthoc_mc_convergence.csv"), index=False)
print(summ.round(3).to_string(index=False))''',

    '''# MANIFEST rows
nb = "11_mc_convergence_CPU"
rows_md = ["| file | notebook | cell | description |", "|---|---|---|---|",
    f"| posthoc_mc_convergence_rows.csv | {nb} | 1 | POST-HOC: 90% intervals of the annual mean at a fixed fit with default and 5x outer draws (2% subsample, h = 1 and 3) |",
    f"| posthoc_mc_convergence.csv | {nb} | 2 | POST-HOC: coverage and median width of those intervals by pollutant, scenario, horizon and draw size |"]
open(os.path.join(OUT, "MANIFEST.md"), "w").write("\\n".join(rows_md) + "\\n")
print("manifest written")''',
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "cells": [{"cell_type": "code", "id": f"c{i}", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": src} for i, src in enumerate(CELLS)]}
    out = Path(__file__).parent / "11_mc_convergence_CPU.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
