"""Write notebooks/10_exclusion_audit_CPU.ipynb.

POST-HOC, descriptive (added during internal review before submission): what the "station moved or method changed" rule
(plan §2 rule 4) actually caught. Reads the station manifest and the EEA sampling-point metadata only; no model.

Inputs are copied from the analysis folder on Drive into data/derived/ (public EEA metadata), then:
  node scripts/colab-run.mjs projects/project-1/j1-env-modeling-assessment notebooks/10_exclusion_audit_CPU.ipynb \
       --input data/derived/manifest.parquet --input data/derived/sampling_points.parquet
"""
import json
from pathlib import Path

CELLS = [
    '''# RUNTIME: CPU (standard) - do NOT pick a GPU
# POST-HOC, descriptive: what the relocation / method-change exclusion rule caught. No model is fitted.
import os
import pandas as pd
ROOT = os.environ.get("PAPER_ROOT", ".")
IN = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
man = pd.read_parquet(os.path.join(IN, "manifest.parquet"))
sp = pd.read_parquet(os.path.join(IN, "sampling_points.parquet"))
x = man[man["excluded_reason"] == "station moved or method changed"]
print(len(man), "series in the manifest;", len(x), "excluded by the relocation / method-change rule")''',

    '''# E1: counts by reason, and whether a break year and a meteorological match exist
summ = x.groupby("pollutant").apply(lambda g: pd.Series({
    "n_excluded": len(g), "moved_only": int((g["moved"] & ~g["method_change"]).sum()),
    "method_change_only": int((~g["moved"] & g["method_change"]).sum()), "both": int((g["moved"] & g["method_change"]).sum()),
    "with_break_year": int(g["break_year"].notna().sum()), "with_met_match": int(g["isd"].notna().sum())})).reset_index()
summ.to_csv(os.path.join(OUT, "posthoc_exclusions_summary.csv"), index=False)
print(summ.to_string(index=False))''',

    '''# E2: the sequence of measurement-method codes in the EEA metadata, for series excluded by a method change
s = sp[sp["sp"].isin(x.loc[x["method_change"], "sp"])].sort_values("Year")
seq = (s.groupby(["pollutant", "sp"])["mmethod"].apply(lambda v: " -> ".join(dict.fromkeys(v.dropna().astype(str))))
         .rename("method_sequence").reset_index())
tab = seq.groupby(["pollutant", "method_sequence"]).size().rename("n").reset_index()
tab["share_of_pollutant"] = tab["n"] / tab.groupby("pollutant")["n"].transform("sum")
tab = tab.sort_values(["pollutant", "n"], ascending=[True, False])
tab.to_csv(os.path.join(OUT, "posthoc_exclusions_methods.csv"), index=False)
print(tab.groupby("pollutant").head(8).to_string(index=False))''',

    '''# MANIFEST rows
nb = "10_exclusion_audit_CPU"
rows_md = ["| file | notebook | cell | description |", "|---|---|---|---|",
    f"| posthoc_exclusions_summary.csv | {nb} | 1 | POST-HOC, descriptive: series excluded by the relocation / method-change rule, by reason, with break year and meteorological match |",
    f"| posthoc_exclusions_methods.csv | {nb} | 2 | POST-HOC, descriptive: sequences of EEA measurement-method codes among series excluded for a method change |"]
open(os.path.join(OUT, "MANIFEST.md"), "w").write("\\n".join(rows_md) + "\\n")
print("manifest written")''',
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "cells": [{"cell_type": "code", "id": f"c{i}", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": src} for i, src in enumerate(CELLS)]}
    out = Path(__file__).parent / "10_exclusion_audit_CPU.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
