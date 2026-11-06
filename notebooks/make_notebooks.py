"""Write the Colab notebooks for this project from short cell lists (run: python make_notebooks.py).

Every notebook: states its runtime in the first cell, mounts Google Drive, records package versions,
does one stage of the analysis, and releases the runtime in its last cell.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
DRIVE_ROOT = "/content/drive/MyDrive/eu-air-quality-2030-attainment"


def setup(name: str, extra_pip: str = "") -> list[str]:
    pip = f"!pip -q install {extra_pip}\n" if extra_pip else ""
    return [
        f"# {name}\n# RUNTIME: CPU (standard) — do NOT pick a GPU or High-RAM.\n"
        "# Project: How likely is compliance? (Environmental Modeling & Assessment).\n"
        "# Data and outputs live on Google Drive; nothing is stored in the Colab VM between runs.",
        pip + "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "import sys\n"
        f"ROOT = '{DRIVE_ROOT}'\n"
        "sys.path.insert(0, ROOT + '/notebooks')\n"
        "import p1lib as L, p1run as R\n"
        "P = L.Paths(ROOT)\n"
        f"R.record_environment(P, '{name}')",
    ]


RELEASE = "from google.colab import runtime\nruntime.unassign()  # free the Colab runtime when finished"

NOTEBOOKS = {
    "01_download_CPU": [
        "# Downloads EEA E1a data for NO2, PM10 and PM2.5 in the EU-27 (2013-2025) and keeps daily means only.\n"
        "# Resumable: countries already on Drive are skipped.\n"
        "log = R.nb01_download(P, threads=16)\nlog",
    ],
    "02_build_CPU": [
        "# Station manifest, inclusion rules, meteorology (NOAA ISD-Lite) matching, data-flow table, missingness audit.\n"
        "flow = R.nb02_build(P, threads=16)\nflow",
    ],
    "03_pilot_CPU": [
        "# 1% pilot: choose global LightGBM settings (time-blocked) and time the full runs.\n"
        "timing, main_pilot, hind_pilot = R.nb03_pilot(P)\ntiming",
    ],
}
for pol, slug in [("NO2", "no2"), ("PM10", "pm10"), ("PM2.5", "pm25")]:
    NOTEBOOKS[f"04_main_{slug}_CPU"] = [
        f"# Main estimates (E1, E2, E3, E5 and the re-use sensitivities) for {pol}. Checkpointed every 50 series.\n"
        f"res = R.nb04_main(P, pollutant='{pol}')\nres['status'].value_counts()",
    ]
    NOTEBOOKS[f"05_hindcast_{slug}_CPU"] = [
        f"# Rolling-origin hindcast (origins 2017-2023, horizons 1-6, plus 2025 check) for {pol}.\n"
        f"res = R.nb05_hindcast(P, pollutant='{pol}')\nlen(res)",
    ]
NOTEBOOKS["06_sensitivity_CPU"] = [
    "# GAM structural alternative on the pre-set 10% subsample.\nout = R.nb06_sensitivity(P)\n{k: len(v) for k, v in out.items()}",
]
NOTEBOOKS["07_results_CPU"] = [
    "# Combine shards, score the hindcast, check the pre-specified success criteria, write summaries and figures.\n"
    "import pandas as pd, p1fig\n"
    "main = pd.concat([pd.read_csv(f) for f in sorted(P.outputs.glob('main_*_station_results.csv'))], ignore_index=True)\n"
    "rows = pd.concat([pd.read_csv(f) for f in sorted(P.outputs.glob('hindcast_*_rows.csv.gz'))], ignore_index=True)\n"
    "scores, coverage = R.score_tables(P, rows)\n"
    "criteria = R.success_check(P, rows, coverage)\n"
    "summary = R.summaries(P, main)\n"
    "sens = R.sensitivity_summary(P, main)\n"
    "p1fig.make_all(P, main, scores)\n"
    "criteria",
]
EXTRA_PIP = {"06_sensitivity_CPU": "pygam"}


def cell(src: str, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


for name, body in NOTEBOOKS.items():
    first, second = setup(name, EXTRA_PIP.get(name, ""))
    cells = [cell(first, "markdown")] + [cell(second)] + [cell(b) for b in body] + [cell(RELEASE)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "colab": {"provenance": []}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / f"{name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", name)
