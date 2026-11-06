# How likely is attainment of the 2030 EU air-quality limit values?

Code and derived results for the paper

> Md Samiul Alom, Md Istiak Ahamed and Rukaiya Taha. *How likely is attainment of the 2030 EU air-quality limit values? Weather-resampled station probabilities and a pre-specified hindcast test.* Manuscript, 2026. The journal reference will be added here once the paper is published.

Directive (EU) 2024/2881 lowers the limit values for PM2.5, PM10 and NO2 from 2030. For every EU-27 monitoring station with a long enough record, we estimate the probability that the station meets both the annual and the daily 2030 limit values at its 2024 level. To do this we combine a station model with whole weather years resampled from 2005–2024 and with resampled residuals. We then test those probabilities in a rolling-origin hindcast (forecast origins 2017–2023, everything refitted at each origin) and on 2025 data.

## What is in this repository

```
notebooks/   analysis code (Python modules) and the Google Colab notebooks that run it
outputs/     derived results: station-level estimates, hindcast forecasts and scores, tables behind the figures
figures/     the three method diagrams (draw.io sources and PDFs)
data/        empty; the raw data are downloaded by the first notebook
```

The main modules are:

- `p1lib.py`: the methods. It holds the inclusion rules, the station model (yearly levels plus a LightGBM weather effect, fitted by backfitting), weather and residual resampling, the estimands E1–E5, the hindcast and the scores.
- `p1run.py`: runs everything in batches, with checkpoints so long runs can resume.
- `p1fig.py`: the result figures.
- `p1mc.py`: the Monte Carlo convergence check of the hindcast intervals.

The notebooks `01`–`07` are the pre-specified analysis, in order. They download the data, build the analysis set, run a pilot, then the main estimates for each pollutant, the hindcast, the sensitivity analyses and the results tables. The notebooks `08`–`12` are the checks added after the pre-specified validation failed. They are labelled post-hoc or exploratory, as in the paper. The `make_*.py` scripts write these notebooks, and `make_extra_figures.py` draws the extra figures from `outputs/`.

`outputs/` holds the corrected run reported in the paper. `outputs/v7_preregistered/` holds the original pre-specified run, kept unchanged for comparison. `outputs/MANIFEST.md` lists every file and the notebook that wrote it.

## Data

Both data sources are public, and neither is copied into this repository:

- **Air quality:** validated E1a data reported by EU member states to the European Environment Agency, from the [EEA Air Quality Download Service](https://eeadmz1-downloads-webapp.azurewebsites.net/).
- **Weather:** daily values from the NOAA [Integrated Surface Database (ISD-Lite)](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database).

Notebook `01_download_CPU.ipynb` (or stage 1 of `run_local.py`) downloads both into `data/`.

## How to run it

**On Google Colab (how the paper's results were produced).** Copy this folder to your Google Drive as `MyDrive/eu-air-quality-2030-attainment`, then open the notebooks in order and run them. Each notebook mounts Drive, records the package versions in `outputs/environment.jsonl` and saves its results to `outputs/`. A standard CPU runtime is enough. The main and hindcast stages take several hours each and resume from their checkpoints if a session ends.

**On your own computer.**

```bash
pip install -r requirements.txt
cd notebooks
python run_local.py ..          # all stages 1-7
python run_local.py .. 4 5      # only stages 4 to 5
```

The random seed is fixed (`20261104`), so a rerun with the same package versions gives the same numbers. The versions we used are in `requirements.txt` and in `outputs/environment.jsonl`.

## Method diagrams

`figures/` contains the draw.io sources of the paper's three method diagrams: the workflow, the inclusion flow and the hindcast design. Open them in [draw.io](https://www.drawio.com/) (free) to edit them.

## Licence

The code is released under the MIT licence (see `LICENSE`). The derived results in `outputs/` may be reused with a citation of the paper. The underlying EEA and NOAA data remain under their providers' terms.

## Contact

Md Samiul Alom, Miyan Research Institute, International University of Business Agriculture and Technology (IUBAT), Dhaka, Bangladesh. Email: samiulalom090@gmail.com
