"""
p1mc — POST-HOC Monte Carlo convergence check of the hindcast predictive intervals (deviation 19 in Supplementary Section S5). Kept in its own module so the main analysis code is unchanged.

At a FIXED fit per series and origin, the 90% interval of the annual mean is computed three times:
  * default: as in the main hindcast (10 level draws x 25 paths with the level frozen; 40 trend draws x 5 paths);
  * default with an independent random stream (to measure Monte Carlo noise at the default size);
  * large: five times as many outer draws (50 level draws; 200 trend draws), same paths per draw.
Coverage and interval width are then compared. Horizons 1 and 3; origins 2017-2023.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import p1lib as L


def _interval(res: list) -> tuple[float, float]:
    means = np.concatenate([o["mean"] for o in res])
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def mc_check(key: dict, series: pd.DataFrame, met: pd.DataFrame, hp=None, origins=range(2017, 2024),
             horizons=(1, 3), n_res=50, n_trend_res=5, seed=L.SEED) -> list[dict]:
    pol = key["pollutant"]
    d = L.prepare_series(series, met)
    rows = []
    for o in origins:
        tr = L.training_rows(d, o)
        cov = tr.groupby("year").size() / 365.0
        if (cov >= L.TRAIN_DAY_SHARE).sum() < 4 or o not in cov.index:
            continue
        model = L.fit_series(tr, hp=hp, seed=seed)                      # one fixed fit per series and origin
        wy = L.weather_year_frames(met, list(range(o - 19, o + 1)))
        if len(wy) < 10:
            continue
        for h in horizons:
            y = o + h
            if y > 2024:
                continue
            ob = L.observed_outcome(series, y, pol)
            mask = ob.pop("_mask")
            if not np.isfinite(ob.get("obs_mean", np.nan)):
                continue
            rec = {**key, "origin": o, "h": h, "target": y, "obs_mean": ob["obs_mean"]}
            for label, stream, n_lv, n_td in [("default", 1, 10, 40), ("default_rep", 2, 10, 40), ("large", 3, 50, 200)]:
                rng = np.random.default_rng([seed, L.key_seed(key), o, h, stream])
                sims = L.base_sims(model, wy, n_res, rng)
                lv = model.a[o] + model.se[o] * rng.standard_normal(n_lv)
                lo, hi = _interval([r for v in lv for r in L.simulate(sims, v, pol, n_paths=n_res // 2, mask=mask)])
                rec.update({f"frozen_{label}_lo": lo, f"frozen_{label}_hi": hi})
                td = L.trend_draws(model.a, model.se, y, n_td, rng, method="wls")
                if not np.isnan(td).all():
                    lo, hi = _interval([r for t in td for r in L.simulate(sims, t, pol, n_paths=n_trend_res, mask=mask)])
                    rec.update({f"trend_{label}_lo": lo, f"trend_{label}_hi": hi})
            rows.append(rec)
    return rows


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Coverage of the observed annual mean and median interval width, by pollutant, scenario, horizon and draw size."""
    out = []
    for sc in ["frozen", "trend"]:
        for (pol, h), g in rows.groupby(["pollutant", "h"]):
            rec = {"pollutant": pol, "scenario": sc, "h": h}
            for lab in ["default", "default_rep", "large"]:
                lo, hi = g.get(f"{sc}_{lab}_lo"), g.get(f"{sc}_{lab}_hi")
                if lo is None:
                    continue
                ok = lo.notna() & hi.notna()
                rec[f"n_{lab}"] = int(ok.sum())
                rec[f"coverage90_{lab}"] = float(((g.obs_mean >= lo) & (g.obs_mean <= hi))[ok].mean())
                rec[f"median_width_{lab}"] = float((hi - lo)[ok].median())
            rec["median_abs_diff_hi_default_vs_rep"] = float((g[f"{sc}_default_hi"] - g[f"{sc}_default_rep_hi"]).abs().median())
            out.append(rec)
    return pd.DataFrame(out)
