"""
p1run — orchestration for the P1 notebooks: building the analysis set, batch runs with Drive checkpoints,
scoring tables, sensitivity analyses and figures. All numbers written here go to outputs/ with a
MANIFEST.md row. See p1lib.py for the methods.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import p1lib as L


# --------------------------------------------------------------------------------------------
# Environment record (rule 9 of colab-builder: seeds, versions, data versions)
# --------------------------------------------------------------------------------------------
def record_environment(P: L.Paths, notebook: str):
    import importlib
    v = {"notebook": notebook, "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
         "python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
         "seed": L.SEED}
    for pkg in ["numpy", "pandas", "lightgbm", "scipy", "pyarrow", "requests", "matplotlib", "pygam", "joblib"]:
        try:
            v[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            v[pkg] = None
    f = P.outputs / "environment.jsonl"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(v) + "\n")
    L.manifest_add(P, "environment.jsonl", "all", "first cell", "package versions, CPU count, seed per notebook run")
    print(json.dumps(v, indent=1))


# --------------------------------------------------------------------------------------------
# 01 download
# --------------------------------------------------------------------------------------------
def nb01_download(P: L.Paths, threads=16):
    meta = L.download_metadata(P)
    recs, failures = [], []
    for cc in L.EU27:
        t = time.time()
        try:
            r = L.download_eea_country(P, cc, threads=threads)
        except Exception as e:
            r = {"country": cc, "status": f"error: {e}"}
            failures.append(cc)
        r["seconds"] = round(time.time() - t)
        recs.append(r)
        L.log(f"{cc}: {r}")
    rows = []
    for cc in L.EU27:
        j = P.daily / f"{cc}_download.json"
        if j.exists():
            d = json.loads(j.read_text())
            rows.append({k: d.get(k) for k in ["country", "files", "failed", "unavailable_404", "rows", "downloaded"]})
    pd.DataFrame(rows).to_csv(P.outputs / "download_log.csv", index=False)
    if failures:
        raise RuntimeError(f"download incomplete for {failures}; rerun 01 before building the analysis set")
    L.manifest_add(P, "download_log.csv", "01_download_CPU", "download", "EEA E1a files per country, failed files, daily rows, download time")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 02 build: manifest, met matching, data flow, missingness audit
# --------------------------------------------------------------------------------------------
def check_downloads_complete(P: L.Paths):
    """Refuse to build the analysis set from incomplete EEA downloads (every EU-27 country needs its
    data file and a completion record with zero failed files)."""
    bad = []
    for cc in L.EU27:
        rec = P.daily / f"{cc}_download.json"
        if not (P.daily / f"{cc}.parquet").exists() or not rec.exists() or json.loads(rec.read_text()).get("failed", 1) != 0:
            bad.append(cc)
    if bad or list(P.daily.glob("*_INCOMPLETE.json")):
        raise RuntimeError(f"EEA download incomplete for {bad}; run stage 1 first")
    pd.DataFrame({"country": L.EU27, "complete": True}).to_csv(P.outputs / "download_completeness.csv", index=False)
    L.manifest_add(P, "download_completeness.csv", "02_build_CPU", "entry check", "every EU-27 country has a complete EEA download (completion record, 0 failed files)")


def nb02_build(P: L.Paths, threads=16):
    check_downloads_complete(P)
    meta = L.download_metadata(P)
    daily = L.load_daily(P)
    spm = L.station_metadata(meta)
    man = L.build_manifest(daily, spm)
    inc = man[(man["excluded_reason"] == "") | man["history_only"]]
    st = inc.drop_duplicates("station")[["station", "lat", "lon", "alt"]]
    hist = L.isd_history(P)
    cands = L.isd_candidates(st, hist, max_km=50.0, max_dz=500.0, k=3)
    ids = sorted(cands["isd"].unique())
    L.log(f"ISD candidates: {len(ids)} stations for {len(st)} AQ stations")
    failed = ids
    for attempt in range(3):                      # retry stations whose download failed
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(lambda i: L._safe(L.download_isd_station, P, i, range(1995, 2025)), failed))
        failed = [i for i, r in zip(failed, res) if isinstance(r, Exception)]
        if not failed:
            break
        time.sleep(20 * (attempt + 1))
    if failed:
        (P.work / "isd_failed.json").write_text(json.dumps(failed))
        raise RuntimeError(f"{len(failed)} ISD stations failed to download after 3 attempts; eligibility not built")
    mets = [pd.read_parquet(P.isd_daily / f"{i}.parquet") for i in ids if (P.isd_daily / f"{i}.parquet").exists()]
    met = pd.concat([m for m in mets if len(m)], ignore_index=True)
    mc = L.met_coverage(met)
    mm = L.match_met(man, cands, mc, L.MAIN_POOL)
    no_met = (mm["excluded_reason"] == "") & mm["isd"].isna()
    mm.loc[no_met, "excluded_reason"] = "no qualifying ISD station within 50 km / 500 m"
    mm["met_within_25km"] = mm["dist_km"] <= 25
    mm.to_parquet(P.work / "manifest.parquet", index=False)
    cands.to_parquet(P.work / "isd_candidates.parquet", index=False)
    met.to_parquet(P.work / "met_all.parquet", index=False)

    # data-flow table (§3.5 supplement S1)
    flow = []
    for pol in L.POLLUTANTS:
        s = mm[mm["pollutant"] == pol]
        n0 = len(s)
        flow.append({"pollutant": pol, "step": "station x pollutant series with E1a data", "n": n0})
        remaining = s
        for reason in ["no metadata", "not EU-27", "not a fixed measurement",
                       "fewer than 8 good training years or 2023/2024 missing", "station moved or method changed",
                       "no qualifying ISD station within 50 km / 500 m"]:
            k = int((remaining["excluded_reason"] == reason).sum())
            remaining = remaining[remaining["excluded_reason"] != reason]
            flow.append({"pollutant": pol, "step": f"excluded: {reason}", "n": k})
        flow.append({"pollutant": pol, "step": "included", "n": int((s["excluded_reason"] == "").sum())})
    flow = pd.DataFrame(flow)
    flow.to_csv(P.outputs / "data_flow.csv", index=False)
    L.manifest_add(P, "data_flow.csv", "02_build_CPU", "data flow", "series counts after each inclusion rule, by pollutant")

    pub = mm.drop(columns=["good_years"]).copy()
    pub.to_csv(P.outputs / "station_manifest.csv", index=False)
    L.manifest_add(P, "station_manifest.csv", "02_build_CPU", "manifest", "every station x pollutant series with sampling point, metadata, met match and exclusion reason")

    # missingness audit: share of missing days by season, and vs. level, included series
    incl = mm[mm["excluded_reason"] == ""]
    d = daily[daily["sp"].isin(incl["sp"])]
    d = d[d["date"].dt.year.isin(L.TRAIN_YEARS)].copy()
    d["season"] = (d["date"].dt.month % 12) // 3
    full = (incl[["sp", "pollutant"]].merge(pd.DataFrame({"date": pd.date_range("2013-01-01", "2024-12-31")}), how="cross"))
    full = full.merge(d[["sp", "date", "value"]], on=["sp", "date"], how="left")
    full["season"] = (full["date"].dt.month % 12) // 3
    full["missing"] = full["value"].isna()
    aud = full.groupby(["pollutant", "season"])["missing"].mean().reset_index(name="missing_share")
    lvl = d.groupby(["sp", d["date"].dt.year])["value"].mean().rename("annual_mean").reset_index()
    ms = full.groupby(["sp", full["date"].dt.year])["missing"].mean().rename("missing_share").reset_index()
    lv = lvl.merge(ms, on=["sp", "date"]).merge(incl[["sp", "pollutant"]], on="sp")
    corr = lv.groupby("pollutant").apply(lambda g: g["annual_mean"].corr(g["missing_share"], method="spearman")).rename("spearman_level_vs_missing").reset_index()
    aud = aud.merge(corr, on="pollutant")
    aud.to_csv(P.outputs / "missingness_audit.csv", index=False)
    L.manifest_add(P, "missingness_audit.csv", "02_build_CPU", "missingness", "share of missing days by pollutant and season (2013-2024), Spearman correlation of station-year level with missing share")

    # observed NO2 1-hour attainment (descriptive only, §1 table)
    hh = pd.concat([pd.read_parquet(p) for p in sorted(P.daily.glob("*_no2_hourly_exc.parquet"))], ignore_index=True)
    hh = hh[hh["sp"].isin(incl.loc[incl["pollutant"] == "NO2", "sp"]) & hh["year"].between(2013, 2025)]
    hh["days"] = np.where(hh["year"] % 4 == 0, 366, 365)
    hh["coverage"] = hh["n_valid_hours"] / (hh["days"] * 24)
    hh["valid"] = hh["coverage"] >= L.COVERAGE_MIN
    hh["attain_1h"] = hh["n_exc_hours"] <= L.LIMITS["NO2"]["hourly_allow"]
    no2h = hh[hh["valid"]].groupby("year").agg(stations=("sp", "nunique"), share_attaining_1h=("attain_1h", "mean")).reset_index()
    no2h.to_csv(P.outputs / "no2_hourly_observed.csv", index=False)
    L.manifest_add(P, "no2_hourly_observed.csv", "02_build_CPU", "NO2 1-hour", "observed share of included NO2 stations meeting the 1-hour limit (<=3 hours >200) per year, valid station-years only")
    return flow


# --------------------------------------------------------------------------------------------
# Series access
# --------------------------------------------------------------------------------------------
class Store:
    def __init__(self, P: L.Paths):
        self.P = P
        self.man = pd.read_parquet(P.work / "manifest.parquet")
        self.inc = self.man[self.man["excluded_reason"] == ""].reset_index(drop=True)
        self.hist = self.man[self.man["history_only"] & self.man["isd"].notna() & self.man["break_year"].notna()].reset_index(drop=True)
        self.daily = L.load_daily(P)
        self.daily = self.daily[self.daily["sp"].isin(set(self.inc["sp"]) | set(self.hist["sp"]))]
        self.by_sp = {k: g[["date", "value"]].sort_values("date") for k, g in self.daily.groupby("sp")}
        met = pd.read_parquet(P.work / "met_all.parquet")
        self.met = {k: g.drop(columns=["isd"]) for k, g in met.groupby("isd")}

    def items(self, pollutant=None, sample=None, history=False):
        base = self.hist if history else self.inc
        d = base if pollutant is None else base[base["pollutant"] == pollutant]
        if sample is not None:
            d = d.loc[sample.index.intersection(d.index)]
        for r in d.itertuples():
            key = {"station": r.station, "pollutant": r.pollutant, "cc": r.cc, "stype": r.stype, "area": r.area}
            if history:
                key["break_year"] = int(r.break_year)
            if r.sp in self.by_sp and r.isd in self.met:
                yield key, self.by_sp[r.sp], self.met[r.isd]


def pilot_sample(inc: pd.DataFrame, frac=0.01, seed=L.SEED) -> pd.DataFrame:
    """1% sample stratified by country × pollutant × station type (at least 1 per stratum with >=20)."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in inc.groupby(["cc", "pollutant", "stype"]):
        n = int(round(len(g) * frac))
        if n == 0 and len(g) >= 20:
            n = 1
        if n:
            parts.append(g.iloc[rng.choice(len(g), n, replace=False)])
    s = pd.concat(parts) if parts else inc.iloc[:0]
    if len(s) < 10:
        s = inc.iloc[rng.choice(len(inc), min(10, len(inc)), replace=False)]
    return s


def fixed_subsample(inc: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(inc) * frac)))
    return inc.iloc[np.sort(rng.choice(len(inc), n, replace=False))]


# --------------------------------------------------------------------------------------------
# Batch runner with Drive checkpoints
# --------------------------------------------------------------------------------------------
def run_batch(P: L.Paths, name: str, fn, items, n_jobs=None, chunk=50, **kw):
    """Run fn(key, series, met, **kw) for every item in parallel, checkpointing every `chunk` items to
    data/work/<name>/part_XXXXX.parquet. Restarting skips finished parts. Returns all rows."""
    from joblib import Parallel, delayed
    import hashlib
    n_jobs = n_jobs or os.cpu_count() or 1
    d = P.work / name
    items = list(items)
    # Checkpoints are reused only if code, settings and the list of series are unchanged
    # otherwise the old folder is set aside, never mixed in.
    here = Path(__file__).parent
    h = hashlib.sha256()
    for f in ("p1lib.py", "p1run.py"):
        h.update((here / f).read_bytes())
    h.update(json.dumps(kw, sort_keys=True, default=str).encode())
    h.update(json.dumps([k for k, _, _ in items], sort_keys=True, default=str).encode())
    h.update(getattr(fn, "__name__", str(fn)).encode())
    h.update(f"chunk={chunk}".encode())
    for k, s_, m_ in items:   # the input data themselves (daily series and meteorology)
        h.update(str(int(pd.util.hash_pandas_object(s_, index=False).sum())).encode())
        h.update(str(int(pd.util.hash_pandas_object(m_, index=False).sum())).encode())
    fp = h.hexdigest()
    fpf = d / "fingerprint.txt"
    if d.exists() and (not fpf.exists() or fpf.read_text().strip() != fp):
        stale = d.with_name(f"{name}_stale_{time.strftime('%Y%m%d%H%M%S')}")
        d.rename(stale)
        L.log(f"{name}: checkpoints from older code/settings moved to {stale.name}")
    d.mkdir(parents=True, exist_ok=True)
    fpf.write_text(fp)
    t0 = time.time()
    for start in range(0, len(items), chunk):
        part = d / f"part_{start:05d}.parquet"
        if part.exists():
            continue
        batch = items[start:start + chunk]
        res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_wrap)(fn, k, s, m, kw) for k, s, m in batch)
        rows = []
        for r in res:
            rows += r if isinstance(r, list) else [r]
        pd.DataFrame(rows).to_parquet(part, index=False)
        done = min(start + chunk, len(items))
        el = time.time() - t0
        L.log(f"{name}: {done}/{len(items)} series, {el/60:.1f} min elapsed")
    parts = sorted(d.glob("part_*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


def _wrap(fn, key, series, met, kw):
    import warnings
    warnings.filterwarnings("ignore")
    try:
        return fn(key, series, met, **kw)
    except Exception as e:  # recorded per series, never dropped silently
        return {**key, "status": f"error: {type(e).__name__}: {e}"}


# --------------------------------------------------------------------------------------------
# 03 pilot: hyperparameters (time-blocked) and timing
# --------------------------------------------------------------------------------------------
HP_GRID = [{"num_leaves": nl, "n_estimators": ne} for nl in (15, 31) for ne in (200, 400)]


def hp_score(key, series, met, hp):
    d = L.prepare_series(series, met)
    ok = d["value"].notna() & d["met_valid"]
    # Tuning fits 2013-2015 and tests on 2016, all before the first hindcast origin (2017), so no
    # hindcast target year informs the settings.
    # test year: the latest year <= 2016 with >=100 usable days; fit on the years before it (>=2 years)
    counts = d[ok].groupby("year").size()
    cands = [y for y in range(2016, 2014, -1) if counts.get(y, 0) >= 100 and (counts.loc[:y - 1] >= 100).sum() >= 2]
    if not cands:
        return {**key, **hp, "status": "too few tuning days"}
    ty = cands[0]
    tr = d[ok & d["year"].between(2013, ty - 1)]
    te = d[ok & (d["year"] == ty)]
    m = L.fit_series(tr, hp={**L.DEFAULT_HP, **hp})
    y = np.log(te["value"].values + L.OFFSET)
    pred = m.a[max(m.a)] + m.f(te[L.FEATURES])
    rmse_level = float(np.sqrt(np.mean((y - pred) ** 2)))
    # day-to-day skill with the year effect re-estimated (isolates f)
    pred2 = pred - np.mean(pred) + np.mean(y)
    return {**key, **hp, "rmse": rmse_level, "rmse_f": float(np.sqrt(np.mean((y - pred2) ** 2))), "n_test": len(te)}


def nb03_pilot(P: L.Paths):
    S = Store(P)
    samp = pilot_sample(S.inc)
    samp.to_csv(P.outputs / "pilot_sample.csv", index=False)
    L.manifest_add(P, "pilot_sample.csv", "03_pilot_CPU", "sample", "1% pilot sample stratified by country x pollutant x station type")
    rows = []
    for hp in HP_GRID:
        r = run_batch(P, f"pilot_hp_{hp['num_leaves']}_{hp['n_estimators']}", hp_score, S.items(sample=samp), hp=hp)
        rows.append(r)
    hpres = pd.concat(rows, ignore_index=True)
    if "rmse_f" not in hpres or hpres["rmse_f"].notna().sum() == 0:
        raise RuntimeError("hyperparameter pilot produced no usable test years")
    hpres = hpres[hpres["rmse_f"].notna()]
    summ = hpres.groupby(["num_leaves", "n_estimators"])[["rmse_f", "rmse"]].mean().reset_index()
    best = summ.sort_values("rmse_f").iloc[0]
    hp = {**L.DEFAULT_HP, "num_leaves": int(best.num_leaves), "n_estimators": int(best.n_estimators)}
    (P.outputs / "hyperparams.json").write_text(json.dumps(hp, indent=1))
    summ.to_csv(P.outputs / "pilot_hp_grid.csv", index=False)
    L.manifest_add(P, "hyperparams.json", "03_pilot_CPU", "tuning", "global LightGBM hyperparameters chosen on the 1% pilot (test on the latest year <= 2016 with >=100 usable days, fit on earlier years; day-to-day RMSE of log concentration)")
    L.manifest_add(P, "pilot_hp_grid.csv", "03_pilot_CPU", "tuning", "mean held-out RMSE for each hyperparameter setting on the pilot")
    # timing
    t = time.time(); m = run_batch(P, "pilot_main", L.run_main, S.items(sample=samp), hp=hp); tm = time.time() - t
    t = time.time(); h = run_batch(P, "pilot_hindcast", L.run_hindcast, S.items(sample=samp), hp=hp); th = time.time() - t
    small = samp.iloc[:max(2, len(samp) // 5)]
    t = time.time(); b = run_batch(P, "pilot_boot", L.run_boot, S.items(sample=small), hp=hp); tb = time.time() - t
    n = len(samp); N = len(S.inc)
    timing = pd.DataFrame([{"stage": "main", "pilot_series": n, "seconds": tm, "projected_hours_all": tm / n * N / 3600},
                           {"stage": "hindcast", "pilot_series": n, "seconds": th, "projected_hours_all": th / n * N / 3600},
                           {"stage": "bootstrap (20% subsample)", "pilot_series": len(small), "seconds": tb,
                            "projected_hours_all": tb / len(small) * 0.2 * N / 3600}])
    timing["cpu_count"] = os.cpu_count()
    timing["errors"] = [int(x["status"].astype(str).str.startswith("error").sum()) if "status" in x else 0
                        for x in (m, h, b)]
    timing.to_csv(P.outputs / "pilot_timing.csv", index=False)
    L.manifest_add(P, "pilot_timing.csv", "03_pilot_CPU", "timing", "pilot run time and projected CPU hours for the full main and hindcast runs")
    return timing, m, h


# --------------------------------------------------------------------------------------------
# 04 main, 05 hindcast
# --------------------------------------------------------------------------------------------
def load_hp(P):
    f = P.outputs / "hyperparams.json"
    return json.loads(f.read_text()) if f.exists() else L.DEFAULT_HP


def nb04_main(P: L.Paths, pollutant=None):
    S = Store(P)
    hp = load_hp(P)
    tag = f"main_{pollutant or 'all'}"
    res = run_batch(P, tag, L.run_main, S.items(pollutant=pollutant), hp=hp)
    out = P.outputs / f"{tag}_station_results.csv"
    res.to_csv(out, index=False)
    L.manifest_add(P, out.name, "04_main_CPU", "main", "station-level E1 (with 90% interval), E2 (WLS, Theil-Sen, with 2020), E3 headroom, E5 variance parts, yearly weather-adjusted levels")
    return res


def nb05_hindcast(P: L.Paths, pollutant=None):
    S = Store(P)
    hp = load_hp(P)
    tag = f"hindcast_{pollutant or 'all'}"
    res = run_batch(P, tag, L.run_hindcast, S.items(pollutant=pollutant), hp=hp)
    out = P.outputs / f"{tag}_rows.csv.gz"
    res.to_csv(out, index=False)
    L.manifest_add(P, out.name, "05_hindcast_CPU", "hindcast", "one row per series x origin x horizon x scenario: forecast probabilities, observed outcomes, baselines, CRPS, PIT, RPS")
    return res


# --------------------------------------------------------------------------------------------
# Scoring tables (§3.4)
# --------------------------------------------------------------------------------------------
def score_tables(P: L.Paths, rows: pd.DataFrame, B=300):
    if "status" in rows:                       # hindcast rows carry 'status' only when a series failed
        rows = rows[rows["status"].isna() | (rows["status"] == "ok")]
    rows = rows.dropna(subset=["h"])
    out = []
    for (st, pol, scen, h), g in rows.groupby(["set", "pollutant", "scenario", "h"]):
        for k in ["att_annual", "att_daily", "att_joint"]:
            o = g[f"obs_{k}"]; p = g[f"p_{k}"]
            dec = L.brier_decomp(p, o)
            rec = {"set": st, "pollutant": pol, "scenario": scen, "h": h, "metric": k, **dec, "base_rate": float(np.nanmean(o))}
            one_year = g["target"].nunique() == 1
            rec["ci_type"] = "one-way (countries): single target year" if one_year else "two-way (target years x countries)"
            for ref in ["clim", "pers", "naive"]:
                stat = lambda df, ref=ref, k=k: L.bss(df[f"p_{k}"], df[f"{ref}_{k}"], df[f"obs_{k}"])
                est, lo, hi = L.cluster_boot(g.dropna(subset=[f"obs_{k}"]), stat, B=B,
                                             c1="cc" if one_year else "target", c2="cc")
                rec.update({f"bss_{ref}": est, f"bss_{ref}_lo90": lo, f"bss_{ref}_hi90": hi})
            if f"pperfect_{k}" in g:
                rec["bs_perfect_weather"] = L.brier_decomp(g[f"pperfect_{k}"], o)["bs"]
            out.append(rec)
    sc = pd.DataFrame(out)
    sc.to_csv(P.outputs / "hindcast_scores.csv", index=False)
    L.manifest_add(P, "hindcast_scores.csv", "05_hindcast_CPU", "scores", "Brier score, binned decomposition (+ remainder) and Brier skill scores vs climatology/persistence/naive trend with 90% two-way cluster-bootstrap CIs, by set (hindcast / 2025 check) x pollutant x scenario x horizon x metric")
    # interval coverage and CRPS for annual mean
    cov = []
    for (st, pol, scen, h), g in rows.groupby(["set", "pollutant", "scenario", "h"]):
        # deviation 17: score coverage only where an interval exists (a missing interval is not a miss)
        gg = g.dropna(subset=["obs_mean", "pi90_lo", "pi90_hi"])
        pit = np.sort(g["pit_exc_obsdays"].dropna().values)
        n_pit = len(pit)
        ks = (float(max(np.max(np.arange(1, n_pit + 1) / n_pit - pit), np.max(pit - np.arange(0, n_pit) / n_pit)))
              if n_pit else np.nan)          # exact two-sided Kolmogorov-Smirnov distance of the PIT values from U(0,1)
        cov.append({"set": st, "pollutant": pol, "scenario": scen, "h": h, "n": len(gg),
                    "n_obs_without_interval": int(g.dropna(subset=["obs_mean"]).shape[0] - len(gg)),
                    "cov80": float(((gg.obs_mean >= gg.pi80_lo) & (gg.obs_mean <= gg.pi80_hi)).mean()) if len(gg) else np.nan,
                    "cov90": float(((gg.obs_mean >= gg.pi90_lo) & (gg.obs_mean <= gg.pi90_hi)).mean()) if len(gg) else np.nan,
                    "crps_mean": float(gg.crps_mean.mean()) if len(gg) else np.nan,
                    "rps_exc_obsdays": float(g.rps_exc_obsdays.mean()),
                    "pit_exc_ks": ks})
    cv = pd.DataFrame(cov)
    cv.to_csv(P.outputs / "hindcast_coverage.csv", index=False)
    L.manifest_add(P, "hindcast_coverage.csv", "05_hindcast_CPU", "coverage", "80%/90% interval coverage, mean CRPS (annual mean), mean RPS and PIT KS distance (exceedance days scored on observed days) by set x pollutant x scenario x horizon")
    # reliability diagram data
    rel = []
    for (pol, scen, h), g in rows[rows["set"] == "hindcast"].groupby(["pollutant", "scenario", "h"]):
        g = g.dropna(subset=["obs_att_joint", "p_att_joint"])
        if not len(g):
            continue
        b = np.clip((g.p_att_joint * 10).astype(int), 0, 9)
        for k, gb in g.groupby(b):
            rel.append({"pollutant": pol, "scenario": scen, "h": h, "bin": int(k), "n": len(gb),
                        "mean_p": float(gb.p_att_joint.mean()), "obs_freq": float(gb.obs_att_joint.mean())})
    pd.DataFrame(rel).to_csv(P.outputs / "reliability_bins.csv", index=False)
    L.manifest_add(P, "reliability_bins.csv", "05_hindcast_CPU", "reliability", "reliability diagram bins for joint attainment")
    return sc, cv


def success_check(P: L.Paths, rows: pd.DataFrame, cv: pd.DataFrame, B=500):
    """Pre-specified success criteria (§3.4). Horizons 5 and 6 are pooled before the two-way cluster
    bootstrap, so the long-horizon check has more than one target year."""
    rows = rows[rows["set"] == "hindcast"]
    out = []
    for pol in sorted(rows["pollutant"].unique()):
        for scen in ["frozen", "trend"]:
            for hs, label in [((1,), "h1"), ((5, 6), "h5-6")]:
                g_all = rows[(rows.pollutant == pol) & (rows.scenario == scen) & (rows.h.isin(hs))]
                g = g_all.dropna(subset=["obs_att_joint"])
                if not len(g):
                    continue
                los = {}
                for ref in ["clim", "pers", "naive"]:
                    stat = lambda df, ref=ref: L.bss(df["p_att_joint"], df[f"{ref}_att_joint"], df["obs_att_joint"])
                    est, lo, hi = L.cluster_boot(g, stat, B=B)
                    los[ref] = (est, lo, hi)
                pos = all(np.isfinite(v[1]) and v[1] > 0 for v in los.values())
                gg = g_all.dropna(subset=["obs_mean", "pi90_lo", "pi90_hi"])   # deviation 17
                cov90 = float(((gg.obs_mean >= gg.pi90_lo) & (gg.obs_mean <= gg.pi90_hi)).mean()) if len(gg) else np.nan
                cov_ok = bool(0.85 <= cov90 <= 0.95) if np.isfinite(cov90) else False
                out.append({"pollutant": pol, "scenario": scen, "horizons": label, "n": len(g),
                            **{f"bss_{r}": v[0] for r, v in los.items()}, **{f"bss_{r}_lo90": v[1] for r, v in los.items()},
                            "coverage90": cov90, "bss_all_positive_lo90": pos,
                            "coverage90_in_85_95": cov_ok, "criteria_met": pos and cov_ok})
    o = pd.DataFrame(out)
    o.to_csv(P.outputs / "success_criteria.csv", index=False)
    L.manifest_add(P, "success_criteria.csv", "05_hindcast_CPU", "criteria", "pre-specified success criteria (BSS lower 90% bound > 0 vs all baselines; 90% coverage in 85-95%), horizons 5-6 pooled, by pollutant x scenario")
    return o


# --------------------------------------------------------------------------------------------
# 06 sensitivity (§3.6)
# --------------------------------------------------------------------------------------------
def nb06_sensitivity(P: L.Paths):
    """Sensitivity analyses that need their own model fits.
    - item 1: GAM (M2) on the pre-set 10% subsample (main estimates and 1-year-ahead hindcast);
    - §3.1:  full refitting bootstrap (50 reps) on the pre-set 20% subsample, checking the s.e. approximation;
    - item 7: stations excluded only for a relocation or method change, refit with a break in the trend.
    Items 2, 3, 5 and 6c come from the main run (columns S_*, E2ts_*, E2c20_*); item 4 and the 25-km
    meteorology match are filters applied in sensitivity_summary()."""
    S = Store(P)
    hp = load_hp(P)
    sub10 = fixed_subsample(S.inc, 0.10, L.SEED + 2)
    sub20 = fixed_subsample(S.inc, 0.20, L.SEED + 3)
    sub10.to_csv(P.outputs / "gam_subsample.csv", index=False)
    sub20.to_csv(P.outputs / "bootstrap_subsample.csv", index=False)
    L.manifest_add(P, "gam_subsample.csv", "06_sensitivity_CPU", "subsample", "pre-set 10% subsample for the GAM sensitivity (fixed seed)")
    L.manifest_add(P, "bootstrap_subsample.csv", "06_sensitivity_CPU", "subsample", "pre-set 20% subsample for the full refitting bootstrap (fixed seed)")
    out = {
        "gam_main": run_batch(P, "sens_gam_main", L.run_main, S.items(sample=sub10), hp=hp, kind="gam", n_trend=100, sens=False),
        "gam_hindcast_h1": run_batch(P, "sens_gam_hindcast", L.run_hindcast, S.items(sample=sub10), hp=hp, kind="gam", horizons=(1,)),
        "bootstrap": run_batch(P, "sens_bootstrap", L.run_boot, S.items(sample=sub20), hp=hp),
        "breaks": run_batch(P, "sens_breaks", run_main_with_break, S.items(history=True), hp=hp),
    }
    desc = {"gam_main": "GAM (M2) instead of LightGBM, main estimates, 10% subsample (§3.6 item 1)",
            "gam_hindcast_h1": "GAM (M2), 1-year-ahead hindcast rows, 10% subsample (§3.6 item 1)",
            "bootstrap": "full refitting bootstrap (50 reps) of E1, E2, E3 and the 2024 level, 20% subsample (§3.1)",
            "breaks": "stations excluded only for a relocation or method change, refit with a trend break (§3.6 item 7)"}
    for name, r in out.items():
        f = P.outputs / f"sens_{name}.csv"
        r.to_csv(f, index=False)
        L.manifest_add(P, f.name, "06_sensitivity_CPU", name, desc[name])
    return out


def nb06b_mc_error(P: L.Paths):
    """Monte Carlo error of all reported estimators at fixed fits (pre-set 2% subsample, 3 repeats)."""
    S = Store(P)
    hp = load_hp(P)
    sub = fixed_subsample(S.inc, 0.02, L.SEED + 5)
    r = run_batch(P, "sens_mc_error", L.run_mc_error, S.items(sample=sub), hp=hp)
    r.to_csv(P.outputs / "mc_error.csv", index=False)
    L.manifest_add(P, "mc_error.csv", "06_sensitivity_CPU", "mc error", "SD across 3 independent integrations at a fixed fit, for every reported estimator (2% subsample)")
    ok = r[r.get("status", "ok") == "ok"] if "status" in r else r
    cols = [c for c in ok.columns if c.startswith("mcsd_")]
    summ = ok.groupby("pollutant")[cols].median().T.reset_index().rename(columns={"index": "estimator"})
    summ.to_csv(P.outputs / "mc_error_summary.csv", index=False)
    L.manifest_add(P, "mc_error_summary.csv", "06_sensitivity_CPU", "mc error", "median Monte Carlo SD per estimator and pollutant")
    return r


def run_main_with_break(key, series, met, **kw):
    d = L.prepare_series(series, met)
    yrs = sorted(set(L.training_rows(d, 2024)["year"]) - {2020})
    if not L.break_supported(yrs, key.get("break_year")):
        return {**key, "status": "unsupported break: needs >= 1 training year on each side"}
    return L.run_main(key, series, met, break_year=key.get("break_year"), sens=False, **kw)


def sensitivity_summary(P: L.Paths, main: pd.DataFrame) -> pd.DataFrame:
    """One table with every pre-specified sensitivity analysis next to the main result (§3.6)."""
    man = pd.read_parquet(P.work / "manifest.parquet")[["station", "pollutant", "elig85", "met_within_25km"]]
    m = main[main["status"] == "ok"].merge(man, on=["station", "pollutant"], how="left")
    rows = []

    def add(variant, pol, series_e1, ref, estimand="E1 joint"):
        both = pd.concat([series_e1.rename("v"), ref.rename("r")], axis=1).dropna()
        rows.append({"variant": variant, "estimand": estimand, "pollutant": pol, "n": len(both),
                     "mean_value": both["v"].mean(), "mean_value_main_same_series": both["r"].mean(),
                     "share_between_0.1_0.9": both["v"].between(0.1, 0.9, inclusive="neither").mean(),
                     "mean_abs_diff_vs_main": (both["v"] - both["r"]).abs().mean()})

    for pol, g in m.groupby("pollutant"):
        ref = g["E1_att_joint"]
        add("main", pol, ref, ref)
        for tag in ["pool1995", "pool2015", "recency", "block3", "block14", "unstrat", "dustdrop5"]:
            c = f"S_{tag}_E1_att_joint"
            if c in g:
                add(f"E1 {tag}", pol, g[c], ref)
        add("training share >=85%", pol, g.loc[g["elig85"] == True, "E1_att_joint"], ref)
        add("meteorology within 25 km", pol, g.loc[g["met_within_25km"] == True, "E1_att_joint"], ref)
        for tag, lab in [("E2", "E2 main (WLS, 2020 out)"), ("E2ts", "E2 Theil-Sen"), ("E2c20", "E2 with 2020")]:
            c = f"{tag}_att_joint"
            if c in g:
                add(lab, pol, g[c], g["E2_att_joint"], estimand="E2 joint")
    for name, col, refcol in [("gam_main", "E1_att_joint", "E1_att_joint"), ("breaks", "E2_att_joint", None)]:
        f = P.outputs / f"sens_{name}.csv"
        if f.exists() and f.stat().st_size > 2:
            r = pd.read_csv(f)
            if "status" not in r:
                continue
            bad = r[r["status"] != "ok"]
            for pol, gb in bad.groupby("pollutant"):
                rows.append({"variant": f"{name}: series that failed (status recorded in sens_{name}.csv)",
                             "estimand": "count", "pollutant": pol, "n": len(gb)})
            r = r[r["status"] == "ok"]
            for pol, g in r.groupby("pollutant"):
                if refcol:
                    j = g.merge(m[["station", "pollutant", refcol]], on=["station", "pollutant"], suffixes=("", "_main"))
                    add(f"GAM ({col})", pol, j[col], j[f"{refcol}_main"])
                else:
                    rows.append({"variant": "breaks: excluded stations refit with a trend break", "estimand": "E2 joint",
                                 "pollutant": pol, "n": len(g), "mean_value": g["E2_att_joint"].mean(),
                                 "share_between_0.1_0.9": g["E2_att_joint"].between(0.1, 0.9, inclusive="neither").mean()})
    full = pd.read_parquet(P.work / "manifest.parquet")
    hist = full[full["history_only"]]
    for pol, g in hist.groupby("pollutant"):
        rows.append({"variant": "breaks: history-only series without a datable break or met match (not refit)",
                     "pollutant": pol, "n": int((g["break_year"].isna() | g["isd"].isna()).sum())})
    # GAM: one-year-ahead hindcast skill next to LightGBM on the same series and targets
    gh = P.outputs / "sens_gam_hindcast_h1.csv"
    if gh.exists() and gh.stat().st_size > 2:
        g1 = pd.read_csv(gh)
        main_rows = [pd.read_csv(f) for f in sorted(P.outputs.glob("hindcast_*_rows.csv.gz"))]
        if main_rows and "p_att_joint" in g1:
            mr = pd.concat(main_rows)
            mr = mr[(mr["h"] == 1) & (mr["set"] == "hindcast")]
            j = g1[(g1["h"] == 1) & (g1["set"] == "hindcast")].merge(
                mr, on=["station", "pollutant", "origin", "target", "scenario"], suffixes=("_gam", "_lgbm"))
            for (pol, scen), gg in j.groupby(["pollutant", "scenario"]):
                rows.append({"variant": f"GAM vs LightGBM, 1-year hindcast Brier ({scen})", "pollutant": pol, "n": len(gg),
                             "brier_gam": L.brier_decomp(gg["p_att_joint_gam"], gg["obs_att_joint_gam"])["bs"],
                             "brier_lgbm": L.brier_decomp(gg["p_att_joint_lgbm"], gg["obs_att_joint_lgbm"])["bs"]})
    out = pd.DataFrame(rows)
    out.to_csv(P.outputs / "sensitivity_summary.csv", index=False)
    L.manifest_add(P, "sensitivity_summary.csv", "07_results_CPU", "sensitivity", "every pre-specified sensitivity analysis next to the main E1/E2 (joint) on the same series (§3.6)")
    b = P.outputs / "sens_bootstrap.csv"
    if b.exists():
        bb = pd.read_csv(b)
        bb = bb[bb["status"] == "ok"]
        m = m.drop(columns=[c for c in m.columns if c.startswith("boot_")], errors="ignore")
        bb = bb.merge(m, on=["station", "pollutant"], suffixes=("", "_main"))
        recs = []
        for pol, g in bb.groupby("pollutant"):
            rec = {"pollutant": pol, "n": len(g),
                   "median_ratio_boot_sd_to_se": float((g["boot_sd_a_2024"] / g["se_2024"]).median())}
            for est in [f"E1_{k}" for k in ("att_annual", "att_daily", "att_joint")] + \
                       [f"E2_{k}" for k in ("att_annual", "att_daily", "att_joint")] + \
                       [f"E3_{k}_{q}" for k in ("att_annual", "att_daily", "att_joint") for q in (90, 50)]:
                if f"boot_{est}_lo" in g and f"{est}_lo" in g:
                    ok = np.isfinite(g[f"boot_{est}_lo"]) & np.isfinite(g[f"boot_{est}_hi"]) & g[f"{est}_lo"].notna()
                    rec[f"{est}_n_comparable"] = int(ok.sum())
                    rec[f"{est}_width_boot"] = float((g.loc[ok, f"boot_{est}_hi"] - g.loc[ok, f"boot_{est}_lo"]).median()) if ok.any() else np.nan
                    if est.startswith("E2_"):
                        # main E2 interval = trend-parameter uncertainty; bootstrap = re-estimated yearly
                        # levels. Different sources: reported side by side, not as a check of each other.
                        rec[f"{est}_width_trend_param"] = float((g.loc[ok, f"{est}_hi"] - g.loc[ok, f"{est}_lo"]).median()) if ok.any() else np.nan
                    else:
                        rec[f"{est}_width_approx"] = float((g.loc[ok, f"{est}_hi"] - g.loc[ok, f"{est}_lo"]).median()) if ok.any() else np.nan
            recs.append(rec)
        chk = pd.DataFrame(recs)
        chk.to_csv(P.outputs / "bootstrap_check.csv", index=False)
        L.manifest_add(P, "bootstrap_check.csv", "07_results_CPU", "bootstrap", "full bootstrap vs approximations: ratio of bootstrap SD of the 2024 level to its s.e., and median 5-95% widths (bootstrap vs approximation) for E1, E2 and E3 for every metric")
    return out


# --------------------------------------------------------------------------------------------
# Summaries for the manuscript (every number in the paper comes from these files)
# --------------------------------------------------------------------------------------------
def censored_median(g: pd.DataFrame, col: str):
    """Median of E3 keeping censored stations: left-censored (target unreachable even with a >98% cut)
    count as -inf, right-censored as +inf. Returns a number, or a string like '< -0.98' / '> 6.39' when
    the median itself falls in a censored tail."""
    if col not in g:
        return np.nan
    cens = g.get(f"{col}_cens", pd.Series("", index=g.index)).fillna("")
    v = g[col].copy()
    v[cens == "left"] = -np.inf
    v[cens == "right"] = np.inf
    v = v.dropna()
    if not len(v):
        return np.nan
    med = float(np.median(v.values))
    if med == -np.inf:
        return f"< {np.exp(L.HEADROOM_GRID[0]) - 1:.2f}"
    if med == np.inf:
        return f"> {np.exp(L.HEADROOM_GRID[-1]) - 1:.2f}"
    return med


def summaries(P: L.Paths, main: pd.DataFrame):
    m = main[main["status"] == "ok"].copy()
    rows = []
    for pol, g in m.groupby("pollutant"):
        for k in ["att_annual", "att_daily", "att_joint"]:
            e1 = g[f"E1_{k}"]
            rows.append({"pollutant": pol, "metric": k, "n_series": len(g),
                         "mean_E1": e1.mean(), "median_E1": e1.median(),
                         "share_E1_ge_0.9": (e1 >= 0.9).mean(), "share_E1_le_0.1": (e1 <= 0.1).mean(),
                         "share_weather_marginal": e1.between(0.1, 0.9, inclusive="neither").mean(),
                         "n_weather_marginal": int(e1.between(0.1, 0.9, inclusive="neither").sum()),
                         "n_E3_unreachable": int(g.get(f"E3_{k}_90", pd.Series(dtype=float)).isna().sum()),
                         "mean_E2": g.get(f"E2_{k}", pd.Series(dtype=float)).mean(),
                         "share_E2_ge_0.9": (g.get(f"E2_{k}", pd.Series(dtype=float)) >= 0.9).mean(),
                         "median_E3_90": censored_median(g, f"E3_{k}_90"),
                         "median_E3_50": censored_median(g, f"E3_{k}_50"),
                         "n_E3_90_nonmono_gt_0.02": int((g.get(f"E3_{k}_90_nonmono", pd.Series(dtype=float)) > 0.02).sum()),
                         "n_E3_50_left_censored": int((g.get(f"E3_{k}_50_cens", pd.Series(dtype=str)) == "left").sum()),
                         "n_E3_50_right_censored": int((g.get(f"E3_{k}_50_cens", pd.Series(dtype=str)) == "right").sum()),
                         "n_E3_50_nonmono_gt_0.02": int((g.get(f"E3_{k}_50_nonmono", pd.Series(dtype=float)) > 0.02).sum()),
                         "n_E3_90_left_censored": int((g.get(f"E3_{k}_90_cens", pd.Series(dtype=str)) == "left").sum()),
                         "n_E3_90_right_censored": int((g.get(f"E3_{k}_90_cens", pd.Series(dtype=str)) == "right").sum()),
                         "median_met_support": g["met_support"].median() if "met_support" in g else np.nan,
                         "median_E1_mcse": g[f"E1_{k}_mcse"].median() if f"E1_{k}_mcse" in g else np.nan,
                         "max_E1_mcse": g[f"E1_{k}_mcse"].max() if f"E1_{k}_mcse" in g else np.nan})
    s = pd.DataFrame(rows)
    s.to_csv(P.outputs / "summary_by_pollutant.csv", index=False)
    L.manifest_add(P, "summary_by_pollutant.csv", "07_figures_CPU", "summary", "E1/E2/E3 summaries by pollutant and metric (headline numbers)")
    grp = []
    for by in [["pollutant", "cc"], ["pollutant", "stype"], ["pollutant", "area"]]:
        for keys, g in m.groupby(by):
            grp.append({**dict(zip(by, keys if isinstance(keys, tuple) else (keys,))), "n": len(g),
                        "share_weather_marginal_joint": g["E1_att_joint"].between(0.1, 0.9, inclusive="neither").mean(),
                        "mean_E1_joint": g["E1_att_joint"].mean()})
    pd.DataFrame(grp).to_csv(P.outputs / "summary_by_group.csv", index=False)
    L.manifest_add(P, "summary_by_group.csv", "07_figures_CPU", "summary", "weather-marginal shares and mean E1 (joint) by country, station type and area")
    # dust / sanding strata (§3.6 6a)
    m["stratum"] = np.where(m.cc.isin(L.DUST_STATES), "dust", np.where(m.cc.isin(L.SANDING_STATES), "sanding", "other"))
    st = m.groupby(["pollutant", "stratum"]).agg(n=("E1_att_joint", "size"), mean_E1_joint=("E1_att_joint", "mean"),
                                                 share_marginal=("E1_att_joint", lambda x: x.between(0.1, 0.9, inclusive="neither").mean())).reset_index()
    st.to_csv(P.outputs / "summary_dust_sanding.csv", index=False)
    L.manifest_add(P, "summary_dust_sanding.csv", "07_figures_CPU", "summary", "E1 joint by dust-affected / sanding / other member states (§3.6 item 6a)")
    # E5 shares
    e5 = []
    for pol, g in m.groupby("pollutant"):
        for metric in ["mean", "exc"]:
            tot = g[f"E5_{metric}_total"]
            e5.append({"pollutant": pol, "metric": metric,
                       "share_trend": (g[f"E5_{metric}_trend"] / tot).median(),
                       "share_weather": (g[f"E5_{metric}_weather"] / tot).median(),
                       "share_resid": (g[f"E5_{metric}_resid"] / tot).median(),
                       "share_weather_first_order": (g[f"E5_{metric}_weather_first"] / tot).median()})
    pd.DataFrame(e5).to_csv(P.outputs / "summary_e5.csv", index=False)
    L.manifest_add(P, "summary_e5.csv", "07_figures_CPU", "summary", "median shares of 2030 variance due to trend, weather and residual (both orders)")
    return s
