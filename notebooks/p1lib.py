"""
p1lib — analysis library for P1 (Environmental Modeling & Assessment):
"How likely is compliance? A weather-resampled, hindcast-validated assessment of European
air-quality stations against the 2030 EU limit values for PM2.5, PM10 and NO2".

Implements the pre-specified analysis plan (v1.2). Section numbers in comments (§x.y) refer to that plan;
later changes are listed in the paper's Supplementary Section S5.
Runs in Google Colab (CPU) or locally for smoke tests. Nothing here invents data: every value comes from
EEA e-Reporting (E1a), NOAA ISD-Lite, or the directive text; resampling only re-uses observed weather years
and observed model residuals to build predictive distributions (§3.2).
"""
from __future__ import annotations

import gzip
import io
import json
import math
import os
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------------------------
# Constants (§1, §2)
# --------------------------------------------------------------------------------------------
EU27 = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV",
        "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"]
# EEA uses "GR" for Greece in the download API; plan text uses EL. Both map to Greece.
DUST_STATES = {"CY", "GR", "ES", "IT", "MT", "PT"}           # §3.6 item 6a (EL = GR)
SANDING_STATES = {"DK", "EE", "FI", "LT", "LV", "SE"}         # §3.6 item 6a

POLLUTANTS = {"NO2": 8, "PM10": 5, "PM2.5": 6001}
# 2030 limit values, Directive (EU) 2024/2881 Annex I Table 1 (verified in the OJ text 2026-11-04)
LIMITS = {
    "NO2": {"annual": 20.0, "daily": 50.0, "allow": 18, "hourly": 200.0, "hourly_allow": 3},
    "PM10": {"annual": 20.0, "daily": 45.0, "allow": 18},
    "PM2.5": {"annual": 10.0, "daily": 25.0, "allow": 18},
}
COVERAGE_MIN = 0.85          # Annex V point B, fixed measurements (deviation 1)
VALID_DAY_HOURS = 18         # Annex V point C
TRAIN_DAY_SHARE = 0.75       # §2 rule 3
TRAIN_YEARS = list(range(2013, 2025))
CHECK_YEAR = 2025
MAIN_POOL = list(range(2005, 2025))
OFFSET = 0.1                 # §3.1 log offset (µg/m³)
RELOCATION_M = 250.0         # deviation 4: a 'relocation' is >250 m between reported positions
SEED = 20261104

FEATURES = ["t_mean", "t_max", "t_min", "dtr", "td_mean", "rh_mean", "slp_mean", "slp_change",
            "ws_mean", "u_mean", "v_mean", "calm_share", "precip", "precip_reported",
            "ws_prev", "precip_prev", "doy_sin", "doy_cos", "dow"]

EEA_API = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/urls"
EEA_META = "https://discomap.eea.europa.eu/App/AQViewer/download?fqn=Airquality_Dissem.b2g.measurements&f=csv"
ISD_BASE = "https://noaa-isd-pds.s3.amazonaws.com"


@dataclass
class Paths:
    root: Path                       # project folder on Drive (or local test folder)

    def __post_init__(self):
        self.root = Path(self.root)
        for p in [self.data, self.raw_meta, self.daily, self.isd_raw, self.isd_daily, self.work, self.outputs]:
            p.mkdir(parents=True, exist_ok=True)

    @property
    def data(self): return self.root / "data"
    @property
    def raw_meta(self): return self.data / "meta"
    @property
    def daily(self): return self.data / "aq_daily"
    @property
    def isd_raw(self): return self.data / "isd_raw"
    @property
    def isd_daily(self): return self.data / "isd_daily"
    @property
    def work(self): return self.data / "work"
    @property
    def outputs(self): return self.root / "outputs"


def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def manifest_add(paths: Paths, file: str, notebook: str, cell: str, description: str):
    """Add/replace a row in outputs/MANIFEST.md (| file | notebook | cell | description |)."""
    mf = paths.outputs / "MANIFEST.md"
    head = "| file | notebook | cell | description |\n|---|---|---|---|\n"
    rows = []
    if mf.exists():
        rows = [l for l in mf.read_text(encoding="utf-8").splitlines()[2:] if l.startswith("|")]
    rows = [r for r in rows if not r.startswith(f"| {file} |")]
    rows.append(f"| {file} | {notebook} | {cell} | {description} |")
    mf.write_text(head + "\n".join(sorted(rows)) + "\n", encoding="utf-8")


def http_get(url: str, tries: int = 5, timeout: int = 120, **kw) -> requests.Response:
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, **kw)
            if r.status_code == 404:
                return r
            r.raise_for_status()
            return r
        except Exception as e:  # network hiccup: back off and retry
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------------------------
# 1. EEA metadata and data (§2)
# --------------------------------------------------------------------------------------------
def download_metadata(paths: Paths) -> pd.DataFrame:
    out = paths.raw_meta / "sampling_points.parquet"
    if out.exists():
        return pd.read_parquet(out)
    log("downloading EEA sampling-point metadata")
    r = http_get(EEA_META, timeout=600)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
    d = d[d["Air Pollutant"].isin(list(POLLUTANTS))].copy()
    keep = ["Country", "Year", "Air Quality Station EoI Code", "Air Quality Station Name", "Sampling Point Id",
            "Air Pollutant", "Longitude", "Latitude", "Altitude", "Air Quality Station Area",
            "Air Quality Station Type", "Operational Activity Begin", "Operational Activity End",
            "Measurement Type", "Measurement Method", "Process Activity Begin", "Process Activity End",
            "Imported"]
    d = d[keep].rename(columns={"Air Quality Station EoI Code": "station", "Sampling Point Id": "sp",
                                "Air Pollutant": "pollutant", "Air Quality Station Area": "area",
                                "Air Quality Station Type": "stype", "Measurement Type": "mtype",
                                "Measurement Method": "mmethod", "Air Quality Station Name": "station_name"})
    d["cc"] = d["station"].str[:2]
    d.to_parquet(out, index=False)
    return d


def eea_urls(country: str, pollutant: str, dataset: int = 2) -> list[str]:
    body = {"countries": [country], "cities": [], "pollutants": [pollutant], "dataset": dataset, "source": "API"}
    for i in range(5):
        try:
            r = requests.post(EEA_API, json=body, timeout=300)
            r.raise_for_status()
            lines = [l.strip() for l in r.text.splitlines()]
            return [l for l in lines if l.startswith("http") and l.endswith(".parquet")]
        except Exception:
            if i == 4:
                raise
            time.sleep(5 * (i + 1))
    return []


def _aggregate_eea_file(content: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One sampling-point parquet → (daily table, NO2 hourly exceedance table). §2 rules 2 and 5."""
    d = pd.read_parquet(io.BytesIO(content), columns=["Samplingpoint", "Pollutant", "Start", "Value",
                                                       "AggType", "Validity"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")
    d["valid"] = (d["Validity"] >= 1) & d["Value"].notna()
    d["sp"] = d["Samplingpoint"].str.split("/").str[-1]
    d["date"] = d["Start"].dt.floor("D")
    out_daily, out_hourly = [], []
    h = d[d["AggType"] == "hour"]
    if len(h):
        hv = h[h["valid"]]
        g = hv.groupby(["sp", "Pollutant", "date"])["Value"].agg(["mean", "size"]).reset_index()
        g = g.rename(columns={"mean": "value", "size": "n_hours"})
        g["value"] = g["value"].where(g["n_hours"] >= VALID_DAY_HOURS)
        g["agg"] = "hour"
        out_daily.append(g)
        if (h["Pollutant"] == POLLUTANTS["NO2"]).any():
            hv2 = hv.assign(year=hv["date"].dt.year, exc=(hv["Value"] > LIMITS["NO2"]["hourly"]))
            e = hv2.groupby(["sp", "year"]).agg(n_valid_hours=("Value", "size"), n_exc_hours=("exc", "sum"))
            out_hourly.append(e.reset_index())
    dd = d[d["AggType"] == "day"]
    if len(dd):
        g = dd[dd["valid"]].groupby(["sp", "Pollutant", "date"])["Value"].mean().reset_index()
        g = g.rename(columns={"Value": "value"})
        g["n_hours"] = np.nan
        g["agg"] = "day"
        out_daily.append(g)
    daily = pd.concat(out_daily, ignore_index=True) if out_daily else pd.DataFrame()
    hourly = pd.concat(out_hourly, ignore_index=True) if out_hourly else pd.DataFrame()
    return daily, hourly


def download_eea_country(paths: Paths, country: str, threads: int = 8) -> dict:
    """Download every E1a sampling-point file for one country and the 3 pollutants, aggregate to daily
    on the fly (hourly raw data are not kept), write data/aq_daily/<cc>.parquet. Resumable."""
    out = paths.daily / f"{country}.parquet"
    outh = paths.daily / f"{country}_no2_hourly_exc.parquet"
    rec_file = paths.daily / f"{country}_download.json"
    if out.exists() and rec_file.exists() and json.loads(rec_file.read_text()).get("failed", 1) == 0:
        return {"country": country, "status": "exists"}       # complete cache (completion record present)
    urls = []
    for pol in POLLUTANTS:
        urls += eea_urls(country, pol)
    urls = sorted(set(urls))
    log(f"{country}: {len(urls)} files")
    dailies, hourlies, failed = [], [], []

    def job(u):
        r = http_get(u, timeout=300)
        if r.status_code == 404:
            return None
        return _aggregate_eea_file(r.content)

    unavailable = []
    pending = list(urls)
    for attempt in range(3):                       # retry failed files before giving up
        failed = []
        with ThreadPoolExecutor(threads) as ex:
            for u, res in zip(pending, ex.map(lambda x: _safe(job, x), pending)):
                if res is None:
                    unavailable.append(u)          # HTTP 404: genuinely not available
                    continue
                if isinstance(res, Exception):
                    failed.append(u)
                    continue
                dd, hh = res
                if len(dd):
                    dailies.append(dd)
                if len(hh):
                    hourlies.append(hh)
        if not failed:
            break
        pending = failed
        time.sleep(10 * (attempt + 1))
    if failed:
        # do NOT write the country file: an incomplete country must be retried, never cached as complete
        (paths.daily / f"{country}_INCOMPLETE.json").write_text(json.dumps({"country": country, "failed_urls": failed}, indent=1))
        raise RuntimeError(f"{country}: {len(failed)} files failed after 3 attempts; rerun to retry")
    daily = pd.concat(dailies, ignore_index=True) if dailies else pd.DataFrame(
        columns=["sp", "Pollutant", "date", "value", "n_hours", "agg"])
    daily["cc"] = country
    daily.to_parquet(out.with_suffix(".tmp"), index=False)
    out.with_suffix(".tmp").replace(out)                    # atomic write; the completion record follows
    (pd.concat(hourlies, ignore_index=True) if hourlies else pd.DataFrame(
        columns=["sp", "year", "n_valid_hours", "n_exc_hours"])).to_parquet(outh, index=False)
    rec = {"country": country, "files": len(urls), "failed": 0, "unavailable_404": len(unavailable),
           "unavailable_urls": unavailable, "rows": int(len(daily)), "downloaded": time.strftime("%Y-%m-%d %H:%M")}
    (paths.daily / f"{country}_INCOMPLETE.json").unlink(missing_ok=True)
    (paths.daily / f"{country}_download.json").write_text(json.dumps(rec, indent=1))
    return rec


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # recorded as a failed file, never silently filled
        return e


# --------------------------------------------------------------------------------------------
# 2. NOAA ISD-Lite meteorology (§2 rule 7, §3.1 features)
# --------------------------------------------------------------------------------------------
def isd_history(paths: Paths) -> pd.DataFrame:
    out = paths.isd_raw / "isd-history.csv"
    if not out.exists():
        out.write_bytes(http_get(f"{ISD_BASE}/isd-history.csv").content)
    h = pd.read_csv(out, dtype={"USAF": str, "WBAN": str})
    h = h.rename(columns={"LAT": "lat", "LON": "lon", "ELEV(M)": "elev"})
    h["isd"] = h["USAF"] + "-" + h["WBAN"]
    h = h.dropna(subset=["lat", "lon"])
    h = h[(h["lat"].between(27, 72)) & (h["lon"].between(-32, 45))]
    h = h[(h["BEGIN"] <= 20050101) & (h["END"] >= 20240101)]
    return h.reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def isd_candidates(stations: pd.DataFrame, hist: pd.DataFrame, max_km=50.0, max_dz=500.0, k=3) -> pd.DataFrame:
    """Up to k nearest ISD stations within max_km and max_dz elevation for each AQ station."""
    rows = []
    hl, hn, hz = hist["lat"].values, hist["lon"].values, hist["elev"].values
    for s in stations.itertuples():
        dist = haversine_km(s.lat, s.lon, hl, hn)
        dz = np.abs(np.nan_to_num(hz, nan=0.0) - (0.0 if pd.isna(s.alt) else s.alt))
        ok = np.where((dist <= max_km) & (dz <= max_dz))[0]
        for j in ok[np.argsort(dist[ok])][:k]:
            rows.append({"station": s.station, "isd": hist["isd"].iat[j], "dist_km": float(dist[j]),
                         "dz_m": float(dz[j])})
    return pd.DataFrame(rows)


def _parse_isd_lite(raw: bytes) -> pd.DataFrame:
    cols = ["y", "m", "d", "h", "t", "td", "slp", "wd", "ws", "sky", "p1", "p6"]
    df = pd.read_csv(io.BytesIO(gzip.decompress(raw)), sep=r"\s+", header=None, names=cols,
                     na_values=[-9999])
    for c in ["t", "td", "slp", "ws", "p1", "p6"]:
        df[c] = df[c] / 10.0
    df["date"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
    return df


def isd_daily_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """UTC calendar-day features from ISD-Lite hours. A valid met day has >=6 hours with both
    temperature and wind speed (§2 rule 7)."""
    h = hourly.copy()
    calm = (h["wd"] == 0) | (h["ws"] == 0)
    wd = np.radians(h["wd"].where(~calm))
    h["u"] = -h["ws"] * np.sin(wd)
    h["v"] = -h["ws"] * np.cos(wd)
    h.loc[calm, ["u", "v"]] = 0.0
    h["calm"] = calm.astype(float).where(h["ws"].notna())
    es = 6.112 * np.exp(17.62 * h["t"] / (243.12 + h["t"]))
    e = 6.112 * np.exp(17.62 * h["td"] / (243.12 + h["td"]))
    h["rh"] = (100 * e / es).clip(0, 100)
    h["both"] = h["t"].notna() & h["ws"].notna()
    g = h.groupby("date")
    f = pd.DataFrame({
        "t_mean": g["t"].mean(), "t_max": g["t"].max(), "t_min": g["t"].min(),
        "td_mean": g["td"].mean(), "rh_mean": g["rh"].mean(), "slp_mean": g["slp"].mean(),
        "ws_mean": g["ws"].mean(), "u_mean": g["u"].mean(), "v_mean": g["v"].mean(),
        "calm_share": g["calm"].mean(), "n_valid": g["both"].sum(),
    })
    p1 = g["p1"].sum(min_count=1)
    p6 = g["p6"].sum(min_count=1)
    f["precip"] = p1.where(p1.notna(), p6)
    f["precip_reported"] = f["precip"].notna().astype(float)
    f["dtr"] = f["t_max"] - f["t_min"]
    f = f.asfreq("D")
    f["slp_change"] = f["slp_mean"].diff()
    f["ws_prev"] = f["ws_mean"].shift(1)
    f["precip_prev"] = f["precip"].shift(1)
    f["met_valid"] = f["n_valid"].fillna(0) >= 6
    return f.reset_index()


def download_isd_station(paths: Paths, isd: str, years) -> pd.DataFrame | None:
    """Daily features for one ISD station. Years return 'ok', 'absent' (HTTP 404: no file exists) or
    raise; a station is cached only when every year is ok or absent, with the status per year recorded
    next to it, so a transient failure is retried on the next run instead of becoming missing data."""
    out = paths.isd_daily / f"{isd}.parquet"
    status_file = paths.isd_daily / f"{isd}_years.json"
    if out.exists() and status_file.exists() and \
            {str(y) for y in years} <= set(json.loads(status_file.read_text())):   # covers every requested year
        try:
            return pd.read_parquet(out)
        except Exception:                                     # truncated by an interrupted run: refetch
            out.unlink(missing_ok=True); status_file.unlink(missing_ok=True)
    parts, status = [], {}
    for y in years:
        r = http_get(f"{ISD_BASE}/isd-lite/data/{y}/{isd}-{y}.gz", timeout=120)   # retries inside; raises on failure
        if r.status_code == 404:
            status[y] = "absent"
            continue
        parts.append(_parse_isd_lite(r.content))                                     # a parse error raises
        status[y] = "ok"
    f = isd_daily_features(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()
    if len(f):
        f["isd"] = isd
    tmp = out.with_suffix(".tmp")
    f.to_parquet(tmp, index=False)
    tmp.replace(out)                                          # atomic: never a half-written cache file
    status_file.write_text(json.dumps(status))                # completion record written last
    return f if len(f) else None


# --------------------------------------------------------------------------------------------
# 3. Building the analysis set (§2 rules 1-7)
# --------------------------------------------------------------------------------------------
def load_daily(paths: Paths) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(paths.daily.glob("??.parquet"))]
    d = pd.concat(parts, ignore_index=True)
    inv = {v: k for k, v in POLLUTANTS.items()}
    d["pollutant"] = d["Pollutant"].map(inv)
    d["date"] = pd.to_datetime(d["date"])
    d["value"] = d["value"].clip(lower=0)
    return d.drop(columns=["Pollutant"])


def station_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per sampling point: latest-year metadata, plus history flags (§2 rule 4)."""
    m = meta.sort_values("Year")
    last = m.groupby("sp").tail(1).set_index("sp")
    def max_disp_m(g):
        la, lo = g["Latitude"].values, g["Longitude"].values
        if len(la) < 2:
            return 0.0
        d = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        return float(np.nanmax(d) * 1000)
    hist = m.groupby("sp").apply(lambda g: pd.Series({
        "max_displacement_m": max_disp_m(g),
        "n_method": g["mmethod"].dropna().nunique()}))
    out = last.join(hist)
    out["moved"] = out["max_displacement_m"] > RELOCATION_M      # deviation 4: >250 m between positions
    out["method_change"] = out["n_method"] > 1
    # year of the latest process start within 2014-2024: used as the break year in sensitivity 7
    pb = pd.to_datetime(m["Process Activity Begin"], errors="coerce").dt.year
    ob = pd.to_datetime(m["Operational Activity Begin"], errors="coerce").dt.year
    pb = pb.where(pb.between(2014, 2024), ob)
    m2 = m.assign(pb=pb)
    by = m2[m2["pb"].between(2014, 2024)].groupby("sp")["pb"].max()
    out["break_year"] = by.reindex(out.index)
    out = out.rename(columns={"Latitude": "lat", "Longitude": "lon", "Altitude": "alt"})
    return out.reset_index()


def coverage_by_year(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.assign(year=daily["date"].dt.year, ok=daily["value"].notna())
    g = d.groupby(["sp", "pollutant", "year"])["ok"].sum().reset_index(name="valid_days")
    g["days_in_year"] = np.where(pd.to_datetime(g["year"].astype(str) + "-12-31").dt.dayofyear == 366, 366, 365)
    g["share"] = g["valid_days"] / g["days_in_year"]
    return g


def build_manifest(daily: pd.DataFrame, spmeta: pd.DataFrame, train_share=TRAIN_DAY_SHARE) -> pd.DataFrame:
    """Choose one sampling point per station × pollutant (most valid days 2013-2024) and apply the
    eligibility rules. Returns the manifest with an 'excluded_reason' column (empty = included)."""
    cov = coverage_by_year(daily)
    tr = cov[cov["year"].isin(TRAIN_YEARS)]
    tot = tr.groupby(["sp", "pollutant"])["valid_days"].sum().reset_index(name="valid_days_train")
    good = tr[tr["share"] >= train_share].groupby(["sp", "pollutant"])["year"].apply(set).reset_index(name="good_years")
    s = tot.merge(good, on=["sp", "pollutant"], how="left")
    s["good_years"] = s["good_years"].apply(lambda x: x if isinstance(x, set) else set())
    s = s.merge(spmeta[["sp", "station", "station_name", "cc", "lat", "lon", "alt", "stype", "area", "mtype",
                        "moved", "method_change", "break_year"]], on="sp", how="left")
    s = s.sort_values("valid_days_train", ascending=False).drop_duplicates(["station", "pollutant"])
    s["n_good_years"] = s["good_years"].apply(len)
    reason = np.full(len(s), "", dtype=object)
    reason[s["cc"].isna().values] = "no metadata"
    reason[(reason == "") & ~s["cc"].isin(EU27).values] = "not EU-27"
    reason[(reason == "") & ~s["mtype"].isin(["automatic", "active"]).values] = "not a fixed measurement"
    elig = s["good_years"].apply(lambda g: len(g) >= 8 and 2023 in g and 2024 in g).values
    reason[(reason == "") & ~elig] = "fewer than 8 good training years or 2023/2024 missing"
    reason[(reason == "") & (s["moved"].fillna(False) | s["method_change"].fillna(False)).values] = \
        "station moved or method changed"
    s["excluded_reason"] = reason
    # sensitivity 4 (training share >=85%) and sensitivity 7 (history changes handled as breaks)
    good85 = tr[tr["share"] >= 0.85].groupby(["sp", "pollutant"])["year"].apply(set)
    s["elig85"] = [len(g) >= 8 and {2023, 2024} <= g for g in
                   (good85.get((a, b), set()) for a, b in zip(s["sp"], s["pollutant"]))]
    s["history_only"] = (s["excluded_reason"] == "station moved or method changed")
    s["good_years"] = s["good_years"].apply(lambda g: ",".join(str(y) for y in sorted(g)))
    return s.reset_index(drop=True)


def match_met(manifest: pd.DataFrame, cands: pd.DataFrame, met_cov: pd.DataFrame, years_needed=None,
              max_km=50.0) -> pd.DataFrame:
    """Nearest candidate ISD station that qualifies (deviation 5): >=75% valid met days in at least 10 of
    the 12 training years 2013-2024 including 2023 and 2024, and >=10 weather-pool years (2005-2024) with
    >=90% valid days (the pool-year rule of weather_year_frames). Days without valid met are never used."""
    c = met_cov.copy()
    tr = c[c["year"].isin(TRAIN_YEARS)]
    good_tr = tr[tr["share"] >= TRAIN_DAY_SHARE].groupby("isd")["year"].apply(set)
    pool = c[c["year"].isin(MAIN_POOL) & c["pool_ok"]].groupby("isd")["year"].nunique()
    ok = {i for i, ys in good_tr.items() if len(ys) >= 10 and {2023, 2024} <= ys and pool.get(i, 0) >= 10}
    cc = cands[(cands["dist_km"] <= max_km) & cands["isd"].isin(ok)].sort_values("dist_km")
    best = cc.drop_duplicates("station")[["station", "isd", "dist_km", "dz_m"]]
    return manifest.merge(best, on="station", how="left")


def met_coverage(met: pd.DataFrame) -> pd.DataFrame:
    """Per ISD station and year: valid-day share, and whether the year qualifies as a weather-pool year
    under exactly the rule used by weather_year_frames (>=97% valid, no gap > 3 days, 365-day calendar)."""
    m = met.assign(date=pd.to_datetime(met["date"]))
    m = m[~((m["date"].dt.month == 2) & (m["date"].dt.day == 29))]
    m = m.assign(year=m["date"].dt.year)
    rows = []
    for (isd, y), g in m.groupby(["isd", "year"]):
        v = g.sort_values("date")["met_valid"].fillna(False).values.astype(bool)
        rows.append({"isd": isd, "year": y, "valid_days": int(v.sum()), "share": v.sum() / 365.0,
                     "pool_ok": pool_year_ok(v) and len(v) >= 365})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 4. Model (§3.1): log C = a_year + f(W, d) + e, backfitting with LightGBM
# --------------------------------------------------------------------------------------------
DEFAULT_HP = {"num_leaves": 31, "n_estimators": 300, "learning_rate": 0.05, "min_child_samples": 20,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.9}


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    doy = d.dt.dayofyear.values
    df = df.copy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["dow"] = d.dt.dayofweek.values
    df["year"] = d.dt.year.values
    df["season"] = ((d.dt.month.values % 12) // 3)  # 0=DJF 1=MAM 2=JJA 3=SON
    return df


@dataclass
class SeriesModel:
    a: dict                       # year -> a_y (log scale, includes centring)
    se: dict                      # year -> s.e. of a_y (effective-n corrected)
    booster: object               # fitted f (LightGBM) or GAM
    f_offset: float               # centring constant subtracted from raw f
    resid: pd.DataFrame           # date, resid, season, fitted (training residuals)
    kind: str = "lgbm"
    iters: int = 0
    terciles: tuple = field(default_factory=tuple)

    def f(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_raw(self.booster, X, self.kind) - self.f_offset


def year_se(r: pd.DataFrame) -> dict:
    """Standard error of each yearly level from that year's residuals, using an effective sample size
    for lag-one autocorrelation computed on consecutive calendar days only."""
    se = {}
    pooled_sd = float(np.std(r["resid"].values, ddof=1))
    for yr, g in r.groupby("year"):
        g = g.sort_values("date")
        e = g["resid"].values
        d = g["date"].values.astype("datetime64[D]").astype(np.int64)
        pairs = np.diff(d) == 1
        n = len(e)
        rho = 0.0
        if n > 10 and pairs.sum() > 10:
            rho = float(np.corrcoef(e[:-1][pairs], e[1:][pairs])[0, 1])
            rho = 0.0 if not np.isfinite(rho) else rho
        rho = min(max(rho, 0.0), 0.95)
        # years with very few days (n < 5) use the station's pooled residual SD, so they get a large
        # but finite s.e. (and little weight in the trend fit) instead of an undefined one
        sd = float(np.std(e, ddof=1)) if n >= 5 else pooled_sd
        n_eff = max(n * (1 - rho) / (1 + rho), 1.0)
        se[int(yr)] = float(sd / math.sqrt(n_eff))
    return se


def _fit_raw(X, y, hp, kind, seed):
    if kind == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(**hp, random_state=seed, n_jobs=1, verbose=-1)
        m.fit(X, y)
        return m
    if kind == "gam":
        from pygam import LinearGAM, s, f as fterm
        cols = list(X.columns)
        terms = None
        for i, c in enumerate(cols):
            t = fterm(i) if c == "dow" else s(i, n_splines=8)
            terms = t if terms is None else terms + t
        m = LinearGAM(terms).fit(X.fillna(X.median()).values, y)
        m._medians = X.median()
        return m
    raise ValueError(kind)


def _predict_raw(m, X, kind):
    if kind == "lgbm":
        return m.predict(X)
    return m.predict(X.fillna(m._medians).values)


def fit_series(train: pd.DataFrame, hp=None, kind="lgbm", seed=SEED, tol=5e-3, max_iter=20) -> SeriesModel:
    """train: rows with date, value (valid daily mean), FEATURES (met may be NaN), year, season."""
    hp = hp or DEFAULT_HP
    y = np.log(train["value"].values + OFFSET)
    years = train["year"].values
    X = train[FEATURES]
    a = pd.Series(y).groupby(years).mean()
    f_hat = np.zeros_like(y)
    it = 0
    for it in range(1, max_iter + 1):
        m = _fit_raw(X, y - a.reindex(years).values, hp, kind, seed)
        f_hat = _predict_raw(m, X, kind)
        a_new = pd.Series(y - f_hat).groupby(years).mean()
        delta = float(np.max(np.abs(a_new - a)))
        a = a_new
        if delta < tol:
            break
    off = float(f_hat.mean())
    a = a + off                                   # centring: mean f over training rows = 0
    fitted = a.reindex(years).values + (f_hat - off)
    resid = y - fitted
    r = pd.DataFrame({"date": pd.to_datetime(train["date"].values), "resid": resid,
                      "season": train["season"].values, "fitted": fitted, "year": years})
    se = year_se(r)
    terc = tuple(np.quantile(fitted, [1 / 3, 2 / 3]))
    return SeriesModel(a={int(k): float(v) for k, v in a.items()}, se=se, booster=m, f_offset=off,
                       resid=r, kind=kind, iters=it, terciles=terc)


# --------------------------------------------------------------------------------------------
# 5. Weather + residual resampling (§3.2)
# --------------------------------------------------------------------------------------------
MET_COLS = [c for c in FEATURES if c not in ("doy_sin", "doy_cos", "dow")]
CORE_MET = ["t_mean", "ws_mean"]


def max_invalid_run(valid: np.ndarray) -> int:
    """Longest run of consecutive False values."""
    best = cur = 0
    for v in valid:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


def pool_year_ok(valid: np.ndarray, min_valid=0.97, max_gap=3) -> bool:
    return len(valid) >= 365 and valid.mean() >= min_valid and max_invalid_run(valid) <= max_gap


def weather_year_frames(met: pd.DataFrame, pool_years, min_valid=0.97, max_gap=3) -> dict:
    """year -> feature frame for that calendar year on a 365-day calendar (29 February removed, so
    every simulated year aligns day-by-day with any target year; deviation 8).
    A pool year needs >=97% valid met days and no run of more than 3 consecutive invalid days. Only the
    invalid days are filled, by linear interpolation in time from the nearest valid days (inputs to the
    simulation only, never reported as data). Missing secondary features on otherwise valid days stay
    missing, exactly as in training, where the model learned how to route them."""
    m = add_calendar(met.assign(date=pd.to_datetime(met["date"])))
    out = {}
    for y in pool_years:
        g = m[m["year"] == y].sort_values("date").reset_index(drop=True)
        g = g[~((g["date"].dt.month == 2) & (g["date"].dt.day == 29))].reset_index(drop=True)
        valid = g["met_valid"].values.astype(bool)
        if len(g) < 365 or not pool_year_ok(valid, min_valid, max_gap):
            continue
        g = g.copy()
        g.loc[~valid, MET_COLS] = np.nan
        filled = g[MET_COLS].interpolate(limit_direction="both")
        g.loc[~valid, MET_COLS] = filled.loc[~valid, MET_COLS].values
        if g[CORE_MET].isna().any().any():
            continue
        out[y] = g
    return out


def met_support(train: pd.DataFrame, wy: dict) -> float:
    """Share of weather-pool days whose core meteorology lies inside the training range (§3.2)."""
    if not wy:
        return np.nan
    lo = train[CORE_MET + ["rh_mean", "slp_mean"]].quantile(0.0)
    hi = train[CORE_MET + ["rh_mean", "slp_mean"]].quantile(1.0)
    allg = pd.concat(wy.values())[CORE_MET + ["rh_mean", "slp_mean"]]
    inside = ((allg >= lo) & (allg <= hi)) | allg.isna()
    return float(inside.all(axis=1).mean())


class ResidualSampler:
    """Season- and tercile-stratified moving-block bootstrap of observed training residuals.

    For every block of a simulated year we draw one observed residual block for each of the three
    tercile classes; the tercile actually used is chosen later from the predicted concentration at the
    block start, so residuals follow any shift of the level (trend draws, headroom search)."""

    def __init__(self, model: SeriesModel, block=7, stratify=True):
        r = model.resid.sort_values("date").reset_index(drop=True)
        self.e = r["resid"].values
        self.block = block
        self.stratify = stratify
        self.t1, self.t2 = model.terciles
        dates = r["date"].values.astype("datetime64[D]").astype(np.int64)
        n = len(r)
        contiguous = np.zeros(n, dtype=bool)
        if n >= block:
            # a block may start at i only if each of the next block-1 steps is exactly one day
            step = np.r_[(np.diff(dates) == 1).astype(int), 0]
            run = np.convolve(step, np.ones(block - 1, dtype=int), mode="full")[block - 2: block - 2 + n]
            contiguous = run == (block - 1)
            contiguous[n - block + 1:] = False
        self.starts_all = np.where(contiguous)[0]
        if len(self.starts_all) == 0:
            raise ValueError("no contiguous residual blocks")
        terc = np.digitize(r["fitted"].values, [self.t1, self.t2])
        seas = r["season"].values
        self.groups = {}
        for s in range(4):
            for t in range(3):
                idx = self.starts_all[(seas[self.starts_all] == s) & (terc[self.starts_all] == t)]
                if len(idx) < 5:
                    idx = self.starts_all[seas[self.starts_all] == s]
                if len(idx) < 5:
                    idx = self.starts_all
                self.groups[(s, t)] = idx

    def draw(self, season: np.ndarray, n_paths: int, rng) -> np.ndarray:
        """Residual blocks, shape (n_paths, n_blocks, 3, block)."""
        L = len(season)
        nb = int(math.ceil(L / self.block))
        out = np.empty((n_paths, nb, 3, self.block))
        for b in range(nb):
            s = int(season[min(b * self.block, L - 1)])
            if self.stratify:
                for t in range(3):
                    st = rng.choice(self.groups[(s, t)], size=n_paths)
                    out[:, b, t, :] = self.e[st[:, None] + np.arange(self.block)[None, :]]
            else:
                st = rng.choice(self.starts_all, size=n_paths)
                out[:, b, :, :] = self.e[st[:, None] + np.arange(self.block)[None, :]][:, None, :]
        return out

    def assemble(self, blocks: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Residual paths (n_paths, len(pred)) using the tercile of `pred` at each block start."""
        L = len(pred)
        nb = blocks.shape[1]
        starts = np.minimum(np.arange(nb) * self.block, L - 1)
        terc = np.digitize(pred[starts], [self.t1, self.t2])
        E = blocks[:, np.arange(nb), terc, :]
        return E.reshape(blocks.shape[0], -1)[:, :L]


def outcomes(logc: np.ndarray, pol: str, mask: np.ndarray | None = None) -> dict:
    """logc: (..., days) simulated log(C+offset). Annual mean, exceedance-day count and attainment flags.
    With `mask` (bool, days), 'exc_obs' counts exceedances only on those days (for scoring counts
    against a partly observed year)."""
    c = np.clip(np.exp(logc) - OFFSET, 0, None)
    lim = LIMITS[pol]
    mean = c.mean(axis=-1)
    over = c > lim["daily"]
    exc = over.sum(axis=-1)
    att_a = mean <= lim["annual"]
    att_d = exc <= lim["allow"]
    out = {"mean": mean, "exc": exc, "att_annual": att_a, "att_daily": att_d, "att_joint": att_a & att_d}
    if mask is not None:
        assert over.shape[-1] == len(mask) == 365, "simulation and mask must share the 365-day calendar"
        out["exc_obs"] = over[..., mask].sum(axis=-1)
    return out


@dataclass
class Sims:
    sampler: ResidualSampler
    years: list
    f: list           # per weather year: f over the year (days)
    blocks: list      # per weather year: residual blocks (n_paths, nb, 3, block)


def base_sims(model: SeriesModel, wy: dict, n_res: int, rng, block=7, stratify=True) -> Sims:
    sampler = ResidualSampler(model, block=block, stratify=stratify)
    ys, fs, bs = [], [], []
    for y, g in wy.items():
        ys.append(y)
        fs.append(model.f(g[FEATURES]))
        bs.append(sampler.draw(g["season"].values, n_res, rng))
    return Sims(sampler, ys, fs, bs)


def simulate(sims: Sims, level: float, pol: str, n_paths=None, mask=None) -> list[dict]:
    """Outcomes for every weather year at one log level (residuals re-conditioned on that level)."""
    res = []
    for fy, B in zip(sims.f, sims.blocks):
        Bk = B if n_paths is None else B[:n_paths]
        pred = level + fy
        E = sims.sampler.assemble(Bk, pred)
        res.append(outcomes(pred[None, :] + E, pol, mask=mask))
    return res


def attain_prob(sims: Sims, level: float, pol: str, weights=None, shifts=None, n_paths=None) -> dict:
    """P(attainment) per metric for each extra log shift (vector). Residuals follow each shifted level."""
    shifts = np.atleast_1d(0.0 if shifts is None else shifts)
    w = np.ones(len(sims.years)) if weights is None else np.asarray(weights, float)
    out = {k: np.zeros(len(shifts)) for k in ["att_annual", "att_daily", "att_joint"]}
    for i, s in enumerate(shifts):
        for wj, o in zip(w, simulate(sims, level + s, pol, n_paths=n_paths)):
            for k in out:
                out[k][i] += wj * o[k].mean()
    return {k: v / w.sum() for k, v in out.items()}


HEADROOM_GRID = np.round(np.arange(-4.0, 2.0001, 0.02), 4)


def headroom_all(sims: Sims, level: float, pol: str, grid=HEADROOM_GRID, n_paths=50) -> dict:
    """E3 for every metric and q in {0.9, 0.5} from one pass over a grid of log shifts.

    Rule (fixed before the main run): the required shift s* is the largest grid value such that
    P(attain) >= q at every grid value <= s* (the first downward crossing, scanning from the lowest
    level). Because residual blocks follow the level by tercile, P need not be exactly monotone; we
    report 'E3_nonmono' = largest increase of P after its first fall below q (a diagnostic).
    Censoring: if P < q already at the lowest shift (a >98% cut), E3 is left-censored ('left', value
    NaN); if P >= q over the whole grid, it is right-censored ('right', value = exp(max)-1, a lower
    bound). Intervals are only given for uncensored values."""
    out = {}
    probs = {k: [] for k in ["att_annual", "att_daily", "att_joint"]}
    for s in grid:
        p = attain_prob(sims, level, pol, shifts=[s], n_paths=n_paths)
        for k in probs:
            probs[k].append(p[k][0])
    step = float(grid[1] - grid[0])
    cache = {}

    def fine(lo_s):
        """P on a 10x finer grid between lo_s and lo_s + step (cached per interval)."""
        if lo_s not in cache:
            fg = np.round(lo_s + step * np.arange(0, 11) / 10, 6)
            res = {k: [] for k in probs}
            for s in fg:
                p = attain_prob(sims, level, pol, shifts=[s], n_paths=n_paths)
                for k in res:
                    res[k].append(p[k][0])
            cache[lo_s] = (fg, {k: np.asarray(v) for k, v in res.items()})
        return cache[lo_s]
    for k, pv in probs.items():
        pv = np.asarray(pv)
        for q in (0.9, 0.5):
            tag = f"E3_{k}_{int(q * 100)}"
            below = np.where(pv < q)[0]
            if len(below) and below[0] == 0:
                out[tag], out[f"{tag}_cens"], out[f"{tag}_nonmono"] = np.nan, "left", np.nan
                continue
            if not len(below):
                out[tag], out[f"{tag}_cens"], out[f"{tag}_nonmono"] = float(np.exp(grid[-1]) - 1), "right", 0.0
                continue
            i = below[0] - 1
            fg, fp = fine(float(grid[i]))                 # refine inside [grid[i], grid[i+1]]
            fb = np.where(fp[k] < q)[0]
            s_star = float(fg[fb[0] - 1]) if len(fb) and fb[0] > 0 else float(fg[-1] if not len(fb) else grid[i])
            out[tag] = float(np.exp(s_star) - 1)
            out[f"{tag}_cens"] = ""
            after = pv[below[0]:]
            out[f"{tag}_nonmono"] = float(max(0.0, after.max() - pv[below[0]]))
    return out


def headroom(sims: Sims, level: float, pol: str, metric: str, q: float, lo=-4.0, hi=2.0, tol=0.005) -> float:
    """E3: proportional change of the underlying level (exp(shift)-1) at which P(attainment) = q,
    by bisection on the log shift in [lo, hi]. P decreases with the shift. Returns -1..+inf;
    NaN if q cannot be reached inside the bracket (reported as a count)."""
    p = lambda s: attain_prob(sims, level, pol, shifts=[s], n_paths=50)[metric][0]
    plo, phi = p(lo), p(hi)
    if plo < q:
        return np.nan            # even a 98% cut does not reach q
    if phi >= q:
        return float(np.exp(hi) - 1)   # reached even at +639%: reported as lower bound
    a, b = lo, hi
    while b - a > tol:
        mid = (a + b) / 2
        if p(mid) >= q:
            a = mid
        else:
            b = mid
    return float(np.exp(a) - 1)


# --------------------------------------------------------------------------------------------
# 6. Trend models for E2 (§3.1, §3.3)
# --------------------------------------------------------------------------------------------
def trend_draws(a: dict, se: dict, target_year: int, n: int, rng, method="wls", exclude=(2020,),
                break_year=None) -> np.ndarray:
    yrs = np.array(sorted(y for y in a if y not in exclude), dtype=float)
    av = np.array([a[int(y)] for y in yrs])
    sv = np.array([se[int(y)] for y in yrs])
    if len(yrs) < 4:
        return np.full(n, np.nan)
    X = np.column_stack([np.ones_like(yrs), yrs - 2020.0])
    x0 = [1.0, target_year - 2020.0]
    if break_year is not None:
        X = np.column_stack([X, (yrs >= break_year).astype(float)])
        x0 = x0 + [1.0]
    x0 = np.array(x0)
    if method == "wls":
        # the same hierarchical predictive sampler as trend_param_draws (one trend-parameter draw per
        # value plus year-level noise tau), so E2, its interval, E5, the bootstrap and the hindcast all
        # use one predictive distribution
        return trend_param_draws(a, se, target_year, n, 1, rng, exclude=exclude, break_year=break_year).ravel()
    if method == "theilsen":
        # robust alternative: Theil-Sen line, bootstrap over years, plus a resampled year-to-year
        # deviation and the estimation error of the yearly levels (mean s.e.)
        from scipy.stats import theilslopes
        out = np.empty(n)
        idx = np.arange(len(yrs))
        se_bar = float(np.sqrt(np.mean(sv ** 2)))
        for i in range(n):
            b = rng.choice(idx, size=len(idx), replace=True)
            if len(np.unique(yrs[b])) < 3:
                b = idx
            sl, ic, _, _ = theilslopes(av[b], yrs[b] - 2020.0)
            res = av - (ic + sl * (yrs - 2020.0))
            out[i] = ic + sl * (target_year - 2020.0) + rng.choice(res) + se_bar * rng.standard_normal()
        return out
    raise ValueError(method)


# --------------------------------------------------------------------------------------------
# 7. Observed outcomes for scoring (§2 rule 5)
# --------------------------------------------------------------------------------------------
def observed_outcome(series: pd.DataFrame, year: int, pol: str) -> dict:
    """Observed attainment in one year under the directive's rules. The count of exceedance days is
    returned twice: over all observed days ('obs_exc_obsdays', with the day mask used to score
    forecasts on the same days) and as a full-year count only when no day is missing."""
    g = series[pd.to_datetime(series["date"]).dt.year == year]
    n_days = 366 if (year % 4 == 0) else 365
    g = g.dropna(subset=["value"])
    v = g["value"].values
    cov = len(v) / n_days
    lim = LIMITS[pol]
    # day mask on the 365-day simulation calendar (29 February has no slot and is not scored)
    mask = np.zeros(365, dtype=bool)
    if len(g):
        dts = pd.to_datetime(g["date"])
        keep = ~((dts.dt.month == 2) & (dts.dt.day == 29))
        doy = dts[keep].dt.dayofyear.values
        leap = (year % 4 == 0)
        doy = np.where(leap & (dts[keep].dt.month.values > 2), doy - 1, doy)
        mask[doy - 1] = True
    out = {"coverage": cov, "obs_mean": np.nan, "obs_att_annual": np.nan, "obs_exc_full": np.nan,
           "obs_exc_obsdays": np.nan, "obs_att_daily": np.nan, "obs_att_joint": np.nan, "_mask": mask}
    if len(v) == 0:
        return out
    exc = int((v > lim["daily"]).sum())
    missing = n_days - len(v)
    dts = pd.to_datetime(g["date"])
    not29 = ~((dts.dt.month == 2) & (dts.dt.day == 29)).values
    out["obs_exc_obsdays"] = int((v[not29] > lim["daily"]).sum())   # scored on the 365-day calendar
    if missing == 0:
        out["obs_exc_full"] = exc
    if cov >= COVERAGE_MIN:
        out["obs_mean"] = float(v.mean())
        out["obs_att_annual"] = float(v.mean() <= lim["annual"])
    if exc > lim["allow"]:
        out["obs_att_daily"] = 0.0
    elif exc + missing <= lim["allow"]:
        out["obs_att_daily"] = 1.0          # cannot fail even if every missing day exceeded
    if not np.isnan(out["obs_att_annual"]) and not np.isnan(out["obs_att_daily"]):
        out["obs_att_joint"] = float(out["obs_att_annual"] and out["obs_att_daily"])
    elif out["obs_att_annual"] == 0.0 or out["obs_att_daily"] == 0.0:
        out["obs_att_joint"] = 0.0
    return out


# --------------------------------------------------------------------------------------------
# 8. Per-series analysis: main (E1, E2, E3, E5), bootstrap, hindcast (§3.1-3.5)
# --------------------------------------------------------------------------------------------
def key_seed(key: dict) -> int:
    """Stable per-series seed (the built-in str hash is salted per process)."""
    return zlib.crc32((key["station"] + "|" + key["pollutant"]).encode()) % 100000


def prepare_series(series: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    s = series[["date", "value"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    m = met.copy()
    m["date"] = pd.to_datetime(m["date"])
    d = s.merge(m, on="date", how="left")
    d = add_calendar(d)
    d.loc[~d["met_valid"].fillna(False).astype(bool), MET_COLS] = np.nan
    d["met_valid"] = d["met_valid"].fillna(False).astype(bool)
    return d


def training_rows(d: pd.DataFrame, last_year: int) -> pd.DataFrame:
    return d[(d["year"] >= 2013) & (d["year"] <= last_year) & d["value"].notna() & d["met_valid"]]


def break_supported(yrs, break_year) -> bool:
    yrs = np.asarray(yrs)
    return len(yrs) >= 4 and (yrs >= break_year).sum() >= 1 and (yrs < break_year).sum() >= 1


def dl_tau2(y: np.ndarray, s: np.ndarray, X: np.ndarray) -> float:
    """Between-year variance of the yearly levels around the trend line: the DerSimonian-Laird method-of-moments
    estimator for meta-regression, tau2 = max(0, (Q - (n - p)) / (tr(W) - tr((X'WX)^-1 X'W^2 X))), with W = 1/s^2
    and Q the weighted residual sum of squares of the fixed-effect fit (deviation 17: the first implementation
    subtracted the sum of all sampling variances from an unweighted residual sum of squares, which is biased low)."""
    w = 1.0 / s ** 2
    XtWX = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    q = float(np.sum(w * (y - X @ beta) ** 2))
    denom = float(w.sum() - np.trace(np.linalg.solve(XtWX, X.T @ ((w ** 2)[:, None] * X))))
    return max(0.0, (q - (len(y) - X.shape[1])) / denom) if denom > 0 else 0.0


def trend_param_draws(a: dict, se: dict, target_year: int, n_par: int, n_within: int, rng, exclude=(2020,),
                      break_year=None) -> np.ndarray:
    """Draws of the 2030 level grouped by trend-parameter draw: shape (n_par, n_within). Each row is the
    conditional predictive N(x0'beta_j, tau^2) for one draw beta_j of the WLS parameters, so the spread of
    the row-wise attainment probabilities gives an interval for E2 (§3.5)."""
    yrs = np.array(sorted(y for y in a if y not in exclude), dtype=float)
    av = np.array([a[int(y)] for y in yrs]); sv = np.array([se[int(y)] for y in yrs])
    if len(yrs) < 4:
        return np.full((n_par, n_within), np.nan)
    X = np.column_stack([np.ones_like(yrs), yrs - 2020.0]); x0 = [1.0, target_year - 2020.0]
    if break_year is not None:
        if not break_supported(yrs, break_year):
            raise ValueError("unsupported break: needs >= 1 training year on each side and >= 4 years")
        X = np.column_stack([X, (yrs >= break_year).astype(float)]); x0 = x0 + [1.0]
    x0 = np.array(x0)
    tau2 = dl_tau2(av, sv, X)
    W = np.diag(1 / (sv ** 2 + tau2))
    cov = np.linalg.inv(X.T @ W @ X); beta = cov @ X.T @ W @ av
    df = max(len(av) - X.shape[1], 1)
    # Deviation 15 (post-hoc): a Student-t scaling here has no finite moments once exponentiated to concentrations.
    # Use a normal draw whose covariance is inflated to the t's variance, df/(df-2), or 3 when df <= 2.
    inflate = df / (df - 2.0) if df > 2 else 3.0
    z = rng.multivariate_normal(np.zeros(len(beta)), cov * inflate, size=n_par)
    mu = (beta[None, :] + z) @ x0
    return mu[:, None] + np.sqrt(tau2) * rng.standard_normal((n_par, n_within))


def e2_block(model, sims, pol, rng, n_trend, n_trend_res, target=2030, break_year=None) -> dict:
    out = {}
    variants = [("wls", (2020,), "E2"), ("theilsen", (2020,), "E2ts"), ("wls", (), "E2c20")]
    if break_year is not None:
        variants = [v for v in variants if v[0] != "theilsen"]   # Theil-Sen has no break term
    for meth, excl, tag in variants:
        td = trend_draws(model.a, model.se, target, n_trend, rng, method=meth, exclude=excl, break_year=break_year)
        if np.isnan(td).all():
            continue
        p = attain_prob(sims, 0.0, pol, shifts=td, n_paths=n_trend_res)
        for k, v in p.items():
            out[f"{tag}_{k}"] = float(v.mean())
        out[f"{tag}_level_mean"] = float(td.mean())
        if tag == "E2":
            out.update(variance_partition(sims, td, pol, n_trend_res))
            # interval for E2: spread of the probability across trend-parameter draws
            grid = trend_param_draws(model.a, model.se, target, 30, 60, rng, exclude=excl, break_year=break_year)
            if not np.isnan(grid).all():
                pj = attain_prob(sims, 0.0, pol, shifts=grid.ravel(), n_paths=n_trend_res)
                for k, v in pj.items():
                    per = v.reshape(grid.shape).mean(axis=1)
                    out[f"E2_{k}_mc"] = out[f"E2_{k}"]          # single-stage Monte Carlo value (kept for E5)
                    out[f"E2_{k}"] = float(per.mean())            # point estimate consistent with its interval
                    out[f"E2_{k}_lo"], out[f"E2_{k}_hi"] = float(np.quantile(per, 0.05)), float(np.quantile(per, 0.95))
    return out


def run_main(key: dict, series: pd.DataFrame, met: pd.DataFrame, hp=None, n_res=100, n_trend=200,
             n_trend_res=10, seed=SEED, kind="lgbm", pool=MAIN_POOL, sens=True, break_year=None) -> dict:
    """E1 (with interval), E2 (WLS, Theil-Sen, 2020 included), E3, E5 and re-use sensitivities."""
    rng = np.random.default_rng(seed + key_seed(key))
    pol = key["pollutant"]
    d = prepare_series(series, met)
    tr = training_rows(d, 2024)
    model = fit_series(tr, hp=hp, kind=kind, seed=seed)
    wy = weather_year_frames(met, pool)
    if len(wy) < 10:
        return {**key, "status": "fewer than 10 weather-pool years"}
    level = model.a[2024]
    sims = base_sims(model, wy, n_res, rng)
    out = {**key, "status": "ok", "iters": model.iters, "n_train_days": len(tr), "n_pool_years": len(wy),
           "met_support": met_support(tr, wy), "a_2024": level, "se_2024": model.se[2024]}
    for k, v in attain_prob(sims, level, pol).items():
        out[f"E1_{k}"] = float(v[0])
    # Monte Carlo error: repeat the integration with independent residual draws at the same fit
    sims_rep = base_sims(model, wy, n_res, np.random.default_rng(seed + 99 + key_seed(key)))
    for k, v in attain_prob(sims_rep, level, pol).items():
        out[f"E1_{k}_mcse"] = float(abs(v[0] - out[f"E1_{k}"]) / np.sqrt(2))
    # E1 interval from the estimation uncertainty of a_2024 (s.e. approximation, checked by run_boot)
    z = level + model.se[2024] * rng.standard_normal(60)
    pz = attain_prob(sims, level, pol, shifts=z - level, n_paths=50)
    for k, v in pz.items():
        out[f"E1_{k}_lo"], out[f"E1_{k}_hi"] = float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))
    se24 = model.se[2024]
    out.update(headroom_all(sims, level, pol))
    for k in ["att_joint", "att_annual", "att_daily"]:
        for q in (0.9, 0.5):
            tag = f"E3_{k}_{int(q * 100)}"
            e3 = out[tag]
            # attainment depends on level + shift only, so if the true level is a_hat + d the required
            # change is (1 + E3) * exp(-d) - 1; the 5-95% interval follows from d ~ N(0, se^2).
            # No interval for censored values.
            if out[f"{tag}_cens"] == "" and np.isfinite(e3):
                out[f"{tag}_lo"] = float((1 + e3) * np.exp(-1.645 * se24) - 1)
                out[f"{tag}_hi"] = float((1 + e3) * np.exp(1.645 * se24) - 1)
            else:
                out[f"{tag}_lo"] = out[f"{tag}_hi"] = np.nan
    out.update(e2_block(model, sims, pol, rng, n_trend, n_trend_res, break_year=break_year))
    if sens:
        out.update(e1_sensitivities(model, met, key, level, rng, sims, tr))
    out["a_series"] = json.dumps({int(k): round(v, 5) for k, v in model.a.items()})
    out["se_series"] = json.dumps({int(k): round(v, 5) for k, v in model.se.items()})
    return out


def e1_sensitivities(model: SeriesModel, met: pd.DataFrame, key: dict, level: float, rng, sims_main: Sims,
                     tr: pd.DataFrame, n_res=100) -> dict:
    """E1 under the pre-specified variants (§3.6 items 2, 3 and 6c), from the same fitted model."""
    pol = key["pollutant"]
    out = {}

    def put(tag, p):
        for k, v in p.items():
            out[f"S_{tag}_E1_{k}"] = float(v[0])

    for tag, pool in [("pool1995", list(range(1995, 2025))), ("pool2015", list(range(2015, 2025)))]:
        wy = weather_year_frames(met, pool)
        if len(wy) >= 5:
            put(tag, attain_prob(base_sims(model, wy, n_res, rng), level, pol))
            out[f"S_{tag}_n_years"] = len(wy)
    w = [float(y - min(sims_main.years) + 1) for y in sims_main.years]
    put("recency", attain_prob(sims_main, level, pol, weights=w))
    wy = weather_year_frames(met, MAIN_POOL)
    for tag, kw in [("block3", dict(block=3)), ("block14", dict(block=14)), ("unstrat", dict(stratify=False))]:
        put(tag, attain_prob(base_sims(model, wy, n_res, rng, **kw), level, pol))
    if pol == "PM10" and key.get("cc") in DUST_STATES:
        lim = LIMITS[pol]
        acc = {"att_annual": 0.0, "att_daily": 0.0, "att_joint": 0.0}
        for fy, B in zip(sims_main.f, sims_main.blocks):
            pred = level + fy
            c = np.clip(np.exp(pred[None, :] + sims_main.sampler.assemble(B, pred)) - OFFSET, 0, None)
            a_ = c.mean(axis=1) <= lim["annual"]
            d_ = (np.sort(c, axis=1)[:, :-5] > lim["daily"]).sum(axis=1) <= lim["allow"]
            acc["att_annual"] += a_.mean(); acc["att_daily"] += d_.mean(); acc["att_joint"] += (a_ & d_).mean()
        put("dustdrop5", {k: np.array([v / len(sims_main.f)]) for k, v in acc.items()})
    return out


def variance_partition(sims: Sims, tdraws, pol, n_paths) -> dict:
    """E5 (§3.5): law of total variance over nested layers trend T ⊃ weather W ⊃ residual R for the 2030
    annual mean and exceedance count; reported in both orders (trend first, weather first)."""
    out = {}
    stack = {"mean": [], "exc": []}
    for t in tdraws:
        sm = simulate(sims, t, pol, n_paths=n_paths)
        for metric in stack:
            stack[metric].append(np.stack([o[metric] for o in sm]))
    for metric, arr in stack.items():
        Y = np.stack(arr).astype(float)                   # T × W × R
        m_tw = Y.mean(axis=2)
        out.update({f"E5_{metric}_total": float(Y.var()),
                    f"E5_{metric}_trend": float(m_tw.mean(axis=1).var()),
                    f"E5_{metric}_weather": float(m_tw.var(axis=1).mean()),
                    f"E5_{metric}_resid": float(Y.var(axis=2).mean()),
                    f"E5_{metric}_weather_first": float(m_tw.mean(axis=0).var()),
                    f"E5_{metric}_trend_second": float(m_tw.var(axis=0).mean())})
    return out


def refresh_residuals(model: SeriesModel, tr: pd.DataFrame) -> SeriesModel:
    """Recompute training residuals of a (bootstrap-)fitted model on the original training frame, so
    residual blocks are taken from real consecutive days with unique dates."""
    y = np.log(tr["value"].values + OFFSET)
    years = tr["year"].values
    fitted = np.array([model.a.get(int(v), np.nan) for v in years]) + model.f(tr[FEATURES])
    ok = ~np.isnan(fitted)
    r = pd.DataFrame({"date": pd.to_datetime(tr["date"].values), "resid": y - fitted, "season": tr["season"].values,
                      "fitted": fitted, "year": years})[ok]
    model.resid = r.reset_index(drop=True)
    model.terciles = tuple(np.quantile(r["fitted"], [1 / 3, 2 / 3]))
    model.se = year_se(model.resid)                # s.e. from real consecutive days, not duplicated dates
    return model


def _year_blocks(tr: pd.DataFrame, rng, block_days=30) -> pd.DataFrame:
    """Resample 30-day blocks of training rows with replacement within each year."""
    parts = []
    for _, g in tr.groupby("year"):
        doy = pd.to_datetime(g["date"]).dt.dayofyear.values
        bid = (doy - 1) // block_days
        ids = np.unique(bid)
        pick = rng.choice(ids, size=len(ids), replace=True)
        parts += [g[bid == b] for b in pick]
    return pd.concat(parts, ignore_index=True)


def run_boot(key: dict, series: pd.DataFrame, met: pd.DataFrame, hp=None, reps=50, seed=SEED,
             n_res=30, n_trend=50, n_trend_res=5) -> dict:
    """Full refitting bootstrap of the yearly levels (§3.1): 30-day blocks resampled within years, the
    whole model refit each time; returns 5-95% ranges of E1, E2 and E3 (pre-set 20% subsample)."""
    rng = np.random.default_rng(seed + 23 + key_seed(key))
    pol = key["pollutant"]
    d = prepare_series(series, met)
    tr = training_rows(d, 2024)
    wy = weather_year_frames(met, MAIN_POOL)
    if len(wy) < 10:
        return {**key, "status": "fewer than 10 weather-pool years"}
    metrics = ["att_annual", "att_daily", "att_joint"]
    vals = {"a_2024": []}
    for k in metrics:
        vals[f"E1_{k}"] = []; vals[f"E2_{k}"] = []
        for q in (90, 50):
            vals[f"E3_{k}_{q}"] = []
    coarse = np.round(np.arange(-4.0, 2.0001, 0.05), 4)   # refined 10x around the crossing (step 0.005)
    for _ in range(reps):
        m = fit_series(_year_blocks(tr, rng), hp=hp, seed=seed)
        if 2024 not in m.a:
            continue
        # Deviation 17: re-centre every bootstrap fit on the ORIGINAL training days, as in the main fit, so the
        # spread of the 2024 level does not include movement of the reference (which cancels in predictions)
        c = float(np.mean(m.f(tr[FEATURES])))
        m.a = {y_: v + c for y_, v in m.a.items()}
        m.f_offset += c
        m = refresh_residuals(m, tr)   # residual blocks always come from real, unique consecutive days
        lvl = m.a[2024]
        crn = np.random.default_rng(seed + 4242 + key_seed(key))   # common random numbers across replicates
        sims = base_sims(m, wy, n_res, crn)
        p1 = attain_prob(sims, lvl, pol)
        td = trend_draws(m.a, m.se, 2030, n_trend, crn)          # the same sampler as the main E2
        p2 = attain_prob(sims, 0.0, pol, shifts=td, n_paths=n_trend_res)
        h3 = headroom_all(sims, lvl, pol, grid=coarse, n_paths=30)  # the same rule as the main E3
        for k in metrics:
            vals[f"E1_{k}"].append(p1[k][0]); vals[f"E2_{k}"].append(p2[k].mean())
            for q in (90, 50):
                c = h3[f"E3_{k}_{q}_cens"]
                vals[f"E3_{k}_{q}"].append(-np.inf if c == "left" else (np.inf if c == "right" else h3[f"E3_{k}_{q}"]))
        vals["a_2024"].append(lvl)
    out = {**key, "status": "ok", "reps": len(vals["a_2024"]), "boot_sd_a_2024": float(np.std(vals["a_2024"], ddof=1))}
    for k, v in vals.items():
        v = np.array(v, float)           # censored replicates kept as -inf / +inf, never dropped
        # order-statistic quantiles: an endpoint is +-inf only if it falls in a censored tail
        lo = float(np.quantile(v, 0.05, method="inverted_cdf")) if len(v) else np.nan
        hi = float(np.quantile(v, 0.95, method="inverted_cdf")) if len(v) else np.nan
        out[f"boot_{k}_lo"], out[f"boot_{k}_hi"] = lo, hi
        out[f"boot_{k}_n_censored"] = int(np.isinf(v).sum())
    return out


def crps_samples(x: np.ndarray, y: float) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or y is None or np.isnan(y):
        return np.nan
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / (n * n))


def _rand_pit(samples, obs, rng):
    if obs is None or np.isnan(obs):
        return np.nan
    s = np.asarray(samples)
    lo, hi = np.mean(s < obs), np.mean(s <= obs)
    return float(lo + (hi - lo) * rng.random())


def _rps(samples, obs):
    """Ranked probability score over the full support 0..max(samples, obs)."""
    if obs is None or np.isnan(obs):
        return np.nan
    s = np.asarray(samples)
    kmax = int(max(s.max(initial=0), obs)) + 1
    ks = np.arange(0, kmax + 1)
    F = (s[None, :] <= ks[:, None]).mean(axis=1)
    O = (obs <= ks).astype(float)
    return float(np.sum((F - O) ** 2))


def _score_block(res: list, ob: dict, rng) -> dict:
    means = np.concatenate([o["mean"] for o in res])
    exc_obs = np.concatenate([o["exc_obs"] for o in res])
    return {"crps_mean": crps_samples(means[:: max(1, len(means) // 500)], ob["obs_mean"]),
            "pred_mean_med": float(np.median(means)),
            "pi80_lo": float(np.quantile(means, .1)), "pi80_hi": float(np.quantile(means, .9)),
            "pi90_lo": float(np.quantile(means, .05)), "pi90_hi": float(np.quantile(means, .95)),
            # counts are scored on the observed days only
            "pit_exc_obsdays": _rand_pit(exc_obs, ob["obs_exc_obsdays"], rng),
            "rps_exc_obsdays": _rps(exc_obs, ob["obs_exc_obsdays"])}


def run_hindcast(key: dict, series: pd.DataFrame, met: pd.DataFrame, origins=range(2017, 2024), hp=None,
                 n_res=50, n_trend=100, n_trend_res=5, n_level=30, seed=SEED, kind="lgbm",
                 horizons=range(1, 7), last_target=2024, extra=((2024, 1),)) -> list[dict]:
    """Rolling-origin hindcast (§3.4). For each origin o everything is refit from data <= o: the model,
    residuals, trend, weather pool (o-19..o) and baselines. Scenario 'frozen' integrates the estimation
    uncertainty of the level at o; 'trend' integrates the trend predictive distribution. A 'perfect
    weather' diagnostic (target year's own weather) is reported separately, never as forecast skill."""
    rng = np.random.default_rng(seed + 7 + key_seed(key))
    pol = key["pollutant"]
    d = prepare_series(series, met)
    rows = []
    plan = {}
    for o in origins:
        for h in horizons:
            if o + h <= last_target:
                plan.setdefault(o, []).append(h)
    for o, h in extra:
        plan.setdefault(o, []).append(h)
    for o, hs in sorted(plan.items()):
        tr = training_rows(d, o)
        cov = tr.groupby("year").size() / 365.0
        if (cov >= TRAIN_DAY_SHARE).sum() < 4 or o not in cov.index:
            continue
        model = fit_series(tr, hp=hp, kind=kind, seed=seed)
        wy = weather_year_frames(met, list(range(o - 19, o + 1)))
        if len(wy) < 10:
            continue
        level = model.a[o]
        sims = base_sims(model, wy, n_res, rng)
        obs_tr = {y: observed_outcome(series, y, pol) for y in range(2013, o + 1)}
        lv_draws = level + model.se[o] * rng.standard_normal(n_level)
        for h in sorted(set(hs)):
            y = o + h
            ob = observed_outcome(series, y, pol)
            mask = ob.pop("_mask")
            base = {**key, "origin": o, "h": h, "target": y, "set": "check2025" if y == 2025 else "hindcast",
                    **{k: v for k, v in ob.items()}}
            # scenario A: level frozen at the origin, integrating its estimation uncertainty
            resA = [o_ for lv in lv_draws[:10] for o_ in simulate(sims, lv, pol, n_paths=n_res // 2, mask=mask)]
            pA = attain_prob(sims, level, pol, shifts=lv_draws - level, n_paths=10)
            rowA = {**base, "scenario": "frozen", **{f"p_{k}": float(v.mean()) for k, v in pA.items()},
                    **_score_block(resA, ob, rng)}
            # scenario B: trend continues
            td = trend_draws(model.a, model.se, y, n_trend, rng, method="wls")
            rowB = {**base, "scenario": "trend"}
            if not np.isnan(td).all():
                pB = attain_prob(sims, 0.0, pol, shifts=td, n_paths=n_trend_res)
                resB = [o_ for t in td[:40] for o_ in simulate(sims, t, pol, n_paths=n_trend_res, mask=mask)]
                rowB.update({f"p_{k}": float(v.mean()) for k, v in pB.items()})
                rowB.update(_score_block(resB, ob, rng))
            # perfect-weather diagnostic (target-year weather; only where ISD covers the target year)
            wt = weather_year_frames(met, [y])
            if wt:
                st = base_sims(model, wt, 200, rng)
                pP = attain_prob(st, level, pol, shifts=lv_draws - level, n_paths=40)
                rowA.update({f"pperfect_{k}": float(v.mean()) for k, v in pP.items()})
                if not np.isnan(td).all():
                    pPB = attain_prob(st, 0.0, pol, shifts=td, n_paths=20)
                    rowB.update({f"pperfect_{k}": float(v.mean()) for k, v in pPB.items()})
            for row in (rowA, rowB):
                row.update(baselines(obs_tr, o, y, pol))
            rows += [rowA, rowB]
    return rows


def baselines(obs_tr: dict, o, y, pol) -> dict:
    """Climatology, persistence and a naive linear trend, from training years only (§3.4)."""
    from scipy.stats import norm
    out = {}
    for k in ["att_annual", "att_daily", "att_joint"]:
        vals = [v[f"obs_{k}"] for v in obs_tr.values() if not np.isnan(v[f"obs_{k}"])]
        out[f"clim_{k}"] = float(np.mean(vals)) if vals else np.nan
        last = obs_tr.get(o, {}).get(f"obs_{k}", np.nan)
        out[f"pers_{k}"] = float(last) if not np.isnan(last) else out[f"clim_{k}"]
    lim = LIMITS[pol]
    yrs = [yy for yy, v in obs_tr.items() if not np.isnan(v["obs_mean"])]
    if len(yrs) >= 3:
        mv = np.array([obs_tr[yy]["obs_mean"] for yy in yrs])
        b = np.polyfit(yrs, mv, 1)
        sd = float(np.std(mv - np.polyval(b, yrs), ddof=2)) if len(yrs) > 3 else float(np.std(mv))
        out["naive_att_annual"] = float(norm.cdf((lim["annual"] - np.polyval(b, y)) / max(sd, 1e-6)))
    else:
        out["naive_att_annual"] = np.nan
    # counts only from years with >=95% of days observed (no scaling or filling)
    yrs2 = [yy for yy, v in obs_tr.items() if v["coverage"] >= 0.95 and not np.isnan(v["obs_exc_obsdays"])]
    if len(yrs2) >= 3:
        ev = np.array([obs_tr[yy]["obs_exc_obsdays"] for yy in yrs2], dtype=float)
        b = np.polyfit(yrs2, ev, 1)
        sd = float(np.std(ev - np.polyval(b, yrs2), ddof=2)) if len(yrs2) > 3 else float(np.std(ev))
        out["naive_att_daily"] = float(norm.cdf((lim["allow"] + 0.5 - np.polyval(b, y)) / max(sd, 1e-6)))
    else:
        out["naive_att_daily"] = np.nan
    a, dd = out["naive_att_annual"], out["naive_att_daily"]
    out["naive_att_joint"] = a * dd if not (np.isnan(a) or np.isnan(dd)) else np.nan
    return out


# --------------------------------------------------------------------------------------------
# 9. Scores (§3.4)
# --------------------------------------------------------------------------------------------
def brier_decomp(p, o, bins=10):
    """Brier score with the binned Murphy decomposition; 'remainder' = bs - (rel - res + unc) is the
    within-bin variance term that the binned decomposition leaves out (reported, not hidden)."""
    p, o = np.asarray(p, float), np.asarray(o, float)
    ok = ~(np.isnan(p) | np.isnan(o))
    p, o = p[ok], o[ok]
    if len(p) == 0:
        return dict(bs=np.nan, rel=np.nan, res=np.nan, unc=np.nan, remainder=np.nan, n=0)
    bs = np.mean((p - o) ** 2)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    ob = o.mean()
    rel = res = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            rel += m.sum() * (p[m].mean() - o[m].mean()) ** 2
            res += m.sum() * (o[m].mean() - ob) ** 2
    n = len(p)
    rel, res, unc = rel / n, res / n, ob * (1 - ob)
    return dict(bs=float(bs), rel=float(rel), res=float(res), unc=float(unc),
                remainder=float(bs - (rel - res + unc)), n=int(n))


def bss(p, ref, o):
    p, ref, o = map(lambda x: np.asarray(x, float), (p, ref, o))
    ok = ~(np.isnan(p) | np.isnan(ref) | np.isnan(o))
    if ok.sum() == 0:
        return np.nan
    b, br = np.mean((p[ok] - o[ok]) ** 2), np.mean((ref[ok] - o[ok]) ** 2)
    return float(1 - b / br) if br > 0 else np.nan


def cluster_boot(df: pd.DataFrame, stat, B=500, seed=SEED, c1="target", c2="cc"):
    """Two-way cluster bootstrap (target years × countries), §3.4. Returns (estimate, lo90, hi90)."""
    rng = np.random.default_rng(seed)
    if df.empty:
        return np.nan, np.nan, np.nan
    est = stat(df)
    if c1 == c2:                                   # one-way cluster bootstrap
        cells1 = {k: g for k, g in df.groupby(c1)}
        g1 = list(cells1)
        vals = []
        for _ in range(B):
            parts = [cells1[a] for a in rng.choice(g1, len(g1), replace=True)]
            vals.append(stat(pd.concat(parts)))
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) < 20:
            return est, np.nan, np.nan
        return est, float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))
    g1, g2 = df[c1].unique(), df[c2].unique()
    cells = {k: g for k, g in df.groupby([c1, c2])}
    vals = []
    for _ in range(B):
        s1 = rng.choice(g1, len(g1), replace=True)
        s2 = rng.choice(g2, len(g2), replace=True)
        parts = [cells[(a, b)] for a in s1 for b in s2 if (a, b) in cells]
        if parts:
            vals.append(stat(pd.concat(parts)))
    vals = np.array([v for v in vals if not np.isnan(v)])
    if len(vals) < 20:
        return est, np.nan, np.nan
    return est, float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def run_mc_error(key: dict, series: pd.DataFrame, met: pd.DataFrame, hp=None, seed=SEED, reps=3) -> dict:
    """Monte Carlo error of every reported estimator at one FIXED fit:
    the integration (residual draws, trend draws, headroom grid) is repeated `reps` times with independent
    seeds and the standard deviation across repeats is reported next to the estimator."""
    pol = key["pollutant"]
    d = prepare_series(series, met)
    tr = training_rows(d, 2024)
    model = fit_series(tr, hp=hp, seed=seed)
    wy = weather_year_frames(met, MAIN_POOL)
    if len(wy) < 10:
        return {**key, "status": "fewer than 10 weather-pool years"}
    level, se24 = model.a[2024], model.se[2024]
    vals = {}
    for r in range(reps):
        rng = np.random.default_rng(seed + 1000 * (r + 1) + key_seed(key))
        sims = base_sims(model, wy, 100, rng)
        rec = {}
        for k, v in attain_prob(sims, level, pol).items():
            rec[f"E1_{k}"] = v[0]
        z = level + se24 * rng.standard_normal(60)
        for k, v in attain_prob(sims, level, pol, shifts=z - level, n_paths=50).items():
            rec[f"E1_{k}_lo"], rec[f"E1_{k}_hi"] = np.quantile(v, 0.05), np.quantile(v, 0.95)
        e2 = e2_block(model, sims, pol, rng, 200, 10)
        for k in ["att_annual", "att_daily", "att_joint"]:
            for c in (f"E2_{k}", f"E2_{k}_lo", f"E2_{k}_hi", f"E2ts_{k}", f"E2c20_{k}"):
                if c in e2:
                    rec[c] = e2[c]
        h = headroom_all(sims, level, pol)
        for k in ["att_annual", "att_daily", "att_joint"]:
            for q in (90, 50):
                c = f"E3_{k}_{q}"
                rec[c] = h[c] if h[f"{c}_cens"] == "" else np.nan
        for c, v in rec.items():
            vals.setdefault(c, []).append(float(v))
    out = {**key, "status": "ok"}
    for c, v in vals.items():
        v = np.array(v, float)
        out[f"mcsd_{c}"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() >= 2 else np.nan
    return out
