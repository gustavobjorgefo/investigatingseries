"""
Build the Brazilian PRE (DI x Pre) constant-maturity curve from B3 TS files.

Three stages, each one cached so the script can be stopped and resumed:

1. download  -> data/raw/b3_ts/TSyymmdd.ex_      (raw B3 files, ALL curves, gitignored)
2. vertices  -> data/raw/br_pre_vertices.parquet  (every PRE vertex, every day, gitignored)
3. curve     -> data/processed/br_pre_curve_cm.parquet  (constant maturities, committed)

Run from the repo root:
    python posts/005-br-yield-curve-pca/build_dataset.py                 # all stages, 2000 -> today
    python posts/005-br-yield-curve-pca/build_dataset.py --start 2024-01-01 --stage download
"""
import argparse
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://www.b3.com.br/pesquisapregao/download?filelist=TS{date:%y%m%d}.ex_"

RAW_DIR = Path("data/raw/b3_ts")
MISSING_LOG = RAW_DIR / "_missing.csv"            # dates with no file (holidays etc.)
VERTICES_FILE = Path("data/raw/br_pre_vertices.parquet")
CURVE_FILE = Path("data/processed/br_pre_curve_cm.parquet")

# Fixed-width layout of the TS file
WIDTHS = [6, 3, 2, 8, 2, 5, 15, 5, 5, 1, 14, 1, 5]
NAMES = ["seq", "seq_compl", "reg_type", "ref_date", "curve_type", "curve_name",
         "description", "cal_days", "bus_days", "sign", "rate", "vertex_type", "vertex_code"]

# Constant maturities in business days (252 per year)
TENORS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "3Y": 756,
          "4Y": 1008, "5Y": 1260, "7Y": 1764, "10Y": 2520}


# ---------------------------------------------------------------- 1. download
def _raw_path(date: pd.Timestamp) -> Path:
    return RAW_DIR / f"TS{date:%y%m%d}.ex_"


def _load_missing() -> set[str]:
    if MISSING_LOG.exists():
        return set(pd.read_csv(MISSING_LOG)["date"])
    return set()


def _save_missing(missing: set[str]) -> None:
    pd.DataFrame({"date": sorted(missing)}).to_csv(MISSING_LOG, index=False)


def _has_content(data: bytes) -> bool:
    """
    True if the B3 response holds a real TS file.
    On holidays B3 still returns a valid zip, but the inner zip is EMPTY.
    """
    try:
        outer = zipfile.ZipFile(io.BytesIO(data))
        inner = zipfile.ZipFile(io.BytesIO(outer.read(outer.namelist()[0])))
        return len(inner.namelist()) > 0
    except (zipfile.BadZipFile, IndexError, KeyError):
        return False


# A date this recent with an empty file may simply not be published yet: retry it next run
RECENT_DAYS = 7


def _is_recent(date: pd.Timestamp) -> bool:
    return date >= pd.Timestamp.today().normalize() - pd.Timedelta(days=RECENT_DAYS)


def clean_cache() -> None:
    """Remove cached files with no content (holidays) and record them in the missing log."""
    missing = _load_missing()
    removed = 0
    for path in RAW_DIR.glob("TS*.ex_"):
        if not _has_content(path.read_bytes()):
            date = pd.to_datetime(path.stem[2:], format="%y%m%d")
            if not _is_recent(date):
                missing.add(str(date.date()))
            path.unlink()
            removed += 1
    if removed:
        _save_missing(missing)
        print(f"Cache cleanup: removed {removed} empty files (holidays / not yet published)")


def download(start: str, end: str, pause: float = 1.0) -> None:
    """Download every business day's TS file not yet in the cache."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    clean_cache()
    missing = _load_missing()
    dates = [d for d in pd.bdate_range(start, end)
             if not _raw_path(d).exists() and str(d.date()) not in missing]
    print(f"Download: {len(dates)} dates to fetch (cache and known holidays skipped)")

    session = requests.Session()
    n_new_missing, n_retry = 0, 0
    for i, date in enumerate(dates, 1):
        try:
            resp = session.get(URL.format(date=date), timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            # Network problem: do NOT mark as missing, it will be retried next run
            print(f"  {date.date()} network error ({e.__class__.__name__}), will retry next run")
            n_retry += 1
            time.sleep(pause * 5)
            continue

        if _has_content(resp.content):
            _raw_path(date).write_bytes(resp.content)
        elif _is_recent(date):
            n_retry += 1                     # probably not published yet: try again next run
        else:
            missing.add(str(date.date()))    # holiday: no file for this day
            n_new_missing += 1

        if i % 100 == 0:
            print(f"  {i}/{len(dates)} done (last: {date.date()})")
            _save_missing(missing)           # checkpoint, in case the run is interrupted
        time.sleep(pause)

    _save_missing(missing)
    print(f"Download finished: {n_new_missing} new dates without file, "
          f"{n_retry} to retry next run")


# ---------------------------------------------------------------- 2. vertices
def parse_pre(path: Path) -> pd.DataFrame:
    """Read one raw TS file and return its PRE curve vertices."""
    outer = zipfile.ZipFile(path)
    inner = zipfile.ZipFile(io.BytesIO(outer.read(outer.namelist()[0])))
    raw = inner.read(inner.namelist()[0])

    ts = pd.read_fwf(io.BytesIO(raw), widths=WIDTHS, names=NAMES, dtype=str, encoding="latin1")
    pre = ts[ts["curve_name"] == "PRE"].copy()
    pre["ref_date"] = pd.to_datetime(pre["ref_date"], format="%Y%m%d")
    pre[["cal_days", "bus_days"]] = pre[["cal_days", "bus_days"]].astype(int)
    pre["rate"] = pd.to_numeric(pre["sign"] + pre["rate"]) / 1e7   # % p.a., base 252
    return pre[["ref_date", "cal_days", "bus_days", "rate", "vertex_type"]]


def build_vertices() -> pd.DataFrame:
    """Parse every cached raw file into one long table of PRE vertices."""
    files = sorted(RAW_DIR.glob("TS*.ex_"))
    print(f"Vertices: parsing {len(files)} raw files")
    frames = []
    for path in files:
        try:
            frames.append(parse_pre(path))
        except Exception as e:  # a corrupt file should not stop the whole build
            print(f"  skipping {path.name}: {e}")
    vertices = pd.concat(frames, ignore_index=True)
    VERTICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    vertices.to_parquet(VERTICES_FILE, index=False)
    print(f"Vertices: {len(vertices):,} rows, {vertices['ref_date'].nunique():,} days -> {VERTICES_FILE}")
    return vertices


# ---------------------------------------------------------------- 3. curve
def flat_forward(bus_days, rates, target_days) -> np.ndarray:
    """
    Flat-forward interpolation (linear in log discount factor), base 252.
    Rates in % p.a. Targets outside the available range return NaN (no extrapolation).
    """
    bus_days = np.asarray(bus_days, float)
    rates = np.asarray(rates, float) / 100
    target_days = np.asarray(target_days, float)

    order = np.argsort(bus_days)
    bus_days, rates = bus_days[order], rates[order]

    log_df = -(bus_days / 252) * np.log1p(rates)
    log_df_target = np.interp(target_days, bus_days, log_df)
    out = (np.exp(-log_df_target * 252 / target_days) - 1) * 100

    outside = (target_days < bus_days.min()) | (target_days > bus_days.max())
    out[outside] = np.nan
    return out


def build_curve(vertices: pd.DataFrame) -> pd.DataFrame:
    """Constant-maturity curve: one row per date, one column per tenor (% p.a.)."""
    targets = list(TENORS.values())
    curve = (vertices.groupby("ref_date")[["bus_days", "rate"]]
             .apply(lambda g: pd.Series(flat_forward(g["bus_days"], g["rate"], targets),
                                        index=list(TENORS)))
             .sort_index())
    curve.index.name = "date"
    CURVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(CURVE_FILE)
    print(f"Curve: {curve.shape[0]:,} days x {curve.shape[1]} tenors -> {CURVE_FILE}")
    print("First date with data, per tenor:")
    print(curve.apply(lambda s: s.first_valid_index()).to_string())
    return curve


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=str(pd.Timestamp.today().date()))
    parser.add_argument("--stage", choices=["all", "download", "vertices", "curve"], default="all")
    args = parser.parse_args()

    if args.stage in ("all", "download"):
        download(args.start, args.end)
    if args.stage in ("all", "vertices"):
        build_vertices()
    if args.stage in ("all", "curve"):
        build_curve(pd.read_parquet(VERTICES_FILE))


if __name__ == "__main__":
    main()