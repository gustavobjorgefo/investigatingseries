"""
Probe how far back B3 keeps the daily TS (reference rates) files available.

For each year, tries business days in early July until one file downloads,
then reports how many PRE (DI x Pre) vertices it has and the longest maturity.

Run from the repo root:
    python posts/004-br-yield-curve-pca/probe_ts_availability.py
"""
import io
import time
import zipfile

import pandas as pd
import requests

URL = "https://www.b3.com.br/pesquisapregao/download?filelist=TS{date:%y%m%d}.ex_"

# Fixed-width layout of the TS file (same as download_curve_data_b3.ipynb)
WIDTHS = [6, 3, 2, 8, 2, 5, 15, 5, 5, 1, 14, 1, 5]
NAMES = ["seq", "seq_compl", "reg_type", "ref_date", "curve_type", "curve_name",
         "description", "cal_days", "bus_days", "sign", "rate", "vertex_type", "vertex_code"]


def fetch_pre_curve(date: pd.Timestamp) -> pd.DataFrame | None:
    """Download the TS file for one date and return the PRE curve, or None if unavailable."""
    try:
        resp = requests.get(URL.format(date=date), timeout=60)
        resp.raise_for_status()
        outer = zipfile.ZipFile(io.BytesIO(resp.content))
        inner = zipfile.ZipFile(io.BytesIO(outer.read(outer.namelist()[0])))
        raw = inner.read(inner.namelist()[0])
    except (requests.RequestException, zipfile.BadZipFile, IndexError):
        return None

    ts = pd.read_fwf(io.BytesIO(raw), widths=WIDTHS, names=NAMES, dtype=str, encoding="latin1")
    pre = ts[ts["curve_name"] == "PRE"].copy()
    if pre.empty:
        return None
    pre["bus_days"] = pre["bus_days"].astype(int)
    pre["rate"] = pd.to_numeric(pre["sign"] + pre["rate"]) / 1e7
    return pre


def main() -> None:
    rows = []
    for year in range(2000, pd.Timestamp.today().year + 1):
        candidates = pd.bdate_range(f"{year}-07-01", f"{year}-07-15")
        result = {"year": year, "date": None, "n_vertices": 0, "max_years": None}
        for date in candidates:
            pre = fetch_pre_curve(date)
            time.sleep(1)  # be polite to B3's server
            if pre is not None:
                result.update(date=date.date(), n_vertices=len(pre),
                              max_years=round(pre["bus_days"].max() / 252, 1))
                break
        rows.append(result)
        status = "OK " if result["date"] else "---"
        print(f"{status} {year}: date={result['date']}  vertices={result['n_vertices']}  "
              f"max maturity={result['max_years']}y")

    print("\nSummary:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()