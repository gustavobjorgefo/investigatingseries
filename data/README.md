# Data

| Folder | In git? | What goes here |
|---|---|---|
| `raw/` | **No** (gitignored) | Downloads exactly as received (B3 files, yfinance pulls). A local cache: code re-downloads anything missing. |
| `processed/` | **Yes** | Small, clean datasets used by the figures of a post (prefer `.parquet`, keep each file < ~5 MB; larger files go to a GitHub Release). |

## Rules
1. Download code checks `raw/` first and only fetches what is missing.
2. Every file in `processed/` is produced by a script in the post folder (e.g. `posts/004-.../build_dataset.py`).
3. Data exported from paid platforms is never committed.
4. No pickle files — use parquet or csv.

## Datasets

| File | Produced by | Source | Downloaded on | Notes |
|---|---|---|---|---|
| _(none yet)_ | | | | |
