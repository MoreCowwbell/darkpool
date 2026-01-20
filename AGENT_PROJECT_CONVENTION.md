# Agent Project Conventions

## Primary rule
Do **not** create, write, or update large data artifacts (DuckDB databases, WAL files, parquet, large CSVs) inside this repo if the repo is located under Dropbox syncing.

This repo uses a portable DB location convention via `DATA_ROOT`.

---

## Database location convention

### Environment variable
- `DATA_ROOT` is a directory, not a file.
- If `DATA_ROOT` is set, DBs are stored outside the repo:

**Windows example**
- `DATA_ROOT=D:\vscode\data`
- DB path becomes:
  - `D:\vscode\data\<project>\<project>.duckdb`

**Linux/AWS example**
- `DATA_ROOT=/data` (recommended for mounted persistent volume)
- DB path becomes:
  - `/data/<project>/<project>.duckdb`

### Fallback (portable default)
If `DATA_ROOT` is **not** set, the DB path falls back to a repo-relative location:
- `./data/<project>.duckdb`

This fallback is acceptable for ephemeral/dev environments. For persistence on AWS, prefer `DATA_ROOT=/data` with a mounted volume.

---

## Implementation requirements

### Single source of truth
All code must obtain DB paths via a single helper function/module (e.g., `get_db_path()`).
Do not hardcode DB paths in multiple files.

### Helper responsibilities
The helper must:
1) Find repo root (prefer `.git` detection; fallback to current working directory).
2) Derive `<project>` from the repo root folder name.
3) If `DATA_ROOT` is set: use `<DATA_ROOT>/<project>/`.
4) Else: use `<repo_root>/data/`.
5) Ensure the directory exists (create it if missing).
6) Return the final DB file path:
   - `<base>/<project>.duckdb` (unless an explicit DB filename override is provided)

---

## .env guidance
The `.env` file is expected to include:

```dotenv
DATA_ROOT=D:\vscode\data
# AWS/Linux example:
# DATA_ROOT=/data
```

---

## Implementation Status

The DATA_ROOT convention is now implemented via `darkpool_analysis/db_path.py`.

### Usage

```python
from db_path import get_db_path

db_path = get_db_path()  # Returns: D:\vscode\data\darkpool\darkpool.duckdb (Windows)
                         #      or: /data/darkpool/darkpool.duckdb (Linux)
                         #      or: ./data/darkpool.duckdb (fallback)
```

### Files Updated (2026-01-20)

| File | Change |
|------|--------|
| `darkpool_analysis/db_path.py` | NEW - Central helper module |
| `darkpool_analysis/config.py` | Uses `get_db_path()` |
| `darkpool_analysis/backfill_itm_otm.py` | Uses `get_db_path()` |
| `Special_tools/reset_daily_metrics.py` | Uses `get_db_path()` |
| `Special_tools/close_db.py` | Uses `get_db_path()` |
| `Special_tools_Score_backtesting/config.py` | Uses `get_db_path()` |
| `Special_tools/circos_v2.ipynb` | Uses `config.db_path` (replaced `find_db_candidates()`) |
| `.env` | Added `DATA_ROOT=D:\vscode\data` |

### Resolved Paths

- **Windows** (DATA_ROOT=D:\vscode\data): `D:\vscode\data\darkpool\darkpool.duckdb`
- **AWS/Linux** (DATA_ROOT=/data): `/data/darkpool/darkpool.duckdb`
- **Fallback** (DATA_ROOT not set): `{repo_root}/data/darkpool.duckdb`
