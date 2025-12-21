# Agent Changelog

This file tracks completed tasks and in-progress work across sessions.  
Each entry should be short, bullet-based, and written immediately after a task is completed.

Format:
- Use checkboxes for completed tasks.
- Use a "⧗" prefix for tasks that are in progress (not yet finished).
- Keep each bullet to one line.
- Keep total log size manageable (older items can be summarized or archived periodically).

---

## Example of logs:
## 2025-01-XX — Session Summary (Agent Name)
- [x] Short description of completed task.
- [x] Another completed task.
- ⧗ In-progress task that should be resumed next session.

## 2025-01-XX — Session Summary (Agent Name)
- [x] Completed adjustment to function X.
- ⧗ Started refactor of Y but not finished.

## 2025-01-14 — Session Summary (Claude Code)
- [x] Added timeout support to Polygon request helper.
- [x] Fixed missing import in historical.py.
- ⧗ Started rewriting intraday_stream.py state machine (continue next session).

---
## Start of Changelog below:
## 2025-12-21 Session Summary (Codex)
- [x] Updated `darkpool/AGENT_CONTEXT.md` to reflect the Option B dark pool ETF analysis plan and execution flow.
- [x] Updated `darkpool/AGENT_INSTRUCTIONS.md` with inference rules, table requirements, outputs, and error handling.
- [x] Logged this session entry in `darkpool/AGENT_CHANGELOG.md`.
## 2025-12-21 Session Summary (Codex)
- [x] Created `darkpool_analysis/` project structure with config and DuckDB scaffolding.
- [x] Implemented FINRA/Polygon ingestion, lit inference, analytics, and plotting modules.
- [x] Added `darkpool_analysis/README.md`, `darkpool_analysis/requirements.txt`, and `darkpool_analysis/.env` template.
- [x] Wired `darkpool_analysis/orchestrator.py` end-to-end and exported tables/plots.

## 2025-12-21 Session Summary (Claude Code)
- [x] Added root `main.py` entry point for running from project root.
- [x] Created `darkpool_analysis/__init__.py` for proper package structure.
- [x] Fixed imports to use relative imports with `try/except ImportError` fallback.
- [x] Fixed DuckDB connection leak by wrapping in `try/finally` with `conn.close()`.
- [x] Added parallel Polygon API fetching using `ThreadPoolExecutor` (configurable via `POLYGON_MAX_WORKERS`).
- [x] Fixed `iterrows()` performance in plotter using vectorized mask operations.
- [x] Fixed `buy_ratio` formula inconsistency: now `bought/(bought+sold)` matching `lit_buy_ratio`.
- [x] Updated `AGENT_CONTEXT.md`, `AGENT_INSTRUCTIONS.md`, and `AGENT_CHANGELOG.md`.
