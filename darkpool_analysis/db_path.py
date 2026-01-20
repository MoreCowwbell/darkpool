"""
Centralized database path resolution.

Convention:
- If DATA_ROOT is set: {DATA_ROOT}/{project}/{project}.duckdb
- Else: ./data/{project}.duckdb (repo-relative fallback)

This module provides a single source of truth for database paths,
ensuring databases are stored outside Dropbox sync directories
when DATA_ROOT is configured.

Usage:
    from db_path import get_db_path

    db_path = get_db_path()  # Returns: D:\vscode\data\darkpool\darkpool.duckdb (Windows)
                             #      or: /data/darkpool/darkpool.duckdb (Linux)
                             #      or: ./data/darkpool.duckdb (fallback)
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    """
    Find repo root by looking for .git directory.

    Walks up the directory tree from this file's location until it finds
    a directory containing .git, indicating the repository root.

    Falls back to assuming darkpool_analysis/ is a direct child of the repo
    if no .git directory is found (e.g., in deployed environments).
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    # Fallback: assume darkpool_analysis/ is direct child of repo
    return Path(__file__).resolve().parent.parent


def get_db_path(db_name: str | None = None) -> Path:
    """
    Get the canonical database path.

    Resolves the database path according to the DATA_ROOT convention:
    - If DATA_ROOT env var is set: {DATA_ROOT}/{project}/{project}.duckdb
    - Otherwise: {repo_root}/data/{project}.duckdb

    The parent directory is created if it doesn't exist.

    Args:
        db_name: Override database filename. Defaults to {project}.duckdb
                 where project is the repo root folder name.

    Returns:
        Path to the database file.

    Examples:
        >>> get_db_path()  # With DATA_ROOT=D:\\vscode\\data
        WindowsPath('D:/vscode/data/darkpool/darkpool.duckdb')

        >>> get_db_path()  # Without DATA_ROOT
        WindowsPath('C:/Users/.../darkpool/data/darkpool.duckdb')

        >>> get_db_path("scanner.duckdb")  # Custom filename
        WindowsPath('D:/vscode/data/darkpool/scanner.duckdb')
    """
    repo_root = _find_repo_root()
    project_name = repo_root.name  # "darkpool"

    data_root = os.environ.get("DATA_ROOT")

    if data_root:
        # External storage: {DATA_ROOT}/{project}/
        base_dir = Path(data_root) / project_name
    else:
        # Fallback: repo-relative ./data/
        base_dir = repo_root / "data"

    # Ensure directory exists
    base_dir.mkdir(parents=True, exist_ok=True)

    filename = db_name or f"{project_name}.duckdb"
    return base_dir / filename


def get_scanner_db_path() -> Path:
    """
    Get the path for the scanner-specific database.

    This is a convenience function for the FINRA scanner which uses
    a separate database file.

    Returns:
        Path to the scanner database file (darkpool_scanner.duckdb).
    """
    return get_db_path("darkpool_scanner.duckdb")
