"""Utility to check/release DuckDB connections."""
import os
import subprocess
import sys
from pathlib import Path

# Add darkpool_analysis to path for db_path import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "darkpool_analysis"))
from db_path import get_db_path

DB_PATH = get_db_path()
WAL_PATH = DB_PATH.with_suffix(".duckdb.wal")

def delete_wal():
    """Delete the WAL file to release locks (safe when DB is idle)."""
    if WAL_PATH.exists():
        try:
            os.remove(WAL_PATH)
            print(f"Deleted WAL file: {WAL_PATH}")
            return True
        except Exception as e:
            print(f"Could not delete WAL: {e}")
            return False
    return True

def main():
    print(f"Database: {DB_PATH}")
    print(f"WAL file: {WAL_PATH} (exists: {WAL_PATH.exists()})")
    print()

    # Try to connect
    try:
        import duckdb
        conn = duckdb.connect(str(DB_PATH))
        print("OK: Database is accessible - no lock!")
        conn.close()
        return
    except Exception as e:
        print(f"LOCKED: {e}")
        print()

    # Offer to force release
    print("Options to release the lock:")
    print("  1. In VS Code: Click DBCode icon (left sidebar) -> Right-click connection -> Disconnect")
    print("  2. Command Palette: Ctrl+Shift+P -> 'DBCode: Disconnect'")
    print("  3. Delete the .wal file (forces release, safe if no active writes)")
    print()

    if WAL_PATH.exists():
        choice = input("Delete WAL file to force release? [y/N]: ").strip().lower()
        if choice == 'y':
            if delete_wal():
                # Try again
                try:
                    conn = duckdb.connect(str(DB_PATH))
                    print("SUCCESS: Database is now accessible!")
                    conn.close()
                except Exception as e:
                    print(f"Still locked: {e}")
                    print("You may need to restart VS Code.")

if __name__ == "__main__":
    main()
