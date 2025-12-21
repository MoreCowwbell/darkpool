#!/usr/bin/env python
"""
Dark Pool ETF Analysis - Entry Point

FINRA does not publish trade direction for off-exchange volume.
Buy/Sell values are inferred estimates derived from lit-market equity trades
and applied proportionally to FINRA OTC volume.

Usage:
    python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add darkpool_analysis to path for module resolution
_package_dir = Path(__file__).resolve().parent / "darkpool_analysis"
if str(_package_dir) not in sys.path:
    sys.path.insert(0, str(_package_dir))

from orchestrator import main

if __name__ == "__main__":
    sys.exit(main() or 0)
