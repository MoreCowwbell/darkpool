"""
Accumulation Signal Backtesting Framework

This package provides tools for systematically testing and optimizing
accumulation score configurations for swing trading signals.

Modules:
    config: Backtest-specific configuration parameters
    schema: Database table definitions (bt_ prefixed)
    trend_classifier: UP/DOWN/SIDEWAYS trend classification
    regime_classifier: RISK_ON/RISK_OFF market regime detection
    options_context: Options premium metrics extraction
    signal_generator: Generate signals with configurable parameters
    forward_returns: Compute forward returns for signals
    exit_detector: Distribution-based exit detection
    metrics: Performance metrics (hit rate, Sharpe, etc.)
    parameter_sweep: Grid search over configurations
    walk_forward: Walk-forward validation harness
"""

from pathlib import Path

# Package root directory
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
OUTPUT_DIR = PACKAGE_DIR / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "charts").mkdir(exist_ok=True)

__version__ = "0.1.0"
__all__ = [
    "config",
    "schema",
    "trend_classifier",
    "regime_classifier",
    "options_context",
    "signal_generator",
    "forward_returns",
    "exit_detector",
    "metrics",
    "parameter_sweep",
    "walk_forward",
]
