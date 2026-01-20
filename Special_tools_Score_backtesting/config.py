"""
Backtest Configuration Module

Central configuration for accumulation signal backtesting parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add parent directory to path for imports
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from darkpool_analysis.config import load_config as load_main_config
from darkpool_analysis.db_path import get_db_path

# =============================================================================
# TICKER UNIVERSE DEFINITIONS
# =============================================================================

# Tier 1: ETF Screening Layer (15 ETFs)
TIER1_GLOBAL_MACRO = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO"]
TIER1_SECTOR_CORE = ["XLF", "XLK", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU"]
TIER1_TICKERS = TIER1_GLOBAL_MACRO + TIER1_SECTOR_CORE

# Tier 3: Always-On High Conviction (18 tickers)
MAG8_TICKERS = ["MSFT", "AAPL", "GOOGL", "AMZN", "NVDA", "AVGO", "META", "TSLA"]
SPECULATIVE_TICKERS = ["AMD", "BE", "CRWV", "HIMS", "HOOD", "IREN", "JPM", "NNE", "PLTR", "SOFI"]
TIER3_TICKERS = MAG8_TICKERS + SPECULATIVE_TICKERS

# Combined POC universe
POC_TICKERS = list(set(TIER1_TICKERS + TIER3_TICKERS))

# ETF to Sector mapping (for parent ETF alignment)
ETF_SECTOR_MAP = {
    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Small Cap",
    "DIA": "Dow 30",
    "TLT": "Treasuries",
    "GLD": "Gold",
    "USO": "Oil",
}

# Ticker to parent ETF mapping (simplified - major holdings)
TICKER_TO_ETF = {
    # Technology (XLK)
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    # Financials (XLF)
    "JPM": "XLF",
    # Consumer Discretionary (XLY)
    "AMZN": "XLY", "TSLA": "XLY",
    # Communication Services
    "META": "XLC", "GOOGL": "XLC",
}

# Liquid options tickers (for options context - high volume)
LIQUID_OPTIONS_TICKERS = set(TIER1_TICKERS + MAG8_TICKERS)


# =============================================================================
# SCORE VARIANT DEFINITIONS
# =============================================================================

@dataclass
class ScoreVariantConfig:
    """Configuration for a score variant."""
    name: str
    short_z_source: str
    description: str
    blend_weights: Optional[Dict[str, float]] = None  # For blended variants


SCORE_VARIANTS = {
    "A": ScoreVariantConfig(
        name="A",
        short_z_source="finra_buy_volume_z",
        description="VWBR-Anchored: Pure institutional buy volume",
    ),
    "B": ScoreVariantConfig(
        name="B",
        short_z_source="short_buy_sell_ratio_z",
        description="Short B/S Ratio: Directional bias (current baseline)",
    ),
    "C": ScoreVariantConfig(
        name="C",
        short_z_source="blended",
        description="Combined: 50/50 blend of A and B",
        blend_weights={"finra_buy_volume_z": 0.5, "short_buy_sell_ratio_z": 0.5},
    ),
    "D": ScoreVariantConfig(
        name="D",
        short_z_source="imbalance_signal_z",
        description="Imbalance-Weighted: (B-S)/(B+S) × log1p(volume)",
    ),
}


# =============================================================================
# WEIGHT PRESETS
# =============================================================================

WEIGHT_PRESETS = {
    "short_heavy": {"w_short": 0.70, "w_lit": 0.20, "w_price": 0.10},
    "balanced": {"w_short": 0.55, "w_lit": 0.30, "w_price": 0.15},
    "lit_heavy": {"w_short": 0.40, "w_lit": 0.40, "w_price": 0.20},
}


# =============================================================================
# PARAMETER SPACE FOR GRID SEARCH
# =============================================================================

@dataclass
class ParameterSpace:
    """Defines the parameter space for grid search."""

    # Score variants to test
    score_variants: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])

    # Component weights
    w_short: List[float] = field(default_factory=lambda: [0.40, 0.50, 0.55, 0.60, 0.70])
    w_lit: List[float] = field(default_factory=lambda: [0.20, 0.25, 0.30, 0.35, 0.40])
    w_price: List[float] = field(default_factory=lambda: [0.10, 0.15, 0.20])

    # Z-score rolling windows
    z_windows: List[int] = field(default_factory=lambda: [10, 15, 20, 30, 40, 60, 90])

    # Signal thresholds
    buy_thresholds: List[int] = field(default_factory=lambda: [60, 65, 70, 75, 80])
    sell_thresholds: List[int] = field(default_factory=lambda: [20, 25, 30, 35, 40])

    # Exit strategies
    exit_strategies: List[str] = field(default_factory=lambda: [
        "fixed_10d", "fixed_20d", "first_dist", "2x_consec_dist", "hybrid_30d"
    ])


# Coarse sweep (Phase A) - reduced parameter space
COARSE_SWEEP_SPACE = ParameterSpace(
    score_variants=["A", "B", "C", "D"],
    w_short=[0.40, 0.55, 0.70],  # 3 presets
    w_lit=[0.20, 0.30, 0.40],
    w_price=[0.10, 0.15, 0.20],
    z_windows=[15, 20, 30],
    buy_thresholds=[70],
    sell_thresholds=[30],
    exit_strategies=["first_dist", "2x_consec_dist"],
)


# =============================================================================
# REGIME DEFINITIONS
# =============================================================================

@dataclass
class RegimeConfig:
    """Market regime classification thresholds."""

    # RISK_ON: SPY > 50d SMA AND VIX < 20
    risk_on_vix_max: float = 20.0

    # RISK_OFF: SPY < 50d SMA AND VIX > 25
    risk_off_vix_min: float = 25.0

    # SMA periods
    spy_sma_period: int = 50


# =============================================================================
# TREND CLASSIFICATION
# =============================================================================

@dataclass
class TrendConfig:
    """Trend classification parameters."""

    # SMA period for trend
    sma_period: int = 20

    # Slope threshold for determining trend direction
    # slope > threshold = UP, slope < -threshold = DOWN, else SIDEWAYS
    slope_threshold: float = 0.001

    # RSI period
    rsi_period: int = 14


# =============================================================================
# EXIT STRATEGY CONFIGS
# =============================================================================

@dataclass
class ExitConfig:
    """Exit strategy parameters."""

    # Fixed holding periods (trading days)
    fixed_periods: Dict[str, int] = field(default_factory=lambda: {
        "fixed_5d": 5,
        "fixed_10d": 10,
        "fixed_20d": 20,
        "fixed_30d": 30,
    })

    # Score decay: exit if score < threshold for N consecutive days
    score_decay_threshold: float = 50.0
    score_decay_days: int = 3

    # Hybrid: exit on distribution OR max days, whichever first
    hybrid_max_days: int = 30


# =============================================================================
# FAILURE MODE DEFINITIONS
# =============================================================================

FAILURE_MODES = [
    "IMMEDIATE_REVERSAL",  # Loss within 2 days
    "SLOW_BLEED",          # Gradual loss over hold period
    "GAP_DOWN",            # Overnight gap killed it
    "SECTOR_DRAG",         # ETF down, constituent followed
    "FALSE_SIGNAL",        # Score never confirmed (quick reversal to neutral)
]


# =============================================================================
# SIGNAL QUALITY THRESHOLDS
# =============================================================================

@dataclass
class SignalQualityConfig:
    """Thresholds for signal quality classification."""

    # HIGH quality thresholds
    high_score_reversal: float = 80.0  # Score threshold for reversal setup
    high_score_aligned: float = 75.0   # Score threshold when ETF aligned
    high_put_call_max: float = 0.7     # Put/call ratio threshold for reversal

    # MEDIUM quality thresholds
    medium_score_min: float = 70.0
    medium_lit_z_min: float = 1.0


# =============================================================================
# METRICS TARGETS
# =============================================================================

@dataclass
class MetricsTargets:
    """Target values for evaluation metrics."""

    hit_rate_min: float = 0.55      # Minimum acceptable hit rate
    avg_return_min: float = 0.01    # Minimum average return per trade (1%)
    win_loss_ratio_min: float = 2.0  # Minimum win/loss ratio
    max_signals_per_month: float = 10.0  # Cap on signal frequency


# =============================================================================
# MAIN BACKTEST CONFIG
# =============================================================================

@dataclass
class BacktestConfig:
    """Main configuration for backtesting."""

    # Database path (use main darkpool database via centralized helper)
    db_path: Path = field(default_factory=get_db_path)

    # Output directory
    output_dir: Path = field(default_factory=lambda: PACKAGE_DIR / "output")

    # Ticker universe
    ticker_universe: str = "TIER1+3"  # Options: "TIER1", "TIER1+3", "ALL"

    # Date range
    date_start: str = "2025-01-01"
    date_end: str = "2025-12-31"

    # Entry timing
    entry_timing: str = "next_day_open"  # Realistic: enter at next day's open

    # Overlap handling
    allow_overlapping_positions: bool = False  # One position per ticker at a time

    # Sub-configs
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    signal_quality: SignalQualityConfig = field(default_factory=SignalQualityConfig)
    metrics_targets: MetricsTargets = field(default_factory=MetricsTargets)

    # Composite score weights (from main config)
    composite_w_short: float = 0.55
    composite_w_lit: float = 0.30
    composite_w_price: float = 0.15
    intensity_scale_min: float = 0.7
    intensity_scale_max: float = 1.3

    def get_tickers(self) -> List[str]:
        """Get ticker list based on universe setting."""
        if self.ticker_universe == "TIER1":
            return TIER1_TICKERS
        elif self.ticker_universe == "TIER1+3":
            return POC_TICKERS
        else:
            # ALL - would need to load from ticker_dictionary
            return POC_TICKERS


def load_config() -> BacktestConfig:
    """Load backtest configuration, inheriting from main config where applicable."""
    main_config = load_main_config()

    config = BacktestConfig(
        db_path=main_config.db_path,
        composite_w_short=main_config.composite_w_short,
        composite_w_lit=main_config.composite_w_lit,
        composite_w_price=main_config.composite_w_price,
        intensity_scale_min=main_config.intensity_scale_min,
        intensity_scale_max=main_config.intensity_scale_max,
    )

    return config
