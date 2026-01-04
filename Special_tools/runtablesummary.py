import sys
from pathlib import Path

# Add parent directory to path so we can import darkpool_analysis
_parent_dir = Path(__file__).resolve().parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from darkpool_analysis.table_renderer_summary import render_sector_summary
from darkpool_analysis.config import load_config

config = load_config()
html_path, png_path = render_sector_summary(
    db_path=config.db_path,
    output_dir=_parent_dir / "darkpool_analysis" / "output" / "tables_summary",
    dates=config.target_dates,  # Uses dates from config (default: last 30 trading days)
    max_dates=10,  # Show up to 10 most recent dates per ticker
)

# # From command line
# python -m darkpool_analysis.table_renderer_summary --dates 2025-12-30,2025-12-27,2025-12-26

# # Or with custom tickers
# python -m darkpool_analysis.table_renderer_summary --dates 2025-12-30 --tickers XLE,XLF,XLK
