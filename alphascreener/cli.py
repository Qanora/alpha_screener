"""CLI entry point."""

import logging

import click

from alphascreener import __version__
from alphascreener.config import get_settings
from alphascreener.db import get_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@click.group()
@click.version_option(version=__version__)
def main():
    """AlphaScreener - AI-Native quantitative strategy experimentation platform."""
    pass


@main.command(name="init-db")
def init_db_cmd():
    """Initialize the SQLite database and all tables."""
    settings = get_settings()
    with get_db(settings.db_path):
        pass
    click.echo(f"Database initialized at {settings.db_path}")


# ---- data commands -------------------------------------------------


@main.group()
def data():
    """Data pipeline: sync, validate, and inspect market data."""


@data.command()
@click.option("--days", default=1, help="Days of OHLCV to fetch")
@click.option("--top", default=100, help="Number of tickers to cross-validate against Stooq")
def sync(days: int, top: int):
    """Fetch index universe, pre-filter, download OHLCV, cross-validate."""
    from datetime import date, timedelta

    from alphascreener.adapters.yfinance_adapter import YFinanceAdapter
    from alphascreener.adapters.stooq_adapter import StooqAdapter, CrossValidator
    from alphascreener.core.universe import fetch_index_universe, pre_filter
    from alphascreener.core.storage import DataStore

    settings = get_settings()
    store = DataStore()
    today = date.today()
    start = today - timedelta(days=days)

    click.echo("Fetching index universe (SP500 + Russell 1000)...")
    tickers = fetch_index_universe()
    click.echo(f"  Got {len(tickers)} unique tickers")

    click.echo("Pre-filtering...")
    universe = pre_filter(tickers)
    click.echo(f"  Passed: {len(universe)} tickers")

    if universe.is_empty():
        click.echo("No tickers passed pre-filter. Aborting.")
        return

    store.write_universe_meta(universe)

    yf_adapter = YFinanceAdapter(rps=settings.yfinance_rps)
    ticker_list = universe["ticker"].to_list()
    click.echo(f"Downloading OHLCV for {len(ticker_list)} tickers...")
    yf_df = yf_adapter.fetch_ohlcv_batch(ticker_list, start, today)
    store.write_ohlcv(yf_df, today)
    click.echo(f"  Saved {len(yf_df)} rows to OHLCV store")

    if not yf_df.is_empty() and top > 0:
        top_tickers = ticker_list[:top]
        click.echo(f"Cross-validating Top {len(top_tickers)} against Stooq...")
        stooq = StooqAdapter()
        validator = CrossValidator(stooq)
        diff_df = validator.validate(yf_df, top_tickers, today)
        if not diff_df.is_empty():
            click.echo(f"  Found {len(diff_df)} discrepancies (> 0.5%)")
        else:
            click.echo("  All clean, no discrepancies")

    click.echo("Data sync complete.")


@data.command()
def universe():
    """Show current universe statistics."""
    from alphascreener.core.storage import DataStore

    store = DataStore()
    meta = store.read_universe_meta()
    if meta is None:
        click.echo("No universe data. Run 'alphascreener data sync' first.")
        return

    click.echo(f"Universe: {len(meta)} tickers")
    if "sector" in meta.columns:
        sectors = meta.group_by("sector").len().sort("len", descending=True)
        click.echo("\nBy sector:")
        for row in sectors.iter_rows(named=True):
            click.echo(f"  {row['sector'] or 'Unknown'}: {row['len']}")
    if "market_cap" in meta.columns:
        mc = meta["market_cap"]
        click.echo(
            f"\nMarket cap (M): min={mc.min() / 1e6:.0f} median={mc.median() / 1e6:.0f} max={mc.max() / 1e6:.0f}"
        )


# ---- analysis commands ---------------------------------------------


@main.command()
@click.option("--market", default="US", help="Target market (US)")
@click.option("--top", default=20, help="Number of top picks to output")
def screen(market: str, top: int):
    """Run a full market scan."""
    click.echo(f"Screening {market} market for top {top} picks...")


@main.command()
@click.option("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
def backtest(start: str):
    """Run backtesting over a historical window."""
    click.echo(f"Running backtest from {start}...")


if __name__ == "__main__":
    main()
