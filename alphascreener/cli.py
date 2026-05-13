"""CLI entry point."""

import logging

import click
import polars as pl

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
@click.option("--top", default=20, help="Number of top picks to output")
def screen(top: int):
    """Run a full market scan."""
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        raise click.BadParameter(f"top must be a positive integer, got {top!r}")
    from datetime import date, timedelta

    from alphascreener.adapters.yfinance_adapter import YFinanceAdapter
    from alphascreener.core.factors import compute_all_factors
    from alphascreener.core.screening import (
        MISSING_RATE_MAX,
        DynamicThreshold,
        compute_missing_rate,
        dedup_by_sector_industry,
        phase1_hard_filter,
        phase2_score,
        standardize_factors,
    )
    from alphascreener.core.storage import DataStore

    settings = get_settings()
    store = DataStore()
    today = date.today()
    yf_adapter = YFinanceAdapter(rps=settings.yfinance_rps)

    meta = store.read_universe_meta()
    if meta is None:
        click.echo("No universe data. Run 'alphascreener data sync' first.")
        return

    tickers = meta["ticker"].to_list()
    click.echo(f"Universe: {len(tickers)} tickers")

    click.echo("Loading OHLCV data...")
    from alphascreener.core.factors import MAX_LOOKBACK

    start = today - timedelta(days=MAX_LOOKBACK)
    ohlcv = yf_adapter.fetch_ohlcv_batch(tickers, start, today)
    if ohlcv.is_empty():
        click.echo("No OHLCV data available.")
        return

    store.write_ohlcv(ohlcv, today)
    click.echo(f"  {ohlcv.n_unique('ticker')} tickers, {len(ohlcv)} rows")

    click.echo("Computing 13 factors...")
    factor_df = compute_all_factors(ohlcv, tickers, yf_adapter)

    missing_rate = compute_missing_rate(factor_df)
    mask = missing_rate <= MISSING_RATE_MAX
    factor_df = factor_df.filter(mask)
    ticker_count = len(factor_df)
    click.echo(f"  {ticker_count} tickers after missing data filter")

    store.write_factors(factor_df, today)

    threshold = DynamicThreshold()
    phase1_df = phase1_hard_filter(factor_df, threshold.thresholds)
    after_filter = len(phase1_df)
    pass_rate = after_filter / max(ticker_count, 1)
    adj_thresholds, status, action = threshold.evaluate(pass_rate, today)

    click.echo(f"Phase 1: {after_filter} passed ({pass_rate:.1%}) [{status}, {action}]")

    if phase1_df.is_empty():
        click.echo("No tickers passed hard filter.")
        return

    z_df = standardize_factors(phase1_df)
    scored = phase2_score(z_df)

    if "sector" in meta.columns and "industry" in meta.columns:
        scored = scored.join(
            meta.select(["ticker", "sector", "industry"]),
            on="ticker",
            how="left",
        )
        top_n = dedup_by_sector_industry(
            scored,
            sector_cap=settings.sector_cap,
            industry_cap=settings.industry_cap,
            top_n=top,
        )
    else:
        top_n = scored.head(top)
    click.echo(f"\nTop {min(top, len(top_n))} by Coarse_Score:")
    for row in top_n.iter_rows(named=True):
        click.echo(f"  {row['ticker']:8s}  Coarse_Score={row.get('coarse_score', 0):+.4f}")

    store.write_signals(top_n, today, track="pure")


@main.command()
@click.option("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="Backtest end date (YYYY-MM-DD), defaults to today")
@click.option(
    "--mode",
    type=click.Choice(["full", "incremental", "backfill"]),
    default="full",
    help="Backtest mode: full (2-year window), incremental (latest day), backfill (paper_trades)",
)
def backtest(start: str, end: str | None, mode: str):
    """Run backtesting using the backtrader engine.

    \b
    Examples:
        alphascreener backtest --start 2023-01-01
        alphascreener backtest --start 2024-01-01 --end 2024-12-31
        alphascreener backtest --start 2024-01-01 --mode incremental
        alphascreener backtest --start 2024-01-01 --mode backfill
    """
    from datetime import date as dt_date, timedelta

    from alphascreener.config import get_settings
    from alphascreener.core.backtest import BacktestEngine
    from alphascreener.core.storage import DataStore

    settings = get_settings()
    store = DataStore()
    engine = BacktestEngine()

    start_date = dt_date.fromisoformat(start)
    if end:
        end_date = dt_date.fromisoformat(end)
    else:
        end_date = dt_date.today()

    click.echo(f"Backtest mode: {mode}")
    click.echo(f"Start date: {start_date}")
    click.echo(f"End date: {end_date}")

    if mode == "backfill":
        click.echo("Backfilling paper trades...")
        engine.backfill_paper_trades(settings.db_path)
        click.echo("Paper trades backfill complete.")
        return

    # Collect signals from parquet storage
    all_signals = []
    d = start_date
    while d <= end_date:
        signals_df = store.read_signals(d, track="llm")
        if signals_df is not None and not signals_df.is_empty():
            all_signals.append(signals_df)
        d += timedelta(days=1)

    if not all_signals:
        click.echo("No signals found in the date range.")
        return

    signals_df = pl.concat(all_signals)

    # Collect OHLCV data for the date range plus lookback and lookahead
    ohlcv_start = start_date - timedelta(days=14)
    ohlcv_end = end_date + timedelta(days=14)

    all_ohlcv = []
    d = ohlcv_start
    while d <= ohlcv_end:
        df = store.read_ohlcv(d)
        if df is not None and not df.is_empty():
            all_ohlcv.append(df)
        d += timedelta(days=1)

    if not all_ohlcv:
        click.echo("No OHLCV data found for the date range.")
        return

    ohlcv_df = pl.concat(all_ohlcv)

    if mode == "incremental":
        # Incremental: only backtest the target date's signals
        target = start_date
        results_df = engine.run_incremental(ohlcv_df, signals_df, target)
        click.echo(f"Incremental backtest for {target}:")
    else:
        # Full backtest
        results_df = engine.run(ohlcv_df, signals_df)

    if results_df.is_empty():
        click.echo("No trades generated.")
        return

    click.echo(f"\nGenerated {len(results_df)} trades:")
    click.echo(f"{'Ticker':<8s} {'Entry':>10s} {'Exit':>10s} {'PnL%':>8s} {'Reason'}")
    click.echo("-" * 55)
    for row in results_df.iter_rows(named=True):
        click.echo(
            f"{row['ticker']:<8s} "
            f"{str(row['entry_date']):>10s} "
            f"{str(row['exit_date']):>10s} "
            f"{row['pnl_pct']:>8.2f} "
            f"{row['exit_reason']}"
        )

    # Print performance metrics
    metrics = BacktestEngine.compute_metrics(results_df)
    click.echo("\nPerformance Metrics:")
    click.echo(f"  Win Rate:          {metrics['win_rate']:.2%}")
    click.echo(f"  Avg Return:        {metrics['avg_return']:.2f}%")
    click.echo(f"  Profit/Loss Ratio: {metrics['profit_loss_ratio']:.2f}")
    click.echo(f"  Annualized Return: {metrics['annualized_return']:.2f}%")
    click.echo(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    click.echo(f"  Max Drawdown:      {metrics['max_drawdown']:.2f}%")


if __name__ == "__main__":
    main()
