"""CLI entry point."""

import click

from alphascreener import __version__
from alphascreener.config import get_settings
from alphascreener.db import get_db


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
