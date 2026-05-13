"""Backtrader-based backtesting engine (issue #6)."""

import bisect
import logging
from datetime import date, timedelta
from typing import Any

import backtrader as bt
import numpy as np
import polars as pl

from alphascreener.adapters.yfinance_adapter import OHLCV_COL_MAP
from alphascreener.types import ExitReason

logger = logging.getLogger(__name__)

_OHLCV_RENAME = {v: k for k, v in OHLCV_COL_MAP.items()}

COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.002
STOP_LOSS_PCT = 0.92
HOLDING_DAYS = 7
MAX_POSITIONS = 20
LOOKBACK_DAYS = HOLDING_DAYS * 2  # padding for T+1 entry + 7-day hold

_TRADE_RESULT_SCHEMA = {
    "ticker": pl.Utf8,
    "entry_date": pl.Date,
    "entry_price": pl.Float64,
    "exit_date": pl.Date,
    "exit_price": pl.Float64,
    "exit_reason": pl.Utf8,
    "pnl_pct": pl.Float64,
}


def _empty_trade_result() -> pl.DataFrame:
    return pl.DataFrame(schema=_TRADE_RESULT_SCHEMA)


class AlphaScreenerStrategy(bt.Strategy):
    """Backtrader strategy implementing alpha screener trading rules.

    Receives pre-computed entry dates per ticker. On each bar:
    - Enters new positions at open if today is an entry date (T+1 from signal)
    - Checks stop loss and holding period for existing positions
    - Exits at close price
    """

    _closed_trades: list[dict[str, Any]]

    def __init__(self, entry_map=None):
        """Initialize strategy.

        Args:
            entry_map: dict[ticker -> set of entry dates (as date objects)]
        """
        self._entry_map: dict[str, set[date]] = entry_map or {}
        self._holdings: dict[str, dict[str, Any]] = {}
        self._closed_trades = []

    def next(self):
        """Called for each bar (day)."""
        current_date = self.datas[0].datetime.date(0)

        # Process entries and exits for each data feed (ticker)
        for data in self.datas:
            ticker = data._name

            # Skip data feeds that don't have valid data for today
            if len(data) < 1:
                continue

            # 1. Check for entry (T+1 buy at open)
            if ticker in self._entry_map and current_date in self._entry_map[ticker]:
                if ticker not in self._holdings and len(self._holdings) < MAX_POSITIONS:
                    entry_price = data.open[0]
                    # Equal weight: allocate 1/MAX_POSITIONS of capital per position
                    size = self.broker.getvalue() / MAX_POSITIONS / entry_price
                    self.buy(data=data, size=size)
                    self._holdings[ticker] = {
                        "entry_price": entry_price,
                        "entry_date": current_date,
                        "bars_held": 0,
                        "size": size,
                    }

            # 2. Check exit conditions for held positions
            if ticker in self._holdings:
                h = self._holdings[ticker]
                h["bars_held"] += 1

                entry_price = h["entry_price"]
                exit_reason: str | None = None
                exit_price: float | None = None
                exit_date = current_date

                # Stop loss: if intraday Low <= entry * 0.92, exit at Close
                if data.low[0] <= entry_price * STOP_LOSS_PCT:
                    exit_reason = ExitReason.stop_loss.value
                    exit_price = data.close[0]

                # Hold expiry: 7 trading days
                elif h["bars_held"] >= HOLDING_DAYS:
                    exit_reason = ExitReason.hold_expiry.value
                    exit_price = data.close[0]

                if exit_reason is not None and exit_price is not None:
                    self.sell(data=data, size=h["size"])
                    # Friction: commission + slippage
                    friction_pct = COMMISSION_RATE * 2 + SLIPPAGE_RATE * 2  # both sides
                    pnl_pct = ((exit_price / entry_price) - 1) * 100 - friction_pct * 100
                    self._closed_trades.append(
                        {
                            "ticker": ticker,
                            "entry_date": h["entry_date"],
                            "entry_price": entry_price,
                            "exit_date": exit_date,
                            "exit_price": exit_price,
                            "exit_reason": exit_reason,
                            "pnl_pct": pnl_pct,
                        }
                    )
                    del self._holdings[ticker]

    def get_closed_trades(self) -> list[dict[str, Any]]:
        return self._closed_trades


class BacktestEngine:
    """Orchestrates backtrader backtesting runs.

    Public interface:
        run(ohlcv_df, signals_df) -> DataFrame of trade results
        run_incremental(ohlcv_df, signals_df, target_date) -> DataFrame
        backfill_paper_trades(db_path) -> None (reads paper_trades, simulates, writes back)
        compute_metrics(trades_df, benchmark_returns) -> dict of performance metrics
    """

    def __init__(self, initial_cash: float = 1_000_000.0):
        self._initial_cash = initial_cash

    def run(
        self,
        ohlcv_df: pl.DataFrame,
        signals_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Run a backtest and return trade results as a DataFrame.

        Args:
            ohlcv_df: OHLCV data with columns: ticker, date, open, high, low, close, volume
            signals_df: Signals with columns: ticker, signal_date, rating

        Returns:
            DataFrame with columns: ticker, entry_date, entry_price, exit_date,
            exit_price, exit_reason, pnl_pct
        """
        empty_result = _empty_trade_result()

        if ohlcv_df.is_empty() or signals_df.is_empty():
            return empty_result

        all_dates = sorted(ohlcv_df["date"].unique().to_list())

        entry_map: dict[str, set[date]] = {}
        for row in signals_df.iter_rows(named=True):
            ticker = row["ticker"]
            signal_date = row["signal_date"]
            idx = bisect.bisect_right(all_dates, signal_date)
            if idx >= len(all_dates):
                continue
            entry_date = all_dates[idx]

            if ticker not in entry_map:
                entry_map[ticker] = set()
            entry_map[ticker].add(entry_date)

        # Set up backtrader
        cerebro = bt.Cerebro()
        cerebro.addstrategy(AlphaScreenerStrategy, entry_map=entry_map)

        # Add data feeds for each ticker present in both OHLCV and signals
        signal_tickers = set(signals_df["ticker"].to_list())
        ohlcv_tickers = set(ohlcv_df["ticker"].unique().to_list())
        relevant_tickers = signal_tickers & ohlcv_tickers

        for ticker in relevant_tickers:
            ticker_df = ohlcv_df.filter(pl.col("ticker") == ticker).sort("date")
            data_df = ticker_df.to_pandas()
            data_df = data_df.set_index("date")
            data_df = data_df.rename(columns=_OHLCV_RENAME)

            data_feed = bt.feeds.PandasData(
                dataname=data_df,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                openinterest=-1,
            )
            data_feed._name = ticker
            cerebro.adddata(data_feed)

        if not relevant_tickers:
            return empty_result

        cerebro.broker.setcash(self._initial_cash)

        # Set commission: 0.1% per side, stock-like
        cerebro.broker.setcommission(commission=COMMISSION_RATE)
        results = cerebro.run()
        strat = results[0]

        trades = strat.get_closed_trades()

        if not trades:
            return empty_result

        return pl.DataFrame(trades)

    def run_incremental(
        self,
        ohlcv_df: pl.DataFrame,
        signals_df: pl.DataFrame,
        target_date: date,
    ) -> pl.DataFrame:
        """Run incremental backtest for a single target date's signals.

        Only backtests the signals for the given target_date against
        the OHLCV data (which should cover the required look-ahead period).

        Args:
            ohlcv_df: OHLCV data covering the target date and holding period
            signals_df: Signals DataFrame
            target_date: Only signals with this signal_date are tested

        Returns:
            Trade results DataFrame (typically 0-20 rows)
        """
        filtered_signals = signals_df.filter(pl.col("signal_date") == target_date)
        if filtered_signals.is_empty():
            return _empty_trade_result()
        return self.run(ohlcv_df, filtered_signals)

    def backfill_paper_trades(self, db_path) -> None:
        """Read paper_trades from DB, simulate with backtrader, write back results.

        Updates the paper_trades table with exit_price, exit_reason, and pnl_pct
        for each trade that has a null exit_price.

        Args:
            db_path: Path to the SQLite database
        """
        from alphascreener.core.storage import DataStore
        from alphascreener.db import get_db

        with get_db(db_path) as conn:
            rows = conn.execute(
                "SELECT id, signal_date, ticker, rating, factor_version "
                "FROM paper_trades "
                "WHERE exit_price IS NULL"
            ).fetchall()

            if not rows:
                return

            store = DataStore()

            for row in rows:
                trade_id, signal_date_str, ticker, rating, factor_version = row
                signal_date = date.fromisoformat(signal_date_str)
                end_date = signal_date + timedelta(days=LOOKBACK_DAYS)

                d = signal_date - timedelta(days=1)
                ohlcv_parts = []
                while d <= end_date:
                    df = store.read_ohlcv(d)
                    if df is not None:
                        ticker_df = df.filter(pl.col("ticker") == ticker)
                        if not ticker_df.is_empty():
                            ohlcv_parts.append(ticker_df)
                    d += timedelta(days=1)

                if not ohlcv_parts:
                    continue

                ohlcv_df = pl.concat(ohlcv_parts)

                signals_df = pl.DataFrame(
                    {
                        "ticker": [ticker],
                        "signal_date": [signal_date],
                        "rating": [rating],
                    }
                )

                result = self.run(ohlcv_df, signals_df)

                if not result.is_empty():
                    exit_price = float(result["exit_price"][0])
                    exit_reason = result["exit_reason"][0]
                    pnl_pct = float(result["pnl_pct"][0])

                    conn.execute(
                        "UPDATE paper_trades SET exit_price = ?, exit_reason = ?, pnl_pct = ? "
                        "WHERE id = ?",
                        (exit_price, exit_reason, pnl_pct, trade_id),
                    )

            conn.commit()

    @staticmethod
    def compute_metrics(trades_df: pl.DataFrame) -> dict[str, float]:
        """Compute performance metrics from trade results.

        Args:
            trades_df: Trade results with pnl_pct column

        Returns:
            dict with keys: win_rate, avg_return, profit_loss_ratio,
            annualized_return, sharpe_ratio, max_drawdown
        """
        if trades_df.is_empty():
            return {
                "win_rate": 0.0,
                "avg_return": 0.0,
                "profit_loss_ratio": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }

        pnl_pcts = trades_df["pnl_pct"].to_numpy().astype(np.float64)

        wins = np.sum(pnl_pcts > 0)
        total = len(pnl_pcts)
        win_rate = wins / total if total > 0 else 0.0

        avg_return = float(np.mean(pnl_pcts))

        positive_returns = pnl_pcts[pnl_pcts > 0]
        negative_returns = pnl_pcts[pnl_pcts < 0]
        avg_win = float(np.mean(positive_returns)) if len(positive_returns) > 0 else 0.0
        avg_loss = float(np.mean(np.abs(negative_returns))) if len(negative_returns) > 0 else 0.0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        if "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
            all_dates = (
                pl.concat([trades_df["entry_date"], trades_df["exit_date"]])
                .unique()
                .sort()
                .to_list()
            )
            if len(all_dates) >= 2:
                days_span = (all_dates[-1] - all_dates[0]).days
                years = max(days_span / 365.25, 0.019)
            else:
                years = 1.0
        else:
            years = 1.0

        cum_array = np.cumprod(1.0 + pnl_pcts / 100.0)
        annualized_return = (cum_array[-1] ** (1.0 / years) - 1.0) * 100.0

        if len(pnl_pcts) > 1:
            mean_return = float(np.mean(pnl_pcts))
            std_return = float(np.std(pnl_pcts, ddof=1))
            sharpe_ratio = (
                (mean_return / std_return) * np.sqrt(252.0 / HOLDING_DAYS)
                if std_return > 0
                else 0.0
            )
        else:
            sharpe_ratio = 0.0

        peak = np.maximum.accumulate(cum_array)
        drawdowns = (cum_array - peak) / peak
        max_drawdown = float(np.min(drawdowns) * 100) if len(drawdowns) > 0 else 0.0

        return {
            "win_rate": win_rate,
            "avg_return": avg_return,
            "profit_loss_ratio": profit_loss_ratio,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }
