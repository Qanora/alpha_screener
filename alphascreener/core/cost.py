"""Cost circuit breaker for LLM API call budgeting (issue #14)."""

import json
from datetime import date
from enum import Enum

from alphascreener.db import get_db


class BreakerLevel(str, Enum):
    """Cost breaker levels, from low to high severity."""

    NORMAL = "normal"
    L1_WARNING = "l1_warning"
    L2_DEGRADE = "l2_degrade"
    L3_SAVINGS = "l3_savings"
    L4_CIRCUIT = "l4_circuit"


class CostCircuitBreaker:
    """Monitors daily and rolling-30-day LLM costs and trips breaker levels."""

    def __init__(self, settings) -> None:
        s = settings
        if s.cost_l1_warning_daily_usd > s.cost_l2_degrade_daily_usd:
            raise ValueError(
                f"L1 warning ({s.cost_l1_warning_daily_usd}) must be <= "
                f"L2 degrade ({s.cost_l2_degrade_daily_usd})"
            )
        if s.cost_l3_savings_monthly_avg_usd > s.cost_l4_circuit_monthly_avg_usd:
            raise ValueError(
                f"L3 savings ({s.cost_l3_savings_monthly_avg_usd}) must be <= "
                f"L4 circuit ({s.cost_l4_circuit_monthly_avg_usd})"
            )
        self._settings = s

    def check(self) -> BreakerLevel:
        """Evaluate cost against thresholds and return the active breaker level.

        Levels are checked from highest to lowest so more severe
        conditions always take precedence.
        """
        with get_db(self._settings.db_path) as conn:
            row = conn.execute(
                "SELECT "
                "COALESCE(SUM(CASE WHEN cost_date = date('now') "
                "THEN total_usd ELSE 0 END), 0), "
                "AVG(total_usd) "
                "FROM llm_cost_daily "
                "WHERE cost_date >= date('now', '-29 days')"
            ).fetchone()
            today_cost: float = row[0]
            rolling_mean: float = row[1] or 0.0

        if rolling_mean >= self._settings.cost_l4_circuit_monthly_avg_usd:
            return BreakerLevel.L4_CIRCUIT
        if rolling_mean >= self._settings.cost_l3_savings_monthly_avg_usd:
            return BreakerLevel.L3_SAVINGS
        if today_cost >= self._settings.cost_l2_degrade_daily_usd:
            return BreakerLevel.L2_DEGRADE
        if today_cost >= self._settings.cost_l1_warning_daily_usd:
            return BreakerLevel.L1_WARNING
        return BreakerLevel.NORMAL

    def record(
        self, cost_date: date, total_usd: float, call_count: int, by_module_json: str
    ) -> None:
        """Insert or accumulate a daily cost row.

        Uses INSERT ... ON CONFLICT DO UPDATE to accumulate total_usd
        and call_count for an existing date, and replace by_module_json
        with the latest value.
        """
        if total_usd < 0:
            raise ValueError(f"total_usd must be >= 0, got {total_usd}")
        if call_count < 0:
            raise ValueError(f"call_count must be >= 0, got {call_count}")
        json.loads(by_module_json)  # validate JSON parseable

        with get_db(self._settings.db_path) as conn:
            conn.execute(
                "INSERT INTO llm_cost_daily "
                "(cost_date, total_usd, call_count, by_module_json) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(cost_date) DO UPDATE SET "
                "total_usd = total_usd + excluded.total_usd, "
                "call_count = call_count + excluded.call_count, "
                "by_module_json = excluded.by_module_json",
                (cost_date.isoformat(), total_usd, call_count, by_module_json),
            )
            conn.commit()
