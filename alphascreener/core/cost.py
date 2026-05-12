"""Cost circuit breaker for LLM API call budgeting (issue #14)."""

import json
import math
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


def _reject_non_finite_json(value):  # type: ignore[no-untyped-def]
    """parse_constant callback: reject NaN/Infinity in JSON."""
    raise ValueError(f"Non-finite JSON constant not allowed: {value!r}")


class CostCircuitBreaker:
    """Monitors daily and rolling-30-day LLM costs and trips breaker levels."""

    def __init__(self, settings) -> None:
        s = settings
        thresholds = [
            ("L1 warning", s.cost_l1_warning_daily_usd),
            ("L2 degrade", s.cost_l2_degrade_daily_usd),
            ("L3 savings", s.cost_l3_savings_monthly_avg_usd),
            ("L4 circuit", s.cost_l4_circuit_monthly_avg_usd),
        ]
        for name, val in thresholds:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(
                    f"{name} threshold must be a finite non-negative number, got {val!r}"
                )

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
        """Evaluate cost against thresholds and return the active breaker level."""
        today_str = date.today().isoformat()
        with get_db(self._settings.db_path) as conn:
            row = conn.execute(
                "SELECT "
                "COALESCE(SUM(CASE WHEN cost_date = ?1 "
                "THEN total_usd ELSE 0 END), 0), "
                "COALESCE(AVG(CASE WHEN total_usd IS NOT NULL AND total_usd = total_usd "
                "THEN total_usd ELSE NULL END), 0) "
                "FROM llm_cost_daily "
                "WHERE cost_date BETWEEN date(?2, '-29 days') AND ?2",
                (today_str, today_str),
            ).fetchone()
            today_cost_raw = row[0]
            rolling_mean_raw = row[1]

        today_cost = float(today_cost_raw)
        rolling_mean = float(rolling_mean_raw)

        if not math.isfinite(today_cost):
            return BreakerLevel.L4_CIRCUIT
        if not math.isfinite(rolling_mean):
            return BreakerLevel.L4_CIRCUIT

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
        """Insert or accumulate a daily cost row."""
        if not isinstance(total_usd, (int, float)) or not math.isfinite(total_usd) or total_usd < 0:
            raise ValueError(f"total_usd must be a finite number >= 0, got {total_usd}")
        if call_count < 0:
            raise ValueError(f"call_count must be >= 0, got {call_count}")
        parsed = json.loads(by_module_json, parse_constant=_reject_non_finite_json)
        if not isinstance(parsed, dict):
            raise ValueError(f"by_module_json must be a JSON object, got {type(parsed).__name__}")

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
