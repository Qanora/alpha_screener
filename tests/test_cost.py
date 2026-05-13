"""Tests for cost circuit breaker (issue #14)."""

import json
from datetime import date, timedelta

import pytest

from alphascreener.db import get_db, init_db


def _today_str() -> str:
    return date.today().isoformat()


class TestBreakerLevel:
    """Verify BreakerLevel enum has all expected values."""

    def test_all_five_enum_values_exist(self):
        from alphascreener.core.cost import BreakerLevel

        assert BreakerLevel.NORMAL == "normal"
        assert BreakerLevel.L1_WARNING == "l1_warning"
        assert BreakerLevel.L2_DEGRADE == "l2_degrade"
        assert BreakerLevel.L3_SAVINGS == "l3_savings"
        assert BreakerLevel.L4_CIRCUIT == "l4_circuit"

        values = {e.value for e in BreakerLevel}
        assert values == {"normal", "l1_warning", "l2_degrade", "l3_savings", "l4_circuit"}


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary DB at the location Settings.db_path expects."""
    db_path = tmp_path / "db" / "metadata.db"
    init_db(db_path)
    return db_path


@pytest.fixture
def settings(tmp_path):
    """Create Settings pointing at the temporary home."""
    from alphascreener.config import Settings

    return Settings(alphascreener_home=str(tmp_path))


class TestCostCircuitBreakerCheck:
    """Test CostCircuitBreaker.check() breaker level logic."""

    def test_returns_normal_when_table_empty(self, temp_db_path, settings):
        from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        assert cb.check() == BreakerLevel.NORMAL

    def test_returns_normal_below_all_thresholds(self, temp_db_path, settings):
        from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker

        with get_db(temp_db_path) as conn:
            conn.execute(
                "INSERT INTO llm_cost_daily (cost_date, total_usd, call_count, by_module_json) "
                "VALUES (?, 0.50, 5, '{}')",
                (_today_str(),),
            )
            conn.commit()

        cb = CostCircuitBreaker(settings)
        assert cb.check() == BreakerLevel.NORMAL

    def test_returns_l1_warning_when_daily_ge_80_cents(self, temp_db_path, settings):
        from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker

        with get_db(temp_db_path) as conn:
            conn.execute(
                "INSERT INTO llm_cost_daily (cost_date, total_usd, call_count, by_module_json) "
                "VALUES (?, 0.80, 10, '{}')",
                (_today_str(),),
            )
            conn.commit()

        cb = CostCircuitBreaker(settings)
        assert cb.check() == BreakerLevel.L1_WARNING

    def test_returns_l2_degrade_when_daily_ge_1_dollar(self, temp_db_path, settings):
        from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker

        with get_db(temp_db_path) as conn:
            conn.execute(
                "INSERT INTO llm_cost_daily (cost_date, total_usd, call_count, by_module_json) "
                "VALUES (?, 1.00, 12, '{}')",
                (_today_str(),),
            )
            conn.commit()

        cb = CostCircuitBreaker(settings)
        assert cb.check() == BreakerLevel.L2_DEGRADE

    def test_returns_l3_savings_when_rolling_mean_ge_2_67(self, temp_db_path, settings):
        from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker

        with get_db(temp_db_path) as conn:
            for i in range(30):
                d = (date.today() - timedelta(days=i)).isoformat()
                conn.execute(
                    "INSERT INTO llm_cost_daily (cost_date, total_usd, call_count, by_module_json) "
                    "VALUES (?, 2.67, 30, '{}')",
                    (d,),
                )
            conn.commit()

        cb = CostCircuitBreaker(settings)
        assert cb.check() == BreakerLevel.L3_SAVINGS

    def test_returns_l4_circuit_when_rolling_mean_ge_3_17(self, temp_db_path, settings):
        from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker

        with get_db(temp_db_path) as conn:
            for i in range(30):
                d = (date.today() - timedelta(days=i)).isoformat()
                conn.execute(
                    "INSERT INTO llm_cost_daily (cost_date, total_usd, call_count, by_module_json) "
                    "VALUES (?, 3.17, 40, '{}')",
                    (d,),
                )
            conn.commit()

        cb = CostCircuitBreaker(settings)
        assert cb.check() == BreakerLevel.L4_CIRCUIT

    def test_l3_over_l2_when_both_triggered(self, temp_db_path, settings):
        """When daily cost >= 1.00 and rolling mean >= 2.67, L3 wins (higher level)."""
        from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker

        with get_db(temp_db_path) as conn:
            conn.execute(
                "INSERT INTO llm_cost_daily (cost_date, total_usd, call_count, by_module_json) "
                "VALUES (?, 1.00, 12, '{}')",
                (_today_str(),),
            )
            for i in range(1, 30):
                d = (date.today() - timedelta(days=i)).isoformat()
                conn.execute(
                    "INSERT INTO llm_cost_daily (cost_date, total_usd, call_count, by_module_json) "
                    "VALUES (?, 2.73, 30, '{}')",
                    (d,),
                )
            conn.commit()

        cb = CostCircuitBreaker(settings)
        assert cb.check() == BreakerLevel.L3_SAVINGS

    def test_l4_over_all_lower_levels(self, temp_db_path, settings):
        """L4_CIRCUIT takes precedence over L3, L2, L1."""
        from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker

        with get_db(temp_db_path) as conn:
            for i in range(30):
                d = (date.today() - timedelta(days=i)).isoformat()
                conn.execute(
                    "INSERT INTO llm_cost_daily (cost_date, total_usd, call_count, by_module_json) "
                    "VALUES (?, 3.17, 40, '{}')",
                    (d,),
                )
            conn.commit()

        cb = CostCircuitBreaker(settings)
        assert cb.check() == BreakerLevel.L4_CIRCUIT

    def test_future_dates_excluded_from_rolling_mean(self, temp_db_path, settings):
        """Rows with cost_date > today must not affect the rolling mean."""
        from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker

        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        with get_db(temp_db_path) as conn:
            conn.execute(
                "INSERT INTO llm_cost_daily (cost_date, total_usd, call_count, by_module_json) "
                "VALUES (?, 10.00, 100, '{}')",
                (tomorrow,),
            )
            conn.commit()

        cb = CostCircuitBreaker(settings)
        assert cb.check() == BreakerLevel.NORMAL


class TestCostCircuitBreakerRecord:
    """Test CostCircuitBreaker.record() insert/update behaviour."""

    def test_inserts_new_row_when_date_not_exists(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        cb.record(date.today(), 0.50, 5, '{"screening": 0.50}')

        with get_db(temp_db_path) as conn:
            row = conn.execute(
                "SELECT total_usd, call_count, by_module_json "
                "FROM llm_cost_daily WHERE cost_date = ?",
                (_today_str(),),
            ).fetchone()

        assert row[0] == 0.50
        assert row[1] == 5
        assert row[2] == '{"screening": 0.50}'

    def test_updates_existing_row_accumulates_amounts(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        cb.record(date.today(), 0.50, 5, '{"screening": 0.50}')
        cb.record(date.today(), 0.30, 3, '{"eval": 0.30}')

        with get_db(temp_db_path) as conn:
            row = conn.execute(
                "SELECT total_usd, call_count, by_module_json "
                "FROM llm_cost_daily WHERE cost_date = ?",
                (_today_str(),),
            ).fetchone()

        assert row[0] == 0.80  # 0.50 + 0.30
        assert row[1] == 8  # 5 + 3
        assert json.loads(row[2]) == {"screening": 0.50, "eval": 0.30}  # merged

    def test_stores_by_module_json_as_provided(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        module_json = json.dumps({"screening": 0.25, "eval": 0.15})

        cb = CostCircuitBreaker(settings)
        cb.record(date.today(), 0.40, 8, module_json)

        with get_db(temp_db_path) as conn:
            row = conn.execute(
                "SELECT by_module_json FROM llm_cost_daily WHERE cost_date = ?",
                (_today_str(),),
            ).fetchone()

        assert row[0] == module_json


class TestCostCircuitBreakerValidation:
    """Test input validation in __init__ and record()."""

    def test_init_raises_when_l1_gt_l2(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        settings.cost_l1_warning_daily_usd = 1.50
        settings.cost_l2_degrade_daily_usd = 1.00
        with pytest.raises(ValueError, match="L1 warning"):
            CostCircuitBreaker(settings)

    def test_init_raises_when_l3_gt_l4(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        settings.cost_l3_savings_monthly_avg_usd = 4.00
        settings.cost_l4_circuit_monthly_avg_usd = 3.00
        with pytest.raises(ValueError, match="L3 savings"):
            CostCircuitBreaker(settings)

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf"), -1.0])
    def test_init_raises_on_non_finite_threshold(self, temp_db_path, settings, bad_value):
        from alphascreener.core.cost import CostCircuitBreaker

        settings.cost_l1_warning_daily_usd = bad_value
        with pytest.raises(ValueError, match="finite"):
            CostCircuitBreaker(settings)

    def test_record_raises_on_negative_cost(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        with pytest.raises(ValueError, match="total_usd"):
            cb.record(date.today(), -0.10, 1, "{}")

    def test_record_raises_on_negative_call_count(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        with pytest.raises(ValueError, match="call_count"):
            cb.record(date.today(), 0.50, -1, "{}")

    def test_record_raises_on_invalid_json(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        with pytest.raises(json.JSONDecodeError):
            cb.record(date.today(), 0.50, 1, "not json")

    def test_record_raises_on_non_object_json(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        with pytest.raises(ValueError, match="JSON object"):
            cb.record(date.today(), 0.50, 1, "[1, 2, 3]")

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
    def test_record_raises_on_non_finite_cost(self, temp_db_path, settings, bad_value):
        from alphascreener.core.cost import CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        with pytest.raises(ValueError, match="finite"):
            cb.record(date.today(), bad_value, 1, "{}")

    def test_record_raises_on_nan_in_json(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        with pytest.raises(ValueError, match="Non-finite"):
            cb.record(date.today(), 0.50, 1, '{"cost": NaN}')

    def test_record_raises_on_float_overflow_in_json(self, temp_db_path, settings):
        from alphascreener.core.cost import CostCircuitBreaker

        cb = CostCircuitBreaker(settings)
        with pytest.raises(ValueError, match="Non-finite"):
            cb.record(date.today(), 0.50, 1, '{"cost": 1e309}')
