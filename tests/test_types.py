"""Tests for domain type definitions."""

from alphascreener.types import (
    FactorStatus,
    FinalRating,
    HARD_KILL_TAGS,
    Regime,
    RiskTag,
)


class TestFactorStatus:
    def test_all_statuses_defined(self):
        assert FactorStatus.proposed.value == "proposed"
        assert FactorStatus.probation.value == "probation"
        assert FactorStatus.active.value == "active"
        assert FactorStatus.degraded.value == "degraded"
        assert FactorStatus.retired.value == "retired"
        assert FactorStatus.rejected.value == "rejected"

    def test_str_compatible(self):
        assert FactorStatus.active.value == "active"
        assert isinstance(FactorStatus.active, str)


class TestRegime:
    def test_all_regimes(self):
        assert Regime.normal.value == "normal"
        assert Regime.low_activity.value == "low_activity"
        assert Regime.style_rotation.value == "style_rotation"
        assert Regime.crisis.value == "crisis"


class TestRiskTag:
    def test_all_tags(self):
        assert RiskTag.no_risk.value == "no_risk"
        assert RiskTag.data_conflict.value == "data_conflict"
        assert RiskTag.liquidity_trap.value == "liquidity_trap"
        assert RiskTag.delisting_risk.value == "delisting_risk"

    def test_hard_kill_tags_contains_expected(self):
        assert RiskTag.data_conflict in HARD_KILL_TAGS
        assert RiskTag.delisting_risk in HARD_KILL_TAGS
        assert RiskTag.no_risk not in HARD_KILL_TAGS
        assert RiskTag.liquidity_trap not in HARD_KILL_TAGS

    def test_hard_kill_tags_is_frozenset(self):
        assert isinstance(HARD_KILL_TAGS, frozenset)


class TestFinalRating:
    def test_all_ratings(self):
        assert FinalRating.strong_buy.value == "Strong Buy"
        assert FinalRating.buy.value == "Buy"
        assert FinalRating.hold.value == "Hold"
        assert FinalRating.avoid.value == "Avoid"
