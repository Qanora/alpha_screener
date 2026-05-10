"""Factor state machine and schedule definitions."""

from enum import Enum


class FactorStatus(str, Enum):
    proposed = "proposed"
    probation = "probation"
    active = "active"
    degraded = "degraded"
    retired = "retired"
    rejected = "rejected"


class Regime(str, Enum):
    normal = "normal"
    low_activity = "low_activity"
    style_rotation = "style_rotation"
    crisis = "crisis"


class RiskTag(str, Enum):
    no_risk = "no_risk"
    data_conflict = "data_conflict"
    liquidity_trap = "liquidity_trap"
    delisting_risk = "delisting_risk"
    earnings_timing_mismatch = "earnings_timing_mismatch"
    catalyst_already_passed = "catalyst_already_passed"


HARD_KILL_TAGS: frozenset[RiskTag] = frozenset({RiskTag.data_conflict, RiskTag.delisting_risk})


class FinalRating(str, Enum):
    strong_buy = "Strong Buy"
    buy = "Buy"
    hold = "Hold"
    avoid = "Avoid"
