"""Refined type models for analyst reports, bull/bear output, and breakout assessments."""

import json
from typing import Literal

from pydantic import BaseModel, Field

from alphascreener.types import FinalRating, HARD_KILL_TAGS, RiskTag


class AnalystReport(BaseModel):
    """Report from a single analyst perspective (Market, News, Fundamentals, Breakout)."""

    analyst_type: Literal["Market", "News", "Fundamentals", "Breakout"]
    ticker: str
    summary: str
    bullish_signals: list[str]
    bearish_signals: list[str]
    concerns: list[str]


class BullBearOutput(BaseModel):
    """Synthesised bull vs bear thesis for a ticker."""

    ticker: str
    bull_thesis: str
    bear_thesis: str


class BreakoutAssessment(BaseModel):
    """Post-screening breakout assessment with risk tags and scoring adjustments."""

    ticker: str = "UNKNOWN"
    score_correction: float = Field(default=1.0, ge=0.90, le=1.05)
    risk_tags: list[RiskTag] = Field(default_factory=list)
    final_rating: FinalRating = Field(default=FinalRating.hold)
    breakout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


def validate_breakout_assessment(
    raw_json: str, default_ticker: str = "UNKNOWN"
) -> BreakoutAssessment:
    """Parse and validate a raw JSON string into a BreakoutAssessment.

    Fallback behaviours:
    1. JSON parse failure → use defaults.
    2. *score_correction* outside [0.90, 1.05] → clamped to nearest bound.
    3. Invalid *risk_tags* values → filtered out (only valid :class:`RiskTag` members kept).
    4. Invalid *final_rating* → defaulted to :attr:`FinalRating.hold`.
    5. Any hard-kill tag present in *risk_tags* → forces *final_rating* to
       :attr:`FinalRating.avoid`.
    """
    try:
        parsed: dict = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        return BreakoutAssessment(ticker=default_ticker)

    if "score_correction" in parsed:
        sc = parsed["score_correction"]
        if isinstance(sc, (int, float)):
            parsed["score_correction"] = max(0.90, min(1.05, sc))

    if "risk_tags" in parsed:
        valid_tags: list[RiskTag] = []
        has_hard_kill = False
        for tag in parsed["risk_tags"]:
            try:
                rt = RiskTag(tag)
                valid_tags.append(rt)
                if rt in HARD_KILL_TAGS:
                    has_hard_kill = True
            except ValueError:
                continue
        parsed["risk_tags"] = valid_tags
        if has_hard_kill:
            parsed["final_rating"] = FinalRating.avoid

    if "final_rating" in parsed and parsed.get("final_rating") is not FinalRating.avoid:
        try:
            parsed["final_rating"] = FinalRating(parsed["final_rating"])
        except ValueError:
            parsed["final_rating"] = FinalRating.hold

    if "ticker" not in parsed:
        parsed["ticker"] = default_ticker

    return BreakoutAssessment(**parsed)
