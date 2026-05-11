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

    Fallback behaviours (in order):
    1. JSON parse failure → retry once, then use defaults.
    2. *score_correction* outside [0.90, 1.05] → clamped to nearest bound.
    3. Invalid *risk_tags* values → filtered out (only valid :class:`RiskTag` members kept).
    4. Invalid *final_rating* → defaulted to :attr:`FinalRating.hold`.
    5. Any hard-kill tag present in *risk_tags* → forces *final_rating* to
       :attr:`FinalRating.avoid`.
    """
    # -- 1. Parse JSON with one retry -------------------------------------------
    parsed: dict | None = None
    for _ in range(2):
        try:
            parsed = json.loads(raw_json)
            break
        except (json.JSONDecodeError, ValueError):
            continue

    if parsed is None:
        return BreakoutAssessment(ticker=default_ticker)

    # -- 2. Clamp score_correction ---------------------------------------------
    if "score_correction" in parsed:
        sc = parsed["score_correction"]
        if isinstance(sc, (int, float)):
            if sc < 0.90:
                parsed["score_correction"] = 0.90
            elif sc > 1.05:
                parsed["score_correction"] = 1.05

    # -- 3. Filter risk_tags to valid RiskTag members -------------------------
    if "risk_tags" in parsed:
        valid_tags: list[RiskTag] = []
        for tag in parsed["risk_tags"]:
            try:
                valid_tags.append(RiskTag(tag))
            except ValueError:
                continue
        parsed["risk_tags"] = valid_tags

    # -- 4. Validate final_rating, default to Hold ----------------------------
    if "final_rating" in parsed:
        try:
            parsed["final_rating"] = FinalRating(parsed["final_rating"])
        except ValueError:
            parsed["final_rating"] = FinalRating.hold

    # -- 5. Hard-kill tags force Avoid -----------------------------------------
    risk_tags = parsed.get("risk_tags", [])
    if any(tag in HARD_KILL_TAGS for tag in risk_tags):
        parsed["final_rating"] = FinalRating.avoid

    # -- Apply default ticker if missing ---------------------------------------
    if "ticker" not in parsed:
        parsed["ticker"] = default_ticker

    return BreakoutAssessment(**parsed)
