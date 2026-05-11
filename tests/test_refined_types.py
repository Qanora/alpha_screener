"""Tests for refined type models and validation in alphascreener.core.refined."""

import json

import pytest
from pydantic import ValidationError

from alphascreener.core.refined import (
    AnalystReport,
    BreakoutAssessment,
    BullBearOutput,
    validate_breakout_assessment,
)
from alphascreener.types import FinalRating, RiskTag


# ---------------------------------------------------------------------------
# AnalystReport
# ---------------------------------------------------------------------------


class TestAnalystReport:
    def test_valid_report_serializes_deserializes(self):
        report = AnalystReport(
            analyst_type="Market",
            ticker="AAPL",
            summary="Strong momentum observed.",
            bullish_signals=["RSI oversold rebound", "Volume spike"],
            bearish_signals=["MACD divergence"],
            concerns=["Earnings next week"],
        )
        data = report.model_dump()
        roundtripped = AnalystReport(**data)
        assert roundtripped.analyst_type == "Market"
        assert roundtripped.ticker == "AAPL"
        assert roundtripped.summary == "Strong momentum observed."
        assert roundtripped.bullish_signals == ["RSI oversold rebound", "Volume spike"]
        assert roundtripped.bearish_signals == ["MACD divergence"]
        assert roundtripped.concerns == ["Earnings next week"]

    def test_invalid_analyst_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AnalystReport(
                analyst_type="UnknownType",  # type: ignore[arg-type]
                ticker="AAPL",
                summary="...",
                bullish_signals=[],
                bearish_signals=[],
                concerns=[],
            )

    def test_empty_bullish_signals_is_valid(self):
        report = AnalystReport(
            analyst_type="News",
            ticker="MSFT",
            summary="No signals today.",
            bullish_signals=[],
            bearish_signals=["Some concern"],
            concerns=[],
        )
        assert report.bullish_signals == []


# ---------------------------------------------------------------------------
# BullBearOutput
# ---------------------------------------------------------------------------


class TestBullBearOutput:
    def test_valid_output_serializes_deserializes(self):
        output = BullBearOutput(
            ticker="GOOGL",
            bull_thesis="AI-driven revenue growth.",
            bear_thesis="Regulatory pressure in EU.",
        )
        data = output.model_dump()
        roundtripped = BullBearOutput(**data)
        assert roundtripped.ticker == "GOOGL"
        assert roundtripped.bull_thesis == "AI-driven revenue growth."
        assert roundtripped.bear_thesis == "Regulatory pressure in EU."

    def test_missing_required_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            BullBearOutput(ticker="GOOGL", bull_thesis="Growth story.")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# BreakoutAssessment (model validation)
# ---------------------------------------------------------------------------


class TestBreakoutAssessmentModel:
    def test_valid_assessment_all_fields(self):
        assessment = BreakoutAssessment(
            ticker="TSLA",
            score_correction=1.02,
            risk_tags=[RiskTag.no_risk],
            final_rating=FinalRating.buy,
            breakout_probability=0.75,
            rationale="Strong breakout pattern.",
        )
        assert assessment.ticker == "TSLA"
        assert assessment.score_correction == 1.02
        assert assessment.risk_tags == [RiskTag.no_risk]
        assert assessment.final_rating == FinalRating.buy
        assert assessment.breakout_probability == 0.75
        assert assessment.rationale == "Strong breakout pattern."

    def test_score_correction_lower_bound_valid(self):
        assessment = BreakoutAssessment(
            ticker="A",
            score_correction=0.90,
            risk_tags=[],
            final_rating=FinalRating.hold,
            breakout_probability=0.5,
            rationale="At lower bound.",
        )
        assert assessment.score_correction == 0.90

    def test_score_correction_upper_bound_valid(self):
        assessment = BreakoutAssessment(
            ticker="A",
            score_correction=1.05,
            risk_tags=[],
            final_rating=FinalRating.hold,
            breakout_probability=0.5,
            rationale="At upper bound.",
        )
        assert assessment.score_correction == 1.05

    def test_score_correction_below_0_90_raises(self):
        with pytest.raises(ValidationError):
            BreakoutAssessment(
                ticker="A",
                score_correction=0.89,
                risk_tags=[],
                final_rating=FinalRating.hold,
                breakout_probability=0.5,
                rationale="Too low.",
            )

    def test_score_correction_above_1_05_raises(self):
        with pytest.raises(ValidationError):
            BreakoutAssessment(
                ticker="A",
                score_correction=1.06,
                risk_tags=[],
                final_rating=FinalRating.hold,
                breakout_probability=0.5,
                rationale="Too high.",
            )

    def test_breakout_probability_0_valid(self):
        assessment = BreakoutAssessment(
            ticker="A",
            score_correction=1.0,
            risk_tags=[],
            final_rating=FinalRating.hold,
            breakout_probability=0.0,
            rationale="Zero prob.",
        )
        assert assessment.breakout_probability == 0.0

    def test_breakout_probability_1_valid(self):
        assessment = BreakoutAssessment(
            ticker="A",
            score_correction=1.0,
            risk_tags=[],
            final_rating=FinalRating.hold,
            breakout_probability=1.0,
            rationale="Full prob.",
        )
        assert assessment.breakout_probability == 1.0

    def test_breakout_probability_below_0_raises(self):
        with pytest.raises(ValidationError):
            BreakoutAssessment(
                ticker="A",
                score_correction=1.0,
                risk_tags=[],
                final_rating=FinalRating.hold,
                breakout_probability=-0.01,
                rationale="Negative prob.",
            )

    def test_breakout_probability_above_1_raises(self):
        with pytest.raises(ValidationError):
            BreakoutAssessment(
                ticker="A",
                score_correction=1.0,
                risk_tags=[],
                final_rating=FinalRating.hold,
                breakout_probability=1.01,
                rationale="Over 1 prob.",
            )

    def test_valid_risk_tags_enum_values(self):
        assessment = BreakoutAssessment(
            ticker="A",
            score_correction=1.0,
            risk_tags=[RiskTag.liquidity_trap, RiskTag.earnings_timing_mismatch],
            final_rating=FinalRating.hold,
            breakout_probability=0.3,
            rationale="Multiple tags.",
        )
        assert RiskTag.liquidity_trap in assessment.risk_tags
        assert RiskTag.earnings_timing_mismatch in assessment.risk_tags


# ---------------------------------------------------------------------------
# validate_breakout_assessment
# ---------------------------------------------------------------------------


class TestValidateBreakoutAssessment:
    def test_valid_json_parses_correctly(self):
        raw = json.dumps(
            {
                "ticker": "NVDA",
                "score_correction": 1.03,
                "risk_tags": ["liquidity_trap"],
                "final_rating": "Strong Buy",
                "breakout_probability": 0.88,
                "rationale": "AI boom continues.",
            }
        )
        result = validate_breakout_assessment(raw)
        assert isinstance(result, BreakoutAssessment)
        assert result.ticker == "NVDA"
        assert result.score_correction == 1.03
        assert result.risk_tags == [RiskTag.liquidity_trap]
        assert result.final_rating == FinalRating.strong_buy
        assert result.breakout_probability == 0.88

    def test_unparseable_json_returns_defaults_after_retry(self):
        result = validate_breakout_assessment("not valid json at all {{{")
        assert isinstance(result, BreakoutAssessment)
        assert result.score_correction == 1.0
        assert result.risk_tags == []
        assert result.final_rating == FinalRating.hold
        assert result.breakout_probability == 0.0

    def test_score_correction_below_0_90_clamped(self):
        raw = json.dumps(
            {
                "ticker": "X",
                "score_correction": 0.50,
                "risk_tags": [],
                "final_rating": "Hold",
                "breakout_probability": 0.5,
                "rationale": "Low score.",
            }
        )
        result = validate_breakout_assessment(raw)
        assert result.score_correction == 0.90

    def test_score_correction_above_1_05_clamped(self):
        raw = json.dumps(
            {
                "ticker": "X",
                "score_correction": 2.00,
                "risk_tags": [],
                "final_rating": "Hold",
                "breakout_probability": 0.5,
                "rationale": "High score.",
            }
        )
        result = validate_breakout_assessment(raw)
        assert result.score_correction == 1.05

    def test_invalid_risk_tags_filtered_out(self):
        raw = json.dumps(
            {
                "ticker": "X",
                "score_correction": 1.0,
                "risk_tags": [
                    "liquidity_trap",
                    "not_a_valid_tag",
                    "earnings_timing_mismatch",
                    "also_invalid",
                ],
                "final_rating": "Hold",
                "breakout_probability": 0.5,
                "rationale": "Mixed tags.",
            }
        )
        result = validate_breakout_assessment(raw)
        assert RiskTag.liquidity_trap in result.risk_tags
        assert RiskTag.earnings_timing_mismatch in result.risk_tags
        assert len(result.risk_tags) == 2  # only the two valid ones survive

    def test_data_conflict_forces_avoid(self):
        raw = json.dumps(
            {
                "ticker": "X",
                "score_correction": 1.0,
                "risk_tags": ["data_conflict"],
                "final_rating": "Strong Buy",
                "breakout_probability": 0.9,
                "rationale": "Conflicting data.",
            }
        )
        result = validate_breakout_assessment(raw)
        assert result.final_rating == FinalRating.avoid

    def test_delisting_risk_forces_avoid(self):
        raw = json.dumps(
            {
                "ticker": "X",
                "score_correction": 1.0,
                "risk_tags": ["delisting_risk"],
                "final_rating": "Buy",
                "breakout_probability": 0.7,
                "rationale": "Delisting risk.",
            }
        )
        result = validate_breakout_assessment(raw)
        assert result.final_rating == FinalRating.avoid

    def test_invalid_final_rating_defaults_to_hold(self):
        raw = json.dumps(
            {
                "ticker": "X",
                "score_correction": 1.0,
                "risk_tags": [],
                "final_rating": "NotAValidRating",
                "breakout_probability": 0.5,
                "rationale": "Weird rating.",
            }
        )
        result = validate_breakout_assessment(raw)
        assert result.final_rating == FinalRating.hold

    def test_missing_fields_use_model_defaults(self):
        # Only provide ticker — all other fields missing
        raw = json.dumps({"ticker": "MISSING_FIELDS"})
        result = validate_breakout_assessment(raw)
        assert isinstance(result, BreakoutAssessment)
        assert result.ticker == "MISSING_FIELDS"

    def test_default_ticker_used_when_missing(self):
        raw = json.dumps(
            {
                "score_correction": 1.0,
                "risk_tags": [],
                "final_rating": "Hold",
                "breakout_probability": 0.5,
                "rationale": "No ticker given.",
            }
        )
        result = validate_breakout_assessment(raw, default_ticker="UNKNOWN")
        assert result.ticker == "UNKNOWN"

    def test_breakout_probability_above_1_clamped(self):
        raw = json.dumps(
            {
                "ticker": "X",
                "score_correction": 1.0,
                "risk_tags": [],
                "final_rating": "Hold",
                "breakout_probability": 1.5,
                "rationale": "Too high probability.",
            }
        )
        result = validate_breakout_assessment(raw)
        assert result.breakout_probability == 1.0

    def test_breakout_probability_below_0_clamped(self):
        raw = json.dumps(
            {
                "ticker": "X",
                "score_correction": 1.0,
                "risk_tags": [],
                "final_rating": "Hold",
                "breakout_probability": -0.5,
                "rationale": "Negative probability.",
            }
        )
        result = validate_breakout_assessment(raw)
        assert result.breakout_probability == 0.0

    def test_json_array_returns_defaults(self):
        result = validate_breakout_assessment("[]")
        assert isinstance(result, BreakoutAssessment)
        assert result.score_correction == 1.0
        assert result.risk_tags == []
        assert result.final_rating == FinalRating.hold

    def test_risk_tags_null_treated_as_empty(self):
        raw = json.dumps({"ticker": "X", "risk_tags": None, "final_rating": "Hold"})
        result = validate_breakout_assessment(raw)
        assert result.risk_tags == []

    def test_risk_tags_number_treated_as_empty(self):
        raw = json.dumps({"ticker": "X", "risk_tags": 123, "final_rating": "Hold"})
        result = validate_breakout_assessment(raw)
        assert result.risk_tags == []

    def test_risk_tags_object_treated_as_empty(self):
        raw = json.dumps({"ticker": "X", "risk_tags": {"a": 1}, "final_rating": "Hold"})
        result = validate_breakout_assessment(raw)
        assert result.risk_tags == []

    def test_final_rating_null_defaults_to_hold(self):
        raw = json.dumps({"ticker": "X", "risk_tags": [], "final_rating": None})
        result = validate_breakout_assessment(raw)
        assert result.final_rating == FinalRating.hold

    def test_final_rating_number_defaults_to_hold(self):
        raw = json.dumps({"ticker": "X", "risk_tags": [], "final_rating": 123})
        result = validate_breakout_assessment(raw)
        assert result.final_rating == FinalRating.hold
