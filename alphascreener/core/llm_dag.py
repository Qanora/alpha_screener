"""LLM refinement DAG: 4 Analysts → Bull∥Bear Researchers → PM Risk Audit (issue #15)."""

import asyncio
import json
import logging
from datetime import date
from typing import Any

import httpx
import numpy as np
from pydantic import ValidationError

from alphascreener.core.cost import BreakerLevel, CostCircuitBreaker
from alphascreener.core.rate_limiter import RateLimiter
from alphascreener.core.refined import (
    AnalystReport,
    BreakoutAssessment,
    BullBearOutput,
    validate_breakout_assessment,
)

logger = logging.getLogger(__name__)

_COST_PER_1M_INPUT = 0.15
_COST_PER_1M_OUTPUT = 0.60


def _compute_llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * _COST_PER_1M_INPUT + completion_tokens * _COST_PER_1M_OUTPUT
    ) / 1_000_000


def _extract_json_from_text(text: str) -> str:
    """Extract JSON content from LLM response text.

    Handles responses wrapped in ```json ... ``` fences or with leading/trailing text.
    """
    text = text.strip()
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    else:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            return text[brace_start : brace_end + 1]
    return text


class BreakoutCaseLibrary:
    """FAISS IndexFlatIP case library for historical breakout patterns.

    Stores case embeddings and supports cosine similarity search (via
    L2-normalized inner product) with a configurable similarity threshold.
    """

    def __init__(self, dimension: int = 384) -> None:
        import faiss

        self._faiss = faiss
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: list[dict[str, Any]] = []

    def add_cases(self, cases: list[dict[str, Any]]) -> None:
        """Add cases to the index. Each case must have an 'embedding' key."""
        if not cases:
            return
        embeddings = np.array([c["embedding"] for c in cases], dtype=np.float32)
        # L2-normalize for cosine similarity via inner product
        self._faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._metadata.extend(cases)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Search for the top-k most similar cases.

        Returns cases with cosine similarity >= min_similarity, sorted descending.
        Each result is a dict with the case metadata + a "similarity" key.
        """
        if self._index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(query)

        distances, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            similarity = float(dist)
            if similarity >= min_similarity:
                case = dict(self._metadata[idx])
                case["similarity"] = similarity
                results.append(case)

        return results

    @property
    def size(self) -> int:
        """Number of cases in the library."""
        return self._index.ntotal


class LLMClient:
    """Async wrapper around OpenAI-compatible chat completion API.

    Handles JSON response parsing, cost tracking, and rate limiting.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._model = settings.llm_model
        self._api_key = getattr(settings, "openai_api_key", None) or ""
        self._base_url = getattr(settings, "openai_base_url", None) or "https://api.openai.com/v1"
        self._rate_limiter = RateLimiter(settings.llm_rps)
        self._cost_breaker = CostCircuitBreaker(settings)
        self._cost_records: list[dict[str, Any]] = []
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0)
            )
        return self._http_client

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """Send a chat completion request and return the response text."""
        await self._rate_limiter.acquire()

        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = await client.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost_usd = _compute_llm_cost(prompt_tokens, completion_tokens)

        self._record_llm_cost("llm_dag", prompt_tokens, completion_tokens, cost_usd)

        return content

    async def chat_with_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_retries: int = 1,
    ) -> str:
        """Send a chat request and extract JSON from the response. Retry once on failure."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                raw_response = await self.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                json_text = _extract_json_from_text(raw_response)
                # Validate it's parseable JSON
                json.loads(json_text)
                return json_text
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning("LLM JSON parse failed, retrying: %s", e)
                    user_prompt = (
                        f"{user_prompt}\n\nYour previous response was not valid JSON. "
                        f"Please respond with ONLY valid JSON, no other text."
                    )
                else:
                    logger.error("LLM JSON parse failed after %d retries: %s", max_retries + 1, e)
                    raise

        # Should not be reached, but satisfy type checker
        raise last_error  # type: ignore[misc]

    async def _parse_analyst_report(self, json_text: str, ticker: str) -> AnalystReport:
        """Parse JSON into AnalystReport with fallback on failure."""
        try:
            parsed = json.loads(json_text)
            if not isinstance(parsed, dict):
                raise ValueError("Not a JSON object")
            return AnalystReport(**parsed)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            return AnalystReport(
                analyst_type="Market",
                ticker=ticker,
                summary="LLM response could not be parsed.",
                bullish_signals=[],
                bearish_signals=[],
                concerns=["LLM parsing failure"],
            )

    async def _parse_bull_bear_output(self, json_text: str, ticker: str) -> BullBearOutput:
        """Parse JSON into BullBearOutput with fallback on failure."""
        try:
            parsed = json.loads(json_text)
            if not isinstance(parsed, dict):
                raise ValueError("Not a JSON object")
            return BullBearOutput(**parsed)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            return BullBearOutput(
                ticker=ticker,
                bull_thesis="No thesis available (LLM parsing failure).",
                bear_thesis="No thesis available (LLM parsing failure).",
            )

    async def _parse_breakout_assessment(self, json_text: str, ticker: str) -> BreakoutAssessment:
        """Parse JSON into BreakoutAssessment using the existing validator."""
        return validate_breakout_assessment(json_text, default_ticker=ticker)

    def _record_llm_cost(
        self,
        module: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None = None,
    ) -> None:
        """Record LLM API cost for later flushing to the circuit breaker."""
        if cost_usd is None:
            cost_usd = _compute_llm_cost(prompt_tokens, completion_tokens)
        self._cost_records.append(
            {
                "module": module,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
            }
        )

    async def flush_costs(self) -> None:
        """Flush all pending cost records to the CostCircuitBreaker."""
        if not self._cost_records:
            return

        # Aggregate by module
        by_module: dict[str, float] = {}
        total_usd = 0.0
        total_calls = len(self._cost_records)

        for rec in self._cost_records:
            module = rec["module"]
            by_module[module] = by_module.get(module, 0.0) + rec["cost_usd"]
            total_usd += rec["cost_usd"]

        try:
            self._cost_breaker.record(
                cost_date=date.today(),
                total_usd=total_usd,
                call_count=total_calls,
                by_module_json=json.dumps(by_module),
            )
            self._cost_records.clear()
        except Exception as e:
            logger.warning("Failed to flush LLM costs, will retry next batch: %s", e)

    def check_breaker(self):
        """Check the cost circuit breaker level. Returns the active BreakerLevel."""
        return self._cost_breaker.check()

    async def close(self) -> None:
        """Close the HTTP client and flush costs."""
        await self.flush_costs()
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# ---------------------------------------------------------------------------
# Analyst functions (parallel per ticker)
# ---------------------------------------------------------------------------


_ANALYST_SYSTEM_PROMPT = """You are a quantitative stock analyst. Your task is to analyze a stock and produce a structured JSON report.
Always respond with ONLY valid JSON. No explanations, no markdown fences, just raw JSON.

The JSON must have these exact fields:
- analyst_type: string, one of "Market", "News", "Fundamentals", "Breakout"
- ticker: string, the stock ticker symbol
- summary: string, a concise 1-2 sentence summary of your analysis
- bullish_signals: array of strings, each a specific bullish signal you identified
- bearish_signals: array of strings, each a specific bearish signal you identified
- concerns: array of strings, risk factors or uncertainties to monitor"""


def _build_analyst_user_prompt(
    analyst_type: str,
    ticker_data: dict,
    extra_context: str = "",
) -> str:
    """Build the user prompt for an analyst LLM call."""
    ticker = ticker_data["ticker"]
    sector = ticker_data.get("sector", "Unknown")
    industry = ticker_data.get("industry", "Unknown")

    ohlcv = ticker_data.get("ohlcv_summary", {})
    factors = ticker_data.get("factor_scores", {})

    lines = [
        f"## {analyst_type} Analyst Report for {ticker}",
        f"Sector: {sector}",
        f"Industry: {industry}",
        "",
        "### OHLCV Summary",
        f"Close: {ohlcv.get('close', 'N/A')}",
        f"5-day Change: {ohlcv.get('change_5d_pct', 'N/A')}%",
        f"20-day Avg Volume: {ohlcv.get('volume_avg_20d', 'N/A')}",
        f"52-week High: {ohlcv.get('high_52w', 'N/A')}",
        f"52-week Low: {ohlcv.get('low_52w', 'N/A')}",
        "",
        "### Factor Scores (z-score normalized, 50 = mean)",
    ]
    for name, score in sorted(factors.items()):
        lines.append(f"- {name}: {score:.1f}")

    if extra_context:
        lines.append("")
        lines.append("### Additional Context")
        lines.append(extra_context)

    lines.append("")
    lines.append(f"Produce a JSON AnalystReport for analyst_type={analyst_type}, ticker={ticker}.")

    return "\n".join(lines)


async def market_analyst(ticker_data: dict, llm_client: LLMClient) -> AnalystReport:
    """Market Analyst: price action, volume, momentum signals, technical patterns."""
    user_prompt = _build_analyst_user_prompt(
        "Market",
        ticker_data,
        "Focus on: price action trends, volume analysis, momentum indicators, "
        "support/resistance levels, and technical patterns. "
        "Consider RSI, MACD, moving average crossovers, and volume anomalies.",
    )
    json_text = await llm_client.chat_with_json(
        system_prompt=_ANALYST_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return await llm_client._parse_analyst_report(json_text, ticker_data["ticker"])


async def news_analyst(ticker_data: dict, llm_client: LLMClient) -> AnalystReport:
    """News Analyst: recent news/catalyst assessment."""
    user_prompt = _build_analyst_user_prompt(
        "News",
        ticker_data,
        "Focus on: recent news sentiment, upcoming catalysts (earnings, product launches), "
        "sector-level news impact, and any material events. "
        "Consider the PEAD (post-earnings announcement drift) flag if present.",
    )
    json_text = await llm_client.chat_with_json(
        system_prompt=_ANALYST_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return await llm_client._parse_analyst_report(json_text, ticker_data["ticker"])


async def fundamentals_analyst(ticker_data: dict, llm_client: LLMClient) -> AnalystReport:
    """Fundamentals Analyst: sector context, market cap, revenue growth, earnings timing."""
    market_cap = ticker_data.get("market_cap", "N/A")
    revenue_growth = ticker_data.get("revenue_growth", "N/A")

    user_prompt = _build_analyst_user_prompt(
        "Fundamentals",
        ticker_data,
        f"Focus on: sector positioning, market cap ({market_cap}), "
        f"revenue growth ({revenue_growth}), earnings timing, "
        "valuation relative to peers, and fundamental quality indicators. "
        "Consider insider buying signals and revenue acceleration.",
    )
    json_text = await llm_client.chat_with_json(
        system_prompt=_ANALYST_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return await llm_client._parse_analyst_report(json_text, ticker_data["ticker"])


async def breakout_analyst(
    ticker_data: dict,
    llm_client: LLMClient,
    case_lib: BreakoutCaseLibrary | None = None,
) -> AnalystReport:
    """Breakout Analyst: cosine similarity search against FAISS case library.

    Searches for Top 5 historical breakout patterns with similarity >= 0.85.
    When case_lib is provided and ticker has an embedding, FAISS search results
    are included in the prompt for the LLM to reference.
    """
    extra_context = [
        "Focus on: breakout pattern recognition, historical analogue cases, "
        "volatility contraction/expansion patterns (BB squeeze, ATR ratio), "
        "and volume confirmation of price breakouts.",
    ]

    embedding = ticker_data.get("embedding")
    if embedding is not None and case_lib is not None and case_lib.size > 0:
        # Perform FAISS similarity search
        matches = case_lib.search(embedding, top_k=5, min_similarity=0.85)
        if matches:
            extra_context.append("")
            extra_context.append("### FAISS Historical Breakout Matches")
            for i, match in enumerate(matches, 1):
                extra_context.append(
                    f"{i}. {match.get('ticker', '?')} — "
                    f"similarity={match['similarity']:.3f}, "
                    f"label={match.get('label', 'N/A')}, "
                    f"date={match.get('date', 'N/A')}"
                )
        else:
            extra_context.append(
                "FAISS search returned no matches above similarity threshold 0.85."
            )
    elif embedding is not None:
        extra_context.append(
            f"Embedding available ({len(embedding)}-dim) but no FAISS case library loaded."
        )
    else:
        extra_context.append("No embedding available for similarity search.")

    user_prompt = _build_analyst_user_prompt(
        "Breakout",
        ticker_data,
        "\n".join(extra_context),
    )
    json_text = await llm_client.chat_with_json(
        system_prompt=_ANALYST_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return await llm_client._parse_analyst_report(json_text, ticker_data["ticker"])


# ---------------------------------------------------------------------------
# Stage 1: Bull ∥ Bear Researchers (parallel per ticker)
# ---------------------------------------------------------------------------


_BULL_RESEARCHER_SYSTEM_PROMPT = """You are a Bullish Research Analyst. Your job is to construct a bullish investment thesis by synthesizing signals from multiple analyst reports. Always respond with ONLY valid JSON. No explanations, no markdown fences, just raw JSON.

The JSON must have these exact fields:
- ticker: string, the stock ticker
- bull_thesis: string, a detailed bullish thesis incorporating the strongest bullish signals
- bear_thesis: string, a brief acknowledgment of the main bearish risk (for balance)"""

_BEAR_RESEARCHER_SYSTEM_PROMPT = """You are a Bearish Research Analyst. Your job is to construct a bearish counter-thesis by synthesizing concerns and bearish signals from multiple analyst reports. Always respond with ONLY valid JSON. No explanations, no markdown fences, just raw JSON.

The JSON must have these exact fields:
- ticker: string, the stock ticker
- bull_thesis: string, a brief acknowledgment of the best bull case (for balance)
- bear_thesis: string, a detailed bearish thesis incorporating the strongest bearish signals and concerns"""


def _build_researcher_user_prompt(analyst_reports: list[AnalystReport], perspective: str) -> str:
    """Build the user prompt for Bull or Bear researcher."""
    if not analyst_reports:
        raise ValueError("analyst_reports must not be empty")
    ticker = analyst_reports[0].ticker
    lines = [
        f"## {perspective} Research Thesis for {ticker}",
        "",
        "### Analyst Reports Summary",
        "",
    ]

    for report in analyst_reports:
        lines.append(f"**{report.analyst_type} Analyst:**")
        lines.append(f"  Summary: {report.summary}")
        lines.append(
            f"  Bullish: {', '.join(report.bullish_signals) if report.bullish_signals else 'None'}"
        )
        lines.append(
            f"  Bearish: {', '.join(report.bearish_signals) if report.bearish_signals else 'None'}"
        )
        lines.append(f"  Concerns: {', '.join(report.concerns) if report.concerns else 'None'}")
        lines.append("")

    if perspective == "Bullish":
        lines.append(
            "Construct a BULLISH thesis. Focus primarily on the bullish signals and "
            "opportunities. Acknowledge the top 1-2 bearish risks briefly for balance. "
            "Be specific and data-driven."
        )
    else:
        lines.append(
            "Construct a BEARISH thesis. Focus primarily on the bearish signals and "
            "concerns. Acknowledge the top 1-2 bullish points briefly for balance. "
            "Be specific and data-driven."
        )

    lines.append("")
    lines.append(f"Produce a JSON BullBearOutput for ticker={ticker}.")

    return "\n".join(lines)


async def bull_researcher(
    analyst_reports: list[AnalystReport], llm_client: LLMClient
) -> BullBearOutput:
    """Bull Researcher: constructs the bullish thesis from analyst signals."""
    user_prompt = _build_researcher_user_prompt(analyst_reports, "Bullish")
    json_text = await llm_client.chat_with_json(
        system_prompt=_BULL_RESEARCHER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return await llm_client._parse_bull_bear_output(json_text, analyst_reports[0].ticker)


async def bear_researcher(
    analyst_reports: list[AnalystReport], llm_client: LLMClient
) -> BullBearOutput:
    """Bear Researcher: constructs the bearish counter-argument."""
    user_prompt = _build_researcher_user_prompt(analyst_reports, "Bearish")
    json_text = await llm_client.chat_with_json(
        system_prompt=_BEAR_RESEARCHER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return await llm_client._parse_bull_bear_output(json_text, analyst_reports[0].ticker)


# ---------------------------------------------------------------------------
# Stage 2: PM Risk Audit (per ticker)
# ---------------------------------------------------------------------------


_PM_SYSTEM_PROMPT = """You are a Portfolio Manager conducting a final risk audit on a stock screening candidate. Your job is to synthesize the bull/bear debate and quantitative factor scores into a final BreakoutAssessment. Always respond with ONLY valid JSON. No explanations, no markdown fences, just raw JSON.

The JSON must have these exact fields:
- ticker: string, the stock ticker symbol
- score_correction: number, multiplicative adjustment to coarse score, must be in [0.90, 1.05]
- risk_tags: array of strings from ["no_risk", "data_conflict", "liquidity_trap", "delisting_risk", "earnings_timing_mismatch", "catalyst_already_passed"]
- final_rating: string, one of "Strong Buy", "Buy", "Hold", "Avoid"
- breakout_probability: number, estimated breakout probability in [0, 1]
- rationale: string, short explanation (2-3 sentences) of your assessment"""


async def pm_risk_audit(
    bull_bear: BullBearOutput,
    factor_scores: dict[str, float],
    llm_client: LLMClient,
) -> BreakoutAssessment:
    """PM Risk Audit: produces BreakoutAssessment from BullBearOutput + factor scores."""
    ticker = bull_bear.ticker

    lines = [
        f"## PM Risk Audit for {ticker}",
        "",
        "### Bull Thesis",
        bull_bear.bull_thesis,
        "",
        "### Bear Thesis",
        bull_bear.bear_thesis,
        "",
        "### Factor Scores (z-score normalized, 50 = mean)",
    ]
    for name, score in sorted(factor_scores.items()):
        lines.append(f"- {name}: {score:.1f}")

    lines.extend(
        [
            "",
            "### Instructions",
            "Evaluate the risk/reward profile. Assign a score_correction in [0.90, 1.05]",
            "(1.0 = neutral, <1.0 = downgrade, >1.0 = upgrade).",
            "Identify any risk tags that apply. Determine the final rating.",
            "Estimate breakout_probability in [0, 1].",
            "Provide a concise rationale.",
            "",
            f"Produce a JSON BreakoutAssessment for ticker={ticker}.",
        ]
    )

    user_prompt = "\n".join(lines)
    json_text = await llm_client.chat_with_json(
        system_prompt=_PM_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return await llm_client._parse_breakout_assessment(json_text, ticker)


# ---------------------------------------------------------------------------
# DAG orchestrator: per-ticker pipeline
# ---------------------------------------------------------------------------


async def _process_single_ticker(
    ticker_data: dict,
    llm_client: LLMClient,
    case_lib: BreakoutCaseLibrary | None = None,
) -> BreakoutAssessment:
    """Run the full DAG pipeline for a single ticker.

    4 Analysts (parallel) → Bull∥Bear Researchers (parallel) → PM Risk Audit.
    """
    ticker = ticker_data.get("ticker", "UNKNOWN")

    try:
        # Stage 0: Check circuit breaker
        breaker_level = llm_client.check_breaker()
        if breaker_level in (BreakerLevel.L3_SAVINGS, BreakerLevel.L4_CIRCUIT):
            logger.warning(
                "Circuit breaker at %s, returning default assessment for %s",
                breaker_level.value,
                ticker,
            )
            return BreakoutAssessment(ticker=ticker)

        # Stage 1: 4 Analysts in parallel
        market_task = market_analyst(ticker_data, llm_client)
        news_task = news_analyst(ticker_data, llm_client)
        fundamentals_task = fundamentals_analyst(ticker_data, llm_client)
        breakout_task = breakout_analyst(ticker_data, llm_client, case_lib)

        analyst_reports_tuple = await asyncio.gather(
            market_task, news_task, fundamentals_task, breakout_task
        )
        analyst_reports = list(analyst_reports_tuple)

        # Stage 1: Bull ∥ Bear Researchers in parallel
        bull_task = bull_researcher(analyst_reports, llm_client)
        bear_task = bear_researcher(analyst_reports, llm_client)

        bull_output, bear_output = await asyncio.gather(bull_task, bear_task)

        # Merge theses: take bull thesis from bull researcher, bear thesis from bear researcher
        combined = BullBearOutput(
            ticker=ticker,
            bull_thesis=bull_output.bull_thesis,
            bear_thesis=bear_output.bear_thesis,
        )

        # Stage 2: PM Risk Audit
        factor_scores = ticker_data.get("factor_scores", {})
        assessment = await pm_risk_audit(combined, factor_scores, llm_client)

        return assessment

    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
            raise
        logger.error("DAG pipeline failed for %s: %s", ticker, e)
        return BreakoutAssessment(ticker=ticker)


async def run_llm_dag(
    ticker_batch: list[dict],
    llm_client: LLMClient,
    case_lib: BreakoutCaseLibrary | None = None,
) -> list[BreakoutAssessment]:
    """Run the LLM refinement DAG for a batch of tickers.

    Each ticker is processed independently in parallel. Cost circuit breaker
    levels are checked before processing and costs are flushed after.

    Args:
        ticker_batch: List of ticker data dicts with OHLCV, factor scores, etc.
        llm_client: Configured LLMClient instance.
        case_lib: Optional BreakoutCaseLibrary for FAISS similarity search.

    Returns:
        List of BreakoutAssessment, one per ticker in the batch.
    """
    if not ticker_batch:
        return []

    # Process all tickers in parallel
    tasks = [_process_single_ticker(td, llm_client, case_lib) for td in ticker_batch]
    results = await asyncio.gather(*tasks)

    # Flush costs after batch completes
    await llm_client.flush_costs()

    return list(results)
