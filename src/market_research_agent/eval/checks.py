"""Deterministic (no-LLM) checks on agent answers, plus threshold loading"""

import re
from contextlib import suppress
from pathlib import Path

import yaml

from market_research_agent.agent.schemas import AgentAnswer
from market_research_agent.eval.golden import GoldenItem

THRESHOLDS_PATH = Path("eval/thresholds.yaml")


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _number_variants(value: float) -> set[str]:
    out = {f"{value:,.0f}", f"{value:.0f}", f"{value:,.2f}", f"{value:.2f}", f"{value:.1f}"}
    for divisor in (1e9, 1e6):
        if abs(value) >= divisor:
            scaled = value / divisor
            for digits in (0, 1, 2, 3):
                out.add(f"{scaled:,.{digits}f}")
    return out


_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[float]:
    vals = []
    for raw in _NUMBER.findall(text):
        with suppress(ValueError):
            vals.append(float(raw.replace(",", "").rstrip(".")))
    return vals


def fact_numbers_present(key_fact: str, answer: str) -> bool:
    """True if the fact's headline numbers (ignoring dates/years) appear in the answer"""
    without_dates = re.sub(r"\d{4}-\d{2}-\d{2}", " ", key_fact)
    nums = [n for n in numbers_in(without_dates) if not (1900 <= abs(n) <= 2100 and n == int(n))]
    if not nums:
        return True
    haystack = answer.replace("$", "")
    return any(any(v in haystack for v in _number_variants(n)) for n in nums)


# An honest "I can't tell from the sources" in an answer to an unanswerable question.
ABSTAIN_RE = re.compile(
    r"\b(does not|doesn't|do not|don't|not (?:contain|include|provide|mention|specify|available|"
    r"address|cover|disclose|found)|no (?:information|mention|data|details|specific|record)|"
    r"unable to|cannot|can't|couldn't|isn't|aren't|outside|only covers?|not covered|"
    r"insufficient|lack|missing)\b",
    re.IGNORECASE,
)


def deterministic_checks(item: GoldenItem, answer: AgentAnswer) -> dict[str, bool | None]:
    """Per-item pass/fail for each applicable check (None = not applicable)"""
    checks: dict[str, bool | None] = {
        # For unanswerable questions either a refusal or an abstention is acceptable.
        "refusal_correct": None
        if item.category == "unanswerable"
        else answer.refused == item.should_refuse,
        "forecast_tool": None,
        "data_tool": None,
        "data_fact": None,
        "cited": None,
        "multi_source": None,
        "abstained": None,
    }
    if item.expected_forecasts:
        got = {(f.ticker, f.horizon) for f in answer.forecasts}
        want = {(t, h) for t in item.expected_tickers for h in item.expected_forecasts}
        checks["forecast_tool"] = want <= got
    if item.category == "data":
        checks["data_tool"] = len(answer.data) >= 1
        checks["data_fact"] = all(fact_numbers_present(f, answer.answer) for f in item.key_facts)
    elif item.category == "mixed" and item.expects_data:
        checks["data_tool"] = len(answer.data) >= 1
    if item.category in ("filings", "multi_source", "mixed"):
        checks["cited"] = len(answer.sources) >= 1
    if item.category == "multi_source":
        cited = {s.ticker for s in answer.sources}
        checks["multi_source"] = set(item.expected_tickers) <= cited
    if item.category == "unanswerable":
        checks["abstained"] = answer.refused or bool(ABSTAIN_RE.search(answer.answer))
    return checks


def rate(results: list[dict[str, bool | None]], key: str) -> float | None:
    vals = [r[key] for r in results if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None
