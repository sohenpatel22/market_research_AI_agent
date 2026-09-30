"""Deterministic (no-LLM) checks on agent answers, plus threshold loading."""

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
    """Ways an answer might legitimately write a number (1,234,000,000 / 1.234 billion / ...)."""
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
    """True if the fact's headline numbers (ignoring dates/years) appear in the answer."""
    without_dates = re.sub(r"\d{4}-\d{2}-\d{2}", " ", key_fact)
    nums = [n for n in numbers_in(without_dates) if not (1900 <= abs(n) <= 2100 and n == int(n))]
    if not nums:
        return True
    haystack = answer.replace("$", "")
    return any(any(v in haystack for v in _number_variants(n)) for n in nums)


def deterministic_checks(item: GoldenItem, answer: AgentAnswer) -> dict[str, bool | None]:
    """Per-item pass/fail for each applicable check (None = not applicable)."""
    checks: dict[str, bool | None] = {
        "refusal_correct": answer.refused == item.should_refuse,
        "forecast_tool": None,
        "data_tool": None,
        "data_fact": None,
        "cited": None,
    }
    if item.category == "forecast":
        got = {f.horizon for f in answer.forecasts}
        checks["forecast_tool"] = set(item.expected_forecasts) <= got
    elif item.category == "data":
        checks["data_tool"] = len(answer.data) >= 1
        checks["data_fact"] = all(fact_numbers_present(f, answer.answer) for f in item.key_facts)
    elif item.category == "filings":
        checks["cited"] = len(answer.sources) >= 1
    return checks


def rate(results: list[dict[str, bool | None]], key: str) -> float | None:
    vals = [r[key] for r in results if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None
