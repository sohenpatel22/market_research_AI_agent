"""Golden evaluation dataset: schema and loader"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

GOLDEN_PATH = Path("eval/golden_dataset.json")

# filings:      one company's filing text (ground-truth source known)
Category = Literal[
    "filings",
    "multi_source",
    "mixed",
    "forecast",
    "data",
    "unanswerable",
    "out_of_scope",
    "adversarial",
]


class GroundTruthSource(BaseModel):
    """Stable pointer to the chunk(s) that answer the question (survives re-ingestion)"""

    ticker: str
    filing_type: str
    accession_number: str
    section: str | None = None
    chunk_index: int
    quote: str = Field(description="Excerpt of the source chunk, for human review")


class GoldenItem(BaseModel):
    id: str
    category: Category
    question: str
    reference_answer: str = ""
    key_facts: list[str] = Field(default_factory=list, description="Facts a good answer contains")
    source: GroundTruthSource | None = None
    expected_tickers: list[str] = Field(default_factory=list)
    expected_forecasts: list[Literal["1w", "1m"]] = Field(default_factory=list)
    should_refuse: bool = False
    expects_data: bool = False
    notes: str = ""


def load_golden(path: Path = GOLDEN_PATH, categories: set[str] | None = None) -> list[GoldenItem]:
    items = [GoldenItem(**row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    return [i for i in items if not categories or i.category in categories]


def save_golden(items: list[GoldenItem], path: Path = GOLDEN_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([i.model_dump() for i in items], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
