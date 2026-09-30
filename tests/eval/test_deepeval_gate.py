"""CI eval gate (DeepEval): runs the real agent on a small golden subset and fails the build when
quality drops below eval/thresholds.yaml.

Deselected by default (`-m "not eval"`) because it calls a paid LLM. Run it with:
    uv run pytest -m eval

Skipped automatically when the configured provider has no API key (e.g. forks without secrets).
"""

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_GRPC_LOGGING", "NO")

import pytest  # noqa: E402

from market_research_agent.config import settings  # noqa: E402
from market_research_agent.eval.checks import deterministic_checks, load_thresholds  # noqa: E402
from market_research_agent.eval.golden import load_golden  # noqa: E402

THRESHOLDS = load_thresholds()
CI_IDS = THRESHOLDS["deepeval"]["ci_items"]
ITEMS = {i.id: i for i in load_golden()}

_provider = settings.judge_provider or settings.llm_provider
pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not (getattr(settings, f"{_provider}_api_key", None) and settings.llm_provider),
        reason="No LLM API key configured; skipping the paid eval gate.",
    ),
]


@pytest.fixture(scope="module")
def deps(engine):
    from market_research_agent.eval import ci_fixture
    from market_research_agent.eval.run_eval import build_deps

    ci_fixture.load()  # idempotent: no-op on a database that already holds the full corpus
    return build_deps(None, None, settings.agent_max_retries, settings.use_reranker)


@pytest.fixture(scope="module")
def judge():
    from market_research_agent.eval.judges import deepeval_judge

    return deepeval_judge()


@pytest.mark.parametrize("item_id", CI_IDS)
def test_golden_item(item_id, deps, judge):
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    from market_research_agent.agent.service import ask

    item = ITEMS[item_id]
    answer = ask(item.question, deps=deps).answer

    failed = [k for k, v in deterministic_checks(item, answer).items() if v is False]
    assert not failed, f"{item_id}: deterministic checks failed: {failed}\n{answer.answer}"

    if item.category != "filings":
        return  # tool use / refusal behaviour is fully covered by the deterministic checks

    case = LLMTestCase(
        input=item.question,
        actual_output=answer.answer,
        expected_output=item.reference_answer,
        retrieval_context=[c.text for c in answer.retrieved_context],
    )
    cfg = THRESHOLDS["deepeval"]
    assert_test(
        case,
        [
            FaithfulnessMetric(
                threshold=cfg["faithfulness"], model=judge, include_reason=False, async_mode=False
            ),
            AnswerRelevancyMetric(
                threshold=cfg["answer_relevancy"],
                model=judge,
                include_reason=False,
                async_mode=False,
            ),
        ],
    )
