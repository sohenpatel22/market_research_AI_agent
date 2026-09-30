import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from market_research_agent import observability
from market_research_agent.agent import prompts
from market_research_agent.agent.schemas import DraftAnswer, GradeResult
from market_research_agent.agent.service import ask
from market_research_agent.config import Settings
from market_research_agent.llm.cache import SQLiteLLMCache
from market_research_agent.observability import langfuse as lf
from tests.test_agent import FILINGS_ROUTE, FakeLLM, make_deps


def test_disabled_without_keys(monkeypatch):
    monkeypatch.setattr(lf.settings, "langfuse_public_key", None)
    lf.get_client.cache_clear()
    assert not lf.is_enabled()
    assert lf.get_client() is None and lf.get_handler() is None
    assert lf.post_scores("abc", {"quality_passed": True}) is None
    with lf.trace_context(session_id="s"):  # must be a harmless no-op
        pass
    lf.flush()
    lf.shutdown()


def test_enabled_requires_both_keys():
    def cfg(**kw):
        return Settings(database_url="x", _env_file=None, **kw)

    assert not lf.is_enabled(cfg(langfuse_public_key="pk"))
    assert lf.is_enabled(cfg(langfuse_public_key="pk", langfuse_secret_key="sk"))


def test_template_conversion_roundtrip():
    for name, text in prompts.PROMPTS.items():
        assert lf.from_langfuse_template(lf.to_langfuse_template(text)) == text, name
    assert lf.to_langfuse_template("Hi {who}, {n}") == "Hi {{who}}, {{n}}"


def test_prompts_use_local_text_by_default():
    assert prompts.get("grade_system") == prompts.GRADE_SYSTEM


class FakePrompt:
    def __init__(self, text):
        self.prompt = text


class FakeClient:
    def __init__(self, remote=None, fail=False):
        self.remote, self.fail, self.scores = remote, fail, []

    def get_prompt(self, name, **kw):
        if self.fail:
            raise RuntimeError("langfuse down")
        return FakePrompt(self.remote)

    def create_score(self, **kw):
        self.scores.append(kw)

    def get_trace_url(self, trace_id=None):
        return f"https://lf.example/trace/{trace_id}"


@pytest.fixture
def fake_client(monkeypatch):
    def install(client):
        monkeypatch.setattr(lf, "get_client", lambda: client)
        monkeypatch.setattr(lf.settings, "langfuse_prompts", True)
        return client

    return install


def test_registry_prompt_used_when_variables_match(fake_client):
    fake_client(FakeClient(remote="Improved: {{question}} / {{answer}} / {{context}}"))
    text = lf.load_prompt("grade_user", prompts.GRADE_USER)
    assert text.startswith("Improved:") and "{question}" in text


def test_registry_prompt_rejected_when_variables_differ_or_error(fake_client):
    fake_client(FakeClient(remote="Only {{question}}"))
    assert lf.load_prompt("grade_user", prompts.GRADE_USER) == prompts.GRADE_USER
    fake_client(FakeClient(fail=True))
    assert lf.load_prompt("grade_user", prompts.GRADE_USER) == prompts.GRADE_USER


def test_scores_are_posted_with_types(fake_client):
    client = fake_client(FakeClient())
    url = lf.post_scores("t1", {"quality_passed": True, "grade_score": 0.8, "skipped": None})
    assert url == "https://lf.example/trace/t1"
    by_name = {s["name"]: s for s in client.scores}
    assert by_name["quality_passed"]["data_type"] == "BOOLEAN"
    assert (
        by_name["grade_score"]["data_type"] == "NUMERIC" and by_name["grade_score"]["value"] == 0.8
    )
    assert "skipped" not in by_name


def test_ask_works_without_langfuse():
    llm = FakeLLM(RouteDecision=[FILINGS_ROUTE], DraftAnswer=[DraftAnswer(answer="a")])
    deps, _ = make_deps(
        llm,
        FakeLLM(GradeResult=[GradeResult(grounded=True, relevant=True, score=0.9, feedback="")]),
    )
    out = ask("q", deps=deps)
    assert out.answer.quality_passed and out.answer.grade_score == 0.9
    assert out.trace_id is None and out.trace_url is None


def test_llm_cache_roundtrip(tmp_path):
    cache = SQLiteLLMCache(str(tmp_path / "c.sqlite"))
    gen = [ChatGeneration(message=AIMessage(content="hello"))]
    assert cache.lookup("p", "llm") is None
    cache.update("p", "llm", gen)
    hit = cache.lookup("p", "llm")
    assert hit[0].message.content == "hello"
    assert cache.lookup("p", "other-llm") is None
    cache.clear()
    assert cache.lookup("p", "llm") is None


def test_observability_exports():
    assert callable(observability.get_handler) and callable(observability.trace_context)
