import json

from fastapi.testclient import TestClient

from market_research_agent.agent.schemas import DraftAnswer
from market_research_agent.api.main import create_app
from market_research_agent.api.ratelimit import SlidingWindowLimiter
from market_research_agent.api.runtime import Runtime
from market_research_agent.api.schemas import ChatRequest, ForecastRequest
from tests.test_agent import BAD, FILINGS_ROUTE, GOOD, FakeLLM, make_deps


def make_client(judge=None, limiter=None, **kw) -> TestClient:
    llm = FakeLLM(
        RouteDecision=[FILINGS_ROUTE],
        DraftAnswer=[DraftAnswer(answer="Supply chain risk [S1].", cited_source_ids=[1])],
        **kw,
    )
    deps, _ = make_deps(llm, judge or FakeLLM(GradeResult=[GOOD]))
    app = create_app(Runtime(deps), limiter or SlidingWindowLimiter(0), with_ui=False)
    return TestClient(app)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_chat_returns_a_validated_answer():
    r = make_client().post("/chat", json={"question": "What risks does Apple face?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]["quality_passed"] is True
    assert body["answer"]["sources"][0]["source_id"] == 1
    assert body["session_id"] and body["latency_s"] >= 0
    assert r.headers["x-request-id"]


def test_chat_validation():
    client = make_client()
    assert client.post("/chat", json={"question": "  "}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422
    bad_session = {"question": "valid question", "session_id": "no spaces allowed!"}
    assert client.post("/chat", json=bad_session).status_code == 422


def test_chat_stream_emits_steps_then_final():
    r = make_client().post("/chat/stream", json={"question": "What risks does Apple face?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [n for e, d in events if e == "step" for n in [d["node"]]] == [
        "route",
        "gather",
        "generate",
        "grade",
    ]
    kind, payload = events[-1]
    assert kind == "final" and payload["answer"]["answer"].startswith("Supply chain")


def test_chat_stream_with_a_retry_reports_it():
    client = make_client(
        judge=FakeLLM(GradeResult=[BAD, GOOD]),
        RewriteResult=[
            __import__("market_research_agent.agent.schemas", fromlist=["x"]).RewriteResult(
                search_query="q2"
            )
        ],
    )
    events = parse_sse(client.post("/chat/stream", json={"question": "risks?"}).text)
    nodes = [d["node"] for e, d in events if e == "step"]
    assert "rewrite" in nodes and events[-1][1]["answer"]["retries"] == 1


def test_stream_errors_are_sanitised():
    class Boom(FakeLLM):
        def with_structured_output(self, schema, **kw):
            raise RuntimeError("secret internal detail sk-123")

    deps, _ = make_deps(Boom(), FakeLLM(GradeResult=[GOOD]))
    client = TestClient(create_app(Runtime(deps), SlidingWindowLimiter(0), with_ui=False))
    events = parse_sse(client.post("/chat/stream", json={"question": "hello there"}).text)
    assert events[-1][0] == "error"
    assert "secret" not in json.dumps(events) and "sk-123" not in json.dumps(events)


def test_missing_llm_key_is_a_503_not_a_500(monkeypatch):
    from market_research_agent.llm.factory import MissingAPIKeyError

    def no_key():
        raise MissingAPIKeyError("DEEPSEEK_API_KEY is not set")

    monkeypatch.setattr("market_research_agent.api.runtime.default_dependencies", no_key)
    client = TestClient(create_app(Runtime(), SlidingWindowLimiter(0), with_ui=False))
    r = client.post("/chat", json={"question": "hello there"})
    assert r.status_code == 503 and "not configured" in r.json()["detail"]
    assert "KEY" not in r.text
    assert client.post("/chat/stream", json={"question": "hello there"}).status_code == 503


def test_forecast_endpoint(monkeypatch):
    from market_research_agent.models.forecast import ForecastError, ForecastResult

    fake = ForecastResult(
        ticker="AAPL", horizon="1w", as_of="2026-01-01", model_version="t", predicted_vol=0.2
    )

    def fake_forecast(ticker, horizon):
        if ticker == "ZZZ":
            raise ForecastError("ZZZ was not in the training set")
        return fake

    monkeypatch.setattr("market_research_agent.api.main.forecast", fake_forecast)
    client = make_client()
    ok = client.post("/forecast", json={"ticker": "aapl", "horizon": "1w"})
    assert ok.status_code == 200 and ok.json()["predicted_vol"] == 0.2
    assert client.post("/forecast", json={"ticker": "ZZZ"}).status_code == 422
    assert client.post("/forecast", json={"ticker": "AAPL", "horizon": "1y"}).status_code == 422
    assert client.post("/forecast", json={"ticker": "A;DROP"}).status_code == 422


def test_health_reports_components(engine):
    body = make_client().get("/health").json()
    assert set(body["components"]) == {"database", "forecast_models", "llm", "langfuse"}
    assert body["status"] in {"ok", "degraded"} and body["version"]
    # The database is reachable and the schema exists; whether it is *populated* differs
    # between a dev machine (full corpus) and CI (empty), so only the wording is checked.
    assert "filing chunks" in body["components"]["database"]["detail"]


def test_tickers_endpoint():
    body = make_client().get("/tickers").json()
    assert "AAPL" in body["tickers"]


def test_rate_limit_returns_429_with_retry_after():
    client = make_client(limiter=SlidingWindowLimiter(2, 60))
    for _ in range(2):
        assert client.post("/chat", json={"question": "valid question"}).status_code == 200
    r = client.post("/chat", json={"question": "valid question"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    # health is never limited
    assert client.get("/health").status_code == 200


def test_limiter_window_expires():
    now = [0.0]
    lim = SlidingWindowLimiter(1, 10, clock=lambda: now[0])
    assert lim.check("a") is None
    assert lim.check("a") is not None and lim.check("b") is None
    now[0] = 11
    assert lim.check("a") is None


def test_request_models_normalise_input():
    assert ForecastRequest(ticker=" nvda ").ticker == "NVDA"
    assert ChatRequest(question="  hello  ").question == "hello"


def test_health_reports_missing_tables_and_empty_corpus_as_degraded(monkeypatch):
    from market_research_agent.api import health

    class FakeConn:
        def __init__(self, fail_on):
            self.fail_on = fail_on

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, stmt):
            sql = str(stmt)
            if self.fail_on and self.fail_on in sql:
                raise RuntimeError("relation does not exist")

            class R:
                @staticmethod
                def scalar_one():
                    return 0

            return R()

    class FakeEngine:
        def __init__(self, fail_on=None):
            self.fail_on = fail_on

        def connect(self):
            return FakeConn(self.fail_on)

    monkeypatch.setattr(health, "get_engine", lambda: FakeEngine(fail_on="documents"))
    missing = health._database()
    assert not missing.ok and "tables are missing" in missing.detail
    monkeypatch.setattr(health, "get_engine", lambda: FakeEngine())
    empty = health._database()
    assert not empty.ok and "0 filing chunks" in empty.detail
