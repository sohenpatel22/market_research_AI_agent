from pathlib import Path

import pytest

from market_research_agent.deploy import space

REPO = Path(__file__).resolve().parent.parent


def test_front_matter_declares_a_docker_space_on_7860():
    assert space.FRONT_MATTER.startswith("---\n") and "sdk: docker" in space.FRONT_MATTER
    assert "app_port: 7860" in space.FRONT_MATTER


def test_bundle_contains_only_what_the_dockerfile_needs(tmp_path):
    out = space.build_bundle(REPO, tmp_path / "bundle")
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    for required in ("Dockerfile", "pyproject.toml", "uv.lock", "README.md"):
        assert required in names
    assert any(n.startswith("src/market_research_agent/") for n in names)
    assert any(n.startswith("artifacts/models/") for n in names)
    # nothing local, bulky or sensitive travels with the deploy
    forbidden = (".env", "data/", "tests/", ".git/", ".venv", "mlflow.db", "eval/results")
    assert not [n for n in names if n == ".env" or n.startswith(forbidden[1:])]
    assert not [n for n in names if n.endswith((".pyc", "desktop.ini"))]
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(space.FRONT_MATTER)
    assert "# Market Research Agent" in readme  # original README follows the front matter


def test_bundle_is_rebuilt_from_scratch(tmp_path):
    stale = tmp_path / "bundle" / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("old")
    out = space.build_bundle(REPO, tmp_path / "bundle")
    assert not (out / "stale.txt").exists()


def test_secrets_and_variables_selection():
    env = {
        "DATABASE_URL": "postgresql://x",
        "DEEPSEEK_API_KEY": "k",
        "OPENAI_API_KEY": "",
        "LANGFUSE_PUBLIC_KEY": "pk",
        "HF_TOKEN": "must-never-be-copied",
        "PATH": "/usr/bin",
    }
    secrets = space.space_secrets(env)
    assert secrets == {
        "DATABASE_URL": "postgresql://x",
        "DEEPSEEK_API_KEY": "k",
        "LANGFUSE_PUBLIC_KEY": "pk",
    }
    variables = space.space_variables(env, "abc123")
    assert variables["GIT_SHA"] == "abc123" and variables["LANGFUSE_ENVIRONMENT"] == "production"
    assert not any("KEY" in k or "TOKEN" in k or "URL" in k for k in variables)


class FakeApi:
    def __init__(self, stages):
        self.stages = list(stages)

    def get_space_runtime(self, space_id):
        stage = self.stages.pop(0) if len(self.stages) > 1 else self.stages[0]
        return type("R", (), {"stage": stage})()


def test_wait_until_running_follows_the_stage_machine():
    api = FakeApi(["BUILDING", "APP_STARTING", "RUNNING"])
    assert space.wait_until_running(api, "o/n", timeout_s=30, poll_s=0) == "RUNNING"


def test_wait_until_running_fails_fast_on_build_errors_and_times_out():
    with pytest.raises(RuntimeError, match="BUILD_ERROR"):
        space.wait_until_running(FakeApi(["BUILDING", "BUILD_ERROR"]), "o/n", 30, poll_s=0)
    with pytest.raises(TimeoutError):
        space.wait_until_running(FakeApi(["BUILDING"]), "o/n", timeout_s=0, poll_s=0)


def test_smoke_test_requires_an_ok_health_and_the_ui(monkeypatch):
    class Resp:
        def __init__(self, status=200, payload=None, text=""):
            self.status_code, self._payload, self.text = status, payload, text

        def json(self):
            return self._payload

    ok = {"status": "ok", "components": {"database": {"ok": True, "detail": "d"}}}
    degraded = {"status": "degraded", "components": {"database": {"ok": False, "detail": "empty"}}}

    def serve(health):
        def get(url, timeout):
            if url.endswith("/health"):
                return Resp(200, health)
            return Resp(200, text="<title>Market Research Agent</title>")

        return get

    monkeypatch.setattr(space.httpx, "get", serve(ok))
    assert space.smoke_test("https://x.hf.space", attempts=1)["status"] == "ok"
    monkeypatch.setattr(space.httpx, "get", serve(degraded))
    with pytest.raises(RuntimeError, match="degraded"):
        space.smoke_test("https://x.hf.space", attempts=1)
