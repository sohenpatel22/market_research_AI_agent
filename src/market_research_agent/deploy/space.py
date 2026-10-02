"""Deploy the app to a Hugging Face Docker Space and smoke-test the live URL.

    python -m market_research_agent.deploy.space --space owner/name [--no-wait] [--chat-smoke]

Reads HF_TOKEN from the environment and the runtime secrets (DATABASE_URL, LLM and Langfuse keys)
from the environment too, so GitHub Actions secrets stay the single source of truth: they are
copied into the *Space's* secrets, never into the uploaded files or the image. Prints no secrets.

What gets uploaded is a minimal build context (Dockerfile, lockfile, source, trained models) plus
a README whose front matter tells Hugging Face this is a Docker Space listening on port 7860.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import httpx

FRONT_MATTER = """---
title: Market Research Agent
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Cited research agent over SEC filings, prices and forecasts
---

"""

# Everything the Dockerfile needs to build the runtime image.
BUNDLE_FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock"]
BUNDLE_DIRS = ["src", "artifacts"]
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "desktop.ini", ".pytest_cache")

# Copied from the environment into the Space's *secrets* (private, hidden from the UI).
SECRET_NAMES = [
    "DATABASE_URL",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
]

SMOKE_QUESTION = "What does NVIDIA say about export controls on its data center products?"

TERMINAL_FAILURES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def build_bundle(repo_root: Path, out_dir: Path) -> Path:
    """Assemble the minimal Space repo contents in `out_dir` (recreated from scratch)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in BUNDLE_FILES:
        shutil.copy2(repo_root / name, out_dir / name)
    for name in BUNDLE_DIRS:
        shutil.copytree(repo_root / name, out_dir / name, ignore=IGNORED)
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    (out_dir / "README.md").write_text(FRONT_MATTER + readme, encoding="utf-8", newline="\n")
    return out_dir


def space_secrets(env: dict[str, str]) -> dict[str, str]:
    """The non-empty runtime secrets present in `env`."""
    return {k: env[k] for k in SECRET_NAMES if env.get(k)}


def space_variables(env: dict[str, str], git_sha: str) -> dict[str, str]:
    """Public, non-sensitive settings. GIT_SHA is also available to the Docker build as an ARG."""
    return {
        "GIT_SHA": git_sha,
        "LLM_PROVIDER": env.get("LLM_PROVIDER", "deepseek"),
        "LANGFUSE_ENVIRONMENT": "production",
        "RATE_LIMIT_PER_MINUTE": env.get("RATE_LIMIT_PER_MINUTE", "20"),
    }


def sync_files(api, space_id: str, bundle: Path, message: str) -> None:
    """Make the Space repo exactly equal to the bundle in one commit (keeping .gitattributes)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    local = {p.relative_to(bundle).as_posix(): p for p in bundle.rglob("*") if p.is_file()}
    remote = set(api.list_repo_files(space_id, repo_type="space"))
    operations = [
        CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in local.items()
    ]
    operations += [
        CommitOperationDelete(path_in_repo=f)
        for f in sorted(remote - set(local))
        if f != ".gitattributes"
    ]
    api.create_commit(
        repo_id=space_id, repo_type="space", operations=operations, commit_message=message
    )


def wait_until_running(api, space_id: str, timeout_s: int, poll_s: int = 15) -> str:
    """Block until the Space is RUNNING; raise with a hint on failure or timeout."""
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        stage = api.get_space_runtime(space_id).stage
        if stage != last:
            print(f"  space stage: {stage}", flush=True)
            last = stage
        if stage == "RUNNING":
            return stage
        if stage in TERMINAL_FAILURES:
            raise RuntimeError(
                f"Space entered {stage}; see its build/container logs on Hugging Face"
            )
        time.sleep(poll_s)
    raise TimeoutError(f"Space did not reach RUNNING within {timeout_s}s (last stage: {last})")


def smoke_test(base_url: str, chat: bool = False, attempts: int = 12) -> dict:
    """Check /health and the UI on the live URL. Returns the health payload."""
    health = None
    for _ in range(attempts):  # the app may still be starting right after RUNNING
        try:
            r = httpx.get(f"{base_url}/health", timeout=30)
            if r.status_code == 200:
                health = r.json()
                break
        except httpx.HTTPError:
            pass
        time.sleep(10)
    if health is None:
        raise RuntimeError("/health did not return 200")
    print(f"  /health: status={health['status']}")
    for name, comp in health["components"].items():
        print(f"    {name}: {'ok' if comp['ok'] else 'NOT OK'} ({comp['detail']})")
    if health["status"] != "ok":
        raise RuntimeError("the app is up but degraded (see components above)")

    page = httpx.get(f"{base_url}/", timeout=30)
    if page.status_code != 200 or "Market Research Agent" not in page.text:
        raise RuntimeError("the UI did not load")
    print("  UI: loads")

    if chat:
        r = httpx.post(
            f"{base_url}/chat",
            json={"question": SMOKE_QUESTION},
            timeout=180,
        )
        r.raise_for_status()
        body = r.json()["answer"]
        if not body["quality_passed"] or not body["sources"]:
            raise RuntimeError("chat smoke test returned an unverified or uncited answer")
        print(f"  /chat: passed quality check with {len(body['sources'])} cited sources")
    return health


def deploy(
    space_id: str,
    repo_root: Path,
    git_sha: str,
    wait: bool = True,
    chat_smoke: bool = False,
    timeout_s: int = 1800,
) -> str:
    from huggingface_hub import HfApi

    env = dict(os.environ)
    token = env.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    api = HfApi(token=token)

    secrets = space_secrets(env)
    missing = [n for n in ("DATABASE_URL", "DEEPSEEK_API_KEY") if n not in secrets]
    if missing:
        raise SystemExit(f"Missing required secrets: {', '.join(missing)}")

    print(f"Deploying {git_sha[:8]} to {space_id}")
    # Secrets/variables first: uploading files is what triggers the (single) build.
    for key, value in secrets.items():
        api.add_space_secret(space_id, key, value)
    print(f"  set {len(secrets)} Space secrets")
    for key, value in space_variables(env, git_sha).items():
        api.add_space_variable(space_id, key, value)

    bundle = build_bundle(repo_root, repo_root / ".space_bundle")
    try:
        sync_files(api, space_id, bundle, f"Deploy {git_sha[:8]}")
    finally:
        shutil.rmtree(bundle, ignore_errors=True)
    print("  uploaded build context")

    host = api.space_info(space_id).host or f"https://{space_id.replace('/', '-').lower()}.hf.space"
    if wait:
        wait_until_running(api, space_id, timeout_s)
        smoke_test(host, chat=chat_smoke)
    print(f"Live: {host}")
    return host


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True, help="Space id, owner/name")
    parser.add_argument("--no-wait", action="store_true", help="return right after uploading")
    parser.add_argument("--chat-smoke", action="store_true", help="also ask one real question")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds to wait for the build")
    args = parser.parse_args()
    sha = os.environ.get("GIT_SHA") or os.environ.get("GITHUB_SHA") or "local"
    deploy(
        args.space,
        Path.cwd(),
        sha,
        wait=not args.no_wait,
        chat_smoke=args.chat_smoke,
        timeout_s=args.timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
