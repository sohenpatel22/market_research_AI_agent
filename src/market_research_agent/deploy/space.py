"""Deploy the app to a Hugging Face Space (Gradio or Docker SDK) and smoke-test the live URL"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

FRONT_MATTER = """---
title: Market Research Agent
emoji: "\\U0001F4C8"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Cited research agent over SEC filings, prices and forecasts
---

"""

GRADIO_FRONT_MATTER = """---
title: Market Research Agent
emoji: "\\U0001F4C8"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "{gradio_version}"
python_version: "3.12.12"
app_file: app.py
pinned: false
short_description: Cited research agent over SEC filings, prices and forecasts
---

"""

# Space entry point: serves the FastAPI app (API plus the mounted Gradio UI) on the Space port.
APP_FILE = """import os
import sys

sys.path.insert(0, "src")
# The Space enables Gradio's Node SSR proxy, which would take the port uvicorn needs.
os.environ["GRADIO_SSR_MODE"] = "False"

import uvicorn  # noqa: E402

from market_research_agent.api.main import create_app  # noqa: E402

try:
    import spaces

    # ZeroGPU Spaces refuse to start without a GPU-decorated function; the app itself runs on CPU.
    @spaces.GPU
    def _zerogpu_probe() -> int:
        return 0

except ImportError:
    pass

uvicorn.run(create_app(), host="0.0.0.0", port=7860)  # noqa: S104
"""

# ZeroGPU accepts only specific torch releases; this is the newest that pyproject.toml allows.
ZEROGPU_TORCH = "torch==2.13.0"

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


def locked_version(repo_root: Path, package: str) -> str:
    """Version of `package` pinned in uv.lock"""
    lines = (repo_root / "uv.lock").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line == f'name = "{package}"':
            return lines[i + 1].split('"')[1]
    raise SystemExit(f"{package} is not in uv.lock")


def export_requirements(repo_root: Path) -> str:
    """requirements.txt for a Gradio Space, pinned from the lockfile"""
    out = subprocess.run(
        ["uv", "export", "--no-dev", "--no-emit-project", "--no-hashes", "--frozen"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pins = [ln for ln in out.splitlines() if ln and not ln.startswith(("#", " "))]
    # Gradio on the Space caps pydantic below the locked version, so leave it to the resolver.
    pins = [ln for ln in pins if not ln.startswith(("pydantic==", "pydantic-core=="))]
    pins = [ZEROGPU_TORCH if ln.startswith("torch==") else ln for ln in pins]
    return "\n".join(dict.fromkeys(pins)) + "\n"


def build_bundle(repo_root: Path, out_dir: Path, sdk: str = "gradio") -> Path:
    """Assemble the minimal Space repo contents in `out_dir` (recreated from scratch)"""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in BUNDLE_DIRS:
        shutil.copytree(repo_root / name, out_dir / name, ignore=IGNORED)
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    if sdk == "docker":
        for name in BUNDLE_FILES:
            shutil.copy2(repo_root / name, out_dir / name)
        front = FRONT_MATTER
    else:
        (out_dir / "app.py").write_text(APP_FILE, encoding="utf-8", newline="\n")
        (out_dir / "requirements.txt").write_text(
            export_requirements(repo_root), encoding="utf-8", newline="\n"
        )
        front = GRADIO_FRONT_MATTER.format(gradio_version=locked_version(repo_root, "gradio"))
    (out_dir / "README.md").write_text(front + readme, encoding="utf-8", newline="\n")
    return out_dir


def space_secrets(env: dict[str, str]) -> dict[str, str]:
    """The non-empty runtime secrets present in `env`"""
    return {k: env[k] for k in SECRET_NAMES if env.get(k)}


def space_variables(env: dict[str, str], git_sha: str) -> dict[str, str]:
    """Public, non-sensitive settings"""
    return {
        "GIT_SHA": git_sha,
        "LLM_PROVIDER": env.get("LLM_PROVIDER", "deepseek"),
        "LANGFUSE_ENVIRONMENT": "production",
        "RATE_LIMIT_PER_MINUTE": env.get("RATE_LIMIT_PER_MINUTE", "20"),
    }


def sync_files(api, space_id: str, bundle: Path, message: str) -> None:
    """Make the Space repo exactly equal to the bundle in one commit (keeping .gitattributes)"""
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


def ensure_space(api, space_id: str, sdk: str = "gradio") -> None:
    """Create the Space on free CPU hardware if it does not exist yet"""
    if api.repo_exists(space_id, repo_type="space"):
        return
    api.create_repo(
        space_id,
        repo_type="space",
        space_sdk=sdk,
        space_hardware="cpu-basic",
        exist_ok=True,
    )


def ensure_cpu_hardware(api, space_id: str, keep_existing: bool = False) -> str:
    """Make sure the Space runs on free CPU hardware"""
    runtime = api.get_space_runtime(space_id)
    requested = str(getattr(runtime, "requested_hardware", None) or "")
    if requested.startswith("cpu"):
        return requested
    try:
        api.request_space_hardware(space_id, "cpu-basic")
    except Exception as exc:  # noqa: BLE001
        if keep_existing:
            print(f"  keeping '{requested or 'unset'}' hardware ({exc})")
            return requested
        raise RuntimeError(
            f"Space {space_id} runs on '{requested or 'unset'}' hardware and could not be moved to "
            "CPU hardware on this account plan. Delete the Space "
            "or point --space at a new name; the deployer creates it on cpu-basic."
        ) from exc
    print(f"  hardware was '{requested or 'unset'}'; switched to cpu-basic (free)")
    return "cpu-basic"


def print_logs(token: str, space_id: str, kind: str, max_lines: int = 60) -> None:
    """Print the tail of the Space's build or container log (server-sent events)"""
    lines: list[str] = []
    try:
        with httpx.stream(
            "GET",
            f"https://huggingface.co/api/spaces/{space_id}/logs/{kind}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(20, read=10),
        ) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    lines.append(line[5:].strip())
    except httpx.HTTPError:
        pass
    print(f"--- {kind} log (last {max_lines} lines) ---")
    print("\n".join(lines[-max_lines:]))


def wait_until_running(api, space_id: str, timeout_s: int, poll_s: int = 15) -> str:
    """Block until the Space is RUNNING; raise with a hint on failure or timeout"""
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
            token = getattr(api, "token", None)
            if token:
                print_logs(token, space_id, "build" if stage == "BUILD_ERROR" else "run")
            raise RuntimeError(
                f"Space entered {stage}; see its build/container logs on Hugging Face"
            )
        time.sleep(poll_s)
    raise TimeoutError(f"Space did not reach RUNNING within {timeout_s}s (last stage: {last})")


def smoke_test(base_url: str, chat: bool = False, attempts: int = 12) -> dict:
    """Check /health and the UI on the live URL"""
    health = None
    for _ in range(attempts):
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
    sdk: str = "gradio",
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
    ensure_space(api, space_id, sdk)
    ensure_cpu_hardware(api, space_id, keep_existing=sdk == "gradio")
    # Secrets/variables first: uploading files is what triggers the (single) build.
    for key, value in secrets.items():
        api.add_space_secret(space_id, key, value)
    print(f"  set {len(secrets)} Space secrets")
    for key, value in space_variables(env, git_sha).items():
        api.add_space_variable(space_id, key, value)

    bundle = build_bundle(repo_root, repo_root / ".space_bundle", sdk)
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
    parser.add_argument("--sdk", choices=["gradio", "docker"], default="gradio")
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
        sdk=args.sdk,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
