"""Log offline experiments (model training runs, evals) to Langfuse as traces with scores.

MLflow remains the system of record for model training; this puts the headline results next to
the LLM traces so one dashboard shows both. No-op when Langfuse is not configured.

Usage:
    uv run python -m market_research_agent.observability.experiments        # log latest bundle
"""

import logging

from market_research_agent.models import registry
from market_research_agent.observability import langfuse as lf

logger = logging.getLogger(__name__)


def log_training_run(meta: dict) -> str | None:
    """One trace per trained bundle; each model's test metrics and the classifier's become scores.
    Returns the trace URL (None if Langfuse is off)."""
    client = lf.get_client()
    if client is None:
        return None
    try:
        version = meta["version"]
        with client.start_as_current_observation(
            name="model-training",
            as_type="span",
            input={"version": version, "tickers": meta["tickers"], "test_rows": meta["test_rows"]},
            metadata={
                "git_sha": lf.git_sha(),
                "classifier": meta["classifier"],
                "vol_test_metrics": meta["vol_test_metrics"],
                "dm_vs_har": meta["dm_vs_har"],
                "classifier_test_metrics": meta["classifier_test_metrics"],
            },
        ) as span:
            for model, metrics in meta["vol_test_metrics"].items():
                for metric, value in metrics.items():
                    span.score_trace(name=f"vol.{model}.{metric}", value=float(value))
            for name, res in meta["dm_vs_har"].items():
                span.score_trace(name=f"vol.{name}.dm_p_vs_har", value=float(res["p_value"]))
            clf = meta["classifier_test_metrics"]
            for key in ("roc_auc", "f1", "accuracy", "base_rate"):
                span.score_trace(name=f"direction.{key}", value=float(clf[key]))
            span.update(
                output={
                    "best_vol_model": min(
                        meta["vol_test_metrics"], key=lambda m: meta["vol_test_metrics"][m]["qlike"]
                    )
                }
            )
            trace_id = client.get_current_trace_id()
        lf.flush()
        return client.get_trace_url(trace_id=trace_id)
    except Exception:  # noqa: BLE001 - observability must never break training
        logger.warning("Failed to log training run to Langfuse", exc_info=True)
        return None


def main() -> None:
    bundle = registry.load_bundle()
    url = log_training_run(bundle["meta"])
    print(url or "Langfuse is not configured (set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY).")


if __name__ == "__main__":
    main()
