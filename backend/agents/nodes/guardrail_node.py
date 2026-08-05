import logging
from typing import Any


from langfuse import get_client, observe


from agents.state import SynapticState
from guardrails.classifier import GuardrailVerdict, guardrail_classifier

logger = logging.getLogger(__name__)


@observe(name="guardrail_node")
async def guardrail_node(state: SynapticState) -> dict[str, Any]:
    callbacks = state.get("langfuse_callbacks", [])

    try:
        verdict = await guardrail_classifier.check_input(
            state["query"], config={"callbacks": callbacks}
        )

    except Exception:

        # fail-close - a broken classfier must not silently pass every query through

        logger.exception("Guardrail classifier failed, failing closed")

        verdict = GuardrailVerdict(
            blocked=True,
            category="classifier_error",
            reason="Guardrail classifier failed; blocking as precaution",
        )

        get_client().update_current_span(
            metadata={
                "blocked": verdict.blocked,
                "category": verdict.category,
                "reason": verdict.reason,
            }
        )

    return {"guardrail_verdict": verdict.model_dump() if verdict.blocked else None}
