from typing import Any

from langfuse import observe

from agents.state import SynapticState

_REFUSAL_BY_CATEGORY = {
    "injection": (
        "I can't follow instructions embedded in a query — I can only help with "
        "technical questions from the indexed knowledge base."
    ),
    "off_topic": (
        "That's outside what I can help with — I can only answer software "
        "engineering questions from the indexed StackOverflow knowledge base."
    ),
    "harmful": "I can't help with that request.",
    "classifier_error": (
        "I'm unable to safely process this request right now. Please try again shortly."
    ),
}
_DEFAULT_REFUSAL = "I can't help with that request."


@observe(name="blocked_node")
async def blocked_node(state: SynapticState) -> dict[str, Any]:
    verdict = state.get("guardrail_verdict") or {}

    category = verdict.get("category")

    return {"final_answer": _REFUSAL_BY_CATEGORY.get(category, _DEFAULT_REFUSAL)}
