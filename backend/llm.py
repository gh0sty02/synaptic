import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from context.budgets import RESERVED_OUTPUT_TOKENS

load_dotenv()

_MODEL = os.environ["LLM_MODEL"]
_BASE_URL = os.environ["LLM_BASE_URL"]
_API_KEY = os.environ["LLM_API_KEY"]

_EVAL_MODEL = os.environ["EVAL_MODEL"]
_EVAL_MODEL_BASE_URL = os.environ["EVAL_MODEL_BASE_URL"]
_EVAL_MODEL_API_KEY = os.environ["EVAL_MODEL_API_KEY"]

# Used by: RagChain (final answer generation), Orchestrator (multi-intent merge)
main_llm = ChatOpenAI(
    model=_MODEL,
    base_url=_BASE_URL,
    api_key=SecretStr(_API_KEY),
    temperature=1.0,
    top_p=0.95,
    max_tokens=RESERVED_OUTPUT_TOKENS,
)

# Used by: Triage, RagChain (condensation), MemoryManager (summarisation),
# ragas eval judge prompts. Low temperature — these are classification/format-constrained
# tasks, not creative generation, and high temp was causing ragas judge output to drift
# out of the expected JSON schema (RagasOutputParserException).
utility_llm = ChatOpenAI(
    model=_MODEL,
    base_url=_BASE_URL,
    api_key=SecretStr(_API_KEY),
    temperature=0.2,
    top_p=0.95,
    max_tokens=RESERVED_OUTPUT_TOKENS,
)

# Used by: ragas eval judge (RagasOutputParserException fixes require a strong,
# stable instruction-follower for structured-output prompts — pinned independent
# of whatever LLM_* points at, so swapping main_llm/utility_llm doesn't affect eval runs)
judge_llm = ChatOpenAI(
    model=_EVAL_MODEL,
    base_url=_EVAL_MODEL_BASE_URL,
    api_key=SecretStr(_EVAL_MODEL_API_KEY),
    temperature=0.2,
    max_tokens=RESERVED_OUTPUT_TOKENS,
)
