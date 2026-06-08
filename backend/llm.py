import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

_MODEL = os.environ["LLM_MODEL"]
_BASE_URL = os.environ["LLM_BASE_URL"]
_API_KEY = os.environ["LLM_API_KEY"]

# Used by: RagChain (final answer generation)
main_llm = ChatOpenAI(
    model=_MODEL,
    base_url=_BASE_URL,
    api_key=SecretStr(_API_KEY),
    temperature=1.0,
    top_p=0.95,
)

# Used by: Triage, Orchestrator, RagChain (condensation), MemoryManager (summarisation)
utility_llm = ChatOpenAI(
    model=_MODEL,
    base_url=_BASE_URL,
    api_key=SecretStr(_API_KEY),
    temperature=1.0,
    top_p=0.95,
)
