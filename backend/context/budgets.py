import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")

# Unsloth Studio Gemma 4 E4B deployment.
N_CTX = 60_928

# Tokens reserved for the model's own generated answer. Enforced via
# llm.py's main_llm(max_tokens=RESERVED_OUTPUT_TOKENS).
RESERVED_OUTPUT_TOKENS = 2_048

# Buffer against tiktoken cl100k_base being only an approximation of
# Gemma's real tokenizer (see ARCHITECTURE.md §3). Not enforced at a
# specific call site — it's headroom baked into TOTAL_INPUT_BUDGET below.
SAFETY_MARGIN_TOKENS = 2_048

TOTAL_INPUT_BUDGET = N_CTX - RESERVED_OUTPUT_TOKENS - SAFETY_MARGIN_TOKENS

AGENT_BUDGET = {
    "rag_agent": {
        "total": TOTAL_INPUT_BUDGET,
        "system_prompt": 1_500,
        "query_and_prompt_scaffolding": 1_100,
        "short_term_memory": 9_000,
        "long_term_memory": 6_800,
        "retrieved_chunks": 36_000,
        "spare": 2_432,
    },
    "orchestrator_node": {
        "total": TOTAL_INPUT_BUDGET,
        "system_prompt": 1_500,
        "query_and_prompt_scaffolding": 1_100,
        "short_term_memory": 7_900,
        "long_term_memory": 11_300,
        "retrieved_chunks": 31_600,
        "spare": 3_432,
    },
}
