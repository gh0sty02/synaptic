from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import utility_llm

CONDENSATION_SYSTEM_PROMPT = (
    "Rewrite the follow-up question as a standalone question by replacing any "
    "pronouns or vague references with the specific topic from the conversation history.\n\n"
    "Example:\n"
    "History: Human: How do I sort a list in Python?\n"
    "Follow-up: How do I do the same in Ruby?\n"
    "Rewritten: How do I sort a list in Ruby?\n\n"
    "Output only the rewritten question, nothing else."
)

_condensation_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSATION_SYSTEM_PROMPT),
            (
                "human",
                "Conversation History:\n{memory_context}\n\nFollow-up Question: {question}",
            ),
        ]
    )
    | utility_llm
    | StrOutputParser()
)


async def condense_query(question: str, memory_context: str) -> str:
    """Rewrite a follow-up question into a standalone retrieval query using
    prior human turns. Returns `question` unchanged if there is no history to
    resolve referential words against (e.g. the first turn in a session)."""
    if not memory_context.strip():
        return question

    human_history = "\n".join(
        line for line in memory_context.splitlines() if line.startswith("Human:")
    )
    if not human_history:
        return question

    condensed = (
        await _condensation_chain.ainvoke(
            {"memory_context": human_history, "question": question}
        )
    ).strip()
    return condensed or question
