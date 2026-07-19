from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import utility_llm

_HYDE_SYSTEM_PROMPT = (
    "Write a short, plausible passage that answers the following question, in the "
    "style of a real document from its subject area — a Stack Overflow answer for "
    "programming questions (with the terminology and code patterns a real answer "
    "would use), or a match report or stats summary for cricket questions (with the "
    "team names, scores, and cricketing terminology a real report would use). "
    "This passage is only used to find similar real documents by embedding "
    "similarity, never shown to the user, so it does not need to be factually "
    "correct — it only needs to match the vocabulary, structure, and level of "
    "detail of a genuine document. Do not hedge, add disclaimers, or mention that "
    "the answer is hypothetical or may be inaccurate. Keep it to 2-4 sentences. "
    "Output only the passage text, nothing else."
)


_hyde_chain = (
    ChatPromptTemplate.from_messages(
        [("system", _HYDE_SYSTEM_PROMPT), ("human", "{question}")]
    )
    | utility_llm
    | StrOutputParser()
)


async def generate_hypothetical_answer(question: str) -> str:
    result = await _hyde_chain.ainvoke({"question": question})

    return result or question
