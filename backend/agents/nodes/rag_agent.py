from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langfuse import observe

from agents.graph import SynapticState
from chain.rag_chain import RagChain
from context.engineer import assemble

rag_chain = None
retriever = None


def init(embeddings: HuggingFaceEmbeddings) -> None:
    global rag_chain, retriever
    _rag = RagChain(embeddings)
    rag_chain = _rag.build()
    retriever = _rag.retriever


@observe(name="rag_agent_node")
async def rag_agent(state: SynapticState) -> dict[str, Any]:

    bundle = assemble(
        "rag_agent",
        short_term_memory=state["short_term_memory"],
        long_term_memory=state["long_term_memory"],
    )
    callbacks = state.get("callbacks", [])
    result = await rag_chain.ainvoke(
        {
            "question": state["query"],
            "memory_context": bundle.memory_context,
            "use_hyde": state.get("use_hyde", False),
        },
        config={"callbacks": callbacks},
    )

    token_counts = {
        **bundle.token_count,
        "retrieved_chunks": result.get("chunks_tokens", 0),
    }

    return {
        "final_answer": result["answer"],
        "retrieved_chunks": result["source_documents"],
        "citations": extract_citations(result["source_documents"]),
        "condensed_query": result.get("condensed_query", state["query"]),
        "token_counts": token_counts,
        "total_tokens": sum(token_counts.values()),
        "decision": bundle.decision,
        "budget_exceeded": bundle.budget_exceeded
        or result.get("chunks_truncated", False),
    }


def extract_citations(documents: list[Document]) -> list[dict[str, Any]]:
    citations = []
    for doc in documents:
        citations.append(
            {
                "title": doc.metadata.get("title"),
                "relevance_score": doc.metadata.get("relevance_score"),
            }
        )

    return citations
