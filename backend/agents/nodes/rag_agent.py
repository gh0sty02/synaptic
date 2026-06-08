from agents.graph import SynapticState
from chain.rag_chain import RagChain
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Any
from langfuse import observe

rag_chain = None
retriever = None


def init(embeddings: HuggingFaceEmbeddings) -> None:
    global rag_chain, retriever
    _rag = RagChain(embeddings)
    rag_chain = _rag.build()
    retriever = _rag.retriever


@observe(name="rag_agent_node")
async def rag_agent(state: SynapticState):
    memory_context = format_memory(state["short_term_memory"])
    callbacks = state.get("callbacks", [])
    result = await rag_chain.ainvoke(
        {"question": state["query"], "memory_context": memory_context},
        config={"callbacks": callbacks},
    )
    return {
        "final_answer": result["answer"],
        "retrieved_chunks": result["source_documents"],
        "citations": extract_citations(result["source_documents"]),
        "condensed_query": result.get("condensed_query", state["query"]),
    }


def format_memory(short_term_memory: list[dict[str, str]]) -> str:

    memory = ""
    for message in short_term_memory:
        role: str

        if message["role"] == "ai":
            role = "AI"
        else:
            role = "Human"
        memory += f"{role}: {message['content']}\n"

    return memory


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
