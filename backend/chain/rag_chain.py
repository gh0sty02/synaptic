import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from ingestion.stackoverflow_loader import EMBEDDING_MODEL, CONN_STR
from retrieval.chunks_retriever import ChunksRetriever
from constants import SYSTEM_PROMPT
from operator import itemgetter

load_dotenv()

LLM_MODEL = os.environ["LLM_MODEL"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]


class RagChain:
    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        self.llm = self._create_llm()
        self.embeddings = embeddings
        self.retriever = ChunksRetriever(
            embeddings=self.embeddings,
            conn_str=CONN_STR,
            k=5,
        )
        self.prompt = self._build_prompt()

    def _create_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=SecretStr(LLM_API_KEY),
            temperature=1.0,
            top_p=0.95,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 1024},
        )

    def _format_docs(self, docs: list[Document]) -> str:
        return "\n\n".join(
            f"[Source: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Previous Conversation:\n{memory_context}\n\nCurrent Question: {question}\n\nContext:\n{context}",
                ),
            ]
        )

    def build(self):
        return (
            {
                "source_documents": itemgetter("question") | self.retriever,
                "question": itemgetter("question"),
                "memory_context": itemgetter("memory_context"),
            }
            | RunnablePassthrough.assign(
                context=lambda x: self._format_docs(x["source_documents"])
            )
            | RunnablePassthrough.assign(
                answer=self.prompt | self.llm | StrOutputParser()
            )
        )
