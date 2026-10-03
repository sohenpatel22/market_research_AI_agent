"""Adapters that let RAGAS and DeepEval use the project's provider-agnostic judge model"""

import sys
import types
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from market_research_agent.data.embeddings import embed_texts
from market_research_agent.llm.factory import get_judge_model


def _shim_ragas_imports() -> None:
    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = stub


class LocalEmbeddings(Embeddings):
    """The project's local sentence-transformer, as a LangChain Embeddings object"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0]


def ragas_judge(callbacks: list | None = None):
    """(llm, embeddings) wrappers for RAGAS metrics"""
    _shim_ragas_imports()
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    kwargs: dict[str, Any] = {"callbacks": callbacks} if callbacks else {}
    return LangchainLLMWrapper(get_judge_model(**kwargs)), LangchainEmbeddingsWrapper(
        LocalEmbeddings()
    )


def deepeval_judge(model: BaseChatModel | None = None):
    """A DeepEval judge backed by the project's judge model"""
    from deepeval.models import DeepEvalBaseLLM

    class ProjectJudge(DeepEvalBaseLLM):
        def __init__(self, chat_model: BaseChatModel):
            self.chat_model = chat_model
            super().__init__(getattr(chat_model, "model_name", None) or "judge")

        def load_model(self):
            return self.chat_model

        @staticmethod
        def _text(msg) -> str:
            return msg.content if isinstance(msg.content, str) else str(msg.content)

        def generate(self, prompt: str, schema=None) -> Any:
            if schema is not None:
                # DeepEval passes a Pydantic schema and expects an instance back.
                result = self.chat_model.with_structured_output(
                    schema, method="function_calling"
                ).invoke(prompt)
                if result is not None:
                    return result
            return self._text(self.chat_model.invoke(prompt))

        async def a_generate(self, prompt: str, schema=None) -> Any:
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return str(self.model_name)

    return ProjectJudge(model or get_judge_model())
