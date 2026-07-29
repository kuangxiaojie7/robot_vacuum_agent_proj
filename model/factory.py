from abc import ABC
from abc import abstractmethod
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Optional
import os
from utils.config_handler import rag_conf


# The workspace provides both chat completion and embeddings through this endpoint.
dashscope_compatible_base_url = str(
    rag_conf.get("dashscope_compatible_base_url", "") or ""
).strip().rstrip("/")


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def __init__(self, config_key: str = "chat_model_name"):
        self.config_key = config_key

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        model_name = str(rag_conf.get(self.config_key, "") or "").strip()
        if not model_name:
            return None
        return ChatOpenAI(
            model=model_name,
            api_key=_get_api_key(),
            base_url=dashscope_compatible_base_url,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            api_key=_get_api_key(),
            base_url=dashscope_compatible_base_url,
            # Bailian's compatible endpoint accepts text, not OpenAI tiktoken ID lists.
            check_embedding_ctx_length=False,
        )


def _get_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY 环境变量")
    if not dashscope_compatible_base_url:
        raise ValueError("未配置 dashscope_compatible_base_url")
    return api_key


chat_model = ChatModelFactory("chat_model_name").generator()
judge_model = ChatModelFactory("judge_model_name").generator()
embed_model = EmbeddingsFactory().generator()
