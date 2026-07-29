from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import (
    csv_loader,
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tools import get_abs_path


def _tokenize_for_bm25(text: str) -> list[str]:
    """Use Chinese character/bigram tokens and English words without another dependency."""
    tokens: list[str] = []
    #[\u4e00-\u9fff]+代表匹配一个或多个连续的中文字符，范围是从\u4e00到\u9fff，涵盖了大部分常用汉字。|表示逻辑“或”，[a-zA-Z0-9]+表示匹配一个或多个连续的英文字母（大小写）或数字。因此，这个正则表达式的整体意思是：匹配一段文本中所有的中文字符序列或者英文单词/数字序列。
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", (text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(part)
            tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


@dataclass
class _BM25Index:
    documents: list[Document]
    term_frequencies: list[Counter[str]]
    document_frequencies: Counter[str]
    average_length: float

    @classmethod
    def build(cls, documents: list[Document]) -> "_BM25Index":
        term_frequencies: list[Counter[str]] = []
        document_frequencies: Counter[str] = Counter()
        total_length = 0

        for document in documents:
            tokens = _tokenize_for_bm25(document.page_content)
            frequencies = Counter(tokens)
            term_frequencies.append(frequencies)
            document_frequencies.update(frequencies.keys())
            total_length += len(tokens)

        average_length = total_length / len(documents) if documents else 0.0
        return cls(documents, term_frequencies, document_frequencies, average_length)

    def search(self, query: str, limit: int) -> list[tuple[Document, float]]:
        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens or not self.documents:
            return []

        k1 = 1.5
        b = 0.75
        total_documents = len(self.documents)
        scores: list[tuple[Document, float]] = []
        for document, frequencies in zip(self.documents, self.term_frequencies):
            document_length = sum(frequencies.values())
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequencies.get(token, 0)
                inverse_document_frequency = log(
                    1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * document_length / (self.average_length or 1.0)
                )
                score += inverse_document_frequency * frequency * (k1 + 1) / denominator

            if score > 0:
                scores.append((document, score))

        return sorted(scores, key=lambda item: item[1], reverse=True)[:limit]


class HybridRetriever:
    """Fuse semantic and lexical rankings with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: Chroma,
        bm25_index: _BM25Index,
        top_k: int,
        candidate_k: int,
        rrf_k: int,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = max(1, rrf_k)

    @staticmethod
    def _document_key(document: Document) -> str:
        source = str((document.metadata or {}).get("source", ""))
        return f"{source}\n{document.page_content}"

    def _semantic_search(self, query: str) -> list[tuple[Document, float]]:
        try:
            return self.vector_store.similarity_search_with_relevance_scores(query, k=self.candidate_k)
        except Exception as error:
            logger.warning(f"[hybrid_retriever] 语义检索评分不可用，使用排名分数: {error}")
            documents = self.vector_store.similarity_search(query, k=self.candidate_k)
            return [(document, float(self.candidate_k - index)) for index, document in enumerate(documents)]

    def invoke(self, query: str) -> list[Document]:
        semantic_candidates = self._semantic_search(query)
        lexical_candidates = self.bm25_index.search(query, self.candidate_k)

        documents: dict[str, Document] = {}
        fused_scores: dict[str, float] = {}
        for rank, (document, _score) in enumerate(semantic_candidates, start=1):
            key = self._document_key(document)
            documents[key] = document
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (document, _score) in enumerate(lexical_candidates, start=1):
            key = self._document_key(document)
            documents[key] = document
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)[: self.top_k]
        return [documents[key] for key in ranked_keys]


class VectorStoreService:
    def __init__(self):
        self.vector_store = self._create_vector_store()
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        self.retrieval_mode = str(chroma_conf.get("retrieval_mode", "vector")).lower()
        self._bm25_index: _BM25Index | None = None
        self._loaded = False

    @staticmethod
    def _create_vector_store() -> Chroma:
        return Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

    @property
    def top_k(self) -> int:
        return int(chroma_conf["k"])

    @property
    def candidate_k(self) -> int:
        return max(self.top_k, int(chroma_conf.get("candidate_k", self.top_k * 3)))

    def get_retriever(self):
        """Keep the original LangChain retriever API for compatibility."""
        return self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

    def set_retrieval_mode(self, mode: str) -> None:
        mode = str(mode or "").lower()
        if mode not in {"vector", "hybrid"}:
            raise ValueError("retrieval_mode 只能是 vector 或 hybrid")
        self.retrieval_mode = mode

    def collection_count(self) -> int:
        return int(self.vector_store._collection.count())

    def is_ready(self) -> bool:
        return self.collection_count() > 0

    def _load_all_documents(self) -> list[Document]:
        payload = self.vector_store.get(include=["documents", "metadatas"])
        texts = payload.get("documents", []) or []
        metadatas = payload.get("metadatas", []) or []
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, metadatas)
            if text
        ]

    def _get_bm25_index(self) -> _BM25Index:
        if self._bm25_index is None:
            self._bm25_index = _BM25Index.build(self._load_all_documents())
        return self._bm25_index

    def retrieve(self, query: str, mode: str | None = None) -> list[Document]:
        selected_mode = str(mode or self.retrieval_mode).lower()
        if selected_mode == "vector":
            return self.vector_store.similarity_search(query, k=self.top_k)
        if selected_mode == "hybrid":
            retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_index=self._get_bm25_index(),
                top_k=self.top_k,
                candidate_k=self.candidate_k,
                rrf_k=int(chroma_conf.get("rrf_k", 60)),
            )
            return retriever.invoke(query)
        raise ValueError("retrieval_mode 只能是 vector 或 hybrid")

    def _md5_store_path(self) -> Path:
        return Path(get_abs_path(chroma_conf["md5_hex_store"]))

    def _read_indexed_hashes(self) -> set[str]:
        md5_path = self._md5_store_path()
        if not md5_path.exists():
            return set()
        return {line.strip() for line in md5_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    def _write_indexed_hashes(self, hashes: set[str]) -> None:
        md5_path = self._md5_store_path()
        md5_path.parent.mkdir(parents=True, exist_ok=True)
        md5_path.write_text("\n".join(sorted(hashes)) + ("\n" if hashes else ""), encoding="utf-8")

    def _reset_collection(self) -> None:
        self.vector_store.delete_collection()
        self.vector_store = self._create_vector_store()
        self._bm25_index = None
        self._write_indexed_hashes(set())
        self._loaded = False

    @staticmethod
    def _load_file_documents(path: str) -> list[Document]:
        if path.endswith("txt"):
            return txt_loader(path)
        if path.endswith("pdf"):
            return pdf_loader(path)
        if path.endswith("csv"):
            return csv_loader(path)
        return []

    def load_document(self, force_rebuild: bool = False) -> dict[str, int]:
    #这个函数的作用是构建本地 Chroma 知识库集合，并返回一个简要的构建摘要。它会检查是否需要强制重建知识库，如果不需要且已经加载过，则直接返回当前知识库的统计信息。如果需要重建，它会清空现有集合，读取允许的文件类型，计算每个文件的 MD5 哈希值，并根据哈希值判断文件是否已经存在于知识库中。对于新文件，它会加载文档内容，进行分片处理，并将分片添加到向量存储中。最后，它会更新已索引的哈希值，并返回一个包含添加、跳过、失败文件数量以及文档总数的摘要字典。
        """Build the local Chroma collection and return a small build summary."""
        if self._loaded and not force_rebuild:
            return {
                "added_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "document_count": self.collection_count(),
            }

        if force_rebuild:
            self._reset_collection()

        indexed_hashes = self._read_indexed_hashes()
        if self.collection_count() == 0 and indexed_hashes:
            # Chroma may have been removed while md5.text remained; rebuilding is safer than skipping every file.
            logger.warning("[加载知识库] 检测到空集合与历史MD5不一致，将重新写入全部知识文件")
            indexed_hashes = set()
        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )
        summary = {"added_files": 0, "skipped_files": 0, "failed_files": 0, "document_count": 0}

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                summary["failed_files"] += 1
                continue
            if md5_hex in indexed_hashes:
                logger.info(f"[加载知识库] {path} 内容已经存在于知识库，跳过")
                summary["skipped_files"] += 1
                continue

            try:
                documents = self._load_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库] {path} 无有效文本内容，跳过")
                    summary["skipped_files"] += 1
                    continue

                split_documents = self.spliter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"[加载知识库] {path} 分片后无内容，跳过")
                    summary["skipped_files"] += 1
                    continue

                self.vector_store.add_documents(split_documents)
                indexed_hashes.add(md5_hex)
                summary["added_files"] += 1
                logger.info(f"[加载知识库] {path} 内容加载成功，共 {len(split_documents)} 个分片")
            except Exception as error:
                summary["failed_files"] += 1
                logger.error(f"[加载知识库] {path} 加载失败：{error}", exc_info=True)

        self._write_indexed_hashes(indexed_hashes)
        self._bm25_index = None
        self._loaded = True
        summary["document_count"] = self.collection_count()
        return summary


if __name__ == "__main__":
    store = VectorStoreService()
    print(store.load_document())
