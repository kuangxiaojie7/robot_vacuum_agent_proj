import unittest

from langchain_core.documents import Document

from rag.rag_service import RagSummarizeService
from rag.vector_store import HybridRetriever, _BM25Index


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def similarity_search_with_relevance_scores(self, _query, k):
        return [(document, 0.7 - index * 0.1) for index, document in enumerate(self.documents[:k])]


class RagRetrievalTests(unittest.TestCase):
    def test_hybrid_retriever_keeps_lexical_match_and_source_reference(self):
        documents = [
            Document(page_content="滤网堵塞会导致吸力下降，清洗后需要晾干。", metadata={"source": "/tmp/维护保养.txt"}),
            Document(page_content="地毯模式可以提高深层清洁能力。", metadata={"source": "/tmp/选购指南.txt"}),
        ]
        retriever = HybridRetriever(
            vector_store=FakeVectorStore(documents),
            bm25_index=_BM25Index.build(documents),
            top_k=1,
            candidate_k=2,
            rrf_k=60,
        )

        result = retriever.invoke("滤网堵塞")
        self.assertEqual(result[0].metadata["source"], "/tmp/维护保养.txt")
        references = RagSummarizeService.build_source_references(result)
        self.assertEqual(references, ["[1] 维护保养.txt：滤网堵塞会导致吸力下降，清洗后需要晾干。"])

    def test_rrf_boosts_a_document_returned_by_both_retrievers(self):
        semantic_only = Document(
            page_content="这是语义检索排第一但没有关键词的片段。",
            metadata={"source": "/tmp/语义片段.txt"},
        )
        shared = Document(
            page_content="设备出现 E42 代码时，需要检查充电触点。",
            metadata={"source": "/tmp/故障排除.txt"},
        )
        retriever = HybridRetriever(
            vector_store=FakeVectorStore([semantic_only, shared]),
            bm25_index=_BM25Index.build([semantic_only, shared]),
            top_k=1,
            candidate_k=2,
            rrf_k=60,
        )

        result = retriever.invoke("E42 代码")
        self.assertEqual(result[0].metadata["source"], "/tmp/故障排除.txt")
