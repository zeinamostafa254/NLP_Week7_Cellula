import re
from typing import Any
from langchain_chroma import Chroma

from document_ai.config import COLLECTION_NAME, VECTOR_DB_DIR

class KeywordSearcher:
    """
    BM25 lexical search over the chunks stored in Chroma.
    Uses rank-bm25 to accurately score keyword relevance via TF-IDF.
    """

    def __init__(self):
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_DB_DIR),
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        # Simple lowercase tokenization
        return re.findall(r"\b[\w.-]+\b", text.lower())

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 is not installed. Run: pip install rank-bm25")
            
        data = self.vector_store.get(include=["documents", "metadatas"])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        
        if not documents:
            return []

        # Tokenize corpus for BM25
        tokenized_corpus = [self._tokens(doc or "") for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = self._tokens(query)
        if not tokenized_query:
            return []
            
        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)
        
        scored = []
        for doc, meta, score in zip(documents, metadatas, scores):
            if score > 0.0:  # Only include hits
                scored.append(
                    {
                        "content": doc,
                        "metadata": meta or {},
                        "score": float(score),
                        "retrieval_method": "keyword (bm25)",
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
