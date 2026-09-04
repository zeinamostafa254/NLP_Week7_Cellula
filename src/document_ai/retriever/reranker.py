import logging
from typing import Any

logger = logging.getLogger(__name__)

class Reranker:
    """
    CrossEncoder Reranker.
    Uses BAAI/bge-reranker-base to accurately score the relevance of retrieved chunks.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None
        
    def _get_model(self):
        if self._model is None:
            logger.info(f"    [Reranker] Loading CrossEncoder model: {self.model_name}...")
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                logger.error("    [Reranker] sentence-transformers not installed.")
                raise
        return self._model

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        if not results:
            return []
            
        model = self._get_model()
        pairs = [[query, item.get("content", "")] for item in results]
        
        # predict returns a list of float scores
        scores = model.predict(pairs)
        
        reranked = []
        for i, item in enumerate(results):
            updated = dict(item)
            updated["initial_score"] = float(item.get("score", 0.0))
            updated["rerank_score"] = float(scores[i])
            updated["score"] = float(scores[i])
            reranked.append(updated)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
