"""
models.py
---------
Shared data structures for the Analyst Agent.

These mirror the "evidence bundle" format produced by Member 1's Retriever
Agent, so the two modules plug together without any translation layer:

    {
        "document": "rag.pdf",
        "page": 5,
        "score": 0.94,
        "content": "...chunk text..."
    }
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    """A single retrieved chunk, as produced by the Retriever Agent."""
    document: str
    page: Optional[int]
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Evidence":
        return Evidence(
            document=d.get("document", "unknown"),
            page=d.get("page"),
            score=float(d.get("score", 0.0)),
            content=d.get("content", ""),
            metadata=d.get("metadata", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document,
            "page": self.page,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class AnalystResult:
    """
    What the Analyst Agent hands to the Orchestrator.

    status == "enough_evidence"  -> ready for the Answer Agent
    status == "need_more_evidence" -> feedback loop back to Retriever Agent
    """
    status: str  # "enough_evidence" | "need_more_evidence"
    analysis: Optional[str] = None
    calculations: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    comparisons: Optional[Dict[str, Any]] = None
    evidence_used: List[Evidence] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    follow_up_query: Optional[str] = None
    iterations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "analysis": self.analysis,
            "calculations": self.calculations,
            "tables": self.tables,
            "comparisons": self.comparisons,
            "evidence_used": [e.to_dict() for e in self.evidence_used],
            "missing_information": self.missing_information,
            "follow_up_query": self.follow_up_query,
            "iterations": self.iterations,
        }
