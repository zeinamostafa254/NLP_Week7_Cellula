from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """One retrieved chunk plus the metadata needed for later citation."""

    doc: str = Field(..., description="Original document filename")
    page: Optional[int] = Field(None, description="1-based page number when known")
    score: float = Field(..., description="Normalized relevance score")
    content: str = Field(..., description="Retrieved text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document metadata like author or date."
    )
    retrieval_method: str = Field(
        "semantic",
        description="The retrieval mechanism used (e.g., semantic, keyword)."
    )


class EvidenceBundle(BaseModel):
    """Stable interface between Member 1 (Retriever) and Member 2 (Analyst)."""

    query: str = Field(
        ..., 
        description="The user's original query."
    )
    rewritten_query: str = Field(
        ...,
        description="The optimized query used for retrieval."
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="List of evidence chunks retrieved from the database."
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters applied during the search."
    )
    source_count: int = Field(
        0,
        description="Number of unique documents the evidence was drawn from."
    )
