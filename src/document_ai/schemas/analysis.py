"""
schemas/analysis.py
--------------------
AnalystResult - the output contract between the Analyst Agent (Member 2)
and the Orchestrator / Answer Agent (Member 3).

status == "enough_evidence"   -> ready for AnswerAgent
status == "need_more_evidence" -> Orchestrator should loop back to Retriever
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from document_ai.schemas.evidence import Evidence


class AnalystResult(BaseModel):
    status: Literal["enough_evidence", "need_more_evidence"] = Field(
        ...,
        description="Whether the retrieved evidence is sufficient to answer the question."
    )

    # Populated when status == "enough_evidence"
    analysis: Optional[str] = Field(
        None, 
        description="The detailed textual analysis answering the user's question."
    )
    calculations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of calculation results performed during analysis."
    )
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted data tables from the evidence."
    )
    comparisons: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured comparison metrics between multiple documents."
    )

    # Always populated
    evidence_used: List[Evidence] = Field(
        default_factory=list,
        description="The chunks of evidence actually referenced in the analysis."
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="List of specific information gaps preventing a complete answer."
    )
    follow_up_query: Optional[str] = Field(
        None,
        description="A suggested search query to find the missing information."
    )
    iterations: int = Field(
        0,
        description="Number of analysis-retrieval loops performed."
    )
