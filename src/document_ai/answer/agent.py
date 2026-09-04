"""
answer/agent.py
----------------
The Answer Agent — Member 3's core deliverable.

Takes an AnalystResult + the original question, uses its three @tools
(citation_formatter, source_formatter, response_formatter) via
llm.bind_tools(), and produces a FinalAnswer.

Member 3 deliverable.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from document_ai.llm.model import get_llm
from document_ai.schemas.analysis import AnalystResult
from document_ai.schemas.answer import Citation, FinalAnswer

from document_ai.answer.citation_formatter import build_citations, format_citations
from document_ai.answer.source_formatter import build_sources, format_sources
from document_ai.answer.response_formatter import assemble_response, format_response

import logging
logger = logging.getLogger(__name__)

# All tools the Answer Agent LLM can call
ANSWER_TOOLS = [format_citations, format_sources, format_response]

ANSWER_SYSTEM_PROMPT = """You are an expert at synthesizing research findings into clear,
well-cited answers.

You will receive:
  1. The user's original question
  2. An analysis produced by the Analyst Agent (may include calculations, tables, comparisons)
  3. The evidence chunks used in the analysis

Your job:
  1. Call format_citations with the evidence list to get numbered citations
  2. Call format_sources with the citations to build the bibliography
  3. Call format_response with the analysis text + citations to produce the final answer

The final answer must:
  - Directly address the question
  - Include inline citation markers like [1], [2] where claims are made
  - End with a numbered Sources / References section
  - Be clear enough for a non-expert to understand
"""


class AnswerAgent:
    """
    Deterministic Answer Agent (No LLM Tool Calling).

    Usage:
        agent = AnswerAgent()
        final = agent.answer(question, analyst_result)
    """

    def answer(self, question: str, analyst_result: AnalystResult) -> FinalAnswer:
        evidence_dicts = [e.model_dump() for e in analyst_result.evidence_used]
        analysis_text = analyst_result.analysis or _fallback_analysis(analyst_result)

        logger.info("    [Answer] Assembling response deterministically...")
        
        # Build everything deterministically
        citations_raw = build_citations(evidence_dicts)
        sources_text = build_sources(citations_raw)
        final_answer_text = assemble_response(analysis_text, citations_raw)

        # Build pydantic Citation objects
        citations = [
            Citation(
                ref_id=c["ref_id"],
                doc=c["doc"],
                page=c.get("page"),
                score=c["score"],
                snippet=c["snippet"],
            )
            for c in citations_raw
        ]

        # Confidence = average relevance score of evidence used
        scores = [e.score for e in analyst_result.evidence_used]
        confidence = sum(scores) / len(scores) if scores else 0.0

        return FinalAnswer(
            question=question,
            answer=final_answer_text,
            citations=citations,
            sources=sources_text,
            confidence=round(confidence, 3),
            metadata={
                "analyst_status": analyst_result.status,
                "iterations": analyst_result.iterations,
                "evidence_count": len(analyst_result.evidence_used),
            },
        )


# ── Helpers ────────────────────────────────────────────────────────────
def _fallback_analysis(result: AnalystResult) -> str:
    """Generate a minimal analysis string when the LLM produced none."""
    docs = sorted({e.doc for e in result.evidence_used})
    lines = [
        f"Based on {len(result.evidence_used)} evidence chunk(s) from "
        f"{len(docs)} document(s): {', '.join(docs)}."
    ]
    for calc in result.calculations:
        if isinstance(calc.get("result"), (int, float)):
            lines.append(f"- {calc.get('metric', calc.get('operation', 'Calculation'))}: "
                         f"{calc['result']:.2f}")
    if result.comparisons:
        for label, info in result.comparisons.get("best_by_metric", {}).items():
            lines.append(f"- Best {label}: {info['document']} at {info['value']}")
    return "\n".join(lines)
