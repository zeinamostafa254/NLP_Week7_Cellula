from __future__ import annotations
from typing import List, Optional, Dict
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from document_ai.schemas.evidence import Evidence, EvidenceBundle
from document_ai.schemas.analysis import AnalystResult
from document_ai.llm.model import get_llm
from document_ai.analyst.calculator import (
    Calculator,
    calculate_average,
    calculate_expression,
    calculate_percent_change,
    calculate_summary_stats,
)
from document_ai.analyst.data_analysis import DataAnalysis, compute_distribution, detect_trend, rank_items
from document_ai.analyst.document_comparison import DocumentComparison, compare_documents
from document_ai.analyst.evidence_assessment import Assessment, AssessmentConfig, EvidenceAssessor
from document_ai.analyst.retrieve_more import MockRetrieverCallback, RetrieveMoreTool, retrieve_more_evidence
from document_ai.analyst.table_extractor import TableExtractor, extract_tables_from_text

logger = logging.getLogger(__name__)

ANALYST_TOOLS = [
    calculate_expression,
    calculate_average,
    calculate_percent_change,
    calculate_summary_stats,
    extract_tables_from_text,
    compare_documents,
    rank_items,
    detect_trend,
    compute_distribution,
    retrieve_more_evidence,
]

ANALYST_SYSTEM_PROMPT = """You are an expert document analyst.
Your job is to read retrieved evidence and answer the user's question accurately.

You have access to the following tools:
- retrieve_more_evidence: Use this to fetch more specific data if current evidence is insufficient.
- calculate_expression: evaluate numeric expressions
- calculate_average / calculate_summary_stats: compute statistics over lists of numbers
- calculate_percent_change: compute percentage changes
- extract_tables_from_text: parse tables from evidence text
- compare_documents: compare metrics across multiple documents
- rank_items: rank items by numeric value
- detect_trend: detect if a series is increasing/decreasing/flat
- compute_distribution: histogram a set of numbers

Use tools as needed, then provide a clear written analysis that directly answers the question.
Always cite evidence by document name and page when possible.
"""

class AnalystAgent:
    def __init__(
        self,
        retriever_callback,
        assessment_config: Optional[AssessmentConfig] = None,
        max_iterations: int = 3,
        max_tool_turns: int = 10,
    ):
        self.assessor = EvidenceAssessor(assessment_config)
        self.retrieve_more = RetrieveMoreTool(retriever_callback)
        self.max_iterations = max_iterations
        self.max_tool_turns = max_tool_turns
        
        self.agent = create_react_agent(
            model=get_llm(),
            tools=ANALYST_TOOLS,
            prompt=ANALYST_SYSTEM_PROMPT
        )

    def analyze(self, question: str, evidence_bundle: EvidenceBundle) -> AnalystResult:
        evidence = list(evidence_bundle.evidence)
        iterations = 0

        while True:
            iterations += 1
            logger.info(f"    [Analyst] Assessment loop {iterations}")
            assessment: Assessment = self.assessor.assess(question, evidence)

            if assessment.enough_evidence:
                logger.info(f"    [Analyst] -> 'enough_evidence'. Running LLM tool loop...")
                return self._run_tool_loop(question, evidence, assessment, iterations)
            elif iterations >= self.max_iterations:
                logger.info(f"    [Analyst] -> Max iterations reached. Forcing LLM tool loop.")
                return self._run_tool_loop(question, evidence, assessment, iterations)

            logger.info(f"    [Analyst] -> 'need_more_evidence' (Missing: {assessment.missing_information})")
            
            new_evidence = self.retrieve_more.request_more_evidence(
                assessment.suggested_follow_up_query or question
            )
            if not new_evidence:
                logger.warning("    [Analyst] Retriever found no additional evidence. Stopping loop.")
                return self._run_tool_loop(question, evidence, assessment, iterations)

            logger.info(f"    [Analyst] Retrieved {len(new_evidence)} additional chunks.")
            evidence = _merge_evidence(evidence, new_evidence)

    def _run_tool_loop(
        self,
        question: str,
        evidence: List[Evidence],
        assessment: Assessment,
        iterations: int,
    ) -> AnalystResult:
        logger.info(f"    [Analyst] Calling LLM tools via ReactAgent")
        status = "enough_evidence" if assessment.enough_evidence else "need_more_evidence"

        evidence_text = "\n\n".join(
            f"[{e.doc}, p.{e.page}, score={e.score:.2f}]\n{e.content}"
            for e in evidence
        )

        message = f"Question: {question}\n\nEvidence:\n{evidence_text}\n\nAnalyze this evidence to answer the question."
        
        response = self.agent.invoke(
            {"messages": [("user", message)]}
        )
        
        analysis_text = response["messages"][-1].content if response["messages"] else ""

        return AnalystResult(
            status=status,
            analysis=analysis_text or None,
            calculations=[],
            tables=[],
            comparisons=None,
            evidence_used=evidence,
            missing_information=assessment.missing_information,
            follow_up_query=assessment.suggested_follow_up_query,
            iterations=iterations,
        )

def _merge_evidence(existing: List[Evidence], new: List[Evidence]) -> List[Evidence]:
    seen = {(e.doc, e.page, e.content) for e in existing}
    merged = list(existing)
    for e in new:
        key = (e.doc, e.page, e.content)
        if key not in seen:
            merged.append(e)
            seen.add(key)
    return merged
