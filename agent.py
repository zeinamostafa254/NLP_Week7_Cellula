"""
agent.py
--------
The Analyst Agent. Matches the diagram's "brain" box: takes an evidence
bundle from the Retriever, uses its 5 tools (Calculator, Table Extractor,
Document Comparison, Data Analysis, Search/Retrieve More Evidence), runs
the Evidence Assessment gate, and either:

  - returns an AnalystResult with status="enough_evidence" (hand off to
    Answer Agent), or
  - loops back through RetrieveMoreTool for additional evidence and
    tries again, up to `max_iterations`.

Reasoning (turning tables/stats into a written analysis) is pluggable:
pass `llm_reasoner` (e.g. a function that calls the Claude API) for a
richer natural-language write-up, or omit it to get a deterministic
template-based summary (useful for unit tests / no-API-key dev).
"""

from typing import Any, Callable, Dict, List, Optional

from calculator import Calculator
from data_analysis import DataAnalysis
from document_comparison import DocumentComparison
from evidence_assessment import AssessmentConfig, EvidenceAssessor
from models import AnalystResult, Evidence
from retrieve_more import RetrieveMoreTool, RetrieverCallback
from table_extractor import TableExtractor

LLMReasoner = Callable[[str, List[Evidence], Dict[str, Any]], str]


class AnalystAgent:
    def __init__(
        self,
        retriever_callback: RetrieverCallback,
        assessment_config: Optional[AssessmentConfig] = None,
        llm_reasoner: Optional[LLMReasoner] = None,
        max_iterations: int = 3,
    ):
        self.calculator = Calculator()
        self.table_extractor = TableExtractor()
        self.document_comparison = DocumentComparison()
        self.data_analysis = DataAnalysis()
        self.assessor = EvidenceAssessor(assessment_config)
        self.retrieve_more = RetrieveMoreTool(retriever_callback)
        self.llm_reasoner = llm_reasoner
        self.max_iterations = max_iterations

    def analyze(self, question: str, evidence_bundle: List[Dict[str, Any]]) -> AnalystResult:
        evidence = [Evidence.from_dict(e) for e in evidence_bundle]
        iterations = 0

        while True:
            iterations += 1
            assessment = self.assessor.assess(question, evidence)

            if assessment.enough_evidence or iterations >= self.max_iterations:
                return self._build_result(
                    question=question,
                    evidence=evidence,
                    enough=assessment.enough_evidence,
                    missing=assessment.missing_information,
                    follow_up=assessment.suggested_follow_up_query,
                    iterations=iterations,
                )

            # Feedback loop: ask the Retriever Agent for more, then merge
            # and re-assess (bounded by max_iterations to avoid infinite loops).
            new_evidence = self.retrieve_more.request_more_evidence(
                assessment.suggested_follow_up_query or question
            )
            if not new_evidence:
                # Retriever had nothing more to give -- stop looping.
                return self._build_result(
                    question=question,
                    evidence=evidence,
                    enough=False,
                    missing=assessment.missing_information,
                    follow_up=assessment.suggested_follow_up_query,
                    iterations=iterations,
                )
            evidence = _merge_evidence(evidence, new_evidence)

    # ------------------------------------------------------------------
    def _build_result(
        self,
        question: str,
        evidence: List[Evidence],
        enough: bool,
        missing: List[str],
        follow_up: Optional[str],
        iterations: int,
    ) -> AnalystResult:
        if not enough:
            return AnalystResult(
                status="need_more_evidence",
                evidence_used=evidence,
                missing_information=missing,
                follow_up_query=follow_up,
                iterations=iterations,
            )

        tables = self.table_extractor.extract_from_evidence(evidence)
        comparison = self.document_comparison.compare(evidence)
        calculations = self._auto_calculations(comparison)
        analysis_text = self._reason(question, evidence, {
            "tables": tables,
            "comparison": comparison,
            "calculations": calculations,
        })

        return AnalystResult(
            status="enough_evidence",
            analysis=analysis_text,
            calculations=calculations,
            tables=tables,
            comparisons=comparison,
            evidence_used=evidence,
            iterations=iterations,
        )

    def _auto_calculations(self, comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
        """For every metric mentioned in 2+ documents, compute a quick
        average via the Calculator so the Answer Agent has a ready number."""
        calcs = []
        for label, doc_values in comparison.get("shared_metrics", {}).items():
            values = list(doc_values.values())
            if len(values) >= 2:
                calcs.append({"metric": label, **self.calculator.average(values)})
        return calcs

    def _reason(self, question: str, evidence: List[Evidence], data: Dict[str, Any]) -> str:
        if self.llm_reasoner:
            return self.llm_reasoner(question, evidence, data)
        return _template_summary(question, evidence, data)


# ------------------------------------------------------------------------
def _merge_evidence(existing: List[Evidence], new: List[Evidence]) -> List[Evidence]:
    seen = {(e.document, e.page, e.content) for e in existing}
    merged = list(existing)
    for e in new:
        key = (e.document, e.page, e.content)
        if key not in seen:
            merged.append(e)
            seen.add(key)
    return merged


def _template_summary(question: str, evidence: List[Evidence], data: Dict[str, Any]) -> str:
    docs = sorted({e.document for e in evidence})
    lines = [f"Analysis based on {len(evidence)} evidence chunk(s) from {len(docs)} document(s): {', '.join(docs)}."]

    calcs = data.get("calculations", [])
    for c in calcs:
        lines.append(f"- Average {c['metric']}: {c['result']:.2f} (from {c['inputs']}).")

    best_by_metric = data.get("comparison", {}).get("best_by_metric", {})
    for label, info in best_by_metric.items():
        lines.append(f"- Highest {label}: {info['document']} at {info['value']}.")

    tables = data.get("tables", [])
    if tables:
        lines.append(f"- Extracted {len(tables)} table(s) from the evidence for reference.")

    if len(lines) == 1:
        lines.append("No numeric patterns were detected; see evidence excerpts for qualitative details.")

    return "\n".join(lines)
