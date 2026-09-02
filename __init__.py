"""
analyst/
--------
Member 2's deliverable: the Analyst Agent and its 5 tools.

Public interface (what Member 3's Orchestrator should import):

    from analyst import AnalystAgent, AnalystResult, Evidence

    agent = AnalystAgent(retriever_callback=my_retriever_function)
    result = agent.analyze(question, evidence_bundle)

    if result.status == "enough_evidence":
        # hand result.analysis / result.tables / result.comparisons
        # to the Answer Agent
        ...
    else:
        # result.status == "need_more_evidence"
        # (in practice AnalystAgent already looped internally via
        # RetrieveMoreTool up to max_iterations, so seeing this status
        # means the Retriever had nothing further to offer)
        ...
"""

from agent import AnalystAgent
from calculator import Calculator, CalculatorError
from data_analysis import DataAnalysis
from document_comparison import DocumentComparison
from evidence_assessment import AssessmentConfig, Assessment, EvidenceAssessor
from models import AnalystResult, Evidence
from retrieve_more import MockRetrieverCallback, RetrieveMoreTool, RetrieverCallback
from table_extractor import TableExtractor

__all__ = [
    "AnalystAgent",
    "AnalystResult",
    "Evidence",
    "Calculator",
    "CalculatorError",
    "DataAnalysis",
    "DocumentComparison",
    "TableExtractor",
    "EvidenceAssessor",
    "Assessment",
    "AssessmentConfig",
    "RetrieveMoreTool",
    "RetrieverCallback",
    "MockRetrieverCallback",
]
