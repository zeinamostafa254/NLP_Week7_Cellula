"""
Quick sanity tests. Run with: python3 test_agent.py
No pytest dependency required, so it works in any barebones env.
"""

from agent import AnalystAgent
from retrieve_more import MockRetrieverCallback


def test_enough_evidence_case():
    """Mirrors the PDF example: 'average CNN accuracy across three papers'."""
    evidence_bundle = [
        {"document": "PaperA.pdf", "page": 4, "score": 0.93, "content": "CNN acc=92%"},
        {"document": "PaperB.pdf", "page": 6, "score": 0.90, "content": "CNN acc=95%"},
        {"document": "PaperC.pdf", "page": 3, "score": 0.88, "content": "CNN acc=89%"},
    ]
    agent = AnalystAgent(retriever_callback=MockRetrieverCallback([]))
    result = agent.analyze("What is the average CNN accuracy across the three papers?", evidence_bundle)

    assert result.status == "enough_evidence", result.missing_information
    assert any(abs(c["result"] - 92.0) < 0.01 for c in result.calculations), result.calculations
    print("PASS: enough_evidence case ->", result.analysis)


def test_feedback_loop_case():
    """Starts with thin/low-score evidence; the mock retriever supplies more on request."""
    thin_bundle = [
        {"document": "PaperA.pdf", "page": 1, "score": 0.30, "content": "some background text"},
    ]
    richer_results = [
        {"document": "PaperA.pdf", "page": 2, "score": 0.85, "content": "CNN acc=92%"},
        {"document": "PaperB.pdf", "page": 5, "score": 0.88, "content": "CNN acc=95%"},
    ]
    agent = AnalystAgent(retriever_callback=MockRetrieverCallback(richer_results), max_iterations=3)
    result = agent.analyze("Compare CNN accuracy between PaperA and PaperB", thin_bundle)

    assert len(agent.retrieve_more.call_log) >= 1, "Expected the feedback loop to call the retriever"
    print("PASS: feedback_loop case -> status:", result.status, "| iterations:", result.iterations)


def test_no_more_evidence_available():
    """Retriever has nothing further to give -> should stop and report need_more_evidence."""
    thin_bundle = [{"document": "PaperA.pdf", "page": 1, "score": 0.1, "content": "irrelevant"}]
    agent = AnalystAgent(retriever_callback=MockRetrieverCallback([]))
    result = agent.analyze("What is the F1 score?", thin_bundle)

    assert result.status == "need_more_evidence"
    print("PASS: no_more_evidence case -> missing:", result.missing_information)


if __name__ == "__main__":
    test_enough_evidence_case()
    test_feedback_loop_case()
    test_no_more_evidence_available()
    print("\nAll analyst module tests passed.")
