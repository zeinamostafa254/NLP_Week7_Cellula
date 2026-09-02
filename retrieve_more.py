"""
retrieve_more.py
------------------
Implements the "Search / Retrieve More Evidence" tool and the feedback
loop arrow that goes from Analyst Agent -> Retriever Agent in the
architecture diagram.

This module intentionally does NOT import Member 1's Retriever code
directly (that would create a hard dependency between modules while
you're building in parallel). Instead it defines a small callback
interface: whoever wires the Orchestrator together (Member 3) passes in
a callable that matches `RetrieverCallback`, and this tool calls it.

During standalone development/testing, use `MockRetrieverCallback` to
simulate the Retriever Agent's responses.
"""

from typing import Any, Callable, Dict, List, Protocol

from models import Evidence


class RetrieverCallback(Protocol):
    """Matches the signature Member 1's Retriever Agent should expose:
    question (str) -> list of evidence dicts."""

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        ...


class RetrieveMoreTool:
    def __init__(self, retriever_callback: RetrieverCallback):
        self.retriever_callback = retriever_callback
        self.call_log: List[str] = []

    def request_more_evidence(self, follow_up_query: str) -> List[Evidence]:
        """Send a refined query back to the Retriever Agent and wrap the
        response as Evidence objects."""
        self.call_log.append(follow_up_query)
        raw_results = self.retriever_callback(follow_up_query)
        return [Evidence.from_dict(r) for r in raw_results]


class MockRetrieverCallback:
    """Stand-in for Member 1's module so Member 2 can develop/test in
    isolation. Replace with the real Retriever Agent at integration time."""

    def __init__(self, canned_results: List[Dict[str, Any]] = None):
        self.canned_results = canned_results or []

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        return self.canned_results
