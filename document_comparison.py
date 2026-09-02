"""
document_comparison.py
-----------------------
Helps the Analyst compare information across multiple source documents:
similarities, differences, and (when numeric) which document "wins" on
a given metric.

Design note: this module does the *structuring* work deterministically
(grouping evidence by document, aligning numeric fields found via
TableExtractor/regex). The actual natural-language write-up of
similarities/differences is left to the Analyst's reasoning step in
agent.py, which receives this structured output as its input -- that
keeps this tool fast, cheap, and unit-testable without an LLM call.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from models import Evidence

_NUM_NEAR_LABEL = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9_\- ]{1,30}?)\s*[:=]?\s*(?P<value>\d+(?:\.\d+)?)\s*%?", re.IGNORECASE
)


class DocumentComparison:
    def group_by_document(self, evidence_list: List[Evidence]) -> Dict[str, List[Evidence]]:
        groups: Dict[str, List[Evidence]] = defaultdict(list)
        for ev in evidence_list:
            groups[ev.document].append(ev)
        return groups

    def extract_numeric_mentions(self, text: str) -> List[Dict[str, Any]]:
        """Pull loose 'label: number' mentions, e.g. 'CNN acc=92%' -> {"label": "CNN acc", "value": 92.0}."""
        found = []
        for m in _NUM_NEAR_LABEL.finditer(text):
            found.append({"label": m.group("label").strip(), "value": float(m.group("value"))})
        return found

    def compare(
        self,
        evidence_list: List[Evidence],
        metric_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build a structured comparison across documents.

        Returns:
            {
              "documents": ["paperA.pdf", "paperB.pdf", ...],
              "per_document": {
                  "paperA.pdf": {
                      "evidence_count": 2,
                      "numeric_mentions": [{"label": "CNN acc", "value": 92.0}, ...],
                      "excerpts": ["...", "..."]
                  },
                  ...
              },
              "shared_metrics": {"CNN acc": {"paperA.pdf": 92.0, "paperB.pdf": 95.0}},
              "best_by_metric": {"CNN acc": {"document": "paperB.pdf", "value": 95.0}}
            }
        """
        groups = self.group_by_document(evidence_list)
        per_document: Dict[str, Any] = {}
        metric_table: Dict[str, Dict[str, float]] = defaultdict(dict)

        for doc, chunks in groups.items():
            mentions = []
            excerpts = []
            for ev in chunks:
                mentions.extend(self.extract_numeric_mentions(ev.content))
                excerpts.append(ev.content[:280])
            if metric_keywords:
                mentions = [m for m in mentions if any(k.lower() in m["label"].lower() for k in metric_keywords)]
            per_document[doc] = {
                "evidence_count": len(chunks),
                "numeric_mentions": mentions,
                "excerpts": excerpts,
            }
            for m in mentions:
                # first mention of a label per doc wins (keeps it simple/deterministic)
                metric_table[m["label"]].setdefault(doc, m["value"])

        best_by_metric = {}
        for label, doc_values in metric_table.items():
            if doc_values:
                best_doc = max(doc_values, key=doc_values.get)
                best_by_metric[label] = {"document": best_doc, "value": doc_values[best_doc]}

        return {
            "documents": list(groups.keys()),
            "per_document": per_document,
            "shared_metrics": dict(metric_table),
            "best_by_metric": best_by_metric,
        }
