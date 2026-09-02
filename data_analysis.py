"""
data_analysis.py
-----------------
Turns raw extracted numbers (from TableExtractor / DocumentComparison)
into higher-level insights: rankings, distributions, and simple trend
direction. Complements Calculator (single operations) by working on
whole datasets at once.
"""

import statistics
from typing import Any, Dict, List, Tuple


class DataAnalysis:
    def summary_stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        if len(values) > 1:
            stats["stdev"] = statistics.stdev(values)
        return stats

    def rank(self, items: Dict[str, float], descending: bool = True) -> List[Tuple[str, float]]:
        """Rank named items by value, e.g. rank({'CNN': 92, 'RNN': 89})."""
        return sorted(items.items(), key=lambda kv: kv[1], reverse=descending)

    def best_and_worst(self, items: Dict[str, float], higher_is_better: bool = True) -> Dict[str, Any]:
        if not items:
            return {}
        ranked = self.rank(items, descending=higher_is_better)
        return {"best": ranked[0], "worst": ranked[-1], "ranked": ranked}

    def trend(self, values: List[float]) -> str:
        """Very lightweight trend detection using slope sign of a linear fit."""
        n = len(values)
        if n < 2:
            return "insufficient_data"
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(values) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
        den = sum((x - x_mean) ** 2 for x in xs)
        if den == 0:
            return "flat"
        slope = num / den
        if abs(slope) < 1e-9:
            return "flat"
        return "increasing" if slope > 0 else "decreasing"

    def distribution(self, values: List[float], bucket_count: int = 5) -> Dict[str, int]:
        if not values:
            return {}
        lo, hi = min(values), max(values)
        if lo == hi:
            return {f"{lo}": len(values)}
        width = (hi - lo) / bucket_count
        buckets = {i: 0 for i in range(bucket_count)}
        for v in values:
            idx = min(int((v - lo) / width), bucket_count - 1)
            buckets[idx] += 1
        return {
            f"{lo + i*width:.2f}-{lo + (i+1)*width:.2f}": count
            for i, count in buckets.items()
        }
