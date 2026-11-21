"""Simple summary-table builders."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, stdev


def summarize_mean_std(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...],
    metric_keys: tuple[str, ...],
) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        buckets[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for key, bucket in sorted(buckets.items()):
        summary_row = {name: value for name, value in zip(group_keys, key)}
        for metric_key in metric_keys:
            values = []
            for item in bucket:
                raw_value = item.get(metric_key)
                if raw_value in (None, ""):
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not math.isnan(value):
                    values.append(value)
            if not values:
                continue
            summary_row[f"{metric_key}_mean"] = mean(values)
            summary_row[f"{metric_key}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary_row)
    return summary_rows
