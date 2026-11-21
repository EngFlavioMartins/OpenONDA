"""Named statistics of a run and their comparison with another run.

A summary maps each statistic name to ``{"value", "standard_error", "unit"}``.
The comparison of two summaries lists, for every statistic of both, the two
values, their difference and the combined standard error
``sqrt(se_a^2 + se_b^2)``; it gives no verdict.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def statistic(value, standard_error=float("nan"), unit: str = "1", **extra) -> dict[str, object]:
    """One named statistic: its value, its standard error from the cycle scatter and its unit.

    ``extra`` adds named fields, such as the frequency resolution of a spectral
    peak (which has no cycle-scatter standard error) or a bootstrap interval.
    """
    return {"value": float(value), "standard_error": float(standard_error), "unit": unit, **extra}


def compare(first: dict, second: dict) -> dict[str, dict[str, object]]:
    """Differences ``second - first`` of every statistic present in both summaries."""
    rows = {}
    for name in sorted(set(first) & set(second)):
        a, b = first[name], second[name]
        if not (isinstance(a, dict) and "value" in a and isinstance(b, dict) and "value" in b):
            continue
        se_a, se_b = float(a["standard_error"]), float(b["standard_error"])
        combined = math.sqrt(se_a**2 + se_b**2) if math.isfinite(se_a + se_b) else float("nan")
        rows[name] = {
            "first": float(a["value"]),
            "second": float(b["value"]),
            "difference": float(b["value"]) - float(a["value"]),
            "combined_standard_error": combined,
            "unit": a["unit"],
        }
    return rows


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def write_json(path: Path, data) -> Path:
    """Write ``data`` as indented JSON, NaN as ``null``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(data), indent=2, sort_keys=False) + "\n")
    return path
