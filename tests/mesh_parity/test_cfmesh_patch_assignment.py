"""Patch-assignment weights remain finite for faces exactly on a surface."""

import json
import subprocess
import sys

import numpy as np
import pytest

from source.solvers.fvm.mesh.cartesian.cfmesh_template import _patch_alignment_weight

_ALIGNMENT_CASES = (
    ("25.0", "0.0", "1.0"),
    ("25.0", "0.0", "0.0"),
    ("25.0", "1.0e-310", "0.5"),
    ("25.0", "1.0e-300", "0.25"),
    ("1.7976931348623157e308", "0.0", "1.0"),
    ("1.7976931348623157e308", "1.7976931348623157e308", "1.0"),
    ("0.0", "0.0", "1.0"),
    ("5e-324", "5e-324", "0.5"),
    ("9.0", "4.0", "0.75"),
)


@pytest.fixture(scope="module")
def ieee_alignment_weights():
    """Evaluate all extreme inputs before a device runtime can alter host FP mode.

    Taichi can leave flush-to-zero enabled after reset. A fresh process keeps
    the smallest-subnormal case independent of previously executed GPU/CPU
    tests, while evaluating the same production function and Decimal oracle.
    Decimal strings also keep parameter construction independent of FP mode.
    """
    script = """
import json
import sys
from decimal import Decimal, localcontext
import numpy as np
from source.solvers.fvm.mesh.cartesian.cfmesh_template import _patch_alignment_weight

rows = []
for maximum_squared, distance_squared, alignment in json.loads(sys.argv[1]):
    maximum_squared, distance_squared, alignment = map(
        float, (maximum_squared, distance_squared, alignment)
    )
    with localcontext() as context:
        context.prec = 100
        denominator = max(distance_squared, float(np.finfo(np.float64).tiny))
        expected = float(
            (Decimal.from_float(maximum_squared) / Decimal.from_float(denominator)).sqrt()
            * Decimal.from_float(alignment)
        )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        weight = _patch_alignment_weight(maximum_squared, distance_squared, alignment)
    rows.append([weight, expected])
print(json.dumps(rows, allow_nan=False))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(_ALIGNMENT_CASES)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout.splitlines()[-1])


@pytest.mark.parametrize(
    "case_index",
    range(len(_ALIGNMENT_CASES)),
    ids=[
        "zero_distance",
        "perpendicular",
        "subnormal_distance",
        "small_distance",
        "largest_numerator",
        "largest_distances",
        "zero_distances",
        "smallest_subnormal",
        "ordinary_distances",
    ],
)
def test_alignment_weight_matches_high_precision_without_overflow(
    case_index, ieee_alignment_weights
):
    weight, expected = ieee_alignment_weights[case_index]
    assert np.isfinite(weight)
    assert weight == pytest.approx(expected, rel=4.0e-16, abs=0.0)


def test_zero_distance_with_perpendicular_normal_does_not_poison_patch_selection():
    candidates = [(0, 0.0, 0.0), (1, 0.0, 0.5), (2, 25.0, 1.0)]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        best = max(candidates, key=lambda item: _patch_alignment_weight(25.0, item[1], item[2]))
    assert best[0] == 1


def test_all_zero_distances_retain_native_candidate_order_tie():
    candidates = [(3, 0.0, 0.0), (1, 0.0, 1.0)]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        best = max(candidates, key=lambda item: _patch_alignment_weight(0.0, item[1], item[2]))
    assert best[0] == 3
