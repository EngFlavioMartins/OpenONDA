"""Fail-fast dense influence-matrix memory budget for the panel solver."""

from __future__ import annotations

import pytest
import taichi

from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import (
    PanelSolver,  # noqa: E402
)


def test_oversized_panel_count_is_rejected_before_any_allocation():
    solver = PanelSolver(max_n_panels=200_000, float_dtype="f32", memory_budget_bytes=10_000_000)
    with pytest.raises(RuntimeError, match="memory_budget_bytes"):
        solver._ensure_initialized()
    assert solver.lattice is None


def test_normal_panel_count_is_allowed():
    if taichi.lang.impl.get_runtime().prog is None:
        taichi.init(arch=taichi.cpu)
    solver = PanelSolver(max_n_panels=16, float_dtype="f32", linear_solver="SCIPY")
    solver._ensure_initialized()
    assert solver.lattice is not None
