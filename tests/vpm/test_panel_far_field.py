"""Regression tests for per-body source monopole/dipole evaluation."""

from __future__ import annotations

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

from test_panel_solver_sphere_analytic import _icosphere_triangles  # noqa: E402

from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import save_stl  # noqa: E402
from source.solvers.vpm.boundary_elements.panels.kernels.far_field import (  # noqa: E402
    PanelFarFieldBody,
    evaluate_source_far_field,
)
from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import (  # noqa: E402
    PanelSolver,
)


def _ensure_taichi_cpu() -> None:
    if taichi.lang.impl.get_runtime().prog is None:
        taichi.init(arch=taichi.cpu)


def _solved_sphere(tmp_path, minimum: int) -> PanelSolver:
    _ensure_taichi_cpu()
    triangles = _icosphere_triangles(2)
    path = tmp_path / f"sphere_{minimum}.stl"
    save_stl(str(path), triangles)
    panel = PanelSolver(
        max_n_panels=400,
        float_dtype="f32",
        linear_solver="SCIPY",
        boundary_condition_type="NEUMANN",
        far_field_min_panels=minimum,
    )
    panel.add_surface("sphere", str(path))
    panel.solve(np.array([1.0, 0.0, 0.0]), None, 0.0)
    return panel


def test_far_field_matches_direct_sum_beyond_acceptance_radius(tmp_path):
    far = _solved_sphere(tmp_path, minimum=256)
    exact = _solved_sphere(tmp_path, minimum=10_000)
    points = np.array([[8.0, 0.4, 0.2], [16.0, -0.2, 0.5]], dtype=np.float32)

    np.testing.assert_allclose(
        far.compute_induced_velocity(points),
        exact.compute_induced_velocity(points),
        rtol=5e-4,
        atol=2e-6,
    )
    assert far.results["diagnostic_history"][-1]["far_field_target_fraction"] == 1.0


def test_below_threshold_uses_exact_path(tmp_path):
    panel = _solved_sphere(tmp_path, minimum=10_000)
    points = np.array([[2.0, 0.2, 0.1]], dtype=np.float32)
    result = panel.compute_induced_velocity(points)
    assert panel.results["diagnostic_history"][-1]["far_field_target_fraction"] == 0.0
    assert np.all(np.isfinite(result))


def test_monopole_is_retained_for_nonzero_net_flux():
    body = PanelFarFieldBody(
        uid="body",
        start_idx=0,
        count=300,
        centre=np.zeros(3),
        radius=1.0,
        monopole=2.0,
        dipole=np.zeros(3),
    )
    np.testing.assert_allclose(
        evaluate_source_far_field(np.array([4.0, 0.0, 0.0]), body),
        np.array([2.0 / (4.0 * np.pi * 16.0), 0.0, 0.0]),
    )
