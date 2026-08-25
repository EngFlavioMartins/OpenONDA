"""Regression coverage for panel body ranges and declarative multi-body setup."""

from __future__ import annotations

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

from test_panel_solver_sphere_analytic import _icosphere_triangles  # noqa: E402

from source.solvers.vpm.boundary_elements.panels.coupling.kinematics import (  # noqa: E402
    TranslatingPanel,
)
from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import save_stl  # noqa: E402
from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import (  # noqa: E402
    PanelSolver,
)
from source.solvers.vpm.config import PanelBodySetup, VPMSetup  # noqa: E402


def _ensure_taichi_cpu() -> None:
    if taichi.lang.impl.get_runtime().prog is None:
        taichi.init(arch=taichi.cpu)


def _write_body(tmp_path, name: str, offset: float = 0.0) -> str:
    triangles = _icosphere_triangles(1) + np.array([offset, 0.0, 0.0])
    path = tmp_path / f"{name}.stl"
    save_stl(str(path), triangles)
    return str(path)


def test_appended_bodies_keep_geometry_ranges_and_group_ids(tmp_path):
    _ensure_taichi_cpu()
    path_a = _write_body(tmp_path, "a")
    path_b = _write_body(tmp_path, "b", offset=6.0)
    panel = PanelSolver(max_n_panels=200, float_dtype="f32", linear_solver="SCIPY")

    panel.add_surface("A", path_a, group_id=3)
    panel.apply_translation_update(np.array([0.5, 0.0, 0.0]), np.zeros(3), (0, 80))
    first_body_after_move = panel.lattice.vertex_position.to_numpy()[:80].copy()
    panel.add_surface("B", path_b, group_id=7)

    assert panel.lattice.n_panels == 160
    assert [(body.uid, body.start_idx, body.count) for body in panel.lattice.bodies] == [
        ("A", 0, 80),
        ("B", 80, 80),
    ]
    np.testing.assert_array_equal(
        panel.lattice.vertex_position.to_numpy()[:80], first_body_after_move
    )
    np.testing.assert_array_equal(
        panel.lattice.group_id.to_numpy()[:160],
        np.r_[np.full(80, 3, dtype=np.int32), np.full(80, 7, dtype=np.int32)],
    )
    assert np.mean(panel.lattice.panel_centre.to_numpy()[80:160], axis=0)[0] > 5.0


def test_each_body_can_move_independently(tmp_path):
    _ensure_taichi_cpu()
    panel = PanelSolver(max_n_panels=200, float_dtype="f32", linear_solver="SCIPY")
    panel.add_surface("A", _write_body(tmp_path, "a"), kinematics=TranslatingPanel([1, 0, 0]))
    panel.add_surface(
        "B", _write_body(tmp_path, "b", offset=6), kinematics=TranslatingPanel([0, 1, 0])
    )
    before = panel.lattice.panel_centre.to_numpy()[:160].copy()

    panel.advance(time=0.0, step=0, time_step_size=0.1, freestream_velocity=np.zeros(3))
    after = panel.lattice.panel_centre.to_numpy()[:160]

    np.testing.assert_allclose(after[:80, 0] - before[:80, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(after[80:, 1] - before[80:, 1], 0.1, atol=1e-6)
    np.testing.assert_allclose(after[:80, 1], before[:80, 1], atol=1e-6)
    np.testing.assert_allclose(after[80:, 0], before[80:, 0], atol=1e-6)


def test_mutual_influence_changes_two_body_solution(tmp_path):
    _ensure_taichi_cpu()
    path_a = _write_body(tmp_path, "a")
    path_b = _write_body(tmp_path, "b", offset=3.0)
    velocity = np.array([1.0, 0.2, 0.0])

    combined = PanelSolver(
        max_n_panels=200,
        float_dtype="f32",
        linear_solver="SCIPY",
        boundary_condition_type="NEUMANN",
    )
    combined.add_surface("A", path_a)
    combined.add_surface("B", path_b)
    combined.solve(velocity, None, 0.0)
    combined.compute_postprocess(velocity, velocity, 1.0)
    per_body = combined.compute_per_surface_forces(1.0, velocity)
    assert set(per_body) == {"A", "B"}
    assert per_body["A"]["panel_count"] == per_body["B"]["panel_count"] == 80
    assert per_body["B"]["reference_point"][0] > per_body["A"]["reference_point"][0]

    independent = PanelSolver(
        max_n_panels=100,
        float_dtype="f32",
        linear_solver="SCIPY",
        boundary_condition_type="NEUMANN",
    )
    independent.add_surface("A", path_a)
    independent.solve(velocity, None, 0.0)

    assert not np.allclose(
        combined.lattice.source_strength.to_numpy()[:80],
        independent.lattice.source_strength.to_numpy()[:80],
        rtol=1e-5,
        atol=1e-6,
    )


def test_declarative_bodies_reject_duplicate_uids_and_round_trip():
    body = PanelBodySetup(stl="body.stl", uid="body", translation=(1, 2, 3))
    setup = VPMSetup(bodies=(body,))
    restored = VPMSetup.from_dict(setup.to_dict())
    assert restored.bodies == (body,)
    with pytest.raises(ValueError, match="Duplicate panel body uid"):
        VPMSetup(bodies=(body, body))
