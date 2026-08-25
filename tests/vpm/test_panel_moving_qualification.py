"""Adversarial qualification tests for moving NEUMANN panel bodies."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

from test_panel_solver_sphere_analytic import _icosphere_triangles  # noqa: E402

from source.solvers.vpm.boundary_elements.panels.coupling.kinematics import (  # noqa: E402
    CompositePanel,
    ManeuverPanel,
    RotatingPanel,
    StaticPanel,
    TranslatingPanel,
)
from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import save_stl  # noqa: E402
from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import (  # noqa: E402
    PanelSolver,
)


def _ensure_taichi_cpu() -> None:
    if taichi.lang.impl.get_runtime().prog is None:
        taichi.init(arch=taichi.cpu)


def _write_sphere(
    tmp_path,
    name: str,
    offset=(0.0, 0.0, 0.0),
    subdivisions: int = 1,
    scale=(1.0, 1.0, 1.0),
) -> str:
    path = tmp_path / f"{name}.stl"
    save_stl(
        str(path),
        _icosphere_triangles(subdivisions) * np.asarray(scale, dtype=np.float64)
        + np.asarray(offset, dtype=np.float64),
    )
    return str(path)


def _panel(max_n_panels: int, linear_solver: str = "SCIPY", **kwargs) -> PanelSolver:
    _ensure_taichi_cpu()
    return PanelSolver(
        max_n_panels=max_n_panels,
        float_dtype="f64",
        linear_solver=linear_solver,
        boundary_condition_type="NEUMANN",
        coupling_scope="vpm_boundary_condition",
        **kwargs,
    )


def test_translating_sphere_is_galilean_invariant(tmp_path):
    path = _write_sphere(tmp_path, "sphere")
    freestream = np.array([1.0, -0.25, 0.1])

    stationary = _panel(100)
    stationary.add_surface("sphere", path)
    stationary.solve(freestream, None, 0.0)

    moving = _panel(100)
    moving.add_surface("sphere", path, kinematics=TranslatingPanel(-freestream))
    moving.advance(freestream_velocity=np.zeros(3), time=0.0, time_step_size=0.1)

    n = stationary.lattice.n_panels
    np.testing.assert_allclose(
        moving.lattice.source_strength.to_numpy()[:n],
        stationary.lattice.source_strength.to_numpy()[:n],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        moving.surface_velocity_relative.to_numpy()[:n],
        stationary.surface_velocity_relative.to_numpy()[:n],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert moving.results["diagnostic_history"][-1]["no_penetration_residual"] < 1.0e-10


def test_vpm_incident_field_is_distinct_from_body_velocity_and_matches_freestream(tmp_path):
    path = _write_sphere(tmp_path, "sphere")
    incident_velocity = np.array([0.7, -0.2, 0.15])

    freestream = _panel(100)
    freestream.add_surface("sphere", path)
    freestream.solve(incident_velocity, None, 0.0)

    from_vpm = _panel(100)
    from_vpm.add_surface("sphere", path)
    incident = taichi.Vector.field(3, dtype=taichi.f64, shape=100)
    incident.from_numpy(np.tile(incident_velocity, (100, 1)))
    from_vpm.solve(np.zeros(3), incident, 0.0)

    n = freestream.lattice.n_panels
    np.testing.assert_allclose(
        from_vpm.lattice.source_strength.to_numpy()[:n],
        freestream.lattice.source_strength.to_numpy()[:n],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        from_vpm.surface_velocity_relative.to_numpy()[:n],
        freestream.surface_velocity_relative.to_numpy()[:n],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_rotation_about_sphere_centre_is_tangential_to_the_exact_sphere(tmp_path):
    path = _write_sphere(tmp_path, "sphere")
    panel = _panel(100)
    panel.add_surface(
        "sphere",
        path,
        kinematics=RotatingPanel(axis=(0.0, 0.0, 1.0), angular_speed=2.0),
    )
    panel.advance(freestream_velocity=np.zeros(3), time=0.0, time_step_size=0.1)

    n = panel.lattice.n_panels
    centre = panel.lattice.panel_centre.to_numpy()[:n]
    radial_normal = centre / np.linalg.norm(centre, axis=1)[:, None]
    body_velocity = panel.lattice.body_velocity.to_numpy()[:n]
    # A triangulated sphere has planar normals, so it can require a small
    # correction for its faceted geometry.  The physical sphere identity is
    # instead tested against the exact radial normal.
    assert np.max(np.abs(np.einsum("ij,ij->i", body_velocity, radial_normal))) < 1.0e-12
    assert panel.results["diagnostic_history"][-1]["no_penetration_residual"] < 1.0e-5


def test_maneuver_uses_one_composed_pose_and_velocity(tmp_path):
    path = _write_sphere(tmp_path, "sphere")
    centre = np.array([0.4, -0.3, 0.2])
    translation = np.array([0.1, -0.2, 0.03])
    panel = _panel(100)
    panel.add_surface(
        "sphere",
        path,
        kinematics=ManeuverPanel(
            translation=TranslatingPanel(translation / 0.1),
            rotation=RotatingPanel(axis=(0.0, 0.0, 1.0), angular_speed=1.0, rotation_centre=centre),
        ),
    )
    reference = panel.lattice.local_vertex_position.to_numpy()[:80].copy()

    panel.advance(freestream_velocity=np.zeros(3), time=0.0, time_step_size=0.1)

    angle = 0.1
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    actual_vertex = panel.lattice.vertex_position.to_numpy()[:80]
    expected_vertex = (reference - centre) @ rotation.T + centre + translation
    np.testing.assert_allclose(actual_vertex, expected_vertex, rtol=0.0, atol=1.0e-12)

    panel_centre = panel.lattice.panel_centre.to_numpy()[:80]
    expected_velocity = translation / 0.1 + np.cross(
        np.array([0.0, 0.0, 1.0]), panel_centre - (centre + translation)
    )
    np.testing.assert_allclose(
        panel.lattice.body_velocity.to_numpy()[:80], expected_velocity, rtol=0.0, atol=1.0e-12
    )


def test_static_component_does_not_reset_a_composed_pose(tmp_path):
    path = _write_sphere(tmp_path, "sphere")
    velocity = np.array([0.3, -0.2, 0.1])
    panel = _panel(100)
    panel.add_surface(
        "sphere",
        path,
        kinematics=CompositePanel([TranslatingPanel(velocity), StaticPanel()]),
    )

    panel.advance(freestream_velocity=np.zeros(3), time=0.0, time_step_size=0.1)

    np.testing.assert_allclose(
        panel.get_body_pose((0, 80)).translation,
        0.1 * velocity,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_moving_body_force_postprocessing_is_rejected(tmp_path):
    path = _write_sphere(tmp_path, "sphere")
    panel = _panel(100)
    panel.add_surface("sphere", path, kinematics=TranslatingPanel((1.0, 0.0, 0.0)))
    panel.advance(freestream_velocity=np.zeros(3), time=0.0, time_step_size=0.1)

    with pytest.raises(NotImplementedError, match="unsteady dphi/dt"):
        panel.compute_postprocess(np.zeros(3), np.array([1.0, 0.0, 0.0]), density=1.225)
    with pytest.raises(NotImplementedError, match="force coefficients"):
        panel.compute_forces_coefficients(1.225, np.array([1.0, 0.0, 0.0]))


def test_moved_multibody_operator_matches_fresh_rebuild(tmp_path):
    path_a = _write_sphere(tmp_path, "a")
    path_b = _write_sphere(tmp_path, "b", offset=(3.0, 0.0, 0.0))
    freestream = np.array([1.0, 0.2, 0.0])

    production = _panel(200)
    production.add_surface("A", path_a)
    production.add_surface("B", path_b, kinematics=TranslatingPanel((1.0, 0.0, 0.0)))
    production.solve(freestream, None, 0.0)
    production.advance(freestream_velocity=freestream, time=0.0, time_step_size=0.1)

    fresh = _panel(200)
    fresh.add_surface("A", path_a)
    fresh.add_surface("B", path_b)
    fresh.apply_translation_update(np.array([0.1, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), (80, 160))
    fresh.solve(freestream, None, 0.1)

    np.testing.assert_allclose(
        production.aerodynamic_influence_coefficient.to_numpy()[:160, :160],
        fresh.aerodynamic_influence_coefficient.to_numpy()[:160, :160],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        production.lattice.source_strength.to_numpy()[:160],
        fresh.lattice.source_strength.to_numpy()[:160],
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_rotated_unequal_three_body_blocks_refresh_and_match_fresh_solver(tmp_path):
    paths = {
        "A": _write_sphere(tmp_path, "a", offset=(-3.0, 0.0, 0.0), scale=(0.8, 1.0, 1.2)),
        "B": _write_sphere(tmp_path, "b", offset=(0.0, 0.0, 0.0), scale=(1.2, 0.7, 1.0)),
        "C": _write_sphere(tmp_path, "c", offset=(3.0, 0.0, 0.0), scale=(0.9, 1.1, 0.8)),
    }
    centre = np.array([1.0, 0.0, 0.0])
    angular_speed = 0.4
    time_step = 0.2
    freestream = np.array([0.9, 0.2, -0.1])

    production = _panel(260)
    production.add_surface("A", paths["A"])
    production.add_surface(
        "B",
        paths["B"],
        kinematics=RotatingPanel(
            axis=(0.0, 0.0, 1.0),
            angular_speed=angular_speed,
            rotation_centre=centre,
        ),
    )
    production.add_surface("C", paths["C"])
    production.solve(freestream, None, 0.0)
    before = production.aerodynamic_influence_coefficient.to_numpy()[:240, :240].copy()
    production.advance(
        freestream_velocity=freestream,
        time=0.0,
        time_step_size=time_step,
    )
    after = production.aerodynamic_influence_coefficient.to_numpy()[:240, :240]

    # Rigid self-blocks and the untouched A-C interaction remain invariant;
    # every cross-block touching the moved body must be rebuilt.
    np.testing.assert_allclose(after[:80, :80], before[:80, :80], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(after[80:160, 80:160], before[80:160, 80:160], atol=1.0e-12)
    np.testing.assert_allclose(after[160:, 160:], before[160:, 160:], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(after[:80, 160:], before[:80, 160:], rtol=0.0, atol=1.0e-12)
    assert not np.allclose(after[:80, 80:160], before[:80, 80:160])
    assert not np.allclose(after[80:160, 160:], before[80:160, 160:])

    angle = angular_speed * time_step
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    fresh = _panel(260)
    for uid in ("A", "B", "C"):
        fresh.add_surface(uid, paths[uid])
    fresh.apply_rotation_update(
        rotation,
        np.array([0.0, 0.0, angular_speed]),
        centre,
        (80, 160),
    )
    fresh.solve(freestream, None, time_step)

    np.testing.assert_allclose(
        after,
        fresh.aerodynamic_influence_coefficient.to_numpy()[:240, :240],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        production.lattice.source_strength.to_numpy()[:240],
        fresh.lattice.source_strength.to_numpy()[:240],
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_static_geometry_reuses_factorization_and_motion_invalidates_it(tmp_path):
    path = _write_sphere(tmp_path, "sphere")
    panel = _panel(100, collect_timing=True)
    panel.add_surface("sphere", path)

    panel.solve(np.array([1.0, 0.0, 0.0]), None, 0.0)
    first = panel.results["diagnostic_history"][-1]
    assert not first["factorization_reused"]
    assert first["factorization_cache_bytes"] > 0
    assert first["aic_rebuilt"]
    assert first["timings_seconds"]["aic_assembly"] > 0.0
    assert first["timings_seconds"]["constraint_factorization"] > 0.0

    panel.solve(np.array([0.5, 0.2, 0.0]), None, 0.1)
    second = panel.results["diagnostic_history"][-1]
    assert second["factorization_reused"]
    assert not second["aic_rebuilt"]
    assert second["timings_seconds"]["aic_assembly"] == 0.0
    assert second["timings_seconds"]["constraint_factorization"] == 0.0

    panel.apply_translation_update(np.array([0.1, 0.0, 0.0]), np.zeros(3), (0, 80))
    panel.solve(np.array([0.5, 0.2, 0.0]), None, 0.2)
    third = panel.results["diagnostic_history"][-1]
    assert not third["factorization_reused"]


def test_constrained_neumann_satisfies_flux_and_wall_together(tmp_path):
    path_a = _write_sphere(tmp_path, "a")
    path_b = _write_sphere(tmp_path, "b", offset=(3.0, 0.0, 0.0))
    panel = _panel(200)
    panel.add_surface("A", path_a)
    panel.add_surface("B", path_b)
    panel.solve(np.array([1.0, 0.2, 0.0]), None, 0.0)

    diagnostic = panel.results["diagnostic_history"][-1]
    assert diagnostic["residual"] < 1.0e-5
    assert diagnostic["constraint_residual"] < 1.0e-12
    assert diagnostic["no_penetration_residual"] < 1.0e-5
    assert max(abs(value) for value in diagnostic["net_source_flux"].values()) < 1.0e-12


def test_projected_gpu_multibody_solve_matches_cpu_oracle(tmp_path):
    path_a = _write_sphere(tmp_path, "a")
    path_b = _write_sphere(tmp_path, "b", offset=(3.0, 0.0, 0.0))
    freestream = np.array([1.0, 0.2, -0.1])

    cpu = _panel(200)
    cpu.add_surface("A", path_a)
    cpu.add_surface("B", path_b)
    cpu.solve(freestream, None, 0.0)

    gpu = _panel(200, linear_solver="BICGSTAB_GPU")
    gpu.add_surface("A", path_a)
    gpu.add_surface("B", path_b)
    gpu.solve(freestream, None, 0.0)

    n = cpu.lattice.n_panels
    relative_difference = np.linalg.norm(
        gpu.lattice.source_strength.to_numpy()[:n] - cpu.lattice.source_strength.to_numpy()[:n]
    ) / np.linalg.norm(cpu.lattice.source_strength.to_numpy()[:n])
    assert relative_difference < 1.0e-9
    diagnostic = gpu.results["diagnostic_history"][-1]
    assert diagnostic["linear_solver"] == "ProjectedCGLS(Taichi)"
    assert diagnostic["relative_constraint_residual"] < 1.0e-10
    assert diagnostic["projected_optimality_residual"] < 1.0e-10


def test_symmetric_two_body_strength_pressure_force_and_moment_identities(tmp_path):
    paths = {
        "A": _write_sphere(tmp_path, "a", offset=(-1.5, 0.0, 0.0)),
        "B": _write_sphere(tmp_path, "b", offset=(1.5, 0.0, 0.0)),
    }
    freestream = np.array([0.0, 1.0, 0.0])
    panel = _panel(200)
    for uid in ("A", "B"):
        panel.add_surface(uid, paths[uid])
    panel.solve(freestream, None, 0.0)
    panel.compute_postprocess(freestream, freestream, density=1.0)

    strength = panel.lattice.source_strength.to_numpy()[:160]
    pressure = panel.lattice.pressure_coefficient.to_numpy()[:160]
    np.testing.assert_allclose(np.sort(strength[:80]), np.sort(strength[80:]), atol=1.0e-12)
    np.testing.assert_allclose(np.sort(pressure[:80]), np.sort(pressure[80:]), atol=1.0e-12)

    forces = panel.compute_per_surface_forces(1.0, freestream)
    assert forces["A"]["force_x"] == pytest.approx(-forces["B"]["force_x"], abs=1.0e-12)
    assert forces["A"]["force_y"] == pytest.approx(forces["B"]["force_y"], abs=1.0e-12)
    assert forces["A"]["force_z"] == pytest.approx(forces["B"]["force_z"], abs=1.0e-12)
    assert forces["A"]["moment_x"] == pytest.approx(forces["B"]["moment_x"], abs=1.0e-12)
    assert forces["A"]["moment_y"] == pytest.approx(-forces["B"]["moment_y"], abs=1.0e-12)
    assert forces["A"]["moment_z"] == pytest.approx(-forces["B"]["moment_z"], abs=1.0e-12)


def test_three_body_insertion_permutations_are_uid_invariant(tmp_path):
    paths = {
        "A": _write_sphere(tmp_path, "a", offset=(-3.0, 0.0, 0.0), scale=(0.8, 1.0, 1.1)),
        "B": _write_sphere(tmp_path, "b", offset=(0.0, 0.0, 0.0), scale=(1.0, 0.9, 1.2)),
        "C": _write_sphere(tmp_path, "c", offset=(3.0, 0.0, 0.0), scale=(1.1, 0.8, 0.9)),
    }
    freestream = np.array([0.8, -0.25, 0.1])

    def solve_order(order):
        panel = _panel(260)
        for uid in order:
            panel.add_surface(uid, paths[uid])
        panel.solve(freestream, None, 0.0)
        panel.compute_postprocess(freestream, freestream, density=1.0)
        strength = panel.lattice.source_strength.to_numpy()[:240]
        pressure = panel.lattice.pressure_coefficient.to_numpy()[:240]
        state = {}
        for body in panel.lattice.bodies:
            body_slice = slice(body.start_idx, body.start_idx + body.count)
            state[body.uid] = {
                "vortex_strength": strength[body_slice].copy(),
                "pressure": pressure[body_slice].copy(),
            }
        return state, panel.compute_per_surface_forces(1.0, freestream)

    reference_state, reference_forces = solve_order(("A", "B", "C"))
    for order in itertools.permutations(("A", "B", "C")):
        state, forces = solve_order(order)
        for uid in ("A", "B", "C"):
            np.testing.assert_allclose(
                state[uid]["vortex_strength"],
                reference_state[uid]["vortex_strength"],
                atol=2.0e-11,
            )
            np.testing.assert_allclose(
                state[uid]["pressure"], reference_state[uid]["pressure"], atol=2.0e-11
            )
            for key in ("force_x", "force_y", "force_z", "moment_x", "moment_y", "moment_z"):
                assert forces[uid][key] == pytest.approx(reference_forces[uid][key], abs=2.0e-11)


def test_neumann_residual_analysis_separates_algebraic_flux_and_wall_metrics(tmp_path):
    path_a = _write_sphere(tmp_path, "a")
    path_b = _write_sphere(tmp_path, "b", offset=(3.0, 0.0, 0.0))
    panel = _panel(200)
    panel.add_surface("A", path_a)
    panel.add_surface("B", path_b)
    panel.solve(np.array([1.0, 0.2, 0.0]), None, 0.0)

    analysis = panel.analyze_neumann_residual()
    diagnostic = panel.results["diagnostic_history"][-1]

    assert analysis["algebraic_residual"] < 1.0e-5
    assert analysis["constraint_residual"] < 1.0e-12
    assert analysis["unconstrained_algebraic_residual"] < 1.0e-10
    assert np.isfinite(analysis["condition_number_2"])
    assert analysis["condition_number_2"] > 1.0
    assert analysis["relative_no_penetration_residual"] == pytest.approx(
        analysis["no_penetration_residual"] / analysis["no_penetration_reference_speed"]
    )
    for name, value in analysis.items():
        assert diagnostic[name] == value


def test_multibody_constraint_compatibility_converges_under_refinement(tmp_path):
    algebraic_residuals = []
    wall_residuals = []
    condition_numbers = []
    convergence_success = []

    for subdivisions in (0, 1, 2):
        panels_per_body = 20 * 4**subdivisions
        panel = _panel(2 * panels_per_body, raise_on_non_convergence=False)
        panel.add_surface(
            "A", _write_sphere(tmp_path, f"a-{subdivisions}", subdivisions=subdivisions)
        )
        panel.add_surface(
            "B",
            _write_sphere(
                tmp_path,
                f"b-{subdivisions}",
                offset=(3.0, 0.0, 0.0),
                subdivisions=subdivisions,
            ),
        )
        panel.solve(np.array([1.0, 0.2, 0.0]), None, 0.0)
        analysis = panel.analyze_neumann_residual(condition_max_panels=1000)

        algebraic_residuals.append(analysis["algebraic_residual"])
        wall_residuals.append(analysis["relative_no_penetration_residual"])
        condition_numbers.append(analysis["condition_number_2"])
        convergence_success.append(panel.results["diagnostic_history"][-1]["linear_solver_success"])
        assert analysis["constraint_residual"] < 1.0e-12
        assert analysis["unconstrained_constraint_residual"] > 1.0e-7

    # The coarse-mesh residual is a compatibility error introduced by the
    # exact per-body flux constraint, not an ill-conditioned solve: it falls
    # rapidly under uniform refinement while the operator remains well
    # conditioned.
    assert algebraic_residuals[1] < algebraic_residuals[0] * 0.15
    assert algebraic_residuals[2] < algebraic_residuals[1] * 0.15
    assert wall_residuals[1] < wall_residuals[0] * 0.15
    assert wall_residuals[2] < wall_residuals[1] * 0.15
    assert max(condition_numbers) < 3.0
    # At the coarsest level r_A exceeds the old equation-residual threshold,
    # yet it is a legitimate constrained optimum: flux and projected KKT
    # optimality, not r_A, define constrained-solver convergence.
    assert algebraic_residuals[0] > 1.0e-5
    assert all(convergence_success)


def test_dirichlet_vpm_refresh_is_rejected_before_velocity_sampling(tmp_path):
    panel = PanelSolver(
        max_n_panels=100,
        float_dtype="f64",
        linear_solver="SCIPY",
        boundary_condition_type="DIRICHLET",
    )
    with pytest.raises(NotImplementedError, match="DIRICHLET panel coupling"):
        panel.refresh_coupled_solution(
            particles=object(),
            physics=object(),
            freestream_velocity=np.array([1.0, 0.0, 0.0]),
            time=0.0,
        )
