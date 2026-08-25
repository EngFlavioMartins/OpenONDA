#!/usr/bin/env python3
"""Reproducible qualification and scaling harness for the Neumann panel solver.

The harness covers the production gates that are intentionally too expensive
for the unit-test suite: cold versus reused factorization timings, analytical
sphere convergence, two-body self-convergence, near-contact domain of validity,
and far-field accuracy/performance sweeps.

Examples::

    python scripts/benchmarks/benchmark_panel_solver.py --mode all \
        --output docs/benchmarks/panel-solver-qualification.json
    python scripts/benchmarks/benchmark_panel_solver.py --mode scaling \
        --scaling-total-panels 256 512 1000 2000 --bodies 1 2 4 8 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import tempfile
import time

import numpy as np
import taichi as ti


def _midpoint_on_unit_sphere(
    first: int,
    second: int,
    vertices: list[np.ndarray],
    cache: dict[tuple[int, int], int],
) -> int:
    key = (min(first, second), max(first, second))
    if key not in cache:
        point = vertices[first] + vertices[second]
        vertices.append(point / np.linalg.norm(point))
        cache[key] = len(vertices) - 1
    return cache[key]


def _icosphere_triangles(subdivisions: int, radius: float = 1.0) -> np.ndarray:
    golden = (1.0 + 5.0**0.5) / 2.0
    vertices = np.array(
        [
            [-1, golden, 0],
            [1, golden, 0],
            [-1, -golden, 0],
            [1, -golden, 0],
            [0, -1, golden],
            [0, 1, golden],
            [0, -1, -golden],
            [0, 1, -golden],
            [golden, 0, -1],
            [golden, 0, 1],
            [-golden, 0, -1],
            [-golden, 0, 1],
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    for _ in range(subdivisions):
        vertex_list = list(vertices)
        cache: dict[tuple[int, int], int] = {}
        refined = []
        for first, second, third in faces:
            first_second = _midpoint_on_unit_sphere(first, second, vertex_list, cache)
            second_third = _midpoint_on_unit_sphere(second, third, vertex_list, cache)
            third_first = _midpoint_on_unit_sphere(third, first, vertex_list, cache)
            refined.extend(
                [
                    [first, first_second, third_first],
                    [second, second_third, first_second],
                    [third, third_first, second_third],
                    [first_second, second_third, third_first],
                ]
            )
        vertices = np.asarray(vertex_list)
        faces = refined
    return (radius * vertices)[np.asarray(faces, dtype=np.int64)]


def _fibonacci_sphere_triangles(vertex_count: int, radius: float = 1.0) -> np.ndarray:
    """Near-uniform convex sphere without icosahedral exactness artifacts."""
    from scipy.spatial import ConvexHull

    if vertex_count < 12:
        raise ValueError("vertex_count must be at least 12")
    index = np.arange(vertex_count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / vertex_count
    azimuth = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    vertices = radius * np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))
    faces = ConvexHull(vertices).simplices.copy()
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    inward = np.einsum("ij,ij->i", cross, triangles.mean(axis=1)) < 0.0
    triangles[inward, 1], triangles[inward, 2] = (
        triangles[inward, 2].copy(),
        triangles[inward, 1].copy(),
    )
    return triangles


def _git_identity() -> dict[str, str | bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": True}


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if platform.system() == "Darwin" else value / 1024.0


def _initialize_backend(backend: str, precision: str) -> str:
    from source.solvers.vpm.runtime.backend import initialize_taichi_backend

    return initialize_taichi_backend(
        preferred_backend=backend.upper(),
        debug_mode=False,
        precision=precision,
    )


def _body_centres(body_count: int, spacing: float = 3.0) -> list[np.ndarray]:
    offset = 0.5 * spacing * (body_count - 1)
    return [np.array([index * spacing - offset, 0.0, 0.0]) for index in range(body_count)]


def _mixed_distance_targets(
    centres: list[np.ndarray], count: int, rng: np.random.Generator
) -> np.ndarray:
    """Targets spanning exact and multipole regions around individual bodies."""
    accepted = []
    centre_array = np.asarray(centres)
    while sum(batch.shape[0] for batch in accepted) < count:
        batch_size = max(64, count - sum(batch.shape[0] for batch in accepted))
        anchors = centre_array[np.arange(batch_size) % len(centre_array)]
        directions = rng.normal(size=(batch_size, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        radii = rng.uniform(2.0, 12.0, size=(batch_size, 1))
        candidates = anchors + directions * radii
        distances = np.linalg.norm(candidates[:, None, :] - centre_array[None, :, :], axis=2)
        accepted.append(candidates[np.min(distances, axis=1) > 1.05])
    return np.concatenate(accepted, axis=0)[:count]


def _build_panel_from_triangles(
    root: Path,
    *,
    triangles: np.ndarray,
    mesh_tag: str,
    centres: list[np.ndarray],
    precision: str,
    linear_solver: str = "SCIPY",
    collect_timing: bool = False,
    far_field_acceptance: float = 5.0,
    far_field_min_panels: int = 256,
):
    from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import save_stl
    from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import PanelSolver

    total_panels = len(centres) * triangles.shape[0]
    panel = PanelSolver(
        max_n_panels=total_panels + 8,
        float_dtype=precision,
        linear_solver=linear_solver,
        boundary_condition_type="NEUMANN",
        coupling_scope="vpm_boundary_condition",
        collect_timing=collect_timing,
        far_field_acceptance=far_field_acceptance,
        far_field_min_panels=far_field_min_panels,
    )
    for index, centre in enumerate(centres):
        path = root / f"sphere-{mesh_tag}-{index}-{centre[0]:.8f}.stl"
        save_stl(str(path), triangles + centre)
        panel.add_surface(f"body-{index}", str(path))
    return panel


def _build_panel(
    root: Path,
    *,
    subdivisions: int,
    centres: list[np.ndarray],
    precision: str,
    linear_solver: str = "SCIPY",
    collect_timing: bool = False,
    far_field_acceptance: float = 5.0,
    far_field_min_panels: int = 256,
):
    return _build_panel_from_triangles(
        root,
        triangles=_icosphere_triangles(subdivisions),
        mesh_tag=f"icosphere-{subdivisions}",
        centres=centres,
        precision=precision,
        linear_solver=linear_solver,
        collect_timing=collect_timing,
        far_field_acceptance=far_field_acceptance,
        far_field_min_panels=far_field_min_panels,
    )


def _median_target_seconds(panel, points: np.ndarray, repeats: int) -> tuple[float, np.ndarray]:
    panel.compute_induced_velocity(points)
    ti.sync()
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = panel.compute_induced_velocity(points)
        ti.sync()
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples)), result


def _observed_orders(cases: list[dict], key: str) -> list[float | None]:
    orders: list[float | None] = [None]
    for coarse, fine in zip(cases, cases[1:], strict=False):
        coarse_error = float(coarse[key])
        fine_error = float(fine[key])
        if coarse_error <= 0.0 or fine_error <= 0.0:
            orders.append(None)
        else:
            orders.append(math.log(coarse_error / fine_error) / math.log(coarse["h"] / fine["h"]))
    return orders


def run_scaling(args, root: Path) -> dict:
    cases = []
    rng = np.random.default_rng(20260825)
    specifications = []
    if args.scaling_total_panels:
        for target_total_panels in args.scaling_total_panels:
            for body_count in args.bodies:
                default_max_bodies = {256: 2, 512: 4, 1000: 4, 2000: 8}.get(target_total_panels)
                if default_max_bodies is not None and body_count > default_max_bodies:
                    continue
                target_panels_per_body = max(20, int(round(target_total_panels / body_count)))
                vertex_count = max(12, int(round((target_panels_per_body + 4) / 2)))
                specifications.append(
                    {
                        "mesh": "fibonacci",
                        "target_total_panels": target_total_panels,
                        "vertices_per_body": vertex_count,
                        "subdivisions": None,
                        "bodies": body_count,
                        "triangles": _fibonacci_sphere_triangles(vertex_count),
                    }
                )
    else:
        for subdivisions in args.subdivisions:
            for body_count in args.bodies:
                specifications.append(
                    {
                        "mesh": "icosphere",
                        "target_total_panels": None,
                        "vertices_per_body": None,
                        "subdivisions": subdivisions,
                        "bodies": body_count,
                        "triangles": _icosphere_triangles(subdivisions),
                    }
                )

    for specification in specifications:
        subdivisions = specification["subdivisions"]
        body_count = specification["bodies"]
        triangles = specification["triangles"]
        centres = _body_centres(body_count)
        mesh_tag = (
            f"fibonacci-{specification['vertices_per_body']}"
            if specification["mesh"] == "fibonacci"
            else f"icosphere-{subdivisions}"
        )
        panel = _build_panel_from_triangles(
            root,
            triangles=triangles,
            mesh_tag=mesh_tag,
            centres=centres,
            precision=args.precision,
            collect_timing=True,
            far_field_min_panels=1,
        )
        freestream = np.array([1.0, 0.2, -0.1])
        panel.solve(freestream, None, 0.0)
        cold = panel.results["diagnostic_history"][-1]
        steady_timings = []
        for repeat in range(args.repeats):
            panel.solve((1.0 + 0.05 * (repeat + 1)) * freestream, None, repeat + 1.0)
            steady_timings.append(panel.results["diagnostic_history"][-1]["timings_seconds"])
        median_steady = {
            name: float(statistics.median(item[name] for item in steady_timings))
            for name in steady_timings[0]
        }

        refactor_timings = []
        for repeat in range(args.repeats):
            panel._constrained_factorization = None
            panel._factorization_geometry_revision = -1
            panel.solve((1.2 + 0.05 * repeat) * freestream, None, 10.0 + repeat)
            refactor_timings.append(panel.results["diagnostic_history"][-1]["timings_seconds"])
        median_refactor = {
            name: float(statistics.median(item[name] for item in refactor_timings))
            for name in refactor_timings[0]
        }

        rebuild_totals = []
        for repeat in range(args.repeats):
            panel.initialize(force=True)
            assembly_seconds = panel._last_aic_assembly_seconds
            panel.solve((1.4 + 0.05 * repeat) * freestream, None, 20.0 + repeat)
            solve_seconds = panel.results["diagnostic_history"][-1]["timings_seconds"][
                "total_solve"
            ]
            rebuild_totals.append(assembly_seconds + solve_seconds)

        oracle_velocity = np.array([0.8, -0.3, 0.15])
        panel.solve(oracle_velocity, None, 30.0)
        cpu_oracle_timing = panel.results["diagnostic_history"][-1]["timings_seconds"]
        cpu_oracle_strength = panel.lattice.source_strength.to_numpy()[: panel.lattice.n_panels]

        projected = _build_panel_from_triangles(
            root,
            triangles=triangles,
            mesh_tag=mesh_tag,
            centres=centres,
            precision=args.precision,
            linear_solver="BICGSTAB_GPU",
            collect_timing=True,
            far_field_min_panels=1,
        )
        projected.solve(oracle_velocity, None, 0.0)
        projected_cold = projected.results["diagnostic_history"][-1]
        projected_strength = projected.lattice.source_strength.to_numpy()[
            : projected.lattice.n_panels
        ]
        projected_steady_timings = []
        for repeat in range(args.repeats):
            projected.solve(
                (1.0 + 0.05 * repeat) * oracle_velocity,
                None,
                repeat + 1.0,
            )
            projected_steady_timings.append(
                projected.results["diagnostic_history"][-1]["timings_seconds"]
            )
        median_projected_steady = {
            name: float(statistics.median(item[name] for item in projected_steady_timings))
            for name in projected_steady_timings[0]
        }
        projected.initialize(force=True)
        projected_assembly_seconds = projected._last_aic_assembly_seconds
        projected.solve(oracle_velocity, None, 40.0)
        projected_reassemble_total = (
            projected_assembly_seconds
            + projected.results["diagnostic_history"][-1]["timings_seconds"]["total_solve"]
        )

        extent = max(np.linalg.norm(centre) for centre in centres) + 1.0
        directions = rng.normal(size=(args.targets, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        radii = rng.uniform(6.0 * extent, 12.0 * extent, size=args.targets)
        points = directions * radii[:, None]
        panel.far_field_min_panels = 10**9
        exact_seconds, exact = _median_target_seconds(panel, points, args.repeats)
        panel.far_field_min_panels = 1
        accelerated_seconds, accelerated = _median_target_seconds(panel, points, args.repeats)
        relative_error = float(
            np.linalg.norm(accelerated - exact) / max(np.linalg.norm(exact), 1.0e-30)
        )
        diagnostic = panel.results["diagnostic_history"][-1]
        cases.append(
            {
                "mesh": specification["mesh"],
                "target_total_panels": specification["target_total_panels"],
                "vertices_per_body": specification["vertices_per_body"],
                "subdivisions": subdivisions,
                "bodies": body_count,
                "panels_per_body": triangles.shape[0],
                "total_panels": panel.lattice.n_panels,
                "cold_seconds": cold["timings_seconds"],
                "steady_reused_seconds": median_steady,
                "steady_refactor_seconds": median_refactor,
                "steady_reassemble_refactor_total_seconds": float(
                    statistics.median(rebuild_totals)
                ),
                "cpu_oracle_seconds": cpu_oracle_timing,
                "projected_taichi_cold_seconds": projected_cold["timings_seconds"],
                "projected_taichi_steady_seconds": median_projected_steady,
                "projected_taichi_reassemble_total_seconds": projected_reassemble_total,
                "projected_taichi_iterations": projected_cold["iterations"],
                "projected_taichi_relative_strength_difference": float(
                    np.linalg.norm(projected_strength - cpu_oracle_strength)
                    / max(np.linalg.norm(cpu_oracle_strength), 1.0e-30)
                ),
                "factorization_reuse_speedup": (
                    median_refactor["total_solve"] / max(median_steady["total_solve"], 1.0e-30)
                ),
                "full_static_reuse_speedup": (
                    statistics.median(rebuild_totals) / max(median_steady["total_solve"], 1.0e-30)
                ),
                "active_aic_bytes": diagnostic["active_aic_bytes"],
                "allocated_aic_bytes": diagnostic["allocated_aic_bytes"],
                "factorization_cache_bytes": diagnostic["factorization_cache_bytes"],
                "exact_panel_to_target_seconds": exact_seconds,
                "accelerated_panel_to_target_seconds": accelerated_seconds,
                "far_field_speedup": exact_seconds / max(accelerated_seconds, 1.0e-30),
                "far_field_relative_l2_error": relative_error,
                "far_field_fraction": panel._last_far_field_fraction,
                "peak_process_rss_mb": _peak_rss_mb(),
            }
        )
    return {"cases": cases}


def _sphere_metrics(panel) -> dict[str, float]:
    panel.compute_postprocess(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1.0)
    n = panel.lattice.n_panels
    centre = panel.lattice.panel_centre.to_numpy()[:n]
    velocity = panel.surface_velocity_relative.to_numpy()[:n]
    pressure = panel.lattice.pressure_coefficient.to_numpy()[:n]
    area = panel.lattice.area.to_numpy()[:n]
    radius = np.linalg.norm(centre, axis=1)
    theta = np.arccos(np.clip(centre[:, 0] / radius, -1.0, 1.0))
    analytic_speed = 1.5 * np.sin(theta)
    analytic_pressure = 1.0 - 2.25 * np.sin(theta) ** 2
    speed = np.linalg.norm(velocity, axis=1)
    total_force = panel.panel_force.to_numpy()[:n].sum(axis=0)
    diagnostic = panel.results["diagnostic_history"][-1]
    return {
        "h": float(np.sqrt(np.sum(area) / n)),
        "surface_speed_relative_l2": float(
            np.linalg.norm(speed - analytic_speed) / np.linalg.norm(analytic_speed)
        ),
        "cp_relative_l2": float(
            np.linalg.norm(pressure - analytic_pressure) / np.linalg.norm(analytic_pressure)
        ),
        "wall_relative_rms": float(diagnostic["relative_no_penetration_residual"]),
        "wall_relative_max": float(
            diagnostic["no_penetration_max_residual"] / diagnostic["no_penetration_reference_speed"]
        ),
        "force_coefficient_error": float(np.linalg.norm(total_force) / (0.5 * np.pi)),
        "discrete_equation_residual": float(diagnostic["discrete_equation_residual"]),
        "relative_flux_residual": float(diagnostic["relative_constraint_residual"]),
    }


def run_convergence(args, root: Path) -> dict:
    sphere_cases = []
    multibody_cases = []
    probe_points = np.array(
        [[0.0, 2.5, 0.2], [0.0, -2.7, 0.4], [4.5, 0.3, -0.2], [-4.2, -0.1, 0.5]]
    )
    for vertex_count in args.convergence_vertices:
        triangles = _fibonacci_sphere_triangles(vertex_count)
        sphere = _build_panel_from_triangles(
            root,
            triangles=triangles,
            mesh_tag=f"fibonacci-{vertex_count}",
            centres=[np.zeros(3)],
            precision=args.precision,
        )
        sphere.solve(np.array([1.0, 0.0, 0.0]), None, 0.0)
        sphere_analysis = sphere.analyze_neumann_residual(condition_max_panels=5000)
        sphere_strength = sphere.lattice.source_strength.to_numpy()[: sphere.lattice.n_panels]
        sphere_projected = _build_panel_from_triangles(
            root,
            triangles=triangles,
            mesh_tag=f"fibonacci-{vertex_count}",
            centres=[np.zeros(3)],
            precision=args.precision,
            linear_solver="BICGSTAB_GPU",
        )
        sphere_projected.solve(np.array([1.0, 0.0, 0.0]), None, 0.0)
        sphere_projected_strength = sphere_projected.lattice.source_strength.to_numpy()[
            : sphere_projected.lattice.n_panels
        ]
        sphere_case = {"vertices": vertex_count, "panels": sphere.lattice.n_panels}
        sphere_case.update(_sphere_metrics(sphere))
        sphere_case.update(
            {
                "condition_number_2": sphere_analysis["condition_number_2"],
                "cpu_projected_relative_strength_difference": float(
                    np.linalg.norm(sphere_projected_strength - sphere_strength)
                    / max(np.linalg.norm(sphere_strength), 1.0e-30)
                ),
            }
        )
        sphere_cases.append(sphere_case)

        pair = _build_panel_from_triangles(
            root,
            triangles=triangles,
            mesh_tag=f"fibonacci-{vertex_count}",
            centres=[np.array([-1.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0])],
            precision=args.precision,
        )
        pair.solve(np.array([0.0, 1.0, 0.0]), None, 0.0)
        pair_analysis = pair.analyze_neumann_residual(condition_max_panels=5000)
        pair_strength = pair.lattice.source_strength.to_numpy()[: pair.lattice.n_panels]
        pair_projected = _build_panel_from_triangles(
            root,
            triangles=triangles,
            mesh_tag=f"fibonacci-{vertex_count}",
            centres=[np.array([-1.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0])],
            precision=args.precision,
            linear_solver="BICGSTAB_GPU",
        )
        pair_projected.solve(np.array([0.0, 1.0, 0.0]), None, 0.0)
        pair_projected_strength = pair_projected.lattice.source_strength.to_numpy()[
            : pair_projected.lattice.n_panels
        ]
        areas = pair.lattice.area.to_numpy()[: pair.lattice.n_panels]
        diagnostic = pair.results["diagnostic_history"][-1]
        multibody_cases.append(
            {
                "vertices_per_body": vertex_count,
                "panels": pair.lattice.n_panels,
                "h": float(np.sqrt(np.sum(areas) / pair.lattice.n_panels)),
                "probe_velocity": pair.compute_induced_velocity(probe_points).tolist(),
                "wall_relative_rms": float(diagnostic["relative_no_penetration_residual"]),
                "wall_relative_max": float(
                    diagnostic["no_penetration_max_residual"]
                    / diagnostic["no_penetration_reference_speed"]
                ),
                "discrete_equation_residual": float(diagnostic["discrete_equation_residual"]),
                "relative_flux_residual": float(diagnostic["relative_constraint_residual"]),
                "condition_number_2": pair_analysis["condition_number_2"],
                "cpu_projected_relative_strength_difference": float(
                    np.linalg.norm(pair_projected_strength - pair_strength)
                    / max(np.linalg.norm(pair_strength), 1.0e-30)
                ),
            }
        )

    for key in ("surface_speed_relative_l2", "cp_relative_l2", "force_coefficient_error"):
        orders = _observed_orders(sphere_cases, key)
        for case, order in zip(sphere_cases, orders, strict=True):
            case[f"{key}_observed_order"] = order
    finest_probe = np.asarray(multibody_cases[-1]["probe_velocity"])
    for case in multibody_cases[:-1]:
        probe = np.asarray(case["probe_velocity"])
        case["probe_velocity_self_error"] = float(
            np.linalg.norm(probe - finest_probe) / max(np.linalg.norm(finest_probe), 1.0e-30)
        )
    multibody_cases[-1]["probe_velocity_self_error"] = None
    sphere_speed_monotone = all(
        fine["surface_speed_relative_l2"] < coarse["surface_speed_relative_l2"]
        for coarse, fine in zip(sphere_cases, sphere_cases[1:], strict=False)
    )
    sphere_cp_monotone = all(
        fine["cp_relative_l2"] < coarse["cp_relative_l2"]
        for coarse, fine in zip(sphere_cases, sphere_cases[1:], strict=False)
    )
    pair_errors = [
        case["probe_velocity_self_error"]
        for case in multibody_cases
        if case["probe_velocity_self_error"] is not None
    ]
    pair_probe_monotone = all(
        fine < coarse for coarse, fine in zip(pair_errors, pair_errors[1:], strict=False)
    )
    qualification = {
        "limits": {
            "finest_surface_speed_relative_l2": 0.01,
            "finest_cp_relative_l2": 0.03,
            "finest_force_coefficient_error": 0.001,
            "wall_relative_rms": 1.0e-4,
            "relative_flux_residual": 1.0e-10,
            "cpu_projected_relative_strength_difference": 1.0e-8,
        },
        "sphere_speed_monotone": sphere_speed_monotone,
        "sphere_cp_monotone": sphere_cp_monotone,
        "two_body_probe_self_error_monotone": pair_probe_monotone,
    }
    limits = qualification["limits"]
    finest = sphere_cases[-1]
    qualification["passed"] = bool(
        sphere_speed_monotone
        and sphere_cp_monotone
        and pair_probe_monotone
        and finest["surface_speed_relative_l2"] <= limits["finest_surface_speed_relative_l2"]
        and finest["cp_relative_l2"] <= limits["finest_cp_relative_l2"]
        and finest["force_coefficient_error"] <= limits["finest_force_coefficient_error"]
        and all(
            case["wall_relative_rms"] <= limits["wall_relative_rms"]
            and case["relative_flux_residual"] <= limits["relative_flux_residual"]
            and case["cpu_projected_relative_strength_difference"]
            <= limits["cpu_projected_relative_strength_difference"]
            for case in sphere_cases + multibody_cases
        )
    )
    return {
        "qualification": qualification,
        "sphere": sphere_cases,
        "two_body": multibody_cases,
    }


def run_near_contact(args, root: Path) -> dict:
    cases = []
    cpu_gpu_limit = (
        args.near_contact_cpu_gpu_limit
        if args.near_contact_cpu_gpu_limit is not None
        else (5.0e-4 if args.precision == "f32" else 1.0e-8)
    )
    for subdivisions in args.near_contact_subdivisions:
        triangles = _icosphere_triangles(subdivisions)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area = 0.5 * np.linalg.norm(cross, axis=1).sum()
        h = math.sqrt(area / triangles.shape[0])
        for gap_ratio in args.gap_ratios:
            gap = gap_ratio * h
            centre_distance = 2.0 + gap
            panel = _build_panel(
                root,
                subdivisions=subdivisions,
                centres=[
                    np.array([-0.5 * centre_distance, 0.0, 0.0]),
                    np.array([0.5 * centre_distance, 0.0, 0.0]),
                ],
                precision=args.precision,
            )
            panel.solve(np.array([0.0, 1.0, 0.0]), None, 0.0)
            analysis = panel.analyze_neumann_residual(condition_max_panels=5000)
            strength = panel.lattice.source_strength.to_numpy()[: panel.lattice.n_panels]
            projected = _build_panel(
                root,
                subdivisions=subdivisions,
                centres=[
                    np.array([-0.5 * centre_distance, 0.0, 0.0]),
                    np.array([0.5 * centre_distance, 0.0, 0.0]),
                ],
                precision=args.precision,
                linear_solver="BICGSTAB_GPU",
            )
            projected.solve(np.array([0.0, 1.0, 0.0]), None, 0.0)
            projected_strength = projected.lattice.source_strength.to_numpy()[
                : projected.lattice.n_panels
            ]
            cpu_gpu_difference = float(
                np.linalg.norm(projected_strength - strength)
                / max(np.linalg.norm(strength), 1.0e-30)
            )
            finite = bool(np.all(np.isfinite(strength)))
            passed = bool(
                finite
                and panel.results["diagnostic_history"][-1]["linear_solver_success"]
                and projected.results["diagnostic_history"][-1]["linear_solver_success"]
                and analysis["condition_number_2"] < args.near_contact_condition_limit
                and analysis["relative_no_penetration_residual"] < args.near_contact_wall_limit
                and cpu_gpu_difference < cpu_gpu_limit
            )
            cases.append(
                {
                    "subdivisions": subdivisions,
                    "panels": panel.lattice.n_panels,
                    "h": h,
                    "gap": gap,
                    "gap_over_h": gap_ratio,
                    "condition_number_2": analysis["condition_number_2"],
                    "max_abs_strength": float(np.max(np.abs(strength))),
                    "wall_relative_rms": analysis["relative_no_penetration_residual"],
                    "wall_relative_max": float(
                        analysis["no_penetration_max_residual"]
                        / analysis["no_penetration_reference_speed"]
                    ),
                    "relative_flux_residual": analysis["relative_constraint_residual"],
                    "projected_taichi_iterations": projected.results["diagnostic_history"][-1][
                        "iterations"
                    ],
                    "cpu_projected_relative_strength_difference": cpu_gpu_difference,
                    "all_finite": finite,
                    "passed": passed,
                }
            )
    refinement = []
    qualified_ratios = []
    for gap_ratio in args.gap_ratios:
        matching = [case for case in cases if case["gap_over_h"] == gap_ratio]
        strengths = [case["max_abs_strength"] for case in matching]
        relative_strength_range = (
            (max(strengths) - min(strengths)) / max(strengths) if strengths else float("inf")
        )
        qualified = bool(
            matching
            and all(case["passed"] for case in matching)
            and relative_strength_range <= args.near_contact_refinement_limit
        )
        if qualified:
            qualified_ratios.append(gap_ratio)
        refinement.append(
            {
                "gap_over_h": gap_ratio,
                "resolutions": len(matching),
                "max_strength_relative_range": relative_strength_range,
                "condition_number_ratio": (
                    max(case["condition_number_2"] for case in matching)
                    / min(case["condition_number_2"] for case in matching)
                    if matching
                    else None
                ),
                "qualified": qualified,
            }
        )
    return {
        "condition_limit": args.near_contact_condition_limit,
        "wall_rms_limit": args.near_contact_wall_limit,
        "cpu_projected_strength_difference_limit": cpu_gpu_limit,
        "refinement_strength_range_limit": args.near_contact_refinement_limit,
        "qualified_minimum_gap_over_h": min(qualified_ratios) if qualified_ratios else None,
        "refinement_sensitivity": refinement,
        "cases": cases,
    }


def run_far_field(args, root: Path) -> dict:
    rng = np.random.default_rng(20260825)
    geometries = []
    for name, centres, subdivisions in (
        ("single-sphere", [np.zeros(3)], 2),
        ("four-sphere", _body_centres(4, spacing=3.0), 1),
    ):
        panel = _build_panel(
            root,
            subdivisions=subdivisions,
            centres=centres,
            precision=args.precision,
            far_field_min_panels=1,
        )
        panel.solve(np.array([1.0, 0.2, 0.0]), None, 0.0)
        points = _mixed_distance_targets(centres, args.targets, rng)
        panel.far_field_min_panels = 10**9
        exact_seconds, exact = _median_target_seconds(panel, points, args.repeats)
        sweeps = []
        panel.far_field_min_panels = 1
        for acceptance in args.far_field_acceptance:
            panel.far_field_acceptance = acceptance
            elapsed, accelerated = _median_target_seconds(panel, points, args.repeats)
            difference = accelerated - exact
            sweeps.append(
                {
                    "acceptance": acceptance,
                    "seconds": elapsed,
                    "speedup": exact_seconds / max(elapsed, 1.0e-30),
                    "relative_l2_error": float(
                        np.linalg.norm(difference) / max(np.linalg.norm(exact), 1.0e-30)
                    ),
                    "max_absolute_error": float(np.max(np.linalg.norm(difference, axis=1))),
                    "accelerated_fraction": panel._last_far_field_fraction,
                }
            )
        geometries.append(
            {
                "name": name,
                "bodies": len(centres),
                "panels": panel.lattice.n_panels,
                "targets": len(points),
                "exact_seconds": exact_seconds,
                "sweep": sweeps,
            }
        )
    accepted = []
    for acceptance in args.far_field_acceptance:
        values = [
            item
            for geometry in geometries
            for item in geometry["sweep"]
            if item["acceptance"] == acceptance
        ]
        if all(item["relative_l2_error"] <= args.far_field_error_limit for item in values):
            accepted.append(acceptance)
    return {
        "relative_l2_error_limit": args.far_field_error_limit,
        "recommended_acceptance": min(accepted) if accepted else None,
        "geometries": geometries,
    }


def _qualification_summary(report: dict) -> dict:
    convergence_passed = bool(report["convergence"]["qualification"]["passed"])
    near_contact_limit = report["near_contact"]["qualified_minimum_gap_over_h"]
    near_contact_passed = near_contact_limit is not None and near_contact_limit <= 0.5
    far_field_passed = report["far_field"]["recommended_acceptance"] is not None
    scaling_cases = report["scaling"]["cases"]
    iterative_limit = 5.0e-4 if report["precision"] == "f32" else 1.0e-8
    projected_oracle_passed = bool(
        scaling_cases
        and all(
            case["projected_taichi_relative_strength_difference"] <= iterative_limit
            for case in scaling_cases
        )
    )
    largest = max(scaling_cases, key=lambda case: case["total_panels"])
    cached_fraction = (
        largest["steady_reused_seconds"]["constrained_rhs_solve"]
        / largest["steady_reused_seconds"]["total_solve"]
    )
    production_qualified = bool(
        convergence_passed and near_contact_passed and far_field_passed and projected_oracle_passed
    )
    return {
        "production_qualified": production_qualified,
        "gates": {
            "physical_convergence": convergence_passed,
            "near_contact_domain_quantified": near_contact_passed,
            "far_field_sweep": far_field_passed,
            "projected_taichi_matches_cpu_oracle": projected_oracle_passed,
            "representative_scaling_documented": bool(scaling_cases),
        },
        "supported_domain": {
            "boundary_condition": "NEUMANN",
            "multiple_closed_bodies": True,
            "moving_rigid_bodies": True,
            "minimum_tested_gap_over_h": near_contact_limit,
            "default_far_field_acceptance": report["far_field"]["recommended_acceptance"],
        },
        "solver_decision": {
            "default": "reusable CPU null-space QR",
            "device_option": "projected Taichi CGLS",
            "device_option_numerically_qualified": projected_oracle_passed,
            "timed_backend": report["backend"],
            "largest_case_total_panels": largest["total_panels"],
            "cached_cpu_rhs_solve_fraction": cached_fraction,
            "reason": (
                "Keep reusable CPU QR as the default because its cached RHS solve is not "
                "the dominant static-step cost; use projected CGLS when avoiding a moving-"
                "geometry refactor is beneficial on the selected Taichi backend."
            ),
        },
        "capabilities": {
            "moving_multibody_neumann_boundary_solver": (
                "PRODUCTION QUALIFIED" if production_qualified else "NOT QUALIFIED"
            ),
            "static_potential_flow_pressure_and_loads": (
                "QUALIFIED" if convergence_passed else "NOT QUALIFIED"
            ),
            "moving_or_unsteady_potential_flow_loads": "UNSUPPORTED",
            "general_vpm_coupled_pressure_and_loads": "NOT QUALIFIED",
            "vpm_dirichlet_coupling": "REJECTED",
            "standalone_dirichlet": "EXPERIMENTAL",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "scaling", "convergence", "near-contact", "far-field"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--precision", choices=("f32", "f64"), default="f64")
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--scaling-total-panels", type=int, nargs="+", default=[256, 512, 1000, 2000]
    )
    parser.add_argument("--bodies", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--convergence-vertices", type=int, nargs="+", default=[42, 82, 162, 322])
    parser.add_argument("--near-contact-subdivisions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--gap-ratios", type=float, nargs="+", default=[8.0, 4.0, 2.0, 1.0, 0.5])
    parser.add_argument("--near-contact-condition-limit", type=float, default=1.0e4)
    parser.add_argument("--near-contact-wall-limit", type=float, default=1.0e-3)
    parser.add_argument("--near-contact-cpu-gpu-limit", type=float)
    parser.add_argument("--near-contact-refinement-limit", type=float, default=0.1)
    parser.add_argument(
        "--far-field-acceptance", type=float, nargs="+", default=[2, 3, 4, 5, 6, 8, 10]
    )
    parser.add_argument("--far-field-error-limit", type=float, default=5.0e-4)
    parser.add_argument("--targets", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.targets < 1:
        raise ValueError("repeats and targets must be positive")

    backend = _initialize_backend(args.backend, args.precision)
    report = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": _git_identity(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        },
        "backend": backend,
        "precision": args.precision,
        "qualification_scope": "moving multi-body Neumann boundary solver",
    }
    with tempfile.TemporaryDirectory(prefix="openonda-panel-qualification-") as directory:
        root = Path(directory)
        if args.mode in ("all", "scaling"):
            report["scaling"] = run_scaling(args, root)
        if args.mode in ("all", "convergence"):
            report["convergence"] = run_convergence(args, root)
        if args.mode in ("all", "near-contact"):
            report["near_contact"] = run_near_contact(args, root)
        if args.mode in ("all", "far-field"):
            report["far_field"] = run_far_field(args, root)
        if args.mode == "all":
            report["qualification"] = _qualification_summary(report)

    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
