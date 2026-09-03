#!/usr/bin/env python3
"""Process-isolated VLM/FVM production-FMM qualification cases.

Usage::

    python coupled_qualification.py vlm FMM VULKAN 100
    python coupled_qualification.py vlm TREECODE VULKAN 100
    python coupled_qualification.py compare-vlm
    python coupled_qualification.py fvm FMM CPU 10
    python coupled_qualification.py fvm TREECODE CPU 10
    python coupled_qualification.py compare-fvm
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

import openonda.vpm as vpm

STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
REPOSITORY = STUDY_DIR.parents[2]


def _induction(name: str):
    normalized = name.upper()
    if normalized == "FMM":
        return vpm.FMMInduction()
    if normalized in {"TREE", "TREECODE"}:
        return vpm.TreecodeInduction()
    raise ValueError(f"unknown induction method {name!r}")


def _diagnostic(induction, name: str, default=0):
    diagnostics = induction.diagnostics
    if isinstance(diagnostics, dict):
        return diagnostics.get(name, default)
    return getattr(diagnostics, name, default)


def load_manifest_source_commit() -> str:
    manifest = json.loads((RESULTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise RuntimeError("study manifest has no non-empty source_commit")
    if manifest.get("source_dirty") is not False:
        raise RuntimeError("study manifest source_dirty must be false")
    return source_commit


def _required_fmm_diagnostic(induction, name: str):
    diagnostics = induction.diagnostics
    if isinstance(diagnostics, dict):
        if name not in diagnostics:
            raise RuntimeError(f"FMM study diagnostic {name!r} is unavailable")
        value = diagnostics[name]
    else:
        if not hasattr(diagnostics, name):
            raise RuntimeError(f"FMM study diagnostic {name!r} is unavailable")
        value = getattr(diagnostics, name)
    if value is None:
        raise RuntimeError(f"FMM study diagnostic {name!r} is unavailable")
    return value


def run_vlm(method: str, backend: str, steps: int) -> None:
    """Run a stationary maintained delta-wing surface with a shedding wake."""
    method = "TREECODE" if method.upper() in {"TREE", "TREECODE"} else "FMM"
    surface_file = (
        REPOSITORY / "tutorials" / "vpm" / "delta_wing" / "assets" / "delta_wing_surface.json"
    )
    if not surface_file.is_file():
        raise FileNotFoundError(surface_file)
    vlm_setup = vpm.VLMSetup(
        surfaces=(vpm.VLMSurfaceSetup(str(surface_file), name="delta_wing"),),
        mesh=vpm.VLMMeshSetup.geometric(ratio=3.0, region="end"),
        kinematic_viscosity=1.0e-3,
        density=1.225,
        sample_surface_forces=True,
        logging_interval_steps=steps + 1,
    )
    with tempfile.TemporaryDirectory(prefix=f"openonda-vlm-{method.lower()}-") as directory:
        case = vpm.VPMCase(
            numerics=vpm.Numerics(
                time_step_size=0.0025,
                compute_device=backend.upper(),
                max_n_particles=20_000,
                max_evaluation_points=20_000,
                integrator=vpm.SSPRK3(),
                vlm=vlm_setup,
                freestream_velocity=[-5.0, 0.0, 0.0],
                viscous=vpm.ViscousConfig.cs(
                    kinematic_viscosity=1.0e-3,
                    particle_spacing=0.04,
                ),
                induction=_induction(method),
                particle_kernel="GAUSSIAN",
                verbose=False,
            ),
            backup=vpm.Backup(interval_steps=steps),
            directory=Path(directory),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            solver = vpm.VPMSolver(case)
        history = []
        start = time.perf_counter()
        for step in range(1, steps + 1):
            with contextlib.redirect_stdout(io.StringIO()):
                solver.advance()
            circulation = solver.vlm_solver.lattice.circulation.to_numpy()[
                : solver.vlm_solver.lattice.n_panels
            ]
            forces = solver.vlm_solver._last_forces or {}
            particle_count = solver.particles.n_particles_total
            position = solver.particle_position
            centroid = position.mean(axis=0) if particle_count else np.zeros(3, dtype=np.float64)
            history.append(
                {
                    "step": step,
                    "circulation": circulation.astype(float).tolist(),
                    "force_coefficient": [
                        float(forces.get("force_coefficient_x", 0.0)),
                        float(forces.get("force_coefficient_y", 0.0)),
                        float(forces.get("force_coefficient_z", 0.0)),
                    ],
                    "wake_particle_count": particle_count,
                    "wake_centroid": centroid.astype(float).tolist(),
                    "raw_rate_defect": float(
                        _diagnostic(solver.induction, "last_relative_rate_defect", 0.0)
                    ),
                }
            )
        elapsed = time.perf_counter() - start
        result = {
            "source_commit": load_manifest_source_commit(),
            "source_dirty": False,
            "method": method,
            "backend": backend.upper(),
            "steps": steps,
            "integrator": "SSPRK3",
            "coupling_policy": "lagged accepted-step VLM solve; stage-position sampling; advection-only",
            "elapsed_seconds": elapsed,
            "final_wake_particle_count": solver.particles.n_particles_total,
            "stage_evaluations": int(_diagnostic(solver.induction, "stage_evaluations", 0)),
            "scheduled_backup_written": (
                Path(directory) / f"solution/vpm_{steps:06d}.h5"
            ).is_file(),
            "scheduled_visualization_written": (
                Path(directory) / f"solution/vpm_{steps:06d}.xdmf"
            ).is_file(),
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "history": history,
        }
        if method == "FMM":
            result.update(
                {
                    "host_particle_transfers": int(
                        _required_fmm_diagnostic(solver.induction, "host_particle_transfers")
                    ),
                    "direct_strength_rate_fallbacks": int(
                        _required_fmm_diagnostic(solver.induction, "direct_strength_rate_fallbacks")
                    ),
                }
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"coupled_vlm_{method.lower()}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


def compare_vlm() -> None:
    fmm = json.loads((RESULTS_DIR / "coupled_vlm_fmm.json").read_text(encoding="utf-8"))
    tree = json.loads((RESULTS_DIR / "coupled_vlm_treecode.json").read_text(encoding="utf-8"))
    fmm_circulation = np.asarray([row["circulation"] for row in fmm["history"]])
    tree_circulation = np.asarray([row["circulation"] for row in tree["history"]])
    fmm_force = np.asarray([row["force_coefficient"] for row in fmm["history"]])
    tree_force = np.asarray([row["force_coefficient"] for row in tree["history"]])
    fmm_centroid = np.asarray(fmm["history"][-1]["wake_centroid"])
    tree_centroid = np.asarray(tree["history"][-1]["wake_centroid"])
    circulation_difference = float(
        np.linalg.norm(fmm_circulation - tree_circulation)
        / max(np.linalg.norm(tree_circulation), np.finfo(float).eps)
    )
    integrated_force_difference = float(
        np.linalg.norm(fmm_force.sum(axis=0) - tree_force.sum(axis=0))
        / max(np.linalg.norm(tree_force.sum(axis=0)), np.finfo(float).eps)
    )
    result = {
        "source_commit": load_manifest_source_commit(),
        "source_dirty": False,
        "bound_circulation_history_relative_difference": circulation_difference,
        "integrated_force_coefficient_relative_difference": integrated_force_difference,
        "final_wake_centroid_distance": float(np.linalg.norm(fmm_centroid - tree_centroid)),
        "wake_centroid_tolerance": 0.1,
        "fmm_maximum_rate_defect": float(max(row["raw_rate_defect"] for row in fmm["history"])),
        "circulation_gate_passed": circulation_difference <= 0.02,
        "force_gate_passed": integrated_force_difference <= 0.02,
        "centroid_gate_passed": float(np.linalg.norm(fmm_centroid - tree_centroid)) <= 0.1,
        "fmm_wake_insertion_passed": fmm["final_wake_particle_count"] > 0,
        "fmm_zero_host_transfer_passed": fmm["host_particle_transfers"] == 0,
        "fmm_zero_fallback_passed": fmm["direct_strength_rate_fallbacks"] == 0,
        "fmm_scheduled_output_passed": bool(
            fmm["scheduled_backup_written"] and fmm["scheduled_visualization_written"]
        ),
    }
    result["comparison_gate_passed"] = bool(
        result["circulation_gate_passed"]
        and result["force_gate_passed"]
        and result["centroid_gate_passed"]
        and result["fmm_wake_insertion_passed"]
        and result["fmm_zero_host_transfer_passed"]
        and result["fmm_zero_fallback_passed"]
        and result["fmm_scheduled_output_passed"]
        and result["fmm_maximum_rate_defect"] <= 1.0e-3
    )
    (RESULTS_DIR / "coupled_vlm_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


def run_fvm(method: str, backend: str, cycles: int) -> None:
    """Run a small vortical native FVM/VPM exchange for complete coupling cycles."""
    from source.coupler import CouplerSetup, FVMVPMCoupler
    from source.solvers.fvm import (
        BoundaryConfig,
        FVMSetup,
        FVMSolver,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh

    method = "TREECODE" if method.upper() in {"TREE", "TREECODE"} else "FMM"
    fvm_dt = 0.025
    vpm_dt = 0.05
    spacing = 0.1
    setup = CouplerSetup(
        freestream_velocity=[1.0, 0.0, 0.0],
        eta_blend_width=0.0,
        backup_interval_steps=cycles,
    )
    with tempfile.TemporaryDirectory(prefix=f"openonda-fvm-{method.lower()}-") as directory:
        case_dir = Path(directory)
        fvm_config = FVMSetup(
            case_name="fmm_coupled_qualification",
            time=TimeConfig(time_step_size=fvm_dt, end_time=cycles * vpm_dt),
            transport=TransportConfig(kinematic_viscosity=0.01),
            boundaries=[
                BoundaryConfig(
                    name="numericalBoundary",
                    velocity_type="fixedValue",
                    velocity_value=setup.freestream_velocity,
                    pressure_type="fixedFluxPressure",
                )
            ],
            initial_velocity=setup.freestream_velocity,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            fvm_solver = FVMSolver(
                fvm_config,
                case_dir=case_dir,
                mesh_data=coupling_box_mesh(
                    (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
                    spacing,
                ),
            )
        n_cells = int(fvm_solver.mesh_data["n_cells"])
        centres = np.asarray(fvm_solver.geo_data["cell_centre"])[:n_cells]
        x, y, z = centres.T
        amplitude = 0.02
        cos_x, cos_y, cos_z = np.cos(np.pi * x), np.cos(np.pi * y), np.cos(np.pi * z)
        fvm_solver.velocity[:n_cells, 0] = 1.0 - (
            2.0 * np.pi * amplitude * cos_x**2 * cos_y * np.sin(np.pi * y) * cos_z**2
        )
        fvm_solver.velocity[:n_cells, 1] = (
            2.0 * np.pi * amplitude * cos_x * np.sin(np.pi * x) * cos_y**2 * cos_z**2
        )
        fvm_solver.velocity_old[:] = fvm_solver.velocity
        fvm_solver.velocity_older[:] = fvm_solver.velocity
        from source.solvers.fvm.assemble import convection

        fvm_solver.volumetric_face_flux[:] = convection.compute_volumetric_face_flux(
            fvm_solver.velocity,
            fvm_solver.mesh_data,
            fvm_solver.geo_data,
        )
        fvm_solver.volumetric_face_flux_old[:] = fvm_solver.volumetric_face_flux
        fvm_solver.volumetric_face_flux_older[:] = fvm_solver.volumetric_face_flux
        vpm_case = vpm.VPMCase(
            numerics=vpm.Numerics(
                time_step_size=vpm_dt,
                compute_device=backend.upper(),
                max_n_particles=50_000,
                max_evaluation_points=50_000,
                integrator=vpm.SSPRK3(),
                freestream_velocity=(1.0, 0.0, 0.0),
                viscous=vpm.ViscousConfig.cs(
                    kinematic_viscosity=0.01,
                    particle_spacing=spacing,
                ),
                induction=_induction(method),
                particle_kernel="GAUSSIAN",
                verbose=False,
            ),
            backup=vpm.Backup(interval_steps=0),
            directory=case_dir,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            vpm_solver = vpm.VPMSolver(vpm_case)
            coupler = FVMVPMCoupler(fvm_solver, vpm_solver, setup)
            coupler.initialize()
        counts = []
        removed_counts = []
        injected_counts = []
        start = time.perf_counter()
        stop = 0
        for _ in range(cycles):
            with contextlib.redirect_stdout(io.StringIO()):
                stop = coupler.solve(
                    start_step=stop,
                    max_coupling_steps=1,
                    backup_at_stop=False,
                )
            counts.append(vpm_solver.particles.n_particles_total)
            transfer_result = coupler._last_transfer_result
            removed_counts.append(int(transfer_result.n_particles_removed))
            injected_counts.append(int(transfer_result.n_particles_injected))
        elapsed = time.perf_counter() - start
        particle_count = vpm_solver.particles.n_particles_total
        result = {
            "source_commit": load_manifest_source_commit(),
            "source_dirty": False,
            "method": method,
            "backend": backend.upper(),
            "cycles": cycles,
            "elapsed_seconds": elapsed,
            "particle_counts": counts,
            "removed_particle_counts": removed_counts,
            "injected_particle_counts": injected_counts,
            "final_particle_count": particle_count,
            "insertion_observed": max(injected_counts, default=0) > 0,
            "replacement_or_removal_observed": max(removed_counts, default=0) > 0,
            "coupled_backup_written": (case_dir / "solution/backups/manifest.json").is_file(),
            "scheduled_particle_output_written": (
                case_dir / f"solution/vpm_{cycles:06d}.h5"
            ).is_file(),
            "finite_fvm_velocity": bool(np.isfinite(fvm_solver.velocity).all()),
            "finite_vpm_position": bool(np.isfinite(vpm_solver.particle_position).all()),
            "stage_evaluations": int(_diagnostic(vpm_solver.induction, "stage_evaluations", 0)),
            "final_fvm_velocity": np.asarray(fvm_solver.velocity).astype(float).tolist(),
            "final_vpm_position": vpm_solver.particle_position.astype(float).tolist(),
            "final_vpm_strength": vpm_solver.particle_vortex_strength.astype(float).tolist(),
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        if method == "FMM":
            result.update(
                {
                    "host_particle_transfers": int(
                        _required_fmm_diagnostic(vpm_solver.induction, "host_particle_transfers")
                    ),
                    "direct_strength_rate_fallbacks": int(
                        _required_fmm_diagnostic(
                            vpm_solver.induction, "direct_strength_rate_fallbacks"
                        )
                    ),
                }
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"coupled_fvm_{method.lower()}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if not key.startswith("final_")}, indent=2
        )
    )


def compare_fvm() -> None:
    fmm = json.loads((RESULTS_DIR / "coupled_fvm_fmm.json").read_text(encoding="utf-8"))
    tree = json.loads((RESULTS_DIR / "coupled_fvm_treecode.json").read_text(encoding="utf-8"))
    fmm_velocity = np.asarray(fmm["final_fvm_velocity"])
    tree_velocity = np.asarray(tree["final_fvm_velocity"])
    fmm_position = np.asarray(fmm["final_vpm_position"])
    tree_position = np.asarray(tree["final_vpm_position"])
    fmm_strength = np.asarray(fmm["final_vpm_strength"])
    tree_strength = np.asarray(tree["final_vpm_strength"])
    velocity_difference = float(
        np.linalg.norm(fmm_velocity - tree_velocity)
        / max(np.linalg.norm(tree_velocity), np.finfo(float).eps)
    )
    position_difference = float(
        np.linalg.norm(fmm_position - tree_position)
        / max(np.linalg.norm(tree_position), np.finfo(float).eps)
    )
    strength_difference = float(
        np.linalg.norm(fmm_strength - tree_strength)
        / max(np.linalg.norm(tree_strength), np.finfo(float).eps)
    )
    comparison_tolerance = 0.02
    result = {
        "source_commit": load_manifest_source_commit(),
        "source_dirty": False,
        "cycles": fmm["cycles"],
        "relative_difference_tolerance": comparison_tolerance,
        "fvm_velocity_relative_difference": velocity_difference,
        "vpm_position_relative_difference": position_difference,
        "vpm_strength_relative_difference": strength_difference,
        "fmm_final_particle_count": fmm["final_particle_count"],
        "treecode_final_particle_count": tree["final_particle_count"],
        "fmm_insertion_observed": fmm["insertion_observed"],
        "fmm_replacement_or_removal_observed": fmm["replacement_or_removal_observed"],
        "fmm_coupled_backup_written": fmm["coupled_backup_written"],
        "fmm_scheduled_particle_output_written": fmm["scheduled_particle_output_written"],
        "fmm_host_particle_transfers": fmm["host_particle_transfers"],
        "fmm_direct_strength_rate_fallbacks": fmm["direct_strength_rate_fallbacks"],
        "finite_fields": bool(
            fmm["finite_fvm_velocity"]
            and fmm["finite_vpm_position"]
            and tree["finite_fvm_velocity"]
            and tree["finite_vpm_position"]
        ),
        "comparison_gate_passed": bool(
            velocity_difference <= comparison_tolerance
            and position_difference <= comparison_tolerance
            and strength_difference <= comparison_tolerance
            and fmm["final_particle_count"] == tree["final_particle_count"]
            and fmm["host_particle_transfers"] == 0
            and fmm["direct_strength_rate_fallbacks"] == 0
        ),
    }
    (RESULTS_DIR / "coupled_fvm_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "vlm" and len(sys.argv) == 5:
        run_vlm(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return
    if command == "compare-vlm" and len(sys.argv) == 2:
        compare_vlm()
        return
    if command == "fvm" and len(sys.argv) == 5:
        run_fvm(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return
    if command == "compare-fvm" and len(sys.argv) == 2:
        compare_fvm()
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
