"""Measure one passive FVM--VPM downstream handoff interval.

This is a diagnostic runner, not a transfer mode.  It restores an atomic
coupled backup, marks the last in-box FVM-derived particle plane with an
otherwise-unused ``group_id``, advances exactly one coupled interval, and
writes the strength budgets surrounding the next absolute replacement.  The
marker is metadata: neither the velocity, vortex strength, positions, nor
particle ownership rules are changed.

The marker is preserved by GBD's existing dominant-contributor group-id
propagation.  Accordingly, the ``tagged`` post-GBD budgets identify the
regular-grid representation owned by that population; the separately recorded
pre-GBD snapshot is the exact advected particle population.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import setup as case  # noqa: E402
import openonda.coupler as coupling  # noqa: E402
import openonda.fvm as fvm  # noqa: E402
import openonda.vpm as vpm  # noqa: E402
from source.coupler.vorticity_transfer import replacement_eta  # noqa: E402


_TAGGED_RELEASE_GROUP = 2_147_483_000


def _summarize_strength(position: np.ndarray, strength: np.ndarray, mask: np.ndarray) -> dict:
    selected_position = position[mask]
    selected_strength = strength[mask]
    return {
        "n_particles": int(mask.sum()),
        "gamma_net": selected_strength.sum(axis=0, dtype=np.float64).tolist(),
        "gamma_l1": float(np.linalg.norm(selected_strength, axis=1).sum(dtype=np.float64)),
        "x_min": None if len(selected_position) == 0 else float(selected_position[:, 0].min()),
        "x_max": None if len(selected_position) == 0 else float(selected_position[:, 0].max()),
    }


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {name: None for name in ("min", "p05", "p50", "p95", "max", "mean")}
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _particle_arrays(solver) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    particles = solver.particles
    return (
        np.asarray(particles.position_cpu(), dtype=np.float64),
        np.asarray(particles.vortex_strength_cpu(), dtype=np.float64),
        np.asarray(particles.velocity_cpu(), dtype=np.float64),
        np.asarray(particles.group_id_cpu(), dtype=np.int32),
    )


def _region_budgets(
    solver,
    *,
    transfer_box: np.ndarray,
    spacing: float,
    only_tagged: bool = False,
) -> dict:
    position, strength, _velocity, group_id = _particle_arrays(solver)
    tagged = group_id == _TAGGED_RELEASE_GROUP
    if only_tagged:
        base = tagged
    else:
        base = np.ones(len(position), dtype=bool)
    x = position[:, 0]
    xmax = float(transfer_box[1])
    face_tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(xmax))
    eta = replacement_eta(position, transfer_box, 0.0)
    deleted = eta >= 1.0
    outside = eta == 0.0
    return {
        "all": _summarize_strength(position, strength, base),
        "last_two_h_inside": _summarize_strength(
            position, strength, base & (x > xmax - 2.0 * spacing) & (x < xmax)
        ),
        "last_h_inside": _summarize_strength(
            position, strength, base & (x > xmax - spacing) & (x < xmax)
        ),
        "on_face": _summarize_strength(
            position, strength, base & (np.abs(x - xmax) <= face_tolerance)
        ),
        "outside": _summarize_strength(position, strength, base & outside),
        "deleted_by_hard_replacement": _summarize_strength(position, strength, base & deleted),
        "survives_hard_replacement": _summarize_strength(position, strength, base & ~deleted),
        "tagged_particles": int(tagged.sum()),
    }


def _write_group_ids(solver, group_id: np.ndarray) -> None:
    """Replace only active group-id metadata without touching physics fields."""
    particles = solver.particles
    active = int(particles.n_particles_total)
    all_groups = particles.group_id.to_numpy()
    all_groups[:active] = group_id
    particles.group_id.from_numpy(all_groups)


def _donor_geometry(transfer, spacing: float) -> dict:
    assert transfer._box is not None
    assert transfer._cell_centre is not None
    assert transfer._cell_volume is not None
    fluid = ~np.asarray(transfer._fvm_solid_mask, dtype=bool)
    eta = replacement_eta(transfer._cell_centre, transfer._box, transfer.eta_blend_width)
    donors = (eta > 0.0) & fluid
    x = transfer._cell_centre[:, 0]
    last_x = float(x[donors].max())
    distances = float(transfer._box[1]) - x[donors]
    layer_distance = np.unique(np.round(distances, 12))[:4]
    layers = []
    for distance in layer_distance:
        layer = donors & np.isclose(float(transfer._box[1]) - x, distance, atol=1.0e-10)
        sizes = np.cbrt(transfer._cell_volume[layer])
        layers.append(
            {
                "distance_to_face": float(distance),
                "n_cells": int(layer.sum()),
                "cell_size_min": float(sizes.min()),
                "cell_size_max": float(sizes.max()),
                "cell_size_median": float(np.median(sizes)),
            }
        )
    anchor = np.asarray(transfer._lattice_anchor, dtype=np.float64)
    n = np.rint((float(transfer._box[1]) - anchor[0]) / spacing).astype(np.int64)
    nearest = anchor[0] + np.arange(n - 2, n + 4) * spacing
    return {
        "x_transfer_face": float(transfer._box[1]),
        "max_donor_x": last_x,
        "distance_face_to_max_donor_x": float(transfer._box[1] - last_x),
        "last_donor_layers": layers,
        "lattice_anchor": anchor.tolist(),
        "lattice_planes_near_downstream_face": nearest.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-from", type=Path, required=True)
    parser.add_argument("--case-directory", type=Path, required=True)
    parser.add_argument("--gbd-threshold-scale", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.gbd_threshold_scale <= 0.0:
        raise ValueError("gbd threshold scale must be positive")

    seed_manifest = json.loads((arguments.restart_from / "manifest.json").read_text())
    seed_time = float(seed_manifest["time"])
    end_time = seed_time + case.VPM_TIME_STEP_SIZE
    case.CASE_DIR = arguments.case_directory.resolve()
    case.CASE_DIR.mkdir(parents=True, exist_ok=True)
    case.FVM_SETUP = replace(case.FVM_SETUP, time=replace(case.FVM_SETUP.time, end_time=end_time))
    case.VPM_CASE = replace(
        case.VPM_CASE,
        numerics=replace(
            case.VPM_CASE.numerics,
            viscous=replace(
                case.VPM_CASE.numerics.viscous,
                gbd_threshold=(
                    case.GBD_VORTICITY_FLOOR
                    * case.VPM_PARTICLE_SPACING**3
                    * arguments.gbd_threshold_scale
                ),
            ),
        ),
        output=replace(
            case.VPM_CASE.output,
            backup=replace(case.VPM_CASE.output.backup, directory="solution"),
            logging=replace(case.VPM_CASE.output.logging, directory="solution"),
        ),
        run=replace(case.VPM_CASE.run, steps=round(end_time / case.VPM_TIME_STEP_SIZE)),
        directory=case.CASE_DIR,
    )

    fvm_solver = fvm.create_fvm_solver(case.FVM_SETUP, case_dir=case.CASE_DIR, mesh=case.FVM_MESH)
    vpm_solver = vpm.VPMSolver(case.VPM_CASE)
    coupler = coupling.create_coupler(fvm_solver, vpm_solver, case.COUPLER_SETUP)
    coupler.initialize()
    # The FVM backup deliberately keeps a strict configuration digest.  A
    # one-step diagnostic necessarily changes only ``end_time``; allow that
    # single restart-time difference while the coupler-level configuration,
    # mesh, numerical methods, and VPM settings retain their exact seed values.
    load_fvm_state = coupler.fvm_solver.load_state

    def load_fvm_for_one_step(path):
        return load_fvm_state(path, allow_config_change=True)

    coupler.fvm_solver.load_state = load_fvm_for_one_step
    try:
        start_step = coupler.load_backup(arguments.restart_from.resolve())
    finally:
        coupler.fvm_solver.load_state = load_fvm_state

    if not coupler._is_master:
        coupler.solve(start_step=start_step)
        return

    assert coupler.vorticity_transfer is not None
    transfer = coupler.vorticity_transfer
    assert transfer._box is not None
    spacing = float(coupler.vpm_particle_spacing)
    report: dict = {
        "seed_time": seed_time,
        "vpm_step_size": float(coupler.vpm_time_step_size),
        "gbd_threshold_scale": float(arguments.gbd_threshold_scale),
        "fvm_restart_config_override": "end_time only",
        "donor_geometry": _donor_geometry(transfer, spacing),
    }

    position, _strength, _velocity, group_id = _particle_arrays(vpm_solver)
    release_mask = (position[:, 0] > float(transfer._box[1]) - spacing) & (
        position[:, 0] < float(transfer._box[1])
    )
    reference_indices = np.flatnonzero(release_mask)
    reference_position = position[reference_indices].copy()
    group_id[release_mask] = _TAGGED_RELEASE_GROUP
    _write_group_ids(vpm_solver, group_id)
    report["post_injection_release_slab"] = _region_budgets(
        vpm_solver, transfer_box=transfer._box, spacing=spacing, only_tagged=True
    )

    original_gbd = vpm_solver.physics.gbd_diffusion

    def record_pre_gbd(particles, time_step_size, *args, **kwargs):
        if time_step_size > 0.0 and "after_advection_before_gbd" not in report:
            p, _g, velocity, ids = _particle_arrays(vpm_solver)
            tagged_index = np.flatnonzero(ids == _TAGGED_RELEASE_GROUP)
            report["after_advection_before_gbd"] = _region_budgets(
                vpm_solver, transfer_box=transfer._box, spacing=spacing, only_tagged=True
            )
            if np.array_equal(tagged_index, reference_indices):
                displacement = p[tagged_index, 0] - reference_position[:, 0]
                report["tagged_advection"] = {
                    "normal_displacement_x": _percentiles(displacement),
                    "streamwise_velocity_x": _percentiles(velocity[tagged_index, 0]),
                    "required_distance_to_leave": _percentiles(
                        float(transfer._box[1]) - reference_position[:, 0]
                    ),
                }
            else:
                report["tagged_advection"] = {
                    "warning": "particle topology changed before the physical GBD call",
                    "tagged_count": int(len(tagged_index)),
                    "reference_count": int(len(reference_indices)),
                }
        return original_gbd(particles, time_step_size, *args, **kwargs)

    vpm_solver.physics.gbd_diffusion = record_pre_gbd
    original_transfer = transfer.transfer

    def record_pre_replacement(vpm_state, velocity, velocity_gradient):
        report["before_next_replacement"] = _region_budgets(
            vpm_state, transfer_box=transfer._box, spacing=spacing, only_tagged=False
        )
        report["before_next_replacement_tagged"] = _region_budgets(
            vpm_state, transfer_box=transfer._box, spacing=spacing, only_tagged=True
        )
        result = original_transfer(vpm_state, velocity, velocity_gradient)
        report["after_next_replacement"] = _region_budgets(
            vpm_state, transfer_box=transfer._box, spacing=spacing, only_tagged=False
        )
        report["after_next_replacement_tagged"] = _region_budgets(
            vpm_state, transfer_box=transfer._box, spacing=spacing, only_tagged=True
        )
        report["replacement_result"] = {
            "n_particles_removed": result.n_particles_removed,
            "n_particles_injected": result.n_particles_injected,
            "replaced_gamma_net": result.replaced_vortex_strength_net.tolist(),
            "replaced_gamma_l1": result.replaced_vortex_strength_l1,
            "injected_gamma_net": result.injected_vortex_strength_net.tolist(),
            "injected_gamma_l1": result.injected_vortex_strength_l1,
        }
        return result

    transfer.transfer = record_pre_replacement
    coupler.solve(start_step=start_step)
    output = case.CASE_DIR / "wake_handoff_diagnostic.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wake-handoff diagnostic written to {output}")


if __name__ == "__main__":
    main()
