"""Run the experimental L1 conservative-lattice transfer from a coupled seed.

L1 is deliberately isolated from the production transfer.  Its FVM authority
box remains the baseline ``[-1.25, 1.25]^3``.  M4' support is allowed to reach
the next VPM plane outside that box; those support particles are updated in
place when present, never removed by the hard ownership operation.
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

import cube_flow_setup as case  # noqa: E402
import openonda.coupler as coupling  # noqa: E402
import openonda.fvm as fvm  # noqa: E402
import openonda.vpm as vpm  # noqa: E402
from source.coupler.lattice_transfer import map_cell_circulation_to_lattice  # noqa: E402
from source.coupler.vorticity_transfer import TransferResult, replacement_eta  # noqa: E402


_L1_RELEASE_GROUP = 2_147_483_001


def _strength_record(position: np.ndarray, strength: np.ndarray, mask: np.ndarray) -> dict:
    selected = strength[mask]
    return {
        "n_particles": int(mask.sum()),
        "gamma_net": selected.sum(axis=0, dtype=np.float64).tolist(),
        "gamma_l1": float(np.linalg.norm(selected, axis=1).sum(dtype=np.float64)),
        "vortex_strength_max": float(np.linalg.norm(selected, axis=1).max(initial=0.0)),
        "x_min": None if not mask.any() else float(position[mask, 0].min()),
        "x_max": None if not mask.any() else float(position[mask, 0].max()),
    }


def _particle_arrays(solver) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    particles = solver.particles
    return (
        np.asarray(particles.position_cpu(), dtype=np.float64),
        np.asarray(particles.vortex_strength_cpu(), dtype=np.float64),
        np.asarray(particles.core_radius_cpu(), dtype=np.float64),
    )


def _node_indices(position: np.ndarray, anchor: np.ndarray, spacing: float) -> np.ndarray:
    return np.rint((position - anchor) / spacing).astype(np.int64)


def _write_group_ids(vpm_solver, group_id: np.ndarray) -> None:
    """Replace active particle metadata without changing physics fields."""
    particles = vpm_solver.particles
    active = int(particles.n_particles_total)
    stored = particles.group_id.to_numpy()
    stored[:active] = group_id
    particles.group_id.from_numpy(stored)


def _tagged_release_record(vpm_solver, box: np.ndarray) -> dict:
    position, strength, _radius = _particle_arrays(vpm_solver)
    group_id = np.asarray(vpm_solver.particles.group_id_cpu(), dtype=np.int32)
    tagged = group_id == _L1_RELEASE_GROUP
    deleted = replacement_eta(position, box, 0.0) >= 1.0
    return {
        "all": _strength_record(position, strength, tagged),
        "downstream_vpm_authority": _strength_record(position, strength, tagged & ~deleted),
        "would_be_deleted_by_baseline_hard_replacement": _strength_record(
            position, strength, tagged & deleted
        ),
    }


def _replace_with_l1(
    vpm_solver,
    transfer,
    fvm_vorticity: np.ndarray,
    *,
    tag_release: bool = False,
) -> tuple[TransferResult, dict]:
    """Apply absolute FVM state on a complete M4' target lattice.

    Only particles in the FVM-authoritative box are removed.  The one-cell M4'
    support outside each face is a representation/release zone: a coincident
    regular VPM node is set to the absolute FVM target strength in place;
    non-coincident particles are untouched and remain VPM-owned.
    """
    assert transfer._box is not None
    assert transfer._cell_centre is not None
    assert transfer._cell_volume is not None
    assert transfer._fvm_solid_mask is not None
    assert transfer._lattice_anchor is not None

    box = transfer._box
    h = float(vpm_solver._viscous_config.gbd_grid_spacing)
    anchor = np.asarray(transfer._lattice_anchor, dtype=np.float64)
    donor = (replacement_eta(transfer._cell_centre, box, 0.0) > 0.0) & ~transfer._fvm_solid_mask
    lattice = map_cell_circulation_to_lattice(
        transfer._cell_centre[donor],
        transfer._cell_volume[donor],
        fvm_vorticity[donor],
        lattice_anchor=anchor,
        spacing=h,
    )
    target_solid = transfer._points_in_solid(lattice.position, include_boundary=True)
    nonzero = np.any(lattice.vortex_strength != 0.0, axis=1) & ~target_solid
    target_position = lattice.position[nonzero]
    target_strength = lattice.vortex_strength[nonzero]
    target_eta = replacement_eta(target_position, box, 0.0)
    target_x = target_position[:, 0]
    release = target_x > box[1]

    position, strength, core_radius = _particle_arrays(vpm_solver)
    eta = replacement_eta(position, box, 0.0)
    remove = eta >= 1.0
    outside = ~remove
    tolerance = 1.0e-5 * h
    existing_index = _node_indices(position, anchor, h)
    reconstructed = anchor + h * existing_index
    regular = np.max(np.abs(position - reconstructed), axis=1) <= tolerance
    target_index = _node_indices(target_position, anchor, h)
    target_by_index = {
        tuple(index): value for index, value in zip(target_index, target_strength, strict=True)
    }
    existing_by_index: dict[tuple[int, int, int], list[int]] = {}
    for index in np.flatnonzero(outside & regular):
        key = tuple(existing_index[index])
        if key in target_by_index:
            existing_by_index.setdefault(key, []).append(int(index))

    update_index: list[int] = []
    update_strength: list[np.ndarray] = []
    duplicate_outside: list[int] = []
    inject = np.ones(len(target_position), dtype=bool)
    for target_number, key_array in enumerate(target_index):
        matches = existing_by_index.get(tuple(key_array), [])
        if not matches:
            continue
        selected = matches[0]
        update_index.append(selected)
        update_strength.append(target_strength[target_number] - strength[selected])
        inject[target_number] = False
        duplicate_outside.extend(matches[1:])
    if duplicate_outside:
        raise RuntimeError("L1 found duplicate regular particles in its release/support region")

    remove_index = np.flatnonzero(remove)
    n_before = int(vpm_solver.particles.n_particles_total)
    n_added = int(inject.sum())
    n_after = n_before - len(remove_index) + n_added
    if n_after > int(vpm_solver.particles.capacity):
        raise RuntimeError(f"L1 requires {n_after:,} particles, exceeding capacity")
    removed_strength = strength[remove]
    overwritten_strength = (
        strength[np.asarray(update_index, dtype=np.int64)] if update_index else np.empty((0, 3))
    )
    injected_net = target_strength.sum(axis=0, dtype=np.float64)
    replaced_net = removed_strength.sum(axis=0, dtype=np.float64) + overwritten_strength.sum(
        axis=0, dtype=np.float64
    )

    if update_index:
        ordered_update = np.argsort(np.asarray(update_index, dtype=np.int64))
        sorted_index = np.asarray(update_index, dtype=np.int64)[ordered_update]
        update_mask = np.isin(np.arange(n_before), sorted_index)
        if update_mask.shape != (n_before,) or int(update_mask.sum()) != len(sorted_index):
            raise RuntimeError("L1 regular-node update mask does not match its target set")
        vpm_solver.update_particle_vortex_strength(
            update_mask,
            np.asarray(update_strength, dtype=np.float64)[ordered_update],
        )
    if tag_release and update_index:
        group_id = np.asarray(vpm_solver.particles.group_id_cpu(), dtype=np.int32)
        release_updates = np.asarray(update_index, dtype=np.int64)[release[~inject]]
        group_id[release_updates] = _L1_RELEASE_GROUP
        _write_group_ids(vpm_solver, group_id)
    if len(remove_index):
        vpm_solver.remove_particles(particle_indices=remove_index.tolist())
    if n_added:
        added = target_position[inject]
        dtype = vpm_solver.np_dtype
        vpm_solver.add_vortex_particles(
            position=np.ascontiguousarray(added, dtype=dtype),
            velocity=np.zeros((n_added, 3), dtype=dtype),
            vortex_strength=np.ascontiguousarray(target_strength[inject], dtype=dtype),
            core_radius=np.full(n_added, transfer.core_radius_ratio * h, dtype=dtype),
            particle_volume=np.full(n_added, h**3, dtype=dtype),
            kinematic_viscosity=np.full(n_added, transfer.kinematic_viscosity, dtype=dtype),
            eddy_viscosity=np.zeros(n_added, dtype=dtype),
            group_id=np.where(release[inject] & tag_release, _L1_RELEASE_GROUP, 0).astype(np.int32),
            zone_id=np.zeros(n_added, dtype=np.int32),
        )
    if int(vpm_solver.particles.n_particles_total) != n_after:
        raise RuntimeError("L1 particle-count mutation did not match its preflight budget")

    diagnostics = {
        "donor_gamma_net": lattice.donor_gamma_net.tolist(),
        "target_gamma_net_before_solid_mask": lattice.target_gamma_net.tolist(),
        "target_gamma_net": injected_net.tolist(),
        "gamma_difference": (injected_net - lattice.donor_gamma_net).tolist(),
        "donor_first_moment": lattice.donor_first_moment.tolist(),
        "target_first_moment_before_solid_mask": lattice.target_first_moment.tolist(),
        "first_moment_error_before_solid_mask": (
            lattice.target_first_moment - lattice.donor_first_moment
        ).tolist(),
        "target_nodes": int(len(lattice.position)),
        "target_nodes_injected_or_updated": int(len(target_position)),
        "target_nodes_excluded_by_solid_mask": int(target_solid.sum()),
        "excluded_gamma_net": lattice.vortex_strength[target_solid]
        .sum(axis=0, dtype=np.float64)
        .tolist(),
        "support_x_min": float(lattice.position[:, 0].min()),
        "support_x_max": float(lattice.position[:, 0].max()),
        "last_replaced_vpm_plane": float(target_x[target_eta >= 1.0].max()),
        "first_persistent_vpm_plane": float(target_x[release].min()),
        "release_plane_gamma": _strength_record(target_position, target_strength, release),
        "n_release_nodes_updated_in_place": int(np.count_nonzero(release & ~inject)),
        "n_release_nodes_added": int(np.count_nonzero(inject & release)),
        "release_nodes_tagged": int(np.count_nonzero(release)) if tag_release else 0,
        "max_target_gamma": float(np.linalg.norm(target_strength, axis=1).max(initial=0.0)),
        "target_core_radius": float(transfer.core_radius_ratio * h),
        "target_particle_volume": h**3,
    }
    result = TransferResult(
        n_particles_before=n_before,
        n_particles_retained=n_before - len(remove_index),
        n_particles_removed=int(len(remove_index)),
        n_particles_blended=0,
        n_particles_injected=n_added,
        n_particles_after=n_after,
        injected_vortex_strength_l1=float(np.linalg.norm(target_strength, axis=1).sum()),
        injected_vortex_strength_net=injected_net,
        replaced_vortex_strength_l1=float(np.linalg.norm(removed_strength, axis=1).sum()),
        replaced_vortex_strength_net=replaced_net,
        state_change_vortex_strength_net=injected_net - replaced_net,
        eta_blending_enabled=False,
    )
    return result, diagnostics


def _record_cloud(vpm_solver, box: np.ndarray) -> dict:
    position, strength, _radius = _particle_arrays(vpm_solver)
    inside = replacement_eta(position, box, 0.0) >= 1.0
    return {
        "all": _strength_record(position, strength, np.ones(len(position), dtype=bool)),
        "fvm_authority": _strength_record(position, strength, inside),
        "vpm_authority": _strength_record(position, strength, ~inside),
        "downstream_vpm_authority": _strength_record(
            position, strength, position[:, 0] > float(box[1])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-from", type=Path, required=True)
    parser.add_argument("--case-directory", type=Path, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    arguments = parser.parse_args()
    manifest = json.loads((arguments.restart_from / "manifest.json").read_text())
    seed_time = float(manifest["time"])
    if arguments.end_time <= seed_time:
        raise ValueError("end time must exceed the checkpoint time")

    case.CASE_DIR = arguments.case_directory.resolve()
    case.CASE_DIR.mkdir(parents=True, exist_ok=True)
    case.FVM_SETUP = replace(
        case.FVM_SETUP, time=replace(case.FVM_SETUP.time, end_time=arguments.end_time)
    )
    case.VPM_SETUP = replace(case.VPM_SETUP, checkpoint_directory=str(case.CASE_DIR / "solution"))
    fvm_solver = fvm.create_fvm_solver(case.FVM_SETUP, case_dir=case.CASE_DIR, mesh=case.FVM_MESH)
    vpm_solver = vpm.create_vpm_solver(case.VPM_SETUP, case_dir=case.CASE_DIR)
    coupler = coupling.create_coupler(fvm_solver, vpm_solver, case.COUPLER_SETUP)
    coupler.initialize()
    original_load = coupler.fvm_solver.load_state
    coupler.fvm_solver.load_state = lambda path: original_load(path, allow_config_change=True)
    try:
        start_step = coupler.load_state(arguments.restart_from)
    finally:
        coupler.fvm_solver.load_state = original_load
    # The FVM field getters are MPI collectives.  Every rank therefore obtains
    # the seed fields before the master applies the time-2.00 L1 replacement.
    # (Only the master receives the assembled buffers.)
    seed_velocity = coupler._get_velocity_field_buffer()
    seed_gradient = coupler._get_velocity_gradient_field_buffer()
    report: dict | None = None
    if coupler._is_master:
        assert coupler.vorticity_transfer is not None
        transfer = coupler.vorticity_transfer
        report = {"seed_time": seed_time, "end_time": arguments.end_time, "steps": []}

        original_gbd = vpm_solver.physics.gbd_diffusion

        def record_pre_gbd(particles, time_step_size, *args, **kwargs):
            if time_step_size > 0.0 and "after_advection_before_gbd_tagged_release" not in report:
                report["after_advection_before_gbd_tagged_release"] = _tagged_release_record(
                    vpm_solver, transfer._box
                )
            return original_gbd(particles, time_step_size, *args, **kwargs)

        vpm_solver.physics.gbd_diffusion = record_pre_gbd

        def l1_transfer(vpm_state, velocity, velocity_gradient):
            if transfer.step > 0:
                report["before_next_l1_tagged_release"] = _tagged_release_record(
                    vpm_state, transfer._box
                )
            transfer.step += 1
            transfer.last_interface_flow = transfer.check_interface_flow(velocity)
            transfer.last_vortex_line_closure = transfer.check_vortex_line_closure(
                velocity_gradient
            )
            result, mapping = _replace_with_l1(
                vpm_state,
                transfer,
                transfer._vorticity_from_gradient(velocity_gradient),
                tag_release=transfer.step == 1,
            )
            report["steps"].append(
                {"mapping": mapping, "after_l1": _record_cloud(vpm_state, transfer._box)}
            )
            if transfer.step == 1:
                report["initial_tagged_release"] = _tagged_release_record(vpm_state, transfer._box)
            else:
                report["after_next_l1_tagged_release"] = _tagged_release_record(
                    vpm_state, transfer._box
                )
            return result

        transfer.transfer = l1_transfer
        initial = l1_transfer(vpm_solver, seed_velocity, seed_gradient)
        report["initial_l1"] = report["steps"].pop()
        report["initial_result"] = {
            "removed": initial.n_particles_removed,
            "added": initial.n_particles_injected,
            "gamma_l1": initial.injected_vortex_strength_l1,
        }
    coupler.solve(start_step=start_step)
    if coupler._is_master:
        assert report is not None
        report["final_cloud"] = _record_cloud(vpm_solver, transfer._box)
        output = case.CASE_DIR / "l1_lattice_diagnostic.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"L1 lattice diagnostic written to {output}")


if __name__ == "__main__":
    main()
