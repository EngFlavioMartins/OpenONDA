"""Test one cube renewal with a donor exactly matching its current VPM velocity.

The oracle supplies curl of the same Gaussian velocity, evaluated using the
saved run's backend and independently checked at selected lattice targets.
It isolates a physical fixed-point condition, without advancing either solver.
An artificial omega_G donor separates the Gaussian coefficient-blend effect
from the omega_G versus curl(u) difference. A coefficient-preserving control
retains identical remapping, masks, pruning and moment restoration.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from cube_wake_operator_audit import curl, evaluate, reflection_asymmetry
from cube_wake_particle_probe import direct_gaussian, load_case, rms
from numba import set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    case = load_case(args.source_tree)
    from source.coupler import stable_renewal as renewal
    from source.solvers.vpm import VPMSolver

    with np.load(args.stage_inputs) as data:
        inputs = {key: data[key].copy() for key in data.files}
    lattice = renewal.build_stable_renewal_lattice(
        np.array([-1.25, 1.25] * 3),
        0.06,
        buffer_length=0.135,
        authority_ramp_width=0.36,
        lattice_anchor=np.array([-0.03] * 3),
        mesh_weight_at_node=lambda p: np.all(np.abs(p) <= 1.5 + 1e-12, axis=1).astype(float),
        fluid_weight_at_node=lambda p: smooth_distance(p, -0.06, 0),
        interior_at_node=lambda p: np.all(np.abs(p) < 0.5, axis=1),
    )
    np.testing.assert_array_equal(lattice.positions, inputs["position"])
    np.testing.assert_array_equal(lattice.fvm_authority, inputs["authority"])
    policy = replace(
        case.VPM_CASE,
        directory=args.output / "runtime",
        samplers=case.vpm.Samplers(),
        numerics=replace(
            case.VPM_CASE.numerics, max_n_particles=300000, max_evaluation_points=300000
        ),
    )
    solver = VPMSolver(policy)
    try:
        solver.load_backup(args.checkpoint)
        solver.refresh_boundary_element_solution()
        p, g, sigma = (
            getattr(solver, name).astype(float)
            for name in ("particle_position", "particle_vortex_strength", "particle_core_radius")
        )
        started = time.perf_counter()
        velocity, jacobian = solver.compute_velocity_and_gradient_at_points(
            lattice.positions, particle_spacing=0.06
        )
        q = curl(np.asarray(jacobian, dtype=float).reshape(-1, 3, 3))
        selected = np.linspace(0, len(q) - 1, 48, dtype=int)
        _, exact_j = direct_gaussian(lattice.positions[selected], p, g, sigma)
        q_error = rms(q[selected] - curl(exact_j))
        omega = (
            renewal.gaussian_represented_vortex_strength(
                inputs["vpm_vortex_strength"],
                lattice.shape,
                0.06,
                core_radius=0.066,
            )
            / 0.06**3
        )
        # Include persistent outer particles in this diagnostic omega_G donor.
        outside = ~np.all(
            (p >= lattice.renewal_bounds[::2]) & (p <= lattice.renewal_bounds[1::2]), axis=1
        )
        if np.any(outside):
            from cube_wake_measured_longitudinal import density_at

            omega += density_at(lattice.positions, p[outside], g[outside], sigma[outside])
        with np.load(args.operator_audit / "fields_t8.npz") as data:
            points = data["points"].copy()
        before = evaluate(points, {"position": p, "vortex_strength": g, "core_radius": sigma})
        records = []
        original_blend = renewal.blend_represented_state
        for name, target in {
            "matching_velocity_curl": q * 0.06**3,
            "matching_gaussian_density": omega * 0.06**3,
            "preserve_coefficients": np.zeros_like(q),
        }.items():
            if name == "preserve_coefficients":

                def preserve(*positional, **keywords):
                    result = original_blend(*positional, **keywords)
                    return replace(result, vortex_strength=positional[0].copy())

                renewal.blend_represented_state = preserve
            else:
                renewal.blend_represented_state = original_blend
            result = renewal.renew_stable_overlap(
                p,
                g,
                lattice,
                fvm_vortex_strength_at_node=lambda _, target=target: target,
                particle_fluid_weight=lambda points: smooth_distance(points, -0.06, 0),
                particle_in_solid=lambda points: np.all(np.abs(points) < 0.5, axis=1),
                prune_threshold=0.05 * 0.06**3,
                core_radius_ratio=1.1,
                amplification_cap=1.8,
                boundary_prune_multiplier=10,
                compute_diagnostics=False,
            )
            fields = evaluate(
                points,
                {
                    "position": result.position,
                    "vortex_strength": result.vortex_strength,
                    "core_radius": result.core_radius,
                },
            )
            row = {"variant": name, "particles": len(result.position), "regions": {}}
            for region, mask in {
                "authority_ramp": points[:, 0] < 1.25,
                "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
                "outer_wake": points[:, 0] > 1.62,
            }.items():
                row["regions"][region] = {
                    "velocity_change_rms": rms((fields["velocity"] - before["velocity"])[mask]),
                    "reflection_rms": rms(reflection_asymmetry(points, fields["velocity"])[mask]),
                    "vorticity_minus_curl_rms": rms((fields["omega"] - fields["curl"])[mask]),
                }
            records.append(row)
            np.savez_compressed(
                args.output / f"{name}.npz",
                points=points,
                before_velocity=before["velocity"],
                **fields,
            )
            print(json.dumps(row), flush=True)
        renewal.blend_represented_state = original_blend
        np.savez_compressed(
            args.output / "oracle_targets.npz",
            position=lattice.positions,
            matched_velocity=velocity,
            matched_curl=q,
            gaussian_vorticity=omega,
        )
        report = {
            "description": __doc__,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "oracle_curl_error_rms_at_48_targets": q_error,
            "wall_seconds_including_diagnostics": time.perf_counter() - started,
            "variants": records,
        }
        (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        solver.close()


def smooth_distance(points, low, high):
    local = np.abs(points) - 0.5
    distance = np.linalg.norm(np.maximum(local, 0), axis=1) + np.minimum(local.max(axis=1), 0)
    phase = np.clip((distance - low) / (high - low), 0, 1)
    return phase**2 * (3 - 2 * phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-inputs", type=Path, required=True)
    parser.add_argument("--operator-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)
