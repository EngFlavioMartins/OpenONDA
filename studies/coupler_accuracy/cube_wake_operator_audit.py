"""Attribute one saved 3D cube step to its native particle operators.

Inputs remain immutable. This diagnostic uses the source that produced the
saved states, evaluates actual RK rates, and records the RK and GBD increments
separately. It does not advance FVM or claim an alternative coupled trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from cube_wake_particle_probe import direct_gaussian, load_case, rms, validate_direct_formula
from cube_wake_vorticity_consistency import gaussian_vorticity_and_divergence
from numba import set_num_threads
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


def snapshot(solver):
    """Copy the active source arrays, excluding stale derived fields."""
    return {
        "position": solver.particle_position.astype(float).copy(),
        "vortex_strength": solver.particle_vortex_strength.astype(float).copy(),
        "core_radius": solver.particle_core_radius.astype(float).copy(),
    }


def curl(jacobian):
    return np.column_stack(
        (
            jacobian[:, 2, 1] - jacobian[:, 1, 2],
            jacobian[:, 0, 2] - jacobian[:, 2, 0],
            jacobian[:, 1, 0] - jacobian[:, 0, 1],
        )
    )


def evaluate(points, state):
    position, strength, radius = (state[k] for k in ("position", "vortex_strength", "core_radius"))
    velocity, jacobian = direct_gaussian(points, position, strength, radius)
    omega, divergence = gaussian_vorticity_and_divergence(points, position, strength, radius)
    return {"velocity": velocity, "curl": curl(jacobian), "omega": omega, "divergence": divergence}


def field_metrics(fields, masks):
    return {
        name: {
            "points": int(mask.sum()),
            "transverse_velocity_rms": rms(fields["velocity"][mask, 1:]),
            "vorticity_minus_curl_rms": rms((fields["omega"] - fields["curl"])[mask]),
            "vorticity_rms": rms(fields["omega"][mask]),
            "divergence_rms": float(np.sqrt(np.mean(fields["divergence"][mask] ** 2))),
        }
        for name, mask in masks.items()
    }


def reflection_asymmetry(points, velocity):
    """Project a closed 3D point orbit onto its odd z-reflection component."""
    distance, index = cKDTree(points).query(points * [1, 1, -1])
    np.testing.assert_allclose(distance, 0, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(index[index], np.arange(len(points)))
    return 0.5 * (velocity - velocity[index] * [1, 1, -1])


def summarize_saved_fields(directory):
    """Reproduce reflection budgets and particle alignment from saved raw arrays."""
    budgets, alignments = [], []
    for path in sorted(directory.glob("fields_t*.npz")):
        time = float(path.stem.removeprefix("fields_t"))
        with np.load(path) as data:
            points = data["points"]
            mask = (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62)
            initial = reflection_asymmetry(points, data["accepted_velocity"])
            energy0 = rms(initial[mask]) ** 2
            budget = {"time": time, "states": {}}
            for name in (
                "accepted",
                "advection_part",
                "stretching_part",
                "after_rk",
                "remap_only",
                "molecular_gbd",
                "native_les_gbd",
            ):
                asymmetry = reflection_asymmetry(points, data[f"{name}_velocity"])
                value = rms(asymmetry[mask])
                budget["states"][name] = {
                    "reflection_asymmetry_rms": value,
                    "energy_change_over_dt": (value**2 - energy0) / 0.01,
                }
            budgets.append(budget)
            p, g = data["particle_position"], data["particle_strength"]
            q = curl(data["stage_gradient"])
            magnitude, omega = np.linalg.norm(g, axis=1), np.linalg.norm(q, axis=1)
            cosine = np.clip(np.sum(g * q, axis=1) / np.maximum(magnitude * omega, 1e-30), -1, 1)
            angle = np.degrees(np.arccos(cosine))
            nearest = cKDTree(p).query(p, k=2, workers=2)[0][:, 1]
            regions = {
                "near_body": np.max(np.abs(p), axis=1) < 0.75,
                "seam": (p[:, 0] >= 1.25) & (p[:, 0] <= 1.65),
                "outer_wake": p[:, 0] > 1.65,
            }
            alignment = {"time": time, "regions": {}}
            for name, selected in regions.items():
                active = selected & (magnitude > 0.1 * 0.06**3)
                alignment["regions"][name] = {
                    "strength_weighted_angle_to_curl_degrees": float(
                        np.average(angle[selected], weights=magnitude[selected])
                    ),
                    "opposed_strength_fraction": float(
                        magnitude[selected & (cosine < 0)].sum() / magnitude[selected].sum()
                    ),
                    "strong_particle_nearest_distance_p99_over_h": float(
                        np.quantile(nearest[active], 0.99) / 0.06
                    ),
                    "near_duplicate_count": int((nearest[selected] < 1e-5).sum()),
                }
            alignments.append(alignment)
    (directory / "asymmetry-budget.json").write_text(json.dumps(budgets, indent=2) + "\n")
    (directory / "alignment-and-overlap.json").write_text(json.dumps(alignments, indent=2) + "\n")


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    case = load_case(args.source_tree)
    from source.solvers.vpm import VPMSolver
    from source.solvers.vpm.physics.induction.base import StageRates, StageState

    policy = replace(
        case.VPM_CASE,
        directory=args.output / "runtime",
        samplers=case.vpm.Samplers(),
        numerics=replace(
            case.VPM_CASE.numerics, max_n_particles=300000, max_evaluation_points=300000
        ),
    )
    x = np.array([0.9, 1.14, 1.26, 1.38, 1.5, 1.62, 1.86, 2.34])
    yz = np.linspace(-0.54, 0.54, 7)
    points = np.stack(np.meshgrid(x, yz, yz, indexing="ij"), axis=-1).reshape(-1, 3)
    masks = {
        "authority_ramp": points[:, 0] < 1.25,
        "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
        "outer_wake": points[:, 0] > 1.62,
    }
    report = {
        "description": __doc__,
        "source_tree": str(args.source_tree.resolve()),
        "state_directory": str(args.solution.resolve()),
        "direct_derivative_relative_error": validate_direct_formula(),
        "probe_count": len(points),
        "times": [],
    }
    solver = VPMSolver(policy)
    try:
        for time in args.times:
            path = args.solution / f"vpm_{round(100 * time):06d}.h5"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            solver.load_backup(path)
            if abs(solver.time - time) > 1e-8:
                raise ValueError("Backup clock differs from requested time")
            solver.physics.configure_body_box(np.array([-0.5, 0.5] * 3))
            solver.physics.configure_grid_lattice_anchor(np.array([-0.03] * 3), 0.06)
            solver.refresh_boundary_element_solution()
            original = snapshot(solver)
            p, g = original["position"], original["vortex_strength"]
            particle_masks = {
                "near_body": np.max(np.abs(p), axis=1) < 0.75,
                "renewal_seam": (p[:, 0] >= 1.2) & (p[:, 0] <= 1.65),
                "outer_wake": p[:, 0] > 1.65,
            }
            rng = np.random.default_rng(20260915)
            selected = np.unique(
                np.concatenate(
                    [
                        rng.choice(np.flatnonzero(m), min(128, int(m.sum())), replace=False)
                        for m in particle_masks.values()
                    ]
                )
            )
            # Evaluate the actual production StageRHS, including its body hooks.
            solver.stage_rhs.evaluate(
                StageState(
                    position=solver.particles.position,
                    vortex_strength=solver.particles.vortex_strength,
                    core_radius=solver.particles.core_radius,
                    count=len(p),
                    time=solver.time,
                    stage_index=0,
                ),
                solver.time,
                StageRates(
                    velocity=solver.integrator.stage_velocity[0],
                    vortex_strength_rate=solver.integrator.stage_strength_rate[0],
                    velocity_gradient=solver.integrator.stage_velocity_gradient,
                ),
            )
            native_u = solver.physics._download_vector_field(
                solver.integrator.stage_velocity[0], len(p)
            ).copy()
            native_rate = solver.physics._download_vector_field(
                solver.integrator.stage_strength_rate[0], len(p)
            ).copy()
            native_j = (
                solver.physics._download_matrix_field(
                    solver.integrator.stage_velocity_gradient, len(p)
                )
                .copy()
                .reshape(-1, 3, 3)
            )
            direct_u, direct_j = direct_gaussian(p[selected], p, g, original["core_radius"])
            body_u = solver.panel_solver.compute_induced_velocity(p[selected])
            body_j = solver.panel_solver.compute_induced_velocity_gradient(p[selected])
            complete_u = direct_u + body_u + [1.0, 0.0, 0.0]
            complete_j = direct_j + body_j
            complete_rate = np.einsum("nji,nj->ni", complete_j, g[selected])
            stage = {}
            for name, mask in particle_masks.items():
                m = mask[selected]
                stage[name] = {
                    "particles": int(mask.sum()),
                    "targets": int(m.sum()),
                    "velocity_error_rms": rms((native_u[selected] - complete_u)[m]),
                    "jacobian_relative_error": rms((native_j[selected] - complete_j)[m])
                    / rms(complete_j[m]),
                    "stretching_relative_error": rms((native_rate[selected] - complete_rate)[m])
                    / rms(complete_rate[m]),
                    "body_velocity_rms": rms(body_u[m]),
                }
            # Prepare the same accepted-step LES state used before native RK.
            solver.stepper._update_velocity_and_gradients()
            solver.stepper._update_les_state()
            solver.stepper._advance_particles(solver.time_step_size)
            after_rk = snapshot(solver)
            states = {
                "accepted": original,
                "after_rk": after_rk,
                "advection_part": {**original, "position": after_rk["position"]},
                "stretching_part": {**original, "vortex_strength": after_rk["vortex_strength"]},
            }
            # Each diffusion query starts from the same post-RK particle arrays.
            # dt=0 isolates scatter/pruning/moment recovery from the heat solve.
            vc = solver._viscous_config
            for label, dt, effective in (
                ("remap_only", 0.0, None),
                ("molecular_gbd", solver.time_step_size, None),
                (
                    "native_les_gbd",
                    solver.time_step_size,
                    solver.particles.effective_viscosity_cpu(),
                ),
            ):
                result = solver.physics.gbd_diffusion(
                    solver.particles,
                    time_step_size=dt,
                    particle_spacing=vc.gbd_grid_spacing,
                    kinematic_viscosity=vc.kinematic_viscosity,
                    domain_padding=vc.gbd_domain_padding,
                    regen_threshold=vc.gbd_threshold,
                    regen_threshold_mode=vc.gbd_threshold_mode,
                    effective_viscosity=effective,
                    max_nodes=vc.gbd_max_nodes,
                    remeshing_kernel=vc.gbd_remeshing_kernel,
                )
                if result is None:
                    raise RuntimeError(f"No particle proposal for {label}")
                states[label] = {k: np.asarray(result[k], dtype=float) for k in original}
            fields = {name: evaluate(points, state) for name, state in states.items()}
            row = {
                "time": time,
                "input_sha256": digest,
                "stage": stage,
                "field_metrics": {name: field_metrics(f, masks) for name, f in fields.items()},
                "increments": {},
            }
            pairs = (
                ("rk", "accepted", "after_rk"),
                ("advection", "accepted", "advection_part"),
                ("stretching", "accepted", "stretching_part"),
                ("remap", "after_rk", "remap_only"),
                ("molecular_diffusion", "remap_only", "molecular_gbd"),
                ("les_contribution", "molecular_gbd", "native_les_gbd"),
                ("native_predictor", "accepted", "native_les_gbd"),
            )
            for label, before, after in pairs:
                du = fields[after]["velocity"] - fields[before]["velocity"]
                row["increments"][label] = {
                    name: {
                        "velocity_rms": rms(du[m]),
                        "transverse_velocity_rms": rms(du[m, 1:]),
                        "transverse_energy_rate": float(
                            np.mean(np.sum(fields[before]["velocity"][m, 1:] * du[m, 1:], axis=1))
                            / solver.time_step_size
                        ),
                    }
                    for name, m in masks.items()
                }
            np.savez_compressed(
                args.output / f"fields_t{time:g}.npz",
                points=points,
                selected=selected,
                particle_position=p,
                particle_strength=g,
                stage_velocity=native_u,
                stage_gradient=native_j,
                stage_strength_rate=native_rate,
                complete_stage_velocity=complete_u,
                complete_stage_gradient=complete_j,
                **{
                    f"{name}_{key}": value for name, f in fields.items() for key, value in f.items()
                },
            )
            for name, state in states.items():
                np.savez_compressed(args.output / f"particles_{name}_t{time:g}.npz", **state)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise AssertionError("Input backup changed")
            report["times"].append(row)
            (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({"time": time, "stage": stage, "increments": row["increments"]}),
                flush=True,
            )
    finally:
        solver.close()
    summarize_saved_fields(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
