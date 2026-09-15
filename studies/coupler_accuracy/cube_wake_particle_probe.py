"""Independent direct-sum check of the saved cube particle velocity/Jacobian."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys

import h5py
from numba import njit, prange, set_num_threads
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


@njit(cache=True, parallel=True)
def direct_gaussian(points, positions, strengths, radii):
    """Untruncated f64 Biot--Savart sum and its analytic Jacobian.

    The independent formula uses Gamma cross (x-X), with the Gaussian
    exp(-r²/sigma²)/(pi**1.5 sigma³) and J[i,j]=du_i/dx_j.
    """
    velocity = np.zeros((len(points), 3))
    jacobian = np.zeros((len(points), 3, 3))
    sqrt_pi = math.sqrt(math.pi)
    for i in prange(len(points)):
        for k in range(len(positions)):
            r = points[i] - positions[k]
            radius2 = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            q2 = radius2 / radii[k] ** 2
            if q2 < 1e-6:
                c = 1.0 / (3.0 * math.pi**1.5 * radii[k] ** 3)
                factor = c * (1.0 - 0.6 * q2 + 3.0 / 14.0 * q2**2)
                derivative = c / radii[k] ** 2 * (-1.2 + 6.0 / 7.0 * q2 - q2**2 / 3.0)
            else:
                radius = math.sqrt(radius2)
                q = math.sqrt(q2)
                g = math.erf(q) - 2.0 * q / sqrt_pi * math.exp(-q2)
                factor = g / (4.0 * math.pi * radius**3)
                derivative = (4.0 * q**3 / sqrt_pi * math.exp(-q2) - 3.0 * g) / (
                    4.0 * math.pi * radius**5
                )
            gamma = strengths[k]
            cross = np.array(
                [
                    gamma[1] * r[2] - gamma[2] * r[1],
                    gamma[2] * r[0] - gamma[0] * r[2],
                    gamma[0] * r[1] - gamma[1] * r[0],
                ]
            )
            for a in range(3):
                velocity[i, a] += factor * cross[a]
                for b in range(3):
                    jacobian[i, a, b] += derivative * cross[a] * r[b]
            jacobian[i, 0, 1] -= factor * gamma[2]
            jacobian[i, 0, 2] += factor * gamma[1]
            jacobian[i, 1, 0] += factor * gamma[2]
            jacobian[i, 1, 2] -= factor * gamma[0]
            jacobian[i, 2, 0] -= factor * gamma[1]
            jacobian[i, 2, 1] += factor * gamma[0]
    return velocity, jacobian


def validate_direct_formula():
    rng = np.random.default_rng(4301)
    positions = rng.normal(size=(9, 3))
    strengths = rng.normal(size=(9, 3))
    radii = rng.uniform(0.06, 0.2, size=9)
    points = np.vstack((rng.normal(size=(11, 3)), positions[0], positions[0] + 1e-8))
    _, jacobian = direct_gaussian(points, positions, strengths, radii)
    differences = []
    for axis in range(3):
        offset = np.eye(3)[axis] * 1e-6
        plus, _ = direct_gaussian(points + offset, positions, strengths, radii)
        minus, _ = direct_gaussian(points - offset, positions, strengths, radii)
        differences.append(np.max(np.abs((plus - minus) / 2e-6 - jacobian[:, :, axis])))
    relative = float(max(differences) / np.max(np.abs(jacobian)))
    if relative > 1e-7:
        raise AssertionError(f"Independent analytic derivative failed: {relative}")
    return relative


def load_case(source_tree):
    sys.path.insert(0, str(source_tree.resolve()))
    setup = source_tree / "tutorials/coupled_fvm_vpm/02_cube_flow/setup.py"
    spec = importlib.util.spec_from_file_location("cube_probe_case", setup)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rms(values):
    return float(np.sqrt(np.mean(np.sum(values.reshape(len(values), -1) ** 2, axis=1))))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    case = load_case(args.source_tree)
    from source.solvers.vpm import VPMSolver

    vpm_case = replace(
        case.VPM_CASE,
        directory=args.output / "runtime",
        samplers=case.vpm.Samplers(),
        backup=replace(case.VPM_CASE.backup, directory="solution", log_directory="solution"),
    )
    if args.particle_capacity is not None:
        vpm_case = replace(
            vpm_case,
            numerics=replace(
                vpm_case.numerics,
                max_n_particles=args.particle_capacity,
                max_evaluation_points=args.particle_capacity,
            ),
        )
    state_directory = args.trial / "solution" if args.trial else args.baseline / "solution/coupled"
    sample_directory = args.trial / "samples" if args.trial else args.baseline / "samples/coupled"
    report = {
        "source_tree": str(args.source_tree.resolve()),
        "state_directory": str(state_directory.resolve()),
        "direct_derivative_relative_error": validate_direct_formula(),
        "times": [],
    }
    lines = {
        name: pd.read_csv(sample_directory / f"vpm_{name}.csv")
        for name in ("centreline", "offaxis_y075")
    }
    solver = VPMSolver(vpm_case)
    try:
        for time in args.times:
            state_path = state_directory / f"vpm_{round(time * 100):06d}.h5"
            solver.load_backup(state_path)
            solver.refresh_boundary_element_solution()
            frames = []
            for name, df in lines.items():
                selected = df[np.isclose(df["time"], time, atol=1e-8, rtol=0)].copy()
                if selected.empty:
                    raise ValueError(f"No {name} sample at physical time {time:g}")
                selected["sample_source"] = name
                frames.append(selected)
            if args.surfaces:
                from cube_wake_drift_audit import frame
                import pyvista as pv

                for name in ("slice_z0", "wake_slice_z0"):
                    grid = pv.read(frame(sample_directory, f"vpm_{name}", time))
                    values = {
                        **dict(zip(["position_" + a for a in "xyz"], grid.points.T, strict=True)),
                        **dict(
                            zip(
                                ["velocity_" + a for a in "xyz"],
                                grid.point_data["velocity"].T,
                                strict=True,
                            )
                        ),
                    }
                    selected = pd.DataFrame(values)
                    selected["sample_source"] = name
                    frames.append(selected)
            saved = pd.concat(frames, ignore_index=True)
            points = saved[["position_" + a for a in "xyz"]].to_numpy()
            points_mask = ~np.all(np.abs(points) <= 0.51, axis=1)
            points = points[points_mask]
            with h5py.File(state_path) as state:
                if abs(float(state["solver"].attrs["time"]) - time) > 1e-8:
                    raise ValueError("Particle state time does not match sampled data")
                group = state["particles"]
                position, strength, radius = (
                    np.asarray(group[key], dtype=float)
                    for key in ("position", "vortex_strength", "core_radius")
                )
            direct, direct_jacobian = direct_gaussian(points, position, strength, radius)
            tree, tree_jacobian = solver.physics.compute_target_velocity_and_gradients_consistent(
                solver.particles, points, include_freestream=False
            )
            tree_jacobian = np.asarray(tree_jacobian).reshape(-1, 3, 3)
            complete = solver.compute_velocity_at_points(points)
            stored = saved.loc[points_mask, ["velocity_" + a for a in "xyz"]].to_numpy()
            if not np.all(np.isfinite(stored)):
                raise ValueError("Nonfinite saved velocity at a fluid probe")
            sources = saved.loc[points_mask, "sample_source"].to_numpy()
            row = {
                "time": time,
                "particle_count": len(position),
                "target_count": len(points),
                "tree_velocity_error_rms": rms(tree - direct),
                "tree_velocity_error_max": float(np.linalg.norm(tree - direct, axis=1).max()),
                "tree_jacobian_error_relative": rms(tree_jacobian - direct_jacobian)
                / rms(direct_jacobian),
                "replayed_vs_saved_velocity_rms": rms(complete - stored),
                "replayed_vs_saved_by_sampler": {
                    name: {
                        "points": int(np.count_nonzero(sources == name)),
                        "rms": rms((complete - stored)[sources == name]),
                        "maximum": float(
                            np.linalg.norm((complete - stored)[sources == name], axis=1).max()
                        ),
                    }
                    for name in sorted(set(sources))
                },
                "body_velocity_rms": rms(complete - tree - np.array([1, 0, 0])),
            }
            report["times"].append(row)
            np.savez_compressed(
                args.output / f"fields_t{time:g}.npz",
                points=points,
                direct=direct,
                direct_jacobian=direct_jacobian,
                tree=tree,
                tree_jacobian=tree_jacobian,
                complete=complete,
                saved=stored,
            )
            (args.output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        solver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--surfaces", action="store_true")
    parser.add_argument("--particle-capacity", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[3, 6, 10, 15])
    args = parser.parse_args()
    set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
