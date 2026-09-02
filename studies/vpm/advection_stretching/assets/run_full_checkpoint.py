#!/usr/bin/env python3
"""Short, full-particle checkpoint replays with stage-level diagnostics."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
from .. import setup

from .core import contract, target_fields
from openonda.vpm import (
    RK2,
    RK4,
    SSPRK3,
    Backup,
    DirectInduction,
    HealthLimits,
    LagrangianCFLLimit,
    Numerics,
    RunPlan,
    Samplers,
    StabilizationConfig,
    TreecodeInduction,
    TurbulenceConfig,
    ViscousConfig,
    VPMCase,
    VPMSolver,
)

CHECKPOINTS = {
    "leapfrog": {
        "path": setup.ROOT
        / "tutorials/vpm/vortex_interactions/solution/leapfrog_les/vpm_000150.h5",
        "dt": 20.0 * 0.035**2 / np.pi,
        "kernel": "GAUSSIAN",
        "theta": 0.30,
        "spacing": 0.035,
        "nu": np.pi / 3000.0,
        "cs": 0.20,
        "production_scheme": "RK3",
        "production_tree": False,
        "freestream": (0.0, 0.0, 0.0),
        "bounds": None,
    },
    "rotor": {
        "path": setup.ROOT / "tutorials/vpm/rotor_flow/solution/vpm_rotor_000520.h5",
        "dt": 0.006,
        "kernel": "WINCKELMANS",
        "theta": 0.20,
        "spacing": min(5.0 / 22.0, 49.0 * 0.006),
        "nu": 1.5e-5,
        "cs": 0.17,
        "production_scheme": "RK2",
        "production_tree": True,
        "freestream": (7.0, 0.0, 0.0),
        "bounds": (-12.0, 120.0, -12.0, 12.0, -12.0, 12.0),
    },
}

CONFIGS = (
    "exact_pair_rk3_isolated",
    "tree_gradient_rk3_isolated",
    "production_numerics_unforced",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(name: str, count: int | None = None) -> dict[str, np.ndarray]:
    spec = CHECKPOINTS[name]
    with h5py.File(spec["path"], "r") as archive:
        particles = archive["particles"]
        total = len(particles["position"])
        index = (
            np.arange(total)
            if count is None or count >= total
            else np.linspace(0, total - 1, count, dtype=np.int64)
        )

        def field(key: str, default: np.ndarray) -> np.ndarray:
            return np.asarray(particles[key][index] if key in particles else default)

        n = len(index)
        return {
            "checkpoint_index": index,
            "position": field("position", np.empty((n, 3), np.float32)).astype("f4"),
            "velocity": field("velocity", np.zeros((n, 3), np.float32)).astype("f4"),
            "vortex_strength": field("vortex_strength", np.empty((n, 3), np.float32)).astype("f4"),
            "core_radius": field("core_radius", np.empty(n, np.float32)).astype("f4"),
            "particle_volume": field("particle_volume", np.empty(n, np.float32)).astype("f4"),
            "kinematic_viscosity": field(
                "kinematic_viscosity", np.full(n, spec["nu"], np.float32)
            ).astype("f4"),
            "eddy_viscosity": field("eddy_viscosity", np.zeros(n, np.float32)).astype("f4"),
            "group_id": field("group_id", np.zeros(n, np.int32)).astype("i4"),
            "zone_id": field("zone_id", np.zeros(n, np.int32)).astype("i4"),
        }


def build_solver(name: str, configuration: str, maximum: int) -> VPMSolver:
    spec = CHECKPOINTS[name]
    production = configuration == "production_numerics_unforced"
    tree = configuration == "tree_gradient_rk3_isolated" or (
        production and bool(spec["production_tree"])
    )
    scheme = str(spec["production_scheme"] if production else "RK3")
    stabilization = (
        StabilizationConfig.bounded_domain(spec["bounds"])
        if production and spec["bounds"] is not None
        else StabilizationConfig.disabled()
    )
    integrator = {"RK2": RK2, "RK3": SSPRK3, "RK4": RK4}[scheme]()
    induction = TreecodeInduction(
        theta=float(spec["theta"]), sort_particle_targets=True, traversal_block_dim=128
    ) if tree else DirectInduction()
    numerics = Numerics(
        time_step_size=float(spec["dt"]),
        integrator=integrator,
        induction=induction,
        viscous=(
            ViscousConfig.cs(
                kinematic_viscosity=float(spec["nu"]), particle_spacing=float(spec["spacing"])
            )
            if production
            else ViscousConfig.inviscid(particle_spacing=float(spec["spacing"]))
        ),
        turbulence=(
            TurbulenceConfig.les_smagorinsky(float(spec["cs"]))
            if production
            else TurbulenceConfig.inviscid()
        ),
        stabilization=stabilization,
        health_limits=HealthLimits(lagrangian_cfl=LagrangianCFLLimit(maximum=None)),
        particle_kernel=str(spec["kernel"]),
        freestream_velocity=tuple(spec["freestream"] if production else (0.0, 0.0, 0.0)),
        max_n_particles=maximum,
        max_evaluation_points=maximum,
        compute_device="VULKAN",
        precision="f32",
        write_precision="f32",
        verbose=False,
    )
    case = VPMCase(
        numerics=numerics,
        backup=Backup(interval_steps=0, directory="work", log_directory="work"),
        samplers=Samplers(),
        run=RunPlan(steps=0, initial_samples=False, final_backup=False),
        directory=setup.RESULTS / "full_checkpoint_work" / name / configuration,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return VPMSolver(case)


def upload(solver: VPMSolver, data: dict[str, np.ndarray], *, replace: bool = False) -> None:
    values = {
        key: data[key]
        for key in (
            "position",
            "velocity",
            "vortex_strength",
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "group_id",
            "zone_id",
        )
    }
    if replace:
        solver.replace_vortex_particles(**values, report_removal=False)
    else:
        solver.add_vortex_particles(**values)


def diagnostic_indices(gamma: np.ndarray, count: int = 64) -> np.ndarray:
    n = len(gamma)
    half = min(count // 2, n)
    strongest = np.argpartition(np.linalg.norm(gamma, axis=1), -half)[-half:]
    spread = np.linspace(0, n - 1, min(count - half, n), dtype=np.int64)
    return np.unique(np.r_[strongest, spread])


def exact_target_gradient(
    target: np.ndarray, position: np.ndarray, gamma: np.ndarray, sigma: np.ndarray, kernel: str
) -> np.ndarray:
    chunks = []
    for start in range(0, len(target), 8):
        chunks.append(target_fields(target[start : start + 8], position, gamma, sigma, kernel)[2])
    return np.concatenate(chunks)


def percentile_fields(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
        f"{prefix}_max": float(np.max(values)),
    }


def replay_one(
    name: str, configuration: str, steps: int
) -> tuple[dict, list[dict], list[dict], dict]:
    import taichi as ti

    spec = CHECKPOINTS[name]
    data = load_checkpoint(name)
    n = len(data["position"])
    initial_x = data["position"].astype(np.float64)
    initial_g = data["vortex_strength"].astype(np.float64)
    selected = diagnostic_indices(initial_g)
    solver = build_solver(name, configuration, n + 32)
    upload(solver, data)
    original_rhs = solver.stage_rhs.evaluate
    stage_rows: list[dict] = []
    stage_number = 0
    active_step = 0

    def traced_rhs(stage_state, stage_time, stage_rates):
        nonlocal stage_number
        original_rhs(stage_state, stage_time, stage_rates)
        ti.sync()
        position = stage_state.position.to_numpy()[:n].astype(np.float64)
        gamma = stage_state.vortex_strength.to_numpy()[:n].astype(np.float64)
        sigma = stage_state.core_radius.to_numpy()[:n].astype(np.float64)
        rate = stage_rates.vortex_strength_rate.to_numpy()[:n].astype(np.float64)
        exact_j = exact_target_gradient(
            position[selected], position, gamma, sigma, str(spec["kernel"])
        )
        s = 0.5 * (exact_j + exact_j.transpose(0, 2, 1))
        w = 0.5 * (exact_j - exact_j.transpose(0, 2, 1))
        rate_ratio = np.linalg.norm(rate, axis=1) / np.maximum(np.linalg.norm(gamma, axis=1), 1e-30)
        row = {
            "checkpoint": name,
            "configuration": configuration,
            "step": active_step,
            "stage": stage_number % solver.integrator.tableau.stages + 1,
            "particles": n,
            "gradient_reference": "independent_f64_source_blob_gradient_on_64_targets",
            "exact_gradient_norm_max": float(np.linalg.norm(exact_j, axis=(1, 2)).max()),
            "exact_chi_s_max": float(spec["dt"] * np.linalg.norm(s, ord=2, axis=(1, 2)).max()),
            "exact_chi_r_max": float(spec["dt"] * np.linalg.norm(w, ord=2, axis=(1, 2)).max()),
            "net_strength_rate_norm": float(np.linalg.norm(rate.sum(axis=0))),
            **percentile_fields(float(spec["dt"]) * rate_ratio, "chi_gamma"),
        }
        row["actual_gradient_evaluator"] = "induction_auxiliary_not_requested"
        exact_rate = contract(exact_j, gamma[selected], "TRANSPOSED")
        row["rate_relative_l2_on_targets"] = float(
            np.linalg.norm(rate[selected] - exact_rate) / max(np.linalg.norm(exact_rate), 1e-30)
        )
        stage_rows.append(row)
        stage_number += 1

    solver.stage_rhs.evaluate = traced_rhs
    wall_times = []
    try:
        for step in range(1, steps + 1):
            active_step = step
            start = time.perf_counter()
            solver.advance(defer_output=True)
            solver.synchronize()
            wall_times.append(time.perf_counter() - start)
        final_x = solver.particles.position_cpu().astype(np.float64)
        final_g = solver.particles.vortex_strength_cpu().astype(np.float64)
    finally:
        solver.close()

    g0mag = np.linalg.norm(initial_g, axis=1)
    gfmag = np.linalg.norm(final_g, axis=1)
    growth = gfmag / np.maximum(g0mag, 1e-30) - 1.0
    displacement = np.linalg.norm(final_x - initial_x, axis=1)
    top = np.argsort(
        np.maximum(np.abs(growth), displacement / np.maximum(data["core_radius"], 1e-30))
    )[-100:][::-1]
    particle_rows = [
        {
            "checkpoint": name,
            "configuration": configuration,
            "rank": rank,
            "checkpoint_index": int(data["checkpoint_index"][i]),
            "x0": initial_x[i, 0],
            "y0": initial_x[i, 1],
            "z0": initial_x[i, 2],
            "initial_strength": g0mag[i],
            "final_strength": gfmag[i],
            "relative_strength_growth": growth[i],
            "displacement": displacement[i],
        }
        for rank, i in enumerate(top, 1)
    ]
    scale = max(
        float(np.linalg.norm(initial_g.sum(axis=0))),
        float(np.linalg.norm(initial_g, axis=1).sum()),
        1e-30,
    )
    summary = {
        "checkpoint": name,
        "configuration": configuration,
        "status": "completed",
        "particles": n,
        "steps": steps,
        "dt": spec["dt"],
        "horizon": steps * float(spec["dt"]),
        "kernel": spec["kernel"],
        "scheme": spec["production_scheme"]
        if configuration == "production_numerics_unforced"
        else "RK3",
        "stretching_evaluator": "canonical_pairwise_rate",
        "forcing": "freestream_only_no_vlm_replay"
        if name == "rotor" and configuration == "production_numerics_unforced"
        else "isolated_self_induced",
        "median_step_wall_s": float(np.median(wall_times)),
        "total_strength_drift_abs": float(
            np.linalg.norm(final_g.sum(axis=0) - initial_g.sum(axis=0))
        ),
        "total_strength_drift_normalized": float(
            np.linalg.norm(final_g.sum(axis=0) - initial_g.sum(axis=0)) / scale
        ),
        "strength_growth_median": float(np.median(growth)),
        "strength_growth_p95": float(np.percentile(growth, 95)),
        "strength_growth_max": float(np.max(growth)),
        "displacement_max": float(np.max(displacement)),
    }
    state = {"position": final_x, "vortex_strength": final_g}
    return summary, stage_rows, particle_rows, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--configuration", choices=CONFIGS, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    setup.mkdirs()
    summary, stages, particles, state = replay_one(args.checkpoint, args.configuration, args.steps)
    stem = f"{args.checkpoint}_{args.configuration}"
    write_csv(setup.RESULTS / f"full_replay_{stem}_stages.csv", stages)
    write_csv(setup.RESULTS / f"full_replay_{stem}_particles.csv", particles)
    (setup.RESULTS / f"full_replay_{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(setup.RESULTS / f"full_replay_{stem}_state.npz", **state)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
