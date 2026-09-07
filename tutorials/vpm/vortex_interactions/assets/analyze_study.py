"""Summarize survival, common-time physics and resolved ring-mode evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..study import STUDY_DIR
from .. import setup
from .render_study import states


def mode_amplitudes(position, strength, groups, bins=64):
    """Azimuthal centroid modes; valid only while each label remains a tube."""
    output = []
    for group in np.unique(groups):
        selected = groups == group
        p, a = position[selected], strength[selected]
        w = np.linalg.norm(a, axis=1)
        theta = np.mod(np.arctan2(p[:, 2], p[:, 1]), 2 * np.pi)
        index = np.minimum((theta / (2 * np.pi) * bins).astype(int), bins - 1)
        mass = np.bincount(index, weights=w, minlength=bins)
        if np.any(mass <= 0):
            continue
        radius = np.hypot(p[:, 1], p[:, 2])
        radial = np.bincount(index, weights=w * radius, minlength=bins) / mass
        axial = np.bincount(index, weights=w * p[:, 0], minlength=bins) / mass
        radial_modes = 2 * np.abs(np.fft.rfft(radial - radial.mean())) / bins
        axial_modes = 2 * np.abs(np.fft.rfft(axial - axial.mean())) / bins
        output.append(
            {
                "group_id": int(group),
                "radial_mode8": radial_modes[8],
                "axial_mode8": axial_modes[8],
                "other_modes_rms": float(
                    np.sqrt(
                        np.sum(radial_modes[1:8] ** 2)
                        + np.sum(radial_modes[9:17] ** 2)
                        + np.sum(axial_modes[1:8] ** 2)
                        + np.sum(axial_modes[9:17] ** 2)
                    )
                ),
            }
        )
    return output


def load_runs(names=None):
    runs = []
    for directory in sorted(STUDY_DIR.glob("*")):
        if ".previous-" in directory.name or (names and directory.name not in names):
            continue
        metadata = directory / "result.json"
        flow = directory / "samples" / "diagnostics" / "flow_integrals.csv"
        if not metadata.exists() or not flow.exists():
            continue
        meta = json.loads(metadata.read_text())
        data = pd.read_csv(flow)
        rings = pd.read_csv(flow.with_name("ring_diagnostics.csv"))
        runs.append((directory.name, directory, meta, data, rings))
    # The diagnostic-fix control was launched with the original tutorial.
    # Include it without copying or renaming its live scientific output.
    baseline_meta = setup.TUTORIAL_DIR / "samples" / "baseline" / "run_metadata.json"
    if (
        (not names or "leapfrog_baseline" in names)
        and not any(run[0] == "leapfrog_baseline" for run in runs)
        and baseline_meta.exists()
    ):
        meta = json.loads(baseline_meta.read_text())
        if meta.get("schema_version", 0) >= 2:
            meta["signature"] = {"scenario": "leapfrog", "method": "baseline"}
            if meta["status"] == "running":
                meta.pop("completed_steps", None)
            flow = baseline_meta.with_name("flow_integrals.csv")
            runs.insert(
                0,
                (
                    "leapfrog_baseline",
                    setup.TUTORIAL_DIR,
                    meta,
                    pd.read_csv(flow),
                    pd.read_csv(flow.with_name("ring_diagnostics.csv")),
                ),
            )
    return runs


def trajectory_rmse(rings):
    reference = pd.read_csv(
        Path(__file__).parent / "references" / "leapfrogging_lbm_trajectory.csv"
    )
    errors = []
    for group, reference_group in ((0, 2), (1, 1)):
        ring = rings[rings.group_id == group].sort_values("vortex_centroid_x")
        target = reference[reference.ring == reference_group]
        x = target.x_over_R0.to_numpy() - 2.5
        inside = (x >= ring.vortex_centroid_x.min()) & (x <= ring.vortex_centroid_x.max())
        if np.any(inside):
            errors.extend(
                np.interp(x[inside], ring.vortex_centroid_x, ring.major_radius)
                - target.R_over_R0.to_numpy()[inside]
            )
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else np.nan


def summarize(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, modes = [], []
    for name, directory, meta, data, rings in runs:
        initial, final = data.iloc[0], data.iloc[-1]
        impulse = data[[f"linear_impulse_{c}" for c in "xyz"]].to_numpy()
        # A head-on pair has zero net impulse: normalize by the sum of the
        # individual initial ring impulse magnitudes, not the vanishing sum.
        ring_initial = rings[rings.step == rings.step.min()]
        impulse_scale = float(ring_initial.linear_impulse_magnitude.sum())
        signature = meta["signature"]
        row = {
            "run": name,
            "scenario": signature["scenario"],
            "method": signature["method"],
            "spacing": signature.get("spacing", setup.PARTICLE_SPACING),
            "support": signature.get("support", "circular"),
            "amplitude": signature.get("amplitude", setup.DISTURBANCE_AMPLITUDE),
            "dt": signature.get("dt", setup.TIME_STEP_SIZE),
            "smagorinsky": signature.get("smagorinsky", setup.SMAGORINSKY_COEFFICIENT),
            "frequency": signature.get("frequency", np.nan),
            "status": meta["status"],
            "steps": meta.get("completed_steps", int(final.step)),
            "sampled_t_star": final.time * setup.RING_CIRCULATION,
            "wall_seconds": meta.get("wall_seconds", np.nan),
            "runtime_energy_ratio": final.total_kinetic_energy / initial.total_kinetic_energy,
            "max_runtime_energy_ratio": data.total_kinetic_energy.max()
            / initial.total_kinetic_energy,
            "max_impulse_drift": float(
                np.linalg.norm(impulse - impulse[0], axis=1).max() / impulse_scale
            ),
            "particles": int(final.n_particles_total),
            "events": int(final.n_stabilization_events),
            "remesh_events": int(final.n_regularization_events),
            "max_divergence": data.vorticity_divergence_error.max(),
            "max_misalignment": data.vortex_strength_misalignment_degrees.max(),
            "max_cfl": data.lagrangian_cfl.max(),
            "trajectory_radius_rmse": trajectory_rmse(rings)
            if signature["scenario"] == "leapfrog" and not final.n_regularization_events
            else np.nan,
            "termination": meta.get("termination_reason"),
        }
        for tstar in (2, 5, 10, 15, 20):
            target = tstar / setup.RING_CIRCULATION
            row[f"runtime_energy_at_tstar_{tstar}"] = (
                np.interp(target, data.time, data.total_kinetic_energy)
                / initial.total_kinetic_energy
                if target <= final.time
                else np.nan
            )
        for state in states(name):
            for mode in mode_amplitudes(
                state["position"], state["vortex_strength"], state["group_id"]
            ):
                modes.append(
                    {
                        "run": name,
                        "time": float(state["time"]),
                        "t_star": float(state["time"]) * setup.RING_CIRCULATION,
                        **mode,
                    }
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(modes).to_csv(output / "ring_modes.csv", index=False)
    for (scenario, spacing, support, amplitude, cs), matched in table.groupby(
        ["scenario", "spacing", "support", "amplitude", "smagorinsky"]
    ):
        selected = [run for run in runs if run[0] in set(matched.run)]
        suffix = f"{scenario}_h{spacing:g}_{support}_a{amplitude:g}_Cs{cs:g}"
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for name, _, _, data, rings in selected:
            t = data.time * setup.RING_CIRCULATION
            label = name.removeprefix(scenario + "_")
            (line,) = axes[0, 0].plot(
                t, data.total_kinetic_energy / data.total_kinetic_energy.iloc[0], label=label
            )
            color = line.get_color()
            axes[0, 1].plot(t, data.vorticity_divergence_error, color=color)
            axes[0, 2].plot(t, data.vortex_strength_misalignment_degrees, color=color)
            axes[1, 0].plot(t, data.n_particles_total / 1000, color=color)
            for group in (0, 1):
                ring = rings[rings.group_id == group]
                axes[1, 1].plot(
                    ring.time * setup.RING_CIRCULATION,
                    ring.major_radius,
                    linestyle="-" if group == 0 else "--",
                    color=color,
                )
                axes[1, 2].plot(
                    ring.time * setup.RING_CIRCULATION,
                    ring.vortex_centroid_x,
                    linestyle="-" if group == 0 else "--",
                    color=color,
                )
        for axis, label in zip(
            axes.flat,
            (
                "E/E0 (runtime estimate; see field audit)",
                "Relative divergence",
                "Misalignment (degrees)",
                "Particles (thousands)",
                "Ring radius proxy / R0",
                "Ring centroid x / R0",
            ),
        ):
            axis.set(xlabel="t Γ0 / R0²", ylabel=label)
            axis.grid(alpha=0.2)
        axes[0, 1].axhline(0.12, color="black", linestyle=":", linewidth=1)
        axes[0, 2].axhline(25, color="black", linestyle=":", linewidth=1)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"{scenario.capitalize()} — h={spacing:g}, {support} support, amplitude={amplitude:g}, Cs={cs:g}"
        )
        fig.savefig(output / f"{suffix}_comparison.png", dpi=150)
        plt.close(fig)
        if modes:
            mode_table = pd.DataFrame(modes)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            for name, *_ in selected:
                series = mode_table[mode_table.run == name]
                for group in (0, 1):
                    tube = series[series.group_id == group]
                    for axis, quantity in zip(axes, ("radial_mode8", "axial_mode8")):
                        axis.plot(
                            tube.t_star,
                            tube[quantity],
                            label=f"{name}, ring {group}",
                            linestyle="-" if group == 0 else "--",
                        )
            for axis, label in zip(axes, ("Radial mode 8 / R0", "Axial mode 8 / R0")):
                axis.set(xlabel="t Γ0 / R0²", ylabel=label)
                axis.grid(alpha=0.2)
            axes[0].legend(fontsize=7)
            fig.suptitle(
                "Azimuthal centroid modes — interpret only while labels remain coherent tubes"
            )
            fig.savefig(output / f"{suffix}_modes.png", dpi=150)
            plt.close(fig)
    lines = [
        "# Recorded interaction experiments",
        "",
        "Survival is not a physics ranking. Runtime energies require a common-box field audit before comparison; old FFT values are superseded by remeasurement.",
        "",
        table[
            [
                "run",
                "status",
                "steps",
                "sampled_t_star",
                "runtime_energy_ratio",
                "max_impulse_drift",
                "particles",
                "events",
                "wall_seconds",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "The energy diagnostic uses a periodic Fourier approximation for large variable-core clouds; changes of representation or Fourier box can alter it. Ring radii and label-based modes are geometric proxies after breakdown/remeshing. Trajectory RMSE is omitted after remeshing because labels are not material tracers.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(
        table[
            ["run", "status", "steps", "sampled_t_star", "runtime_energy_ratio", "particles"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures" / "study")
    args = parser.parse_args()
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit("No recorded experiments with diagnostic samples")
    summarize(runs, args.output)
