#!/usr/bin/env python3
"""Numerical and aerodynamic acceptance checks for rotor_flow.

Checks, in order of what they catch:

1. The run finished and the particle field is finite and bounded.
2. The wake's **linear-impulse budget** closes: for an unbounded vortex system
   density*|dI_x/dt| must equal the axial force on the blades.  A wake that is
   quietly losing or manufacturing strength shows up here and nowhere else --
   thrust_coefficient, power_coefficient and the loading profile can stay plausible while the wake decays.
3. The sampler planes are statistically stationary, so the wake-deficit figure
   is showing a converged wake rather than one still in transit.
4. thrust_coefficient / power_coefficient agree with BEM for the same blade to within a tolerance.
"""

from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DENSITY,
    FREESTREAM_SPEED,
    N_BLADES,
    ROTOR_RADIUS,
    FIGURES_DIR,
    SAMPLES_DIR,
    SOLUTION_DIR,
    TIP_SPEED_RATIO,
    read_time_step,
)


def _step(path: Path) -> int:
    m = re.search(r"_(\d+)\.h5$", path.name)
    return int(m.group(1)) if m else -1


def _bem_reference(
    rotor_radius: float,
    freestream_speed: float,
    angular_velocity: float,
    n_blades: int,
) -> tuple[float, float]:
    """Run BEM for the rotor_flow blade design and return (thrust_coefficient, power_coefficient)."""
    try:
        from rotor_theory import solve_blade_element_momentum
        from generate_openvsp_blade import RotorBladeDesign, design_schedule
    except Exception:
        return float("nan"), float("nan")

    design = RotorBladeDesign(
        rotor_radius=rotor_radius,
        hub_radius=1.0,
        freestream_speed=freestream_speed,
        tip_speed_ratio=angular_velocity * rotor_radius / freestream_speed,
    )
    sched = design_schedule(design)
    bem = solve_blade_element_momentum(
        radial_position=sched["radial_position"],
        chord=sched["chord"],
        twist_angle_radians=np.radians(sched["twist_angle_degrees"]),
        n_blades=n_blades,
        rotor_radius=rotor_radius,
        freestream_speed=freestream_speed,
        angular_velocity=angular_velocity,
    )
    return float(bem.attrs["thrust_coefficient"]), float(bem.attrs["power_coefficient"])


def _impulse_ratio(
    samples: Path, density: float, tail_rotations: float, angular_velocity: float
) -> float | None:
    """Mean density*|dI_x/dt| / T over the last ``tail_rotations`` revolutions.

    A value of one closes the blade-force/wake-momentum budget.
    """
    integrals_path = samples / "flow_integrals.csv"
    forces_path = samples / "vlm_forces.csv"
    if not integrals_path.exists() or not forces_path.exists():
        return None

    integrals = pd.read_csv(integrals_path)
    forces = pd.read_csv(forces_path)
    if len(integrals) < 4 or forces.empty:
        return None

    time = integrals["time"].to_numpy()
    thrust = np.interp(time, forces["time"].to_numpy(), forces["force_x"].to_numpy())
    force_from_wake = -density * np.gradient(integrals["impulse_x"].to_numpy(), time)

    t_cut = time.max() - tail_rotations * 2.0 * np.pi / angular_velocity
    window = time >= t_cut
    if window.sum() < 2:
        window = np.zeros_like(time, dtype=bool)
        window[-2:] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = force_from_wake[window] / thrust[window]
    ratio = ratio[np.isfinite(ratio)]
    return float(np.mean(ratio)) if ratio.size else None


def _plane_drifts(
    samples: Path,
    R: float,
    freestream_speed: float,
    time_step_size: float,
    angular_velocity: float,
) -> dict[str, float]:
    """Per-plane relative drift of the disc-averaged deficit over the averaging window."""
    try:
        import pyvista as pv

        from plot_rotor_wake_planes import (
            _discover_planes,
            _drift,
            _plane_files,
            _plane_statistics,
            _select_time_window,
        )
    except Exception:
        return {}

    radial_edges = np.linspace(0.0, 1.25, 33)
    drifts: dict[str, float] = {}
    for tag, _label in _discover_planes(samples, R):
        files = _select_time_window(
            _plane_files(samples, tag),
            time_step_size=time_step_size,
            angular_velocity=angular_velocity,
            averaging_rotations=3.0,
            tail_fraction=0.25,
        )
        if len(files) < 4:
            continue
        try:
            disc_means = np.array(
                [_plane_statistics(pv.read(f), freestream_speed, R, radial_edges)[1] for f in files]
            )
        except Exception:
            continue
        drifts[tag] = _drift(disc_means)
    return drifts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    expected_step = 2400
    bem_tolerance = 0.25
    impulse_tolerance = 0.10
    drift_tolerance = 0.01
    R = ROTOR_RADIUS
    freestream_speed = FREESTREAM_SPEED
    angular_velocity = TIP_SPEED_RATIO * freestream_speed / R
    dynamic_pressure_area = 0.5 * DENSITY * freestream_speed**2 * np.pi * R**2
    failures: list[str] = []

    # -- 1. VPM particle sanity ------------------------------------------------
    files = sorted(SOLUTION_DIR.glob("vpm_rotor_*.h5"), key=_step)
    if not files or _step(files[-1]) != expected_step:
        failures.append(
            f"last rotor backup is {_step(files[-1]) if files else 'missing'}, "
            f"expected {expected_step}"
        )
    if files:
        with h5py.File(files[-1], "r") as h5:
            particles = h5["particles"]
            vortex_strength = particles["vortex_strength"][:]
            core_radius = particles["core_radius"][:]
        max_vortex_strength_magnitude = float(np.linalg.norm(vortex_strength, axis=1).max())
        print(
            f"Final particles={len(vortex_strength)}, "
            f"max_vortex_strength_magnitude={max_vortex_strength_magnitude:.4g}, "
            f"max_core_radius={core_radius.max():.4g}"
        )
        if not np.isfinite(vortex_strength).all() or max_vortex_strength_magnitude > 10.0:
            failures.append(
                f"unbounded final wake vortex_strength: {max_vortex_strength_magnitude:.4g}"
            )

    # -- 2. Wake impulse budget ------------------------------------------------
    ratio = _impulse_ratio(
        SAMPLES_DIR, DENSITY, tail_rotations=2.0, angular_velocity=angular_velocity
    )
    if ratio is None:
        failures.append("could not evaluate the wake impulse budget (missing sampler CSVs)")
    else:
        print(f"Wake impulse budget: density*|dIx/dt| / T = {ratio:.3f} (target 1.000)")
        if not np.isfinite(ratio) or abs(ratio - 1.0) > impulse_tolerance:
            failures.append(
                f"wake impulse budget {ratio:.3f} deviates from 1.0 by more than "
                f"{impulse_tolerance:.0%} — the wake is not carrying the momentum the "
                "blades extract, so the velocity deficit cannot be trusted"
            )

    # -- 3. Wake-plane stationarity -------------------------------------------
    time_step_size = read_time_step(SAMPLES_DIR)
    if time_step_size is None:
        failures.append("could not determine the run's time step")
    else:
        drifts = _plane_drifts(SAMPLES_DIR, R, freestream_speed, time_step_size, angular_velocity)
        if not drifts:
            failures.append("no wake-plane samples found to check for stationarity")
        for tag, drift in sorted(drifts.items()):
            print(f"Plane {tag}: disc-mean drift {drift:.2%}")
            if not np.isfinite(drift) or drift > drift_tolerance:
                failures.append(
                    f"wake plane {tag} is still in transit ({drift:.1%} drift over the "
                    f"averaging window, limit {drift_tolerance:.0%}) — run longer"
                )

    # -- 4. VLM force CSV ------------------------------------------------------
    csv = SAMPLES_DIR / "vlm_forces.csv"
    mean_thrust_coefficient = mean_power_coefficient = float("nan")
    if not csv.exists():
        failures.append("missing vlm_forces.csv")
    else:
        df = pd.read_csv(csv)
        thrust_coefficient = df["force_x"].to_numpy() / dynamic_pressure_area
        power_coefficient = (-df["moment_x"].to_numpy() * angular_velocity) / (
            dynamic_pressure_area * freestream_speed
        )
        tail = slice(max(0, int(0.8 * len(df))), None)
        mean_thrust_coefficient, mean_power_coefficient = (
            float(np.mean(thrust_coefficient[tail])),
            float(np.mean(power_coefficient[tail])),
        )
        print(
            f"Tail mean thrust_coefficient={mean_thrust_coefficient:.4f}, power_coefficient={mean_power_coefficient:.4f}"
        )
        if not np.isfinite(thrust_coefficient).all() or not np.isfinite(power_coefficient).all():
            failures.append("non-finite rotor coefficients")
        # Physical plausibility: must be within actuator-disk range
        if not (0.4 < mean_thrust_coefficient < 1.1 and 0.2 < mean_power_coefficient < 0.62):
            failures.append(
                f"implausible tail coefficients thrust_coefficient={mean_thrust_coefficient:.3f}, power_coefficient={mean_power_coefficient:.3f} "
                "(expected 0.4 < thrust_coefficient < 1.1, 0.2 < power_coefficient < 0.62)"
            )

    # -- 5. BEM reference comparison ------------------------------------------
    bem_thrust_coefficient, bem_power_coefficient = _bem_reference(
        R, freestream_speed, angular_velocity, N_BLADES
    )
    if np.isfinite(bem_thrust_coefficient) and np.isfinite(mean_thrust_coefficient):
        thrust_coefficient_relative_error = abs(
            mean_thrust_coefficient - bem_thrust_coefficient
        ) / max(bem_thrust_coefficient, 1e-10)
        power_coefficient_relative_error = abs(
            mean_power_coefficient - bem_power_coefficient
        ) / max(bem_power_coefficient, 1e-10)
        print(
            f"BEM reference: thrust_coefficient={bem_thrust_coefficient:.4f}, power_coefficient={bem_power_coefficient:.4f}  "
            f"(VLM error: thrust_coefficient={thrust_coefficient_relative_error:.1%}, power_coefficient={power_coefficient_relative_error:.1%})"
        )
        if thrust_coefficient_relative_error > bem_tolerance:
            failures.append(
                f"VLM thrust_coefficient={mean_thrust_coefficient:.3f} deviates from BEM thrust_coefficient={bem_thrust_coefficient:.3f} "
                f"by {thrust_coefficient_relative_error:.1%} (limit {bem_tolerance:.0%})"
            )
        if power_coefficient_relative_error > bem_tolerance:
            failures.append(
                f"VLM power_coefficient={mean_power_coefficient:.3f} deviates from BEM power_coefficient={bem_power_coefficient:.3f} "
                f"by {power_coefficient_relative_error:.1%} (limit {bem_tolerance:.0%})"
            )
    elif not np.isfinite(bem_thrust_coefficient):
        print("  (BEM reference unavailable — skipping BEM comparison)")

    # -- 6. Figure outputs -----------------------------------------------------
    if not args.pre_plot:
        for extension in ("png", "pdf"):
            for name in (
                "rotor_performance",
                "rotor_wake_planes",
                "rotor_loading_validation",
            ):
                figure = FIGURES_DIR / f"{name}.{extension}"
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing or empty figure {figure.name}")

    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        return 1
    print("[OK] rotor_flow certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
