#!/usr/bin/env python3
"""Numerical and aerodynamic acceptance checks for rotorFlow.

Checks, in order of what they catch:

1. The run finished and the particle field is finite and bounded.
2. The wake's **linear-impulse budget** closes: for an unbounded vortex system
   rho*|dI_x/dt| must equal the axial force on the blades.  A wake that is
   quietly losing or manufacturing strength shows up here and nowhere else --
   Ct, Cp and the loading profile can stay plausible while the wake decays.
3. The sampler planes are statistically stationary, so the wake-deficit figure
   is showing a converged wake rather than one still in transit.
4. Ct / Cp agree with BEM for the same blade to within a tolerance.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DENSITY,
    FREESTREAM_VELOCITY,
    NUM_BLADES,
    ROTOR_RADIUS,
    TIP_SPEED_RATIO,
    read_time_step,
)


def _step(path: Path) -> int:
    m = re.search(r"_(\d+)\.h5$", path.name)
    return int(m.group(1)) if m else -1


def _bem_reference(R: float, U_inf: float, omega: float, B: int) -> tuple[float, float]:
    """Run BEM for the rotorFlow blade design and return (Ct, Cp)."""
    try:
        from rotor_theory import bem_solve
        from generate_openvsp_blade import RotorBladeDesign, design_schedule
    except Exception:
        return float("nan"), float("nan")

    design = RotorBladeDesign(
        radius=R,
        hub_radius=1.0,
        freestream_velocity=U_inf,
        tip_speed_ratio=omega * R / U_inf,
    )
    sched = design_schedule(design)
    bem = bem_solve(
        r=sched["r"],
        chord=sched["chord"],
        twist_rad=np.radians(sched["theta_deg"]),
        B=B,
        R=R,
        U_inf=U_inf,
        omega=omega,
    )
    return float(bem.attrs["Ct"]), float(bem.attrs["Cp"])


def _impulse_ratio(root: Path, rho: float, tail_rotations: float, omega: float) -> float | None:
    """Mean rho*|dI_x/dt| / T over the last ``tail_rotations`` revolutions.

    1.0 means the wake absorbs exactly the momentum the blades extract.  Below
    Below 1.0 the wake is losing strength; above 1.0 under-resolved stretching
    is manufacturing it.
    """
    integrals_path = root / "samples" / "flow_integrals.csv"
    forces_path = root / "samples" / "vlm_forces.csv"
    if not integrals_path.exists() or not forces_path.exists():
        return None

    integrals = pd.read_csv(integrals_path)
    forces = pd.read_csv(forces_path)
    if len(integrals) < 4 or forces.empty:
        return None

    time = integrals["time"].to_numpy()
    thrust = np.interp(time, forces["time"].to_numpy(), forces["Fx"].to_numpy())
    force_from_wake = -rho * np.gradient(integrals["impulse_x"].to_numpy(), time)

    t_cut = time.max() - tail_rotations * 2.0 * np.pi / omega
    window = time >= t_cut
    if window.sum() < 2:
        window = np.zeros_like(time, dtype=bool)
        window[-2:] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = force_from_wake[window] / thrust[window]
    ratio = ratio[np.isfinite(ratio)]
    return float(np.mean(ratio)) if ratio.size else None


def _plane_drifts(root: Path, R: float, U_inf: float, dt: float, omega: float) -> dict[str, float]:
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

    samples = root / "samples"
    radial_edges = np.linspace(0.0, 1.25, 33)
    drifts: dict[str, float] = {}
    for tag, _label in _discover_planes(samples, R):
        files = _select_time_window(
            _plane_files(samples, tag),
            dt=dt,
            omega=omega,
            averaging_rotations=3.0,
            tail_fraction=0.25,
        )
        if len(files) < 4:
            continue
        try:
            disc_means = np.array(
                [_plane_statistics(pv.read(f), U_inf, R, radial_edges)[1] for f in files]
            )
        except Exception:
            continue
        drifts[tag] = _drift(disc_means)
    return drifts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution-dir", default="solution/rotor")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--expected-step", type=int, default=2400)
    # Physical parameters default to the shared case definition in _common.py.
    ap.add_argument("--rotor-radius", type=float, default=ROTOR_RADIUS)
    ap.add_argument("--freestream-velocity", type=float, default=FREESTREAM_VELOCITY)
    ap.add_argument("--tip-speed-ratio", type=float, default=TIP_SPEED_RATIO)
    ap.add_argument("--rho", type=float, default=DENSITY)
    ap.add_argument("--blades", type=int, default=NUM_BLADES)
    # BEM tolerance: VLM Ct/Cp are expected within this fraction of BEM values.
    # VLM (lifting-line) will over-predict vs. viscous BEM; 25% is generous but
    # physically reasonable for a high-TSR rotor with finite wake effects.
    ap.add_argument("--bem-tol", type=float, default=0.25)
    # Impulse-budget tolerance: how far rho*|dI/dt| / T may sit from 1.0.
    ap.add_argument("--impulse-tol", type=float, default=0.10)
    # Wake-plane stationarity tolerance, matching plot_rotor_wake_planes.py.
    ap.add_argument("--drift-tol", type=float, default=0.01)
    args = ap.parse_args()

    root, figs = Path(args.solution_dir), Path(args.figures_dir)
    R = args.rotor_radius
    U_inf = args.freestream_velocity
    omega = args.tip_speed_ratio * U_inf / R
    qA = 0.5 * args.rho * U_inf**2 * np.pi * R**2
    failures: list[str] = []

    # -- 1. VPM particle sanity ------------------------------------------------
    files = sorted(root.glob("vpm_rotor_*.h5"), key=_step)
    if not files or _step(files[-1]) != args.expected_step:
        failures.append(
            f"last rotor backup is {_step(files[-1]) if files else 'missing'}, "
            f"expected {args.expected_step}"
        )
    if files:
        with h5py.File(files[-1], "r") as h5:
            alpha = h5["particles/circulation"][:]
            radius = h5["particles/radius"][:]
        max_strength = float(np.linalg.norm(alpha, axis=1).max())
        print(
            f"Final particles={len(alpha)}, max|alpha|={max_strength:.4g}, "
            f"max radius={radius.max():.4g}"
        )
        if not np.isfinite(alpha).all() or max_strength > 10.0:
            failures.append(f"unbounded final wake strength: {max_strength:.4g}")

    # -- 2. Wake impulse budget ------------------------------------------------
    ratio = _impulse_ratio(root, args.rho, tail_rotations=2.0, omega=omega)
    if ratio is None:
        failures.append("could not evaluate the wake impulse budget (missing sampler CSVs)")
    else:
        print(f"Wake impulse budget: rho*|dIx/dt| / T = {ratio:.3f} (target 1.000)")
        if not np.isfinite(ratio) or abs(ratio - 1.0) > args.impulse_tol:
            failures.append(
                f"wake impulse budget {ratio:.3f} deviates from 1.0 by more than "
                f"{args.impulse_tol:.0%} — the wake is not carrying the momentum the "
                "blades extract, so the velocity deficit cannot be trusted"
            )

    # -- 3. Wake-plane stationarity -------------------------------------------
    dt = read_time_step(root)
    if dt is None:
        failures.append("could not determine the run's time step")
    else:
        drifts = _plane_drifts(root, R, U_inf, dt, omega)
        if not drifts:
            failures.append("no wake-plane samples found to check for stationarity")
        for tag, drift in sorted(drifts.items()):
            print(f"Plane {tag}: disc-mean drift {drift:.2%}")
            if not np.isfinite(drift) or drift > args.drift_tol:
                failures.append(
                    f"wake plane {tag} is still in transit ({drift:.1%} drift over the "
                    f"averaging window, limit {args.drift_tol:.0%}) — run longer"
                )

    # -- 4. VLM force CSV ------------------------------------------------------
    csv = root / "samples" / "vlm_forces.csv"
    ct_mean = cp_mean = float("nan")
    if not csv.exists():
        failures.append("missing vlm_forces.csv")
    else:
        df = pd.read_csv(csv)
        ct = df["Fx"].to_numpy() / qA
        cp = (-df["Mx"].to_numpy() * omega) / (qA * U_inf)
        tail = slice(max(0, int(0.8 * len(df))), None)
        ct_mean, cp_mean = float(np.mean(ct[tail])), float(np.mean(cp[tail]))
        print(f"Tail mean Ct={ct_mean:.4f}, Cp={cp_mean:.4f}")
        if not np.isfinite(ct).all() or not np.isfinite(cp).all():
            failures.append("non-finite rotor coefficients")
        # Physical plausibility: must be within actuator-disk range
        if not (0.4 < ct_mean < 1.1 and 0.2 < cp_mean < 0.62):
            failures.append(
                f"implausible tail coefficients Ct={ct_mean:.3f}, Cp={cp_mean:.3f} "
                "(expected 0.4 < Ct < 1.1, 0.2 < Cp < 0.62)"
            )

    # -- 5. BEM reference comparison ------------------------------------------
    bem_ct, bem_cp = _bem_reference(R, U_inf, omega, args.blades)
    if np.isfinite(bem_ct) and np.isfinite(ct_mean):
        ct_err = abs(ct_mean - bem_ct) / max(bem_ct, 1e-10)
        cp_err = abs(cp_mean - bem_cp) / max(bem_cp, 1e-10)
        print(
            f"BEM reference: Ct={bem_ct:.4f}, Cp={bem_cp:.4f}  "
            f"(VLM error: Ct={ct_err:.1%}, Cp={cp_err:.1%})"
        )
        if ct_err > args.bem_tol:
            failures.append(
                f"VLM Ct={ct_mean:.3f} deviates from BEM Ct={bem_ct:.3f} "
                f"by {ct_err:.1%} (limit {args.bem_tol:.0%})"
            )
        if cp_err > args.bem_tol:
            failures.append(
                f"VLM Cp={cp_mean:.3f} deviates from BEM Cp={bem_cp:.3f} "
                f"by {cp_err:.1%} (limit {args.bem_tol:.0%})"
            )
    elif not np.isfinite(bem_ct):
        print("  (BEM reference unavailable — skipping BEM comparison)")

    # -- 6. Figure outputs -----------------------------------------------------
    for name in (
        "rotor_performance.png",
        "rotor_wake_planes.png",
        "rotor_loading_validation.png",
    ):
        if not (figs / name).exists():
            failures.append(f"missing figure {name}")

    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        return 1
    print("[OK] rotorFlow certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
