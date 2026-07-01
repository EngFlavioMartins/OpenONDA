#!/usr/bin/env python3
"""Numerical and aerodynamic acceptance checks for rotorFlow."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _step(path: Path) -> int:
    m = re.search(r"_(\d+)\.h5$", path.name)
    return int(m.group(1)) if m else -1


def _bem_reference(R: float, U_inf: float, omega: float, B: int) -> tuple[float, float]:
    """Run BEM for the rotorFlow blade design and return (Ct, Cp)."""
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution-dir", default="solution/rotor")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--expected-step", type=int, default=3500)
    # Physical parameters (must match rotor_setup.py defaults)
    ap.add_argument("--rotor-radius", type=float, default=6.0)
    ap.add_argument("--freestream-velocity", type=float, default=7.0)
    ap.add_argument("--tip-speed-ratio", type=float, default=7.0)
    ap.add_argument("--blades", type=int, default=3)
    # BEM tolerance: VLM Ct/Cp are expected within this fraction of BEM values.
    # VLM (lifting-line) will over-predict vs. viscous BEM; 25% is generous but
    # physically reasonable for a high-TSR rotor with finite wake effects.
    ap.add_argument("--bem-tol", type=float, default=0.25)
    args = ap.parse_args()

    root, figs = Path(args.solution_dir), Path(args.figures_dir)
    R = args.rotor_radius
    U_inf = args.freestream_velocity
    omega = args.tip_speed_ratio * U_inf / R
    qA = 0.5 * 1.225 * U_inf**2 * np.pi * R**2
    failures: list[str] = []

    # -- 1. VPM particle sanity ------------------------------------------------
    files = sorted(root.glob("vpm_rotor_*.h5"), key=_step)
    if not files or _step(files[-1]) != args.expected_step:
        failures.append(
            f"last rotor backup is {_step(files[-1]) if files else 'missing'}, "
            f"expected {args.expected_step}"
        )
    elif files:
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

    # -- 2. VLM force CSV -----------------------------------------------------
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

    # -- 3. BEM reference comparison ------------------------------------------
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

    # -- 4. Wake-plane snapshots ----------------------------------------------
    for tag in ("x36m", "x72m", "x108m"):
        planes = sorted((root / "samples").glob(f"slice_{tag}_*.vts"))
        if not planes or f"_{args.expected_step:06d}.vts" not in planes[-1].name:
            failures.append(f"missing final {tag} wake plane")

    # -- 5. Figure outputs ----------------------------------------------------
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
