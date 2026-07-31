#!/usr/bin/env python3
"""
D4 Diagnostic: VPM → VLM Induced Downwash per Span Station
===========================================================
Loads the final VPM particle snapshot and evaluates VPM-induced velocity at
the VLM collocation points (at their correct final-step lab-frame position)
in a single forward pass — no time-marching. Groups by span station and
compares w_j = V_VPM · n_hat to the Glauert-required induced AoA.

This quantifies whether the VPM wake delivers a flat or tip-peaked downwash
profile, localising the source of the flat-loading Bug 2e deficiency.

Usage:
    cd tutorials/VPM/flatPlate
    python assets/diag_downwash.py
    python assets/diag_downwash.py --h5 solution/exp_moving_aoa05/vpm_exp_moving_aoa05_000306.h5

Output:
    solution/exp_moving_aoa05/samples/exp_moving_aoa05_downwash.csv

Author:  Flavio A. C. Martins, OpenONDA Team
Date: June 2026
"""

from __future__ import annotations

import os, sys, argparse, math
import numpy as np
import pandas as pd
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CASE_DIR = _SCRIPT_DIR.parent
from openonda.vpm import BackupSystem, Solver, VPMSetup
from openonda.vpm import (
    ForceConfig,
    VLMMeshSetup,
    VLMSurfaceSetup,
    VLMSetup,
)
from openonda.vpm import VLMLoadingDistribution
from openonda.vpm import SmoothRampVLM

from generate_surface import create_flat_plate, save_surface
from theoretical_model import liftingline_circulation


def main():
    os.chdir(_CASE_DIR)

    p = argparse.ArgumentParser(
        description="D4: VPM→VLM downwash diagnostic (no time-marching)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--h5",
        default="solution/exp_moving_aoa05/vpm_exp_moving_aoa05_000306.h5",
        help="VPM particle snapshot to load",
    )
    p.add_argument("--steps", type=int, default=306, help="Simulation step count for displacement")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--aoa", type=float, default=5.0)
    p.add_argument("--span", type=float, default=10.0)
    p.add_argument("--chord", type=float, default=1.0)
    p.add_argument("--U-inf", type=float, default=10.0)
    p.add_argument("--tau-ramp", type=float, default=0.6, help="Ramp length [chord-lengths]")
    p.add_argument("--panels-chord", type=int, default=4)
    p.add_argument("--panels-span", type=int, default=16)
    p.add_argument("--name", default="exp_moving_aoa05")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    U = args.U_inf
    c = args.chord
    dt = args.dt
    h5_path = str(Path(args.h5).resolve())
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"solution/{args.name}/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Surface geometry (body frame: plate tilted at AoA) --------------------
    surface_file = str(_CASE_DIR / "assets" / "flat_plate_surface.json")
    surface = create_flat_plate(
        chord=c, span=args.span, alpha=args.aoa, n_chord=args.panels_chord, n_span=args.panels_span
    )
    save_surface(surface, surface_file)

    # -- VLM solver ------------------------------------------------------------
    U_ref = np.array([U, 0.0, 0.0])
    t_ramp = 2.0 * args.tau_ramp * c / U
    kin = SmoothRampVLM(U_final=[-U, 0.0, 0.0], acceleration_time=t_ramp)

    vlm_setup = VLMSetup(
        surfaces=(VLMSurfaceSetup(surface_file, kinematics=kin),),
        mesh=VLMMeshSetup.geometric(ratio=4.0, region="end"),
        density=1.0,
        viscosity=1e-2,
        freestream_velocity=tuple(U_ref),
        force=ForceConfig.kutta_joukowski(),
        sigma_factor=2.5,
        sample_surface_forces=True,
    )

    # -- Minimal Solver (initialises ti.init + VPM physics) -------------------
    _tmp_dir = "/tmp/diag_downwash"
    Path(_tmp_dir).mkdir(parents=True, exist_ok=True)
    solver = Solver(
        setup=VPMSetup.les_simulation(
            cs=0.30,
            time_step_size=dt,
            vlm=vlm_setup,
            background_velocity=[0.0, 0.0, 0.0],
            logging_frequency=999999,
            backup_frequency=999999,
            backup_file_name="diag_downwash",
            backup_directory=_tmp_dir,
        )
    )
    vlm = solver.vlm_solver
    if vlm is None:
        raise RuntimeError("Downwash diagnostic requires its declared VLM solver")

    # -- Plate displacement at final step (pure translation, sin²-ramp) --------
    # v(t) = 0.5*U*(1 - cos(π*t/t_ramp)) for t ≤ t_ramp, then U constant.
    # integral → dist_ramp = 0.5*U*t_ramp; dist_cruise = U*(t_total - t_ramp).
    t_total = args.steps * dt
    if t_total >= t_ramp:
        dist = 0.5 * U * t_ramp + U * (t_total - t_ramp)
    else:
        dist = 0.5 * U * (t_total - (t_ramp / math.pi) * math.sin(math.pi * t_total / t_ramp))
    displacement = np.array([-dist, 0.0, 0.0])
    print(f"\n  Plate displacement at step {args.steps}: dx={displacement[0]:.3f} m")

    # -- Collocation points and normals at final plate position ----------------
    n_p = vlm.lattice.num_panels
    coll_init = vlm.lattice.collocation.to_numpy()[:n_p]  # (N, 3) — initial geometry
    coll_final = coll_init + displacement  # translated to step-306 position
    normals_np = vlm.lattice.normals.to_numpy()[:n_p]  # (N, 3) — pure translation, same normals
    bm_all = vlm.lattice.bound_midpoints.to_numpy()[:n_p]  # for y_station

    print(
        f"  Collocation x range at final step: [{coll_final[:, 0].min():.2f}, {coll_final[:, 0].max():.2f}]"
    )

    # -- Load final particles --------------------------------------------------
    BackupSystem._load_numerical_data(solver, h5_path)
    n_part = len(solver.particles)
    print(f"  Particles loaded: {n_part}")

    # -- VPM-induced velocity at collocation points (no freestream) -----------
    v_vpm = solver.compute_target_velocities(coll_final, include_freestream=False)
    v_mag = np.linalg.norm(v_vpm, axis=1)
    print(f"  v_VPM: min={v_mag.min():.4f}  max={v_mag.max():.4f}  mean={v_mag.mean():.4f}")

    # -- Per-station downwash w_j = V_VPM · n_hat -----------------------------
    blocks = VLMLoadingDistribution.build_surface_grid_index(vlm, "flat_plate")
    rows: list[dict] = []
    for blk in blocks:
        ns, nc = blk["ns"], blk["nc"]
        for half, idx2d in [("orig", blk["orig_idx"]), ("mirror", blk["mirror_idx"])]:
            if idx2d is None:
                continue
            for j in range(ns):
                cidx = idx2d[j]
                w_panels = np.einsum("ki,ki->k", v_vpm[cidx], normals_np[cidx])
                rows.append(
                    {
                        "half": half,
                        "span_index": j,
                        "y": float(np.mean(bm_all[cidx, 1])),
                        "w_VPM": float(np.mean(w_panels)),
                        "w_VPM_std": float(np.std(w_panels)),
                    }
                )

    df_dw = pd.DataFrame(rows)
    orig = df_dw[df_dw.half == "orig"].sort_values("span_index").reset_index(drop=True)
    y_sta = orig["y"].to_numpy()
    w_VPM = orig["w_VPM"].to_numpy()

    # -- Glauert reference -----------------------------------------------------
    df_ll = liftingline_circulation(
        y_sta, b=args.span, c=c, alpha_rad=math.radians(args.aoa), U_inf=U
    )
    cl_gl = df_ll["cl"].to_numpy()
    # Induced AoA: α_i = α − α_eff  where α_eff = arcsin(cl/(2π)) ≈ cl/(2π)
    alpha_rad_val = math.radians(args.aoa)
    alpha_i_req = np.degrees(alpha_rad_val - cl_gl / (2.0 * math.pi))
    # VPM-delivered induced AoA: positive downwash (w<0) reduces effective AoA
    alpha_i_VPM = np.degrees(np.arctan2(-w_VPM, U))
    ratio = np.where(np.abs(alpha_i_req) > 0.01, alpha_i_VPM / alpha_i_req, float("nan"))

    # -- Print table -----------------------------------------------------------
    print(
        f"\n  --- D4: VPM→VLM induced downwash per station (step {args.steps}, τ={(args.steps * dt * U / c):.1f}c) ---"
    )
    print(
        f"  {'j':>3}  {'y/b':>6}  {'w_VPM[m/s]':>11}  {'α_VPM[°]':>9}  {'α_req[°]':>9}  {'ratio':>6}"
    )
    for k in range(len(y_sta)):
        yob = 2.0 * y_sta[k] / args.span
        print(
            f"  {k:>3}  {yob:>6.3f}  {w_VPM[k]:>11.4f}  "
            f"{alpha_i_VPM[k]:>9.3f}  {alpha_i_req[k]:>9.3f}  {ratio[k]:>6.3f}"
        )

    # -- Save CSV --------------------------------------------------------------
    df_out = orig.copy()
    df_out["y_over_b"] = 2.0 * df_out["y"] / args.span
    df_out["alpha_i_VPM_deg"] = alpha_i_VPM
    df_out["alpha_i_required_deg"] = alpha_i_req
    df_out["delivery_ratio"] = ratio
    dw_csv = out_dir / f"{args.name}_downwash.csv"
    df_out.to_csv(dw_csv, index=False)
    print(f"\n  Saved: {dw_csv}\n")

    # -- Summary ---------------------------------------------------------------
    root_ratio = ratio[0]
    tip_ratio = ratio[-1]
    print(f"  delivery_ratio: root={root_ratio:.3f}  tip={tip_ratio:.3f}")
    print(f"  (1.0 = VPM delivers exactly the required Glauert downwash)")
    print(f"  (< 1.0 = under-delivered; deficit explains flat loading)\n")


if __name__ == "__main__":
    main()
