#!/usr/bin/env python3
"""
Flat Plate VLM-VPM Setup Runner
================================
Single-case parametric runner for all flat-plate VLM-VPM experiments.
All physical and numerical settings are passed via command-line arguments.

The allrun.sh bash script hard-codes one call per experiment, giving a
clear, auditable record of every configuration that was run.

Supported configurations
-------------------------
Kinematics  : static, ramp (impulsive start), pitching (harmonic)
Frame       : wind (fixed wing, moving air) or body (moving wing, still air)
Force       : Kutta-Joukowski
Wake        : truncation, conservative remeshing, or none

Usage example::

    python setup_plate.py \
        --name exp_moving_aoa05 \
        --kinematics ramp --frame body \
        --aoa 5 --span 10 --tau-max 5.5 --dt 0.01

Author:  Flavio A. C. Martins, OpenONDA Team
Date: March 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Ensure OpenONDA is importable when run from any working directory
# ------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_CASE_DIR = _SCRIPT_DIR  # tutorial root (flatPlate/)
_REPO_ROOT = _SCRIPT_DIR.parents[2]  # OpenONDA/
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_CASE_DIR / "assets"))

from source.solvers.VPM import Solver, SolverConfig
from source.solvers.VPM.boundary_elements.vlm import VLMSolver, ForceConfig
from source.solvers.VPM.boundary_elements.vlm.solver import VLMLoadingDistribution
from source.solvers.VPM.boundary_elements.vlm.coupling.kinematics import (
    StaticVLM,
    TranslatingVLM,
    SmoothRampVLM,
    PitchingVLM,
)
from source.solvers.VPM.config.types import AdaptationConfig
from source.solvers.VPM.utils.field_samplers import SurfaceSampler

from generate_surface import create_flat_plate, save_surface


# ======================================================================
# Argument parser
# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified flat-plate VLM-VPM runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Output ────────────────────────────────────────────────────────
    p.add_argument(
        "--name", required=True, help="Case name; CSV saved as solution/{name}/samples/{name}.csv"
    )
    p.add_argument(
        "--backup-freq", type=int, default=0, help="XDMF backup frequency (0 = disabled)"
    )
    p.add_argument("--log-freq", type=int, default=5, help="CSV logging frequency [steps]")

    # ── Geometry ──────────────────────────────────────────────────────
    p.add_argument("--chord", type=float, default=1.0, help="Chord [m]")
    p.add_argument("--span", type=float, default=5.0, help="Full span [m]")
    p.add_argument(
        "--aoa", type=float, default=0.0, help="Angle of attack [deg] (static/ramp only)"
    )
    p.add_argument("--panels-chord", type=int, default=8, help="Chordwise panels")
    p.add_argument("--panels-span", type=int, default=16, help="Spanwise panels per side")

    # ── Kinematics ────────────────────────────────────────────────────
    p.add_argument(
        "--kinematics", choices=["static", "ramp", "pitching"], default="static", help="Motion type"
    )
    p.add_argument(
        "--frame",
        choices=["body", "wind"],
        default="wind",
        help='Reference frame: "wind" = static wing in freestream; '
        '"body" = moving wing in quiescent fluid',
    )

    # ── Ramp parameters ───────────────────────────────────────────────
    p.add_argument(
        "--tau-ramp", type=float, default=0.6, help="Ramp length [chord-lengths] (ramp only)"
    )

    # ── Pitching parameters ───────────────────────────────────────────
    p.add_argument("--pitch-amplitude", type=float, default=2.0, help="Pitch amplitude [deg]")
    p.add_argument("--reduced-freq", type=float, default=0.3, help="Reduced frequency k = ωc/(2U)")
    p.add_argument(
        "--n-periods", type=int, default=8, help="Number of oscillation periods to simulate"
    )
    p.add_argument("--pts-per-period", type=int, default=60, help="Time steps per period")

    # ── Solver physics ────────────────────────────────────────────────
    p.add_argument("--U-inf", type=float, default=10.0, help="Freestream speed [m/s]")
    p.add_argument("--density", type=float, default=1.0, help="Fluid density [kg/m³]")
    p.add_argument("--viscosity", type=float, default=1e-2, help="Kinematic viscosity [m²/s]")
    p.add_argument("--cs", type=float, default=0.30, help="Smagorinsky constant")
    p.add_argument(
        "--sigma-factor",
        type=float,
        default=2.5,
        help="Near-wake particle core sizing factor (overlap floor = sigma_factor·V·dt)",
    )
    p.add_argument(
        "--steady-vlm",
        action="store_true",
        help="DIAGNOSTIC: skip time-marching; do one UNCOUPLED steady VLM solve "
        "(full semi-infinite horseshoe) and dump spanwise cl. Isolates whether the "
        "geometry/AIC alone produces the finite-wing taper.",
    )

    # ── Time control (priority: --steps > --tau-max / --n-periods) ────
    p.add_argument("--steps", type=int, default=None, help="Override total step count")
    p.add_argument(
        "--tau-max", type=float, default=5.5, help="Simulation end [chord-lengths] (static/ramp)"
    )
    p.add_argument(
        "--dt", type=float, default=None, help="Override time step [s] (auto=0.01 for static/ramp)"
    )

    # ── Wake adaptation ───────────────────────────────────────────────
    p.add_argument(
        "--cutoff-x",
        type=float,
        default=None,
        help="Truncate wake: remove particles beyond --cutoff-x chords "
        "downstream of the trailing edge",
    )
    p.add_argument(
        "--remesh-frequency", type=int, default=None, help="Remesh every N steps (None = disabled)"
    )
    p.add_argument("--remesh-spacing", type=float, default=0.1, help="Remeshing grid spacing [m]")

    # ── Field sampling ────────────────────────────────────────────────
    p.add_argument(
        "--sample-midspan",
        action="store_true",
        help="Attach mid-span SurfaceSampler (y = span/2 plane)",
    )
    p.add_argument(
        "--sample-spacing", type=float, default=0.08, help="Grid spacing for mid-span sampler [m]"
    )
    p.add_argument(
        "--sample-crossflow",
        action="store_true",
        help="Attach 3 streamwise-normal crossflow plane samplers at x = 5, 15, 25 "
        "(y in [-6, 6], z in [-0.5, 5.0]) to visualise the trailing wake.",
    )

    return p


# ======================================================================
# Time extent helper
# ======================================================================


def compute_time_params(args):
    """
    Return (dt, n_steps, t_ramp, chord_travel_fn).

    Priority:
      1. --steps  (explicit override)
      2. --n-periods * --pts-per-period  (pitching)
      3. --tau-max                        (static / ramp)
    """
    U = args.U_inf
    c = args.chord

    if args.kinematics == "pitching":
        omega = 2.0 * args.reduced_freq * U / c  # rad/s
        freq = omega / (2.0 * math.pi)  # Hz
        T = 1.0 / freq  # period [s]
        dt = args.dt if args.dt else T / args.pts_per_period
        if args.steps:
            n_steps = args.steps
        else:
            n_steps = int(round(args.n_periods * args.pts_per_period))
        t_ramp = 0.0

        def chord_travel_fn(step_indices, _dt=dt, _U=U, _c=c):
            return step_indices * _dt * _U / _c

    elif args.kinematics == "static":
        dt = args.dt if args.dt else 0.01
        t_ramp = 0.0
        if args.steps:
            n_steps = args.steps
        else:
            n_steps = int(round(args.tau_max * c / (U * dt)))

        def chord_travel_fn(step_indices, _dt=dt, _U=U, _c=c):
            return step_indices * _dt * _U / _c

    else:  # smooth translational ramp
        dt = args.dt if args.dt else 0.01

        t_ramp = 2.0 * args.tau_ramp * c / U  # ramp duration [s] (sin² ramp)
        if args.steps:
            n_steps = args.steps
        else:
            t_cruise = max(0.0, args.tau_max - args.tau_ramp) * c / U
            n_steps = int(round((t_ramp + t_cruise) / dt))

        def chord_travel_fn(step_indices, _dt=dt, _U=U, _c=c, _t_ramp=t_ramp):
            t_vals = step_indices * _dt
            # Displacement for sin² ramp: integral of 0.5*U*(1 - cos(π*t/t_ramp))
            t_r = max(_t_ramp, 1e-12)
            dist = np.where(
                t_vals < _t_ramp,
                0.5 * _U * (t_vals - (t_r / math.pi) * np.sin(math.pi * t_vals / t_r)),
                0.5 * _U * t_r + _U * (t_vals - _t_ramp),
            )
            return dist / _c

    return dt, n_steps, t_ramp, chord_travel_fn


# ======================================================================
# Build kinematics
# ======================================================================


def build_kinematics(args, t_ramp: float):
    """Return (kinematics_object, bg_velocity_list).

    Wind frame (static): plate is fixed, freestream encodes AoA.
    Body frame (ramp/translating): plate moves in quiescent fluid.
    Pitching: plate pitches about quarter-chord in freestream.
    """
    U = args.U_inf
    c = args.chord
    alpha_rad = math.radians(args.aoa)

    if args.kinematics == "pitching":
        omega = 2.0 * args.reduced_freq * U / c
        freq = omega / (2.0 * math.pi)
        kin = PitchingVLM(
            amplitude_deg=args.pitch_amplitude,
            frequency=freq,
            phase=0.0,
            pitch_axis=[0, 1, 0],
            pivot=[c / 4.0, 0.0, 0.0],
        )
        bg = [U, 0.0, 0.0]

    elif args.kinematics == "ramp":
        kin = SmoothRampVLM(
            U_final=[-U, 0.0, 0.0],
            acceleration_time=t_ramp,
        )
        bg = [0.0, 0.0, 0.0]

    else:  # static
        if args.frame == "body":
            kin = TranslatingVLM(velocity=np.array([-U, 0.0, 0.0]))
            bg = [0.0, 0.0, 0.0]
        else:
            kin = StaticVLM()
            bg = [U * math.cos(alpha_rad), 0.0, +U * math.sin(alpha_rad)]

    return kin, bg


# ======================================================================
# Main simulation
# ======================================================================


def main():
    os.chdir(_CASE_DIR)  # always run relative to tutorial directory

    parser = build_parser()
    args = parser.parse_args()

    print(f"\n{'=' * 65}")
    print(f"  Flat Plate VLM-VPM: {args.name}")
    print(f"  kinematics={args.kinematics}, frame={args.frame}, aoa={args.aoa}°")
    print(f"{'=' * 65}\n")

    # ── Derived parameters ────────────────────────────────────────────
    dt, n_steps, t_ramp, chord_travel_fn = compute_time_params(args)
    kin, bg_vel = build_kinematics(args, t_ramp)

    print(f"  dt={dt:.5f}s  n_steps={n_steps}  t_ramp={t_ramp:.3f}s")

    # ── Surface geometry ──────────────────────────────────────────────
    # Wind-frame static: AoA in background velocity, plate is horizontal.
    # Body-frame and pitching: AoA encoded in plate geometry.
    if args.kinematics == "static" and args.frame == "wind":
        geom_aoa = 0.0
    elif args.kinematics == "pitching":
        geom_aoa = 0.0  # mean attitude = 0; pitching adds ±amplitude_deg
    else:
        geom_aoa = args.aoa

    surface_file = str(_CASE_DIR / "assets" / "flat_plate_surface.json")
    surface = create_flat_plate(
        chord=args.chord,
        span=args.span,
        alpha=geom_aoa,
        n_chord=args.panels_chord,
        n_span=args.panels_span,
    )
    save_surface(surface, surface_file)

    # ── Force configuration ────────────────────────────────────────────
    force_cfg = ForceConfig.kutta_joukowski()

    alpha_rad = math.radians(args.aoa)

    # ── VLM solver ────────────────────────────────────────────────────
    # U_ref encodes the aerodynamic freestream (relative wind seen by the body).
    # This sets D_hat = U_ref/|U_ref| in _decompose_wind_axes, so the drag axis
    # must point in the direction of the incoming flow, not the body velocity.
    #
    # Body frame (plate moves in -X at U, quiescent air):
    #   relative wind = +X  →  U_ref = [+U, 0, 0]
    # Wind frame (plate fixed, flow at AoA):
    #   freestream direction = [cos α, 0, sin α]  →  encode angle here
    # Pitching / ramp (flow is always +X):
    #   U_ref = [+U, 0, 0]
    if args.kinematics in ("ramp", "static") and args.frame == "body":
        U_ref = np.array([args.U_inf, 0.0, 0.0])
    elif args.kinematics == "static" and args.frame == "wind":
        U_ref = np.array(
            [
                args.U_inf * math.cos(alpha_rad),
                0.0,
                args.U_inf * math.sin(alpha_rad),
            ]
        )
    else:
        U_ref = np.array([args.U_inf, 0.0, 0.0])

    vlm = VLMSolver(
        max_panels=max(1024, args.panels_chord * args.panels_span * 4),
        density=args.density,
        viscosity=args.viscosity,
        linear_solver="SCIPY",
        U_inf=U_ref,
        force=force_cfg,
        sigma_factor=args.sigma_factor,
        sample_surface_forces=True,
    )
    vlm.add_surface(surface_file, kinematics=kin)

    # ── Samplers ──────────────────────────────────────────────────────
    backup_dir = f"solution/{args.name}"
    samplers = []

    if args.sample_midspan:
        y_mid = args.span / 2.0
        samplers.append(
            SurfaceSampler(
                point=[0.0, y_mid, 0.0],
                normal=[0, 1, 0],
                bounds=[-7.0, 3.0, -2.0, 2.0],
                spacing=args.sample_spacing,
                file_name=f"{args.name}_midspan",
                output_dir=backup_dir + "/samples",
            )
        )

    if args.sample_crossflow:
        # Streamwise-normal (y-z) crossflow planes at x = 5, 15, 25 to capture the
        # trailing-vortex roll-up.  Grid spacing ~ shed-particle spacing (U*dt).
        cf_spacing = max(args.U_inf * (args.dt if args.dt else 0.01), 0.05)
        for x_loc in (5.0, 15.0, 25.0):
            samplers.append(
                SurfaceSampler(
                    point=[x_loc, 0.0, 0.0],
                    normal=[1, 0, 0],
                    bounds=[-6.0, 6.0, -0.5, 5.0],
                    spacing=cf_spacing,
                    file_name=f"{args.name}_crossflow_x{int(x_loc)}",
                    output_dir=backup_dir + "/samples",
                )
            )

    # ── Solver config ─────────────────────────────────────────────────
    cfg_kwargs = dict(
        time_step_size=dt,
        vlm_solver=vlm,
        background_velocity=bg_vel,
        logging_frequency=2,
        backup_frequency=2,
        backup_file_name=args.name,
        backup_directory=backup_dir,
        samplers=samplers if samplers else None,
    )

    solver_config = SolverConfig.les_simulation(cs=args.cs, **cfg_kwargs)

    # ── Remove stale CSVs before run ──────────────────────────────────
    # The diagnostics/loading-distribution hooks APPEND to these every log step,
    # so a leftover file from a previous run of the same --name would accumulate
    # (and a single-case re-run without allclean would mix old+new rows at the
    # same step number). Clear them here so each run starts fresh.
    samples_dir = Path(backup_dir) / "samples"
    src_csv = samples_dir / "vlm_forces.csv"
    for _stale in ["vlm_forces.csv", "vlm_spanwise_flat_plate.csv", "vlm_chordwise_flat_plate.csv"]:
        _p = samples_dir / _stale
        if _p.exists():
            _p.unlink()

    # ── Solver instantiation ──────────────────────────────────────────
    solver = Solver(config=solver_config)

    # Generate mesh AFTER Solver() (ti.init required first)
    vlm.generate_mesh(
        spanwise_spacing="geometric",
        spanwise_spacing_ratio=4.0,
        spanwise_spacing_region="end",
    )

    # ── DIAGNOSTIC: standalone uncoupled steady VLM (D1) ──────────────
    # One solve with the FULL semi-infinite horseshoe (coupled=False) = the exact
    # linear finite-wing answer. If cl(y) tapers to ~0 at the tip, the geometry/AIC
    # is fine and the coupled VPM wake is what under-delivers the tip downwash.
    if args.steady_vlm:
        n_p = vlm.lattice.num_panels
        V_ext = np.tile(U_ref.astype(float), (n_p, 1))
        vlm.solve(V_external=V_ext, dt=None, coupled=False)
        vlm.compute_postprocess(V_ext, U_ref, args.density, dt=None, coupled=False)
        dists = VLMLoadingDistribution.extract_distributions(vlm, "flat_plate", U_ref, args.density)
        sp = dists["spanwise"]
        out = Path(backup_dir) / "samples"
        out.mkdir(parents=True, exist_ok=True)
        sp.to_csv(out / f"{args.name}_spanwise.csv", index=False)
        o = sp[sp.half == "orig"].sort_values("span_index")
        print("\n  ─── Uncoupled steady VLM (diagnostic) ───")
        print(
            f"  cl_root={o.cl.iloc[0]:.4f}  cl_tip={o.cl.iloc[-1]:.4f}  "
            f"Gamma_root={o.Gamma.iloc[0]:.4f}  Gamma_tip={o.Gamma.iloc[-1]:.4f}"
        )
        print(f"  cl(y) orig half: {np.round(o.cl.to_numpy(), 4).tolist()}")
        print(f"  Output: {out / f'{args.name}_spanwise.csv'}\n")
        return

    # Pre-compute chords traveled for each step
    step_indices = np.arange(n_steps, dtype=float)
    chords_traveled = chord_travel_fn(step_indices)

    # ── Simulation loop ───────────────────────────────────────────────
    for _step in range(n_steps):
        solver.update_state()

    # ── Post-process ──────────────────────────────────────────────────
    if src_csv.exists():
        df = pd.read_csv(src_csv)
        df["chords"] = chords_traveled[: len(df)]
        dst_csv = src_csv.parent / f"{args.name}.csv"
        df.to_csv(dst_csv, index=False)
        src_csv.unlink()

        if "CL" in df.columns and len(df) > 0:
            cl_final = df["CL"].iloc[-1]
            cd_final = df["CD"].iloc[-1] if "CD" in df.columns else float("nan")
            n_part = df["n_particles"].iloc[-1] if "n_particles" in df.columns else "?"
            print(f"\n  ─── Summary ───────────────────────────────────────────")
            print(f"  CL (final)  = {cl_final:8.5f}")
            print(f"  CD (final)  = {cd_final:8.5f}")
            print(f"  Particles   = {n_part}")
            print(f"  Output      = {dst_csv}")
            print(f"  ───────────────────────────────────────────────────────\n")

            if "chords" in df and df["chords"].max() >= 5.0:
                tail = df[df["chords"] >= df["chords"].max() - 5.0]
                cl_scale = max(abs(float(tail["CL"].mean())), 1e-12)
                rel_range = float(tail["CL"].max() - tail["CL"].min()) / cl_scale
                print(f"  CL tail range (last 5c) = {100.0 * rel_range:.4f}%")
                if rel_range > 2e-3:
                    print("  [WARNING] CL has not met the 0.2% steady-state criterion.")
    else:
        print("  [WARNING] vlm_forces.csv not found; no CSV output produced.")

    # Copy final-step rows of spanwise distribution CSV for easy validation access
    span_src = Path(backup_dir) / "samples" / "vlm_spanwise_flat_plate.csv"
    if span_src.exists():
        df_sp = pd.read_csv(span_src)
        if not df_sp.empty:
            final_step = df_sp["step"].max()
            df_sp_final = df_sp[df_sp["step"] == final_step].copy()
            span_dst = span_src.parent / f"{args.name}_spanwise.csv"
            df_sp_final.to_csv(span_dst, index=False)

    # ── D4: Per-station VPM→VLM induced downwash diagnostic ──────────────────
    # external_velocity holds the last step's VPM-induced velocity (body frame:
    # include_freestream=False so this is PURE VPM contribution). Grouped by span
    # station, w_j = V_ext·n_hat gives the normal downwash the VPM wake feeds into
    # the no-penetration BC. Compare to Glauert-required induced AoA to localize
    # the tip downwash deficit (Bug 2e).
    try:
        from theoretical_model import liftingline_circulation

        n_p = vlm.lattice.num_panels
        V_ext_all = vlm.lattice.external_velocity.to_numpy()[:n_p]  # (N, 3)
        normals_all = vlm.lattice.normals.to_numpy()[:n_p]  # (N, 3)
        bm_all = vlm.lattice.bound_midpoints.to_numpy()[:n_p]  # (N, 3)

        blocks = VLMLoadingDistribution.build_surface_grid_index(vlm, "flat_plate")

        rows = []
        for blk in blocks:
            ns, nc = blk["ns"], blk["nc"]
            for half, idx2d in [("orig", blk["orig_idx"]), ("mirror", blk["mirror_idx"])]:
                if idx2d is None:
                    continue
                for j in range(ns):
                    cidx = idx2d[j]
                    w_panels = np.einsum("ki,ki->k", V_ext_all[cidx], normals_all[cidx])
                    w_j = float(np.mean(w_panels))
                    y_j = float(np.mean(bm_all[cidx, 1]))
                    rows.append({"half": half, "span_index": j, "y": y_j, "w_VPM": w_j})

        df_dw = pd.DataFrame(rows)
        orig = df_dw[df_dw.half == "orig"].sort_values("span_index")
        y_stations = orig["y"].to_numpy()
        w_VPM = orig["w_VPM"].to_numpy()

        # Glauert-required induced AoA: α_i = α − cl_glauert/(2π)
        df_ll = liftingline_circulation(
            y_stations,
            b=args.span,
            c=args.chord,
            alpha_rad=math.radians(args.aoa),
            U_inf=args.U_inf,
        )
        cl_gl = df_ll["cl"].to_numpy()
        alpha_rad_val = math.radians(args.aoa)
        alpha_i_req = np.degrees(alpha_rad_val - cl_gl / (2.0 * math.pi))  # induced AoA [deg]
        # VPM delivered induced AoA: α_VPM = -w_j/U_inf (w_j<0 means downwash)
        alpha_i_VPM = np.degrees(np.arctan2(-w_VPM, args.U_inf))
        ratio = np.where(np.abs(alpha_i_req) > 0.01, alpha_i_VPM / alpha_i_req, float("nan"))

        print(f"\n  ─── D4: VPM→VLM induced downwash per span station ───")
        print(f"  {'j':>3}  {'y/b':>6}  {'w_VPM':>8}  {'α_VPM':>7}  {'α_i_req':>8}  {'ratio':>6}")
        for k in range(len(y_stations)):
            yob = 2.0 * y_stations[k] / args.span
            print(
                f"  {k:>3}  {yob:>6.3f}  {w_VPM[k]:>8.4f}  "
                f"{alpha_i_VPM[k]:>7.3f}°  {alpha_i_req[k]:>8.3f}°  {ratio[k]:>6.2f}"
            )

        df_out = orig.copy()
        df_out["alpha_i_VPM_deg"] = alpha_i_VPM
        df_out["alpha_i_required_deg"] = alpha_i_req
        df_out["delivery_ratio"] = ratio
        dw_csv = Path(backup_dir) / "samples" / f"{args.name}_downwash.csv"
        df_out.to_csv(dw_csv, index=False)
        print(f"  Saved: {dw_csv}")
    except Exception as _exc:
        print(f"  [WARNING] D4 downwash dump failed: {_exc}")


if __name__ == "__main__":
    main()
