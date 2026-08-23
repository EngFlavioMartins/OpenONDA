#!/usr/bin/env python3
"""
Verify the rotor sheds vorticity from the TRAILING edge (not the LE).
=====================================================================
The blade is built with its leading edge on the −Z side (chord points +Z from
LE→TE).  For the blade to advance LEADING-EDGE first — so the bound circulation
develops with a positive angle of attack and vorticity is shed from the
TRAILING edge — the rotor must spin so the +Y blade moves toward −Z, i.e.
``ManeuverVLM(axis=[-1, 0, 0])``.

This check uses the OpenVSP blade design schedule from ``generate_openvsp_blade``
(the same geometry used in ``rotor_setup.py``) and, for each radial station,
forms the local relative wind ``U∞ − ω×r`` and tests its projection on the
chord direction (from LE→TE, i.e. toward +Z at zero-twist):

    relwind · chord  > 0   → LE faces the wind  (correct: TE shedding)
    relwind · chord  < 0   → TE faces the wind  (reversed: LE shedding)

It also prints the local angle of attack.  Exits non-zero if the configured
rotation does not shed from the TE, so it can gate the tutorial.

Usage::

    python assets/verify_shedding.py            # checks the corrected config (axis_x=-1)
    python assets/verify_shedding.py --axis 1   # show the (wrong) +x case
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from generate_openvsp_blade import RotorBladeDesign, design_schedule


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify rotor TE shedding.")
    ap.add_argument(
        "--axis",
        type=float,
        default=-1.0,
        help="x-component of the rotation axis (rotor_setup uses -1).",
    )
    args = ap.parse_args()

    design = RotorBladeDesign()
    sched = design_schedule(design)
    r_stations = sched["radial_position"]
    theta_deg = sched["twist_angle_degrees"]
    chord = sched["chord"]

    omega_vec = np.array([args.axis, 0.0, 0.0]) * design.angular_velocity
    freestream = np.array([design.freestream_speed, 0.0, 0.0])

    print(
        f"Rotor TE-shedding check  (axis=[{args.axis:+.0f},0,0], "
        f"angular_velocity={design.angular_velocity:.3f} rad/s, freestream_speed={design.freestream_speed} m/s)"
    )
    print(
        f"{'r [m]':>7} {'c [m]':>6} {'theta [deg]':>12} {'AoA [deg]':>10} "
        f"{'relwind·chord':>14}  verdict"
    )

    all_ok = True
    for r, theta, c in zip(r_stations, theta_deg, chord):
        theta_rad = np.radians(theta)
        # Chord direction for blade at azimuth=0 (pointing in +Y):
        # At twist θ = arctan(axial_velocity / tangential_velocity) - α, the chord lies in the XZ plane.
        # The LE→TE unit vector (after twist about Y-axis from the +Z direction):
        # chord_dir = [sin(theta), 0, cos(theta)]  (in XZ plane)
        chord_dir = np.array([np.sin(theta_rad), 0.0, np.cos(theta_rad)])

        # A blade section at radius r: position mid-chord ~ [0, r, 0] at azimuth=0
        mid = np.array([0.0, r, 0.0])
        relwind = freestream - np.cross(omega_vec, mid)
        rw_norm = relwind / (np.linalg.norm(relwind) + 1e-20)
        dot = float(np.dot(rw_norm, chord_dir))

        # Angle of attack = angle between relwind and chord_dir
        aoa = np.degrees(np.arccos(np.clip(abs(dot), 0.0, 1.0)))
        # sign: positive AoA when LE faces wind (dot > 0)
        if dot < 0.0:
            aoa = -aoa

        ok = dot > 0.0
        all_ok &= ok
        verdict = "LE faces wind (TE shedding) OK" if ok else "TE faces wind (LE shedding) REVERSED"
        print(f"{r:7.2f} {c:6.3f} {theta:12.2f} {aoa:10.2f} {dot:14.3f}  {verdict}")

    print()
    if all_ok:
        print(
            "PASS: leading edge faces the relative wind at every station — "
            "vorticity is shed from the trailing edge."
        )
        return 0
    print(
        "FAIL: the blade advances trailing-edge first — vorticity is shed from "
        "the LEADING edge. Flip the rotation axis x-component to -1."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
