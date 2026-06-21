#!/usr/bin/env python3
"""
Verify the rotor sheds vorticity from the TRAILING edge (not the LE).
=====================================================================
The blade is built with its leading edge on the −Z side (chord points +Z from
LE→TE).  For the blade to advance LEADING-EDGE first — so the bound circulation
develops with a positive angle of attack and vorticity is shed from the
TRAILING edge — the rotor must spin so the +Y blade moves toward −Z, i.e.
``RotatingVLM(axis=[-1, 0, 0])``.

This check rebuilds the exact blade geometry used by ``rotor_setup.py`` and, for
each radial station, forms the local relative wind ``U∞ − ω×r`` and tests its
projection on the chord (LE→TE):

    relwind · chord  > 0   → LE faces the wind  (correct: TE shedding)
    relwind · chord  < 0   → TE faces the wind  (reversed: LE shedding)

It also prints the local angle of attack.  Exits non-zero if the configured
rotation does not shed from the TE, so it can gate the tutorial.

Usage:
    python assets/verify_shedding.py            # checks the corrected config
    python assets/verify_shedding.py --axis 1   # show the (wrong) +x case
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from generate_blade import create_blade, save_surface  # noqa: E402


# Physical parameters — must match rotor_setup.py
U_INF = 7.0
TSR = 7.0
ROTOR_RADIUS = 6.0
HUB_RADIUS = 1.0
OMEGA = TSR * U_INF / ROTOR_RADIUS
ALPHA_DES = np.radians(5.0)
A_BETZ = 1.0 / 3.0


def build_blade():
    """Reproduce rotor_setup.py's Betz-twisted blade and return its segments."""
    n_sched = 11
    r = np.linspace(HUB_RADIUS, ROTOR_RADIUS, n_sched)
    phi = np.arctan2(U_INF * (1.0 - A_BETZ), OMEGA * r)
    beta = 90.0 - np.degrees(phi - ALPHA_DES)
    aircraft = create_blade(
        radius=ROTOR_RADIUS,
        hub_radius=HUB_RADIUS,
        root_chord=0.6,
        tip_chord=0.35,
        n_span=20,
        n_chord=6,
        r_schedule=r,
        beta_schedule=beta,
        pitch_schedule=np.zeros(n_sched),
    )
    # Serialise to the canonical JSON (list-structured) and read it back, so we
    # don't depend on the in-memory Aircraft container layout.
    tmp = Path(tempfile.gettempdir()) / "_verify_blade.json"
    save_surface(aircraft, str(tmp))
    data = json.loads(tmp.read_text())
    return [seg["vertices"] for w in data["wings"] for seg in w["segments"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify rotor TE shedding.")
    ap.add_argument("--axis", type=float, default=-1.0,
                    help="x-component of the rotation axis (rotor_setup uses -1).")
    args = ap.parse_args()

    wvec = np.array([args.axis, 0.0, 0.0]) * OMEGA
    freestream = np.array([U_INF, 0.0, 0.0])
    segs = build_blade()

    print(f"Rotor TE-shedding check  (axis=[{args.axis:+.0f},0,0], "
          f"omega={OMEGA:.3f} rad/s, U_inf={U_INF} m/s)")
    print(f"{'r [m]':>6} {'AoA [deg]':>10} {'relwind.chord':>14}  verdict")
    all_ok = True
    for v in segs:
        LE = np.asarray(v["b"], float)   # tip leading edge
        TE = np.asarray(v["c"], float)   # tip trailing edge
        r = 0.5 * (LE[1] + TE[1])
        chord = TE - LE
        chord /= np.linalg.norm(chord)
        mid = 0.5 * (LE + TE)
        relwind = freestream - np.cross(wvec, mid)
        rw = relwind / np.linalg.norm(relwind)
        dot = float(np.dot(rw, chord))
        aoa = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        ok = dot > 0.0
        all_ok &= ok
        print(f"{r:6.1f} {aoa:10.1f} {dot:14.3f}  "
              f"{'LE faces wind (TE shedding) OK' if ok else 'TE faces wind (LE shedding) REVERSED'}")

    print()
    if all_ok:
        print("PASS: leading edge faces the relative wind at every station — "
              "vorticity is shed from the trailing edge.")
        return 0
    print("FAIL: the blade advances trailing-edge first — vorticity is shed from "
          "the LEADING edge. Flip the rotation: RotatingVLM(axis=[-1,0,0]).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
