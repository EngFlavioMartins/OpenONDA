#!/usr/bin/env python3
"""Acceptance checks for the Saffman vortex-ring validation."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import T_REF, load_ring_data, load_ring_speed, saffman_speed, U_REF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution-dir", default="solution")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--expected-files", type=int, default=100)
    args = ap.parse_args()
    root, figs = Path(args.solution_dir), Path(args.figures_dir)
    failures: list[str] = []

    for name, speed_tol in (("DNS_transposed", 0.10), ("LES_rvpm", 0.12)):
        files = sorted(glob.glob(str(root / name / f"vpm_{name}_*.h5")))
        if len(files) != args.expected_files:
            failures.append(f"{name}: {len(files)} backups, expected {args.expected_files}")
            continue
        raw = load_ring_data(files)
        entries = raw.get(0, [])
        if len(entries) != len(files):
            failures.append(f"{name}: unreadable or unbounded snapshots")
            continue
        r = np.array([x["major_R"] for x in entries])
        impulse = np.array([x["impulse_x"] for x in entries])
        gamma = np.array([x["gamma"] for x in entries])
        r_drift = float(np.max(np.abs(r / r[0] - 1.0)))
        i_drift = float(np.max(np.abs(impulse / impulse[0] - 1.0)))
        g_drift = float(np.max(np.abs(gamma / gamma[0] - 1.0)))
        t, u = load_ring_speed(files)
        ref = saffman_speed(t * T_REF) / U_REF
        rel_rmse = float(np.sqrt(np.mean((u - ref) ** 2)) / np.mean(ref))
        print(f"{name}: dR={r_drift:.3%}, dI={i_drift:.3%}, dGamma={g_drift:.3%}, speed RMSE={rel_rmse:.3%}")
        if i_drift > 0.05 or g_drift > 0.08 or rel_rmse > speed_tol:
            failures.append(f"{name}: conservation/speed tolerance exceeded")

    for name in ("vortex_ring_motion.png", "vortex_ring_energy.png", "vortex_ring_circulation.png"):
        if not (figs / name).exists():
            failures.append(f"missing figure {name}")
    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        return 1
    print("[OK] vortexRing certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
