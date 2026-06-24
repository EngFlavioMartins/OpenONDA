#!/usr/bin/env python3
"""Numerical and aerodynamic acceptance checks for rotorFlow."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import h5py
import numpy as np
import pandas as pd


def _step(path: Path) -> int:
    m = re.search(r"_(\d+)\.h5$", path.name)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution-dir", default="solution/rotor")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--expected-step", type=int, default=3500)
    args = ap.parse_args()
    root, figs = Path(args.solution_dir), Path(args.figures_dir)
    failures: list[str] = []

    files = sorted(root.glob("vpm_rotor_*.h5"), key=_step)
    if not files or _step(files[-1]) != args.expected_step:
        failures.append(f"last rotor backup is {_step(files[-1]) if files else 'missing'}, expected {args.expected_step}")
    elif files:
        with h5py.File(files[-1], "r") as h5:
            alpha = h5["particles/circulation"][:]
            radius = h5["particles/radius"][:]
        max_strength = float(np.linalg.norm(alpha, axis=1).max())
        print(f"Final particles={len(alpha)}, max|alpha|={max_strength:.4g}, max radius={radius.max():.4g}")
        if not np.isfinite(alpha).all() or max_strength > 10.0:
            failures.append(f"unbounded final wake strength: {max_strength:.4g}")

    csv = root / "samples" / "vlm_forces.csv"
    if not csv.exists():
        failures.append("missing vlm_forces.csv")
    else:
        df = pd.read_csv(csv)
        qA = 0.5 * 1.225 * 7.0**2 * np.pi * 6.0**2
        omega = 7.0 * 7.0 / 6.0
        ct = df.Fx.to_numpy() / qA
        cp = -df.Mx.to_numpy() * omega / (qA * 7.0)
        tail = slice(max(0, int(0.8 * len(df))), None)
        ct_mean, cp_mean = float(np.mean(ct[tail])), float(np.mean(cp[tail]))
        print(f"Tail mean Ct={ct_mean:.4f}, Cp={cp_mean:.4f}")
        if not np.isfinite(ct).all() or not np.isfinite(cp).all():
            failures.append("non-finite rotor coefficients")
        if not (0.2 < ct_mean < 1.2 and 0.1 < cp_mean < 0.65):
            failures.append(f"implausible tail coefficients Ct={ct_mean:.3f}, Cp={cp_mean:.3f}")

    for tag in ("x36m", "x72m", "x108m"):
        planes = sorted((root / "samples").glob(f"slice_{tag}_*.vts"))
        if not planes or f"_{args.expected_step:06d}.vts" not in planes[-1].name:
            failures.append(f"missing final {tag} wake plane")
    for name in ("rotor_performance.png", "rotor_wake_planes.png"):
        if not (figs / name).exists():
            failures.append(f"missing figure {name}")

    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        return 1
    print("[OK] rotorFlow certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
