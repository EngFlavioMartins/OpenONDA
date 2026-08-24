#!/usr/bin/env python3
"""Acceptance checks for the Saffman vortex-ring validation."""

from __future__ import annotations

import glob
import re
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    SOLUTION_DIR,
    REFERENCE_TIME,
    REFERENCE_VELOCITY,
    load_ring_data,
    load_ring_speed,
    saffman_speed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    expected_steps = set(range(25, 601, 25))
    failures: list[str] = []

    speed_tolerances = {
        "dns_direct": 0.15,
        "dns_transposed": 0.10,
        "dns_mixed": 0.15,
        "les_transposed": 0.12,
    }
    for name, speed_tol in speed_tolerances.items():
        all_files = sorted(glob.glob(str(SOLUTION_DIR / f"vpm_{name}_*.h5")))
        numbered = {
            int(match.group(1)): path
            for path in all_files
            if (match := re.search(rf"vpm_{name}_(\d{{6}})\.h5$", path))
        }
        if set(numbered) != expected_steps:
            failures.append(
                f"{name}: numbered checkpoints {sorted(numbered)}; expected {sorted(expected_steps)}"
            )
            continue
        files = [numbered[step] for step in sorted(numbered)]
        raw = load_ring_data(files)
        entries = raw.get(0, [])
        if len(entries) != len(files):
            failures.append(f"{name}: unreadable or unbounded snapshots")
            continue
        r = np.array([x["major_radius"] for x in entries])
        impulse = np.array([x["linear_impulse_x"] for x in entries])
        tube_circulation = np.array([x["tube_circulation"] for x in entries])
        r_drift = float(np.max(np.abs(r / r[0] - 1.0)))
        i_drift = float(np.max(np.abs(impulse / impulse[0] - 1.0)))
        g_drift = float(np.max(np.abs(tube_circulation / tube_circulation[0] - 1.0)))
        if r_drift > 0.05:
            failures.append(f"{name}: major-radius drift {r_drift:.3%} > 5%")
        t, u = load_ring_speed(files)
        ref = saffman_speed(t * REFERENCE_TIME) / REFERENCE_VELOCITY
        rel_rmse = float(np.sqrt(np.mean((u - ref) ** 2)) / np.mean(ref))
        print(
            f"{name}: dR={r_drift:.3%}, dI={i_drift:.3%}, dcirculation={g_drift:.3%}, speed RMSE={rel_rmse:.3%}"
        )
        if i_drift > 0.05 or g_drift > 0.08 or rel_rmse > speed_tol:
            failures.append(f"{name}: conservation/speed tolerance exceeded")

        samples = SAMPLES_DIR / name
        for csv_name in ("flow_integrals.csv", "ring_diagnostics.csv", "ring_modes.csv"):
            csv_path = samples / csv_name
            if not csv_path.is_file():
                failures.append(f"{name}: missing {csv_name}")
                continue
            try:
                data = pd.read_csv(csv_path)
            except (OSError, ValueError, pd.errors.ParserError) as error:
                failures.append(f"{name}: unreadable {csv_name} ({error})")
                continue
            if data.empty or not np.isfinite(data.select_dtypes(include=[np.number])).all().all():
                failures.append(f"{name}: empty or non-finite {csv_name}")
        final = SOLUTION_DIR / f"vpm_{name}_final.h5"
        if not final.is_file():
            failures.append(f"{name}: missing final restart state")

    if not args.pre_plot:
        for extension in ("png", "pdf"):
            for name in (
                "vortex_ring_motion",
                "vortex_ring_energy",
                "vortex_ring_circulation",
            ):
                figure = FIGURES_DIR / f"{name}.{extension}"
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing or empty figure {figure.name}")
    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        return 1
    print("[OK] vortex_ring certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
