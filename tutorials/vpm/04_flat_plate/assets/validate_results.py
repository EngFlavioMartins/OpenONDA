#!/usr/bin/env python3
"""Certification checks for the flat-plate suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"
from .results import load_forces

AOA_TAGS = [
    "aoan10",
    "aoan05",
    "aoan02",
    "aoa00",
    "aoa02",
    "aoa05",
    "aoa08",
    "aoa10",
    "aoa12",
    "aoa15",
]
COEFFICIENTS = (
    "lift_coefficient",
    "drag_coefficient",
    "pitching_moment_coefficient_quarter_chord",
)


def check_polar(polar: dict[tuple[str, str], np.ndarray]) -> list[str]:
    """Check frame equivalence and reflection symmetry of the settled loads."""
    failures = []

    def compare(label, actual, expected):
        # Use the same 0.2% tolerance as the accepted late lift variation, with
        # an absolute floor for the exactly unloaded zero-incidence plate.
        tolerance = 2e-3 * np.maximum(np.abs(actual), np.abs(expected)) + 1e-8
        for field, error, limit in zip(COEFFICIENTS, np.abs(actual - expected), tolerance):
            if error > limit:
                failures.append(f"{label}: {field} difference {error:.3e} > {limit:.3e}")

    for tag in AOA_TAGS:
        if ("moving", tag) in polar and ("static", tag) in polar:
            compare(f"{tag} moving/static", polar["moving", tag], polar["static", tag])
    for frame in ("moving", "static"):
        for angle in ("02", "05", "10"):
            negative, positive = (frame, f"aoan{angle}"), (frame, f"aoa{angle}")
            if negative in polar and positive in polar:
                compare(
                    f"{frame} +/-{angle} reflection",
                    polar[negative],
                    polar[positive] * [-1, 1, -1],
                )
        zero = (frame, "aoa00")
        if zero in polar:
            compare(f"{frame} zero incidence", polar[zero], np.zeros(3))
    return failures


def vector_strength_closure(flow: pd.DataFrame) -> float:
    """Normalize the native coupled vector budget by peak bound strength."""
    coupled_fields = [f"coupled_vortex_strength_{axis}" for axis in "xyz"]
    bound_fields = [f"bound_vortex_strength_{axis}" for axis in "xyz"]
    if not set(coupled_fields + bound_fields).issubset(flow.columns):
        return float("inf")
    coupled = flow[coupled_fields].to_numpy()
    bound = flow[bound_fields].to_numpy()
    if flow.empty or not np.isfinite(coupled).all() or not np.isfinite(bound).all():
        return float("inf")
    return float(
        np.linalg.norm(coupled, axis=1).max() / max(np.linalg.norm(bound, axis=1).max(), 1e-15)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    case_dir = Path(__file__).resolve().parents[1]
    root = case_dir / "samples"
    figs = case_dir / "figures"
    failures: list[str] = []
    polar = {}

    for frame in ("moving", "static"):
        for tag in AOA_TAGS:
            name = f"exp_{frame}_{tag}"
            path = root / name / "vlm_forces.csv"
            if not path.exists():
                failures.append(f"missing {path}")
                continue
            df = load_forces(case_dir, name)
            load_fields = [
                f"{quantity}_{axis}" for quantity in ("force", "moment") for axis in "xyz"
            ]
            if (
                df is None
                or df.empty
                or not np.isfinite(df[[*load_fields, *COEFFICIENTS]].to_numpy()).all()
            ):
                failures.append(f"{name}: missing, incomplete or non-finite force history")
                continue
            metadata = json.loads((case_dir / "solution" / name / "vpm_metadata.json").read_text())
            state = metadata["state"]
            vlm = metadata["configuration"]["numerics"]["vlm"]
            final_step = state["initial_step"] + state["requested_steps"]
            if state["step"] != final_step:
                failures.append(f"{name}: solver stopped before the requested final step")
            expected_steps = np.arange(state["initial_step"] + 1, final_step + 1)
            expected_steps = expected_steps[expected_steps % vlm["logging_interval_steps"] == 0]
            if not np.array_equal(df["step"].to_numpy(), expected_steps):
                failures.append(f"{name}: missing, repeated or unordered force samples")
            if not vlm["force"].get("unsteady", False):
                failures.append(f"{name}: missing the authored unsteady pressure force model")
                continue
            if metadata["configuration"]["numerics"]["induction"]["method"] != "DIRECT":
                failures.append(
                    f"{name}: preceding result does not use the direct reference backend"
                )
                continue
            if (
                "nondimensional_distance_travelled" not in df
                or df["nondimensional_distance_travelled"].max() < 23.5
            ):
                failures.append(f"{name}: did not reach 24 chord lengths")
                continue
            tail = df[
                df["nondimensional_distance_travelled"]
                >= df["nondimensional_distance_travelled"].max() - 5.0
            ]
            polar[frame, tag] = tail[list(COEFFICIENTS)].mean().to_numpy()
            scale = max(abs(float(tail["lift_coefficient"].mean())), 1e-12)
            rel = float(tail["lift_coefficient"].max() - tail["lift_coefficient"].min()) / scale
            if rel > 2e-3:
                failures.append(f"{name}: lift_coefficient tail range {100 * rel:.3f}% > 0.2%")
            strength = df[["bound_vortex_strength_y", "wake_vortex_strength_y"]].to_numpy()
            if not np.isfinite(strength).all():
                failures.append(f"{name}: non-finite bound/wake strength history")
                continue
            scale = max(float(np.max(np.abs(strength[:, 0]))), 1e-15)
            closure = float(np.max(np.abs(strength.sum(axis=1))) / scale)
            print(f"{name}: bound/wake strength closure {closure:.3e}")
            if closure > 1e-4:
                failures.append(f"{name}: bound/wake strength closure {closure:.3e} > 1e-4")
            flow_path = root / name / "flow_integrals.csv"
            if not flow_path.is_file():
                failures.append(f"missing {flow_path}")
                continue
            vector_closure = vector_strength_closure(pd.read_csv(flow_path))
            print(f"{name}: full-vector strength closure {vector_closure:.3e}")
            if vector_closure > 1e-4:
                failures.append(f"{name}: full-vector strength closure {vector_closure:.3e} > 1e-4")

    failures.extend(check_polar(polar))

    if not args.pre_plot:
        for extension in ("png", "pdf"):
            for name in (
                "plate_polar",
                "plate_staticvsmoving",
                "plate_startup",
                "plate_spanwise",
                "flat_plate_kelvin",
                "plate_velocity",
                "plate_impulse",
            ):
                figure = figs / f"{name}.{extension}"
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing or empty figure {figure.name}")

    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        return 1
    print("[OK] flat_plate certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
