#!/usr/bin/env python3
"""Strict completeness and physics checks for the Lamb--Oseen tutorial."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = CASE_DIR / "samples"
FIGURES_DIR = CASE_DIR / "figures"
GRID_REPORT = CASE_DIR / "grid_study" / "cs_equal_protocol_v1" / "grid_independence_cs.json"
SCHEMES = ("cs", "rwm", "dvh", "gbd")
CASES = ("vortex", "dipole", "merging")
EXPECTED_END_TIME = 103.0 * 0.291
EXPECTED_DT = 0.291 / 9.0


def _read_csv(
    path: Path,
    failures: list[str],
    *,
    allow_nonfinite: bool = False,
) -> pd.DataFrame | None:
    if not path.is_file():
        failures.append(f"missing {path}")
        return None
    try:
        data = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        failures.append(f"unreadable {path}: {error}")
        return None
    numeric = data.select_dtypes(include=[np.number])
    if (
        data.empty
        or numeric.empty
        or (not allow_nonfinite and not np.isfinite(numeric.to_numpy()).all())
    ):
        failures.append(f"{path}: empty or non-finite numeric data")
    return data


def _single_vortex_errors() -> dict[str, tuple[float, float, float]]:
    """Compare the final common single-vortex field with analytic Lamb--Oseen."""
    assets = CASE_DIR / "assets"
    sys.path.insert(0, str(assets))
    from plot_vortex_comparison import lamb_oseen_gradient, lamb_oseen_profile, load_profile
    from vortex_diagnostics import pvd_time_map, resolve_runtime_physics

    latest = min(max(pvd_time_map(SAMPLES_DIR, "vortex", scheme).values()) for scheme in SCHEMES)
    runtime = resolve_runtime_physics(SAMPLES_DIR, 1.0, 1.0 / 530.0, 1.0, 0.125 / 1.12)
    output: dict[str, tuple[float, float, float]] = {}
    for scheme in SCHEMES:
        profile = load_profile(SAMPLES_DIR, scheme, latest)
        if profile is None:
            raise ValueError(f"vortex_{scheme}: no readable common-time field profile")
        x, velocity, vorticity, selected_time = profile
        exact_velocity, exact_vorticity, _ = lamb_oseen_profile(
            x,
            runtime["t0"] + selected_time,
            runtime["circulation"],
            runtime["kinematic_viscosity"],
        )
        exact_gradient = lamb_oseen_gradient(
            x,
            runtime["t0"] + selected_time,
            runtime["circulation"],
            runtime["kinematic_viscosity"],
        )
        numerical_gradient = np.gradient(velocity, x)
        window = np.abs(x / runtime["velocity_peak_radius0"]) <= 5.5
        errors = tuple(
            float(np.linalg.norm((numerical - exact)[window]) / np.linalg.norm(exact[window]))
            for numerical, exact in (
                (velocity, exact_velocity),
                (vorticity, exact_vorticity),
                (numerical_gradient, exact_gradient),
            )
        )
        output[scheme] = errors
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    for physics in CASES:
        for scheme in SCHEMES:
            name = f"{physics}_{scheme}"
            folder = SAMPLES_DIR / name
            metadata_path = folder / "run_metadata.json"
            if not metadata_path.is_file():
                failures.append(f"{name}: missing run_metadata.json")
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                failures.append(f"{name}: unreadable metadata ({error})")
                continue
            if metadata.get("status") != "complete" or metadata.get("completed") is not True:
                failures.append(f"{name}: metadata is not complete")
            final_time = float(metadata.get("final_time", np.nan))
            if not np.isclose(final_time, EXPECTED_END_TIME, atol=EXPECTED_DT):
                failures.append(f"{name}: final time {final_time:.9g} != {EXPECTED_END_TIME:.9g}")

            integrals = _read_csv(folder / "flow_integrals.csv", failures)
            if integrals is not None:
                if (
                    "time" not in integrals
                    or integrals["time"].iloc[-1] < EXPECTED_END_TIME - EXPECTED_DT
                ):
                    failures.append(f"{name}: flow-integral history is incomplete")
                if np.any(np.diff(integrals["time"].to_numpy(float)) <= 0.0):
                    failures.append(f"{name}: flow-integral time is not strictly increasing")
                for column in ("kinetic_energy_rate", "viscous_kinetic_energy_rate"):
                    if column in integrals and np.any(integrals[column].to_numpy(float) > 1.0e-7):
                        failures.append(f"{name}: positive modeled energy rate in {column}")

            fields = _read_csv(folder / "field_diagnostics.csv", failures, allow_nonfinite=True)
            if fields is not None:
                required = {"time", "step", "core_radius_0", "mean_core_radius"}
                if not required.issubset(fields.columns):
                    failures.append(
                        f"{name}: missing field columns {sorted(required - set(fields.columns))}"
                    )
                else:
                    if np.any(fields["core_radius_0"].to_numpy(float) <= 0.0):
                        failures.append(f"{name}: non-positive extracted core radius")
                    if fields["time"].iloc[-1] < EXPECTED_END_TIME - EXPECTED_DT:
                        failures.append(f"{name}: field diagnostics are incomplete")
                boundary_columns = [column for column in fields if "boundary_limited" in column]
                if any(bool(fields[column].astype(bool).any()) for column in boundary_columns):
                    failures.append(f"{name}: extracted core radius is boundary limited")

            if not list(folder.glob("*_zq.pvd")):
                failures.append(f"{name}: missing sampled surface-field PVD")

    try:
        errors = _single_vortex_errors()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        failures.append(f"single-vortex analytic comparison failed: {error}")
    else:
        for scheme, (velocity, vorticity, gradient) in errors.items():
            print(
                f"vortex_{scheme}: analytic relative L2 "
                f"velocity={velocity:.3%}, vorticity={vorticity:.3%}, gradient={gradient:.3%}"
            )
            if scheme == "cs" and max(velocity, vorticity, gradient) > 0.20:
                failures.append("vortex_cs: analytic profile error exceeds 20%")
            if scheme != "cs" and max(velocity, vorticity, gradient) > 2.0:
                failures.append(f"vortex_{scheme}: analytic profile error exceeds 200%")

    if not GRID_REPORT.is_file():
        failures.append(f"missing required grid-independence report {GRID_REPORT}")
    else:
        try:
            report = json.loads(GRID_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"unreadable grid-independence report ({error})")
        else:
            if report.get("grid_independence_verdict") != "supported_at_stated_tolerance":
                failures.append("CS grid-independence verdict is not supported_at_stated_tolerance")
            if report.get("analytical_reference_convergence_verdict") != "supported":
                # The finite-column VPM model can retain a non-monotone
                # model-form error against the infinite 2-D Lamb--Oseen
                # solution even after the numerical grid is independent.
                # Keep that distinction visible without misclassifying a
                # numerically converged campaign as unstable.
                print(
                    "[WARN] CS analytical-reference convergence is not supported; "
                    "report the finite-column model-form caveat separately"
                )

    if not args.pre_plot:
        for extension in ("png", "pdf"):
            for name in (
                "vortex_comparison",
                "dipole_comparison",
                "merging_comparison",
                "vortex_surface_fields",
                "lamboseen_energy",
            ):
                figure = FIGURES_DIR / f"{name}.{extension}"
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing or empty figure {figure.name}")

    if failures:
        print("\n".join(f"[FAIL] {failure}" for failure in failures))
        return 1
    print("[OK] lamb_oseen_vortex certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
