#!/usr/bin/env python3
"""Validate one completed Lamb--Oseen result from the root samples tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TUTORIAL_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = TUTORIAL_DIR / "samples"
SCHEMES = ("cs", "rwm", "dvh", "gbd")

VELOCITY_ERROR_LIMIT = 0.10
VORTICITY_ERROR_LIMIT = 0.15
PAIR_CIRCULATION_ERROR_LIMIT = 0.20
RUNTIME_LIMIT_SECONDS = 30.0 * 60.0
BETA_RMAX = 1.12


def _flow_time(path: Path) -> float:
    with path.open(encoding="utf-8") as handle:
        line = handle.readline().strip()
    if not line.startswith("# flow_time="):
        raise ValueError(f"missing flow_time header in {path}")
    return float(line.split("=", 1)[1])


def _relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(values - reference) / np.linalg.norm(reference))


def _validate_completion(folder: Path, metadata: dict) -> list[str]:
    failures = []
    flow_integrals = folder / "flow_integrals.csv"
    if not flow_integrals.is_file():
        return ["flow_integrals.csv is missing"]

    history = pd.read_csv(flow_integrals)
    if len(history) < 10:
        failures.append(f"only {len(history)} flow-integral samples were saved")
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number])).all().all():
        failures.append("flow-integral history is empty or non-finite")
        return failures

    end_time = float(history["time"].iloc[-1])
    tolerance = max(float(metadata["time_step"]), float(metadata["sample_interval"]))
    if abs(end_time - float(metadata["total_time"])) > tolerance:
        failures.append(f"history stops at {end_time:.3f}s, expected {metadata['total_time']:.3f}s")

    runtime = metadata.get("wall_time_seconds")
    if runtime is not None and float(runtime) > RUNTIME_LIMIT_SECONDS:
        failures.append(f"runtime {float(runtime) / 60.0:.1f}min exceeds 30min")
    return failures


def _validate_vortex(folder: Path, metadata: dict) -> tuple[list[str], str]:
    path = folder / f"{folder.name}_x.csv"
    if not path.is_file():
        return [f"{path.name} is missing"], ""

    data = pd.read_csv(path, comment="#")
    required = {"x", "Uy", "omega_z"}
    if data.empty or not required.issubset(data.columns):
        return [f"{path.name} has no usable profile"], ""
    if not np.isfinite(data[list(required)]).all().all():
        return [f"{path.name} contains non-finite values"], ""

    x = data["x"].to_numpy()
    gamma = abs(float(metadata["circulations"][0]))
    viscosity = float(metadata["viscosity"])
    core_radius = float(metadata["core_radius"])
    half_length = float(metadata["column_half_length"])
    physical_time = _flow_time(path)
    vortex_age = (core_radius / BETA_RMAX) ** 2 / (4.0 * viscosity)
    radius_squared = 4.0 * viscosity * (vortex_age + physical_time)

    radius = np.abs(x)
    reference_velocity = np.zeros_like(x)
    away_from_axis = radius > np.finfo(float).eps
    reference_velocity[away_from_axis] = (
        gamma
        / (2.0 * np.pi * radius[away_from_axis])
        * (1.0 - np.exp(-(radius[away_from_axis] ** 2) / radius_squared))
        * half_length
        / np.sqrt(half_length**2 + radius[away_from_axis] ** 2)
        * np.sign(x[away_from_axis])
    )
    reference_vorticity = gamma / (np.pi * radius_squared) * np.exp(-(x**2) / radius_squared)

    comparison = np.abs(x) <= 5.5 * core_radius
    velocity_error = _relative_error(
        data.loc[comparison, "Uy"].to_numpy(),
        reference_velocity[comparison],
    )
    vorticity_error = _relative_error(
        data.loc[comparison, "omega_z"].to_numpy(),
        reference_vorticity[comparison],
    )

    failures = []
    if velocity_error > VELOCITY_ERROR_LIMIT:
        failures.append(f"velocity L2 error is {velocity_error:.1%}")
    if vorticity_error > VORTICITY_ERROR_LIMIT:
        failures.append(f"vorticity L2 error is {vorticity_error:.1%}")
    return failures, f"velocity={velocity_error:.1%}, vorticity={vorticity_error:.1%}"


def _validate_pair(folder: Path, metadata: dict) -> tuple[list[str], str]:
    path = folder / "pair_diagnostics.csv"
    if not path.is_file():
        return ["pair_diagnostics.csv is missing"], ""

    data = pd.read_csv(path)
    required = {"flow_time", "x_core", "y_core", "surface_circulation"}
    if len(data) < 10 or not required.issubset(data.columns):
        return ["pair diagnostics are missing or too sparse"], ""
    if not np.isfinite(data[list(required)]).all().all():
        return ["pair diagnostics contain non-finite primary quantities"], ""
    if not data["flow_time"].is_monotonic_increasing:
        return ["pair diagnostic time is not monotonic"], ""

    target_time = float(metadata["total_time"])
    tolerance = max(float(metadata["time_step"]), float(metadata["sample_interval"]))
    end_time = float(data["flow_time"].iloc[-1])
    failures = []
    if abs(end_time - target_time) > tolerance:
        failures.append(f"pair history stops at {end_time:.3f}s, expected {target_time:.3f}s")

    circulations = [abs(float(value)) for value in metadata["circulations"]]
    expected_circulation = circulations[0] if metadata["case"] == "dipole" else sum(circulations)
    measured_circulation = abs(float(data["surface_circulation"].iloc[-1]))
    circulation_error = abs(measured_circulation - expected_circulation) / expected_circulation
    if circulation_error > PAIR_CIRCULATION_ERROR_LIMIT:
        failures.append(f"surface-circulation error is {circulation_error:.1%}")

    if metadata["case"] == "dipole":
        core_radius = data["core_radius"].to_numpy(float)
        initial_core_radius = float(metadata["core_radius"])
        expected_core_radius = np.sqrt(
            initial_core_radius**2
            + 4.0
            * BETA_RMAX**2
            * float(metadata["viscosity"])
            * float(metadata["total_time"])
        )
        x_core = data["x_core"].to_numpy(float)
        y_core = data["y_core"].to_numpy(float)
        if not np.isfinite(core_radius).all() or np.any(core_radius <= 0.0):
            failures.append("final dipole core radius is invalid")
        elif (
            core_radius[-1] <= 2.5 * initial_core_radius
            or np.min(np.diff(core_radius)) < -0.02
        ):
            failures.append("dipole core does not show sustained diffusive growth")
        elif abs(core_radius[-1] / expected_core_radius - 1.0) > 0.35:
            failures.append(
                "final dipole core radius differs from the diffusive scale by "
                f"{abs(core_radius[-1] / expected_core_radius - 1.0):.1%}"
            )
        if np.any(np.diff(x_core) < -1.0e-3) or x_core[-1] < 3.5 * float(metadata["separation"]):
            failures.append("dipole does not translate monotonically through the expected distance")
        if np.max(np.abs(y_core - y_core[0])) > 0.20 * float(metadata["separation"]):
            failures.append("dipole symmetry plane drifts excessively")
    else:
        pair_fields = data[["separation", "core_area", "angle_rad"]]
        if not np.isfinite(pair_fields).all().all():
            failures.append("merging-pair geometry contains non-finite values")
        else:
            separation = data["separation"].to_numpy(float)
            angle = np.unwrap(data["angle_rad"].to_numpy(float))
            initial_separation = float(metadata["separation"])
            if not 0.80 * initial_separation <= separation[0] <= 1.15 * initial_separation:
                failures.append("merging pair starts at an invalid separation")
            if separation.min() >= 0.70 * initial_separation:
                failures.append("merging pair does not approach coalescence")
            if np.ptp(angle) <= np.pi:
                failures.append("merging pair does not complete a physical rotation")
            if data["core_area"].iloc[-1] <= data["core_area"].iloc[0]:
                failures.append("merging core area does not grow")

    details = f"circulation={measured_circulation:.3f}, t={end_time:.3f}s"
    if metadata["case"] == "dipole" and len(data):
        details += (
            f", x={float(data['x_core'].iloc[-1]):.3f}, a={float(data['core_radius'].iloc[-1]):.3f}"
        )
    return failures, details


def validate(case: str, scheme: str) -> tuple[bool, str]:
    folder = SAMPLES_DIR / f"{case}_{scheme}"
    metadata_path = folder / "run_metadata.json"
    if not metadata_path.is_file():
        return False, f"FAIL {case}/{scheme}: run_metadata.json is missing"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    failures = _validate_completion(folder, metadata)
    if case == "vortex":
        case_failures, metrics = _validate_vortex(folder, metadata)
    else:
        case_failures, metrics = _validate_pair(folder, metadata)
    failures.extend(case_failures)

    if failures:
        return False, f"FAIL {case}/{scheme}: " + "; ".join(failures)
    return True, f"PASS {case}/{scheme}: {metrics}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("vortex", "dipole", "merging"))
    parser.add_argument("scheme", choices=SCHEMES)
    arguments = parser.parse_args()

    passed, message = validate(arguments.case, arguments.scheme)
    print(message)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
