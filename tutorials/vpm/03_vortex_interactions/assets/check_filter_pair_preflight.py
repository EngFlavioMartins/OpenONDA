#!/usr/bin/env python3
"""Dry-validate the held seeded CS ``Cs=.20/0`` qualification pair.

This script constructs immutable case configurations and analytic initial
particle sets, then compares their primary arrays and mutual induction.  It
does not construct a solver, initialize Taichi, or advance either case.
Run it from the tutorial directory so imports resolve to the installed runtime
generation used by the eventual pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from openonda.tutorial_runner import case_package

if not __package__:
    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

from source.solvers.vpm.config.fingerprint import numerical_configuration
from source.solvers.vpm.kernels import gaussian

from .. import setup_les
from .check_initial_coverage_field import (
    _centreline_probes,
    _evaluate_field,
    _relative_rms,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "cs_filter_pair_preflight"
CONTROL_NAME = "cs_breakdown_filter_cs020_cpu_t6_step080"
MOLECULAR_NAME = "cs_breakdown_filter_cs000_cpu_t6_step080"
EXPECTED_GAUSSIAN_SHA256 = "de05441d7da429c171f4609554c19f71c569ec02616a3eea95129914667484aa"
PRIMARY_FIELDS = (
    "position",
    "velocity",
    "vortex_strength",
    "core_radius",
    "particle_volume",
    "kinematic_viscosity",
    "group_id",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _flatten_differences(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                rows.append({"path": path, "control": left.get(key), "molecular": right.get(key)})
            else:
                rows.extend(_flatten_differences(left[key], right[key], path))
        return rows
    if isinstance(left, list) and isinstance(right, list):
        rows = []
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                rows.append(
                    {
                        "path": path,
                        "control": left[index] if index < len(left) else None,
                        "molecular": right[index] if index < len(right) else None,
                    }
                )
            else:
                rows.extend(_flatten_differences(left[index], right[index], path))
        return rows
    if left != right:
        return [{"path": prefix, "control": left, "molecular": right}]
    return []


def _build_case(name: str, smagorinsky: float):
    return setup_les.build_case(
        "baseline",
        scenario="seeded_breakdown",
        compute_device="CPU",
        steps=80,
        wall_minutes=5,
        qualification=True,
        particle_spacing=0.06,
        particle_core_radius=0.06,
        smagorinsky_coefficient=smagorinsky,
        case_name=name,
    )


def _cloud(case) -> dict[str, np.ndarray]:
    rings = [initial_condition.build() for initial_condition in case.initial_conditions]
    return {
        field: np.concatenate([np.asarray(getattr(ring, field)) for ring in rings])
        for field in PRIMARY_FIELDS
    }


def _mutual_field(cloud: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    targets = np.vstack((_centreline_probes(-0.5), _centreline_probes(0.5)))
    results = []
    for target_group, source_group in ((0, 1), (1, 0)):
        selected = cloud["group_id"] == source_group
        sources = {
            "position": cloud["position"][selected],
            "vortex_strength": cloud["vortex_strength"][selected],
            "core_radius": cloud["core_radius"][selected],
        }
        target = targets[target_group * 64 : (target_group + 1) * 64]
        results.append(_evaluate_field(target, sources))
    return np.vstack([row[0] for row in results]), np.vstack([row[1] for row in results])


def _command(case_name: str, smagorinsky: float) -> str:
    return " ".join(
        (
            "env TI_CPU_MAX_NUM_THREADS=6",
            "/opt/anaconda3/envs/OpenONDA/bin/python",
            "setup_les.py",
            "--variant baseline",
            "--scenario seeded_breakdown",
            "--compute-device CPU",
            "--qualification",
            "--steps 80",
            "--wall-minutes 5",
            "--particle-spacing .06",
            "--particle-core-radius .06",
            f"--smagorinsky {smagorinsky:g}",
            f"--case-name {case_name}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    gaussian_path = Path(gaussian.__file__).resolve()
    if "site-packages" not in gaussian_path.parts:
        raise RuntimeError(
            f"preflight must use the installed source generation; resolved {gaussian_path}"
        )
    gaussian_hash = _sha256_file(gaussian_path)
    if gaussian_hash != EXPECTED_GAUSSIAN_SHA256:
        raise RuntimeError(
            f"installed Gaussian generation differs from the approved pair: {gaussian_hash}"
        )

    cases = {
        "control_cs020": _build_case(CONTROL_NAME, 0.20),
        "molecular_cs000": _build_case(MOLECULAR_NAME, 0.0),
    }
    configurations = {name: numerical_configuration(case.numerics) for name, case in cases.items()}
    differences = _flatten_differences(
        configurations["control_cs020"], configurations["molecular_cs000"]
    )
    expected_difference = [
        {
            "path": "turbulence.smagorinsky_coefficient",
            "control": 0.2,
            "molecular": 0.0,
        }
    ]
    if differences != expected_difference:
        raise RuntimeError(f"unexpected numerical-configuration differences: {differences}")

    clouds = {name: _cloud(case) for name, case in cases.items()}
    array_checks = []
    for field in PRIMARY_FIELDS:
        control = clouds["control_cs020"][field]
        molecular = clouds["molecular_cs000"][field]
        exact = bool(np.array_equal(control, molecular))
        row = {
            "field": field,
            "shape": list(control.shape),
            "dtype": str(control.dtype),
            "exact": exact,
            "maximum_absolute_difference": float(
                np.max(np.abs(control.astype(np.float64) - molecular.astype(np.float64)))
            ),
            "control_sha256": _sha256_array(control),
            "molecular_sha256": _sha256_array(molecular),
        }
        array_checks.append(row)
        if not exact:
            raise RuntimeError(f"initial primary array differs: {field}")

    mutual = {name: _mutual_field(cloud) for name, cloud in clouds.items()}
    velocity_difference = mutual["molecular_cs000"][0] - mutual["control_cs020"][0]
    gradient_difference = mutual["molecular_cs000"][1] - mutual["control_cs020"][1]
    field_checks = {
        "probe_definition": "64 analytic disturbed-centreline probes per target ring",
        "mutual_velocity_relative_rms": _relative_rms(
            mutual["molecular_cs000"][0], mutual["control_cs020"][0]
        ),
        "mutual_velocity_maximum_absolute_difference": float(np.max(np.abs(velocity_difference))),
        "mutual_gradient_relative_rms": _relative_rms(
            mutual["molecular_cs000"][1], mutual["control_cs020"][1]
        ),
        "mutual_gradient_maximum_absolute_difference": float(np.max(np.abs(gradient_difference))),
    }
    if any(value != 0.0 for key, value in field_checks.items() if key != "probe_definition"):
        raise RuntimeError(f"initial mutual field differs: {field_checks}")

    viscosity = np.pi / 3415.0
    mode_wavelength = 2.0 * np.pi / 8.0
    molecular_particle_core = np.sqrt(0.06**2 + 4.0 * viscosity * 0.6)
    molecular_physical_core = np.sqrt(0.1**2 + 4.0 * viscosity * 0.6)
    historical_control_core = 0.1046576553144179
    signal = {
        "time": 0.6,
        "mode8_wavelength_at_R0": mode_wavelength,
        "seed_amplitude": 0.05,
        "seed_over_wavelength": 0.05 / mode_wavelength,
        "molecular_only_particle_core": molecular_particle_core,
        "historical_cs020_particle_core": historical_control_core,
        "expected_particle_core_separation": historical_control_core - molecular_particle_core,
        "expected_separation_over_wavelength": (historical_control_core - molecular_particle_core)
        / mode_wavelength,
        "expected_separation_over_seed": (historical_control_core - molecular_particle_core) / 0.05,
        "molecular_physical_core": molecular_physical_core,
        "historical_cross_plane_core_width_group_0": 0.13086376777542777,
        "historical_cross_plane_core_width_group_1": 0.12329109393632033,
    }
    commands = {
        "control_first": _command(CONTROL_NAME, 0.20),
        "molecular_second_if_budget_allows": _command(MOLECULAR_NAME, 0.0),
    }
    result = {
        "status": "PREPARED_NOT_LAUNCHED",
        "python": sys.executable,
        "installed_gaussian_path": str(gaussian_path),
        "installed_gaussian_sha256": gaussian_hash,
        "run_order": [CONTROL_NAME, MOLECULAR_NAME],
        "commands": commands,
        "configuration_differences": differences,
        "primary_initial_arrays": array_checks,
        "optional_initial_fields": {"zone_id": "absent from both analytic particle sets"},
        "initial_mutual_field": field_checks,
        "expected_early_signal": signal,
        "excluded_initial_equality_fields": {
            "eddy_viscosity": "expected to differ because Cs is the isolated parameter",
            "effective_viscosity": "expected to differ because it includes eddy viscosity",
        },
        "budget_protocol": {
            "remaining_before_pair": "about 13:14",
            "native_cap_each_seconds": 300,
            "reserved_finalization_and_overshoot": "about 3:14",
            "note": (
                "native caps are checked between steps; record actual control time and "
                "withhold or reduce the second leg if the original envelope would be exceeded"
            ),
        },
    }
    (output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Seeded CS filter-pair preflight",
        "",
        "**Status: prepared, not launched.**",
        "",
        f"Installed Gaussian: `{gaussian_path}`",
        f"SHA-256: `{gaussian_hash}`",
        "",
        "All seven primary analytic particle arrays are exactly equal; optional `zone_id` is absent from both initial particle sets. The numerical configuration differs only at `turbulence.smagorinsky_coefficient` (`.20` versus `0`). Initial mutual velocity and gradient are exactly equal on 128 disturbed-centreline probes. Initial eddy/effective viscosity is excluded from the equality gate because it is the intended consequence of changing `Cs`.",
        "",
        "## Held sequential commands",
        "",
        "Run from this tutorial directory with `TI_CPU_MAX_NUM_THREADS=6`, only after explicit compute-slot release.",
        "",
        "```sh",
        commands["control_first"],
        "```",
        "",
        "Record actual control wall time and remaining envelope before starting the second leg.",
        "",
        "```sh",
        commands["molecular_second_if_budget_allows"],
        "```",
        "",
        "Each command has a five-minute native cap. The aggregate reservation is not a hard process-wall maximum because caps are checked only between accepted steps.",
    ]
    (output / "preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"filter-pair preflight passed; wrote {output}")


if __name__ == "__main__":
    main()
