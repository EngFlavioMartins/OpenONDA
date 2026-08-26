#!/usr/bin/env python3
"""Post-run validation and manifest generation for the Lamb--Oseen tutorial.

Default mode runs strict completeness/physics checks.
Use ``--manifest`` to write the JSON status/provenance manifest instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent

if __package__:
    from .vortex_diagnostics import (
        CASES,
        FIGURES_DIR,
        SAMPLES_DIR,
        SCHEMES,
        pvd_time_map,
        resolve_runtime_physics,
    )
else:
    sys.path.insert(0, str(ASSETS_DIR))
    from vortex_diagnostics import (
        CASES,
        FIGURES_DIR,
        SAMPLES_DIR,
        SCHEMES,
        pvd_time_map,
        resolve_runtime_physics,
    )

sys.path.insert(0, str(CASE_DIR))
from lamb_oseen_setup import (
    ADVECTION_SCHEME,
    COLUMN_LENGTH,
    CORE_RADIUS,
    DVH_MAX_NODES,
    DVH_RD_RATIO,
    FIELD_SPACING,
    GBD_MAX_NODES,
    SPACING,
    TIME_STEP_SIZE,
    TREECODE_MULTIPOLE_ORDER,
    TREECODE_THETA,
)

EXPECTED_END_TIME = 103.0 * 0.291
EXPECTED_DT = 0.291 / 9.0


# =============================================================
# Validation helpers
# =============================================================


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
    from plot_vortex_comparison import lamb_oseen_gradient, lamb_oseen_profile, load_profile

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


def validate(pre_plot: bool) -> int:
    failures: list[str] = []
    for physics_id, _, _ in CASES:
        for scheme in SCHEMES:
            name = f"{physics_id}_{scheme}"
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

    if not pre_plot:
        for extension in ("png", "pdf"):
            for fig_name in (
                "vortex_comparison",
                "dipole_comparison",
                "merging_comparison",
                "vortex_surface_fields",
                "lamboseen_energy",
            ):
                figure = FIGURES_DIR / f"{fig_name}.{extension}"
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing or empty figure {figure.name}")

    if failures:
        print("\n".join(f"[FAIL] {failure}" for failure in failures))
        return 1
    print("[OK] lamb_oseen_vortex certification passed")
    return 0


# =============================================================
# Manifest generation
# =============================================================


def _metadata(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _last_time(path: Path, column: str) -> tuple[int, float | None]:
    try:
        frame = pd.read_csv(path, on_bad_lines="skip")
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return 0, None
    return len(frame), (float(values.max()) if not values.empty else None)


def _quality_warnings(scheme: str, metadata: dict, max_particles: float | None) -> list[str]:
    warnings = []
    if scheme == "rwm" and metadata:
        warnings.append("RWM is a single realization; it is not an ensemble estimate.")
    cap = {"dvh": DVH_MAX_NODES, "gbd": GBD_MAX_NODES}.get(scheme)
    if cap and max_particles is not None and max_particles >= float(cap):
        warnings.append(
            f"{scheme.upper()} reached its particle-count guard; inspect late-time sensitivity."
        )
    return warnings


def build_manifest(samples_dir: Path, figures_dir: Path) -> dict:
    runs = {}
    for case_id, _, _ in CASES:
        for scheme in SCHEMES:
            name = f"{case_id}_{scheme}"
            folder = samples_dir / name
            metadata = _metadata(folder / "run_metadata.json")
            field_rows, field_time = _last_time(folder / "field_diagnostics.csv", "time")
            integral_rows, integral_time = _last_time(folder / "flow_integrals.csv", "time")
            _, max_particles = _last_time(folder / "flow_integrals.csv", "n_particles_total")
            has_samples = field_rows > 0 or integral_rows > 0 or any(folder.glob("*_zq_*.vts"))
            complete = metadata.get("completed") is True or metadata.get("status") == "complete"
            if complete:
                status = "complete"
            elif metadata or has_samples:
                status = str(metadata.get("status", "partial"))
            else:
                status = "missing"
            runs[name] = {
                "status": status,
                "complete": complete,
                "field_rows": field_rows,
                "last_field_time": field_time,
                "integral_rows": integral_rows,
                "last_integral_time": integral_time,
                "end_time": metadata.get("end_time"),
                "core_radius_definition": "gaussian_1_over_e_vorticity_radius",
                "sample_plane_z": 0.25 * COLUMN_LENGTH,
                "particle_spacing_ratio": SPACING / CORE_RADIUS,
                "field_spacing_ratio": FIELD_SPACING / CORE_RADIUS,
                "max_n_particles_sampled": max_particles,
                "time_step_size": TIME_STEP_SIZE,
                "advection_scheme": ADVECTION_SCHEME,
                "treecode_theta": TREECODE_THETA,
                "treecode_multipole_order": TREECODE_MULTIPOLE_ORDER,
                "dvh_rd_ratio": DVH_RD_RATIO,
                "dvh_max_nodes": DVH_MAX_NODES,
                "gbd_max_nodes": GBD_MAX_NODES,
                "circulation_normalization": "per_vortex_after_strength_cutoff",
                "quality_warnings": _quality_warnings(scheme, metadata, max_particles),
            }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Figures intentionally plot every readable sample; missing/incomplete runs do not fail.",
        "runs": runs,
        "figures": sorted(path.name for path in figures_dir.glob("*.png")),
    }


def write_manifest() -> int:
    manifest = build_manifest(SAMPLES_DIR, FIGURES_DIR)
    output = FIGURES_DIR / "postprocessing_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output)
    counts = {}
    for run in manifest["runs"].values():
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    print(f"  [status] {counts}; wrote {output}")
    return 0


# =============================================================
# CLI
# =============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-plot", action="store_true", help="skip figure existence checks")
    parser.add_argument("--manifest", action="store_true", help="write JSON status manifest")
    args = parser.parse_args()
    if args.manifest:
        return write_manifest()
    return validate(pre_plot=args.pre_plot)


if __name__ == "__main__":
    raise SystemExit(main())
