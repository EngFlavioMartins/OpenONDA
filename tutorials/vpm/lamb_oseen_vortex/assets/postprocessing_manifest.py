#!/usr/bin/env python3
"""Write a non-blocking status/provenance manifest for Lamb-Oseen figures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__:
    from .vortex_diagnostics import FIGURES_DIR, SAMPLES_DIR, SCHEMES
else:
    from vortex_diagnostics import FIGURES_DIR, SAMPLES_DIR, SCHEMES


CASES = ("vortex", "dipole", "merging")


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
    cap_key = {"dvh": "dvh_max_nodes", "gbd": "gbd_max_nodes"}.get(scheme)
    cap = metadata.get(cap_key) if cap_key else None
    if cap and max_particles is not None and max_particles >= float(cap):
        warnings.append(
            f"{scheme.upper()} reached its particle-count guard; inspect late-time sensitivity "
            f"to {cap_key}."
        )
    return warnings


def build_manifest(samples_dir: Path, figures_dir: Path) -> dict:
    runs = {}
    for case in CASES:
        for scheme in SCHEMES:
            name = f"{case}_{scheme}"
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
                "end_time": metadata.get("end_time", metadata.get("total_time")),
                "core_radius_definition": metadata.get("core_radius_definition")
                if metadata
                else None,
                "sample_plane_z": metadata.get("sample_plane_z"),
                "particle_spacing_ratio": (
                    metadata.get("in_plane_spacing", 0.0)
                    / metadata.get("velocity_peak_radius", metadata.get("core_radius", 1.0))
                    if metadata
                    else None
                ),
                "field_spacing_ratio": (
                    metadata.get("field_spacing", 0.0)
                    / metadata.get("velocity_peak_radius", metadata.get("core_radius", 1.0))
                    if metadata
                    else None
                ),
                "max_n_particles_sampled": max_particles,
                "time_step_size": metadata.get("time_step_size"),
                "advection_scheme": metadata.get("advection_scheme"),
                "treecode_theta": metadata.get("treecode_theta"),
                "treecode_multipole_order": metadata.get("treecode_multipole_order"),
                "dvh_rd_ratio": metadata.get("dvh_rd_ratio"),
                "dvh_max_nodes": metadata.get("dvh_max_nodes"),
                "gbd_max_nodes": metadata.get("gbd_max_nodes"),
                "circulation_normalization": metadata.get("circulation_normalization"),
                "raw_retained_circulation_fraction": metadata.get(
                    "raw_retained_circulation_fraction"
                ),
                "random_seed": metadata.get("random_seed"),
                "quality_warnings": _quality_warnings(scheme, metadata, max_particles),
            }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Figures intentionally plot every readable sample; missing/incomplete runs do not fail.",
        "runs": runs,
        "figures": sorted(path.name for path in figures_dir.glob("*.png")),
    }


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
