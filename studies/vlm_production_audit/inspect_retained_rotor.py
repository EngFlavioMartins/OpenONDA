"""Read retained rotor evidence without modifying native solution files."""

import ast
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["ROTOR_OUTPUT_TAG"] = ""

import pandas as pd

from source.solvers.vpm.io.checkpoint_audit import audit_checkpoint
from tests._tutorial_helpers import load_tutorial_module


def main():
    """Persist load statistics, native checkpoint clocks and documentation gaps."""
    common = load_tutorial_module("vpm/rotor_flow", "assets._common")
    p = common.rotor_inputs()
    force, bem = common.performance(), common.bem_reference()
    end = float(force.time.max())
    tail = force[force.time > end - 5 * p.rotation_period]
    result = {
        "status": "NOT_PRODUCTION_QUALIFIED",
        "lifecycle": p.metadata["lifecycle"],
        "state": p.metadata["state"],
        "actual_radius": p.rotor_radius,
        "force_time_range": [float(force.time.min()), end],
        "force_clock_duplicates": int(force.time.duplicated().sum()),
        "force_clock_max_gap": float(force.time.diff().max()),
        "averaging_window": [float(tail.time.min()), float(tail.time.max())],
    }
    result["loads"] = {}
    for key, attr in (("CT", "thrust_coefficient"), ("CP", "power_coefficient")):
        values = tail[key]
        split = len(values) // 2
        mean = float(values.mean())
        reference = float(bem.attrs[attr])
        result["loads"][key] = {
            "mean": mean,
            "bem": reference,
            "bem_relative_difference": abs(mean / reference - 1),
            "half_window_relative_drift": abs(
                values.iloc[split:].mean() - values.iloc[:split].mean()
            )
            / abs(mean),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    result["native_planes"] = {}
    for file in sorted(p.samples_dir.glob("wake_*.pvd")):
        entries = ET.parse(file).findall(".//DataSet")
        times = [float(entry.attrib["timestep"]) for entry in entries]
        result["native_planes"][file.stem] = {
            "frames": len(times),
            "time_range": [min(times), max(times)],
        }
    result["streamwise_samples_present"] = sorted(
        file.name for file in p.samples_dir.glob("streamwise_*.csv")
    )
    wake = load_tutorial_module("vpm/rotor_flow", "assets.plot_rotor_wake_planes")
    validator = load_tutorial_module("vpm/rotor_flow", "assets.validate_results")
    try:
        result["finite_distance_diagnostics"] = [
            {
                "name": row["name"],
                "complete_window": row["complete"],
                "axial_scaled_rms": validator._scaled_profile_error(
                    row["actual_axial_induction"], row["reference_axial_induction"], 0.05
                ),
                "azimuthal_scaled_rms": validator._scaled_profile_error(
                    row["actual_tangential_induction"], row["reference_tangential_induction"], 0.02
                ),
                "axial_velocity_rms": validator._rms_error(
                    row["actual_axial_velocity"], row["reference_axial_velocity"]
                ),
                "azimuthal_velocity_rms": validator._rms_error(
                    row["actual_tangential_velocity"], row["reference_tangential_velocity"]
                ),
            }
            for row in wake.finite_distance_profiles(p)
        ]
    except Exception as error:
        result["finite_distance_diagnostics"] = {"error": str(error)}
    result["checkpoints"] = []
    for file in sorted(p.solution_dir.glob("vpm_*.h5")):
        try:
            record = audit_checkpoint(file)
        except Exception as error:
            record = {"path": str(file), "error": str(error)}
        result["checkpoints"].append(record)
    integrals = pd.read_csv(p.samples_dir / "flow_integrals.csv")
    names = [
        "time",
        "n_particles_total",
        "strain_increment_infinity",
        "maximum_particle_vorticity",
        "total_enstrophy",
    ]
    result["last_integrals"] = integrals[names].tail(6).to_dict("records")
    missing = []
    definitions = 0
    for file in (ROOT / "source/solvers/vpm/boundary_elements/vlm").rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions += 1
                if ast.get_docstring(node) is None:
                    missing.append(
                        {
                            "file": str(file.relative_to(ROOT)),
                            "line": node.lineno,
                            "name": node.name,
                        }
                    )
    result["docstrings"] = {"functions_including_nested": definitions, "missing": missing}
    output = Path(__file__).with_name("retained_rotor.json")

    def finite_json(value):
        """Represent unavailable scientific scores as null, never JSON NaN."""
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(item) for item in value]
        return value

    result = finite_json(result)
    output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "loads", "native_planes", "streamwise_samples_present")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
