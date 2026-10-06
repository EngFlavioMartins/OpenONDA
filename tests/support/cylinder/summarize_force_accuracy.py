"""Record accepted complete force cycles from private autonomous controls."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from .analyze_force_cycles import FIELDS, complete_cycles

REPORT = Path(__file__).parent / "force_accuracy_study/20261006"
CONTROLS = (
    "retain80",
    "wallpotential",
    "particle002-fixedwidth",
    "advection0008",
    "fvmx4-developed-v2",
    "exchange0008-developed",
    "exchange0008-developed-continuation",
    "slip10-v2",
    "fvmx4-retain80-slip10-v2",
    "retain80-slip10-exchange0008",
    "precut-baseline16",
    "precut-retain80",
    "precut-tolerance0001",
    "precut-retain80-tolerance0001",
    "precut-baseline-tolerance0001-continuation",
)


def main():
    registry, measurements = {}, {}
    snapshots = REPORT / "accepted_force_snapshots"
    snapshots.mkdir(exist_ok=True)
    reference = json.loads((REPORT / "baseline/force_comparison.json").read_text())["histories"][
        "reference"
    ]["statistics"]
    for name in CONTROLS:
        directory = Path(f"/tmp/openonda-force-{name}-20261006")
        report = json.loads((directory / "experiment.json").read_text())
        content = (directory / "samples/forces_history.csv").read_bytes()
        content = content[: content.rfind(b"\n") + 1]
        destination = snapshots / f"{name}.csv"
        destination.write_bytes(content)
        values = np.genfromtxt(destination, names=True, delimiter=",")
        if name == "exchange0008-developed-continuation":
            prefix = np.genfromtxt(
                "/tmp/openonda-force-exchange0008-at24-20261006/forces_through24.csv",
                names=True,
                delimiter=",",
            )
            values = np.concatenate([prefix, values[values["time"] > prefix["time"][-1] + 1e-8]])
        registry[name] = {
            "directory": str(directory),
            "status": report["status"],
            "parameters": report["parameters"],
            "initial_time_s": report.get("initial_time"),
            "last_accepted_sample_s": float(values["time"][-1]),
            "execution_scheduling": report.get("execution_scheduling"),
        }
        row = {
            "time": float(values["time"][-1]),
            "status": report["status"],
            "source_sha256": hashlib.sha256(content).hexdigest(),
            "source_snapshot": str(destination.resolve()),
        }
        for field in FIELDS:
            cycles = complete_cycles(values, field, 2.0)
            for cycle in cycles:
                selected = (values["time"] >= cycle["left_trough_time"]) & (
                    values["time"] <= cycle["right_trough_time"]
                )
                cycle["mean_drag_coefficient"] = float(
                    trapezoid(values["drag_coefficient"][selected], values["time"][selected])
                    / cycle["period"]
                )
                cycle["amplitude_ratio_to_reference"] = (
                    cycle["peak_to_peak_drift_corrected"]
                    / reference[field]["median_peak_to_peak_drift_corrected"]
                )
                cycle["mean_drag_ratio_to_reference"] = (
                    cycle["mean_drag_coefficient"] / reference["drag_coefficient"]["mean"]
                )
            row[field] = cycles
        diagnostics = [
            json.loads(line)
            for line in (directory / "solution/coupler_diagnostics.jsonl").read_text().splitlines()
            if line.endswith("}")
        ]
        row["accepted_exchanges"] = len(diagnostics)
        row["unconverged_exchanges"] = sum(
            not item["interface_iteration"]["converged"] for item in diagnostics
        )
        measurements[name] = row
        last = row["lift_coefficient"][-1] if row["lift_coefficient"] else None
        print(
            name,
            row["time"],
            [
                last[quantity]
                for quantity in (
                    "mean_drag_ratio_to_reference",
                    "amplitude_ratio_to_reference",
                    "period",
                )
            ]
            if last
            else "no complete shedding cycle",
        )
    now = datetime.now(UTC).isoformat()
    (REPORT / "experiment_plan.json").write_text(
        json.dumps({"updated_utc": now, "experiments": registry}, indent=2) + "\n"
    )
    (REPORT / "current_force_cycles.json").write_text(
        json.dumps(
            {
                "updated_utc": now,
                "scope": "Provisional accepted cycles. Settling is measured, not assumed; no fitted force or phase corrections.",
                "reference_mature_interval_s": [55, 95],
                "controls": measurements,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
