"""Compare matched wake-boundary controls from the saved 16 s physical state."""

from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from .analyze_force_cycles import FIELDS, complete_cycles

REPORT = Path(__file__).parent / "force_accuracy_study/20261006"
CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def read_accepted(path, name, sources):
    content = path.read_bytes()
    content = content[: content.rfind(b"\n") + 1]
    target = REPORT / "accepted_force_snapshots" / f"pre_truncation_{name}.csv"
    target.write_bytes(content)
    sources[name] = {
        "source": str(path.resolve()),
        "accepted_snapshot": str(target.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return np.genfromtxt(io.BytesIO(content), names=True, delimiter=",")


def main():
    sources = {}
    histories = {
        name: read_accepted(CASE / relative / "samples/forces_history.csv", name, sources)
        for name, relative in (("reference", "reference_flow"), ("original", "."))
    }
    prefix = read_accepted(
        Path("/tmp/openonda-force-precut-tolerance0001-20261006/samples/forces_history.csv"),
        "baseline_prefix",
        sources,
    )
    baseline = read_accepted(
        Path(
            "/tmp/openonda-force-precut-baseline-tolerance0001-continuation-20261006/samples/forces_history.csv"
        ),
        "baseline_continuation",
        sources,
    )
    baseline = np.concatenate([prefix, baseline[baseline["time"] > prefix["time"][-1] + 1e-8]])
    retained = read_accepted(
        Path(
            "/tmp/openonda-force-precut-retain80-tolerance0001-20261006/samples/forces_history.csv"
        ),
        "wake_retained",
        sources,
    )
    baseline["time"] += 16
    retained["time"] += 16
    histories.update(baseline=baseline, wake_retained=retained)
    end = float(min(baseline["time"][-1], retained["time"][-1]))
    measurements = {}
    for name, values in histories.items():
        selected = (values["time"] >= 18) & (values["time"] <= end)
        samples = values[selected]
        time = samples["time"]
        row = {}
        for field in FIELDS:
            reference = np.interp(
                time, histories["reference"]["time"], histories["reference"][field]
            )
            row[field] = {
                "raw_correlation_to_reference": float(np.corrcoef(samples[field], reference)[0, 1]),
                "raw_rmse_to_reference": float(np.sqrt(np.mean((samples[field] - reference) ** 2))),
                "mean": float(trapezoid(samples[field], time) / (time[-1] - time[0])),
                "complete_cycles": [
                    cycle
                    for cycle in complete_cycles(values, field, 2)
                    if cycle["left_trough_time"] >= 18 and cycle["right_trough_time"] <= end
                ],
            }
        measurements[name] = row
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained")
    for i, field in enumerate(FIELDS):
        for name, values in histories.items():
            selected = (values["time"] >= 18) & (values["time"] <= end)
            axes[i, 0].plot(values["time"][selected], values[field][selected], label=name, lw=1)
            cycles = measurements[name][field]["complete_cycles"]
            axes[i, 1].plot(
                [c["peak_time"] for c in cycles],
                [c["peak_to_peak_drift_corrected"] for c in cycles],
                ".-",
                label=name,
            )
        axes[i, 0].set_ylabel(field.replace("_", " "))
        axes[i, 1].set_ylabel("Complete cycle peak-to-peak")
        for axis in axes[i]:
            axis.grid(alpha=0.25)
            axis.set_xlabel("Physical age (s)")
            axis.set_xlim(18, end)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Wake retention from saved physical age 16 s; matched interface tolerance 1e-4")
    figure.savefig(REPORT / "pre_truncation_force_comparison.png", dpi=160)
    plt.close(figure)
    report = {
        "updated_utc": datetime.now(UTC).isoformat(),
        "scope": "Autonomous runs from identical simultaneous FVM and particle states; x=15 versus x=80 particle boundary, identical body mesh, steps, cores and 1e-4 interface tolerance.",
        "time_alignment": "New native clocks plus the known 16 s physical initial age. No fitted phase or force adjustments.",
        "comparison_interval_s": [18, end],
        "sources": sources,
        "measurements": measurements,
    }
    (REPORT / "pre_truncation_force_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print("Accepted common physical interval", [18, end])


if __name__ == "__main__":
    main()
