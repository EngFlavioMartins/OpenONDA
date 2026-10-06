"""Compare force deterioration with the saved particle wake's downstream extent."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from .analyze_force_cycles import complete_cycles

CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
REPORT = Path(__file__).parent / "force_accuracy_study/20261006"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    histories, cycles, hashes = {}, {}, {}
    for name, directory in (("coupled", CASE), ("reference", CASE / "reference_flow")):
        path = directory / "samples/forces_history.csv"
        hashes[str(path)] = digest(path)
        values = np.genfromtxt(path, names=True, delimiter=",")
        histories[name] = values
        rows = complete_cycles(values, "lift_coefficient", 2.0)
        for row in rows:
            selected = (values["time"] >= row["left_trough_time"]) & (
                values["time"] <= row["right_trough_time"]
            )
            row["mean_drag_coefficient"] = float(
                trapezoid(values["drag_coefficient"][selected], values["time"][selected])
                / row["period"]
            )
        cycles[name] = rows
    wake, configurations = [], set()
    for path in sorted((CASE / "solution/vpm").glob("vpm_*.h5")):
        with h5py.File(path) as stored:
            age = float(stored["solver"].attrs["time"])
            configurations.add(stored["solver"].attrs["numerical_configuration_sha256"])
            if age > 40:
                continue
            position = stored["particles/position"][:]
            strength = stored["particles/vortex_strength"][:, 2].astype(float)
            belt = position[:, 0] > 14.5
            row = {
                "time_s": age,
                "particle_count": len(position),
                "maximum_x_m": float(position[:, 0].max()),
                "downstream_belt_particle_count": int(belt.sum()),
                "downstream_belt_absolute_vortex_strength_m3_per_s": float(
                    abs(strength[belt]).sum()
                ),
                "downstream_belt_net_vortex_strength_m3_per_s": float(strength[belt].sum()),
                "source": str(path),
                "sha256": digest(path),
            }
            wake.append(row)
    windows = []
    coupled, reference = histories["coupled"], histories["reference"]
    for start in range(5, 91, 5):
        selected = (coupled["time"] >= start) & (coupled["time"] < start + 10)
        row = {"interval_s": [start, start + 10]}
        for field in ("drag_coefficient", "lift_coefficient"):
            exact = np.interp(coupled["time"][selected], reference["time"], reference[field])
            candidate = coupled[field][selected]
            row[field] = {
                "same_clock_correlation": float(np.corrcoef(candidate, exact)[0, 1]),
                "same_clock_rms_error": float(np.sqrt(np.mean((candidate - exact) ** 2))),
            }
        windows.append(row)
    figure, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, layout="constrained")
    for name, values in histories.items():
        axes[0].plot(values["time"], values["lift_coefficient"], label=name, lw=1)
        rows = cycles[name]
        axes[1].plot(
            [r["peak_time"] for r in rows],
            [r["peak_to_peak_drift_corrected"] for r in rows],
            ".-",
            label=name,
        )
        axes[2].plot(
            [r["peak_time"] for r in rows],
            [r["mean_drag_coefficient"] for r in rows],
            ".-",
            label=name,
        )
    time = [r["time_s"] for r in wake]
    axes[3].plot(time, [r["maximum_x_m"] for r in wake], ".-", color="#404040")
    axes[3].axhline(15, color="#a23c32", ls="--", label="particle cutoff")
    labels = ("Lift coefficient", "Lift cycle peak-to-peak", "Cycle mean drag", "Wake extent x (m)")
    for axis, label in zip(axes, labels, strict=True):
        axis.set_ylabel(label)
        axis.axvspan(16, 18, color="#aaaaaa", alpha=0.2)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[3].legend(frameon=False)
    axes[3].set_xlim(5, 40)
    axes[3].set_xlabel("Original physical time (s); no fitted time shift")
    figure.savefig(REPORT / "force_correlation_onset.png", dpi=170)
    plt.close(figure)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "force_source_sha256": hashes,
        "vpm_numerical_configuration_sha256": sorted(configurations),
        "wake_snapshots": wake,
        "complete_lift_cycles": cycles,
        "same_clock_correlations": windows,
        "interpretation": "Wake extent saturates at the 15 m cutoff around 17 s; stronger shed vorticity reaches the downstream belt around 22-23 s. The coupled lift amplitude drops between its 27.52 s and 33.04 s peaks while reference amplitude continues growing. Drag has an earlier mean deficit. Timing is a mechanism clue; the pre-truncation long-wake experiment is the causal test. Correlation includes accumulated phase drift and does not measure amplitude agreement alone.",
    }
    (REPORT / "force_correlation_onset.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(digest(Path(path)) != expected for path, expected in hashes.items()):
        raise RuntimeError("Original force history changed during read-only audit")
    print("Recorded", len(wake), "wake snapshots and", len(windows), "force correlation windows")


if __name__ == "__main__":
    main()
