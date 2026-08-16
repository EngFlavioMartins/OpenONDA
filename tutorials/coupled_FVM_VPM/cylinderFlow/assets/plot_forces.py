#!/usr/bin/env python3
"""Compare cylinder force histories and shedding statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()

    reference = util.load_forces("reference")
    hybrid = util.load_forces("fvm")
    if reference is None or hybrid is None:
        raise SystemExit("Run referenceFlow/allrun.sh and ./allrun.sh before plotting forces.")

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.2), sharex=True, constrained_layout=True)
    styles = {
        "reference": dict(color=util.COLORS["reference"], ls="-.", label="Reference FVM"),
        "hybrid": dict(color=util.COLORS["hybrid"], ls="-", label="Hybrid FVM-VPM"),
    }
    for name, data in (("reference", reference), ("hybrid", hybrid)):
        axes[0].plot(data["time"], data["Cd"], **styles[name])
        axes[1].plot(data["time"], data["Cl"], **styles[name])
    common_start = max(float(reference["time"][0]), float(hybrid["time"][0]))
    common_end = min(float(reference["time"][-1]), float(hybrid["time"][-1]))
    time = np.linspace(common_start, common_end, 600)
    cd_ref = np.interp(time, reference["time"], reference["Cd"])
    cd_hybrid = np.interp(time, hybrid["time"], hybrid["Cd"])
    scale_start = (
        util.SHEDDING_START
        if util.SHEDDING_START < common_end
        else 0.5 * (common_start + common_end)
    )
    scale = max(float(np.mean(np.abs(cd_ref[time >= scale_start]))), 1.0e-8)
    axes[2].plot(time, 100.0 * np.abs(cd_hybrid - cd_ref) / scale)
    axes[0].set(ylabel=r"$C_D$", title="Cylinder force history")
    axes[1].set(ylabel=r"$C_L$")
    axes[2].set(
        xlabel=r"$tU_\infty/D$",
        ylabel=r"$|\Delta C_D|/\langle|C_D^{ref}|\rangle$ [\%]",
    )
    axes[0].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=0.2)
    util.save(fig, "force_history", args.format)
    plt.close(fig)

    ref_metrics = util.settled_force_metrics(reference)
    hybrid_metrics = util.settled_force_metrics(hybrid)
    print("  settled reference:", ref_metrics)
    print("  settled hybrid:   ", hybrid_metrics)


if __name__ == "__main__":
    main()
