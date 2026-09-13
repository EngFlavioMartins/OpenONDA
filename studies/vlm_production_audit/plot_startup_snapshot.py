"""Render the retained smoke snapshot honestly as startup data, not validation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    """Plot native signed components at the recorded accepted time."""
    summary = json.loads((ROOT / "rotor_smoke_summary.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    for name, radius in (("000", 0.0), ("025", 0.25), ("065", 0.65), ("110", 1.1)):
        data = pd.read_csv(ROOT / f"rotor_smoke/samples/rotor/streamwise_r{name}.csv", comment="#")
        assert np.isfinite(data.to_numpy()).all()
        fields = (1 - data.velocity_x / 7, data.velocity_y / 7, data.velocity_z / 7)
        for axis, field in zip(axes, fields, strict=True):
            axis.plot(data.position_x / 12, field, label=f"r/R = {radius:g}", lw=1.2)
    for axis, label in zip(
        axes,
        ("Axial deficit 1 − ux/U", "Signed transverse uy/U", "Signed transverse uz/U"),
        strict=True,
    ):
        axis.set_ylabel(label)
        axis.axvline(0, color=".5", ls=":", lw=0.8)
        axis.grid(alpha=0.2)
    axes[0].set_title(
        f"Rotor startup snapshot at t = {summary['time']:.3f} s\nRun stopped at wall-time limit; aerodynamic accuracy unqualified"
    )
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("x / design diameter (rotor at 0)")
    fig.savefig(ROOT / "rotor_startup_streamwise.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
