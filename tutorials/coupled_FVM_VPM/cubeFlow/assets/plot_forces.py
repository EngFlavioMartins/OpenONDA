#!/usr/bin/env python3
"""Wall-force comparison: coupled FVM vs referenceFlow.

Both series are body-fitted FVM wall integrals over the same cube, so this is an
FVM-vs-FVM comparison: it asks whether replacing the far field with a VPM wake
changed the load on the body, not whether two different force models agree.

Row 0: Cd(t) for both runs.  Row 1: the difference, plus the mean over the
overlapping window once the initial transient has passed.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

TRANSIENT_FRACTION = 0.25


def _series(source: str) -> tuple[np.ndarray, np.ndarray] | None:
    data = util.load_forces(source)
    if data is None or "Cd" not in data or "time" not in data:
        return None
    order = np.argsort(data["time"])
    return np.asarray(data["time"])[order], np.asarray(data["Cd"])[order]


def fig_forces(coupled, reference, fmt: str = "png", dpi: int = 400) -> None:
    t_c, cd_c = coupled
    t_r, cd_r = reference

    lo = max(t_c.min(), t_r.min())
    hi = min(t_c.max(), t_r.max())
    if hi <= lo:
        print(
            f"  forces: not enough overlap yet - coupled [{t_c.min():.3f}, {t_c.max():.3f}] s, "
            f"reference [{t_r.min():.3f}, {t_r.max():.3f}] s"
        )
        return

    grid = np.linspace(lo, hi, 400)
    cd_c_i = np.interp(grid, t_c, cd_c)
    cd_r_i = np.interp(grid, t_r, cd_r)
    delta = cd_c_i - cd_r_i

    settled = grid >= lo + TRANSIENT_FRACTION * (hi - lo)
    mean_c = float(cd_c_i[settled].mean())
    mean_r = float(cd_r_i[settled].mean())
    rel = (mean_c - mean_r) / mean_r if mean_r else float("nan")

    fig, (ax_cd, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 5.4))

    ax_cd.plot(t_r, cd_r, color=util.colour("reference"), label=util.label("reference"))
    ax_cd.plot(t_c, cd_c, color=util.colour("fvm"), label=util.label("fvm"))
    ax_cd.axvspan(lo, lo + TRANSIENT_FRACTION * (hi - lo), color="0.85", zorder=0)
    ax_cd.set_ylabel(r"$C_d$")
    ax_cd.legend(loc="best")
    ax_cd.set_title(
        rf"$\overline{{C_d}}$: coupled {mean_c:.4f}  vs  reference {mean_r:.4f}   "
        rf"($\Delta$ = {rel:+.2%}, shaded = transient, excluded)"
    )

    ax_err.plot(grid, delta, color=util.colour("vpm"))
    ax_err.axhline(0.0, color="0.4", lw=0.8)
    ax_err.set_ylabel(r"$C_d^{\rm coupled} - C_d^{\rm ref}$")
    ax_err.set_xlabel("t [s]")

    util.save(fig, "forces_cd", fmt, dpi)
    plt.close(fig)

    print(f"  settled window : {lo + TRANSIENT_FRACTION * (hi - lo):.3f} .. {hi:.3f} s")
    print(f"  mean Cd coupled: {mean_c:.4f}")
    print(f"  mean Cd ref    : {mean_r:.4f}")
    print(f"  relative diff  : {rel:+.2%}")


def main() -> None:
    coupled = _series("fvm")
    reference = _series("reference")
    if coupled is None or reference is None:
        missing = [
            name
            for name, series in (("coupled", coupled), ("reference", reference))
            if series is None
        ]
        print(f"  forces: no forces_history.csv with a Cd column for {', '.join(missing)}")
        return
    fig_forces(coupled, reference)


if __name__ == "__main__":
    main()
