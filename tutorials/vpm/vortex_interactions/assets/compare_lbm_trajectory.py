"""Compare saved stabilization trajectories with the digitized Cheng LBM curves.

Chart contract: six matched h=.035 circular-support panels, radius versus axial
position; blue VPM and neutral LBM, solid/dashed ring identity. Static scientific
figure with fixed axes, no phase fitting, stretching or extrapolation. Scores use
uniform shared axial grids in explicitly specified windows, equal ring weights.
RMS particle-cloud radius is a core-centre proxy, not an identical observable.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .analyze_study import load_runs

HERE = Path(__file__).parent
OUT = HERE.parent / "figures" / "study" / "lbm_comparison"
CASES = [
    ("leapfrog_baseline", "Baseline"),
    ("qualified_leapfrog_splitting_h035", "Baseline + splitting"),
    ("leapfrog_stretching_viscosity", "Baseline + residual viscosity"),
    ("qualified_leapfrog_pedrizzetti_h035", "Baseline + normalized realignment (0.3)"),
    ("leapfrog_p_moments", "Baseline + moment-corrected P (0.03)"),
    ("qualified_leapfrog_remeshing_h035", "Baseline + remeshing (cost screen)"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(HERE / "references" / "leapfrogging_lbm_trajectory.csv")
    runs = {r[0]: r for r in load_runs([n for n, _ in CASES])}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.3), sharex=True, sharey=True)
    scores = []
    for ax, (name, label) in zip(axes.flat, CASES):
        _, _, meta, _, rings = runs[name]
        for group, refgroup, style in [(0, 2, "-"), (1, 1, "--")]:
            ref = reference[reference.ring == refgroup].sort_values("x_over_R0")
            ring = rings[rings.group_id == group].sort_values("step")
            if not np.all(np.diff(ring.vortex_centroid_x) > 0):
                raise ValueError(f"Non-monotonic path cannot be scored as R(x): {name}")
            ax.plot(ref.x_over_R0 - 2.5, ref.R_over_R0, color="#444444", ls=style, lw=1.6)
            ax.plot(ring.vortex_centroid_x, ring.major_radius, color="#2874A6", ls=style, lw=1.6)
        ax.set_title(f"{label}\n{int(rings.step.max())} steps sampled", fontsize=10)
        ax.set(xlim=(-0.6, 7.3), ylim=(0.6, 1.4))
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        for lower, upper in [(0.55, 1.5), (0.55, 3.5), (0.55, 7.0)]:
            errors = []
            for group, refgroup in [(0, 2), (1, 1)]:
                ring = rings[rings.group_id == group].sort_values("vortex_centroid_x")
                ref = reference[reference.ring == refgroup].sort_values("x_over_R0")
                rx = ref.x_over_R0.to_numpy() - 2.5
                x = ring.vortex_centroid_x.to_numpy()
                if lower < max(x.min(), rx.min()) or upper > min(x.max(), rx.max()):
                    break
                targets = np.linspace(lower, upper, 401)
                errors.append(
                    np.interp(targets, x, ring.major_radius) - np.interp(targets, rx, ref.R_over_R0)
                )
            if len(errors) == 2:
                scores.append(
                    dict(
                        run=name,
                        method=label,
                        x_min=lower,
                        x_max=upper,
                        radius_rmse_over_R0=float(np.sqrt(np.mean(np.square(errors)))),
                    )
                )
    for ax in axes[-1]:
        ax.set_xlabel("Axial position from initial midpoint, x/R₀")
    for ax in axes[:, 0]:
        ax.set_ylabel("Ring radius, R/R₀")
    fig.suptitle("Vortex radius versus travelled axial position", fontsize=16, y=0.98)
    fig.legend(
        handles=[
            Line2D([], [], color="#2874A6", label="VPM"),
            Line2D([], [], color="#444444", label="LBM: Cheng et al., Fig. 5(b)"),
            Line2D([], [], color="#777777", ls="-", label="Initially trailing ring"),
            Line2D([], [], color="#777777", ls="--", label="Initially leading ring"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.05,
        0.02,
        "Same h = 0.035, radial seed = 0.05 R₀, Cs = 0.2. LBM reference is unperturbed. Only axial origin shifted; no phase fitting.\nVPM radius is a strength-weighted RMS cloud radius; LBM tracks the core centre. Short curves are not extrapolated.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(top=0.81, bottom=0.16, hspace=0.36, wspace=0.13)
    fig.savefig(OUT / "radius_vs_distance.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(scores).to_csv(OUT / "radius_errors.csv", index=False)
    print(pd.DataFrame(scores).to_string(index=False))


if __name__ == "__main__":
    main()
