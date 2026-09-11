"""Field peaks and material-label proxies against the Cheng trajectory.

Chart contract: static scientific small multiples, identical x/R0 and R/R0
axes, blue computed data and neutral published trajectories. Field peaks are
observations with no forced identities after merger. Dotted material-label
proxies are explicitly distinguished, and are not used to certify agreement.
"""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from .. import setup

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .study import STUDY_DIR
from .ring_metrics import load_study_metadata, metadata_settings
from source.solvers.vpm.diagnostics.axisymmetric_field import azimuthal_vorticity

OUT = setup.TUTORIAL_DIR / "figures/study/physics"
PEAKS = setup.TUTORIAL_DIR / "figures/study/core_diagnosis"


def label(settings):
    scheme = settings.get("diffusion", "CS")
    name = {
        "CS": "Core spreading",
        "GBD": "GBD: " + settings.get("gbd_remeshing", "M4_PRIME").replace("_PRIME", "′"),
        "RWM": "Random walk",
    }[scheme]
    return name + (f"\nh/R₀ = {settings['spacing']:g}, σ₀/h = {settings.get('core_ratio', 2):g}")


def history(run, filename, *, peaks=False):
    """Read one run's recorded samples without reconstructing continuations."""
    folder = STUDY_DIR / run
    path = PEAKS / f"{run}_peaks.csv" if peaks else folder / "samples/diagnostics" / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--section-time", type=float)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(Path(__file__).parent / "references/leapfrogging_lbm_trajectory.csv")
    fig, axes = plt.subplots(
        1, len(args.runs), figsize=(4.6 * len(args.runs), 4.8), squeeze=False, sharey=True
    )
    for ax, run in zip(axes.flat, args.runs):
        folder = STUDY_DIR / run
        meta = load_study_metadata(folder)
        if not meta:
            raise FileNotFoundError(folder / "solution/vpm_metadata.json")
        settings = metadata_settings(meta)
        status = meta.get("lifecycle", {}).get("status", "unknown")
        rings = history(run, "ring_diagnostics.csv")
        integrals = history(run, "flow_integrals.csv")
        if settings["scenario"] == "leapfrog":
            for _, group in ref.groupby("ring"):
                ax.plot(group.x_over_R0 - 2.5, group.R_over_R0, color="#555555", lw=1.4)
        for _, group in rings.groupby("group_id"):
            ax.plot(group.vortex_centroid_x, group.major_radius, color="#2874A6", ls=":", lw=1.3)
        peaks = history(run, "", peaks=True)
        if not peaks.empty:
            ax.scatter(peaks.x, peaks.radius, color="#2874A6", s=27, zorder=4)
        capacity_reached = bool((integrals.n_particles_total >= settings["capacity"]).any())
        cap_note = (
            "Particle cap reached: screening only"
            if capacity_reached
            else "No sampled particle-cap contact"
        )
        ax.set(
            title=label(settings),
            xlabel="Axial position from initial midpoint, x/R₀",
            xlim=(-0.6, 8),
            ylim=(0.5, 1.45),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.text(
            0.03,
            0.03,
            f"t = {rings.time.max():.2f} · {status}\n{cap_note}",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )
    axes[0, 0].set_ylabel("Core / ring radius, R/R₀")
    fig.suptitle("Core transport controls: agreement with LBM is the test", fontsize=15, y=0.98)
    fig.legend(
        handles=[
            Line2D([], [], color="#555555", label="LBM core trajectory"),
            Line2D(
                [], [], color="#2874A6", marker="o", ls="", label="Azimuthally averaged field peaks"
            ),
            Line2D([], [], color="#2874A6", ls=":", label="Material-label radius proxy"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.06,
        0.02,
        "Cheng et al. (2015), Fig. 5(b), ReΓ = 3000. Only the axial origin is shifted. No time or distance fitting.\nMaterial labels may split or mix during regeneration; their continuing curves do not prove coherent rings.",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.72, bottom=0.24, wspace=0.12)
    fig.savefig(OUT / "trajectories.png", dpi=180)
    plt.close(fig)
    if args.section_time is not None:
        sections(args.runs, args.section_time)


def sections(runs, target_time):
    fields = []
    for run in runs:
        folder = STUDY_DIR / run
        candidates = list((folder / "samples/diagnostics/particles").glob("*.npz"))
        selected = None
        for path in candidates:
            with np.load(path) as state:
                if np.isclose(float(state["time"]), target_time, atol=1e-8, rtol=0):
                    selected = path
                    break
        if selected is None:
            raise ValueError(f"{run} has no field at t={target_time}")
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
        cache = OUT / f"{run}_section_t{target_time:g}.npz"
        if cache.exists() and str(np.load(cache)["sha256"]) == digest:
            fields.append(dict(np.load(cache)))
            continue
        state = dict(np.load(selected))
        weight = np.linalg.norm(state["vortex_strength"], axis=1)
        centre = np.average(state["position"][:, 0], weights=weight)
        x = np.linspace(centre - 1.3, centre + 1.3, 111)
        r = np.linspace(0.35, 1.5, 76)
        xx, rr = np.meshgrid(x, r, indexing="ij")
        omega = azimuthal_vorticity(
            state["position"],
            state["vortex_strength"],
            state["core_radius"],
            np.c_[xx.ravel(), rr.ravel()],
        ).reshape(xx.shape)
        data = dict(x=x, r=r, omega=omega, sha256=digest)
        np.savez_compressed(cache, **data)
        fields.append(data)
        print("section", run, "t", target_time, flush=True)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 4), squeeze=False, sharey=True)
    for ax, run, field in zip(axes.flat, runs, fields):
        meta = load_study_metadata(STUDY_DIR / run)
        if not meta:
            raise FileNotFoundError(STUDY_DIR / run / "solution/vpm_metadata.json")
        contour = ax.contourf(
            field["x"],
            field["r"],
            field["omega"].T / 100,
            levels=np.linspace(0, 0.7, 15),
            cmap="Blues",
            extend="both",
        )
        ax.contour(
            field["x"],
            field["r"],
            field["omega"].T / 100,
            levels=[0.01, 0.03, 0.1, 0.2, 0.3, 0.5],
            colors="#2874A6",
            linewidths=0.5,
        )
        ax.set(title=label(metadata_settings(meta)), xlabel="x/R₀", aspect="equal")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("r/R₀")
    fig.colorbar(
        contour,
        ax=list(axes.flat),
        label="Mean azimuthal vorticity / initial peak",
        shrink=0.65,
        pad=0.02,
    )
    fig.suptitle(
        f"Represented core sections at the same time: tΓ₀/R₀² = {target_time * np.pi:.2f}",
        fontsize=14,
    )
    fig.text(
        0.06,
        0.04,
        "Azimuthal average of the Gaussian vorticity field. Independent x windows follow each system; coordinates retain the same origin.\nCore deformation is visible here; a 3D instability requires an additional azimuthal-mode or full-field check.",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(OUT / f"core_sections_t{target_time:g}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
