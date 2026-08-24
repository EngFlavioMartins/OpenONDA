#!/usr/bin/env python3
"""Rotor wake validation — axial velocity vs momentum theory.

Reads the streamwise-normal crossflow sampler planes and overlays
time- and azimuthally-averaged axial-velocity profiles against
actuator-disk momentum theory:

    rotor disk:   u/U∞ = 1 − a
    far wake:     u/U∞ = 1 − 2a

The induction factor is inferred from the sampled thrust coefficient. Planes
that have not reached a statistically stationary state are shown dashed.

Saves: figures/rotor_wake_planes.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    build_arg_parser,
    build_rotor_style_map,
    DENSITY,
    FIGURES_DIR,
    FREESTREAM_SPEED,
    HUB_RADIUS,
    load_theme,
    read_operating_point,
    read_time_step,
    ROTOR_RADIUS,
    SAMPLES_DIR,
    TIP_SPEED_RATIO,
)
from rotor_theory import axial_induction_factor_from_thrust_coefficient  # noqa: E402


# ==============================================================================
# File helpers
# ==============================================================================


def _step_number(path: Path) -> int:
    match = re.search(r"_(\d+)\.vts$", path.name)
    return int(match.group(1)) if match else -1


def _plane_files(samples_dir: Path, tag: str) -> list[Path]:
    return sorted(samples_dir.glob(f"slice_{tag}_*.vts"), key=_step_number)


def _discover_planes(samples_dir: Path, rotor_radius: float) -> list[tuple[str, str]]:
    tags: list[tuple[float, str]] = []
    for pvd in sorted(samples_dir.glob("slice_x*m.pvd")):
        match = re.fullmatch(r"slice_x(?P<x>[-+]?\d+(?:\.\d+)?)m\.pvd", pvd.name)
        if match:
            tags.append((float(match.group("x")), f"x{match.group('x')}m"))

    if not tags:
        tags = [(1.5 * rotor_radius, f"x{int(round(1.5 * rotor_radius))}m")]
        tags += [(3.0 * rotor_radius, f"x{int(round(3.0 * rotor_radius))}m")]
        tags += [(4.5 * rotor_radius, f"x{int(round(4.5 * rotor_radius))}m")]

    tags.sort(key=lambda item: item[0])
    return [(tag, rf"${x_loc / rotor_radius:g}R$") for x_loc, tag in tags]


def _select_time_window(
    files: list[Path],
    time_step_size: float,
    angular_velocity: float,
    averaging_rotations: float,
    tail_fraction: float,
) -> list[Path]:
    """Trailing window of ``averaging_rotations`` rotor revolutions."""
    if not files:
        return []
    if averaging_rotations > 0.0 and time_step_size > 0.0 and angular_velocity > 0.0:
        t_max = _step_number(files[-1]) * time_step_size
        t_cut = t_max - averaging_rotations * 2.0 * np.pi / angular_velocity
        selected = [f for f in files if _step_number(f) * time_step_size >= t_cut]
    else:
        n_tail = max(1, int(np.ceil(len(files) * tail_fraction)))
        selected = files[-n_tail:]
    return selected or [files[-1]]


# ==============================================================================
# Profiles and stationarity
# ==============================================================================


def _plane_statistics(grid, freestream_speed: float, rotor_radius: float, radial_edges: np.ndarray):
    """Return (radial profile of u_x/U∞, disc-averaged u_x/U∞) for one snapshot."""
    pts = np.asarray(grid.points)
    vel = np.asarray(grid.point_data["velocity"])
    normalized_radial_position = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2) / rotor_radius
    ux_over_U = vel[:, 0] / freestream_speed

    bin_idx = np.digitize(normalized_radial_position, radial_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(radial_edges) - 1) & np.isfinite(ux_over_U)
    sums = np.zeros(len(radial_edges) - 1)
    counts = np.zeros(len(radial_edges) - 1)
    np.add.at(sums, bin_idx[valid], ux_over_U[valid])
    np.add.at(counts, bin_idx[valid], 1.0)
    profile = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0.0)

    disc = (normalized_radial_position <= 1.0) & np.isfinite(ux_over_U)
    disc_mean = float(np.mean(ux_over_U[disc])) if disc.any() else np.nan
    return profile, disc_mean


def _drift(disc_means: np.ndarray) -> float:
    """Relative drift of the disc-averaged deficit across the averaging window.

    Compares the first and second halves of the window.  A wake front still in
    transit produces a large one-sided drift; a converged wake produces noise
    about zero.
    """
    finite = disc_means[np.isfinite(disc_means)]
    if finite.size < 4:
        return np.inf
    half = finite.size // 2
    first, second = np.mean(finite[:half]), np.mean(finite[half:])
    scale = abs(np.mean(finite))
    return np.inf if scale < 1e-12 else abs(second - first) / scale


# ==============================================================================
# Plot
# ==============================================================================


def plot_wake_planes(args) -> int:
    if pv is None:
        print("  [WARNING] pyvista unavailable — skipping rotor wake-plane plot.")
        return 1

    samples = SAMPLES_DIR
    fmt = getattr(args, "format", "png")
    out = FIGURES_DIR / f"rotor_wake_planes.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    rotor_radius = ROTOR_RADIUS
    freestream_speed = FREESTREAM_SPEED
    angular_velocity = TIP_SPEED_RATIO * freestream_speed / rotor_radius
    averaging_rotations = 3.0
    tail_fraction = 0.25
    radial_bins = 32
    radial_limit = 1.25
    drift_tolerance = 0.01

    # -- Time step: from the run, never a hardcoded guess -----------------
    time_step_size = read_time_step(samples)
    if time_step_size is None:
        print(
            "  [WARNING] could not determine the run's time step — falling back to tail fraction."
        )
        time_step_size = 0.0

    # -- Reference induction: the run's own operating point ---------------
    operating_point = read_operating_point(
        samples,
        density=DENSITY,
        freestream_speed=freestream_speed,
        rotor_radius=rotor_radius,
        tip_speed_ratio=TIP_SPEED_RATIO,
    )
    if operating_point is not None:
        a_ref = axial_induction_factor_from_thrust_coefficient(operating_point[0])
        a_source = rf"from run $C_T={operating_point[0]:.3f}$"
    else:
        a_ref, a_source = 1.0 / 3.0, "Betz design value"
        print("  [WARNING] vlm_forces.csv not found — falling back to the Betz a = 1/3.")

    planes = _discover_planes(samples, rotor_radius)
    colors, _ = load_theme()
    styles = build_rotor_style_map(colors)
    s_ref = styles["reference"]
    s_vpm = styles["vpm"]
    fig, ax = plt.subplots(figsize=(12.8 / 2.54, 7.4 / 2.54))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.14, top=0.90)

    markers = ["s", "o", "^"]
    n_planes = len(planes)
    alphas = np.linspace(0.5, 1.0, n_planes) if n_planes > 1 else [0.8]
    radial_edges = np.linspace(0.0, radial_limit, radial_bins + 1)
    radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])

    found = False
    unconverged: list[str] = []
    ux_min, ux_max = np.inf, -np.inf
    for i, (tag, label) in enumerate(planes):
        files = _select_time_window(
            _plane_files(samples, tag),
            time_step_size=time_step_size,
            angular_velocity=angular_velocity,
            averaging_rotations=averaging_rotations,
            tail_fraction=tail_fraction,
        )
        if not files:
            continue
        try:
            stats = [
                _plane_statistics(
                    pv.read(sample_file), freestream_speed, rotor_radius, radial_edges
                )
                for sample_file in files
            ]
        except Exception as exc:
            print(f"  [WARNING] could not read wake samples for {tag}: {exc}")
            continue

        profiles = np.vstack([s[0] for s in stats])
        disc_means = np.array([s[1] for s in stats])
        ux = np.nanmean(profiles, axis=0)

        drift = _drift(disc_means)
        converged = drift <= drift_tolerance
        if converged:
            plot_label = rf"{label}, $\langle u_x\rangle_{{t,\theta}}$"
        else:
            # '%' is a comment character in the LaTeX text pipeline — escape it.
            plot_label = rf"{label}, not converged ({100 * drift:.0f}\% drift)"
            unconverged.append(f"{tag} ({drift:.1%})")

        ax.plot(
            radial_centres,
            ux,
            color=s_vpm["color"],
            alpha=alphas[i],
            marker=markers[i % len(markers)],
            markersize=s_vpm["markersize"],
            lw=s_vpm["linewidth"],
            ls="-" if converged else ":",
            label=plot_label,
        )
        print(
            f"  {tag}: {len(files)} snapshots, disc-mean drift {drift:.2%} "
            f"→ {'converged' if converged else 'NOT CONVERGED'}"
        )
        ux_min = min(ux_min, float(np.nanmin(ux)))
        ux_max = max(ux_max, float(np.nanmax(ux)))
        found = True

    if not found:
        print("  [WARNING] no rotor wake-plane samples found — run rotor_flow first.")
        return 0

    if unconverged:
        print(
            "  [WARNING] wake still in transit at: "
            + ", ".join(unconverged)
            + " — run longer before reading these profiles as physics."
        )

    # Momentum-theory references (text-annotated, not in legend)
    ax.axhline(1.0, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"])
    ax.axhline(1.0 - a_ref, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"])
    ax.axhline(
        1.0 - 2.0 * a_ref, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"]
    )

    x_text = 1.03
    ax.text(x_text, 1.03, r"$U_\infty$", color=s_ref["color"], va="center", ha="left")
    ax.text(x_text, 1.03 - a_ref, r"$(1-a)U_\infty$", color=s_ref["color"], va="center", ha="left")
    ax.text(
        x_text,
        1.03 - 2.0 * a_ref,
        r"$(1-2a)U_\infty$",
        color=s_ref["color"],
        va="center",
        ha="left",
    )

    ax.axvspan(
        HUB_RADIUS / rotor_radius,
        1.0,
        color=colors["background_light"],
        alpha=0.5,
        linewidth=0.0,
    )
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\langle u_x/U_\infty\rangle_{t,\theta}$")
    ax.set_xlim([0.0, radial_limit])
    # Keep the far-wake reference line in frame without leaving half the axis empty.
    ax.set_ylim([min(1.0 - 2.0 * a_ref - 0.08, ux_min - 0.05), max(1.1, ux_max + 0.05)])
    ax.set_title(rf"Azimuthally averaged wake deficit ($a={a_ref:.3f}$, {a_source})")
    ax.legend(loc="best")

    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    return plot_wake_planes(
        build_arg_parser("Rotor wake-plane validation vs momentum theory.").parse_args()
    )


if __name__ == "__main__":
    raise SystemExit(main())
