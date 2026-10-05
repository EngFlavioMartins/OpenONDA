"""Compare each native wake plane with disk and expanded far-wake limits."""

from __future__ import annotations

import numpy as np
from matplotlib.legend_handler import HandlerTuple

from openonda import plotting as theme
from openonda.plotting import centered_subplots_adjust
from openonda.results import read_pvd_frames
from openonda.rotor_theory import axial_induction_factor_from_thrust_coefficient
from openonda.validation import time_mean as _time_weighted_mean
from source.solvers.vpm.io.postprocess import surface_series

from ._common import (
    FIELD_STATIONARITY_COMPARISON_REVOLUTIONS,
    FIGURES_DIR,
    OPERATING_WINDOW_REVOLUTIONS,
    bem_reference,
    build_arg_parser,
    rotor_inputs,
    rotor_subplots,
    save_rotor_figure,
)
from .finite_distance_theory import build_system, induced_velocity


def relative_drift(values):
    """Compare two halves of a time window, normalized by its mean magnitude."""
    values = np.asarray(values)
    half = len(values) // 2
    return abs(values[:half].mean() - values[half:].mean()) / max(abs(values.mean()), 1e-12)


def _annulus_profile(coordinates, edges, values):
    """Mean of ``values`` inside each annular bin, indexed by ``coordinates``.

    ``coordinates`` are normalized radial positions against ``edges`` (radial
    bounds in the same units).  Bins with no samples receive NaN:
    the surrounding consumers mask those annuli before combining plots.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    edges = np.asarray(edges, dtype=float)
    values = np.asarray(values, dtype=float)
    radius = 0.5 * (edges[1:] + edges[:-1])
    indices = np.digitize(coordinates, edges) - 1
    valid = (indices >= 0) & (indices < len(radius))
    counts = np.bincount(indices[valid], minlength=len(radius))
    sums = np.bincount(indices[valid], weights=values[valid], minlength=len(radius))
    return (radius, np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0))


def _plane_statistics(grid, *, freestream_speed, rotor_radius, radial_edges):
    """Annulus-averaged axial velocity of one published native plane.

    ``grid`` is the PyVista plane as written by the surface sampler: point
    coordinates with the axial station on axis 0 and point data ``velocity``.
    Returns the mean axial velocity (normalized by ``freestream_speed``) per
    radial bin and its arithmetic mean over the whole plane.
    """
    points = np.asarray(grid.points, dtype=float)
    velocity = np.asarray(grid.point_data["velocity"], dtype=float)
    axial = velocity[:, 0] / freestream_speed
    radial = np.linalg.norm(points[:, 1:], axis=1) / rotor_radius
    _, profile = _annulus_profile(radial, np.asarray(radial_edges, dtype=float), axial)
    disc_mean = float(np.mean(axial[np.isfinite(axial)]))
    return (profile, disc_mean)


def induced_field_drift(
    times,
    velocity,
    background,
    period,
    comparison_rotations=FIELD_STATIONARITY_COMPARISON_REVOLUTIONS,
    *,
    window_start=None,
    window_end=None,
):
    """Compare first-two versus final-three whole-revolution means.

    For the five-revolution native window, the comparison covers all included
    time: two complete revolutions at the front versus three at the end. When
    explicit boundaries are supplied, every boundary must be bracketed by
    recorded native frames; no extrapolation is allowed.
    """
    comparison_rotations = int(comparison_rotations)
    if window_start is None:
        window_end = float(times[-1]) if window_end is None else float(window_end)
        window_start = window_end - comparison_rotations * period
    else:
        window_start, window_end = (float(window_start), float(window_end))
    early_rotations = comparison_rotations // 2
    split = window_start + early_rotations * period
    early = _time_weighted_mean(times, velocity, window_start, split)
    late = _time_weighted_mean(times, velocity, split, window_end)
    scale = np.linalg.norm(0.5 * (early + late) - background)
    if scale <= 1e-12:
        return (np.nan, comparison_rotations)
    return (np.linalg.norm(late - early) / scale, comparison_rotations)


def native_plane_windows(p, rotations=OPERATING_WINDOW_REVOLUTIONS):
    """Average the native velocity over the selected number of rotor revolutions."""
    items = p.metadata["configuration"]["samplers"]["items"]
    names = [item["file_name"] for item in items if item["type"] == "SurfaceSampler"]
    collections = {name: read_pvd_frames(p.samples_dir / f"{name}.pvd") for name in names}
    end = min((frames[-1][0] for frames in collections.values()))
    end = min(end, p.metadata["state"]["time"])
    start = end - rotations * p.rotation_period
    background = np.asarray(p.metadata["configuration"]["numerics"]["freestream_velocity"])
    records = []
    for name, _frames in collections.items():
        times, points, velocity = surface_series(p.samples_dir / f"{name}.pvd")
        first = max(0, np.searchsorted(times, start, side="right") - 1)
        last = np.searchsorted(times, end, side="left")
        clock, velocity = times[first : last + 1], velocity[first : last + 1]
        drift, compared = induced_field_drift(
            clock, velocity, background, p.rotation_period, window_start=start, window_end=end
        )
        records.append(
            {
                "name": name,
                "points": points,
                "times": clock,
                "velocity": velocity,
                "window_start": start,
                "window_end": end,
                "window_mean_velocity": _time_weighted_mean(clock, velocity, start, end),
                "induced_field_drift": drift,
                "compared_rotations": compared,
            }
        )
    return records


def plane_profiles(p, rotations=OPERATING_WINDOW_REVOLUTIONS):
    """Time/azimuth averages of the solver's published native velocity planes."""
    records = []
    for plane in native_plane_windows(p, rotations):
        points = plane["points"]
        extent = min(-points[:, 1:].min(axis=0).max(), points[:, 1:].max(axis=0).min())
        edges = np.linspace(0, extent / p.rotor_radius, 49)
        r = np.linalg.norm(points[:, 1:], axis=1) / p.rotor_radius
        mean_velocity = plane["window_mean_velocity"][:, 0] / p.freestream_speed
        radius, mean = _annulus_profile(r, edges, mean_velocity)
        records.append(
            {
                "name": plane["name"],
                "x": points[0, 0],
                "radius": radius,
                "mean": mean,
                "times": plane["times"],
                "window_start": plane["window_start"],
                "window_end": plane["window_end"],
                "induced_field_drift": plane["induced_field_drift"],
                "compared_rotations": plane["compared_rotations"],
            }
        )
    return sorted(records, key=lambda row: row["x"])


def _binned_mean(values, indices, valid, counts, size):
    sums = np.bincount(indices[valid], weights=np.asarray(values)[valid], minlength=size)
    return np.divide(sums, counts, out=np.full(size, np.nan), where=counts > 0)


def finite_distance_profiles(p, rotations=OPERATING_WINDOW_REVOLUTIONS):
    """Return native axial/azimuthal profiles beside the right-cylinder theory.

    The native velocity is time-averaged at each fixed plane point before
    annular binning.  The reference uses the matched BEM circulation table and
    the analytical finite-distance cylinder functions; it is deliberately not
    fitted to the native samples.
    """
    bem = bem_reference()
    system = build_system(
        bem,
        number_of_blades=p.n_blades,
        freestream_speed=p.freestream_speed,
        angular_velocity=p.angular_velocity,
        hub_radius=p.hub_radius,
        rotor_radius=p.rotor_radius,
    )
    configuration = p.metadata["configuration"]
    signed_angular_velocity = float(
        configuration["numerics"]["vlm"]["surfaces"][0]["kinematics"]["angular_speed"]
    )
    rotation_sign = np.sign(signed_angular_velocity) or 1.0
    records = []
    for plane in native_plane_windows(p, rotations):
        points = plane["points"]
        if points[0, 0] <= 0:
            continue
        extent = min(-points[:, 1:].min(axis=0).max(), points[:, 1:].max(axis=0).min())
        edges = np.linspace(0.0, extent / p.rotor_radius, 49)
        radius = 0.5 * (edges[1:] + edges[:-1])
        point_radius = np.linalg.norm(points[:, 1:], axis=1)
        radial_coordinate = point_radius / p.rotor_radius
        indices = np.digitize(radial_coordinate, edges) - 1
        valid = (indices >= 0) & (indices < len(radius)) & (point_radius > 1e-10)
        counts = np.bincount(indices[valid], minlength=len(radius))
        mean_velocity = plane["window_mean_velocity"]
        axial = mean_velocity[:, 0]
        tangential = -points[:, 2] * mean_velocity[:, 1] + points[:, 1] * mean_velocity[:, 2]
        tangential = np.divide(
            tangential,
            point_radius,
            out=np.full_like(tangential, np.nan, dtype=float),
            where=point_radius > 1e-10,
        )
        actual_axial_velocity = _binned_mean(axial, indices, valid, counts, len(radius))
        actual_axial = 1.0 - actual_axial_velocity / p.freestream_speed
        actual_tangential = -_binned_mean(tangential, indices, valid, counts, len(radius)) / (
            signed_angular_velocity * radius * p.rotor_radius
        )
        reference_axial = np.full_like(radius, np.nan, dtype=float)
        reference_axial_velocity = np.full_like(radius, np.nan, dtype=float)
        reference_tangential = np.full_like(radius, np.nan, dtype=float)
        for index, normalized_radius in enumerate(radius):
            if not np.isfinite(actual_axial[index]):
                continue
            prediction = induced_velocity(
                normalized_radius * p.rotor_radius,
                points[0, 0],
                system,
                freestream_speed=p.freestream_speed,
                angular_velocity=p.angular_velocity,
            )
            reference_axial_velocity[index] = p.freestream_speed + prediction["axial_velocity"]
            reference_axial[index] = prediction["axial_induction"]
            reference_tangential[index] = prediction["tangential_induction"]
        reference_tangential_velocity = (
            -reference_tangential * p.angular_velocity * radius * p.rotor_radius * rotation_sign
        )
        actual_tangential_velocity = _binned_mean(tangential, indices, valid, counts, len(radius))
        records.append(
            {
                "name": plane["name"],
                "x": points[0, 0],
                "radius": radius,
                "actual_axial_induction": actual_axial,
                "reference_axial_induction": reference_axial,
                "actual_axial_velocity": actual_axial_velocity,
                "reference_axial_velocity": reference_axial_velocity,
                "actual_tangential_induction": actual_tangential,
                "reference_tangential_induction": reference_tangential,
                "actual_tangential_velocity": actual_tangential_velocity,
                "reference_tangential_velocity": reference_tangential_velocity,
                "induced_field_drift": plane["induced_field_drift"],
                "compared_rotations": plane["compared_rotations"],
            }
        )
    return sorted(records, key=lambda row: row["x"])


def main():
    args = build_arg_parser(__doc__).parse_args()
    p = rotor_inputs()
    profiles = plane_profiles(p)
    disk = [row for row in profiles if np.isclose(row["x"], 0.0, atol=1e-09)]
    if disk:
        profiles = [disk[0], max(profiles, key=lambda row: row["x"])]
    else:
        print("No disk-plane samples in this record; showing the recorded downstream stations.")
    ct = bem_reference().attrs["thrust_coefficient"]
    induction = axial_induction_factor_from_thrust_coefficient(ct)
    wake_radius = np.sqrt((1 - induction) / (1 - 2 * induction))
    extent = max((row["radius"].max() for row in profiles))
    fig, axes = rotor_subplots(len(profiles), height_cm=11, sharex=True)
    for axis, row in zip(axes, profiles, strict=True):
        axis.plot(
            row["mean"],
            row["radius"],
            color=theme.COLORS["vpm"],
            ls="-",
            marker="o",
            markevery=4,
            label="VLM+VPM",
            zorder=4,
        )
        axis.plot(
            [1 - induction, 1 - induction, 1, 1],
            [0, 1, 1, max(extent, 1)],
            color=theme.COLORS["reference"],
            ls=":",
            label="$1-a$",
        )
        axis.plot(
            [1 - 2 * induction, 1 - 2 * induction, 1, 1],
            [0, wake_radius, wake_radius, max(extent, wake_radius)],
            color=theme.COLORS["reference"],
            ls="--",
            label="$1-2a$",
        )
        axis.text(
            0.04,
            0.94,
            f"$x/D={row['x'] / p.station_diameter:g}$",
            transform=axis.transAxes,
            va="top",
        )
        axis.set_ylabel("Radius, $r/R$")
        print(f"{row['name']}: induced-field drift {row['induced_field_drift']:.2%}")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        [handles[0], (handles[1], handles[2])],
        ["VLM+VPM", "$1-a$, $1-2a$"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        borderaxespad=0,
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="0.8",
        facecolor="white",
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.6,
    )
    low = min(0, *(axis.get_xlim()[0] for axis in axes))
    high = max(1.08, *(axis.get_xlim()[1] for axis in axes))
    for axis in axes:
        axis.set_xlim(low, high)
        axis.set_ylim(0, extent)
    axes[-1].set_xlabel("$u_x/U_\\infty$")
    print(f"BEM disk reference: CT={ct:.6f}, a={induction:.6f}, Rw/R={wake_radius:.6f}")
    centered_subplots_adjust(fig, outer=0.113, bottom=0.22, top=0.99, hspace=0.1)
    save_rotor_figure(
        fig, FIGURES_DIR / "rotor_wake_planes.png", figure_format=args.format, dpi=args.dpi
    )
    return 0


if __name__ == "__main__":
    main()
