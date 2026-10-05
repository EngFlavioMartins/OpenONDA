"""Shared plotting utilities for the quadcopter tutorial."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from openonda.results import read_csv_table, read_json

CASE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = CASE_DIR / "samples" / "quadcopter"
FIGURES_DIR = CASE_DIR / "figures"


def _load_theme():
    from openonda import plotting as theme

    return theme


_theme = _load_theme()
_COLORS = _theme.COLORS


def load_integrals(samples_dir: Path) -> pd.DataFrame:
    csv_path = samples_dir / "flow_integrals.csv"
    return pd.DataFrame(read_csv_table(csv_path))


def plot_vorticity_history(
    samples_dir: Path, figures_dir: Path, figure_format: str = "png"
) -> None:
    _theme.set_thesis_style()
    data = load_integrals(samples_dir)
    fig, ax = plt.subplots(figsize=_theme.figure_size("single"))
    ax.plot(data["time"], data["total_enstrophy"], "-o", color=_COLORS["vpm"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Enstrophy [m$^3$/s$^2$]")
    _theme.centered_subplots_adjust(fig, outer=0.104, bottom=0.22, top=0.923)
    _theme.save_fig(
        fig,
        figures_dir / "quadcopter_vorticity_history.png",
        figure_format=figure_format,
        bbox_inches=None,
    )


def rotor_inputs(metadata_path: Path | None = None):
    """Read geometry, motion and reference scales from this run's native records."""
    from types import SimpleNamespace

    import numpy as np

    metadata_path = metadata_path or CASE_DIR / "solution/vpm_metadata.json"
    metadata = read_json(metadata_path)
    config = metadata["configuration"]
    vlm = config["numerics"]["vlm"]
    first = vlm["surfaces"][0]
    geometry = _theme.read_vlm_surface(first)
    segment = geometry["wings"][0]["segments"][0]
    a, b, c, d = (np.asarray(segment["vertex_position"][key]) for key in "abcd")
    root, tip = sorted(
        ((a, d), (b, c)), key=lambda ends: np.linalg.norm((0.75 * ends[0] + 0.25 * ends[1])[:2])
    )
    inner, outer = (np.linalg.norm((0.75 * le + 0.25 * te)[:2]) for le, te in (root, tip))
    edges = np.linspace(inner, outer, 65)
    radius = 0.5 * (edges[1:] + edges[:-1])
    eta = (radius - inner) / (outer - inner)
    chord_vectors = (1 - eta[:, None]) * (root[1] - root[0]) + eta[:, None] * (tip[1] - tip[0])
    chord = np.linalg.norm(chord_vectors, axis=1)
    pitch = np.arcsin(-chord_vectors[:, 2] / chord)
    omega = abs(first["kinematics"]["angular_speed"])
    groups = {surface["group_id"] for surface in vlm["surfaces"]}
    return SimpleNamespace(
        metadata=metadata,
        density=vlm["density"],
        radius=outer,
        hub_radius=inner,
        radial_position=radius,
        radial_widths=np.diff(edges),
        chord=chord,
        pitch=pitch,
        omega=omega,
        period=2 * np.pi / omega,
        n_rotors=len(groups),
        n_blades=len(vlm["surfaces"]) / len(groups),
        climb=np.linalg.norm(config["numerics"]["freestream_velocity"]),
        samples_dir=CASE_DIR / "samples" / config["samplers"]["directory"],
    )


def bem_reference(p):
    from openonda.rotor_theory import solve_blade_element_momentum

    return solve_blade_element_momentum(
        p.radial_position,
        p.chord,
        p.pitch,
        p.n_blades,
        p.radius,
        p.climb,
        p.omega,
        hub_radius=p.hub_radius,
        radial_widths=p.radial_widths,
        density=p.density,
        mode="propeller",
    )


def performance(samples_dir, p):
    """Positive propeller thrust and input power; torques are never summed first."""
    import numpy as np

    data = pd.DataFrame(read_csv_table(samples_dir / "vlm_surface_forces.csv"))
    data["rotor"] = data.surface.str.rsplit("_blade_", n=1).str[0]
    grouped = (
        data.groupby(["rotor", "step"], sort=True)
        .agg(
            time=("time", "first"),
            thrust=("force_z", "sum"),
            fluid_power=("rotational_power", "sum"),
        )
        .reset_index()
    )
    reference = p.density * np.pi * p.radius**2 * (p.omega * p.radius) ** 2
    grouped["input_power"] = -grouped.fluid_power
    grouped["CT"] = grouped.thrust / reference
    grouped["CP"] = grouped.input_power / (reference * p.omega * p.radius)
    grouped["revolutions"] = grouped.time / p.period
    return grouped


def plot_performance(samples_dir, figures_dir, figure_format="png", metadata_path=None):
    import numpy as np

    _theme.set_thesis_style()
    p = rotor_inputs(metadata_path)
    data = performance(samples_dir, p)
    bem = bem_reference(p)
    reference = p.density * np.pi * p.radius**2 * (p.omega * p.radius) ** 2
    ct_bem = bem.attrs["thrust"] / reference
    cp_bem = bem.attrs["power"] / (reference * p.omega * p.radius)
    fig, rows = plt.subplots(4, 1, figsize=(12.5 * _theme.CM, 16 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.21, bottom=0.09, top=0.955, hspace=0.55)
    axes = np.array([[rows[0], rows[2]], [rows[1], rows[3]]])
    for index, (rotor, rows) in enumerate(data.groupby("rotor")):
        style = {
            "color": _theme.COLOR_CYCLE[index],
            "marker": ("o", "s", "D", "^")[index],
            "markevery": 60,
            "ms": 2.5,
        }
        axes[0, 0].plot(rows.revolutions, rows.CT, label=rotor.replace("_", " "), **style)
        axes[1, 0].plot(rows.revolutions, rows.CP, **style)
    axes[0, 0].axhline(ct_bem, color=_COLORS["reference"], ls="--", label="BEM")
    axes[1, 0].axhline(cp_bem, color=_COLORS["reference"], ls="--")
    axes[0, 0].set(ylabel="$C_T$")
    axes[1, 0].set(xlabel="Nominal revolutions", ylabel="$C_P$")
    axes[0, 0].legend(ncol=2, loc="upper right", handlelength=1.2, columnspacing=0.7)
    tail = data[data.time > data.time.max() - 6 * p.period]
    start, end = tail.revolutions.agg(["min", "max"])
    print(f"Performance mean: revolutions {start:.1f}–{end:.1f}")
    ct_max = max(ct_bem, tail.CT.max()) * 1.15
    ct = np.linspace(0, ct_max, 150)
    advance = p.climb / (p.omega * p.radius)
    ideal = ct * 0.5 * (advance + np.sqrt(advance**2 + 2 * ct))
    axes[0, 1].plot(ct, ideal, color=_COLORS["reference"], ls=":", label="Axial momentum")
    for _rotor, rows in tail.groupby("rotor"):
        axes[0, 1].plot(rows.CT.mean(), rows.CP.mean(), "o", ms=4)
    axes[0, 1].plot(ct_bem, cp_bem, "x", color=_COLORS["reference"], label="BEM")
    axes[0, 1].set(xlabel="$C_T$", ylabel="$C_P$")
    axes[0, 1].legend()
    total = data.groupby("step").agg(
        time=("time", "first"), thrust=("thrust", "sum"), power=("input_power", "sum")
    )
    axes[1, 1].plot(
        total.time / p.period, total.thrust, color=_COLORS["teal"], label="Total thrust"
    )
    power_axis = axes[1, 1].twinx()
    power_axis.plot(
        total.time / p.period, total.power, color=_COLORS["vpm"], ls="-", label="Total shaft input"
    )
    axes[1, 1].set(xlabel="Nominal revolutions", ylabel="Thrust [N]")
    power_axis.set_ylabel("Power [W]", color=_COLORS["vpm"])
    _theme.centered_subplots_adjust(fig, outer=0.092, top=0.966)
    _theme.export_figure(
        fig, figures_dir / "quadcopter_performance.png", figure_format=figure_format
    )


def wake_windows(samples_dir, period, revolutions=6):
    """Read the final window of each published native velocity plane."""
    from source.solvers.vpm.io.postprocess import surface_series

    records = []
    for pvd in sorted(samples_dir.glob("sampled_zplane*.pvd")):
        times, points, velocity = surface_series(pvd)
        selected = times > times[-1] - revolutions * period
        records.append(
            {
                "name": pvd.stem,
                "times": times[selected],
                "points": points,
                "velocity": velocity[selected],
            }
        )
    return records


def plot_wake(samples_dir, figures_dir, figure_format="png"):
    import numpy as np
    from matplotlib.patches import Circle

    _theme.set_thesis_style()
    p = rotor_inputs()
    planes = wake_windows(samples_dir, p.period)
    fig, axes = plt.subplots(
        1, len(planes), figsize=(12.5 * _theme.CM, 7.4 * _theme.CM), sharey=True, squeeze=False
    )
    _theme.centered_subplots_adjust(fig, outer=0.135, bottom=0.375, top=0.922, wspace=0.33)
    records = []
    for ax, plane in zip(axes.flat, planes, strict=True):
        start, end = plane["times"][[0, -1]]
        mean = -plane["velocity"][:, :, 2].mean(axis=0)
        records.append((ax, plane["points"], mean))
    low = min((record[2].min() for record in records))
    high = max((record[2].max() for record in records))
    centers = {
        tuple(surface["translation"])
        for surface in p.metadata["configuration"]["numerics"]["vlm"]["surfaces"]
    }
    for ax, points, mean in records:
        artist = ax.tricontourf(
            points[:, 0],
            points[:, 1],
            mean,
            levels=np.linspace(low, high, 24),
            cmap=_theme.COLORMAPS["velocity"],
        )
        for x, y, _ in centers:
            ax.add_patch(Circle((x, y), p.radius, fill=False, color="white", lw=0.6, ls="--"))
        ax.set(xlabel="$x$ [m]", aspect="equal", title=f"$z={points[0, 2]:.2g}$ m")
        ax.locator_params(axis="both", nbins=3)
    axes[0, 0].set_ylabel("y [m]")
    outer = 0.135
    cax = fig.add_axes([outer, 0.175, 1 - 2 * outer, 0.035])
    fig.colorbar(
        artist,
        cax=cax,
        orientation="horizontal",
        label="$-\\overline{u_z}$ [m/s]",
        format="%.2g",
        ticks=np.linspace(low, high, 3),
    )
    _theme.export_figure(fig, figures_dir / "quadcopter_wake.png", figure_format=figure_format)
