"""Shared plotting utilities for the quadcopter tutorial."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    if not csv_path.exists():
        raise SystemExit(f"No sampled flow integrals found in {samples_dir}")
    return pd.read_csv(csv_path)


def plot_vorticity_history(
    samples_dir: Path,
    figures_dir: Path,
    figure_format: str = "png",
) -> None:
    _theme.set_thesis_style()
    data = load_integrals(samples_dir)
    fig, ax = plt.subplots(figsize=_theme.figure_size("single"))
    ax.plot(data["time"], data["total_enstrophy"], "-o", color=_COLORS["VPMpurple"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Enstrophy [m$^3$/s$^2$]")
    ax.set_title("Quadcopter wake enstrophy history")
    figures_dir.mkdir(parents=True, exist_ok=True)
    _theme.save_fig(
        fig,
        figures_dir / "quadcopter_vorticity_history.png",
        figure_format=figure_format,
        bbox_inches=None,
    )


def rotor_inputs(metadata_path: Path | None = None):
    """Read geometry, motion and reference scales from this run's native records."""
    import json
    from types import SimpleNamespace
    import numpy as np

    metadata_path = metadata_path or CASE_DIR / "solution/vpm_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    config = metadata["configuration"]
    vlm = config["numerics"]["vlm"]
    first = vlm["surfaces"][0]
    geometry = _theme.read_vlm_surface(first, CASE_DIR / "assets")
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

    data = pd.read_csv(samples_dir / "vlm_surface_forces.csv")
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
    fig, rows = plt.subplots(
        4, 1, figsize=(12.5 * _theme.CM, 22 * _theme.CM), constrained_layout=True
    )
    axes = np.array([[rows[0], rows[2]], [rows[1], rows[3]]])
    for rotor, rows in data.groupby("rotor"):
        axes[0, 0].plot(rows.revolutions, rows.CT, label=rotor.replace("_", " "))
        axes[1, 0].plot(rows.revolutions, rows.CP)
    axes[0, 0].axhline(ct_bem, color="0.3", ls="--", label="Isolated-rotor BEM")
    axes[1, 0].axhline(cp_bem, color="0.3", ls="--")
    axes[0, 0].set(ylabel=r"Thrust coefficient $C_T$")
    axes[1, 0].set(xlabel="Nominal revolutions", ylabel=r"Power coefficient $C_P$")
    axes[0, 0].legend()
    tail = data[data.time > data.time.max() - 6 * p.period]
    start, end = tail.revolutions.agg(["min", "max"])
    ct_max = max(ct_bem, tail.CT.max()) * 1.15
    ct = np.linspace(0, ct_max, 150)
    advance = p.climb / (p.omega * p.radius)
    ideal = ct * 0.5 * (advance + np.sqrt(advance**2 + 2 * ct))
    axes[0, 1].plot(ct, ideal, color="0.3", ls=":", label="Ideal axial momentum")
    for rotor, rows in tail.groupby("rotor"):
        axes[0, 1].plot(rows.CT.mean(), rows.CP.mean(), "o", ms=4)
    axes[0, 1].plot(ct_bem, cp_bem, "x", color="black", label="Isolated-rotor BEM")
    axes[0, 1].set(
        xlabel=r"$C_T$",
        ylabel=r"$C_P$",
        title=f"Mean, rev {start:.1f}–{end:.1f}",
    )
    axes[0, 1].legend()
    total = data.groupby("step").agg(
        time=("time", "first"), thrust=("thrust", "sum"), power=("input_power", "sum")
    )
    axes[1, 1].plot(
        total.time / p.period, total.thrust, color=_COLORS["TUDcyan"], label="Total thrust"
    )
    power_axis = axes[1, 1].twinx()
    power_axis.plot(
        total.time / p.period,
        total.power,
        color=_COLORS["VPMpurple"],
        ls="--",
        label="Total shaft input",
    )
    axes[1, 1].set(xlabel="Nominal revolutions", ylabel="Thrust [N]")
    power_axis.set_ylabel("Power [W]", color=_COLORS["VPMpurple"])
    _theme.save_fig(
        fig,
        figures_dir / "quadcopter_performance.png",
        figure_format=figure_format,
        bbox_inches=None,
    )


def wake_windows(samples_dir, period, revolutions=6):
    """Read the final window of each published native velocity plane."""
    from defusedxml import ElementTree
    import numpy as np
    import pyvista as pv

    records = []
    for pvd in sorted(samples_dir.glob("sampled_zplane*.pvd")):
        frames = ElementTree.parse(pvd).findall(".//DataSet")
        times = np.array([float(frame.attrib["timestep"]) for frame in frames])
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"{pvd.name}: missing, duplicate or unordered frame times")
        selected = np.flatnonzero(times > times[-1] - revolutions * period)
        grids = [pv.read(pvd.parent / frames[index].attrib["file"]) for index in selected]
        points = np.asarray(grids[0].points)
        if not np.isfinite(points).all() or any(
            not np.array_equal(grid.points, points) for grid in grids
        ):
            raise ValueError(f"{pvd.name}: velocity plane grid changes within the window")
        velocity = np.array([np.asarray(grid["velocity"]) for grid in grids])
        if velocity.shape != (len(selected), len(points), 3) or not np.isfinite(velocity).all():
            raise ValueError(f"{pvd.name}: missing or non-finite velocity vectors")
        records.append(dict(name=pvd.stem, times=times[selected], points=points, velocity=velocity))
    return records


def plot_wake(samples_dir, figures_dir, figure_format="png"):
    import numpy as np
    from matplotlib.patches import Circle

    _theme.set_thesis_style()
    p = rotor_inputs()
    planes = wake_windows(samples_dir, p.period)
    if not planes:
        raise FileNotFoundError(f"No published quadcopter wake planes in {samples_dir}")
    fig, axes = plt.subplots(
        len(planes),
        1,
        figsize=(12.5 * _theme.CM, 14 * _theme.CM),
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    records = []
    for ax, plane in zip(axes.flat, planes, strict=True):
        start, end = plane["times"][[0, -1]]
        mean = -plane["velocity"][:, :, 2].mean(axis=0)
        records.append((ax, plane["points"], mean))
        ax.set_title(
            f"z = {plane['points'][0, 2]:g} m\nrev {start / p.period:.1f}–{end / p.period:.1f}"
        )
    low = min(record[2].min() for record in records)
    high = max(record[2].max() for record in records)
    centers = {
        tuple(surface["translation"])
        for surface in p.metadata["configuration"]["numerics"]["vlm"]["surfaces"]
    }
    for ax, points, mean in records:
        artist = ax.tricontourf(
            points[:, 0], points[:, 1], mean, levels=np.linspace(low, high, 24), cmap="viridis"
        )
        for x, y, _ in centers:
            ax.add_patch(Circle((x, y), p.radius, fill=False, color="white", lw=0.6, ls="--"))
        ax.set(xlabel="x [m]", aspect="equal")
    axes[0, 0].set_ylabel("y [m]")
    fig.colorbar(
        artist,
        ax=axes,
        label=r"Mean downward velocity $-u_z$ [m/s]",
        format="%.2f",
        ticks=np.linspace(low, high, 5),
    )
    _theme.save_fig(
        fig, figures_dir / "quadcopter_wake.png", figure_format=figure_format, bbox_inches=None
    )
