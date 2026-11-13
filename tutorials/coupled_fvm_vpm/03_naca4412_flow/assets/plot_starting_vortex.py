#!/usr/bin/env python3
"""Passage of the starting vortex and the tip vortices through the cross-flow planes x = 4 and 8 m.

At every saved state of the coupled run (solution/coupled/coupled_*) and of the FVM-only reference
(reference_flow/solution/fvm/fvm_*), the vorticity on the two planes, on a 0.04 m grid:
- coupled: the particle-represented vorticity, sum of alpha_p zeta(r / sigma_p) / sigma_p^3 with
  the run's core function; both planes lie beyond the FVM box, in the particle wake;
- reference: the curl of an affine fit of the FVM cell velocity on the 20 nearest cells of each
  grid point (cells of 0.08 m at x = 4 m, 0.16 m at x = 8 m).

Each state is cached in samples/starting_vortex/, so a rerun only adds new states. Writes PNG
figures to figures/starting_vortex/:
- x<plane>_omega_z, x<plane>_omega_x: the maps at the five common states around the passage of
  the starting vortex, coupled above reference;
- histories: through each plane, the integrated spanwise vorticity (the starting vortex) and the
  streamwise circulation of each half-plane (the tip vortex of each wing half), against time.

    python -m assets.plot_starting_vortex [--coupled-case DIR]

``--coupled-case`` reads the coupled run from another copy of this case (default: this case).
"""

import argparse
import glob
import json
from pathlib import Path

import h5py
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from openonda.plotting import (
    CM,
    COLORMAPS,
    COLORS,
    REFERENCE_STYLE,
    export_figure,
    set_thesis_style,
)
from openonda.vpm.kernels.registry import kernel_definition

CASE_DIR = Path(__file__).absolute().parents[1]
FIGURES = CASE_DIR / "figures/starting_vortex"
CACHE = CASE_DIR / "samples/starting_vortex"
REFERENCE_SOLUTION = CASE_DIR / "reference_flow/solution/fvm"
REFERENCE_MESH = CASE_DIR / "reference_flow/assets/reference_naca4412_flow_mesh.npz"

PLANES = (4.0, 8.0)  # m, x of the cross-flow planes
CROSS_BOUNDS = (-2.4, 2.4, -4.0, 4.0)  # m, y and z bounds (VPM_SLICE_CROSS of setup.py)
SPACING = 0.04  # m, plane grid
FIT_CELLS = 20  # reference cells of each affine velocity fit
KERNEL_SUPPORT = 5.0  # core radii within which a particle contributes
SHOWN_STATES = 5  # maps around the passage of the starting vortex

Y = np.linspace(
    CROSS_BOUNDS[0], CROSS_BOUNDS[1], round((CROSS_BOUNDS[1] - CROSS_BOUNDS[0]) / SPACING) + 1
)
Z = np.linspace(
    CROSS_BOUNDS[2], CROSS_BOUNDS[3], round((CROSS_BOUNDS[3] - CROSS_BOUNDS[2]) / SPACING) + 1
)
AREA = SPACING**2  # m^2 per grid point


def plane_points(x):
    """Grid points of the plane x, shape (len(Y) * len(Z), 3), y the slow index."""
    y, z = np.meshgrid(Y, Z, indexing="ij")
    return np.column_stack([np.full(y.size, x), y.ravel(), z.ravel()])


def stored(npz, name):
    """An array of a backup file, decoded from its storage layout when it has one."""
    if "storage_layout" in npz.files:
        dtype, shape = json.loads(str(npz["storage_layout"]))[name]
        return np.ascontiguousarray(npz[name].T).view(np.dtype(dtype)).reshape(shape)
    return npz[name]


# ------------------------------------------------------------------------------------------------
# Vorticity on the planes
# ------------------------------------------------------------------------------------------------


def particle_vorticity(backup, zeta):
    """Particle-represented vorticity on each plane, shape (3, len(Y), len(Z)) per plane."""
    particles = h5py.File(glob.glob(f"{backup}/vpm_*.h5")[0], "r")["particles"]
    position = particles["position"][:].astype(np.float64)
    strength = particles["vortex_strength"][:].astype(np.float64)
    radius = particles["core_radius"][:].astype(np.float64)
    reach = KERNEL_SUPPORT * radius.max()
    fields = []
    for x in PLANES:
        near = np.abs(position[:, 0] - x) < reach
        points = plane_points(x)
        omega = np.zeros_like(points)
        if np.any(near):
            pairs = cKDTree(points).sparse_distance_matrix(
                cKDTree(position[near]), reach, output_type="coo_matrix"
            )
            sigma = radius[near][pairs.col]
            weight = zeta(pairs.data / sigma) / sigma**3
            for axis in range(3):
                omega[:, axis] = np.bincount(
                    pairs.row, weight * strength[near][pairs.col, axis], minlength=len(points)
                )
        fields.append(omega.T.reshape(3, len(Y), len(Z)))
    return fields


class ReferenceMesh:
    """Centres of the reference cells near the planes, and their fit stencils."""

    def __init__(self):
        mesh = np.load(REFERENCE_MESH)
        offsets, vertices = mesh["face_offsets"], mesh["face_vertices"]
        face_centre = (
            np.add.reduceat(mesh["vertex_position"][vertices], offsets[:-1])
            / np.diff(offsets)[:, None]
        )
        cell_faces, cell_offset = mesh["cell_face_indices"], mesh["cell_face_offset"]
        centre = (
            np.add.reduceat(face_centre[cell_faces], cell_offset[:-1])
            / np.diff(cell_offset)[:, None]
        )
        self.n_cells = len(centre)
        self.stencils = []
        for x in PLANES:
            box = (
                (np.abs(centre[:, 0] - x) < 0.6)
                & (np.abs(centre[:, 1]) < CROSS_BOUNDS[1] + 0.6)
                & (np.abs(centre[:, 2]) < CROSS_BOUNDS[3] + 0.6)
            )
            cells = np.flatnonzero(box)
            points = plane_points(x)
            _, nearest = cKDTree(centre[cells]).query(points, k=FIT_CELLS)
            offset = centre[cells][nearest] - points[:, None, :]
            design = np.concatenate([np.ones((*nearest.shape, 1)), offset], axis=2)
            weight = 1.0 / (np.sum(offset**2, axis=2) + (0.1 * SPACING) ** 2)
            normal = np.einsum("nki,nk,nkj->nij", design, weight, design)
            # Rows 1-3 of the solved fit are the velocity gradient; each grid point keeps the
            # weighted least-squares operator of its stencil.
            operator = np.linalg.solve(normal, np.einsum("nki,nk->nik", design, weight))
            self.stencils.append((cells[nearest], operator[:, 1:, :]))


def fvm_vorticity(backup, mesh):
    """Vorticity of the affine fit of the FVM velocity on each plane, as particle_vorticity."""
    velocity = np.zeros((mesh.n_cells, 3))
    for rank in glob.glob(f"{backup}/rank-*.npz"):
        data = np.load(rank)
        cells = stored(data, "global_cell_id")
        velocity[cells] = stored(data, "velocity")[: len(cells)]
    fields = []
    for cells, operator in mesh.stencils:
        gradient = np.einsum("njk,nki->nji", operator, velocity[cells])  # d u_i / d x_j
        omega = np.column_stack(
            [
                gradient[:, 1, 2] - gradient[:, 2, 1],
                gradient[:, 2, 0] - gradient[:, 0, 2],
                gradient[:, 0, 1] - gradient[:, 1, 0],
            ]
        )
        fields.append(omega.T.reshape(3, len(Y), len(Z)))
    return fields


def reference_time(backup):
    return float(np.load(sorted(glob.glob(f"{backup}/rank-*.npz"))[0])["time"])


def coupled_time(backup):
    return float(json.loads(Path(backup, "checkpoint_info.json").read_text())["time"])


def cached_states(run, backups, time_of, compute):
    """(time, [field per plane]) of every backup, from the cache or computed and cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    states = []
    for backup in backups:
        path = CACHE / f"{run}_{Path(backup).name}.npz"
        source = str(Path(backup).resolve())
        if path.exists():
            data = np.load(path)
            if str(data["source"]) == source:
                states.append((float(data["time"]), [data[f"x{x:g}"] for x in PLANES]))
                continue
        time = time_of(backup)
        fields = compute(backup)
        np.savez(path, source=source, time=time, **{f"x{x:g}": f for x, f in zip(PLANES, fields)})
        print(f"{run}: t = {time:.2f} s cached")
        states.append((time, fields))
    return sorted(states, key=lambda state: state[0])


# ------------------------------------------------------------------------------------------------
# Measures and figures
# ------------------------------------------------------------------------------------------------


def measures(states):
    """Per plane: times, integrated omega_z, peak omega_z and the circulation of each half."""
    times = np.array([t for t, _ in states])
    out = {}
    for index, x in enumerate(PLANES):
        omega = np.stack([fields[index] for _, fields in states])
        upper = Z > 0.0
        out[x] = {
            "time": times,
            "omega_z_integral": omega[:, 2].sum(axis=(1, 2)) * AREA,
            "omega_z_peak": np.array([w.flat[np.argmax(np.abs(w))] for w in omega[:, 2]]),
            "circulation_upper": omega[:, 0][:, :, upper].sum(axis=(1, 2)) * AREA,
            "circulation_lower": omega[:, 0][:, :, ~upper].sum(axis=(1, 2)) * AREA,
        }
    return out


def passage_states(reference, coupled, index):
    """The common times around the largest |integrated omega_z| of the reference on a plane."""
    common = sorted(
        set(np.round([t for t, _ in reference], 6)) & set(np.round([t for t, _ in coupled], 6))
    )
    if not common:
        return []
    values = {round(t, 6): abs(f[index][2].sum()) for t, f in reference}
    peak = common.index(max(common, key=lambda t: values[t]))
    start = min(max(0, peak - SHOWN_STATES // 2), max(0, len(common) - SHOWN_STATES))
    return common[start : start + SHOWN_STATES]


def map_figure(name, component, x, index, shown, reference, coupled):
    by_time = {
        "coupled": {round(t, 6): f[index][component] for t, f in coupled},
        "reference": {round(t, 6): f[index][component] for t, f in reference},
    }
    limit = max(np.abs(by_time[run][t]).max() for run in by_time for t in shown) or 1.0
    width, left, gap = 12.5, 1.3, 0.3
    panel_w = (width - 2 * left - gap) / 2
    panel_h = panel_w * (Y[-1] - Y[0]) / (Z[-1] - Z[0])
    top, row_gap, bar = 0.9, 0.35, 2.8
    height = top + len(shown) * panel_h + (len(shown) - 1) * row_gap + bar
    fig = plt.figure(figsize=(width * CM, height * CM))
    symbol = "xyz"[component]
    axes = []
    for row, t in enumerate(shown):
        for column, run in enumerate(("coupled", "reference")):
            x0 = left + column * (panel_w + gap)
            y0 = height - top - (row + 1) * panel_h - row * row_gap
            ax = fig.add_axes((x0 / width, y0 / height, panel_w / width, panel_h / height))
            image = ax.pcolormesh(
                Z,
                Y,
                by_time[run][t],
                cmap=COLORMAPS["vorticity"],
                vmin=-limit,
                vmax=limit,
                shading="nearest",
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set_xticks([-2, 0, 2] if row == len(shown) - 1 else [])
            ax.set_yticks([-2, 0, 2] if column == 0 else [])
            if row == 0:
                ax.set_title("coupled" if run == "coupled" else "FVM only")
            if column == 1:
                ax.text(
                    1.03,
                    0.5,
                    f"$t={t:.2f}$ s",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="left",
                )
            axes.append(ax)
    axes[-2].set_xlabel("$z$ (m)")
    axes[-1].set_xlabel("$z$ (m)")
    fig.text(0.25 / width, 0.5, "$y$ (m)", rotation=90, va="center")
    bar_axis = fig.add_axes((left / width, 1.3 / height, (width - 2 * left) / width, 0.18 / height))
    colorbar = fig.colorbar(image, cax=bar_axis, orientation="horizontal")
    colorbar.set_label(rf"$\omega_{symbol}$ (1/s) at $x={x:g}$ m")
    export_figure(fig, FIGURES / name)


def history_figure(coupled, reference):
    width, left, gap, height = 12.5, 1.6, 1.5, 11.8
    panel_w = (width - 2 * left - gap) / 2
    panel_h = (height - 1.3 - 2.2 - 1.0) / 2
    fig = plt.figure(figsize=(width * CM, height * CM))
    rows = (
        ("omega_z_integral", r"$\int\omega_z\,dA$ (m$^2$/s)"),
        ("circulation", r"$\Gamma_x$ (m$^2$/s)"),
    )
    for column, x in enumerate(PLANES):
        for row, (key, label) in enumerate(rows):
            x0 = left + column * (panel_w + gap)
            y0 = 1.3 + (1 - row) * (panel_h + 1.0)
            ax = fig.add_axes((x0 / width, y0 / height, panel_w / width, panel_h / height))
            for data, run, style in (
                (coupled[x], "coupled", {"color": COLORS["hybrid"]}),
                (reference[x], "FVM only", REFERENCE_STYLE),
            ):
                if key == "circulation":
                    ax.plot(
                        data["time"],
                        data["circulation_upper"],
                        marker=".",
                        **style,
                        label=f"{run}, $z>0$",
                    )
                    ax.plot(
                        data["time"],
                        data["circulation_lower"],
                        marker="x",
                        **style,
                        label=f"{run}, $z<0$",
                    )
                else:
                    ax.plot(data["time"], data[key], marker=".", **style, label=run)
            ax.axhline(0.0, color=COLORS["light_gray"], linewidth=0.5)
            ax.set_xlim(0.0, max(reference[x]["time"].max(), coupled[x]["time"].max()))
            if row == 0:
                ax.set_title(f"$x={x:g}$ m")
            if row == 1:
                ax.set_xlabel("$t$ (s)")
            if column == 0:
                ax.set_ylabel(label)
            if row == 0 and column == 0:
                handles = ax.get_lines()[:2]
    # In the circulation panels the marker tells the half-plane apart.
    markers = [Line2D([], [], color=COLORS["text"], marker=m, linestyle="none") for m in (".", "x")]
    fig.legend(
        [*handles, *markers],
        ("coupled", "FVM only", "$\\Gamma_x$, $z>0$", "$\\Gamma_x$, $z<0$"),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0 - 0.1 / height),
    )
    export_figure(fig, FIGURES / "histories")


def report(coupled, reference):
    """Both runs at their common times: the passage of the starting vortex and the tip circulation."""
    for x in PLANES:
        c, r = coupled[x], reference[x]
        times = sorted(set(np.round(c["time"], 6)) & set(np.round(r["time"], 6)))
        if not times:
            continue
        ci = [int(np.argmin(np.abs(c["time"] - t))) for t in times]
        ri = [int(np.argmin(np.abs(r["time"] - t))) for t in times]
        passage = int(np.argmax(np.abs(r["omega_z_integral"][ri])))
        last = -1
        print(
            f"x = {x:g} m, passage at t = {times[passage]:.2f} s: int omega_z coupled "
            f"{c['omega_z_integral'][ci[passage]]:+.3f}, reference "
            f"{r['omega_z_integral'][ri[passage]]:+.3f} m^2/s; peak omega_z coupled "
            f"{c['omega_z_peak'][ci[passage]]:+.2f}, reference {r['omega_z_peak'][ri[passage]]:+.2f} 1/s"
        )
        print(
            f"x = {x:g} m, t = {times[last]:.2f} s: tip circulation z>0 coupled "
            f"{c['circulation_upper'][ci[last]]:+.3f}, reference {r['circulation_upper'][ri[last]]:+.3f}; "
            f"z<0 coupled {c['circulation_lower'][ci[last]]:+.3f}, reference "
            f"{r['circulation_lower'][ri[last]]:+.3f} m^2/s"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--coupled-case", type=Path, default=CASE_DIR, help="case folder of the coupled run"
    )
    arguments = parser.parse_args()

    coupled_backups = sorted(glob.glob(f"{arguments.coupled_case}/solution/coupled/coupled_*"))
    reference_backups = sorted(
        p for p in glob.glob(f"{REFERENCE_SOLUTION}/fvm_*") if Path(p).is_dir()
    )
    if not coupled_backups:
        print(
            f"starting vortex: no coupled backups in {arguments.coupled_case}/solution/coupled; skipped"
        )
        return
    if not reference_backups or not REFERENCE_MESH.exists():
        print(
            f"starting vortex: no reference backups or mesh in {REFERENCE_SOLUTION.parent}; skipped"
        )
        return
    metadata = json.loads(Path(arguments.coupled_case, "solution/vpm_metadata.json").read_text())
    zeta = kernel_definition(metadata["configuration"]["numerics"]["particle_kernel"]).zeta

    coupled = cached_states(
        "coupled", coupled_backups, coupled_time, lambda b: particle_vorticity(b, zeta)
    )
    mesh = None

    def reference_fields(backup):
        nonlocal mesh
        mesh = mesh or ReferenceMesh()
        return fvm_vorticity(backup, mesh)

    reference = cached_states("reference", reference_backups, reference_time, reference_fields)

    set_thesis_style()
    for index, x in enumerate(PLANES):
        shown = passage_states(reference, coupled, index)
        if not shown:
            print(f"starting vortex: no common state at x = {x:g} m; maps skipped")
            continue
        for component, symbol in ((2, "z"), (0, "x")):
            map_figure(f"x{x:g}_omega_{symbol}", component, x, index, shown, reference, coupled)
    coupled_measures, reference_measures = measures(coupled), measures(reference)
    history_figure(coupled_measures, reference_measures)
    report(coupled_measures, reference_measures)


if __name__ == "__main__":
    main()
