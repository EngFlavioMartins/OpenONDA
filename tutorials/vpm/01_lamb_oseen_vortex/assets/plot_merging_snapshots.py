"""Depth-tested sphere views of the GBD merger, with vector LaTeX labels.

Figure contents: compare the initial and final finite columns, particle vorticity
and instantaneous in-plane streamlines. Analytic ray-sphere intersections use
an orthographic camera, surface lighting and a depth buffer. Sphere radii
scale with particle-strength magnitude to power 0.65 within each frame. Colour limits are
clipped per frame and displayed explicitly. No particles are filtered out.
Both states share one field of view enclosing both clouds and the sampling plane.
The 600-dpi scene is embedded in a 125 x 73 mm PDF with vector labels.
Blue particles and amber streamlines use the approved thesis field maps.
"""

import argparse
from pathlib import Path

import numpy as np

from openonda.scenes import (
    render_spheres,
    render_streamlines,
    sample_axial_velocity,
    sample_axial_vorticity,
)
from source.solvers.vpm.io.postprocess import particle_state

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

from .postprocess import _metadata, load_theme

CASE_DIR = Path(__file__).resolve().parents[1]
OUT = CASE_DIR / "figures"


def project(points):
    view = np.array([0.7, 0.61, -0.37])
    view /= np.linalg.norm(view)
    right = np.array([0.0, 0.0, 1.0]) - view[2] * view
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    angle = np.deg2rad(0)
    basis = np.array(
        [np.cos(angle) * right - np.sin(angle) * up, np.sin(angle) * right + np.cos(angle) * up]
    )
    return (points @ basis.T, points @ view)


def _initial_particle_state(case_dir: Path):
    state = particle_state(case_dir / "solution/merging_gbd/vpm/vpm_000000.h5")
    return (state["position"], state["vortex_strength"], state["core_radius"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--format", choices=("pdf", "png", "both"), default="both")
    args = parser.parse_args()
    _, theme = load_theme()
    plt.rcParams.update({"axes.linewidth": 0.6, "pdf.compression": 9})
    run = _metadata(CASE_DIR / "solution/merging_gbd/vpm_metadata.json")
    viscosity = float(run["kinematic_viscosity"])
    a0 = float(run["velocity_peak_radius"])
    circulation = abs(float(run["circulations"][0]))
    sample_z = 0.5 * float(run["column_half_length"])
    pos0, alpha0, sigma0 = _initial_particle_state(CASE_DIR)
    completed_steps = int(run["completed_steps"])
    path = CASE_DIR / "solution/merging_gbd/vpm" / f"vpm_{completed_steps:06d}.h5"
    final = particle_state(path)
    pos1, alpha1, sigma1, time = (
        final["position"],
        final["vortex_strength"],
        final["core_radius"],
        final["time"],
    )
    uc = circulation / (2 * np.pi * a0)
    wc = circulation / (np.pi * a0 * a0)
    x = np.linspace(-1.4, 1.4, 149)
    y = x.copy()
    xx, yy = np.meshgrid(x, y)
    query = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, sample_z)))
    cases = []
    for pos, alpha, sigma, t in [(pos0, alpha0, sigma0, 0.0), (pos1, alpha1, sigma1, time)]:
        omega = sample_axial_vorticity(pos, pos, alpha, sigma, cutoff=5.0) / wc
        vel = sample_axial_velocity(query, pos, alpha, sigma).reshape(len(y), len(x), 2)
        cases.append((pos, alpha, omega, vel, t))
        print(
            f"Reconstructed nu*t/a_c0^2={t * viscosity / a0**2:.2g}: {len(pos)} particles",
            flush=True,
        )
    plane_corners = np.array([[px, py, sample_z] for px in [x[0], x[-1]] for py in [y[0], y[-1]]])
    bounds = np.concatenate([project(pos0)[0], project(pos1)[0], project(plane_corners)[0]])
    lower = bounds.min(axis=0)
    upper = bounds.max(axis=0)
    max_radius = max(
        (
            float(
                np.max(
                    0.035
                    * (np.linalg.norm(alpha, axis=1) / np.linalg.norm(alpha, axis=1).max()) ** 0.65
                )
            )
            for _, alpha, _, _, _ in cases
        )
    )
    padding = 0.018 * (upper - lower) + max_radius
    lower -= padding
    upper += padding
    width = 3000
    height = int(np.ceil((width - 1) * (upper[1] - lower[1]) / (upper[0] - lower[0]))) + 1
    upper[1] = lower[1] + (height - 1) * (upper[0] - lower[0]) / (width - 1)
    axes_rectangle = [0.015, 0.02, 0.97, 0.97]
    view_limits = []
    for (pos, alpha, omega, vel, t), name in zip(
        cases, ["merging_render_t0.pdf", "merging_render_final.pdf"], strict=False
    ):
        fig = plt.figure(figsize=(125 / 25.4, 73 / 25.4), facecolor="white")
        ax = fig.add_axes(axes_rectangle)
        ax.set_aspect("equal", anchor="N")
        ax.axis("off")
        projected, depth = project(pos)
        strength = np.linalg.norm(alpha, axis=1)
        radii = 0.035 * (strength / strength.max()) ** 0.65
        wmax = min(float(np.quantile(omega, 0.95)), 0.8 * float(np.max(omega)))
        wmin = 0.08 * wmax
        speed = np.linalg.norm(vel, axis=2) / uc
        umax = float(np.quantile(speed, 0.99))
        norms = [Normalize(wmin, wmax), Normalize(0, umax)]
        particle_cmap = plt.get_cmap("thesis_blue").copy()
        particle_cmap.set_under((0.78, 0.82, 0.87, 1.0))
        colors = particle_cmap(norms[0](omega))[:, :3]
        scratch, sax = plt.subplots()
        streams = sax.streamplot(
            x,
            y,
            vel[:, :, 0],
            vel[:, :, 1],
            density=0.75,
            arrowsize=0.001,
            broken_streamlines=False,
            color=speed,
            cmap="thesis_amber",
            norm=norms[1],
            linewidth=0.6,
        )
        raw = streams.lines.get_segments()
        line_colors = plt.get_cmap("thesis_amber")(norms[1](streams.lines.get_array()))[:, :3]
        segments = []
        for line in raw:
            points = np.column_stack((line, np.full(len(line), sample_z)))
            plane, plane_depth = project(points)
            segments.append(np.column_stack((plane, plane_depth)))
        plt.close(scratch)
        segments = np.array(segments)
        canvas, zbuffer = render_spheres(
            np.column_stack((projected, depth)),
            radii,
            colors,
            lower,
            upper,
            width,
            height,
            light=np.array([-0.45, 0.65, 0.65]),
        )
        canvas = render_streamlines(canvas, zbuffer, segments, line_colors, lower, upper, 2.65)
        ax.imshow(
            np.uint8(np.clip(canvas, 0, 1) * 255),
            extent=[lower[0], upper[0], lower[1], upper[1]],
            interpolation="none",
        )
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.apply_aspect()
        view_limits.append([list(lower), list(upper)])
        colorbar_axes = []
        for box, cmap, norm, label, extension, title_location in [
            (
                [0.015, 0.705, 0.023, 0.22],
                particle_cmap,
                norms[0],
                "$\\omega_z/\\omega_{c,0}$",
                "both",
                "left",
            ),
            (
                [0.89, 0.2, 0.023, 0.2],
                "thesis_amber",
                norms[1],
                "$|\\mathbf{u}|/U_{c,0}$",
                "max",
                "right",
            ),
        ]:
            cax = fig.add_axes(box)
            colorbar_axes.append(cax)
            cb = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax,
                ticks=[norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax],
                format=FormatStrFormatter("%.2g"),
                extend=extension,
            )
            cb.ax.tick_params(width=0.6, length=2, pad=2)
            for tick_label in cb.ax.get_yticklabels():
                tick_label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.4})
            cb.ax.set_title(
                label,
                pad=7,
                loc=title_location,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
            )
        tau = t * viscosity / a0**2
        time_relation = "=" if t == 0.0 else "\\approx"
        fig.text(
            0.98, 0.045, f"$\\nu t/a_{{c,0}}^2{time_relation}{tau:.2g}$", ha="right", va="bottom"
        )
        theme.export_figure(fig, args.output_dir / name, figure_format=args.format, dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    main()
