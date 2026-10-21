"""Animate the particle wake shed by the moving flat plate, with its lift and drag.

Each frame shows the particles and the plate surface written by the animated
case (``ANIMATED_CASE`` in ``setup.py``) at one output time, coloured by the
vorticity magnitude. The inset traces the lift and drag coefficients up to that
time. ``assets/flat_plate_wake.gif``, shown in the README gallery, is a copy
of the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pyvista as pv

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution" / "exp_moving_aoa12"
FORCES = CASE_DIR / "samples" / "exp_moving_aoa12" / "vlm_forces.csv"
OUTPUT = CASE_DIR / "figures" / "flat_plate_wake.gif"

CHORD = 1.0  # [m]
FREESTREAM_SPEED = 10.0  # [m/s], plate speed after the ramp
IMAGE_SIZE = (800, 500)  # [px], as the other gallery animations
RENDER_SCALE = 2  # render at twice the size, then downsample smoothly

# README gallery style: white DejaVu Serif text on black, a stepped blue vorticity scale
GALLERY_STYLE = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 22,
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.edgecolor": "white",
    "axes.linewidth": 1.5,
    "xtick.color": "white",
    "ytick.color": "white",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": False,
}
VORTICITY_COLORMAP = LinearSegmentedColormap.from_list(
    "gallery_vorticity", ["#031a3c", "#0d58b7", "#3791ed", "#73c0ff", "#c9e6fd"], N=16
)


def read_series(collection: Path) -> list[tuple[float, pv.DataSet]]:
    """Every dataset of a ParaView collection with its time."""
    reader = pv.get_reader(str(collection))
    frames = []
    for time in reader.time_values:
        reader.set_active_time_value(time)
        frames.append((float(time), reader.read()[0]))
    return frames


def camera(bounds: np.ndarray) -> dict[str, np.ndarray]:
    """A fixed oblique view, from above and behind the plate's side, over the whole wake."""
    lower, upper = bounds[::2], bounds[1::2]
    centre = 0.5 * (lower + upper)
    length = float(np.max(upper - lower))
    toward_camera = np.array([0.3, -1.0, 1.0])
    toward_camera /= np.linalg.norm(toward_camera)
    view_up = np.array([0.0, 0.0, 1.0])
    screen_up = view_up - (view_up @ toward_camera) * toward_camera
    screen_up /= np.linalg.norm(screen_up)
    # Aim above the wake, so it sits below the colour bar and the force inset
    focal_point = centre + 0.12 * length * screen_up
    return {
        "focal_point": focal_point,
        "position": focal_point + 1.85 * length * toward_camera,
        "view_up": view_up,
    }


def render_scene(plotter, particles, surface, view, omega_max):
    """Render one frame of particles and plate, returning its RGB image."""
    plotter.clear()
    # The plate starts from rest, before it has shed any particle
    if particles.n_points:
        cloud = pv.PolyData(np.asarray(particles.points))
        cloud["vorticity magnitude"] = np.linalg.norm(np.asarray(particles["vorticity"]), axis=1)
        plotter.add_mesh(
            cloud,
            scalars="vorticity magnitude",
            cmap=VORTICITY_COLORMAP,
            clim=(0.0, omega_max),
            # Weak vorticity fades out, so the rolled-up tip vortices stand out
            opacity=np.linspace(0.2, 1.0, 16),
            render_points_as_spheres=True,
            point_size=3.0 * RENDER_SCALE,
            show_scalar_bar=False,
        )
    plotter.add_mesh(surface, color="#9c9c9c", smooth_shading=True)
    plotter.camera.position = view["position"]
    plotter.camera.focal_point = view["focal_point"]
    plotter.camera.up = view["view_up"]
    return plotter.screenshot(return_img=True)


def compose(image, time, forces, omega_max):
    """Lay the rendered scene, the colour bar and the force inset on one frame."""
    with plt.rc_context(GALLERY_STYLE):
        figure = plt.figure(figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100), dpi=100)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        # Vorticity magnitude: its two end values, the title above the bar
        bar = figure.add_axes((0.045, 0.52, 0.04, 0.28))
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(0.0, omega_max), VORTICITY_COLORMAP),
            cax=bar,
            ticks=(0.0, omega_max),
        )
        bar.set_yticklabels(["0", f"{omega_max:g}"], fontsize=30)
        bar.tick_params(direction="out", length=10, width=1.5)
        bar.set_title(r"$|\omega|$ [1/s]", fontsize=30, pad=28, loc="left")

        # Lift and drag up to this time; the rest of the history stays faint
        inset = figure.add_axes((0.64, 0.64, 0.32, 0.26))
        inset.set_facecolor("none")
        travel = forces["time"].to_numpy() * FREESTREAM_SPEED / CHORD
        now = forces["time"].to_numpy() <= time + 1e-9
        for name, label, colour in (
            ("lift_coefficient", "$C_L$", "#f39c1f"),
            ("drag_coefficient", "$C_D$", "#9bd75a"),
        ):
            values = forces[name].to_numpy()
            inset.plot(travel, values, color=colour, alpha=0.25, linewidth=1.5)
            inset.plot(travel[now], values[now], color=colour, linewidth=2.5, label=label)
            if np.any(now):
                inset.plot(
                    travel[now][-1],
                    values[now][-1],
                    "o",
                    color=colour,
                    markeredgecolor="white",
                    markersize=8,
                )
        inset.set_xlim(0.0, 25.0)
        inset.set_ylim(0.0, 1.5)
        inset.set_xticks((0, 10, 20))
        inset.set_yticks((0.0, 0.5, 1.0))
        inset.set_xlabel(r"$t\,U_\infty/c$")
        inset.legend(
            loc="center right",
            bbox_to_anchor=(1.06, 0.4),
            ncols=2,
            handlelength=1.1,
            handletextpad=0.4,
            columnspacing=0.8,
        )
        inset.spines[["top", "right"]].set_visible(False)

        figure.canvas.draw()
        frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3])
        plt.close(figure)
    return frame.quantize(colors=256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    particles = read_series(SOLUTION_DIR / "vpm.pvd")
    surfaces = read_series(SOLUTION_DIR / "vlm.pvd")
    forces = pd.read_csv(FORCES)

    # One colour scale and one view for every frame, set by the final wake
    final = particles[-1][1]
    omega_max = float(np.ceil(np.linalg.norm(final["vorticity"], axis=1).max()))
    bounds = np.array(final.bounds)
    bounds[::2] = np.minimum(bounds[::2], surfaces[-1][1].bounds[::2])
    bounds[1::2] = np.maximum(bounds[1::2], surfaces[-1][1].bounds[1::2])
    view = camera(bounds)

    plotter = pv.Plotter(off_screen=True, window_size=[RENDER_SCALE * size for size in IMAGE_SIZE])
    plotter.set_background("black")
    frames = []
    for (time, cloud), (_, surface) in zip(particles, surfaces, strict=True):
        image = render_scene(plotter, cloud, surface, view, omega_max)
        frames.append(compose(image, time, forces, omega_max))
    plotter.close()

    # Hold the final wake for one second before the animation repeats
    durations = [round(1000 / args.fps)] * (len(frames) - 1) + [1000]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
