"""Animate the wakes shed by the two heaving wings, with their lift.

Each frame shows the particles and wing surfaces written at one output time,
coloured by the vorticity magnitude. The inset traces the lift coefficient of
each wing up to that time. ``assets/delta_wing_30fps.gif``, shown at the top of
the README, is a copy of the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyvista as pv

from assets.postprocess import SOLUTION_DIR, force_history, lift_and_thrust, motion_period

OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "delta_wing_30fps.gif"

IMAGE_SIZE = (1280, 576)  # [px], the README banner
RENDER_SCALE = 2  # render at twice the size, then downsample smoothly
VIEW_DIRECTION = (-0.15, 1.0, 0.9)  # toward the camera
VIEW_DISTANCE = 2.2  # camera distance per streamwise extent of the wing pair
WAKE_SHIFT = 0.5  # focal point downstream of the wing pair, per its streamwise extent

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
WING_COLOURS = {"front_wing": "#f39c1f", "rear_wing": "#9bd75a"}


def read_series(collection: Path) -> list[tuple[float, pv.DataSet]]:
    """Every dataset of a ParaView collection with its time."""
    reader = pv.get_reader(str(collection))
    frames = []
    for time in reader.time_values:
        reader.set_active_time_value(time)
        frames.append((float(time), reader.read()[0]))
    return frames


def camera(bounds: np.ndarray) -> dict[str, np.ndarray]:
    """A fixed oblique view from beside and above, with the wings at the left."""
    lower, upper = bounds[::2], bounds[1::2]
    length = float(upper[0] - lower[0])
    toward_camera = np.asarray(VIEW_DIRECTION, dtype=float)
    toward_camera /= np.linalg.norm(toward_camera)
    focal_point = 0.5 * (lower + upper) - np.array([WAKE_SHIFT * length, 0.0, 0.0])
    return {
        "focal_point": focal_point,
        "position": focal_point + VIEW_DISTANCE * length * toward_camera,
        "view_up": np.array([0.0, 0.0, 1.0]),
    }


def render_scene(plotter, particles, surface, view, omega_max):
    """Render one frame of particles and wings, returning its RGB image."""
    plotter.clear()
    if particles.n_points:
        cloud = pv.PolyData(np.asarray(particles.points))
        cloud["vorticity magnitude"] = np.linalg.norm(np.asarray(particles["vorticity"]), axis=1)
        plotter.add_mesh(
            cloud,
            scalars="vorticity magnitude",
            cmap=VORTICITY_COLORMAP,
            clim=(0.0, omega_max),
            opacity=np.linspace(0.15, 1.0, 16),
            render_points_as_spheres=True,
            point_size=2.5 * RENDER_SCALE,
            show_scalar_bar=False,
        )
    plotter.add_mesh(surface, color="#9c9c9c", smooth_shading=True)
    plotter.camera.position = view["position"]
    plotter.camera.focal_point = view["focal_point"]
    plotter.camera.up = view["view_up"]
    plotter.reset_camera_clipping_range()
    plotter.render()
    return plotter.screenshot(return_img=True)


def compose(image, time, loads, frequency, omega_max):
    """Lay the rendered scene, the colour bar and the lift inset on one frame."""
    with plt.rc_context(GALLERY_STYLE):
        figure = plt.figure(figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100), dpi=100)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        bar = figure.add_axes((0.035, 0.1, 0.022, 0.3))
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(0.0, omega_max), VORTICITY_COLORMAP),
            cax=bar,
            ticks=(0.0, omega_max),
        )
        bar.set_yticklabels(["0", f"{omega_max:g}"], fontsize=26)
        bar.tick_params(direction="out", length=8, width=1.5)
        bar.set_title(r"$|\omega|$ [1/s]", fontsize=26, pad=20, loc="left")

        inset = figure.add_axes((0.735, 0.62, 0.235, 0.27))
        inset.set_facecolor("none")
        for surface, rows in loads.groupby("surface"):
            cycles = rows["time"].to_numpy() * frequency
            values = rows["lift_coefficient"].to_numpy()
            now = rows["time"].to_numpy() <= time + 1e-9
            colour = WING_COLOURS[surface]
            label = surface.split("_")[0]
            inset.plot(cycles, values, color=colour, alpha=0.25, linewidth=1.5)
            inset.plot(cycles[now], values[now], color=colour, linewidth=2.5, label=label)
            if np.any(now):
                inset.plot(
                    cycles[now][-1],
                    values[now][-1],
                    "o",
                    color=colour,
                    markeredgecolor="white",
                    markersize=8,
                    clip_on=False,
                )
        cycles = loads["time"].to_numpy() * frequency
        inset.set_xlim(0.0, np.ceil(cycles.max()))
        low, high = loads["lift_coefficient"][cycles > 0.5].quantile([0.0, 1.0])
        inset.set_ylim(np.floor(2 * low) / 2, np.ceil(2 * high) / 2)
        inset.locator_params(axis="x", nbins=3)
        inset.locator_params(axis="y", nbins=3)
        inset.set_xlabel(r"$t\,f$")
        inset.set_ylabel(r"$C_L$")
        inset.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.97),
            ncols=2,
            handlelength=1.1,
            handletextpad=0.4,
            columnspacing=0.8,
        )
        inset.spines[["top", "right"]].set_visible(False)

        figure.canvas.draw()
        frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3])
        plt.close(figure)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    particles = read_series(SOLUTION_DIR / "vpm.pvd")
    surfaces = read_series(SOLUTION_DIR / "vlm.pvd")
    data = force_history()
    loads = lift_and_thrust(data)
    frequency = 1.0 / motion_period(data)

    final = particles[-1][1]
    omega_max = float(f"{np.percentile(np.linalg.norm(final['vorticity'], axis=1), 99):.1g}")
    view = camera(np.array(surfaces[-1][1].bounds))

    plotter = pv.Plotter(off_screen=True, window_size=[RENDER_SCALE * size for size in IMAGE_SIZE])
    plotter.set_background("black")
    frames = []
    for (time, cloud), (_, surface) in zip(particles, surfaces, strict=True):
        image = render_scene(plotter, cloud, surface, view, omega_max)
        frames.append(compose(image, time, loads, frequency, omega_max))
    plotter.close()

    sample = Image.new("RGB", (IMAGE_SIZE[0], 4 * IMAGE_SIZE[1]))
    for row, frame in enumerate(frames[len(frames) // 4 :: len(frames) // 4][:4]):
        sample.paste(frame, (0, row * IMAGE_SIZE[1]))
    palette = sample.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]

    durations = [round(1000 / args.fps)] * (len(frames) - 1) + [1000]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
