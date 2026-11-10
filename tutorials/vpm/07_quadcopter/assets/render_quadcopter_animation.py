"""Animate the particle wakes of the four rotors, with their thrust and power.

Each frame shows the particles and blade surfaces written at one output time,
coloured by the rotor that shed them and brightened by their vorticity. The
inset traces the thrust and power coefficients, averaged over the four rotors,
up to that time.
``assets/quadcopter_flow.gif``, shown in the README gallery, is a copy of the
result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyvista as pv

from assets._quadcopter_plots import performance, rotor_inputs

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution" / "quadcopter"
OUTPUT = CASE_DIR / "figures" / "quadcopter_flow.gif"

IMAGE_SIZE = (800, 500)  # [px], as the other gallery animations
RENDER_SCALE = 2  # render at twice the size, then downsample smoothly
VIEW_DIRECTION = (-0.45, -1.0, 0.55)  # toward the camera
VIEW_DISTANCE = 2.9  # camera distance per height of the final wake
SCREEN_SHIFT = 0.17  # wake left of the frame centre, per height of the final wake

# README gallery style: white DejaVu Serif text on black, a stepped blue vorticity scale
GALLERY_STYLE = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 20,
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
ROTOR_COLOURS = ("#3d9bff", "#ff4fa3", "#ffd23f", "#3fe0b8")  # rotors 0 to 3
ROTOR_COLORMAPS = tuple(
    LinearSegmentedColormap.from_list(
        f"rotor_{index}", [to_rgb(colour) * np.array(0.45), colour, "#ffffff"], N=16
    )
    for index, colour in enumerate(ROTOR_COLOURS)
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
    """A fixed view from above one side, with the vehicle at the top and its wake below."""
    lower, upper = bounds[::2], bounds[1::2]
    centre = 0.5 * (lower + upper)
    height = float(upper[2] - lower[2])
    toward_camera = np.asarray(VIEW_DIRECTION, dtype=float)
    toward_camera /= np.linalg.norm(toward_camera)
    view_up = np.array([0.0, 0.0, 1.0])
    screen_right = np.cross(view_up, toward_camera)
    screen_right /= np.linalg.norm(screen_right)
    # Aim right of the wake, so it clears the force inset
    focal_point = centre + SCREEN_SHIFT * height * screen_right
    return {
        "focal_point": focal_point,
        "position": focal_point + VIEW_DISTANCE * height * toward_camera,
        "view_up": view_up,
    }


def render_scene(plotter, particles, blades, view, omega_max):
    """Render one frame of particles and blades, returning its RGB image."""
    plotter.clear()
    # The rotors start from rest, before they have shed any particle
    if particles.n_points:
        points = np.asarray(particles.points)
        vorticity = np.linalg.norm(np.asarray(particles["vorticity"]), axis=1)
        rotor = np.asarray(particles["group_id"]) - 1
        # One hue per rotor; weak vorticity fades out, so the tip vortices stand out
        for index, colormap in enumerate(ROTOR_COLORMAPS):
            shed = rotor == index
            if not np.any(shed):
                continue
            cloud = pv.PolyData(points[shed])
            cloud["vorticity magnitude"] = vorticity[shed]
            plotter.add_mesh(
                cloud,
                scalars="vorticity magnitude",
                cmap=colormap,
                clim=(0.0, omega_max),
                opacity=np.linspace(0.2, 1.0, 16),
                render_points_as_spheres=True,
                point_size=2.0 * RENDER_SCALE,
                show_scalar_bar=False,
            )
    plotter.add_mesh(blades, color="#9c9c9c", smooth_shading=True)
    plotter.camera.position = view["position"]
    plotter.camera.focal_point = view["focal_point"]
    plotter.camera.up = view["view_up"]
    plotter.reset_camera_clipping_range()
    plotter.render()
    return plotter.screenshot(return_img=True)


def compose(image, time, coefficients, period):
    """Lay the rendered scene and the force inset on one frame."""
    with plt.rc_context(GALLERY_STYLE):
        figure = plt.figure(figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100), dpi=100)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        # Thrust and power up to this time; the rest of the history stays faint
        inset = figure.add_axes((0.64, 0.6, 0.32, 0.22))
        inset.set_facecolor("none")
        revolutions = coefficients.index.to_numpy() / period
        now = coefficients.index.to_numpy() <= time + 1e-9
        for name, scale, label, colour in (
            ("CT", 1.0e3, r"$10^3C_T$", "#f39c1f"),
            ("CP", 1.0e4, r"$10^4C_P$", "#9bd75a"),
        ):
            values = scale * coefficients[name].to_numpy()
            inset.plot(revolutions, values, color=colour, alpha=0.25, linewidth=1.5)
            inset.plot(revolutions[now], values[now], color=colour, linewidth=2.5, label=label)
            if np.any(now):
                inset.plot(
                    revolutions[now][-1],
                    values[now][-1],
                    "o",
                    color=colour,
                    markeredgecolor="white",
                    markersize=8,
                    clip_on=False,
                )
        inset.set_xlim(0.0, np.ceil(revolutions.max()))
        settled = revolutions > 1.0
        low = min(
            1.0e3 * coefficients["CT"][settled].min(), 1.0e4 * coefficients["CP"][settled].min()
        )
        high = max(
            1.0e3 * coefficients["CT"][settled].max(), 1.0e4 * coefficients["CP"][settled].max()
        )
        inset.set_ylim(np.floor(low - 0.5), np.ceil(high + 0.5))
        inset.locator_params(axis="x", nbins=3)
        inset.locator_params(axis="y", nbins=3)
        inset.set_xlabel("revolutions")
        inset.legend(
            loc="lower center",
            bbox_to_anchor=(0.45, 0.98),
            ncols=2,
            fontsize=18,
            handlelength=1.0,
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
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    particles = read_series(SOLUTION_DIR / "vpm.pvd")
    blades = read_series(SOLUTION_DIR / "vlm.pvd")
    rotor = rotor_inputs()
    coefficients = performance(rotor.samples_dir, rotor).groupby("time")[["CT", "CP"]].mean()

    # One brightness scale for every frame, the median of their vorticity peaks, and one view
    peaks = [
        np.percentile(np.linalg.norm(cloud["vorticity"], axis=1), 99.5)
        for _, cloud in particles
        if cloud.n_points
    ]
    omega_max = float(f"{np.median(peaks):.1g}")
    final = particles[-1][1]
    bounds = np.array(final.bounds)
    view = camera(bounds)

    plotter = pv.Plotter(off_screen=True, window_size=[RENDER_SCALE * size for size in IMAGE_SIZE])
    plotter.set_background("black")
    frames = []
    for (time, cloud), (_, surface) in zip(particles, blades, strict=True):
        image = render_scene(plotter, cloud, surface, view, omega_max)
        frames.append(compose(image, time, coefficients, rotor.period))
    plotter.close()

    # One palette for every frame, from four frames across the animation, keeps the GIF small
    sample = Image.new("RGB", (IMAGE_SIZE[0], 4 * IMAGE_SIZE[1]))
    for row, frame in enumerate(frames[len(frames) // 4 :: len(frames) // 4][:4]):
        sample.paste(frame, (0, row * IMAGE_SIZE[1]))
    palette = sample.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]

    # Hold the final wake for one second before the animation repeats
    durations = [round(1000 / args.fps)] * (len(frames) - 1) + [1000]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
