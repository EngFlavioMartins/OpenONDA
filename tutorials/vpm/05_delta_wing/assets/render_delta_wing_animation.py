"""Animate the wakes shed by the two heaving wings, with their lift.

Each frame shows the particles and wing surfaces written at one output time,
coloured by the wing that shed them and brightened by their vorticity. The
inset traces the lift coefficient of each wing up to that time. By default the
animation plays at half the physical speed. The result is
``assets/delta_wing.gif``, shown at the top of the main README.

python -m assets.render_delta_wing_animation               the README banner GIF, 1280 x 460
python -m assets.render_delta_wing_animation presentation  a 1920 x 1080 H.264 video for slides, in
                                                           docs/assets/presentation/, one frame per
                                                           saved state; its frames are kept in
                                                           solution/delta_wing/presentation_frames/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pyvista as pv
from scipy.signal import find_peaks

from openonda.plotting import (
    PRESENTATION_DIR,
    PRESENTATION_SIZE,
    PRESENTATION_TEXT_SCALE,
    vorticity_discs,
    write_presentation_video,
)

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution" / "delta_wing"
FORCES = CASE_DIR / "samples" / "delta_wing" / "vlm_surface_forces.csv"
OUTPUT = CASE_DIR / "assets" / "delta_wing.gif"
PRESENTATION = PRESENTATION_DIR / "delta_wing.mp4"
PRESENTATION_FRAMES = SOLUTION_DIR / "presentation_frames"

IMAGE_SIZE = (1280, 460)  # [px], the README banner
RENDER_SCALE = 2  # GIF: render at twice the size, then downsample smoothly
VIEW_DIRECTION = (-0.15, 1.0, 0.9)  # toward the camera
# The GIF frame is a figure at 100 dpi, the video frame a figure 5 in tall at height / 5 dpi, in
# which `text` scales the gallery font sizes for a 1080p slide. `distance` is the camera distance
# and `wake_shift` the focal point downstream of the wing pair, per its streamwise extent; the
# video steps back to fit the banner's wake in the 16:9 picture.
FORMATS = {
    "gif": dict(size=IMAGE_SIZE, dpi=100, text=1.0, discs=False, distance=1.9, wake_shift=0.78,
                inset=(0.735, 0.6, 0.235, 0.27)),
    "presentation": dict(size=PRESENTATION_SIZE, dpi=PRESENTATION_SIZE[1] / 5.0,
                         text=PRESENTATION_TEXT_SCALE, discs=True, distance=3.0, wake_shift=0.78,
                         inset=(0.75, 0.72, 0.2, 0.16)),
}
VIDEO_FRAMES_PER_SECOND = 10  # video: saved states per second
# Video: the particles as discs (openonda.plotting) in their wing's colour map, transparent below
# DISC_CUT and opaque from DISC_OPAQUE, as fractions of the colour scale.
DISC_RADIUS = 0.8 * 5.0 * 0.005  # [m], 0.8 of the particle spacing, U dt
DISC_CUT, DISC_OPAQUE = 0.25, 0.7

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
WING_COLOURS = {"front_wing": "#f39c1f", "rear_wing": "#9bd75a"}
WING_COLORMAPS = tuple(  # by group id: the front wing sheds group 0, the rear wing group 1
    LinearSegmentedColormap.from_list(
        name, [to_rgb(colour) * np.array(0.45), colour, "#ffffff"], N=16
    )
    for name, colour in WING_COLOURS.items()
)


def wing_loads() -> pd.DataFrame:
    """Sampled forces of each wing, with its lift coefficient on its own reference area."""
    data = pd.read_csv(FORCES).sort_values(["time", "surface"])
    numerics = json.loads((SOLUTION_DIR / "vpm_metadata.json").read_text(encoding="utf-8"))[
        "configuration"
    ]["numerics"]
    vlm = numerics["vlm"]
    speed = np.linalg.norm(numerics["freestream_velocity"])
    area = {surface["name"]: surface["geometry"]["refs"]["area"] for surface in vlm["surfaces"]}
    reference = 0.5 * vlm["density"] * speed**2 * data.surface.map(area)
    return data.assign(lift_coefficient=data.force_z / reference)


def motion_period(data: pd.DataFrame) -> float:
    """Heave period, measured from the sampled heave velocity of the front wing."""
    front = data[data.surface == "front_wing"].sort_values("time")
    peaks, _ = find_peaks(front.translation_velocity_z)
    return float(np.median(np.diff(front.time.to_numpy()[peaks])))


def read_series(collection: Path) -> list[tuple[float, pv.DataSet]]:
    """Every dataset of a ParaView collection with its time."""
    reader = pv.get_reader(str(collection))
    frames = []
    for time in reader.time_values:
        reader.set_active_time_value(time)
        frames.append((float(time), reader.read()[0]))
    return frames


def camera(bounds: np.ndarray, distance: float, wake_shift: float) -> dict[str, np.ndarray]:
    """A fixed oblique view from beside and above, with the wings at the left."""
    lower, upper = bounds[::2], bounds[1::2]
    length = float(upper[0] - lower[0])
    toward_camera = np.asarray(VIEW_DIRECTION, dtype=float)
    toward_camera /= np.linalg.norm(toward_camera)
    focal_point = 0.5 * (lower + upper) - np.array([wake_shift * length, 0.0, 0.0])
    return {
        "focal_point": focal_point,
        "position": focal_point + distance * length * toward_camera,
        "view_up": np.array([0.0, 0.0, 1.0]),
    }


def render_scene(plotter, particles, surface, view, omega_max, discs=False):
    """Render one frame of particles and wings, returning its RGB image; the video draws the
    particles as discs."""
    plotter.clear()
    if particles.n_points:
        points = np.asarray(particles.points)
        vorticity = np.linalg.norm(np.asarray(particles["vorticity"]), axis=1)
        wing = np.asarray(particles["group_id"])
        # One hue per wing; weak vorticity fades out, so the tip vortices stand out
        for index, colormap in enumerate(WING_COLORMAPS):
            shed = wing == index
            if not np.any(shed):
                continue
            if discs:
                plotter.add_actor(vorticity_discs(
                    points[shed], vorticity[shed], view["position"], radius=DISC_RADIUS,
                    cut=DISC_CUT * omega_max, opaque=DISC_OPAQUE * omega_max, maximum=omega_max,
                    colormap=colormap,
                ))
                continue
            cloud = pv.PolyData(points[shed])
            cloud["vorticity magnitude"] = vorticity[shed]
            plotter.add_mesh(
                cloud,
                scalars="vorticity magnitude",
                cmap=colormap,
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


def compose(image, time, loads, frequency, omega_max, layout):
    """Lay the rendered scene and the lift inset on one frame in the given format."""
    size, dpi, text = layout["size"], layout["dpi"], layout["text"]
    with plt.rc_context({**GALLERY_STYLE, "font.size": GALLERY_STYLE["font.size"] * text}):
        figure = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        inset = figure.add_axes(layout["inset"])
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
                    markersize=8 * text**0.5,
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("format", nargs="?", choices=tuple(FORMATS), default="gif")
    parser.add_argument("--speed", type=float, default=0.5, help="playback speed; 1 is real time")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    layout = FORMATS[args.format]
    size = layout["size"]
    render_scale = 1 if layout["discs"] else RENDER_SCALE

    particles = read_series(SOLUTION_DIR / "vpm.pvd")
    surfaces = read_series(SOLUTION_DIR / "vlm.pvd")
    loads = wing_loads()
    frequency = 1.0 / motion_period(loads)

    final = particles[-1][1]
    omega_max = float(f"{np.percentile(np.linalg.norm(final['vorticity'], axis=1), 99):.1g}")
    view = camera(np.array(surfaces[-1][1].bounds), layout["distance"], layout["wake_shift"])

    plotter = pv.Plotter(off_screen=True, window_size=[render_scale * pixels for pixels in size])
    plotter.set_background("black")
    if layout["discs"]:
        plotter.enable_anti_aliasing("fxaa")
        PRESENTATION_FRAMES.mkdir(parents=True, exist_ok=True)
        for old in PRESENTATION_FRAMES.glob("frame_*.png"):
            old.unlink()
    frames = []
    for (time, cloud), (_, surface) in zip(particles, surfaces, strict=True):
        image = render_scene(plotter, cloud, surface, view, omega_max, layout["discs"])
        frames.append(compose(image, time, loads, frequency, omega_max, layout))
        if layout["discs"]:
            frames[-1].save(PRESENTATION_FRAMES / f"frame_{len(frames) - 1:04d}.png")
    plotter.close()

    if layout["discs"]:
        write_presentation_video(
            frames, PRESENTATION, VIDEO_FRAMES_PER_SECOND, frame_dir=PRESENTATION_FRAMES
        )
        print(f"  Saved: {PRESENTATION} and {len(frames)} frames in {PRESENTATION_FRAMES}")
        return

    sample = Image.new("RGB", (IMAGE_SIZE[0], 4 * IMAGE_SIZE[1]))
    for row, frame in enumerate(frames[len(frames) // 4 :: len(frames) // 4][:4]):
        sample.paste(frame, (0, row * IMAGE_SIZE[1]))
    palette = sample.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]

    # Each frame lasts its physical time step, so the wings heave at their real rate
    times = np.array([time for time, _ in particles])
    durations = [round(1000 * step / args.speed) for step in np.diff(times)] + [1000]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
