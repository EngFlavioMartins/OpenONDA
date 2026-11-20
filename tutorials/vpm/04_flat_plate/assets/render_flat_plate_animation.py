"""Animate the particle wake shed by the moving flat plate, with its lift and drag.

Each frame shows the particles and the plate surface written by the animated
case (``ANIMATED_CASE`` in ``setup.py``) at one output time, coloured by the
vorticity magnitude. The inset traces the lift and drag coefficients up to that
time. The result is ``assets/flat_plate_wake.gif``, shown in the README gallery.

python -m assets.render_flat_plate_animation               the README gallery GIF, 800 x 500
python -m assets.render_flat_plate_animation presentation  a 1920 x 1080 H.264 video for slides, in
                                                           docs/assets/presentation/, one frame per
                                                           saved state; its frames are kept in
                                                           solution/exp_moving_aoa12/
                                                           presentation_frames/
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

from openonda.plotting import (
    PRESENTATION_DIR,
    PRESENTATION_SIZE,
    PRESENTATION_TEXT_SCALE,
    vorticity_discs,
    write_presentation_video,
)

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution" / "exp_moving_aoa12"
FORCES = CASE_DIR / "samples" / "exp_moving_aoa12" / "vlm_forces.csv"
OUTPUT = CASE_DIR / "assets" / "flat_plate_wake.gif"
PRESENTATION = PRESENTATION_DIR / "flat_plate_wake.mp4"
PRESENTATION_FRAMES = SOLUTION_DIR / "presentation_frames"

CHORD = 1.0  # [m]
FREESTREAM_SPEED = 10.0  # [m/s], plate speed after the ramp
IMAGE_SIZE = (800, 500)  # [px], as the other gallery animations
RENDER_SCALE = 2  # GIF: render at twice the size, then downsample smoothly
# Each frame is a figure 5 in tall at height / 5 dpi; `text` scales the gallery font sizes for a
# 1080p slide. The GIF scene is drawn at twice its size and reduced, the video scene at its own.
# `distance` is the camera distance and `rise` the focal point above the wake centre, per extent
# of the final wake; the video is framed closer, the wake across the 16:9 picture.
FORMATS = {
    "gif": dict(size=IMAGE_SIZE, render_scale=RENDER_SCALE, text=1.0, discs=False,
                distance=1.6, rise=0.1,
                bar=(0.045, 0.52, 0.04, 0.28), inset=(0.64, 0.70, 0.32, 0.24)),
    "presentation": dict(size=PRESENTATION_SIZE, render_scale=1, text=PRESENTATION_TEXT_SCALE,
                         discs=True, distance=1.35, rise=0.03,
                         bar=(0.04, 0.58, 0.016, 0.27), inset=(0.72, 0.74, 0.22, 0.16)),
}
VIDEO_FRAMES_PER_SECOND = 10  # video: saved states per second
# Video: the particles as vorticity discs (openonda.plotting), transparent below DISC_CUT and
# opaque from DISC_OPAQUE, as fractions of the colour scale.
DISC_RADIUS = 0.8 * FREESTREAM_SPEED * 0.0125  # [m], 0.8 of the particle spacing, U dt
DISC_CUT, DISC_OPAQUE = 0.15, 0.7

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


def camera(bounds: np.ndarray, distance: float, rise: float) -> dict[str, np.ndarray]:
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
    focal_point = centre + rise * length * screen_up
    return {
        "focal_point": focal_point,
        "position": focal_point + distance * length * toward_camera,
        "view_up": view_up,
    }


def render_scene(plotter, particles, surface, view, omega_max, discs=False):
    """Render one frame of particles and plate, returning its RGB image; the video draws the
    particles as discs."""
    plotter.clear()
    # The plate starts from rest, before it has shed any particle
    if discs and particles.n_points:
        vorticity = np.linalg.norm(np.asarray(particles["vorticity"]), axis=1)
        plotter.add_actor(vorticity_discs(
            particles.points, vorticity, view["position"], radius=DISC_RADIUS,
            cut=DISC_CUT * omega_max, opaque=DISC_OPAQUE * omega_max, maximum=omega_max,
            colormap=VORTICITY_COLORMAP,
        ))
    elif particles.n_points:
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
    plotter.reset_camera_clipping_range()
    plotter.render()
    return plotter.screenshot(return_img=True)


def compose(image, time, forces, omega_max, layout):
    """Lay the rendered scene, the colour bar and the force inset on one frame in a format."""
    size, text = layout["size"], layout["text"]
    with plt.rc_context({**GALLERY_STYLE, "font.size": GALLERY_STYLE["font.size"] * text}):
        figure = plt.figure(figsize=(5.0 * size[0] / size[1], 5.0), dpi=size[1] / 5.0)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        # Vorticity magnitude: its two end values, the title above the bar
        bar = figure.add_axes(layout["bar"])
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(0.0, omega_max), VORTICITY_COLORMAP),
            cax=bar,
            ticks=(0.0, omega_max),
        )
        bar.set_yticklabels(["0", f"{omega_max:g}"], fontsize=30 * text)
        bar.tick_params(direction="out", length=10 * text, width=1.5)
        bar.set_title(r"$|\omega|$ [1/s]", fontsize=30 * text, pad=28 * text, loc="left")

        # Lift and drag up to this time; the rest of the history stays faint
        inset = figure.add_axes(layout["inset"])
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
                    markersize=8 * text**0.5,
                    clip_on=False,
                )
        inset.set_xlim(0.0, 25.0)
        # Start below zero, so the small drag stays clear of the axis
        inset.set_ylim(-0.25, 1.5)
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
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("format", nargs="?", choices=tuple(FORMATS), default="gif")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    layout = FORMATS[args.format]
    size, render_scale = layout["size"], layout["render_scale"]

    particles = read_series(SOLUTION_DIR / "vpm.pvd")
    surfaces = read_series(SOLUTION_DIR / "vlm.pvd")
    forces = pd.read_csv(FORCES)

    # One colour scale and one view for every frame, set by the final wake
    final = particles[-1][1]
    omega_max = float(np.ceil(np.linalg.norm(final["vorticity"], axis=1).max()))
    bounds = np.array(final.bounds)
    bounds[::2] = np.minimum(bounds[::2], surfaces[-1][1].bounds[::2])
    bounds[1::2] = np.maximum(bounds[1::2], surfaces[-1][1].bounds[1::2])
    view = camera(bounds, layout["distance"], layout["rise"])

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
        frames.append(compose(image, time, forces, omega_max, layout))
        if layout["discs"]:
            frames[-1].save(PRESENTATION_FRAMES / f"frame_{len(frames) - 1:04d}.png")
    plotter.close()

    if layout["discs"]:
        write_presentation_video(
            frames, PRESENTATION, VIDEO_FRAMES_PER_SECOND, frame_dir=PRESENTATION_FRAMES
        )
        print(f"  Saved: {PRESENTATION} and {len(frames)} frames in {PRESENTATION_FRAMES}")
        return

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
