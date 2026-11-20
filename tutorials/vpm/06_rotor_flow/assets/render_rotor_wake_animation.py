"""Animate the particle wake of the three-bladed rotor, with its thrust and power.

Each frame shows the particles and blade surfaces written at every third
output time, coloured by the vorticity magnitude. The inset traces the thrust
and power coefficients up to that time. The result is ``assets/rotor_flow.gif``,
shown in the README gallery.

python -m assets.render_rotor_wake_animation               the README gallery GIF, 800 x 500
python -m assets.render_rotor_wake_animation presentation  a 1920 x 1080 H.264 video for slides, in
                                                           docs/assets/presentation/, one frame per
                                                           saved state; its frames are kept in
                                                           solution/rotor/presentation_frames/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

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
from openonda.vpm.io.postprocess import accepted_history

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution" / "rotor"
FORCES = CASE_DIR / "samples" / "rotor" / "vlm_forces.csv"
OUTPUT = CASE_DIR / "assets" / "rotor_flow.gif"
PRESENTATION = PRESENTATION_DIR / "rotor_flow.mp4"
PRESENTATION_FRAMES = SOLUTION_DIR / "presentation_frames"

IMAGE_SIZE = (800, 500)  # [px], as the other gallery animations
RENDER_SCALE = 2  # GIF: render at twice the size, then downsample smoothly
VIEW_DIRECTION = (-0.6, -1.0, 0.7)  # toward the camera, from upstream of the rotor
OMEGA_MAX = 8.0  # [1/s], the tip vortices; the stronger root and blade vorticity saturates
# Each frame is a figure 5 in tall at height / 5 dpi; `text` scales the gallery font sizes for a
# 1080p slide, whose scene is drawn at its own size. `stride` keeps every third particle output
# in the GIF, which keeps it small, and every output in the video. `distance` is the camera
# distance, `shift` and `drop` the focal point right of and below the wake centre, per extent of
# the final wake; the video steps back to show the whole final wake and the blades.
FORMATS = {
    "gif": dict(size=IMAGE_SIZE, text=1.0, discs=False, stride=3,
                distance=1.35, shift=-0.05, drop=0.06,
                bar=(0.05, 0.62, 0.035, 0.26), inset=(0.64, 0.17, 0.32, 0.16)),
    "presentation": dict(size=PRESENTATION_SIZE, text=PRESENTATION_TEXT_SCALE, discs=True, stride=1,
                         distance=2.4, shift=0.03, drop=0.06,
                         bar=(0.04, 0.62, 0.016, 0.25), inset=(0.72, 0.13, 0.22, 0.12)),
}
VIDEO_FRAMES_PER_SECOND = 10  # video: saved states per second
# Video: the particles as vorticity discs (openonda.plotting), transparent below DISC_CUT and
# opaque from DISC_OPAQUE.
DISC_RADIUS = 0.8 * (6.0 - 1.0) / 22  # [m], 0.8 of the particle spacing, blade span / 22 panels
DISC_CUT, DISC_OPAQUE = 2.4, 6.4  # [1/s]

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
VORTICITY_COLORMAP = LinearSegmentedColormap.from_list(
    "gallery_vorticity", ["#031a3c", "#0d58b7", "#3791ed", "#73c0ff", "#c9e6fd"], N=16
)


def performance() -> tuple[pd.DataFrame, float]:
    """Thrust and power coefficients on the rotor disk, and the rotation period."""
    metadata = json.loads((SOLUTION_DIR / "vpm_metadata.json").read_text(encoding="utf-8"))
    numerics = metadata["configuration"]["numerics"]
    vlm = numerics["vlm"]
    blade = vlm["surfaces"][0]
    # Rotor radius: the outermost quarter-chord point of the first blade's panels
    radius = 0.0
    for wing in blade["geometry"]["wings"]:
        for segment in wing["segments"]:
            b, c = (np.array(segment["vertex_position"][key]) for key in "bc")
            radius = max(radius, float(np.linalg.norm((0.75 * b + 0.25 * c)[1:])))
    speed = float(np.linalg.norm(numerics["freestream_velocity"]))
    period = 2 * np.pi / abs(float(blade["kinematics"]["angular_speed"]))
    data = accepted_history(pd.read_csv(FORCES), metadata)
    reference = 0.5 * vlm["density"] * speed**2 * np.pi * radius**2
    data["CT"] = data.force_x / reference
    data["CP"] = data.rotational_power / (reference * speed)
    return data, period


def frame_files(collection: Path) -> list[tuple[float, Path]]:
    """Times and files of a ParaView collection."""
    entries = ET.parse(collection).getroot().iter("DataSet")
    return [
        (float(entry.get("timestep")), collection.parent / entry.get("file")) for entry in entries
    ]


def camera(bounds: np.ndarray, distance: float, shift: float, drop: float) -> dict[str, np.ndarray]:
    """A fixed view from upstream of one side, with the rotor near the left of the frame."""
    lower, upper = bounds[::2], bounds[1::2]
    centre = 0.5 * (lower + upper)
    length = float(np.max(upper - lower))
    toward_camera = np.asarray(VIEW_DIRECTION, dtype=float)
    toward_camera /= np.linalg.norm(toward_camera)
    view_up = np.array([0.0, 0.0, 1.0])
    screen_right = np.cross(-toward_camera, view_up)
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(screen_right, -toward_camera)
    # The rotor near the left edge and the wake rising across the frame, clear of the inset
    focal_point = centre + length * (shift * screen_right - drop * screen_up)
    return {
        "focal_point": focal_point,
        "position": focal_point + distance * length * toward_camera,
        "view_up": view_up,
    }


def render_scene(plotter, particles, blades, view, omega_max, discs=False):
    """Render one frame of particles and blades, returning its RGB image; the video draws the
    particles as discs."""
    plotter.clear()
    # The rotor starts from rest, before it has shed any particle
    if discs and particles.n_points:
        vorticity = np.linalg.norm(np.asarray(particles["vorticity"]), axis=1)
        plotter.add_actor(vorticity_discs(
            particles.points, vorticity, view["position"], radius=DISC_RADIUS, cut=DISC_CUT,
            opaque=DISC_OPAQUE, maximum=omega_max, colormap=VORTICITY_COLORMAP,
        ))
    elif particles.n_points:
        cloud = pv.PolyData(np.asarray(particles.points))
        cloud["vorticity magnitude"] = np.linalg.norm(np.asarray(particles["vorticity"]), axis=1)
        plotter.add_mesh(
            cloud,
            scalars="vorticity magnitude",
            cmap=VORTICITY_COLORMAP,
            clim=(0.0, omega_max),
            # Weak vorticity fades out, so the helical tip vortices stand out
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


def compose(image, time, coefficients, period, omega_max, layout):
    """Lay the rendered scene, the colour bar and the force inset on one frame in a format."""
    size, text = layout["size"], layout["text"]
    with plt.rc_context({**GALLERY_STYLE, "font.size": GALLERY_STYLE["font.size"] * text}):
        figure = plt.figure(figsize=(5.0 * size[0] / size[1], 5.0), dpi=size[1] / 5.0)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        # Vorticity magnitude, top left: its two end values, the title above the bar
        bar = figure.add_axes(layout["bar"])
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(0.0, omega_max), VORTICITY_COLORMAP),
            cax=bar,
            ticks=(0.0, omega_max),
        )
        bar.set_yticklabels(["0", f"{omega_max:g}"], fontsize=24 * text)
        bar.tick_params(direction="out", length=10 * text, width=1.5)
        bar.set_title(r"$|\omega|$ [1/s]", fontsize=24 * text, pad=14 * text, loc="left")

        # Thrust and power up to this time; the rest of the history stays faint
        inset = figure.add_axes(layout["inset"])
        inset.set_facecolor("none")
        times = coefficients["time"].to_numpy()
        revolutions = times / period
        now = times <= time + 1e-9
        for name, label, colour in (
            ("CT", "$C_T$", "#f39c1f"),
            ("CP", "$C_P$", "#9bd75a"),
        ):
            values = coefficients[name].to_numpy()
            inset.plot(revolutions, values, color=colour, alpha=0.25, linewidth=1.5)
            inset.plot(revolutions[now], values[now], color=colour, linewidth=2.5, label=label)
            if np.any(now):
                inset.plot(
                    revolutions[now][-1],
                    values[now][-1],
                    "o",
                    color=colour,
                    markeredgecolor="white",
                    markersize=8 * text**0.5,
                    clip_on=False,
                )
        inset.set_xlim(0.0, 10.0)
        inset.set_ylim(0.4, 1.0)
        inset.set_xticks((0, 5, 10))
        inset.set_yticks((0.5, 0.75, 1.0))
        inset.set_xlabel("revolutions")
        inset.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.98),
            ncols=2,
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("format", nargs="?", choices=tuple(FORMATS), default="gif")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    layout = FORMATS[args.format]
    size = layout["size"]
    render_scale = 1 if layout["discs"] else RENDER_SCALE

    particles = frame_files(SOLUTION_DIR / "vpm.pvd")[:: layout["stride"]]
    blades = frame_files(SOLUTION_DIR / "vlm.pvd")[:: layout["stride"]]
    coefficients, period = performance()

    # One view for every frame, set by the final wake
    bounds = np.array(pv.read(particles[-1][1]).bounds)
    view = camera(bounds, layout["distance"], layout["shift"], layout["drop"])

    plotter = pv.Plotter(off_screen=True, window_size=[render_scale * pixels for pixels in size])
    plotter.set_background("black")
    if layout["discs"]:
        plotter.enable_anti_aliasing("fxaa")
        PRESENTATION_FRAMES.mkdir(parents=True, exist_ok=True)
        for old in PRESENTATION_FRAMES.glob("frame_*.png"):
            old.unlink()
    frames = []
    for (time, particle_file), (_, blade_file) in zip(particles, blades, strict=True):
        particle_data, blade_data = pv.read(particle_file), pv.read(blade_file)
        image = render_scene(plotter, particle_data, blade_data, view, OMEGA_MAX, layout["discs"])
        frames.append(compose(image, time, coefficients, period, OMEGA_MAX, layout))
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
