"""Animate the vortex ring of the default variant as it travels and its instability grows.

Each frame shows the particles at one output time as spheres sized and
coloured by their strength relative to the largest initial strength. The camera
follows the ring; a ruler along its path marks the distance travelled in ring
radii. The animation holds the final ring, then fades back to the first frame.
The result is ``assets/vortex_ring.gif``, shown in the README gallery.

python -m assets.render_vortex_ring_animation               the README gallery GIF, 800 x 500
python -m assets.render_vortex_ring_animation presentation  a 1920 x 1080 H.264 video for slides,
                                                            in docs/assets/presentation/, one frame
                                                            per saved state; its frames are kept in
                                                            solution/dns_transposed/
                                                            presentation_frames/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyvista as pv

from openonda.plotting import (
    PRESENTATION_DIR,
    PRESENTATION_SIZE,
    PRESENTATION_TEXT_SCALE,
    vorticity_discs,
    write_presentation_video,
)

plt.switch_backend("Agg")

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution" / "dns_transposed"
OUTPUT = CASE_DIR / "assets" / "vortex_ring.gif"
PRESENTATION = PRESENTATION_DIR / "vortex_ring.mp4"
PRESENTATION_FRAMES = SOLUTION_DIR / "presentation_frames"

RING_RADIUS = 1.0  # [m], R0
STRENGTH_RANGE = (0.04, 0.8)  # |alpha_p| / max |alpha_q,0|, the colour scale
SPHERE_RADIUS = 0.025  # [R0], drawn radius of a particle at the largest initial strength
RULER_OFFSET = (0.0, -1.6)  # [R0], y and z of the ruler below the ring path
TICK_LENGTH = 0.25  # [R0]
LABEL_EVERY = 5  # ruler ticks every R0, labelled every five

IMAGE_SIZE = (800, 500)  # [px], as the other gallery animations
VIEW_DIRECTION = (-0.6, -1.0, 0.35)  # toward the camera, from upstream of the ring
VIEW_DISTANCE = 6.0  # [R0]
FADE_FRAMES = 10  # frames of the fade from the final ring back to the first
# Each frame is a figure 5 in tall at height / 5 dpi; `text` scales the gallery font sizes for a
# 1080p slide. The GIF scene is drawn at twice its size and reduced, the video scene at its own.
FORMATS = {
    "gif": dict(size=IMAGE_SIZE, render_scale=2, text=1.0, discs=False,
                bar=(0.04, 0.53, 0.04, 0.29)),
    "presentation": dict(size=PRESENTATION_SIZE, render_scale=1, text=PRESENTATION_TEXT_SCALE,
                         discs=True, bar=(0.04, 0.58, 0.016, 0.27)),
}
VIDEO_FRAMES_PER_SECOND = 10  # video: saved states per second
# Video: the particles as discs coloured by relative strength (openonda.plotting), transparent
# below DISC_CUT and opaque from DISC_OPAQUE, so that the many weak particles close the tube.
DISC_RADIUS = 0.8 * 0.035  # [R0], 0.8 of the particle spacing
DISC_CUT, DISC_OPAQUE = 0.12, 0.25  # |alpha_p| / max |alpha_q,0|

# README gallery style: white DejaVu Serif text on black
GALLERY_STYLE = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 22,
    "text.color": "white",
    "axes.edgecolor": "white",
    "axes.linewidth": 1.5,
    "ytick.color": "white",
    "ytick.direction": "out",
    "ytick.major.width": 1.5,
}
RULER_COLOUR = "#b4b4b4"
# Sequential blue scale, dark for weak particles
STRENGTH_COLORMAP = LinearSegmentedColormap.from_list(
    "strength_blue", ["#0D3D45", "#156074", "#41A6C4", "#C8EDF5"], N=256
)


def frame_files(collection: Path) -> list[tuple[float, Path]]:
    """Times and files of a ParaView collection."""
    entries = ET.parse(collection).getroot().iter("DataSet")
    return [
        (float(entry.get("timestep")), collection.parent / entry.get("file")) for entry in entries
    ]


def particle_spheres(particles, strength_reference) -> pv.PolyData:
    """Particle spheres, sized and coloured by relative strength."""
    relative = np.linalg.norm(np.asarray(particles["vortex_strength"]), axis=1) / strength_reference
    cloud = pv.PolyData(np.asarray(particles.points))
    cloud["relative strength"] = relative
    cloud["radius"] = SPHERE_RADIUS * RING_RADIUS * relative**0.65
    sphere = pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12)
    return cloud.glyph(geom=sphere, scale="radius", orient=False)


def particle_discs(particles, strength_reference, camera_position):
    """The particles as discs coloured by relative strength over STRENGTH_RANGE."""
    relative = np.linalg.norm(np.asarray(particles["vortex_strength"]), axis=1) / strength_reference
    low, high = STRENGTH_RANGE
    return vorticity_discs(particles.points, relative - low, camera_position,
                           radius=DISC_RADIUS * RING_RADIUS, cut=DISC_CUT - low,
                           opaque=DISC_OPAQUE - low, maximum=high - low,
                           colormap=STRENGTH_COLORMAP, steps=256)


def ruler(x_start: float, x_end: float) -> tuple[pv.PolyData, list[float]]:
    """The ruler line with a tick every R0, and the positions of its labelled ticks."""
    y, z = (RING_RADIUS * offset for offset in RULER_OFFSET)
    ticks = np.arange(np.floor(x_start), np.ceil(x_end) + 1.0) * RING_RADIUS
    lines = [pv.Line((x_start, y, z), (x_end, y, z))]
    lines += [pv.Line((x, y, z), (x, y, z + TICK_LENGTH * RING_RADIUS)) for x in ticks]
    labelled = [float(x) for x in ticks if round(x / RING_RADIUS) % LABEL_EVERY == 0]
    return pv.merge(lines), labelled


def render_scene(plotter, particles, strength_reference, centre, path_length, layout):
    """Render the particles and the ruler, following the ring; return the image and label anchors.

    The GIF draws the particles as spheres, the video as discs.
    """
    plotter.clear()
    toward_camera = np.asarray(VIEW_DIRECTION, dtype=float)
    toward_camera /= np.linalg.norm(toward_camera)
    position = centre + VIEW_DISTANCE * RING_RADIUS * toward_camera
    if layout["discs"]:
        plotter.add_actor(particle_discs(particles, strength_reference, position))
    else:
        plotter.add_mesh(
            particle_spheres(particles, strength_reference),
            scalars="relative strength",
            cmap=STRENGTH_COLORMAP,
            clim=STRENGTH_RANGE,
            smooth_shading=True,
            specular=0.6,
            specular_power=30.0,
            show_scalar_bar=False,
        )
    size, render_scale = layout["size"], layout["render_scale"]
    line, labelled = ruler(-2.0 * RING_RADIUS, path_length + 6.0 * RING_RADIUS)
    line_width = 1.5 * render_scale * size[1] / IMAGE_SIZE[1]
    plotter.add_mesh(line, color=RULER_COLOUR, line_width=line_width)
    plotter.camera.focal_point = centre
    plotter.camera.position = position
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.reset_camera_clipping_range()
    plotter.render()

    # Screen positions of the labelled ticks, so their numbers are typeset with the other text
    y, z = (RING_RADIUS * offset for offset in RULER_OFFSET)
    anchors = []
    for x in labelled:
        plotter.renderer.SetWorldPoint(x, y, z, 1.0)
        plotter.renderer.WorldToDisplay()
        column, row, depth = plotter.renderer.GetDisplayPoint()
        # Only numbers that fit inside the frame
        inside = 0.03 < column / (render_scale * size[0]) < 0.97
        if inside and 0.0 < depth < 1.0:
            anchors.append((round(x / RING_RADIUS), column / render_scale, row / render_scale))
    return plotter.screenshot(return_img=True), anchors


def compose(image, anchors, layout):
    """Lay the rendered scene, the colour bar and the ruler labels on one frame in a format."""
    size, text = layout["size"], layout["text"]
    with plt.rc_context({**GALLERY_STYLE, "font.size": GALLERY_STYLE["font.size"] * text}):
        figure = plt.figure(figsize=(5.0 * size[0] / size[1], 5.0), dpi=size[1] / 5.0)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        # Relative strength: its two end values, the title above the bar
        bar = figure.add_axes(layout["bar"])
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(*STRENGTH_RANGE), STRENGTH_COLORMAP),
            cax=bar,
            ticks=STRENGTH_RANGE,
        )
        bar.set_yticklabels([f"{value:g}" for value in STRENGTH_RANGE], fontsize=30 * text)
        bar.tick_params(direction="out", length=10 * text, width=1.5)
        bar.set_title(
            r"$|\alpha_p|\,/\max\,|\alpha_{q,0}|$", fontsize=30 * text, pad=24 * text, loc="left"
        )

        # Ruler numbers below their ticks, in the size and colour of the ruler caption
        for value, column, row in anchors:
            figure.text(
                column / size[0],
                row / size[1] - 0.015,
                f"{value}",
                color=RULER_COLOUR,
                ha="center",
                va="top",
            )
        figure.text(0.97, 0.04, r"ruler: $x/R_0$", color=RULER_COLOUR, ha="right", va="bottom")

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

    files = frame_files(SOLUTION_DIR / "vpm.pvd")
    first = pv.read(files[0][1])
    strength_reference = float(np.linalg.norm(np.asarray(first["vortex_strength"]), axis=1).max())
    path_length = float(np.mean(np.asarray(pv.read(files[-1][1]).points)[:, 0]))

    plotter = pv.Plotter(off_screen=True, window_size=[render_scale * pixels for pixels in size])
    plotter.set_background("black")
    if layout["discs"]:
        plotter.enable_anti_aliasing("fxaa")
        PRESENTATION_FRAMES.mkdir(parents=True, exist_ok=True)
        for old in PRESENTATION_FRAMES.glob("frame_*.png"):
            old.unlink()
    frames = []
    for _, path in files:
        particles = pv.read(path)
        centre = np.asarray(particles.points).mean(axis=0)
        image, anchors = render_scene(
            plotter, particles, strength_reference, centre, path_length, layout
        )
        frames.append(compose(image, anchors, layout))
        if layout["discs"]:
            frames[-1].save(PRESENTATION_FRAMES / f"frame_{len(frames) - 1:04d}.png")
    plotter.close()

    if layout["discs"]:
        write_presentation_video(
            frames, PRESENTATION, VIDEO_FRAMES_PER_SECOND, frame_dir=PRESENTATION_FRAMES
        )
        print(f"  Saved: {PRESENTATION} and {len(frames)} frames in {PRESENTATION_FRAMES}")
        return

    # Fade from the final ring back to the first frame, so the animation repeats smoothly
    fade = [
        Image.blend(frames[-1], frames[0], step / (FADE_FRAMES + 1))
        for step in range(1, FADE_FRAMES + 1)
    ]
    frames += fade

    # One palette for every frame, from four frames across the animation, keeps the GIF small
    sample = Image.new("RGB", (IMAGE_SIZE[0], 4 * IMAGE_SIZE[1]))
    for row, frame in enumerate(frames[len(frames) // 4 :: len(frames) // 4][:4]):
        sample.paste(frame, (0, row * IMAGE_SIZE[1]))
    palette = sample.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]

    # Hold the final ring briefly before the fade
    durations = [round(1000 / args.fps)] * len(frames)
    durations[len(files) - 1] = 350
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
