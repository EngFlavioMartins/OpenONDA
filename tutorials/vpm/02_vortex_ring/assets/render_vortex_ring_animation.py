"""Animate the vortex ring of the default variant as it travels and its instability grows.

Each frame shows the particles at one output time as spheres sized and
coloured by their strength relative to the largest initial strength. The camera
follows the ring; a ruler along its path marks the distance travelled in ring
radii. The animation holds the final ring, then fades back to the first frame.
``assets/vortex_ring.gif``, shown in the README gallery, is a copy of the
result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from matplotlib import colormaps
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyvista as pv

import openonda.plotting  # noqa: F401  registers the thesis_blue scale

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution" / "dns_transposed"
OUTPUT = CASE_DIR / "figures" / "vortex_ring.gif"

RING_RADIUS = 1.0  # [m], R0
STRENGTH_RANGE = (0.04, 0.8)  # |alpha_p| / max |alpha_q,0|, the colour scale
SPHERE_RADIUS = 0.025  # [R0], drawn radius of a particle at the largest initial strength
RULER_OFFSET = (0.0, -1.6)  # [R0], y and z of the ruler below the ring path
TICK_LENGTH = 0.25  # [R0]
LABEL_EVERY = 5  # ruler ticks every R0, labelled every five

IMAGE_SIZE = (800, 500)  # [px], as the other gallery animations
RENDER_SCALE = 2  # render at twice the size, then downsample smoothly
VIEW_DIRECTION = (-0.6, -1.0, 0.35)  # toward the camera, from upstream of the ring
VIEW_DISTANCE = 6.0  # [R0]
FADE_FRAMES = 10  # frames of the fade from the final ring back to the first

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
STRENGTH_COLORMAP = colormaps["thesis_blue"]


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


def ruler(x_start: float, x_end: float) -> tuple[pv.PolyData, list[float]]:
    """The ruler line with a tick every R0, and the positions of its labelled ticks."""
    y, z = (RING_RADIUS * offset for offset in RULER_OFFSET)
    ticks = np.arange(np.floor(x_start), np.ceil(x_end) + 1.0) * RING_RADIUS
    lines = [pv.Line((x_start, y, z), (x_end, y, z))]
    lines += [pv.Line((x, y, z), (x, y, z + TICK_LENGTH * RING_RADIUS)) for x in ticks]
    labelled = [float(x) for x in ticks if round(x / RING_RADIUS) % LABEL_EVERY == 0]
    return pv.merge(lines), labelled


def render_scene(plotter, spheres, centre, path_length):
    """Render the particles and the ruler, following the ring; return the image and label anchors."""
    plotter.clear()
    plotter.add_mesh(
        spheres,
        scalars="relative strength",
        cmap=STRENGTH_COLORMAP,
        clim=STRENGTH_RANGE,
        smooth_shading=True,
        specular=0.6,
        specular_power=30.0,
        show_scalar_bar=False,
    )
    line, labelled = ruler(-2.0 * RING_RADIUS, path_length + 6.0 * RING_RADIUS)
    plotter.add_mesh(line, color=RULER_COLOUR, line_width=1.5 * RENDER_SCALE)
    toward_camera = np.asarray(VIEW_DIRECTION, dtype=float)
    toward_camera /= np.linalg.norm(toward_camera)
    plotter.camera.focal_point = centre
    plotter.camera.position = centre + VIEW_DISTANCE * RING_RADIUS * toward_camera
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
        inside = 0.03 < column / (RENDER_SCALE * IMAGE_SIZE[0]) < 0.97
        if inside and 0.0 < depth < 1.0:
            anchors.append((round(x / RING_RADIUS), column / RENDER_SCALE, row / RENDER_SCALE))
    return plotter.screenshot(return_img=True), anchors


def compose(image, anchors):
    """Lay the rendered scene, the colour bar and the ruler labels on one frame."""
    with plt.rc_context(GALLERY_STYLE):
        figure = plt.figure(figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100), dpi=100)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        # Relative strength: its two end values, the title above the bar
        bar = figure.add_axes((0.04, 0.53, 0.04, 0.29))
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(*STRENGTH_RANGE), STRENGTH_COLORMAP),
            cax=bar,
            ticks=STRENGTH_RANGE,
        )
        bar.set_yticklabels([f"{value:g}" for value in STRENGTH_RANGE], fontsize=30)
        bar.tick_params(direction="out", length=10, width=1.5)
        bar.set_title(r"$|\alpha_p|\,/\max\,|\alpha_{q,0}|$", fontsize=30, pad=24, loc="left")

        # Ruler numbers below their ticks, in the size and colour of the ruler caption
        for value, column, row in anchors:
            figure.text(
                column / IMAGE_SIZE[0],
                row / IMAGE_SIZE[1] - 0.015,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    files = frame_files(SOLUTION_DIR / "vpm.pvd")
    first = pv.read(files[0][1])
    strength_reference = float(np.linalg.norm(np.asarray(first["vortex_strength"]), axis=1).max())
    path_length = float(np.mean(np.asarray(pv.read(files[-1][1]).points)[:, 0]))

    plotter = pv.Plotter(off_screen=True, window_size=[RENDER_SCALE * size for size in IMAGE_SIZE])
    plotter.set_background("black")
    frames = []
    for _, path in files:
        particles = pv.read(path)
        centre = np.asarray(particles.points).mean(axis=0)
        image, anchors = render_scene(
            plotter, particle_spheres(particles, strength_reference), centre, path_length
        )
        frames.append(compose(image, anchors))
    plotter.close()

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
