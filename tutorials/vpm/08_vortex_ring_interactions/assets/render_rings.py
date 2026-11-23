"""Render the particles of saved backups, coloured by the ring they descend from.

Ring 0 is drawn in blue and ring 1 in amber; regeneration on the lattice gives each
new particle the ring that holds most of the strength at its node, so the colours
follow how the rings exchange vorticity. Brightness and size grow with the particle
vorticity relative to the peak vorticity of the initial rings; particles below
``VORTICITY_RANGE[0]`` of it are not drawn.

``python -m assets.render_rings`` writes ``figures/<flow>_scenes.png`` (DNS, Smagorinsky LES
and spectral-cusp LES in rows at the same times) and ``assets/<flow>.gif`` (the three side by
side).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

from assets.turbulence_statistics import backups

CASE_DIR = Path(__file__).resolve().parents[1]
VORTICITY_RANGE = (0.06, 0.8)  # |w| / max |w| at t = 0: hidden below, brightest above
RING_COLOURS = (
    LinearSegmentedColormap.from_list("ring0", ["#0B2A4A", "#1D6FA3", "#6FC3E8", "#E4F6FF"]),
    LinearSegmentedColormap.from_list("ring1", ["#4A2306", "#B8620F", "#F2A93B", "#FFF0CF"]),
)
POINT_SIZE = (2.0, 6.0)  # [px] at the bottom and top of VORTICITY_RANGE, before supersampling
IMAGE_SIZE = (800, 600)  # [px] of one view
SUPERSAMPLE = 2
BACKGROUND = "#05070A"
SCENE_TIMES = {
    "collision": (0.0, 3.0, 5.0, 6.0, 7.5),
    "leapfrog": (0.0, 4.0, 8.0, 12.0, 14.5),
}
# Camera position relative to the focal point, in R_0, and the focal point rule
CAMERAS = {
    "collision": {"offset": (8.5, -5.9, 6.5), "focus": "origin", "view_up": (0.0, 0.0, 1.0)},
    "leapfrog": {"offset": (1.5, -7.0, 3.0), "focus": "centroid", "view_up": (0.0, 0.0, 1.0)},
}
FRAME_INTERVAL = 0.5  # [s] between animation frames
MODELS = ("dns", "les", "les_cusp")
LABELS = {"dns": "DNS", "les": "LES, Smagorinsky", "les_cusp": "LES, spectral cusp"}


def load(path):
    with h5py.File(path, "r") as handle:
        particles = handle["particles"]
        return (
            particles["position"][()],
            np.linalg.norm(particles["vorticity"][()], axis=1),
            particles["group_id"][()],
            float(handle["solver"].attrs["time"]),
        )


def colours(vorticity, group, reference):
    """RGBA per particle and the drawn mask."""
    low, high = VORTICITY_RANGE
    level = np.clip((vorticity / reference - low) / (high - low), 0.0, 1.0)
    rgba = np.empty((len(vorticity), 4))
    for ring, colormap in enumerate(RING_COLOURS):
        selected = group == ring
        rgba[selected] = colormap(0.15 + 0.85 * level[selected])
    return rgba, vorticity >= low * reference, level


def render(path, flow, reference, *, label=None):
    """One view of one backup as an RGB array."""
    position, vorticity, group, time = load(path)
    rgba, drawn, level = colours(vorticity, group, reference)
    width, height = (SUPERSAMPLE * n for n in IMAGE_SIZE)
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background(BACKGROUND)
    # Two passes by brightness so the strongest filaments are drawn as larger points
    for bright in (False, True):
        selected = drawn & ((level >= 0.5) == bright)
        if not np.any(selected):
            continue
        cloud = pv.PolyData(position[selected])
        cloud["rgba"] = (255 * rgba[selected]).astype(np.uint8)
        plotter.add_mesh(
            cloud,
            scalars="rgba",
            rgba=True,
            render_points_as_spheres=True,
            point_size=SUPERSAMPLE * POINT_SIZE[int(bright)],
            lighting=True,
            ambient=0.35,
            diffuse=0.75,
            specular=0.25,
        )
    camera = CAMERAS[flow]
    focus = (
        np.zeros(3)
        if camera["focus"] == "origin"
        else np.average(position, axis=0, weights=vorticity)
    )
    plotter.camera_position = [tuple(focus + camera["offset"]), tuple(focus), camera["view_up"]]
    plotter.camera.view_angle = 30.0
    image = plotter.screenshot(return_img=True)
    plotter.close()
    picture = Image.fromarray(image).resize(IMAGE_SIZE, Image.LANCZOS)
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype(plt.matplotlib.get_data_path() + "/fonts/ttf/DejaVuSerif.ttf", 26)
    text = f"tΓ₀/R₀² = {time:g}"
    draw.text((18, 14), text if label is None else f"{label}   {text}", fill="white", font=font)
    return np.asarray(picture)


def reference_vorticity(case):
    return float(load(backups(case)[0][1])[1].max())


def nearest_backup(case, time):
    timed = backups(case)
    index = int(np.argmin([abs(t - time) for t, _ in timed]))
    return timed[index] if abs(timed[index][0] - time) < 0.26 else None


def scene_figure(flow):
    """One row per run at the scene times; a run that stopped leaves the rest of its row dark."""
    rows = []
    for model in MODELS:
        case = f"{flow}_{model}"
        reference = reference_vorticity(case)
        tiles = []
        for time in SCENE_TIMES[flow]:
            found = nearest_backup(case, time)
            tile = (
                render(found[1], flow, reference, label=LABELS[model])
                if found is not None
                else np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 5, dtype=np.uint8)
            )
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    output = CASE_DIR / "figures" / f"{flow}_scenes.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(output)


def animation(flow):
    """The three runs side by side; a view holds its last state once its run stops."""
    cases = [f"{flow}_{model}" for model in MODELS]
    references = [reference_vorticity(case) for case in cases]
    end = max(backups(case)[-1][0] for case in cases)
    frames, last = [], [None] * len(cases)
    for time in np.arange(0.0, end + 1e-6, FRAME_INTERVAL):
        views = []
        for index, case in enumerate(cases):
            found = nearest_backup(case, time)
            if found is not None:
                last[index] = render(found[1], flow, references[index], label=LABELS[MODELS[index]])
            views.append(last[index])
        frames.append(Image.fromarray(np.concatenate(views, axis=1)).quantize(colors=255))
        print(f"{flow}: frame t = {time:.1f} s", flush=True)
    frames[0].save(
        CASE_DIR / "assets" / f"{flow}.gif",
        save_all=True,
        append_images=frames[1:] + [frames[-1]] * 8,
        duration=90,
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("flows", nargs="*", default=["collision", "leapfrog"])
    parser.add_argument("--no-animation", action="store_true")
    arguments = parser.parse_args()
    for name in arguments.flows:
        scene_figure(name)
        if not arguments.no_animation:
            animation(name)
