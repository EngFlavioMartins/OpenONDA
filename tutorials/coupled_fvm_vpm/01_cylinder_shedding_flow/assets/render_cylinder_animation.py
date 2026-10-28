"""Animate the coupled cylinder wake over four shedding periods, with its lift and drag.

Each frame shows the FVM vorticity on the mid-plane of the FVM domain and the
particles outside it, on one vorticity-magnitude scale, at one field output
time of developed shedding. Fields are written every 0.2 s from 40 s on; the
frames span four shedding periods, so the animation repeats without a jump.
The inset traces the lift and drag coefficients up to that time.
``assets/cylinder_shedding.gif``, shown in the README gallery, is a copy of the
result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pyvista as pv

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution"
FORCES = CASE_DIR / "samples" / "forces_history.csv"
OUTPUT = CASE_DIR / "figures" / "cylinder_shedding.gif"

DIAMETER = 1.0  # [m]
FREESTREAM_SPEED = 1.0  # [m/s]
FVM_DOMAIN = (-2.4, 4.4, -2.4, 2.4)  # [m], x and y bounds
START_TIME = 75.6  # [s], an upward zero crossing of the lift in developed shedding
SHEDDING_PERIODS = 4
SHEDDING_PERIOD = 5.488  # [s], between upward zero crossings of the lift
FVM_HEIGHT = 1.0  # [m], drawn height of the FVM domain and cylinder; the span is one cell

# An oblique view from above and downstream: FVM domain top left, wake toward bottom right
VIEW_DIRECTION = (0.9, -0.9, 1.2)  # toward the camera
FOCAL_POINT = (6.5, 0.2, 0.0)  # [m]
VIEW_DISTANCE = 36.0  # [m]
VIEW_ANGLE = 17.0  # [deg], a long lens keeps the far wake from looming

IMAGE_SIZE = (800, 500)  # [px], as the other gallery animations
RENDER_SCALE = 2  # render at twice the size, then downsample smoothly
OMEGA_MAX = 1.5  # [1/s], vorticity magnitude of the shed vortices in the particle wake

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


def frame_files(collection: Path, start: float, end: float) -> list[tuple[float, Path]]:
    """Times and files of a ParaView collection from start to end."""
    entries = ET.parse(collection).getroot().iter("DataSet")
    frames = [
        (float(entry.get("timestep")), collection.parent / entry.get("file")) for entry in entries
    ]
    return [(time, path) for time, path in frames if start - 1e-6 <= time <= end + 1e-6]


def camera(toward_camera, focal_point, distance, view_angle) -> dict[str, np.ndarray]:
    """A fixed oblique view from above the plane of the flow."""
    toward_camera = np.asarray(toward_camera, dtype=float)
    toward_camera /= np.linalg.norm(toward_camera)
    focal_point = np.asarray(focal_point, dtype=float)
    return {
        "focal_point": focal_point,
        "position": focal_point + distance * toward_camera,
        "view_up": np.array([0.0, 0.0, 1.0]),
        "view_angle": view_angle,
    }


def fvm_plane(path: Path) -> pv.PolyData:
    """FVM vorticity magnitude on the mid-plane of the one-cell span, at the mesh points."""
    plane = pv.read(path).slice(normal="z", origin=(0.0, 0.0, 0.0)).cell_data_to_point_data()
    plane["vorticity magnitude"] = np.linalg.norm(np.asarray(plane["vorticity"]), axis=1)
    return plane


def wake_particles(path: Path) -> pv.PolyData:
    """Particles outside the FVM domain with their vorticity magnitude."""
    particles = pv.read(path)
    x, y = np.asarray(particles.points)[:, 0], np.asarray(particles.points)[:, 1]
    xmin, xmax, ymin, ymax = FVM_DOMAIN
    outside = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)
    cloud = pv.PolyData(np.asarray(particles.points)[outside])
    cloud["vorticity magnitude"] = np.linalg.norm(
        np.asarray(particles["vorticity"])[outside], axis=1
    )
    return cloud


def render_scene(plotter, plane, cloud, view):
    """Render one frame of FVM plane, particles and cylinder, returning its RGB image."""
    plotter.clear()
    # FVM and particle vorticity share one scale; weak vorticity fades out, so the street is continuous
    vorticity = {
        "scalars": "vorticity magnitude",
        "cmap": VORTICITY_COLORMAP,
        "clim": (0.0, OMEGA_MAX),
        "opacity": np.linspace(0.0, 1.0, 16),
        "show_scalar_bar": False,
    }
    plotter.add_mesh(plane, **vorticity)
    plotter.add_mesh(
        cloud, render_points_as_spheres=True, point_size=3.0 * RENDER_SCALE, **vorticity
    )
    # FVM domain edges, over the drawn height of the planar flow
    xmin, xmax, ymin, ymax = FVM_DOMAIN
    box = pv.Box(bounds=(xmin, xmax, ymin, ymax, -0.5 * FVM_HEIGHT, 0.5 * FVM_HEIGHT))
    plotter.add_mesh(
        box.extract_feature_edges().tube(radius=0.04), color="#d9d9d9", smooth_shading=True
    )
    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=0.5 * DIAMETER,
        height=FVM_HEIGHT,
        resolution=120,
    )
    plotter.add_mesh(cylinder, color="#9c9c9c", smooth_shading=True, specular=0.3)
    plotter.camera.position = view["position"]
    plotter.camera.focal_point = view["focal_point"]
    plotter.camera.up = view["view_up"]
    plotter.camera.view_angle = view["view_angle"]
    plotter.reset_camera_clipping_range()
    plotter.render()
    return plotter.screenshot(return_img=True)


def compose(image, time, forces):
    """Lay the rendered scene, the colour bars and the force inset on one frame."""
    with plt.rc_context(GALLERY_STYLE):
        figure = plt.figure(figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100), dpi=100)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        # Vorticity magnitude of the FVM field and the particles: end values, title above the bar
        bar = figure.add_axes((0.80, 0.53, 0.04, 0.29))
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(0.0, OMEGA_MAX), VORTICITY_COLORMAP),
            cax=bar,
            ticks=(0.0, OMEGA_MAX),
        )
        bar.set_yticklabels(["0", f"{OMEGA_MAX:g}"], fontsize=30)
        bar.tick_params(direction="out", length=10, width=1.5)
        bar.set_title(r"$|\omega|$ [1/s]", fontsize=30, pad=28)

        # Lift and drag up to this time; the rest of the history stays faint
        inset = figure.add_axes((0.12, 0.17, 0.29, 0.24))
        inset.set_facecolor("none")
        times = forces["time"].to_numpy()
        travel = times * FREESTREAM_SPEED / DIAMETER
        now = times <= time + 1e-9
        for name, label, colour in (
            ("lift_coefficient", "$C_L$", "#f39c1f"),
            ("drag_coefficient", "$C_D$", "#9bd75a"),
        ):
            values = forces[name].to_numpy()
            inset.plot(travel, values, color=colour, alpha=0.25, linewidth=1.5)
            inset.plot(travel[now], values[now], color=colour, linewidth=2.5)
            inset.plot(
                travel[now][-1],
                values[now][-1],
                "o",
                color=colour,
                markeredgecolor="white",
                markersize=8,
                clip_on=False,
            )
            # Each curve is named at its right end, clear of the other
            inset.annotate(
                label,
                (1.0, values.mean()),
                xycoords=("axes fraction", "data"),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                color=colour,
            )
        inset.set_xlim(travel[0], travel[-1])
        inset.set_ylim(-0.6, 1.6)
        inset.set_xticks((80, 90))
        inset.set_yticks((-0.5, 0.5, 1.5))
        inset.set_xlabel(r"$t\,U_\infty/D$")
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

    # Four shedding periods of output times; the next one repeats the first
    end_time = START_TIME + SHEDDING_PERIODS * SHEDDING_PERIOD - 0.1
    particles = frame_files(SOLUTION_DIR / "vpm.pvd", START_TIME, end_time)
    planes = frame_files(SOLUTION_DIR / "fvm.pvd", START_TIME, end_time)
    forces = pd.read_csv(FORCES)
    forces = forces[(forces["time"] >= START_TIME - 1e-9) & (forces["time"] <= end_time + 1e-9)]

    view = camera(VIEW_DIRECTION, FOCAL_POINT, VIEW_DISTANCE, VIEW_ANGLE)
    plotter = pv.Plotter(off_screen=True, window_size=[RENDER_SCALE * size for size in IMAGE_SIZE])
    plotter.set_background("black")
    plotter.enable_depth_peeling()
    frames = []
    for (time, particle_file), (_, plane_file) in zip(particles, planes, strict=True):
        image = render_scene(plotter, fvm_plane(plane_file), wake_particles(particle_file), view)
        frames.append(compose(image, time, forces))
    plotter.close()

    # One palette for every frame, from four frames across the loop, keeps the GIF small
    sample = Image.new("RGB", (IMAGE_SIZE[0], 4 * IMAGE_SIZE[1]))
    for row, frame in enumerate(frames[:: len(frames) // 4][:4]):
        sample.paste(frame, (0, row * IMAGE_SIZE[1]))
    palette = sample.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]

    # The frames span whole shedding periods, so the animation loops without a pause
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
    )
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
