"""Animate the developed cylinder wake over four shedding periods.

python -m assets.render_cylinder_animation               the README gallery GIF, 800 x 500
python -m assets.render_cylinder_animation presentation  a 1920 x 1080 H.264 video for slides, in
                                                         docs/assets/presentation/, one frame per
                                                         saved state; its frames are kept in
                                                         solution/presentation_frames/ of the
                                                         baseline run
"""

from pathlib import Path
import sys
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

CASE_DIR = Path(__file__).resolve().parents[1]
BASELINE_DIR = CASE_DIR / "solution/sensitivity_analysis/box2.4_sub5_inset8.75"  # baseline run
SOLUTION_DIR = BASELINE_DIR / "solution"
FORCES = BASELINE_DIR / "samples" / "forces_history.csv"
OUTPUT = CASE_DIR / "figures" / "cylinder_shedding.gif"
PRESENTATION = PRESENTATION_DIR / "cylinder_shedding.mp4"
PRESENTATION_FRAMES = SOLUTION_DIR / "presentation_frames"
FRAMES_PER_SECOND = 20.0  # GIF: saved states per second
VIDEO_FRAMES_PER_SECOND = 10  # video: saved states per second

DIAMETER = 1.0  # [m]
FREESTREAM_SPEED = 1.0  # [m/s]
FVM_DOMAIN = (-2.4, 4.4, -2.4, 2.4)  # [m], x and y bounds
START_TIME = 75.6  # [s], an upward zero crossing of the lift in developed shedding
SHEDDING_PERIODS = 4
SHEDDING_PERIOD = 5.488  # [s], between upward zero crossings of the lift
FVM_HEIGHT = 1.0  # m, drawn span of the planar FVM domain

IMAGE_SIZE = (800, 500)  # px
RENDER_SCALE = 2  # GIF: render at twice the size, then downsample smoothly
OMEGA_MAX = 1.5  # 1/s

# Oblique view from above and downstream, toward_camera pointing to the camera. Each frame is a
# figure 5 in tall at height / 5 dpi; `text` scales the gallery font sizes for a 1080p slide, whose
# scene is drawn at its own size. The video moves the FVM box to the left, the wake across the
# 16:9 picture.
FORMATS = {
    "gif": dict(size=IMAGE_SIZE, text=1.0,
                bar=(0.80, 0.53, 0.04, 0.29), inset=(0.12, 0.17, 0.29, 0.24),
                view=dict(toward_camera=(0.9, -0.9, 1.2), focal_point=(6.5, 0.2, 0.0),
                          distance=36.0, view_angle=17.0)),
    "presentation": dict(size=PRESENTATION_SIZE, text=PRESENTATION_TEXT_SCALE,
                         bar=(0.90, 0.58, 0.016, 0.27), inset=(0.07, 0.13, 0.22, 0.16),
                         view=dict(toward_camera=(0.9, -0.9, 1.2), focal_point=(7.2, 0.9, 0.0),
                                   distance=36.0, view_angle=17.0)),
}
# Video: the FVM cells and the particles outside the FVM domain as vorticity discs
# (openonda.plotting), transparent below DISC_CUT and opaque from DISC_OPAQUE.
DISC_RADIUS = 0.8 * 0.04  # [m], 0.8 of the cell and particle spacing
DISC_CUT, DISC_OPAQUE = 0.3, 1.2  # [1/s]

# Dark gallery style and blue vorticity scale.
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


def disc_actor(fvm_file: Path, particle_file: Path, view):
    """The FVM cells and the particles outside the FVM domain as vorticity discs."""
    fvm = pv.read(fvm_file)
    cloud = wake_particles(particle_file)
    points = np.vstack([np.asarray(fvm.cell_centers().points), np.asarray(cloud.points)])
    vorticity = np.concatenate([
        np.linalg.norm(np.asarray(fvm.cell_data["vorticity"]), axis=1),
        np.asarray(cloud["vorticity magnitude"]),
    ])
    return vorticity_discs(points, vorticity, view["position"], radius=DISC_RADIUS, cut=DISC_CUT,
                           opaque=DISC_OPAQUE, maximum=OMEGA_MAX, colormap=VORTICITY_COLORMAP)


def render_scene(plotter, plane, cloud, view, discs=None):
    """Render one frame of FVM plane, particles and cylinder, returning its RGB image; the video
    draws the discs of disc_actor instead of the plane and the particle spheres."""
    plotter.clear()
    # Use one vorticity scale for FVM and particles.
    vorticity = {
        "scalars": "vorticity magnitude",
        "cmap": VORTICITY_COLORMAP,
        "clim": (0.0, OMEGA_MAX),
        "opacity": np.linspace(0.0, 1.0, 16),
        "show_scalar_bar": False,
    }
    if discs is not None:
        plotter.add_actor(discs)
    else:
        plotter.add_mesh(plane, **vorticity)
        plotter.add_mesh(
            cloud, render_points_as_spheres=True, point_size=3.0 * RENDER_SCALE, **vorticity
        )
    # Draw the FVM-domain edges and cylinder.
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


def compose(image, time, forces, layout):
    """Lay the rendered scene, the colour bars and the force inset on one frame in a format."""
    size, text = layout["size"], layout["text"]
    with plt.rc_context({**GALLERY_STYLE, "font.size": GALLERY_STYLE["font.size"] * text}):
        figure = plt.figure(figsize=(5.0 * size[0] / size[1], 5.0), dpi=size[1] / 5.0)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        # Common vorticity scale for FVM and particle fields.
        bar = figure.add_axes(layout["bar"])
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(0.0, OMEGA_MAX), VORTICITY_COLORMAP),
            cax=bar,
            ticks=(0.0, OMEGA_MAX),
        )
        bar.set_yticklabels(["0", f"{OMEGA_MAX:g}"], fontsize=30 * text)
        bar.tick_params(direction="out", length=10 * text, width=1.5)
        bar.set_title(r"$|\omega|$ [1/s]", fontsize=30 * text, pad=28 * text)

        # Force history up to the current frame.
        inset = figure.add_axes(layout["inset"])
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
                markersize=8 * text**0.5,
                clip_on=False,
            )
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
    presentation = "presentation" in sys.argv[1:]
    layout = FORMATS["presentation" if presentation else "gif"]
    # Four shedding periods return the wake to its starting phase.
    end_time = START_TIME + SHEDDING_PERIODS * SHEDDING_PERIOD - 0.1
    particles = frame_files(SOLUTION_DIR / "vpm.pvd", START_TIME, end_time)
    planes = frame_files(SOLUTION_DIR / "fvm.pvd", START_TIME, end_time)
    forces = pd.read_csv(FORCES)
    forces = forces[(forces["time"] >= START_TIME - 1e-9) & (forces["time"] <= end_time + 1e-9)]

    view = camera(**layout["view"])
    render_scale = 1 if presentation else RENDER_SCALE
    window_size = [render_scale * pixels for pixels in layout["size"]]
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("black")
    if presentation:
        plotter.enable_anti_aliasing("fxaa")
        PRESENTATION_FRAMES.mkdir(parents=True, exist_ok=True)
        for old in PRESENTATION_FRAMES.glob("frame_*.png"):
            old.unlink()
    else:
        plotter.enable_depth_peeling()
    frames = []
    for (time, particle_file), (_, plane_file) in zip(particles, planes, strict=True):
        if presentation:
            discs = disc_actor(plane_file, particle_file, view)
            image = render_scene(plotter, None, None, view, discs)
        else:
            plane, cloud = fvm_plane(plane_file), wake_particles(particle_file)
            image = render_scene(plotter, plane, cloud, view)
        frames.append(compose(image, time, forces, layout))
        if presentation:
            frames[-1].save(PRESENTATION_FRAMES / f"frame_{len(frames) - 1:04d}.png")
    plotter.close()

    if presentation:
        write_presentation_video(
            frames, PRESENTATION, VIDEO_FRAMES_PER_SECOND, frame_dir=PRESENTATION_FRAMES
        )
        print(f"  Saved: {PRESENTATION} and {len(frames)} frames in {PRESENTATION_FRAMES}")
        return

    # Reuse one palette for a compact GIF.
    sample = Image.new("RGB", (IMAGE_SIZE[0], 4 * IMAGE_SIZE[1]))
    for row, frame in enumerate(frames[:: len(frames) // 4][:4]):
        sample.paste(frame, (0, row * IMAGE_SIZE[1]))
    palette = sample.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / FRAMES_PER_SECOND),
        loop=0,
    )
    print(f"  Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
