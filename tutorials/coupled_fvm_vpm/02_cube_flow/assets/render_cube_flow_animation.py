"""Animate the coupled cube wake and drag history."""

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
OUTPUT = CASE_DIR / "assets" / "cube_flow.gif"

CUBE_SIDE = 1.0  # [m]
FREESTREAM_SPEED = 1.0  # [m/s]
FVM_BOUNDS = (-1.8, 5.085, -1.8, 1.8, -1.8, 1.8)  # [m]
# ParaView's box clip removes this part of the VPM field to reveal the overlap.
VPM_CLIP_BOUNDS = (-1.0, 5.5, -2.5, -0.115, -0.05, 1.45)  # [m]
IMAGE_SIZE = (800, 500)  # [px], matching the README gallery
RENDER_SCALE = 2
FRAMES_PER_SECOND = 20
VORTICITY_MAX = 5.0  # [1/s]

# White DejaVu Serif text on black, with the README gallery's blue vorticity scale.
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


def frame_files(collection: Path) -> list[tuple[float, Path]]:
    """Return the dataset paths and times from a ParaView collection."""
    entries = ET.parse(collection).getroot().iter("DataSet")
    return [
        (float(entry.get("timestep")), collection.parent / entry.get("file"))
        for entry in entries
    ]


def fvm_plane(dataset: pv.DataSet) -> pv.PolyData:
    """Return mid-height FVM vorticity on the mesh points."""
    plane = dataset.slice(normal="z", origin=(0.0, 0.0, 0.0)).cell_data_to_point_data()
    plane["vorticity magnitude"] = np.linalg.norm(np.asarray(plane["vorticity"]), axis=1)
    return plane


def wake_particles(dataset: pv.DataSet) -> pv.PolyData:
    """Return VPM particles outside the ParaView overlap-visibility clip."""
    points = np.asarray(dataset.points)
    lower = np.asarray(VPM_CLIP_BOUNDS)[::2]
    upper = np.asarray(VPM_CLIP_BOUNDS)[1::2]
    outside_clip = np.any((points < lower) | (points > upper), axis=1)
    cloud = pv.PolyData(points[outside_clip])
    vorticity = np.asarray(dataset["vorticity"])[outside_clip]
    cloud["vorticity magnitude"] = np.linalg.norm(vorticity, axis=1)
    return cloud


def camera() -> dict[str, np.ndarray]:
    """Set an oblique view over the cube and the downstream wake."""
    return {
        "focal_point": np.array([9.9569245156, -0.8551330725, 0.5457414744]),
        "position": np.array([35.17181586, -28.61535302, 22.62267563]),
        "view_up": np.array([-0.35132246, 0.36616274, 0.86168287]),
        "view_angle": 15.0,
    }


def render_scene(plotter, plane, cloud, view):
    """Render the FVM plane, VPM wake and cube for one time."""
    plotter.clear()
    field = {
        "scalars": "vorticity magnitude",
        "cmap": VORTICITY_COLORMAP,
        "clim": (0.0, VORTICITY_MAX),
        "opacity": np.linspace(0.2, 1.0, 16),
        "show_scalar_bar": False,
    }
    plotter.add_mesh(plane, **field)
    if cloud.n_points:
        plotter.add_mesh(
            cloud,
            render_points_as_spheres=True,
            point_size=3.0 * RENDER_SCALE,
            **field,
        )

    fvm_box = pv.Box(bounds=FVM_BOUNDS)
    plotter.add_mesh(
        fvm_box.extract_feature_edges().tube(radius=0.025),
        color="#d9d9d9",
        smooth_shading=True,
    )
    cube = pv.Cube(
        bounds=(
            -0.5 * CUBE_SIDE,
            0.5 * CUBE_SIDE,
            -0.5 * CUBE_SIDE,
            0.5 * CUBE_SIDE,
            -0.5 * CUBE_SIDE,
            0.5 * CUBE_SIDE,
        )
    )
    plotter.add_mesh(cube, color="#9c9c9c", smooth_shading=True, specular=0.3)

    plotter.camera.position = view["position"]
    plotter.camera.focal_point = view["focal_point"]
    plotter.camera.up = view["view_up"]
    plotter.camera.view_angle = view["view_angle"]
    plotter.reset_camera_clipping_range()
    plotter.render()
    return plotter.screenshot(return_img=True)


def compose(image, time, forces):
    """Add the vorticity scale and drag history to a rendered frame."""
    with plt.rc_context(GALLERY_STYLE):
        figure = plt.figure(figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100), dpi=100)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        bar = figure.add_axes((0.89, 0.53, 0.035, 0.29))
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(0.0, VORTICITY_MAX), VORTICITY_COLORMAP),
            cax=bar,
            ticks=(0.0, VORTICITY_MAX),
        )
        bar.set_yticklabels(["0", f"{VORTICITY_MAX:g}"], fontsize=30)
        bar.tick_params(direction="out", length=10, width=1.5)
        bar.set_title(r"$|\omega|$ [1/s]", fontsize=30, pad=28, x=0.2)

        inset = figure.add_axes((0.12, 0.17, 0.29, 0.24))
        inset.set_facecolor("none")
        times = forces["time"].to_numpy()
        travel = times * FREESTREAM_SPEED / CUBE_SIDE
        now = times <= time + 1e-9
        drag = forces["drag_coefficient"].to_numpy()
        inset.plot(travel, drag, color="#9bd75a", alpha=0.25, linewidth=1.5)
        inset.plot(travel[now], drag[now], color="#9bd75a", linewidth=2.5)
        inset.plot(
            travel[now][-1],
            drag[now][-1],
            "o",
            color="#9bd75a",
            markeredgecolor="white",
            markersize=8,
            clip_on=False,
        )
        inset.annotate(
            "$C_D$",
            (1.0, drag.mean()),
            xycoords=("axes fraction", "data"),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color="#9bd75a",
        )
        inset.set_xlim(travel[0], travel[-1])
        inset.set_ylim(0.5, 2.5)
        inset.set_xticks((0, 20, 40))
        inset.set_yticks((0.5, 1.5, 2.5))
        inset.set_xlabel(r"$t\,U_\infty/D$")
        inset.spines[["top", "right"]].set_visible(False)

        figure.canvas.draw()
        frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        plt.close(figure)
    return frame


def main() -> None:
    particles = frame_files(SOLUTION_DIR / "vpm.pvd")
    fvm = frame_files(SOLUTION_DIR / "fvm.pvd")
    forces = pd.read_csv(FORCES)
    view = camera()

    plotter = pv.Plotter(
        off_screen=True,
        window_size=[RENDER_SCALE * size for size in IMAGE_SIZE],
    )
    plotter.set_background("black")
    frames = []
    for (time, particle_path), (_, fvm_path) in zip(particles, fvm, strict=True):
        fvm_data = pv.read(fvm_path)
        particle_data = pv.read(particle_path)
        plane = fvm_plane(fvm_data)
        cloud = wake_particles(particle_data)
        image = render_scene(plotter, plane, cloud, view)
        frame = compose(image, time, forces)
        frames.append(frame)
    plotter.close()

    sample = Image.new("RGB", (IMAGE_SIZE[0], 4 * IMAGE_SIZE[1]))
    sample_frames = (
        frames[0],
        frames[len(frames) // 3],
        frames[2 * len(frames) // 3],
        frames[-1],
    )
    for row, frame in enumerate(sample_frames):
        sample.paste(frame, (0, row * IMAGE_SIZE[1]))
    palette = sample.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]

    durations = [round(1000 / FRAMES_PER_SECOND)] * (len(frames) - 1) + [1000]
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
