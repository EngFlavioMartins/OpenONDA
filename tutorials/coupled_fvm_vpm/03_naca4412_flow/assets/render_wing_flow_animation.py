"""Animate the coupled NACA 4412 wake, seen from above, with the lift and drag histories."""

from pathlib import Path
import sys
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
WING = CASE_DIR / "assets" / "naca4412_wing.stl"
OUTPUT = CASE_DIR / "assets" / "wing_flow.gif"

CHORD = 1.0  # [m]
FREESTREAM_SPEED = 1.0  # [m/s]
FVM_BOUNDS = (-1.6, 2.56, -1.32, 1.24, -3.6, 3.6)  # [m]
# The particles inside the FVM box on the camera's side of the mid-span plane are left out, so
# that the FVM vorticity on that plane shows.
VPM_CLIP_BOUNDS = (-2.0, 3.0, -2.0, 2.0, 0.05, 4.0)  # [m]
IMAGE_SIZE = (800, 500)  # [px], matching the README gallery
RENDER_SCALE = 2
FRAMES_PER_SECOND = 12
VORTICITY_MAX = 3.0  # [1/s]
# Opacity of the 16 colour steps: the weak vorticity of the wake sheet, which fills a long volume,
# is left out so that the shed vortices show.
OPACITY = np.concatenate([np.zeros(3), np.linspace(0.25, 1.0, 13)])
FORCE_RANGE = (0.0, 0.6)  # inset axis of C_L and C_D; the impulsive start above it is cut
LIFT_COLOUR, DRAG_COLOUR = "#9bd75a", "#f4a261"

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


def frame_files(collection: Path) -> dict[float, Path]:
    """Return the dataset paths of a ParaView collection by time; a time written twice keeps the later file."""
    entries = ET.parse(collection).getroot().iter("DataSet")
    return {round(float(entry.get("timestep")), 6): collection.parent / entry.get("file") for entry in entries}


def fvm_plane(dataset: pv.DataSet) -> pv.PolyData:
    """Return the FVM vorticity on the mid-span plane z = 0, on the mesh points."""
    plane = dataset.slice(normal="z", origin=(0.0, 0.0, 0.0)).cell_data_to_point_data()
    plane["vorticity magnitude"] = np.linalg.norm(np.asarray(plane["vorticity"]), axis=1)
    return plane


def wake_particles(dataset: pv.DataSet) -> pv.PolyData:
    """Return the particles outside the box that reveals the mid-span plane."""
    points = np.asarray(dataset.points)
    lower = np.asarray(VPM_CLIP_BOUNDS)[::2]
    upper = np.asarray(VPM_CLIP_BOUNDS)[1::2]
    outside_clip = np.any((points < lower) | (points > upper), axis=1)
    cloud = pv.PolyData(points[outside_clip])
    vorticity = np.asarray(dataset["vorticity"])[outside_clip]
    cloud["vorticity magnitude"] = np.linalg.norm(vorticity, axis=1)
    return cloud


def camera() -> dict[str, np.ndarray]:
    """Set an oblique view from above the suction side, the span across the picture."""
    # The view of the cube animation turned so that y, the lift direction, points up: the wing lies
    # flat, seen from above its suction side and downstream, framed close on the FVM box and the
    # wake, with the box about 25 px clear of the left and top edges.
    focal_point = np.array([7.7, 0.6, 0.3])
    toward_camera = np.array([25.2, 22.1, 27.8])
    toward_camera /= np.linalg.norm(toward_camera)
    view_up = np.array([0.0, 1.0, 0.0]) - toward_camera[1] * toward_camera
    return {
        "focal_point": focal_point,
        "position": focal_point + 43.5 * toward_camera,
        "view_up": view_up / np.linalg.norm(view_up),
        "view_angle": 15.0,
    }


def render_scene(plotter, plane, cloud, wing, view):
    """Render the FVM plane, VPM wake and wing for one time."""
    plotter.clear()
    field = {
        "scalars": "vorticity magnitude",
        "cmap": VORTICITY_COLORMAP,
        "clim": (0.0, VORTICITY_MAX),
        "opacity": OPACITY,
        "show_scalar_bar": False,
    }
    plotter.add_mesh(plane, **field)
    if cloud.n_points:
        plotter.add_mesh(
            cloud,
            render_points_as_spheres=True,
            point_size=2.0 * RENDER_SCALE,
            **field,
        )

    fvm_box = pv.Box(bounds=FVM_BOUNDS)
    plotter.add_mesh(
        fvm_box.extract_feature_edges().tube(radius=0.025),
        color="#d9d9d9",
        smooth_shading=True,
    )
    plotter.add_mesh(wing, color="#9c9c9c", smooth_shading=True, specular=0.3)

    plotter.camera.position = view["position"]
    plotter.camera.focal_point = view["focal_point"]
    plotter.camera.up = view["view_up"]
    plotter.camera.view_angle = view["view_angle"]
    plotter.reset_camera_clipping_range()
    plotter.render()
    return plotter.screenshot(return_img=True)


def compose(image, time, forces):
    """Add the vorticity scale and the lift and drag histories to a rendered frame."""
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
        travel = times * FREESTREAM_SPEED / CHORD
        now = times <= time + 1e-9
        for column, colour, label in (
            ("lift_coefficient", LIFT_COLOUR, "$C_L$"),
            ("drag_coefficient", DRAG_COLOUR, "$C_D$"),
        ):
            values = forces[column].to_numpy()
            inset.plot(travel, values, color=colour, alpha=0.25, linewidth=1.5)
            inset.plot(travel[now], values[now], color=colour, linewidth=2.5)
            inset.plot(
                travel[now][-1],
                np.clip(values[now][-1], *FORCE_RANGE),
                "o",
                color=colour,
                markeredgecolor="white",
                markersize=8,
                clip_on=False,
            )
            inset.annotate(
                label,
                (1.0, values[travel > 2.0].mean()),
                xycoords=("axes fraction", "data"),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                color=colour,
            )
        inset.set_xlim(travel[0], travel[-1])
        inset.set_ylim(*FORCE_RANGE)
        inset.locator_params(axis="x", nbins=4)
        inset.set_yticks((FORCE_RANGE[0], sum(FORCE_RANGE) / 2, FORCE_RANGE[1]))
        inset.set_xlabel(r"$t\,U_\infty/c$")
        inset.spines[["top", "right"]].set_visible(False)

        figure.canvas.draw()
        frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        plt.close(figure)
    return frame


def main() -> None:
    particles = frame_files(SOLUTION_DIR / "vpm.pvd")
    fvm = frame_files(SOLUTION_DIR / "fvm.pvd")
    times = sorted(set(particles) & set(fvm))
    if len(sys.argv) > 1:  # a preview of one time: python -m assets.render_wing_flow_animation 24
        times = [min(times, key=lambda t: abs(t - float(sys.argv[1])))]
    forces = pd.read_csv(FORCES).drop_duplicates("time", keep="last").sort_values("time")
    wing = pv.read(WING)
    view = camera()

    plotter = pv.Plotter(
        off_screen=True,
        window_size=[RENDER_SCALE * size for size in IMAGE_SIZE],
    )
    plotter.set_background("black")
    frames = []
    for time in times:
        plane = fvm_plane(pv.read(fvm[time]))
        cloud = wake_particles(pv.read(particles[time]))
        image = render_scene(plotter, plane, cloud, wing, view)
        frames.append(compose(image, time, forces))
        print(f"  frame t = {time:g}", flush=True)
    plotter.close()

    if len(times) == 1:
        preview = OUTPUT.with_name(f"wing_flow_t{times[0]:g}.png")
        frames[0].save(preview)
        print(f"  Saved: {preview}")
        return

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
