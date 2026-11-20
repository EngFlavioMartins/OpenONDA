"""Animate the coupled cube wake and drag history.

python -m assets.render_cube_flow_animation               the README gallery GIF, 800 x 500
python -m assets.render_cube_flow_animation presentation  a 1920 x 1080 H.264 video for slides, in
                                                          docs/assets/presentation/, one frame per
                                                          saved state; its frames are kept in
                                                          solution/presentation_frames/
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
SOLUTION_DIR = CASE_DIR / "solution"
FORCES = CASE_DIR / "samples" / "forces_history.csv"
OUTPUT = CASE_DIR / "assets" / "cube_flow.gif"
PRESENTATION = PRESENTATION_DIR / "cube_flow.mp4"
PRESENTATION_FRAMES = SOLUTION_DIR / "presentation_frames"

CUBE_SIDE = 1.0  # [m]
FREESTREAM_SPEED = 1.0  # [m/s]
FVM_BOUNDS = (-1.8, 5.085, -1.8, 1.8, -1.8, 1.8)  # [m]
# ParaView's box clip removes this part of the VPM field to reveal the overlap.
VPM_CLIP_BOUNDS = (-1.0, 5.5, -2.5, -0.115, -0.05, 1.45)  # [m]
IMAGE_SIZE = (800, 500)  # [px], matching the README gallery
# Each frame is a figure 5 in tall at height / 5 dpi, so lines keep their proportion to the
# height; `text` scales the gallery font sizes, which suit an 800 x 500 GIF but are oversized on a
# 1080p slide. The GIF scene is drawn at twice its size and reduced; the video scene at its own
# size, which keeps the GPU memory of the window small beside a running solver.
# The video is framed closer than the GIF, with the wake flatter across the 16:9 picture, the
# FVM box near the left edge and the wake reaching the right edge.
FORMATS = {
    "gif": dict(size=IMAGE_SIZE, render_scale=2, text=1.0, discs=False, drag_range=(0.5, 1.5),
                bar=(0.89, 0.53, 0.035, 0.29), inset=(0.12, 0.17, 0.29, 0.24)),
    "presentation": dict(size=PRESENTATION_SIZE, render_scale=1, text=PRESENTATION_TEXT_SCALE,
                         discs=True, drag_range=(0.75, 1.25),
                         bar=(0.925, 0.58, 0.016, 0.27), inset=(0.07, 0.13, 0.22, 0.16),
                         view=dict(toward_camera=(16.0, -30.0, 20.0), focal_point=(7.4, -0.6, 0.3),
                                   distance=44.0, view_angle=13.0)),
}
FRAMES_PER_SECOND = 12  # GIF: saved states per second
VIDEO_FRAMES_PER_SECOND = 10  # video: saved states per second
VORTICITY_MAX = 5.0  # [1/s]
OPACITY = np.linspace(0.2, 1.0, 16)  # opacity of the 16 colour steps
# Video: the FVM cells and the particles outside the FVM box are drawn alike, as unlit discs with
# a soft edge, blended from the farthest to the nearest, as in the 3D renders of the paper: the
# vorticity is transparent below DISC_CUT and opaque from DISC_OPAQUE, linearly in between, so
# that the weak vorticity does not veil the hairpin vortices.
DISC_RADIUS = 0.8 * 0.045  # [m], 0.8 of the cell and particle spacing
DISC_CUT, DISC_OPAQUE = 0.6, 4.0  # [1/s]

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


def camera(toward_camera=None, focal_point=None, distance=None, view_angle=None) -> dict:
    """Set an oblique view over the cube and the downstream wake; the arguments set another one,
    from a direction toward the camera, with z up."""
    if toward_camera is not None:
        toward = np.asarray(toward_camera) / np.linalg.norm(toward_camera)
        view_up = np.array([0.0, 0.0, 1.0]) - toward[2] * toward
        return {
            "focal_point": np.asarray(focal_point),
            "position": np.asarray(focal_point) + distance * toward,
            "view_up": view_up / np.linalg.norm(view_up),
            "view_angle": view_angle,
        }
    return {
        "focal_point": np.array([9.9569245156, -0.8551330725, 0.5457414744]),
        "position": np.array([35.17181586, -28.61535302, 22.62267563]),
        "view_up": np.array([-0.35132246, 0.36616274, 0.86168287]),
        "view_angle": 15.0,
    }


def disc_actor(fvm_data: pv.DataSet, particle_data: pv.DataSet, view):
    """The FVM cells and the particles outside the FVM box as vorticity discs."""
    particles = np.asarray(particle_data.points)
    bounds = np.asarray(FVM_BOUNDS)
    outside = np.any((particles < bounds[::2]) | (particles > bounds[1::2]), axis=1)
    points = np.vstack([np.asarray(fvm_data.cell_centers().points), particles[outside]])
    vorticity = np.concatenate([
        np.linalg.norm(np.asarray(fvm_data.cell_data["vorticity"]), axis=1),
        np.linalg.norm(np.asarray(particle_data["vorticity"]), axis=1)[outside],
    ])
    return vorticity_discs(points, vorticity, view["position"], radius=DISC_RADIUS, cut=DISC_CUT,
                           opaque=DISC_OPAQUE, maximum=VORTICITY_MAX, colormap=VORTICITY_COLORMAP)


def render_scene(plotter, plane, cloud, view, point_size, discs=None):
    """Render the FVM and VPM vorticity and the cube for one time.

    The GIF draws the FVM mid-plane and the particles as spheres of point_size pixels; the video
    draws the discs of disc_actor.
    """
    plotter.clear()
    field = {
        "scalars": "vorticity magnitude",
        "cmap": VORTICITY_COLORMAP,
        "clim": (0.0, VORTICITY_MAX),
        "opacity": OPACITY,
        "show_scalar_bar": False,
    }
    if discs is not None:
        plotter.add_actor(discs)
    else:
        plotter.add_mesh(plane, **field)
    if discs is None and cloud.n_points:
        plotter.add_mesh(
            cloud,
            render_points_as_spheres=True,
            point_size=point_size,
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


def compose(image, time, forces, layout):
    """Add the vorticity scale and drag history to a rendered frame in the given format."""
    size, text, drag_range = layout["size"], layout["text"], layout["drag_range"]
    with plt.rc_context({**GALLERY_STYLE, "font.size": GALLERY_STYLE["font.size"] * text}):
        figure = plt.figure(figsize=(5.0 * size[0] / size[1], 5.0), dpi=size[1] / 5.0)
        figure.patch.set_facecolor("black")
        scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        scene.imshow(image, interpolation="antialiased")
        scene.set_axis_off()

        bar = figure.add_axes(layout["bar"])
        figure.colorbar(
            plt.cm.ScalarMappable(Normalize(0.0, VORTICITY_MAX), VORTICITY_COLORMAP),
            cax=bar,
            ticks=(0.0, VORTICITY_MAX),
        )
        bar.set_yticklabels(["0", f"{VORTICITY_MAX:g}"], fontsize=30 * text)
        bar.tick_params(direction="out", length=10 * text, width=1.5)
        bar.set_title(r"$|\omega|$ [1/s]", fontsize=30 * text, pad=28 * text, x=0.2)

        inset = figure.add_axes(layout["inset"])
        inset.set_facecolor("none")
        times = forces["time"].to_numpy()
        travel = times * FREESTREAM_SPEED / CUBE_SIDE
        now = times <= time + 1e-9
        drag = forces["drag_coefficient"].to_numpy()
        inset.plot(travel, drag, color="#9bd75a", alpha=0.25, linewidth=1.5)
        inset.plot(travel[now], drag[now], color="#9bd75a", linewidth=2.5)
        inset.plot(
            travel[now][-1],
            np.clip(drag[now][-1], *drag_range),
            "o",
            color="#9bd75a",
            markeredgecolor="white",
            markersize=8 * text**0.5,
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
        inset.set_ylim(*drag_range)
        inset.locator_params(axis="x", nbins=4)
        inset.set_yticks((drag_range[0], sum(drag_range) / 2, drag_range[1]))
        inset.set_xlabel(r"$t\,U_\infty/D$")
        inset.spines[["top", "right"]].set_visible(False)

        figure.canvas.draw()
        frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        plt.close(figure)
    return frame


def main() -> None:
    presentation = "presentation" in sys.argv[1:]
    layout = FORMATS["presentation" if presentation else "gif"]
    size, render_scale = layout["size"], layout["render_scale"]
    particles = frame_files(SOLUTION_DIR / "vpm.pvd")
    fvm = frame_files(SOLUTION_DIR / "fvm.pvd")
    forces = pd.read_csv(FORCES)
    view = camera(**layout.get("view", {}))
    point_size = 3.0 * render_scale * size[1] / IMAGE_SIZE[1]

    plotter = pv.Plotter(
        off_screen=True,
        window_size=[render_scale * pixels for pixels in size],
        lighting="light kit",
    )
    plotter.set_background("black")
    if layout["discs"]:
        plotter.enable_anti_aliasing("fxaa")
        PRESENTATION_FRAMES.mkdir(parents=True, exist_ok=True)
        for old in PRESENTATION_FRAMES.glob("frame_*.png"):
            old.unlink()
    frames = []
    for (time, particle_path), (_, fvm_path) in zip(particles, fvm, strict=True):
        fvm_data = pv.read(fvm_path)
        particle_data = pv.read(particle_path)
        if layout["discs"]:
            discs = disc_actor(fvm_data, particle_data, view)
            image = render_scene(plotter, None, None, view, point_size, discs)
        else:
            plane = fvm_plane(fvm_data)
            cloud = wake_particles(particle_data)
            image = render_scene(plotter, plane, cloud, view, point_size)
        frame = compose(image, time, forces, layout)
        if layout["discs"]:
            frame.save(PRESENTATION_FRAMES / f"frame_{len(frames):04d}.png")
        frames.append(frame)
    plotter.close()

    if presentation:
        write_presentation_video(
            frames, PRESENTATION, VIDEO_FRAMES_PER_SECOND, frame_dir=PRESENTATION_FRAMES
        )
        print(f"  Saved: {PRESENTATION} and {len(frames)} frames in {PRESENTATION_FRAMES}")
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
