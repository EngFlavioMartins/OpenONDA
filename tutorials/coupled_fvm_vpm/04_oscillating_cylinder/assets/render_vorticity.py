#!/usr/bin/env python3
"""Render the vortical structures of each run at two instants of its last oscillation cycle.

The FVM vorticity on the mid-plane of the box and the particles of the outer
wake share one colour scale. The instants are the last upper turning point of
the cross-flow displacement and the following downward crossing of its mean;
the images are placed in the inertial frame, the frame of the paper's
figures 14 and 15, and typeset with the thesis fonts.
"""

import subprocess
import tempfile
import xml.etree.ElementTree as ET

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
import pyvista as pv

from assets.response import CASE_DIR, DIAMETER, FREESTREAM_SPEED, completed_runs, read_run
from assets.response import variant

FIGURES = CASE_DIR / "figures"
FVM_DOMAIN = (-2.4, 4.4, -2.4, 2.4)  # m, x and y bounds in the frame of the box
FVM_HEIGHT = 1.0  # m, drawn span of the planar FVM domain

WIDTH_MM, HEIGHT_MM = 125.0, 50.0  # one panel at the thesis print width
DPI = 300
PIXELS_PER_MM = DPI / 25.4
OMEGA_MAX = 3.0  # omega D / U_inf
OPAQUE_FROM = 1.0  # omega D / U_inf, weaker vorticity fades into the page

# Oblique view from above and downstream, following the mean cylinder position.
VIEW_DIRECTION = (0.45, -1.0, 1.2)  # toward the camera
FOCAL_POINT = (7.3, 0.46, 0.0)  # m
VIEW_DISTANCE = 32.0  # m
VIEW_ANGLE = 15.5  # deg
ARROW_START = -4.7  # m, x of the freestream arrow tail
ARROW_LENGTH = 2.0  # m

# Surface materials: colour, metallic fraction and roughness.
FRAME_MATERIAL = ("#c3c8cc", 0.1, 0.4)
CYLINDER_MATERIAL = ("#c8c8c8", 0.05, 0.45)
ARROW_MATERIAL = ("#9a9a9a", 0.0, 0.5)
SPRING_MATERIAL = ("#8c8c8c", 0.2, 0.5)

# Blue vorticity scale for a white background.
VORTICITY_COLORMAP = LinearSegmentedColormap.from_list(
    "print_vorticity", ["#c9e6fd", "#73c0ff", "#3791ed", "#0d58b7", "#031a3c"], N=16
)


def frame_files(collection):
    """Times and files of a ParaView collection."""
    entries = ET.parse(collection).getroot().iter("DataSet")
    return [
        (float(entry.get("timestep")), collection.parent / entry.get("file")) for entry in entries
    ]


def chosen_instants(motion, saved_times):
    """Saved times nearest the last upper turning point of y and its following mean crossing."""
    time, y = motion["time"], motion["displacement_y"]
    centred = y - y.mean()
    interior = np.arange(1, len(y) - 1)
    peaks = interior[(centred[1:-1] > centred[:-2]) & (centred[1:-1] >= centred[2:])]
    peaks = [k for k in peaks if time[k] >= saved_times[0] - 1e-9]
    instants = {}
    if peaks:
        instants["top"] = time[peaks[-1]]
        after = np.flatnonzero(
            (centred[:-1] > 0.0) & (centred[1:] <= 0.0) & (time[:-1] > time[peaks[-1]])
        )
        if len(after):
            instants["crossing"] = time[after[0]]
    if not instants:
        instants["last"] = saved_times[-1]
    saved = np.asarray(saved_times)
    return {name: saved[np.argmin(np.abs(saved - value))] for name, value in instants.items()}


def displacement_at(motion, time):
    """Cylinder displacement (x, y) in m at a saved time."""
    return np.array(
        [
            np.interp(time, motion["time"], motion[name])
            for name in ("displacement_x", "displacement_y")
        ]
    )


def fvm_plane(path):
    """FVM vorticity magnitude on the mid-plane of the one-cell span, at the mesh points."""
    plane = pv.read(path).slice(normal="z", origin=(0.0, 0.0, 0.0)).cell_data_to_point_data()
    plane["vorticity magnitude"] = (
        np.linalg.norm(np.asarray(plane["vorticity"]), axis=1) * DIAMETER / FREESTREAM_SPEED
    )
    return plane


def wake_particles(path, shift):
    """Particles outside the FVM box, moved from the box frame to the inertial frame."""
    particles = pv.read(path)
    points = np.asarray(particles.points)
    xmin, xmax, ymin, ymax = FVM_DOMAIN
    outside = (
        (points[:, 0] < xmin)
        | (points[:, 0] > xmax)
        | (points[:, 1] < ymin)
        | (points[:, 1] > ymax)
    )
    cloud = pv.PolyData(points[outside] + np.array([shift[0], shift[1], 0.0]))
    cloud["vorticity magnitude"] = (
        np.linalg.norm(np.asarray(particles["vorticity"])[outside], axis=1)
        * DIAMETER
        / FREESTREAM_SPEED
    )
    return cloud


def screen_mm(plotter, point):
    """Position of a scene point on the printed panel, in mm from its lower-left corner."""
    plotter.renderer.SetWorldPoint(*point, 1.0)
    plotter.renderer.WorldToDisplay()
    column, row, _ = plotter.renderer.GetDisplayPoint()
    return column / PIXELS_PER_MM, row / PIXELS_PER_MM


def physical_material(material):
    """add_mesh arguments of a physically based material."""
    colour, metallic, roughness = material
    return {
        "color": colour,
        "pbr": True,
        "metallic": metallic,
        "roughness": roughness,
        "smooth_shading": True,
    }


def studio_environment():
    """A soft studio sky for the physically based materials: bright above, grey below the horizon."""
    elevation = np.linspace(1.0, -1.0, 128)[:, None, None]
    brightness = np.where(elevation > 0.0, 0.85 + 0.15 * elevation, 0.85 + 0.35 * elevation)
    sky = np.broadcast_to(brightness, (128, 256, 3))
    return pv.Texture((255 * sky).astype(np.uint8))


def camera():
    """The fixed oblique view."""
    toward = np.asarray(VIEW_DIRECTION, dtype=float)
    toward /= np.linalg.norm(toward)
    focal = np.asarray(FOCAL_POINT, dtype=float)
    return focal, focal + VIEW_DISTANCE * toward


def render_panel(plane, cloud, shift):
    """Render one instant on white; return the RGB image and the freestream-label anchor."""
    window_size = [round(WIDTH_MM * PIXELS_PER_MM), round(HEIGHT_MM * PIXELS_PER_MM)]
    plotter = pv.Plotter(off_screen=True, window_size=window_size, lighting="light kit")
    plotter.set_background("white")
    plotter.set_environment_texture(studio_environment())
    plotter.enable_depth_peeling()
    vorticity = {
        "scalars": "vorticity magnitude",
        "cmap": VORTICITY_COLORMAP,
        "clim": (0.0, OMEGA_MAX),
        "opacity": np.minimum(np.linspace(0.0, OMEGA_MAX / OPAQUE_FROM, 16), 1.0),
        "show_scalar_bar": False,
    }
    plotter.add_mesh(plane, **vorticity)
    if cloud.n_points:
        plotter.add_mesh(
            cloud,
            render_points_as_spheres=True,
            point_size=window_size[0] / 170,
            specular=0.3,
            specular_power=20.0,
            **vorticity,
        )
    xmin, xmax, ymin, ymax = FVM_DOMAIN
    box = pv.Box(
        bounds=(
            xmin + shift[0],
            xmax + shift[0],
            ymin + shift[1],
            ymax + shift[1],
            -0.5 * FVM_HEIGHT,
            0.5 * FVM_HEIGHT,
        )
    )
    plotter.add_mesh(
        box.extract_feature_edges().tube(radius=0.03, n_sides=24),
        **physical_material(FRAME_MATERIAL),
    )
    cylinder = pv.Cylinder(
        center=(shift[0], shift[1], 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=0.5 * DIAMETER,
        height=FVM_HEIGHT,
        resolution=240,
    )
    plotter.add_mesh(cylinder, **physical_material(CYLINDER_MATERIAL))
    # The rest position of the cylinder axis, a thin post through the box
    plotter.add_mesh(
        pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=0.03, height=1.6 * FVM_HEIGHT
        ),
        **physical_material(SPRING_MATERIAL),
    )
    arrow = pv.Arrow(
        start=(ARROW_START, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        scale=ARROW_LENGTH,
        tip_length=0.3,
        tip_radius=0.1,
        shaft_radius=0.04,
        tip_resolution=64,
        shaft_resolution=64,
    )
    plotter.add_mesh(arrow, **physical_material(ARROW_MATERIAL))
    focal, position = camera()
    plotter.camera.position = position
    plotter.camera.focal_point = focal
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.view_angle = VIEW_ANGLE
    plotter.reset_camera_clipping_range()
    plotter.enable_ssao(radius=0.4, bias=0.005, kernel_size=128)
    plotter.enable_anti_aliasing("ssaa")
    plotter.render()
    image = plotter.screenshot(return_img=True)
    label_anchor = screen_mm(plotter, (ARROW_START + 0.5 * ARROW_LENGTH, 0.0, 0.0))
    plotter.close()
    return image, label_anchor


def colour_bar(left, bottom):
    """TikZ for the stepped vorticity colour bar, 25 x 2 mm, with its ticks and label below."""
    width, height = 25.0, 2.0
    steps = VORTICITY_COLORMAP.N
    lines = []
    for step in range(steps):
        red, green, blue, _ = VORTICITY_COLORMAP((step + 0.5) / steps)
        lines.append(rf"\definecolor{{bar{step}}}{{rgb}}{{{red:.4f},{green:.4f},{blue:.4f}}}")
        x0 = left + width * step / steps
        lines.append(
            rf"\fill[bar{step}] ({x0:.3f},{bottom}) rectangle ++({width / steps:.3f},{height});"
        )
    lines.append(rf"\draw[line width=.6pt] ({left},{bottom}) rectangle ++({width},{height});")
    for fraction, value in ((0.0, "0"), (0.5, f"{0.5 * OMEGA_MAX:g}"), (1.0, f"{OMEGA_MAX:g}")):
        x = left + fraction * width
        lines.append(rf"\draw[line width=.6pt] ({x},{bottom}) -- ++(0,-0.8);")
        lines.append(rf"\node[anchor=north,inner sep=.3mm] at ({x},{bottom - 0.8}) {{${value}$}};")
    lines.append(
        rf"\node[anchor=north,inner sep=.3mm] at ({left + 0.5 * width},{bottom - 5.2})"
        r" {$|\boldsymbol{\omega}|\,D/U_\infty$};"
    )
    return "\n".join(lines)


def overlay(name, panels):
    """The standalone TikZ figure: the panels stacked at print width with their labels."""
    lines = [
        r"\documentclass[11pt,tikz,border=0pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{newpxtext,amsmath,newpxmath,graphicx}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1mm,y=1mm,every node/.style={font=\normalsize}]",
        rf"\path[use as bounding box] (0,0) rectangle ({WIDTH_MM:g},{HEIGHT_MM * len(panels):g});",
    ]
    for row, (label, time, (label_x, label_y), image_name) in enumerate(panels):
        bottom = HEIGHT_MM * (len(panels) - 1 - row)
        lines += [
            rf"\node[anchor=south west,inner sep=0] at (0,{bottom:g})"
            rf" {{\includegraphics[width={WIDTH_MM:g}mm]{{{image_name}}}}};",
            rf"\node[anchor=south,inner sep=.8mm] at ({label_x:.2f},{bottom + label_y + 0.6:.2f})"
            r" {$\mathbf{U}_\infty$};",
            rf"\node[anchor=north west,inner sep=1mm] at (0,{bottom + HEIGHT_MM:g})"
            rf" {{({'ab'[row]}) {label}, $t={time:.1f}$ s}};",
        ]
    lines += [
        colour_bar(WIDTH_MM - 28.0, HEIGHT_MM * len(panels) - 5.0),
        r"\end{tikzpicture}",
        r"\end{document}",
        "",
    ]
    return "\n".join(lines)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for run in completed_runs(CASE_DIR / "samples"):
        solution = CASE_DIR / "solution" / variant(*run)
        fvm_frames = dict(frame_files(solution / "fvm.pvd"))
        vpm_frames = dict(frame_files(solution / "vpm.pvd"))
        saved_times = sorted(set(fvm_frames) & set(vpm_frames))
        if not saved_times:
            continue
        motion, _, _ = read_run(CASE_DIR / "samples" / variant(*run), run[1])
        name = f"vorticity_{variant(*run)}"
        panels = []
        for label, time in chosen_instants(motion, saved_times).items():
            shift = displacement_at(motion, time)
            image, anchor = render_panel(
                fvm_plane(fvm_frames[time]), wake_particles(vpm_frames[time], shift), shift
            )
            image_name = f"{name}_{label}_raw.png"
            Image.fromarray(image).save(FIGURES / image_name, dpi=(DPI, DPI))
            titles = {
                "top": "upper turning point",
                "crossing": "downward crossing",
                "last": "last saved state",
            }
            panels.append((titles[label], time, anchor, image_name))
        (FIGURES / f"{name}.tex").write_text(overlay(name, panels))
        # Typeset the print overlay without retaining LaTeX auxiliary files.
        with tempfile.TemporaryDirectory() as build:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", build, f"{name}.tex"],
                cwd=FIGURES,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            # Only the PNG is kept; the PDF stays in the temporary build directory.
            subprocess.run(
                ["pdftoppm", "-r", f"{DPI}", "-png", "-singlefile", f"{build}/{name}.pdf", name],
                cwd=FIGURES,
                check=True,
            )
        print(f"  Saved: {FIGURES / name}.png")


if __name__ == "__main__":
    main()
