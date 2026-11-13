#!/usr/bin/env python3
"""Render the rotor, its FVM disk and the particle wake at the last saved time for print.

The blades, the FVM vorticity on the plane through the rotor axis and the
particles of the wake share one vorticity colour scale on a white page; the
text is typeset with the thesis fonts over the render.
"""

import subprocess
import tempfile
import xml.etree.ElementTree as ET

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
import pyvista as pv

from assets.rotor_data import CASE_DIR, FIGURES

SOLUTION = CASE_DIR / "solution"
WIDTH_MM, HEIGHT_MM = 125.0, 62.5  # the thesis print width
DPI = 300
PIXELS_PER_MM = DPI / 25.4
OMEGA_MAX = 8.0  # 1/s, the tip vortices; the stronger blade vorticity saturates
OPAQUE_FROM = 2.0  # 1/s, weaker vorticity fades into the page

# Oblique view from upstream of one side, the rotor at the left of the page.
VIEW_DIRECTION = (-0.6, -1.0, 0.7)  # toward the camera
VIEW_ANGLE = 22.0  # deg
FOCAL_SHIFT = 0.35  # of the wake length, downstream of the rotor
ARROW_LENGTH = 4.0  # m, freestream arrow upstream of the rotor

# Surface materials: colour, metallic fraction and roughness.
BLADE_MATERIAL = ("#c8c8c8", 0.05, 0.45)
FRAME_MATERIAL = ("#c3c8cc", 0.1, 0.4)
ARROW_MATERIAL = ("#9a9a9a", 0.0, 0.5)

VORTICITY_COLORMAP = LinearSegmentedColormap.from_list(
    "print_vorticity", ["#c9e6fd", "#73c0ff", "#3791ed", "#0d58b7", "#031a3c"], N=16
)


def frame_files(collection):
    """Times and files of a ParaView collection."""
    entries = ET.parse(collection).getroot().iter("DataSet")
    return [
        (float(entry.get("timestep")), collection.parent / entry.get("file")) for entry in entries
    ]


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


def screen_mm(plotter, point):
    """Position of a scene point on the printed figure, in mm from its lower-left corner."""
    plotter.renderer.SetWorldPoint(*point, 1.0)
    plotter.renderer.WorldToDisplay()
    column, row, _ = plotter.renderer.GetDisplayPoint()
    return column / PIXELS_PER_MM, row / PIXELS_PER_MM


def render_scene(fvm_frame, vpm_frame):
    """Render the blades, the FVM mid-plane and the wake on white; return the image and label anchor."""
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
    mesh = pv.read(fvm_frame)
    # The plane through the rotor axis, in the orientation of the turned mesh
    plane = mesh.slice(normal="z", origin=(0.0, 0.0, 0.0)).cell_data_to_point_data()
    plane["vorticity magnitude"] = np.linalg.norm(np.asarray(plane["vorticity"]), axis=1)
    plotter.add_mesh(plane, **vorticity)
    # The edges of the turned FVM disk
    boundary = mesh.extract_surface()
    plotter.add_mesh(
        boundary.extract_feature_edges(feature_angle=89.0).tube(radius=0.05, n_sides=16),
        **physical_material(FRAME_MATERIAL),
    )
    particles = pv.read(vpm_frame)
    if particles.n_points:
        cloud = pv.PolyData(np.asarray(particles.points))
        cloud["vorticity magnitude"] = np.linalg.norm(np.asarray(particles["vorticity"]), axis=1)
        plotter.add_mesh(
            cloud,
            render_points_as_spheres=True,
            point_size=window_size[0] / 300,
            specular=0.3,
            specular_power=20.0,
            **vorticity,
        )
    for blade_file in sorted((CASE_DIR / "assets").glob("blade_*.stl")):
        plotter.add_mesh(pv.read(blade_file), **physical_material(BLADE_MATERIAL))
    arrow_start = -4.0 - ARROW_LENGTH
    arrow = pv.Arrow(
        start=(arrow_start, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        scale=ARROW_LENGTH,
        tip_length=0.3,
        tip_radius=0.1,
        shaft_radius=0.04,
        tip_resolution=64,
        shaft_resolution=64,
    )
    plotter.add_mesh(arrow, **physical_material(ARROW_MATERIAL))

    extent = max(float(particles.bounds[1]) if particles.n_points else 0.0, 12.0)
    focal = np.array([FOCAL_SHIFT * extent, 0.0, 0.0])
    toward = np.asarray(VIEW_DIRECTION, dtype=float)
    toward /= np.linalg.norm(toward)
    plotter.camera.position = focal + 1.6 * (extent + 12.0) * toward
    plotter.camera.focal_point = focal
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.view_angle = VIEW_ANGLE
    plotter.reset_camera_clipping_range()
    plotter.enable_ssao(radius=0.6, bias=0.005, kernel_size=128)
    plotter.enable_anti_aliasing("ssaa")
    plotter.render()
    image = plotter.screenshot(return_img=True)
    label_anchor = screen_mm(plotter, (arrow_start + 0.5 * ARROW_LENGTH, 0.0, 0.0))
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
        r" {$|\boldsymbol{\omega}|$ [1/s]};"
    )
    return "\n".join(lines)


def overlay(name, time, label_anchor):
    """The standalone TikZ figure: the render at print width and its text in the thesis fonts."""
    label_x, label_y = label_anchor
    return "\n".join(
        [
            r"\documentclass[11pt,tikz,border=0pt]{standalone}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{newpxtext,amsmath,newpxmath,graphicx}",
            r"\begin{document}",
            r"\begin{tikzpicture}[x=1mm,y=1mm,every node/.style={font=\normalsize}]",
            rf"\path[use as bounding box] (0,0) rectangle ({WIDTH_MM:g},{HEIGHT_MM:g});",
            r"\node[anchor=south west,inner sep=0] at (0,0)"
            rf" {{\includegraphics[width={WIDTH_MM:g}mm]{{{name}_raw.png}}}};",
            rf"\node[anchor=south,inner sep=.8mm] at ({label_x:.2f},{label_y + 0.6:.2f})"
            r" {$\mathbf{U}_\infty$};",
            rf"\node[anchor=north west,inner sep=1mm] at (0,{HEIGHT_MM:g}) {{$t={time:.3f}$ s}};",
            colour_bar(WIDTH_MM - 28.0, HEIGHT_MM - 5.0),
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    fvm_frames = dict(frame_files(SOLUTION / "fvm.pvd"))
    vpm_frames = dict(frame_files(SOLUTION / "vpm.pvd"))
    time = max(set(fvm_frames) & set(vpm_frames))
    name = f"rotor_wake_t{time:g}"
    image, label_anchor = render_scene(fvm_frames[time], vpm_frames[time])
    Image.fromarray(image).save(FIGURES / f"{name}_raw.png", dpi=(DPI, DPI))
    (FIGURES / f"{name}.tex").write_text(overlay(name, time, label_anchor))
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
