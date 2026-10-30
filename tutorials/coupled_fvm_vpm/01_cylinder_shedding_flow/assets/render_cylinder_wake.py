"""Render a developed coupled cylinder wake for print."""

import shutil
import subprocess
import tempfile

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
import pyvista as pv

from assets.render_cylinder_animation import (
    CASE_DIR,
    DIAMETER,
    FVM_DOMAIN,
    FVM_HEIGHT,
    SOLUTION_DIR,
    camera,
    frame_files,
    fvm_plane,
    wake_particles,
)

ASSETS_DIR = CASE_DIR / "assets"
TIME = 86.0  # [s], developed shedding, a lift minimum
NAME = f"cylinder_wake_t{TIME:g}"

WIDTH_MM, HEIGHT_MM = 125.0, 50.0  # the thesis print width
DPI = 300
PIXELS_PER_MM = DPI / 25.4
OMEGA_MAX = 3.0  # omega D/U_inf, as the vorticity field figures
OPAQUE_FROM = 1.0  # omega D/U_inf, weaker vorticity fades into the page

# Oblique view from above and downstream.
VIEW_DIRECTION = (0.45, -1.0, 1.2)  # toward the camera
FOCAL_POINT = (7.3, 0.46, 0.0)  # [m]
VIEW_DISTANCE = 32.0  # [m]
VIEW_ANGLE = 15.5  # [deg]

# Freestream arrow upstream of the FVM domain.
ARROW_START = -4.7  # [m], x of the arrow tail
ARROW_LENGTH = 2.0  # [m]

# Surface materials: colour, metallic fraction, and roughness.
FRAME_MATERIAL = ("#c3c8cc", 0.1, 0.4)  # light grey tubes of the FVM domain edges
CYLINDER_MATERIAL = ("#c8c8c8", 0.05, 0.45)
ARROW_MATERIAL = ("#9a9a9a", 0.0, 0.5)

# Blue vorticity scale for a white background.
VORTICITY_COLORMAP = LinearSegmentedColormap.from_list(
    "print_vorticity", ["#c9e6fd", "#73c0ff", "#3791ed", "#0d58b7", "#031a3c"], N=16
)


def screen_mm(plotter, point) -> tuple[float, float]:
    """Position of a scene point on the printed figure, in mm from its lower-left corner."""
    plotter.renderer.SetWorldPoint(*point, 1.0)
    plotter.renderer.WorldToDisplay()
    column, row, _ = plotter.renderer.GetDisplayPoint()
    return column / PIXELS_PER_MM, row / PIXELS_PER_MM


def physical_material(material) -> dict:
    """add_mesh arguments of a physically based material."""
    colour, metallic, roughness = material
    return {
        "color": colour,
        "pbr": True,
        "metallic": metallic,
        "roughness": roughness,
        "smooth_shading": True,
    }


def studio_environment() -> pv.Texture:
    """A soft studio sky for the physically based materials: bright above, grey below the horizon."""
    elevation = np.linspace(1.0, -1.0, 128)[:, None, None]
    brightness = np.where(elevation > 0.0, 0.85 + 0.15 * elevation, 0.85 + 0.35 * elevation)
    sky = np.broadcast_to(brightness, (128, 256, 3))
    return pv.Texture((255 * sky).astype(np.uint8))


def render_scene(plane, cloud, view):
    """Render the scene on white; return the RGB image and the freestream-label anchor.

    The label sits above the middle of the freestream arrow.
    """
    window_size = [round(WIDTH_MM * PIXELS_PER_MM), round(HEIGHT_MM * PIXELS_PER_MM)]
    plotter = pv.Plotter(off_screen=True, window_size=window_size, lighting="light kit")
    plotter.set_background("white")
    plotter.set_environment_texture(studio_environment())
    plotter.enable_depth_peeling()
    # Use one vorticity scale for FVM and particles.
    vorticity = {
        "scalars": "vorticity magnitude",
        "cmap": VORTICITY_COLORMAP,
        "clim": (0.0, OMEGA_MAX),
        "opacity": np.minimum(np.linspace(0.0, OMEGA_MAX / OPAQUE_FROM, 16), 1.0),
        "show_scalar_bar": False,
    }
    plotter.add_mesh(plane, **vorticity)
    plotter.add_mesh(
        cloud,
        render_points_as_spheres=True,
        point_size=window_size[0] / 170,
        specular=0.3,
        specular_power=20.0,
        **vorticity,
    )
    xmin, xmax, ymin, ymax = FVM_DOMAIN
    box = pv.Box(bounds=(xmin, xmax, ymin, ymax, -0.5 * FVM_HEIGHT, 0.5 * FVM_HEIGHT))
    frame = box.extract_feature_edges().tube(radius=0.03, n_sides=24)
    plotter.add_mesh(frame, **physical_material(FRAME_MATERIAL))
    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=0.5 * DIAMETER,
        height=FVM_HEIGHT,
        resolution=240,
    )
    plotter.add_mesh(cylinder, **physical_material(CYLINDER_MATERIAL))
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
    plotter.camera.position = view["position"]
    plotter.camera.focal_point = view["focal_point"]
    plotter.camera.up = view["view_up"]
    plotter.camera.view_angle = view["view_angle"]
    plotter.reset_camera_clipping_range()
    # Ambient occlusion separates the body, box, and vorticity plane.
    plotter.enable_ssao(radius=0.4, bias=0.005, kernel_size=128)
    plotter.enable_anti_aliasing("ssaa")
    plotter.render()
    image = plotter.screenshot(return_img=True)
    label_anchor = screen_mm(plotter, (ARROW_START + 0.5 * ARROW_LENGTH, 0.0, 0.0))
    plotter.close()
    return image, label_anchor


def colour_bar(left: float, bottom: float) -> str:
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


def overlay(label_anchor) -> str:
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
            rf" {{\includegraphics[width={WIDTH_MM:g}mm]{{{NAME}_raw.png}}}};",
            rf"\node[anchor=south,inner sep=.8mm] at ({label_x:.2f},{label_y + 0.6:.2f})"
            r" {$\mathbf{U}_\infty$};",
            colour_bar(WIDTH_MM - 28.0, HEIGHT_MM - 5.0),
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )


def main() -> None:
    plane_file = frame_files(SOLUTION_DIR / "fvm.pvd", TIME, TIME)[0][1]
    particle_file = frame_files(SOLUTION_DIR / "vpm.pvd", TIME, TIME)[0][1]
    view = camera(VIEW_DIRECTION, FOCAL_POINT, VIEW_DISTANCE, VIEW_ANGLE)
    image, label_anchor = render_scene(fvm_plane(plane_file), wake_particles(particle_file), view)

    Image.fromarray(image).save(ASSETS_DIR / f"{NAME}_raw.png", dpi=(DPI, DPI))
    (ASSETS_DIR / f"{NAME}.tex").write_text(overlay(label_anchor))

    # Typeset the print overlay without retaining LaTeX auxiliary files.
    with tempfile.TemporaryDirectory() as build:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", build, f"{NAME}.tex"],
            cwd=ASSETS_DIR,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        shutil.copy(f"{build}/{NAME}.pdf", ASSETS_DIR / f"{NAME}.pdf")
    subprocess.run(
        ["pdftoppm", "-r", f"{DPI}", "-png", "-singlefile", f"{NAME}.pdf", NAME],
        cwd=ASSETS_DIR,
        check=True,
    )
    print(f"  Saved: {ASSETS_DIR / NAME}.pdf")
    print(f"  Saved: {ASSETS_DIR / NAME}.png")


if __name__ == "__main__":
    main()
