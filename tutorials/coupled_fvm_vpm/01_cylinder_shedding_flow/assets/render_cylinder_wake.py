"""Render the coupled cylinder wake at one time of developed shedding, for print.

The scene of the README animation on a white page: the FVM vorticity on the
mid-plane of the FVM domain and the particles outside it, on one
vorticity-magnitude scale, the FVM domain edges and the cylinder. On white,
weak vorticity fades to light blue and strong vorticity is dark. Writes
``assets/cylinder_wake.png`` and ``.pdf`` at the 125 mm print width.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from assets.render_cylinder_animation import (
    CASE_DIR,
    DIAMETER,
    FOCAL_POINT,
    FVM_DOMAIN,
    FVM_HEIGHT,
    SOLUTION_DIR,
    VIEW_ANGLE,
    VIEW_DIRECTION,
    VIEW_DISTANCE,
    camera,
    frame_files,
    fvm_plane,
    wake_particles,
)
from openonda.plotting import CM, FONT_SIZE_PT, export_figure, set_thesis_style

OUTPUT = CASE_DIR / "assets" / "cylinder_wake"
TIME = 86.0  # [s], developed shedding, a lift minimum

WIDTH_CM, HEIGHT_CM = 12.5, 7.0
PIXELS_PER_CM = 200  # 508 ppi
OMEGA_MAX = (
    3.0  # omega D/U_inf, as the vorticity field figures; the shear layers keep their gradation
)
OPAQUE_FROM = 1.0  # omega D/U_inf, weaker vorticity fades into the page

# The gallery blue scale, reversed, so weak vorticity fades into the white page
VORTICITY_COLORMAP = LinearSegmentedColormap.from_list(
    "print_vorticity", ["#c9e6fd", "#73c0ff", "#3791ed", "#0d58b7", "#031a3c"], N=16
)


def render_scene(plane, cloud, view, window_size):
    """Render the FVM plane, particles, domain edges and cylinder on white, returning the RGB image."""
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.enable_depth_peeling()
    # FVM and particle vorticity share one scale; weak vorticity fades out, so the street is continuous
    vorticity = {
        "scalars": "vorticity magnitude",
        "cmap": VORTICITY_COLORMAP,
        "clim": (0.0, OMEGA_MAX),
        "opacity": np.minimum(np.linspace(0.0, OMEGA_MAX / OPAQUE_FROM, 16), 1.0),
        "show_scalar_bar": False,
    }
    plotter.add_mesh(plane, **vorticity)
    plotter.add_mesh(
        cloud, render_points_as_spheres=True, point_size=window_size[0] / 170, **vorticity
    )
    xmin, xmax, ymin, ymax = FVM_DOMAIN
    box = pv.Box(bounds=(xmin, xmax, ymin, ymax, -0.5 * FVM_HEIGHT, 0.5 * FVM_HEIGHT))
    plotter.add_mesh(
        box.extract_feature_edges().tube(radius=0.03, n_sides=24),
        color="#60686c",
        smooth_shading=True,
    )
    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=0.5 * DIAMETER,
        height=FVM_HEIGHT,
        resolution=240,
    )
    plotter.add_mesh(
        cylinder, color="#b4b4b4", smooth_shading=True, specular=0.4, specular_power=20
    )
    plotter.camera.position = view["position"]
    plotter.camera.focal_point = view["focal_point"]
    plotter.camera.up = view["view_up"]
    plotter.camera.view_angle = view["view_angle"]
    plotter.reset_camera_clipping_range()
    plotter.enable_anti_aliasing("ssaa")
    plotter.render()
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def main() -> None:
    plane_file = frame_files(SOLUTION_DIR / "fvm.pvd", TIME, TIME)[0][1]
    particle_file = frame_files(SOLUTION_DIR / "vpm.pvd", TIME, TIME)[0][1]
    view = camera(VIEW_DIRECTION, FOCAL_POINT, VIEW_DISTANCE, VIEW_ANGLE)
    window_size = [round(WIDTH_CM * PIXELS_PER_CM), round(HEIGHT_CM * PIXELS_PER_CM)]
    image = render_scene(fvm_plane(plane_file), wake_particles(particle_file), view, window_size)

    set_thesis_style()
    figure = plt.figure(figsize=(WIDTH_CM * CM, HEIGHT_CM * CM))
    scene = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    scene.imshow(image, interpolation="antialiased")
    scene.set_axis_off()

    # Vorticity magnitude of the FVM field and the particles: end values, title above the bar
    bar = figure.add_axes((0.86, 0.5, 0.018, 0.3))
    figure.colorbar(
        plt.cm.ScalarMappable(Normalize(0.0, OMEGA_MAX), VORTICITY_COLORMAP),
        cax=bar,
        ticks=(0.0, OMEGA_MAX),
    )
    bar.set_yticklabels(["0", f"{OMEGA_MAX:g}"], fontsize=FONT_SIZE_PT)
    bar.set_title(r"$|\omega| D/U_\infty$", fontsize=FONT_SIZE_PT, pad=10)
    figure.text(0.03, 0.05, rf"$t\,U_\infty/D = {TIME:g}$", fontsize=FONT_SIZE_PT)

    export_figure(figure, OUTPUT)


if __name__ == "__main__":
    main()
