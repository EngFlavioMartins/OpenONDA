"""Reproduce the final moving-flat-plate wake rendering and its LaTeX overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv

from openonda import plotting as theme
from openonda.scenes import export_scene
from source.solvers.vpm.io.postprocess import coupled_frames, particle_state

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
SOLUTION_DIR = CASE_DIR / "solution" / "exp_moving_aoa12"
FIGURE_DIR = CASE_DIR / "figures"
RAW_OUTPUT = FIGURE_DIR / "flat_plate_wake_raw.png"
TEX_OUTPUT = FIGURE_DIR / "flat_plate_wake.tex"
PDF_OUTPUT = FIGURE_DIR / "flat_plate_wake.pdf"
PNG_OUTPUT = FIGURE_DIR / "flat_plate_wake.png"
MANIFEST_OUTPUT = FIGURE_DIR / "flat_plate_wake_scene.json"
FIGURE_WIDTH_MM = 125.0
FIGURE_HEIGHT_MM = 45.0


def native_inputs() -> tuple[Path, Path]:
    particle_path = coupled_frames(SOLUTION_DIR)[-1][1]
    step = int(particle_path.stem.removeprefix("vpm_"))
    return (particle_path, SOLUTION_DIR / "vlm" / f"vlm_{step:06d}.vtp")


def read_native_state(particle_input: Path, surface_input: Path) -> dict[str, object]:
    saved = particle_state(particle_input)
    corners = saved["vlm"]["panel_corner_position"]
    plate_velocity = saved["vlm"]["kinematic_velocity"].mean(axis=0)
    bounds = np.array([corners.min(axis=(0, 1)), corners.max(axis=(0, 1))])
    return {
        "position": saved["position"],
        "omega": np.linalg.norm(saved["vorticity"], axis=1),
        "plate_bounds": bounds,
        "plate_corners": corners.reshape(-1, 3),
        "kinematic_velocity": plate_velocity,
        "particle_count": len(saved["position"]),
        "panel_count": len(corners),
        "step": saved["step"],
        "time": saved["time"],
    }


def prepare_geometry(state: dict[str, object], directory: Path) -> tuple[Path, Path]:
    position = np.asarray(state["position"])
    omega = np.asarray(state["omega"])
    omega_max = float(omega.max())
    bounds = np.asarray(state["plate_bounds"])
    span = float(np.ptp(bounds[:, 1]))
    glyph_radius = span * (0.0012 + 0.0065 * np.power(omega / omega_max, 0.55))
    particles = pv.PolyData(position)
    particles.point_data["vorticity_magnitude"] = omega.astype(np.float32)
    particles.point_data["glyph_radius"] = glyph_radius.astype(np.float32)
    particle_path = directory / "particles.vtp"
    particles.save(particle_path, binary=True)
    velocity = np.asarray(state["kinematic_velocity"])
    direction = velocity / np.linalg.norm(velocity)
    plate_centre = bounds.mean(axis=0)
    start = plate_centre + direction * (0.03 * span) + [0.0, 0.0, 0.12 * span]
    arrow_length = 0.24 * span
    arrow = pv.Arrow(
        start=start,
        direction=direction,
        tip_length=0.26,
        tip_radius=0.1,
        shaft_radius=0.045,
        scale=arrow_length,
    )
    arrow_path = directory / "motion_arrow.vtp"
    arrow.save(arrow_path, binary=True)
    state["arrow_starts"] = [start.tolist()]
    state["arrow_ends"] = [(start + arrow_length * direction).tolist()]
    state["glyph_radius_range"] = [float(glyph_radius.min()), float(glyph_radius.max())]
    state["arrow_length"] = arrow_length
    state["arrow_shaft_radius"] = 0.045 * arrow_length
    return (particle_path, arrow_path)


def camera_for_state(state: dict[str, object]) -> dict[str, object]:
    points = np.asarray(state["position"])
    all_points = np.vstack(
        (points, state["plate_corners"], state["arrow_starts"], state["arrow_ends"])
    )
    margin = max(state["glyph_radius_range"][-1], 0.1 * state["arrow_length"])
    offsets = np.array(np.meshgrid(*[[-margin, margin]] * 3)).T.reshape(-1, 3)
    all_points = (all_points[:, None, :] + offsets).reshape(-1, 3)
    centre = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    toward_camera = np.array([-0.55, -1.0, 0.7])
    toward_camera /= np.linalg.norm(toward_camera)
    view_direction = -toward_camera
    screen_right = np.array([1.0, 0.0, 0.0]) - toward_camera[0] * toward_camera
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(screen_right, view_direction)
    relative = all_points - centre
    horizontal = relative @ screen_right
    vertical = relative @ screen_up
    depth = relative @ toward_camera
    view_angle = 24.0
    tangent = np.tan(np.radians(view_angle / 2))
    aspect = FIGURE_WIDTH_MM / FIGURE_HEIGHT_MM
    distance = 1.015 * max(
        np.max(depth + np.abs(horizontal) / (aspect * tangent * 0.94)),
        np.max((vertical / tangent + 0.88 * depth) / 0.63),
        np.max((-vertical / tangent + 0.52 * depth) / 0.7),
    )

    def focus_intervals(distance):
        remaining = distance - depth
        return (
            np.max(horizontal - 0.94 * aspect * tangent * remaining),
            np.min(horizontal + 0.94 * aspect * tangent * remaining),
            np.max(vertical - 0.88 * tangent * remaining),
            np.min(vertical + 0.52 * tangent * remaining),
        )

    near = float(depth.max()) + margin
    far = distance
    for _ in range(60):
        candidate = (near + far) / 2
        left, right, bottom, top = focus_intervals(candidate)
        if left <= right and bottom <= top:
            far = candidate
        else:
            near = candidate
    distance = far * 1.015
    left, right, bottom, top = focus_intervals(distance)
    focal_point = centre + 0.5 * (left + right) * screen_right + 0.5 * (bottom + top) * screen_up
    position = focal_point + distance * toward_camera
    return {
        "projection": "perspective",
        "focal_point_m": focal_point.tolist(),
        "position_m": position.tolist(),
        "view_up": screen_up.tolist(),
        "view_angle_degrees": view_angle,
        "image_pixels": [2500, 900],
    }


def vorticity_colormap():
    return matplotlib.colormaps[theme.get_colormap("field_vorticity")]


def plate_velocity_tex(velocity: np.ndarray) -> str:
    terms = []
    for value, axis in zip(velocity, "xyz", strict=True):
        if abs(value) > 1e-10:
            terms.append(f"{value:+.2g}\\,\\mathbf{{e}}_{axis}")
    return "".join(terms).lstrip("+")


def overlay_document(
    omega_min: float, omega_max: float, camera: dict, arrow_tip: np.ndarray
) -> str:
    position = np.asarray(camera["position_m"])
    forward = np.asarray(camera["focal_point_m"]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera["view_up"])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = arrow_tip - position
    tangent = np.tan(np.radians(camera["view_angle_degrees"] / 2))
    depth = relative @ forward
    aspect = FIGURE_WIDTH_MM / FIGURE_HEIGHT_MM
    label_x = FIGURE_WIDTH_MM * (0.5 + relative @ right / (2 * aspect * depth * tangent))
    label_y = FIGURE_HEIGHT_MM * (0.5 + relative @ up / (2 * depth * tangent)) + 4.0
    label_y = min(label_y, FIGURE_HEIGHT_MM - 6.5)
    colormap = vorticity_colormap()
    bar_bottom = 6.5
    bar_top = 9.1
    tick_y = 5.9
    segments = []
    for index in range(48):
        red, green, blue, _ = colormap((index + 0.5) / 48.0)
        name = f"omega{index:02d}"
        x0 = 3.0 + 39.0 * index / 48.0
        x1 = 3.0 + 39.0 * (index + 1) / 48.0
        segments.append(f"\\definecolor{{{name}}}{{rgb}}{{{red:.6f},{green:.6f},{blue:.6f}}}")
        segments.append(
            f"\\fill[{name}] ({x0:.4f},{bar_bottom:.1f}) rectangle ({x1:.4f},{bar_top:.1f});"
        )
    bar = "\n".join(segments)
    source = f"\\documentclass[tikz,border=0pt]{{standalone}}\n\\usepackage[T1]{{fontenc}}\n\\usepackage{{newpxtext,newpxmath}}\n\\usepackage{{graphicx}}\n\\usepackage{{tikz}}\n\\begin{{document}}\n\\begin{{tikzpicture}}[x=1mm,y=1mm,font=\\fontsize{{10.95}}{{13.1}}\\selectfont]\n  \\node[anchor=south west,inner sep=0] at (0,0)\n    {{\\includegraphics[width={FIGURE_WIDTH_MM:.1f}mm]{{flat_plate_wake_raw.png}}}};\n  \\node[anchor=south,inner sep=0.7mm]\n        at ({label_x:.3f},{label_y:.3f}) {{$\\mathbf{{U}}_{{\\mathrm{{plate}}}}$}};\n  \\node[anchor=west] at (45.0,7.8) {{$\\lvert\\boldsymbol{{\\omega}}\\rvert\\;[\\mathrm{{s}}^{{-1}}]$}};\n{bar}\n  \\draw[line width=0.6pt] (3.0,{bar_bottom:.1f}) rectangle (42.0,{bar_top:.1f});\n  \\node[anchor=north west] at (3.0,{tick_y:.1f}) {{{omega_min:.2g}}};\n  \\node[anchor=north east] at (42.0,{tick_y:.1f}) {{{omega_max:.2g}}};\n\\end{{tikzpicture}}\n\\end{{document}}\n"
    return source


def scene_assets(work, state, surface_input, figure_format):
    omega = np.asarray(state["omega"])
    omega_min, omega_max = float(omega.min()), float(omega.max())
    particles, arrows = prepare_geometry(state, work)
    camera = camera_for_state(state)
    colormap = vorticity_colormap()
    color_points = [[fraction, *colormap(fraction)[:3]] for fraction in np.linspace(0, 1, 9)]
    arguments = [
        "--particles",
        particles,
        "--surface",
        surface_input,
        "--arrows",
        arrows,
        "--output",
        work / RAW_OUTPUT.name,
        "--state-output",
        work / "flat_plate_wake.pvsm",
        "--omega-min",
        repr(omega_min),
        "--omega-max",
        repr(omega_max),
        "--camera",
        json.dumps(camera),
        "--color-points",
        json.dumps(color_points),
    ]
    overlay = overlay_document(omega_min, omega_max, camera, np.asarray(state["arrow_ends"][0]))
    return {
        "renderer": (ASSETS_DIR / "render_flat_plate_paraview.py", arguments),
        "images": {RAW_OUTPUT.name: RAW_OUTPUT.name},
        "files": {
            "particles.vtp": "auxiliary/flat_plate_wake/particles.vtp",
            "motion_arrow.vtp": "auxiliary/flat_plate_wake/motion_arrow.vtp",
            "flat_plate_wake.pvsm": "auxiliary/flat_plate_wake/flat_plate_wake.pvsm",
        },
        "documents": [(TEX_OUTPUT.stem, overlay)],
        "format": figure_format,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    args = parser.parse_args()
    particle_input, surface_input = native_inputs()
    state = read_native_state(particle_input, surface_input)
    export_scene(
        FIGURE_DIR, lambda work: scene_assets(work, state, surface_input, args.format), dpi=400
    )
    final_output = PNG_OUTPUT if args.format == "png" else PDF_OUTPUT
    print(f"  Saved: {final_output}")


if __name__ == "__main__":
    main()
