#!/usr/bin/env python3
"""Reproduce the final moving-flat-plate wake rendering and its LaTeX overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import h5py
import matplotlib
import numpy as np
import pyvista as pv

from openonda.executables import find_executable
from openonda import plotting as theme
from source.solution_layout import vpm_backup_files


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
    """Select the latest jointly published accepted particle and VLM state."""
    metadata = json.loads((SOLUTION_DIR / "vpm_metadata.json").read_text())
    accepted_step = int(metadata["state"]["step"])
    vlm_directory = SOLUTION_DIR / "vlm"
    for particle_path in reversed(vpm_backup_files(SOLUTION_DIR)):
        step = int(particle_path.stem.removeprefix("vpm_"))
        if step <= accepted_step:
            surface_path = vlm_directory / f"vlm_{step:06d}.vtp"
            if surface_path.is_file():
                return particle_path, surface_path
    raise FileNotFoundError("No jointly published accepted particle/VLM state is available")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_pvpython() -> Path:
    return Path(find_executable("pvpython"))


def read_native_state(particle_input: Path, surface_input: Path) -> dict[str, object]:
    with h5py.File(particle_input, "r") as state:
        position = np.asarray(state["particles/position"], dtype=float)
        vorticity = np.asarray(state["particles/vorticity"], dtype=float)
        corners = np.asarray(state["solver/vlm/panel_corner_position"], dtype=float)
        kinematic_velocity = np.asarray(state["solver/vlm/kinematic_velocity"], dtype=float)
        step = int(state["solver"].attrs["step"])
        time = float(state["solver"].attrs["time"])
    if step != int(particle_input.stem.removeprefix("vpm_")) or not np.isfinite(time):
        raise ValueError("Native backup step/time do not match the selected file")
    accepted = json.loads((SOLUTION_DIR / "vpm_metadata.json").read_text())["state"]
    if step > int(accepted["step"]) or time > float(accepted["time"]) + 1e-10:
        raise ValueError("Native backup is beyond the accepted solver state")
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or not len(position)
        or not np.isfinite(position).all()
    ):
        raise ValueError("Native particle positions are empty or invalid")
    if vorticity.shape != position.shape or not np.isfinite(vorticity).all():
        raise ValueError("Native particle vorticity is missing or invalid")
    if (
        corners.ndim != 3
        or corners.shape[1:] != (4, 3)
        or not len(corners)
        or not np.isfinite(corners).all()
    ):
        raise ValueError("Native VLM panels are empty or invalid")
    if kinematic_velocity.shape != (len(corners), 3) or not np.isfinite(kinematic_velocity).all():
        raise ValueError("Native plate kinematics are missing or invalid")
    plate_velocity = kinematic_velocity.mean(axis=0)
    if np.linalg.norm(plate_velocity) <= 1e-12 or not np.allclose(
        kinematic_velocity, plate_velocity, atol=1e-7, rtol=0
    ):
        raise ValueError("Native plate kinematics must have one nonzero velocity")
    surface = pv.read(surface_input)
    plate_bounds = np.array([corners.min(axis=(0, 1)), corners.max(axis=(0, 1))])
    surface_bounds = np.array(
        [
            [surface.bounds[0], surface.bounds[2], surface.bounds[4]],
            [surface.bounds[1], surface.bounds[3], surface.bounds[5]],
        ]
    )
    if surface.n_cells != len(corners) or not np.allclose(
        surface_bounds, plate_bounds, atol=1e-5, rtol=0
    ):
        raise ValueError("Native VLM surface does not match embedded backup geometry")
    surface_time = np.asarray(surface["time"])
    if not np.allclose(surface_time, time, atol=1e-10, rtol=0):
        raise ValueError("Native VLM surface time does not match backup time")
    omega = np.linalg.norm(vorticity, axis=1)
    if not np.isfinite(omega).all() or omega.max() <= 0:
        raise ValueError("Native particle vorticity has no finite nonzero signal")
    return {
        "position": position,
        "omega": omega,
        "plate_bounds": plate_bounds,
        "plate_corners": corners.reshape(-1, 3),
        "kinematic_velocity": plate_velocity,
        "particle_count": len(position),
        "panel_count": len(corners),
        "step": step,
        "time": time,
    }


def prepare_geometry(state: dict[str, object], directory: Path) -> tuple[Path, Path]:
    position = np.asarray(state["position"])
    omega = np.asarray(state["omega"])
    omega_max = float(omega.max())
    # The spheres are visual glyphs. Their radii are not particle core radii.
    bounds = np.asarray(state["plate_bounds"])
    span = float(np.ptp(bounds[:, 1]))
    if span <= 0:
        raise ValueError("Native plate has no span")
    glyph_radius = span * (0.0012 + 0.0065 * np.power(omega / omega_max, 0.55))
    particles = pv.PolyData(position)
    particles.point_data["vorticity_magnitude"] = omega.astype(np.float32)
    particles.point_data["glyph_radius"] = glyph_radius.astype(np.float32)
    particle_path = directory / "particles.vtp"
    particles.save(particle_path, binary=True)

    velocity = np.asarray(state["kinematic_velocity"])
    direction = velocity / np.linalg.norm(velocity)
    plate_centre = bounds.mean(axis=0)
    # One substantial physical vector above the plate, pointing into its motion.
    start = plate_centre + direction * (0.03 * span) + [0.0, 0.0, 0.12 * span]
    arrow_length = 0.24 * span
    arrow = pv.Arrow(
        start=start,
        direction=direction,
        tip_length=0.26,
        tip_radius=0.10,
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
    return particle_path, arrow_path


def camera_for_state(state: dict[str, object]) -> dict[str, object]:
    points = np.asarray(state["position"])
    all_points = np.vstack(
        (points, state["plate_corners"], state["arrow_starts"], state["arrow_ends"])
    )
    # Include sphere and arrow radii in the perspective framing, without cropping.
    margin = max(state["glyph_radius_range"][-1], 0.10 * state["arrow_length"])
    offsets = np.array(np.meshgrid(*[[-margin, margin]] * 3)).T.reshape(-1, 3)
    all_points = (all_points[:, None, :] + offsets).reshape(-1, 3)
    centre = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    # View from ahead of the moving plate and above its near-side tip.
    toward_camera = np.array([-0.55, -1.0, 0.70])
    toward_camera /= np.linalg.norm(toward_camera)
    view_direction = -toward_camera
    # Keep the projected chord/motion axis horizontal as the camera rises.
    # This fills the wide canvas without cropping the span or the wake.
    screen_right = np.array([1.0, 0.0, 0.0]) - toward_camera[0] * toward_camera
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(screen_right, view_direction)
    relative = all_points - centre
    horizontal = relative @ screen_right
    vertical = relative @ screen_up
    depth = relative @ toward_camera
    view_angle = 24.0
    tangent = np.tan(np.radians(view_angle / 2))
    # Explicit scene windows on the shorter canvas: x=3..97%, y=24..94%.
    # The lower band holds the vector colour bar; the upper edge stays compact.
    aspect = FIGURE_WIDTH_MM / FIGURE_HEIGHT_MM
    distance = 1.015 * max(
        np.max(depth + np.abs(horizontal) / (aspect * tangent * 0.94)),
        np.max((vertical / tangent + 0.88 * depth) / 0.63),
        np.max((-vertical / tangent + 0.52 * depth) / 0.70),
    )

    # A Cartesian bounding-box centre leaves the long perspective wake off-centre
    # in projection. Fit the focal point within the same authored scene window.
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
    assert left <= right and bottom <= top
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
    if not terms:
        raise ValueError("Plate velocity is zero")
    return "".join(terms).lstrip("+")


def write_overlay(omega_min: float, omega_max: float, camera: dict, arrow_tip: np.ndarray) -> None:
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
    label_x = FIGURE_WIDTH_MM * (0.5 + (relative @ right) / (2 * aspect * depth * tangent))
    label_y = FIGURE_HEIGHT_MM * (0.5 + (relative @ up) / (2 * depth * tangent)) + 4.0
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
    source = rf"""\documentclass[tikz,border=0pt]{{standalone}}
\usepackage[T1]{{fontenc}}
\usepackage{{newpxtext,newpxmath}}
\usepackage{{graphicx}}
\usepackage{{tikz}}
\begin{{document}}
\begin{{tikzpicture}}[x=1mm,y=1mm,font=\fontsize{{10.95}}{{13.1}}\selectfont]
  \node[anchor=south west,inner sep=0] at (0,0)
    {{\includegraphics[width={FIGURE_WIDTH_MM:.1f}mm]{{flat_plate_wake_raw.png}}}};
  \node[anchor=south,inner sep=0.7mm]
        at ({label_x:.3f},{label_y:.3f}) {{$\mathbf{{U}}_{{\mathrm{{plate}}}}$}};
  \node[anchor=west] at (45.0,7.8) {{$\lvert\boldsymbol{{\omega}}\rvert\;[\mathrm{{s}}^{{-1}}]$}};
{bar}
  \draw[line width=0.6pt] (3.0,{bar_bottom:.1f}) rectangle (42.0,{bar_top:.1f});
  \node[anchor=north west] at (3.0,{tick_y:.1f}) {{{omega_min:.2g}}};
  \node[anchor=north east] at (42.0,{tick_y:.1f}) {{{omega_max:.2g}}};
\end{{tikzpicture}}
\end{{document}}
"""
    TEX_OUTPUT.write_text(source)


def compile_overlay(figure_format: str) -> None:
    with tempfile.TemporaryDirectory(prefix="flat-plate-overlay-") as temporary:
        compiled_pdf = Path(temporary) / PDF_OUTPUT.name
        subprocess.run(
            [
                find_executable("pdflatex"),
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                temporary,
                TEX_OUTPUT.name,
            ],
            cwd=FIGURE_DIR,
            check=True,
        )
        if figure_format in ("png", "both"):
            pdftoppm = find_executable("pdftoppm")
            subprocess.run(
                [
                    pdftoppm,
                    "-png",
                    "-r",
                    "400",
                    "-singlefile",
                    str(compiled_pdf),
                    str(PNG_OUTPUT.with_suffix("")),
                ],
                check=True,
            )
        if figure_format in ("pdf", "both"):
            shutil.copy2(compiled_pdf, PDF_OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    particle_input, surface_input = native_inputs()
    state = read_native_state(particle_input, surface_input)
    omega = np.asarray(state["omega"])
    omega_min = float(omega.min())
    omega_max = float(omega.max())
    pvpython = find_pvpython()
    geometry_directory = FIGURE_DIR / "auxiliary" / "flat_plate_wake"
    geometry_directory.mkdir(parents=True, exist_ok=True)
    particles, arrows = prepare_geometry(state, geometry_directory)
    camera = camera_for_state(state)
    colormap = vorticity_colormap()
    color_points = [[fraction, *colormap(fraction)[:3]] for fraction in np.linspace(0, 1, 9)]
    subprocess.run(
        [
            str(pvpython),
            str(ASSETS_DIR / "render_flat_plate_paraview.py"),
            "--particles",
            str(particles),
            "--surface",
            str(surface_input),
            "--arrows",
            str(arrows),
            "--output",
            str(RAW_OUTPUT),
            "--state-output",
            str(geometry_directory / "flat_plate_wake.pvsm"),
            "--omega-min",
            repr(omega_min),
            "--omega-max",
            repr(omega_max),
            "--camera",
            json.dumps(camera),
            "--color-points",
            json.dumps(color_points),
        ],
        check=True,
    )
    if not RAW_OUTPUT.is_file():
        raise FileNotFoundError(f"ParaView did not produce {RAW_OUTPUT}.")
    write_overlay(omega_min, omega_max, camera, np.asarray(state["arrow_ends"][0]))
    compile_overlay(args.format)

    manifest = {
        "description": "Latest jointly published accepted moving-flat-plate wake rendering",
        "input": {
            "particles": str(particle_input),
            "particles_sha256": sha256(particle_input),
            "vlm_surface": str(surface_input),
            "vlm_surface_sha256": sha256(surface_input),
        },
        "state": {
            "step": state["step"],
            "accepted_state_step": int(
                json.loads((SOLUTION_DIR / "vpm_metadata.json").read_text())["state"]["step"]
            ),
            "time_s": state["time"],
            "particle_count": state["particle_count"],
            "panel_count": state["panel_count"],
            "plate_velocity_m_per_s": np.asarray(state["kinematic_velocity"]).tolist(),
            "vorticity_magnitude_range_per_s": [omega_min, omega_max],
        },
        "glyphs": {
            "meaning": "visual particle spheres, not particle core radii",
            "field": "kernel-reconstructed vorticity at particle positions",
            "radius_m": "plate_span*(0.0012 + 0.0065*(|omega|/max(|omega|))^0.55)",
            "radius_range_m": state["glyph_radius_range"],
            "colour_map": f"{theme.get_colormap('field_vorticity')}, linear in |omega|",
        },
        "motion_arrows": {
            "count": 1,
            "colour_rgb": [0.64, 0.64, 0.64],
            "direction": (
                np.asarray(state["kinematic_velocity"])
                / np.linalg.norm(state["kinematic_velocity"])
            ).tolist(),
            "starts_m": state["arrow_starts"],
            "length_m": state["arrow_length"],
            "shaft_radius_m": state["arrow_shaft_radius"],
        },
        "surface": {
            "visible": True,
            "color_rgb": [0.3764705882, 0.4078431373, 0.4235294118],
            "meaning": "Saved VLM plate surface, shown with shaded neutral material.",
        },
        "lighting": {
            "reference": "author's vortex_stretching.pvsm light kit",
            "shading": True,
            "tone_mapping": "filmic",
        },
        "camera": camera,
        "output": {
            "raw_png": str(RAW_OUTPUT),
            "raw_png_sha256": sha256(RAW_OUTPUT),
            "overlay_tex": str(TEX_OUTPUT),
            "paraview_state": str(geometry_directory / "flat_plate_wake.pvsm"),
            "particle_glyph_input": str(particles),
            "motion_arrow_input": str(arrows),
        },
    }
    final_output = PNG_OUTPUT if args.format == "png" else PDF_OUTPUT
    manifest["output"][f"final_{args.format}"] = str(final_output)
    manifest["output"][f"final_{args.format}_sha256"] = sha256(final_output)
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  Saved: {final_output}")
    print(f"  Scene record: {MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
