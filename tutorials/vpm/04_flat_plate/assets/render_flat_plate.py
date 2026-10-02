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
FIGURE_HEIGHT_MM = 62.5


def native_inputs() -> tuple[Path, Path]:
    """Select the latest jointly published accepted particle and VLM state."""
    metadata = json.loads((SOLUTION_DIR / "vpm_metadata.json").read_text())
    accepted_step = int(metadata["state"]["step"])
    vlm_directory = SOLUTION_DIR / "vlm"
    if not vlm_directory.is_dir():
        vlm_directory = SOLUTION_DIR
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
    offset = np.array([0.0, 0.0, 0.055 * span])
    arrow_origin = plate_centre - direction * (0.2 * span) + offset
    arrows = []
    arrow_starts = []
    for span_fraction in (-0.55, 0.0, 0.55):
        start = arrow_origin + np.array([0.0, span_fraction * span, 0.0])
        arrow_starts.append(start.tolist())
        arrows.append(
            pv.Arrow(
                start=start,
                direction=direction,
                tip_length=0.28,
                tip_radius=0.11,
                shaft_radius=0.045,
                scale=0.18 * span,
            )
        )
    merged = arrows[0].merge(arrows[1:])
    arrow_path = directory / "motion_arrows.vtp"
    merged.save(arrow_path, binary=True)
    state["arrow_starts"] = arrow_starts
    state["glyph_radius_range"] = [float(glyph_radius.min()), float(glyph_radius.max())]
    state["arrow_length"] = 0.18 * span
    return particle_path, arrow_path


def camera_for_state(state: dict[str, object]) -> dict[str, object]:
    points = np.asarray(state["position"])
    plate_bounds = np.asarray(state["plate_bounds"])
    corners = np.array(np.meshgrid(*zip(*plate_bounds))).T.reshape(-1, 3)
    all_points = np.vstack((points, corners, np.asarray(state["arrow_starts"])))
    centre = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    extent = np.ptp(all_points, axis=0)
    span = max(float(extent.max()), 1e-6)
    position = centre + span * np.array([0.0, -1.1, 0.5])
    view_direction = centre - position
    view_direction /= np.linalg.norm(view_direction)
    screen_right = np.cross(view_direction, [0.0, 0.0, 1.0])
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(screen_right, view_direction)
    projected_width = np.ptp(all_points @ screen_right)
    projected_height = np.ptp(all_points @ screen_up)
    scale = 1.12 * max(projected_height / 2, projected_width / 4)
    return {
        "projection": "parallel",
        "focal_point_m": centre.tolist(),
        "position_m": position.tolist(),
        "view_up": [0.0, 0.0, 1.0],
        "parallel_scale": float(scale),
        "image_pixels": [1800, 900],
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


def write_overlay(
    omega_min: float, omega_max: float, velocity: np.ndarray, *, time: float | None = None
) -> None:
    colormap = vorticity_colormap()
    bar_bottom = 7.7
    bar_top = 10.3
    bar_label_y = 11.1
    tick_y = 7.1
    segments = []
    for index in range(48):
        red, green, blue, _ = colormap((index + 0.5) / 48.0)
        name = f"omega{index:02d}"
        x0 = 80.0 + 39.0 * index / 48.0
        x1 = 80.0 + 39.0 * (index + 1) / 48.0
        segments.append(f"\\definecolor{{{name}}}{{rgb}}{{{red:.6f},{green:.6f},{blue:.6f}}}")
        segments.append(
            f"\\fill[{name}] ({x0:.4f},{bar_bottom:.1f}) rectangle ({x1:.4f},{bar_top:.1f});"
        )
    bar = "\n".join(segments)
    # Glyph interpretation is recorded in the scene JSON; retain only physical time.
    caption = rf"$t={time:.2g}$ s" if time is not None else ""
    source = rf"""\documentclass[tikz,border=0pt]{{standalone}}
\usepackage[T1]{{fontenc}}
\usepackage{{newpxtext,newpxmath}}
\usepackage{{graphicx}}
\usepackage{{tikz}}
\begin{{document}}
\begin{{tikzpicture}}[x=1mm,y=1mm,font=\fontsize{{10.95}}{{13.1}}\selectfont]
  \node[anchor=south west,inner sep=0] at (0,0)
    {{\includegraphics[width={FIGURE_WIDTH_MM:.1f}mm]{{flat_plate_wake_raw.png}}}};
  \node[anchor=north west,fill=white,fill opacity=0.90,text opacity=1,
        rounded corners=0.8mm,inner xsep=2.2mm,inner ysep=1.5mm]
        at (2.8,{FIGURE_HEIGHT_MM - 2.8:.1f})
        {{ $\mathbf{{U}}_{{\mathrm{{plate}}}}={plate_velocity_tex(velocity)}\;\mathrm{{m\,s^{{-1}}}}$}};
  \node[anchor=south west,fill=white,inner sep=1.2mm] at (2.8,2.8) {{{caption}}};
  \fill[white,fill opacity=0.90] (76.8,3.5) rectangle (122.3,14.5);
  \node[anchor=south] at (99.5,{bar_label_y:.1f}) {{$\lvert\boldsymbol{{\omega}}\rvert\;[\mathrm{{s}}^{{-1}}]$}};
{bar}
  \draw[line width=0.6pt] (80.0,{bar_bottom:.1f}) rectangle (119.0,{bar_top:.1f});
  \node[anchor=north west] at (80.0,{tick_y:.1f}) {{{omega_min:.2g}}};
  \node[anchor=north east] at (119.0,{tick_y:.1f}) {{{omega_max:.2g}}};
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
    with tempfile.TemporaryDirectory(prefix="flat-plate-render-") as temporary:
        temporary_path = Path(temporary)
        particles, arrows = prepare_geometry(state, temporary_path)
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
    write_overlay(
        omega_min, omega_max, np.asarray(state["kinematic_velocity"]), time=float(state["time"])
    )
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
            "direction": (
                np.asarray(state["kinematic_velocity"])
                / np.linalg.norm(state["kinematic_velocity"])
            ).tolist(),
            "starts_m": state["arrow_starts"],
            "length_m": state["arrow_length"],
        },
        "camera": camera,
        "output": {
            "raw_png": str(RAW_OUTPUT),
            "raw_png_sha256": sha256(RAW_OUTPUT),
            "overlay_tex": str(TEX_OUTPUT),
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
