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


ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
SOLUTION_DIR = CASE_DIR / "solution" / "exp_moving_aoa12"
FIGURE_DIR = CASE_DIR / "figures"
PARTICLE_INPUT = SOLUTION_DIR / "vpm" / "vpm_000197.h5"
SURFACE_INPUT = SOLUTION_DIR / "vlm" / "vlm_000197.vtp"
RAW_OUTPUT = FIGURE_DIR / "flat_plate_wake_raw.png"
TEX_OUTPUT = FIGURE_DIR / "flat_plate_wake.tex"
PDF_OUTPUT = FIGURE_DIR / "flat_plate_wake.pdf"
PNG_OUTPUT = FIGURE_DIR / "flat_plate_wake.png"
MANIFEST_OUTPUT = FIGURE_DIR / "flat_plate_wake_scene.json"
FIGURE_WIDTH_MM = 125.0
FIGURE_HEIGHT_MM = 62.5
EXPECTED_STEP = 197
EXPECTED_TIME = 2.4625
EXPECTED_PARTICLES = 11_032
EXPECTED_PANELS = 224


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_pvpython() -> Path:
    return Path(find_executable("pvpython"))


def read_native_state() -> dict[str, object]:
    if not PARTICLE_INPUT.is_file() or not SURFACE_INPUT.is_file():
        raise FileNotFoundError("The accepted step-197 HDF5 and VLM VTP files are required.")
    with h5py.File(PARTICLE_INPUT, "r") as state:
        position = np.asarray(state["particles/position"], dtype=float)
        vorticity = np.asarray(state["particles/vorticity"], dtype=float)
        corners = np.asarray(state["solver/vlm/panel_corner_position"], dtype=float)
        kinematic_velocity = np.asarray(state["solver/vlm/kinematic_velocity"], dtype=float)
        step = int(state["solver"].attrs["step"])
        time = float(state["solver"].attrs["time"])
    if step != EXPECTED_STEP or not np.isclose(time, EXPECTED_TIME, atol=1e-12):
        raise ValueError(
            f"Expected step/time {EXPECTED_STEP}/{EXPECTED_TIME}, found {step}/{time}."
        )
    if position.shape != (EXPECTED_PARTICLES, 3):
        raise ValueError(f"Expected {EXPECTED_PARTICLES} particles, found {position.shape[0]}.")
    if corners.shape != (EXPECTED_PANELS, 4, 3):
        raise ValueError(f"Expected {EXPECTED_PANELS} VLM panels, found {corners.shape[0]}.")
    expected_velocity = np.array([-10.0, 0.0, 0.0])
    if not np.allclose(kinematic_velocity, expected_velocity, atol=1e-7):
        raise ValueError("The rendered plate does not have the expected -x kinematic velocity.")
    omega = np.linalg.norm(vorticity, axis=1)
    return {
        "position": position,
        "omega": omega,
        "plate_bounds": np.array([corners.min(axis=(0, 1)), corners.max(axis=(0, 1))]),
        "kinematic_velocity": expected_velocity,
        "step": step,
        "time": time,
    }


def prepare_geometry(state: dict[str, object], directory: Path) -> tuple[Path, Path]:
    position = np.asarray(state["position"])
    omega = np.asarray(state["omega"])
    omega_max = float(omega.max())
    # The spheres are visual glyphs. Their radii are not particle core radii.
    glyph_radius = 0.012 + 0.065 * np.power(omega / omega_max, 0.55)
    particles = pv.PolyData(position)
    particles.point_data["vorticity_magnitude"] = omega.astype(np.float32)
    particles.point_data["glyph_radius"] = glyph_radius.astype(np.float32)
    particle_path = directory / "particles.vtp"
    particles.save(particle_path, binary=True)

    bounds = np.asarray(state["plate_bounds"])
    arrow_start_x = float(bounds[1, 0] + 1.55)
    arrow_z = float(bounds[1, 2] + 0.55)
    arrows = []
    arrow_starts = []
    for span_fraction in (-0.55, 0.0, 0.55):
        start = np.array([arrow_start_x, span_fraction * 10.0, arrow_z])
        arrow_starts.append(start.tolist())
        arrows.append(
            pv.Arrow(
                start=start,
                direction=(-1.0, 0.0, 0.0),
                tip_length=0.28,
                tip_radius=0.11,
                shaft_radius=0.045,
                scale=1.8,
            )
        )
    merged = arrows[0].merge(arrows[1:])
    arrow_path = directory / "motion_arrows.vtp"
    merged.save(arrow_path, binary=True)
    state["arrow_starts"] = arrow_starts
    state["glyph_radius_range"] = [float(glyph_radius.min()), float(glyph_radius.max())]
    return particle_path, arrow_path


def write_overlay(omega_min: float, omega_max: float) -> None:
    colormap = matplotlib.colormaps["viridis"]
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
        {{Plate velocity: $\mathbf{{U}}_{{\mathrm{{plate}}}}=-10\,\mathbf{{e}}_x\;\mathrm{{m\,s^{{-1}}}}$}};
  \fill[white,fill opacity=0.90] (76.8,3.5) rectangle (122.3,14.5);
  \node[anchor=south] at (99.5,{bar_label_y:.1f}) {{$\lvert\boldsymbol{{\omega}}\rvert\;[\mathrm{{s}}^{{-1}}]$}};
{bar}
  \draw[line width=0.35pt] (80.0,{bar_bottom:.1f}) rectangle (119.0,{bar_top:.1f});
  \node[anchor=north west] at (80.0,{tick_y:.1f}) {{{omega_min:.3g}}};
  \node[anchor=north east] at (119.0,{tick_y:.1f}) {{{omega_max:.3g}}};
\end{{tikzpicture}}
\end{{document}}
"""
    TEX_OUTPUT.write_text(source)


def compile_overlay(figure_format: str) -> None:
    with tempfile.TemporaryDirectory(prefix="flat-plate-overlay-") as temporary:
        compiled_pdf = Path(temporary) / PDF_OUTPUT.name
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                temporary,
                TEX_OUTPUT.name,
            ],
            cwd=FIGURE_DIR,
            check=True,
        )
        if figure_format == "png":
            pdftoppm = shutil.which("pdftoppm")
            if pdftoppm is None:
                raise FileNotFoundError("pdftoppm is required to export the rendered PNG.")
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
        else:
            shutil.copy2(compiled_pdf, PDF_OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    state = read_native_state()
    omega = np.asarray(state["omega"])
    omega_min = float(omega.min())
    omega_max = float(omega.max())
    pvpython = find_pvpython()
    with tempfile.TemporaryDirectory(prefix="flat-plate-render-") as temporary:
        temporary_path = Path(temporary)
        particles, arrows = prepare_geometry(state, temporary_path)
        subprocess.run(
            [
                str(pvpython),
                str(ASSETS_DIR / "render_flat_plate_paraview.py"),
                "--particles",
                str(particles),
                "--surface",
                str(SURFACE_INPUT),
                "--arrows",
                str(arrows),
                "--output",
                str(RAW_OUTPUT),
                "--omega-min",
                repr(omega_min),
                "--omega-max",
                repr(omega_max),
            ],
            check=True,
        )
    if not RAW_OUTPUT.is_file():
        raise FileNotFoundError(f"ParaView did not produce {RAW_OUTPUT}.")
    write_overlay(omega_min, omega_max)
    compile_overlay(args.format)

    manifest = {
        "description": "Final moving-flat-plate wake rendering",
        "input": {
            "particles": str(PARTICLE_INPUT),
            "particles_sha256": sha256(PARTICLE_INPUT),
            "vlm_surface": str(SURFACE_INPUT),
            "vlm_surface_sha256": sha256(SURFACE_INPUT),
        },
        "state": {
            "step": state["step"],
            "time_s": state["time"],
            "particle_count": EXPECTED_PARTICLES,
            "panel_count": EXPECTED_PANELS,
            "plate_velocity_m_per_s": np.asarray(state["kinematic_velocity"]).tolist(),
            "vorticity_magnitude_range_per_s": [omega_min, omega_max],
        },
        "glyphs": {
            "meaning": "visual particle spheres, not particle core radii",
            "field": "kernel-reconstructed vorticity at particle positions",
            "radius_m": "0.012 + 0.065*(|omega|/max(|omega|))^0.55",
            "radius_range_m": state["glyph_radius_range"],
            "colour_map": "viridis, linear in |omega|",
        },
        "motion_arrows": {
            "direction": [-1.0, 0.0, 0.0],
            "starts_m": state["arrow_starts"],
            "length_m": 1.8,
        },
        "camera": {
            "projection": "parallel",
            "focal_point_m": [-11.6, 0.0, -0.75],
            "position_m": [-11.6, -28.0, 12.0],
            "view_up": [0.0, 0.0, 1.0],
            "parallel_scale": 7.0,
            "image_pixels": [1800, 900],
        },
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
