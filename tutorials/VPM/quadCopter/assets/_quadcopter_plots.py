#!/usr/bin/env python3
"""Generate quick diagnostics from QuadCopter VPM backups."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution" / "quadcopter"
FIGURES_DIR = CASE_DIR / "figures"
THEME_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"


def _load_theme() -> tuple[dict[str, str], object | None]:
    if not THEME_PATH.exists():
        raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
    spec = importlib.util.spec_from_file_location("mpl_setup", THEME_PATH)
    theme = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theme)
    theme.set_style()
    return dict(theme.COLORS), theme


_COLORS, _theme = _load_theme()


def extract_step(path: Path) -> int:
    match = re.search(r"_(\d{6})\.h5$", path.name)
    return int(match.group(1)) if match else -1


def find_backups(solution_dir: Path, pattern: str) -> list[Path]:
    files = sorted(solution_dir.glob(pattern), key=extract_step)
    if files:
        return files
    # Backward-compatible fallback.
    return sorted(solution_dir.glob("particle_data_*.h5"), key=extract_step)


def read_series(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = []
    particle_count = []
    vorticity_l2 = []

    for file in files:
        with h5py.File(file, "r") as handle:
            solver = handle["solver"].attrs
            times.append(float(solver.get("flow_time", 0.0)))
            particle_count.append(int(solver.get("number_of_particles", 0)))

            if "particles" in handle and "vorticity" in handle["particles"]:
                vort = handle["particles"]["vorticity"][:]
                vorticity_l2.append(float(np.linalg.norm(vort, axis=1).sum()))
            else:
                vorticity_l2.append(0.0)

    return np.array(times), np.array(particle_count), np.array(vorticity_l2)


def load_series(solution_dir: Path, pattern: str = "vpm_*.h5"):
    files = find_backups(solution_dir, pattern)
    if not files:
        raise SystemExit(f"No backup files found in {solution_dir}")
    return read_series(files)

def plot_particle_count(solution_dir: Path, figures_dir: Path, figure_format: str = "png") -> None:
    times, particle_count, _ = load_series(solution_dir)
    fig, ax = plt.subplots(figsize=_theme.figure_size("single"))
    ax.plot(times, particle_count, "-o", color=_COLORS["TUDcyan"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Particles [-]")
    ax.set_title("Quadcopter particle count evolution")
    out = figures_dir / "quadcopter_particle_count.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _theme.save_fig(fig, out, figure_format=figure_format)


def plot_vorticity_history(solution_dir: Path, figures_dir: Path, figure_format: str = "png") -> None:
    times, _, vorticity_l2 = load_series(solution_dir)
    fig, ax = plt.subplots(figsize=_theme.figure_size("single"))
    ax.plot(times, vorticity_l2, "-o", color=_COLORS["VPMpurple"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\sum \|\omega\|$ [1/s]")
    ax.set_title("Quadcopter vorticity magnitude history")
    out = figures_dir / "quadcopter_vorticity_history.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _theme.save_fig(fig, out, figure_format=figure_format)
