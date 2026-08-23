"""Shared plotting utilities for the quadcopter tutorial."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CASE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = CASE_DIR / "samples" / "quadcopter"
FIGURES_DIR = CASE_DIR / "figures"
THEME_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"


def _load_theme():
    if not THEME_PATH.exists():
        raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
    spec = importlib.util.spec_from_file_location("mpl_setup", THEME_PATH)
    theme = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theme)
    theme.set_style()
    return theme


_theme = _load_theme()
_COLORS = _theme.COLORS


def load_integrals(samples_dir: Path) -> pd.DataFrame:
    csv_path = samples_dir / "flow_integrals.csv"
    if not csv_path.exists():
        raise SystemExit(f"No sampled flow integrals found in {samples_dir}")
    return pd.read_csv(csv_path)


def plot_particle_count(
    samples_dir: Path,
    figures_dir: Path,
    figure_format: str = "png",
) -> None:
    data = load_integrals(samples_dir)
    fig, ax = plt.subplots(figsize=_theme.figure_size("single"))
    ax.plot(data["time"], data["n_particles_total"], "-o", color=_COLORS["TUDcyan"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Particles [-]")
    ax.set_title("Quadcopter particle count evolution")
    figures_dir.mkdir(parents=True, exist_ok=True)
    _theme.save_fig(
        fig,
        figures_dir / "quadcopter_particle_count.png",
        figure_format=figure_format,
    )


def plot_vorticity_history(
    samples_dir: Path,
    figures_dir: Path,
    figure_format: str = "png",
) -> None:
    data = load_integrals(samples_dir)
    fig, ax = plt.subplots(figsize=_theme.figure_size("single"))
    ax.plot(data["time"], data["total_enstrophy"], "-o", color=_COLORS["VPMpurple"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Enstrophy [m$^3$/s$^2$]")
    ax.set_title("Quadcopter wake enstrophy history")
    figures_dir.mkdir(parents=True, exist_ok=True)
    _theme.save_fig(
        fig,
        figures_dir / "quadcopter_vorticity_history.png",
        figure_format=figure_format,
    )
