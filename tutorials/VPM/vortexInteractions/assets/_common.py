"""Shared utilities for vortexInteractions plot scripts.

Each plot script lives in assets/ and imports from here via::

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from _common import load_theme, build_arg_parser, ...
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ── Directory layout ─────────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).resolve().parent  # …/assets/
SCRIPT_DIR = ASSETS_DIR.parent  # …/vortexInteractions/
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"
THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"
FONT_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "DejaVuSerif.ttf"

# ── Physical constants  (match rings_setup.py) ────────────────────────────────
R0 = 1.0  # ring major radius [m]
GAMMA = np.pi  # circulation [m²/s]
CORE_RADIUS = 0.1  # initial core radius [m]
T_REF = R0**2 / GAMMA  # T₀ = R₀²/Γ  [s]

_eps = CORE_RADIUS / R0
_C = -0.558 - 1.12 * _eps**2 - 5.0 * _eps**4
U_REF = GAMMA / (4 * np.pi * R0) * (np.log(8 / _eps) + _C)  # Saffman speed [m/s]

# Reference energy dissipation rate scale (per unit density)
E_REF = GAMMA**2 * R0  # [m⁵/s²]  kinetic energy scale for a ring
P_REF = E_REF / T_REF  # [m⁵/s³]  dissipation rate scale = Γ³/R₀

CM = 1 / 2.54  # cm → inch


# ── Theme ─────────────────────────────────────────────────────────────────────


def load_theme() -> tuple[dict[str, str], object | None]:
    """Load the OpenONDA matplotlib theme. Returns (COLORS dict, theme module)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    theme = None
    if THEME_PATH.exists():
        spec = importlib.util.spec_from_file_location("mpl_setup", THEME_PATH)
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        try:
            theme.set_style()
        except Exception:
            pass

    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "DejaVu Serif"

    if theme is not None and hasattr(theme, "COLORS"):
        return dict(theme.COLORS), theme
    return {}, theme


# ── Argument parser ───────────────────────────────────────────────────────────


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--solution-dir", default=str(SOLUTION_DIR), help="Root solution directory.")
    p.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Output directory for figures.")
    p.add_argument("--dpi", type=int, default=400, help="Figure DPI.")
    return p


# ── H5 helpers ────────────────────────────────────────────────────────────────


def load_total_circulation(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_star, circ_norm) from H5 backups.

    Computes Σ|Γᵢ| at each snapshot and normalises by the initial value
    (first snapshot), so the curve starts at exactly 1.
    """
    times, circs = [], []
    for path in sorted(h5_files):
        try:
            with h5py.File(path, "r") as f:
                circ = f["particles/circulation"][:]
                t = float(f["solver"].attrs.get("flow_time", 0.0))
                total_circ = float(np.sum(np.linalg.norm(circ, axis=1)))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
        times.append(t)
        circs.append(total_circ)

    if not circs:
        return np.array([]), np.array([])

    t_arr = np.array(times) / T_REF
    c_arr = np.array(circs)
    Gamma0 = c_arr[0]  # normalise by the actual initial total strength

    blow_up = c_arr > 500.0 * Gamma0
    if blow_up.any():
        idx = int(blow_up.argmax())
        print(f"Stopping at {Path(h5_files[idx]).name}: blow-up detected.")
        t_arr = t_arr[:idx]
        c_arr = c_arr[:idx]

    return t_arr, c_arr / Gamma0


def _ring_props_from_h5(path) -> dict | None:
    """Return {ring_id: {time, x_centroid, major_R, strength_max}} or None."""
    try:
        with h5py.File(path, "r") as f:
            pos = f["particles/position"][:]
            vort = f["particles/vorticity"][:]
            gid = f["particles/group_id"][:]
            smag = np.linalg.norm(f["particles/circulation"][:], axis=1)
            t = float(f["solver"].attrs.get("flow_time", 0.0))
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

    out: dict = {}
    for rid in np.unique(gid):
        m_ = gid == rid
        vm = np.linalg.norm(vort[m_], axis=1)
        core = vm > 0.1 * vm.max()
        if not core.any():
            continue
        pc, vc = pos[m_][core], vm[core]
        w = vc.sum()
        xc = (pc[:, 0] * vc).sum() / w
        Rc = (np.sqrt(pc[:, 1] ** 2 + pc[:, 2] ** 2) * vc).sum() / w
        out[rid] = dict(time=t, x_centroid=xc, major_R=Rc, strength_max=smag[m_].max())
    return out


def load_ring_data(h5_files: list) -> dict:
    """Read all H5 backups; stop at blow-up. Returns {ring_id: [dict, ...]}."""
    data: dict = {}
    for path in h5_files:
        res = _ring_props_from_h5(path)
        if not res:
            continue
        if any(r["strength_max"] > 500.0 for r in res.values()):
            print(f"Stopping at {Path(path).name}: blow-up detected.")
            break
        for rid, vals in res.items():
            data.setdefault(rid, []).append(vals)
    return data


def normalise_ring_data(raw: dict) -> dict:
    """Convert raw ring dicts to {rid: {x_norm, R_norm}} arrays, masking outliers."""
    out: dict = {}
    for rid, entries in raw.items():
        x = np.array([d["x_centroid"] for d in entries]) / R0 + 0.5 / R0
        R = np.array([d["major_R"] for d in entries]) / R0
        valid = (np.abs(x) < 1000) & (np.abs(R) < 1000)
        out[rid] = {"x_norm": x[valid], "R_norm": R[valid]}
    return out


# ── Log-file parser ───────────────────────────────────────────────────────────


def parse_log(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (flow_times, -nuΩ, dE/dt) arrays from a VPM solver log."""
    path = Path(path)
    if not path.exists():
        print(f"(Warning) Log not found: {path}")
        return np.array([]), np.array([]), np.array([])

    t_pat = re.compile(r"Time-step:\s*\d+\s+Flow time:\s*([\d.E+\-]+)\s*s")
    nuEns_pat = re.compile(r"Viscous dissipation.{1,15}:\s*([-\d.e+]+)")
    de_pat = re.compile(r"Energy decay rate.{1,15}:\s*([-\d.e+]+)")

    times, nu_ens_values, dts = [], [], []
    cur_t = cur_nu_ens = None
    for line in path.open(encoding="utf-8", errors="replace"):
        if mt := t_pat.search(line):
            cur_t, cur_nu_ens = float(mt.group(1)), None
        elif mn := nuEns_pat.search(line):
            cur_nu_ens = float(mn.group(1))
        elif (md := de_pat.search(line)) and cur_t is not None and cur_nu_ens is not None:
            times.append(cur_t)
            nu_ens_values.append(cur_nu_ens)
            dts.append(float(md.group(1)))
            cur_nu_ens = None

    t = np.array(times)
    nuEns = np.array(nu_ens_values)
    de = np.array(dts)
    valid = (np.abs(nuEns) < 1000) & (np.abs(de) < 1000)
    return t[valid], nuEns[valid], de[valid]


# ── Figure helpers ────────────────────────────────────────────────────────────


def save_fig(fig, path, dpi: int = 400) -> None:
    import matplotlib.pyplot as plt

    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    print(f"Generated: {path}")
    plt.close(fig)


def read_csv(assets_dir, fname: str, xcol: str, ycol: str):
    path = Path(assets_dir) / fname
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    return df[xcol].values, df[ycol].values
