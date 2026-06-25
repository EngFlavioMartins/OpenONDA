"""Shared utilities for vortexRing plot scripts.

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
SCRIPT_DIR = ASSETS_DIR.parent  # …/vortexRings/
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"
THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"
FONT_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "DejaVuSerif.ttf"

CM = 1 / 2.54  # cm → inch

# ── Physical constants  (match ring_setup.py) ─────────────────────────────────────
R0 = 1.0  # ring major radius [m]
GAMMA = np.pi  # circulation [m²/s]
CORE_RADIUS = 0.1  # initial core radius [m]
NU = GAMMA / 3000.0  # kinematic viscosity [m²/s]  (Re = Γ/nu = 3000)
T_REF = R0**2 / GAMMA  # T₀ = R₀²/Γ  [s]

# Saffman (1970) self-induced speed at t=0 with Archer et al. (2008) correction
_eps0 = CORE_RADIUS / R0
_C0 = 0.558 + 1.12 * _eps0**2 + 5.0 * _eps0**4
U_REF = GAMMA / (4.0 * np.pi * R0) * (np.log(8.0 / _eps0) - _C0)

# Reference energy & dissipation rate scales (per unit density)
E_REF = GAMMA**2 * R0  # [m⁵/s²]  kinetic energy scale for a ring
P_REF = E_REF / T_REF  # [m⁵/s³]  dissipation rate scale = Γ³/R₀

# Consistent colour & marker per solution variant (used by every figure).
# DNS → dashed line (--),  LES → solid line (-).
VARIANT_STYLE: dict[str, dict[str, str]] = {
    "DNS_direct": {"color": "#1A6B9A", "marker": "o"},
    "DNS_transposed": {"color": "#5C3D9B", "marker": "s"},
    "DNS_mixed": {"color": "#B85C2A", "marker": "^"},
    "LES_direct": {"color": "#0E8A85", "marker": "D"},
    "LES_transposed": {"color": "#1A8C88", "marker": "v"},
    "LES_mixed": {"color": "#2B7A4E", "marker": "p"},
    "LES_rvpm": {"color": "#7B2969", "marker": "v"},
}

# Display labels (override underscore-substitution for special cases).
VARIANT_LABEL: dict[str, str] = {
    **{k: k.replace("_", " ") for k in VARIANT_STYLE},
    "LES_rvpm": "LES rVPM",
}

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


def load_length_integrated_strength(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_star, strength_norm) from H5 backups.

    Computes Σ|alpha_i| at each snapshot and normalises by the initial value.
    For a vortex ring this is a length-integrated strength measure, not the
    scalar tube circulation: changes in ring radius or strength direction can
    change this quantity even when the tube circulation is nearly unchanged.
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


def load_ring_circulation(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_star, Gamma_tube/Gamma_tube0) for a single vortex ring.

    The ring's physically relevant scalar circulation is inferred from the
    length-integrated particle strength and impulse-derived ring radius:

        Gamma_tube = Σ|alpha_i| / (2*pi*R)

    This is the quantity used by the Saffman ring-speed model.  It avoids
    mistaking a change in ring length or strength-vector alignment for a
    change in tube circulation.
    """
    raw = load_ring_data(h5_files)
    if not raw:
        return np.array([]), np.array([])

    rid = min(raw.keys())
    entries = raw[rid]
    if not entries:
        return np.array([]), np.array([])

    t_arr = np.array([d["time"] for d in entries]) / T_REF
    gamma = np.array([d["gamma"] for d in entries])
    valid = np.isfinite(gamma) & (gamma > 0.0)
    if not valid.any():
        return np.array([]), np.array([])

    t_arr = t_arr[valid]
    gamma = gamma[valid]
    return t_arr, gamma / gamma[0]


def load_total_circulation(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for the ring tube-circulation diagnostic."""
    return load_ring_circulation(h5_files)


def _ring_props_from_h5(path) -> dict | None:
    """Return impulse/strength-based properties for each vortex ring."""
    try:
        with h5py.File(path, "r") as f:
            pos = f["particles/position"][:]
            gid = f["particles/group_id"][:]
            strength = f["particles/circulation"][:]
            t = float(f["solver"].attrs.get("flow_time", 0.0))
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

    out: dict = {}
    for rid in np.unique(gid):
        m_ = gid == rid
        pc = pos[m_]
        alpha = strength[m_]
        amag = np.linalg.norm(alpha, axis=1)
        total_length_strength = float(amag.sum())
        if total_length_strength <= 1e-30:
            continue
        xc = float(np.dot(amag, pc[:, 0]) / total_length_strength)
        impulse_x = float(0.5 * np.sum(pc[:, 1] * alpha[:, 2] - pc[:, 2] * alpha[:, 1]))
        major_R = abs(2.0 * impulse_x / total_length_strength)
        gamma = (
            total_length_strength / (2.0 * np.pi * major_R) if major_R > 1e-12 else np.nan
        )
        out[rid] = dict(
            time=t,
            x_centroid=xc,
            major_R=major_R,
            gamma=gamma,
            impulse_x=impulse_x,
            strength_max=float(amag.max()),
        )
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
    """Convert raw ring dicts to {rid: {t_norm, x_norm, R_norm}} arrays, masking outliers."""
    out: dict = {}
    for rid, entries in raw.items():
        t = np.array([d["time"] for d in entries]) / T_REF
        x = np.array([d["x_centroid"] for d in entries]) / R0
        R = np.array([d["major_R"] for d in entries]) / R0
        valid = (np.abs(x) < 1000) & (np.abs(R) < 1000)
        out[rid] = {"t_norm": t[valid], "x_norm": x[valid], "R_norm": R[valid]}
    return out


def load_ring_speed(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_star, U_norm) for a single vortex ring.

    Computes the self-induced velocity from a local least-squares slope of the
    strength-weighted centroid.  It is normalised by the analytical U_REF,
    rather than by its own first noisy finite difference.
    Reuses load_ring_data so blow-up detection is inherited.
    """
    raw = load_ring_data(h5_files)
    if not raw:
        return np.array(
            [],
        ), np.array([])

    rid = min(raw.keys())
    entries = raw[rid]
    if len(entries) < 2:
        return np.array(
            [],
        ), np.array([])

    t = np.array([d["time"] for d in entries])
    x = np.array([d["x_centroid"] for d in entries])

    U_num = np.empty_like(x)
    for i in range(len(t)):
        lo = max(0, i - 2)
        hi = min(len(t), i + 3)
        U_num[i] = np.polyfit(t[lo:hi], x[lo:hi], 1)[0]
    return t / T_REF, U_num / U_REF


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

    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def saffman_speed(t_arr: np.ndarray, k_nu: float = 4.0) -> np.ndarray:
    """Saffman (1970) self-induced velocity with Archer et al. (2008) correction.

    Gaussian core diffusion:  a²(t) = a₀² + k_nu·nu·t   (k_nu=4 for laminar).
    Finite-core correction:  C(ε) = 0.558 + 1.12·ε² + 5.0·ε⁴
    Ring speed:              U(t) = Γ/(4πR₀) · [ln(8R₀/a) - C(a/R₀)]
    """
    t_s = CORE_RADIUS**2 / (k_nu * NU)
    a_t = np.sqrt(k_nu * NU * (np.asarray(t_arr) + t_s))
    eps = a_t / R0
    C = 0.558 + 1.12 * eps**2 + 5.0 * eps**4
    return GAMMA / (4.0 * np.pi * R0) * (np.log(8.0 / eps) - C)


def read_csv(assets_dir, fname: str, xcol: str, ycol: str):
    path = Path(assets_dir) / fname
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    return df[xcol].values, df[ycol].values
