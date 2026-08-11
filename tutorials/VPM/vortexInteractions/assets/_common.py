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

# -- Directory layout ---------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent  # …/assets/
SCRIPT_DIR = ASSETS_DIR.parent  # …/vortexInteractions/
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"
THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"

# -- Physical constants  (match rings_setup.py) --------------------------------
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


# -- Theme ---------------------------------------------------------------------

_THEME_MODULE = None


def _theme():
    """Return the shared OpenONDA matplotlib theme module."""
    global _THEME_MODULE
    if _THEME_MODULE is None:
        if not THEME_PATH.exists():
            raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
        spec = importlib.util.spec_from_file_location("openonda_matplotlib_setup", THEME_PATH)
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        _THEME_MODULE = theme
    return _THEME_MODULE


def load_theme() -> tuple[dict[str, str], object | None]:
    """Load the OpenONDA matplotlib theme. Returns (COLORS dict, theme module)."""
    theme = _theme()
    theme.set_style()
    return dict(theme.COLORS), theme


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a named figure size from the shared plot theme."""
    return _theme().figure_size(name)


def reference_style() -> dict:
    """Return the shared reference-line style."""
    return dict(_theme().REFERENCE_STYLE)


def reference_fill_style(kind: str = "normal") -> dict:
    """Return the shared reference-band style."""
    if kind == "strong":
        return dict(_theme().REFERENCE_STRONG_FILL_STYLE)
    return dict(_theme().REFERENCE_FILL_STYLE)


def legend_handle_style(style: dict) -> dict:
    """Return the shared legend-handle style for a case style."""
    return _theme().legend_handle_style(style)


def compact_case_legend_handles(include_families: bool = True) -> list:
    """Build one compact method key and, optionally, one family key."""
    from matplotlib.lines import Line2D

    theme = _theme()
    handles = []
    for variant in theme.VARIANT_ORDER:
        style = case_style(f"leapfrog_{variant}", include_family=False)
        style["linestyle"] = "-"
        handles.append(Line2D([0], [0], **legend_handle_style(style)))
    if include_families:
        for family in ("leapfrog", "collide"):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=theme.PALETTE["black"],
                    linestyle=theme.FAMILY_LINESTYLE[family],
                    linewidth=theme.LINE_WIDTH,
                    label=theme.FAMILY_LABEL[family],
                )
            )
    return handles


def mark_every(name: str = "default") -> int:
    """Return the shared marker cadence for a plot kind."""
    return _theme().MARK_EVERY[name]


def secondary_line_style() -> dict:
    """Return the shared style for secondary lines related to a primary case."""
    theme = _theme()
    return {
        "linestyle": theme.SECONDARY_LINESTYLE,
        "linewidth": theme.SECONDARY_LINE_WIDTH,
    }


# -- Argument parser -----------------------------------------------------------


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--solution-dir", default=str(SOLUTION_DIR), help="Root solution directory.")
    p.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Output directory for figures.")
    p.add_argument("--format", choices=_theme().EXPORT_FORMATS, default="png")
    p.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Figure DPI.")
    return p


_BLOWUP_FACTOR = 50.0  # plotting guard only; solver termination uses field-health metrics

_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_STEP_TIME_RE = re.compile(rf"Time-step:\s*(?P<step>\d+)\s+Flow time:\s*(?P<time>{_FLOAT_RE})\s*s")
_DIAG_TITLE_RE = re.compile(
    rf"FLOW DIAGNOSTICS\s+\(step\s+(?P<step>\d+),\s*t\s*=\s*(?P<time>{_FLOAT_RE})\s*s\)"
)
_BLOWUP_RE = re.compile(
    rf"BLOWUP CHECK\s+step=(?P<step>\d+)\s+time=(?P<time>{_FLOAT_RE})\s+"
    rf"max_gamma=(?P<max_gamma>{_FLOAT_RE})\s+threshold=(?P<threshold>{_FLOAT_RE})"
    rf"(?:\s+n_particles=(?P<n_particles>\d+))?"
)


def _case_parts(name: str) -> tuple[str, str]:
    family, _, variant = name.partition("_")
    return family, variant


def case_style(
    name: str,
    include_family: bool = True,
    colors: dict[str, str] | None = None,
) -> dict:
    """Return a consistent style dictionary from the shared plot theme."""
    return _theme().case_style(name, include_family=include_family)


def discover_cases(solution_dir, family: str | None = None) -> list[Path]:
    """Return sorted case directories that hold run data, optionally by family prefix.

    A directory counts as a case if it carries a solver log or full-state
    backups. CSV metric files are also accepted when present.
    """
    sol = Path(solution_dir)
    if not sol.is_dir():
        return []
    cases = []
    intended = _theme().INTENDED_CASE_ORDER
    for d in sol.iterdir():
        if not d.is_dir() or d.name not in intended:
            continue
        case_family, _ = _case_parts(d.name)
        if family and case_family != family:
            continue
        if (
            any(d.glob("*.log"))
            or (d / "stability_metrics.csv").exists()
            or (d / "samples" / "flow_integrals.csv").exists()
            or any(d.glob("vpm_*.h5"))
        ):
            cases.append(d)
    return sorted(cases, key=lambda path: intended[path.name])


def _latest_log(case_dir) -> Path | None:
    case_dir = Path(case_dir)
    expected = case_dir / f"{case_dir.name}.log"
    if expected.exists():
        return expected
    logs = sorted(case_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _first_float_after_colon(line: str) -> float | None:
    _, _, tail = line.partition(":")
    match = re.search(_FLOAT_RE, tail)
    return float(match.group(0)) if match else None


def _vector_after_colon(line: str) -> list[float]:
    _, _, tail = line.partition(":")
    return [float(value) for value in re.findall(_FLOAT_RE, tail)]


def _trim_to_last_monotone_segment(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns or len(df) <= 1:
        return df
    times = df["time"].to_numpy(float)
    last_restart = 0
    for i in range(1, len(times)):
        if np.isfinite(times[i]) and np.isfinite(times[i - 1]) and times[i] < times[i - 1]:
            last_restart = i
    if last_restart:
        df = df.iloc[last_restart:].reset_index(drop=True)
    return df


def read_log_diagnostics(case_dir) -> pd.DataFrame:
    """Read flow diagnostics and blow-up checks from a case log."""
    log_path = _latest_log(case_dir)
    if log_path is None:
        return pd.DataFrame()

    rows: list[dict] = []
    current_step: int | None = None
    current_time: float | None = None
    active: dict | None = None

    def flush_active() -> None:
        nonlocal active
        if active is not None and any(key not in {"step", "time"} for key in active):
            rows.append(active)
        active = None

    for line in log_path.open(encoding="utf-8", errors="replace"):
        if match := _BLOWUP_RE.search(line):
            rows.append(
                {
                    "step": int(match.group("step")),
                    "time": float(match.group("time")),
                    "max_gamma": float(match.group("max_gamma")),
                    "blowup_threshold": float(match.group("threshold")),
                    "n_particles": (
                        int(match.group("n_particles"))
                        if match.group("n_particles") is not None
                        else np.nan
                    ),
                }
            )
            continue

        if match := _STEP_TIME_RE.search(line):
            flush_active()
            current_step = int(match.group("step"))
            current_time = float(match.group("time"))
            continue

        if "FLOW DIAGNOSTICS" in line:
            flush_active()
            if match := _DIAG_TITLE_RE.search(line):
                current_step = int(match.group("step"))
                current_time = float(match.group("time"))
            active = {"step": current_step, "time": current_time}
            continue

        if active is None:
            continue

        if "Number of Particles" in line:
            value = _first_float_after_colon(line)
            active["n_particles"] = int(value) if value is not None else np.nan
        elif "Total Circulation (Σ|Γ|)" in line:
            active["sum_gamma_magnitude"] = _first_float_after_colon(line)
        elif "Total Circulation (ΣΓ)" in line:
            values = _vector_after_colon(line)
            for axis, value in zip("xyz", values[:3]):
                active[f"strength_{axis}"] = value
        elif "Linear Impulse" in line:
            values = _vector_after_colon(line)
            for axis, value in zip("xyz", values[:3]):
                active[f"impulse_{axis}"] = value
        elif "Angular Impulse" in line:
            values = _vector_after_colon(line)
            for axis, value in zip("xyz", values[:3]):
                active[f"angular_impulse_{axis}"] = value
        elif "Total Enstrophy" in line:
            active["enstrophy"] = _first_float_after_colon(line)
        elif "Total Helicity" in line:
            active["helicity"] = _first_float_after_colon(line)
        elif "Total Energy, E" in line:
            active["kinetic_energy"] = _first_float_after_colon(line)
        elif "Modeled dissipation" in line or "Viscous dissipation" in line:
            active["neg_nu_enstrophy"] = _first_float_after_colon(line)
        elif "Energy decay rate" in line:
            value = _first_float_after_colon(line)
            active["dEdt_solver"] = value
            active["dEdt"] = value

    flush_active()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    if "time" in df.columns:
        df = df.dropna(subset=["time"]).sort_values(["time", "step"], kind="stable")
    return _trim_to_last_monotone_segment(df.reset_index(drop=True))


def read_metric(case_dir, column: str, truncate_blowup: bool = True):
    """Return (t*, column) from a case log.

    ``t*`` is the time normalised by ``T_REF``.  When ``truncate_blowup`` is set
    the series is cut at the first snapshot where ``max_gamma`` exceeds the
    blow-up factor, so diverging runs do not swamp the axis scale.
    """
    df = read_log_diagnostics(case_dir)
    if df.empty or column not in df.columns or df[column].dropna().empty:
        integrals = read_integrals(case_dir)
        csv_column = "strength_magnitude" if column == "sum_gamma_magnitude" else column
        if integrals is not None and csv_column in integrals.columns:
            df = integrals.rename(columns={csv_column: column})
    if df.empty or column not in df.columns or df[column].dropna().empty:
        csv = Path(case_dir) / "stability_metrics.csv"
        if not csv.exists():
            return np.array([]), np.array([])
        df = pd.read_csv(csv)
    if column not in df.columns or "time" not in df.columns:
        return np.array([]), np.array([])
    df = df.dropna(subset=["time", column])
    if df.empty:
        return np.array([]), np.array([])
    t = df["time"].to_numpy(float) / T_REF
    y = df[column].to_numpy(float)
    if truncate_blowup and "max_gamma" in df.columns:
        mg = df["max_gamma"].to_numpy(float)
        if mg.size and mg[0] > 0.0:
            bad = np.flatnonzero(mg > _BLOWUP_FACTOR * mg[0])
            if bad.size:
                t, y = t[: bad[0] + 1], y[: bad[0] + 1]
    return t, y


def read_integrals(case_dir):
    """Return flow-integral diagnostics from CSV, falling back to the case log.

    Keeps only the last monotonically increasing time segment so appended
    restart rows do not double back on the time axis.
    """
    csv = Path(case_dir) / "samples" / "flow_integrals.csv"
    if csv.exists():
        df = pd.read_csv(csv)
    else:
        df = read_log_diagnostics(case_dir)
        if df.empty:
            return None
    if "strength_magnitude" in df.columns and "sum_gamma_magnitude" not in df.columns:
        df["sum_gamma_magnitude"] = df["strength_magnitude"]
    keep = [
        "time",
        "step",
        "kinetic_energy",
        "enstrophy",
        "dEdt",
        "dEdt_solver",
        "neg_nu_enstrophy",
        "helicity",
        "sum_gamma_magnitude",
        "strength_magnitude",
        "strength_x",
        "strength_y",
        "strength_z",
        "impulse_x",
        "impulse_y",
        "impulse_z",
        "angular_impulse_x",
        "angular_impulse_y",
        "angular_impulse_z",
        "n_particles",
        # Discretization health (solver ``export_discretization_health``).
        # Invariant drift cannot tell a conservative-but-unresolved run from a
        # trustworthy one; these can.
        "core_radius_mean",
        "overlap_ratio",
        "overlap_ratio_max",
        "vorticity_divergence_error",
        "strength_misalignment_deg",
        "enstrophy_test",
        "max_gamma",
        "turbulent_viscosity_mean",
        "turbulent_viscosity_max",
        "effective_viscosity_mean",
        "effective_viscosity_max",
        "stabilization_viscosity_mean",
        "stabilization_viscosity_max",
        "stabilization_viscosity_active_fraction",
        "invariant_projection_correction_ratio",
    ]
    keep = [col for col in keep if col in df.columns]
    if "time" not in keep or "kinetic_energy" not in keep:
        return None
    df = df[keep].dropna(subset=["time", "kinetic_energy"])
    if df.empty:
        return None
    df = _trim_to_last_monotone_segment(df.reset_index(drop=True))
    return df


# -- H5 helpers ----------------------------------------------------------------


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


# -- Log-file parser -----------------------------------------------------------


def parse_log(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (flow_times, -nuΩ, dE/dt) arrays from a VPM solver log."""
    path = Path(path)
    if not path.exists():
        print(f"(Warning) Log not found: {path}")
        return np.array([]), np.array([]), np.array([])

    t_pat = re.compile(r"Time-step:\s*\d+\s+Flow time:\s*([\d.E+\-]+)\s*s")
    nuEns_pat = re.compile(r"(?:Modeled|Viscous) dissipation.{1,30}:\s*([-\d.e+]+)")
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


# -- Figure helpers ------------------------------------------------------------


def save_fig(
    fig,
    path,
    dpi: int | None = None,
    figure_format: str = "png",
    tight_rect: tuple[float, float, float, float] | None = None,
) -> None:
    _theme().save_fig(
        fig,
        path,
        figure_format=figure_format,
        dpi=dpi,
        tight_rect=tight_rect,
    )


def read_csv(assets_dir, fname: str, xcol: str, ycol: str):
    path = Path(assets_dir) / fname
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    return df[xcol].values, df[ycol].values
