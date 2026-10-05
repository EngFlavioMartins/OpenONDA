"""Paths, physical constants, and H5/CSV diagnostics loaders for the
vortex_ring plot scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from openonda import plotting as theme
from openonda.results import read_history_table, read_json
from source.solvers.vpm.io.postprocess import particle_state

ASSETS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ASSETS_DIR.parent
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"
SAMPLES_DIR = SCRIPT_DIR / "samples"
RING_RADIUS = 1.0
RING_CIRCULATION = np.pi
CORE_RADIUS = 0.1
KINEMATIC_VISCOSITY = RING_CIRCULATION / 3000.0
REFERENCE_TIME = RING_RADIUS**2 / RING_CIRCULATION
_eps0 = CORE_RADIUS / RING_RADIUS
_C0 = 0.558 + 1.12 * _eps0**2 + 5.0 * _eps0**4
REFERENCE_VELOCITY = RING_CIRCULATION / (4.0 * np.pi * RING_RADIUS) * (np.log(8.0 / _eps0) - _C0)
SAFFMAN_MAX_CORE_RATIO = 0.3
REFERENCE_KINETIC_ENERGY = RING_CIRCULATION**2 * RING_RADIUS
P_REF = REFERENCE_KINETIC_ENERGY / REFERENCE_TIME


def _theme():
    return theme


VARIANT_STYLE = _theme().VORTEX_RING_VARIANT_STYLE
VARIANT_LABEL = _theme().VORTEX_RING_VARIANT_LABEL
CURRENT_VARIANTS = ("dns_direct", "dns_transposed", "dns_mixed", "les_transposed")
EXPECTED_STRETCHING_SCHEME = {
    "dns_direct": "DIRECT",
    "dns_transposed": "TRANSPOSED",
    "dns_mixed": "MIXED",
    "les_transposed": "TRANSPOSED",
}


def metadata_path(variant: str, samples_dir: Path = SAMPLES_DIR) -> Path:
    """Return the solver-owned metadata path corresponding to a sample root."""
    solution_dir = SOLUTION_DIR if samples_dir == SAMPLES_DIR else samples_dir.parent / "solution"
    return solution_dir / variant / "vpm_metadata.json"


def load_metadata(variant: str, samples_dir: Path = SAMPLES_DIR) -> dict:
    return read_json(metadata_path(variant, samples_dir))


def _stretching_scheme(metadata: dict) -> str:
    return metadata["configuration"]["numerics"]["induction"]["stretching_scheme"]


def plot_variants(samples_dir: Path = SAMPLES_DIR) -> tuple[str, ...]:
    return CURRENT_VARIANTS


def load_stability_results(samples_dir: Path = SAMPLES_DIR) -> tuple[dict, ...]:
    results = []
    for variant in CURRENT_VARIANTS:
        metadata = load_metadata(variant, samples_dir)
        state = metadata["state"]
        results.append(
            {
                "variant": variant,
                "status": metadata["run_status"]["status"],
                "step": state["step"],
                "time": state["time"],
                "normalized_time": state["time"] / REFERENCE_TIME,
            }
        )
    return tuple(results)


def load_theme() -> tuple[dict[str, str], object | None]:
    """Load the OpenONDA matplotlib theme. Returns (COLORS dict, theme module)."""
    theme = _theme()
    theme.set_thesis_style()
    return (dict(theme.COLORS), theme)


def figure_size(name: str = "single") -> tuple[float, float]:
    return _theme().figure_size(name)


def centered_subplots_adjust(fig, *, outer: float, **kwargs) -> None:
    """Apply the shared horizontally-centred thesis layout."""
    _theme().centered_subplots_adjust(fig, outer=outer, **kwargs)


def mark_every(name: str = "default") -> int:
    return _theme().MARK_EVERY[name]


def reference_style() -> dict:
    return dict(_theme().REFERENCE_STYLE)


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--format", choices=_theme().FORMAT_CHOICES, default="both")
    p.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Figure DPI.")
    return p


def save_fig(fig, path, dpi=None, tight_rect=None, figure_format="both"):
    return _theme().export_figure(fig, path, figure_format=figure_format, dpi=dpi)


def load_length_integrated_strength(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (nondimensional_time, strength_norm) from H5 backups.

    Computes Σ|alpha_i| at each snapshot and normalises by the initial value.
    For a vortex ring this is a length-integrated strength measure, not the
    scalar tube circulation: changes in ring radius or strength direction can
    change this quantity even when the tube circulation is nearly unchanged.
    """
    times, vortex_strength_magnitude_sums = ([], [])
    for path in sorted(h5_files):
        state = particle_state(path)
        t = state["time"]
        vortex_strength_magnitude_sum = float(
            np.sum(np.linalg.norm(state["vortex_strength"], axis=1))
        )
        times.append(t)
        vortex_strength_magnitude_sums.append(vortex_strength_magnitude_sum)
    t_arr = np.array(times) / REFERENCE_TIME
    c_arr = np.array(vortex_strength_magnitude_sums)
    initial_vortex_strength_magnitude_sum = c_arr[0]
    return (t_arr, c_arr / initial_vortex_strength_magnitude_sum)


def load_ring_circulation(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (nondimensional_time, circulation_tube/circulation_tube0) for a single vortex ring.

    The ring's physically relevant scalar circulation is inferred from the
    length-integrated particle strength and orientation-independent ring
    radius:

        circulation_tube = Σ|alpha_i| / (2*pi*R_cov)

    ``R_cov`` is computed from the two dominant eigenvalues of the
    strength-weighted position covariance.  Unlike an impulse-x radius, it does
    not report a false circulation spike when the ring tilts away from the
    initial x-axis.
    """
    raw = load_ring_data(h5_files)
    rid = min(raw.keys())
    entries = raw[rid]
    t_arr = np.array([d["time"] for d in entries]) / REFERENCE_TIME
    tube_circulation = np.array([d["tube_circulation"] for d in entries])
    valid = np.isfinite(tube_circulation) & (tube_circulation > 0.0)
    t_arr = t_arr[valid]
    tube_circulation = tube_circulation[valid]
    return (t_arr, tube_circulation / tube_circulation[0])


def load_vector_circulation_error(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return drift in the conserved vector sum, normalized by initial strength.

    Transposed stretching conserves ``Σ alpha`` under exact pair summation;
    accelerated evaluation may introduce approximation error. For a closed
    vortex ring this vector sum is close to zero, so the drift is scaled
    by the initial length-integrated strength ``Σ|alpha|`` rather than by
    ``|Σ alpha_0|``.
    """
    raw = load_ring_data(h5_files)
    rid = min(raw.keys())
    entries = raw[rid]
    t_arr = np.array([d["time"] for d in entries]) / REFERENCE_TIME
    sum_vec = np.array([d["net_vortex_strength"] for d in entries])
    initial_vortex_strength_magnitude_sum = float(entries[0]["vortex_strength_magnitude_sum"])
    err = np.linalg.norm(sum_vec - sum_vec[0], axis=1) / initial_vortex_strength_magnitude_sum
    return (t_arr, err)


def _ring_props_from_h5(path) -> dict:
    """Return impulse/strength-based properties for each vortex ring."""
    state = particle_state(path)
    position, gid, vortex_strength = (
        state[name] for name in ("position", "group_id", "vortex_strength")
    )
    t = state["time"]
    out: dict = {}
    for rid in np.unique(gid):
        m_ = gid == rid
        pc = position[m_]
        alpha = vortex_strength[m_]
        amag = np.linalg.norm(alpha, axis=1)
        total_length_strength = float(amag.sum())
        if total_length_strength <= 1e-30:
            continue
        net_vortex_strength = alpha.sum(axis=0)
        vortex_centroid = np.einsum("i,ij->j", amag, pc) / total_length_strength
        xc = float(vortex_centroid[0])
        impulse = 0.5 * np.sum(np.cross(pc, alpha), axis=0)
        linear_impulse_x = float(impulse[0])
        linear_impulse_magnitude = float(np.linalg.norm(impulse))
        impulse_radius = 2.0 * linear_impulse_magnitude / total_length_strength
        centred_position = pc - vortex_centroid
        cov = (centred_position * amag[:, None]).T @ centred_position / total_length_strength
        eig = np.linalg.eigvalsh(cov)
        major_radius = float(np.sqrt(max(eig[-1] + eig[-2], 0.0)))
        tube_circulation = (
            total_length_strength / (2.0 * np.pi * major_radius) if major_radius > 1e-12 else np.nan
        )
        out[rid] = {
            "time": t,
            "vortex_centroid_x": xc,
            "major_radius": major_radius,
            "tube_circulation": tube_circulation,
            "linear_impulse_x": linear_impulse_x,
            "linear_impulse_magnitude": linear_impulse_magnitude,
            "impulse_radius": impulse_radius,
            "vortex_strength_magnitude_sum": total_length_strength,
            "net_vortex_strength": net_vortex_strength,
            "max_vortex_strength_magnitude": float(amag.max()),
        }
    return out


def load_ring_data(h5_files: list) -> dict:
    """Derive physical ring quantities from every recorded native backup."""
    data: dict = {}
    for path in h5_files:
        res = _ring_props_from_h5(path)
        for rid, vals in res.items():
            data.setdefault(rid, []).append(vals)
    return data


def normalise_ring_data(raw: dict) -> dict:
    """Normalize the full recorded ring trajectory by its physical scales."""
    out: dict = {}
    for rid, entries in raw.items():
        t = np.array([d["time"] for d in entries]) / REFERENCE_TIME
        x = np.array([d["vortex_centroid_x"] for d in entries]) / RING_RADIUS
        R = np.array([d["major_radius"] for d in entries]) / RING_RADIUS
        out[rid] = {"t_norm": t, "x_norm": x, "R_norm": R}
    return out


def load_ring_speed(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (nondimensional_time, nondimensional_velocity) for a single vortex ring.

    Computes the self-induced velocity from a local least-squares slope of the
    strength-weighted vortex_centroid.  It is normalised by the analytical REFERENCE_VELOCITY,
    rather than by its own first noisy finite difference.
    """
    raw = load_ring_data(h5_files)
    rid = min(raw.keys())
    entries = raw[rid]
    t = np.array([d["time"] for d in entries])
    x = np.array([d["vortex_centroid_x"] for d in entries])
    U_num = np.empty_like(x)
    for i in range(len(t)):
        lo = max(0, i - 2)
        hi = min(len(t), i + 3)
        U_num[i] = np.polyfit(t[lo:hi], x[lo:hi], 1)[0]
    return (t / REFERENCE_TIME, U_num / REFERENCE_VELOCITY)


def load_sampled_ring_data(csv_path: Path) -> pd.DataFrame:
    """Load the compact ring sampler history."""
    return pd.DataFrame(read_history_table(csv_path))


def load_sampled_ring_speed(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled normalized time and vortex_centroid speed."""
    data = load_sampled_ring_data(csv_path)
    time = data["time"].to_numpy(float)
    position = data["vortex_centroid_x"].to_numpy(float)
    speed = np.empty_like(position)
    boundaries = np.r_[
        0, np.flatnonzero(np.diff(time) > 1.5 * np.median(np.diff(time))) + 1, len(time)
    ]
    for first, last in zip(boundaries[:-1], boundaries[1:], strict=False):
        for index in range(first, last):
            lower = max(first, index - 2)
            upper = min(last, index + 3)
            speed[index] = (
                np.polyfit(time[lower:upper], position[lower:upper], 1)[0]
                if upper - lower >= 2
                else np.nan
            )
    keep = time > 0.0
    return (time[keep] / REFERENCE_TIME, speed[keep] / REFERENCE_VELOCITY)


def saffman_valid_time_limit(maximum_core_ratio: float = SAFFMAN_MAX_CORE_RATIO) -> float:
    """Return the last physical time retained for the thin-core comparison."""
    return ((maximum_core_ratio * RING_RADIUS) ** 2 - CORE_RADIUS**2) / (4.0 * KINEMATIC_VISCOSITY)


def load_sampled_ring_circulation(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return tube circulation relative to the saved unevolved state.

    Retain initialization and the full adjustment history. Subscript zero
    consequently has the same meaning for scalar and vector diagnostics.
    """
    data = load_sampled_ring_data(csv_path)
    time = data["time"].to_numpy(float)
    circulation = data["tube_circulation"].to_numpy(float)
    return (time / REFERENCE_TIME, circulation / circulation[0])


def load_sampled_vector_circulation_error(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled drift of the conserved vector circulation."""
    data = load_sampled_ring_data(csv_path)
    vector = data[
        ["net_vortex_strength_x", "net_vortex_strength_y", "net_vortex_strength_z"]
    ].to_numpy(float)
    strength0 = float(data["vortex_strength_magnitude_sum"].iloc[0])
    time = data["time"].to_numpy(float)
    error = np.linalg.norm(vector - vector[0], axis=1) / strength0
    keep = np.isfinite(time) & (time > 0.0) & np.isfinite(error) & (error > 0.0)
    return (time[keep] / REFERENCE_TIME, error[keep])


def saffman_speed(t_arr: np.ndarray, k_nu: float = 4.0) -> np.ndarray:
    """Saffman (1970) self-induced velocity with Archer et al. (2008) correction.

    Gaussian core diffusion:  a²(t) = a₀² + k_nu·kinematic_viscosity·t   (k_nu=4 for laminar).
    Finite-core correction:  C(ε) = 0.558 + 1.12·ε² + 5.0·ε⁴
    Ring speed:              U(t) = Γ/(4πR₀) · [ln(8R₀/a) - C(a/R₀)]
    """
    t_s = CORE_RADIUS**2 / (k_nu * KINEMATIC_VISCOSITY)
    a_t = np.sqrt(k_nu * KINEMATIC_VISCOSITY * (np.asarray(t_arr) + t_s))
    eps = a_t / RING_RADIUS
    C = 0.558 + 1.12 * eps**2 + 5.0 * eps**4
    return RING_CIRCULATION / (4.0 * np.pi * RING_RADIUS) * (np.log(8.0 / eps) - C)


def with_sample_gaps(time, *values):
    """Break lines across missing sampler events without inventing measurements."""
    time = np.asarray(time, dtype=float)
    if len(time) < 3:
        return (time, *values)
    gaps = np.flatnonzero(np.diff(time) > 1.5 * np.median(np.diff(time))) + 1
    return (
        np.insert(time, gaps, np.nan),
        *(np.insert(np.asarray(value, dtype=float), gaps, np.nan) for value in values),
    )
