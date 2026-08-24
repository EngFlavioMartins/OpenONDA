"""Paths, physical constants, and H5/CSV diagnostics loaders for the
vortex_ring plot scripts."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# -- Directory layout ---------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent  # …/assets/
SCRIPT_DIR = ASSETS_DIR.parent  # …/vortex_rings/
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"
SAMPLES_DIR = SCRIPT_DIR / "samples"

# -- Physical constants  (match ring_setup.py) -------------------------------------
RING_RADIUS = 1.0  # ring major radius [m]
RING_CIRCULATION = np.pi  # circulation [m²/s]
CORE_RADIUS = 0.1  # initial core radius [m]
KINEMATIC_VISCOSITY = (
    RING_CIRCULATION / 3000.0
)  # kinematic viscosity [m²/s]  (Re = Γ/kinematic_viscosity = 3000)
REFERENCE_TIME = RING_RADIUS**2 / RING_CIRCULATION  # T₀ = R₀²/Γ  [s]

# Saffman (1970) self-induced speed at t=0 with Archer et al. (2008) correction
_eps0 = CORE_RADIUS / RING_RADIUS
_C0 = 0.558 + 1.12 * _eps0**2 + 5.0 * _eps0**4
REFERENCE_VELOCITY = RING_CIRCULATION / (4.0 * np.pi * RING_RADIUS) * (np.log(8.0 / _eps0) - _C0)

# Reference energy & dissipation rate scales (per unit density)
REFERENCE_KINETIC_ENERGY = (
    RING_CIRCULATION**2 * RING_RADIUS
)  # [m⁵/s²]  kinetic energy scale for a ring
P_REF = REFERENCE_KINETIC_ENERGY / REFERENCE_TIME  # [m⁵/s³]  dissipation rate scale = Γ³/R₀


# -- H5 helpers ----------------------------------------------------------------


def _checkpoint_vortex_strength(handle: h5py.File) -> np.ndarray:
    """Read canonical particle vortex strength."""
    return handle["particles/vortex_strength"][:]


def _checkpoint_time(handle: h5py.File) -> float:
    """Read canonical checkpoint time."""
    return float(handle["solver"].attrs["time"])


def _sample_time_column(data: pd.DataFrame) -> str:
    """Return the canonical sample time column."""
    if "time" not in data.columns:
        raise KeyError("sample data requires canonical 'time' column")
    return "time"


def _sample_step_column(data: pd.DataFrame) -> str:
    """Return the canonical sample step column."""
    if "step" not in data.columns:
        raise KeyError("sample data requires canonical 'step' column")
    return "step"


def load_length_integrated_strength(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (nondimensional_time, strength_norm) from H5 checkpoints.

    Computes Σ|alpha_i| at each snapshot and normalises by the initial value.
    For a vortex ring this is a length-integrated strength measure, not the
    scalar tube circulation: changes in ring radius or strength direction can
    change this quantity even when the tube circulation is nearly unchanged.
    """
    times, vortex_strength_magnitude_sums = [], []
    for path in sorted(h5_files):
        try:
            with h5py.File(path, "r") as f:
                vortex_strength = _checkpoint_vortex_strength(f)
                t = _checkpoint_time(f)
                vortex_strength_magnitude_sum = float(
                    np.sum(np.linalg.norm(vortex_strength, axis=1))
                )
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
        times.append(t)
        vortex_strength_magnitude_sums.append(vortex_strength_magnitude_sum)

    if not vortex_strength_magnitude_sums:
        return np.array([]), np.array([])

    t_arr = np.array(times) / REFERENCE_TIME
    c_arr = np.array(vortex_strength_magnitude_sums)
    initial_vortex_strength_magnitude_sum = c_arr[0]

    blow_up = c_arr > 500.0 * initial_vortex_strength_magnitude_sum
    if blow_up.any():
        idx = int(blow_up.argmax())
        print(f"Stopping at {Path(h5_files[idx]).name}: blow-up detected.")
        t_arr = t_arr[:idx]
        c_arr = c_arr[:idx]

    return t_arr, c_arr / initial_vortex_strength_magnitude_sum


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
    if not raw:
        return np.array([]), np.array([])

    rid = min(raw.keys())
    entries = raw[rid]
    if not entries:
        return np.array([]), np.array([])

    t_arr = np.array([d["time"] for d in entries]) / REFERENCE_TIME
    tube_circulation = np.array([d["tube_circulation"] for d in entries])
    valid = np.isfinite(tube_circulation) & (tube_circulation > 0.0)
    if not valid.any():
        return np.array([]), np.array([])

    t_arr = t_arr[valid]
    tube_circulation = tube_circulation[valid]
    return t_arr, tube_circulation / tube_circulation[0]


def load_vector_circulation_error(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return drift in the conserved vector sum, normalized by initial strength.

    The direct transposed stretching operator conserves ``Σ alpha``.  For a
    closed vortex ring this vector sum is close to zero, so the drift is scaled
    by the initial length-integrated strength ``Σ|alpha|`` rather than by
    ``|Σ alpha_0|``.
    """
    raw = load_ring_data(h5_files)
    if not raw:
        return np.array([]), np.array([])

    rid = min(raw.keys())
    entries = raw[rid]
    if not entries:
        return np.array([]), np.array([])

    t_arr = np.array([d["time"] for d in entries]) / REFERENCE_TIME
    sum_vec = np.array([d["net_vortex_strength"] for d in entries])
    initial_vortex_strength_magnitude_sum = float(entries[0]["vortex_strength_magnitude_sum"])
    if initial_vortex_strength_magnitude_sum <= 0.0:
        return np.array([]), np.array([])
    err = np.linalg.norm(sum_vec - sum_vec[0], axis=1) / initial_vortex_strength_magnitude_sum
    return t_arr, err


def _ring_props_from_h5(path) -> dict | None:
    """Return impulse/strength-based properties for each vortex ring."""
    try:
        with h5py.File(path, "r") as f:
            position = f["particles/position"][:]
            gid = f["particles/group_id"][:]
            vortex_strength = _checkpoint_vortex_strength(f)
            t = _checkpoint_time(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

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

        # A circular ring has covariance eigenvalues (R^2/2, R^2/2, 0).
        # Summing the two dominant eigenvalues therefore recovers R^2, while
        # remaining independent of the ring normal direction.
        centred_position = pc - vortex_centroid
        cov = (centred_position * amag[:, None]).T @ centred_position / total_length_strength
        eig = np.linalg.eigvalsh(cov)
        major_radius = float(np.sqrt(max(eig[-1] + eig[-2], 0.0)))
        tube_circulation = (
            total_length_strength / (2.0 * np.pi * major_radius) if major_radius > 1e-12 else np.nan
        )
        out[rid] = dict(
            time=t,
            vortex_centroid_x=xc,
            major_radius=major_radius,
            tube_circulation=tube_circulation,
            linear_impulse_x=linear_impulse_x,
            linear_impulse_magnitude=linear_impulse_magnitude,
            impulse_radius=impulse_radius,
            vortex_strength_magnitude_sum=total_length_strength,
            net_vortex_strength=net_vortex_strength,
            max_vortex_strength_magnitude=float(amag.max()),
        )
    return out


def load_ring_data(h5_files: list) -> dict:
    """Read all H5 checkpoints; stop at blow-up. Returns {ring_id: [dict, ...]}."""
    data: dict = {}
    for path in h5_files:
        res = _ring_props_from_h5(path)
        if not res:
            continue
        if any(r["max_vortex_strength_magnitude"] > 500.0 for r in res.values()):
            print(f"Stopping at {Path(path).name}: blow-up detected.")
            break
        for rid, vals in res.items():
            data.setdefault(rid, []).append(vals)
    return data


def normalise_ring_data(raw: dict) -> dict:
    """Convert raw ring dicts to {rid: {t_norm, x_norm, R_norm}} arrays, masking outliers."""
    out: dict = {}
    for rid, entries in raw.items():
        t = np.array([d["time"] for d in entries]) / REFERENCE_TIME
        x = np.array([d["vortex_centroid_x"] for d in entries]) / RING_RADIUS
        R = np.array([d["major_radius"] for d in entries]) / RING_RADIUS
        valid = (np.abs(x) < 1000) & (np.abs(R) < 1000)
        out[rid] = {"t_norm": t[valid], "x_norm": x[valid], "R_norm": R[valid]}
    return out


def load_ring_speed(h5_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (nondimensional_time, nondimensional_velocity) for a single vortex ring.

    Computes the self-induced velocity from a local least-squares slope of the
    strength-weighted vortex_centroid.  It is normalised by the analytical REFERENCE_VELOCITY,
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
    x = np.array([d["vortex_centroid_x"] for d in entries])

    U_num = np.empty_like(x)
    for i in range(len(t)):
        lo = max(0, i - 2)
        hi = min(len(t), i + 3)
        U_num[i] = np.polyfit(t[lo:hi], x[lo:hi], 1)[0]
    return t / REFERENCE_TIME, U_num / REFERENCE_VELOCITY


def load_sampled_ring_data(csv_path: Path) -> pd.DataFrame | None:
    """Load the compact ring sampler history."""
    if not csv_path.is_file():
        return None
    data = pd.read_csv(csv_path)
    if data.empty:
        return None
    return data.sort_values(_sample_time_column(data)).drop_duplicates(
        _sample_step_column(data), keep="last"
    )


def load_sampled_ring_speed(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled normalized time and vortex_centroid speed."""
    data = load_sampled_ring_data(csv_path)
    if data is None or len(data) < 2:
        return np.array([]), np.array([])

    time = data[_sample_time_column(data)].to_numpy(float)
    position = data["vortex_centroid_x"].to_numpy(float)
    speed = np.empty_like(position)
    for index in range(len(time)):
        lower = max(0, index - 2)
        upper = min(len(time), index + 3)
        speed[index] = np.polyfit(time[lower:upper], position[lower:upper], 1)[0]
    return time / REFERENCE_TIME, speed / REFERENCE_VELOCITY


def load_sampled_ring_circulation(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled normalized tube circulation."""
    data = load_sampled_ring_data(csv_path)
    if data is None:
        return np.array([]), np.array([])
    circulation = data["tube_circulation"].to_numpy(float)
    valid = np.isfinite(circulation) & (circulation > 0.0)
    if not valid.any():
        return np.array([]), np.array([])
    time = data[_sample_time_column(data)].to_numpy(float)[valid] / REFERENCE_TIME
    circulation = circulation[valid]
    return time, circulation / circulation[0]


def load_sampled_vector_circulation_error(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled drift of the conserved vector circulation."""
    data = load_sampled_ring_data(csv_path)
    if data is None:
        return np.array([]), np.array([])
    vector = data[
        ["net_vortex_strength_x", "net_vortex_strength_y", "net_vortex_strength_z"]
    ].to_numpy(float)
    strength0 = float(data["vortex_strength_magnitude_sum"].iloc[0])
    if strength0 <= 0.0:
        return np.array([]), np.array([])
    time = data[_sample_time_column(data)].to_numpy(float) / REFERENCE_TIME
    error = np.linalg.norm(vector - vector[0], axis=1) / strength0
    return time, error


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


def read_csv(assets_dir, fname: str, xcol: str, ycol: str):
    path = Path(assets_dir) / fname
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    return df[xcol].values, df[ycol].values
