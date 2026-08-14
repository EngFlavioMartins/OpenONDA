#!/usr/bin/env python3
"""Regenerate the plot samples from existing VPM snapshots, without re-running.

``vortex_setup.py`` writes the HDF5 particle snapshots under ``solution/`` but
the sampled figures (line profiles, z = 0 surface fields, pair trajectories,
flow integrals) only reach ``samples/<case>/`` while the simulation runs.  This
script reconstructs ``samples/<case>/`` from the snapshots alone, so the plot
scripts always have something to draw:

  * ``flow_integrals.csv``       — copied from the solver's own log
  * ``run_metadata.json``        — physical constants for the reference curves
  * ``vortex_<scheme>_x/y.csv``  — mid-plane line profiles (final snapshot)
  * ``vortex_<scheme>_z0_<step>.vts`` + ``vortex_<scheme>_z0.pvd``
                                — z = 0 surface fields (final snapshot)
  * ``pair_diagnostics.csv``     — dipole / merging core trajectories (all snapshots)

Existing artifacts are left untouched, so running this before plotting is a
no-op for a fresh run whose ``samples/`` tree is already populated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import taichi as ti

_SCRIPT_DIR = Path(__file__).resolve().parent
_TUTORIAL_DIR = _SCRIPT_DIR.parent
if str(_TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(_TUTORIAL_DIR))

from assets.pair_diagnostics import CSV_COLUMNS as PAIR_CSV_COLUMNS
from source.solvers.VPM.io.sampling import SAMPLER_CSV_COLUMNS

# ---- Tutorial constants (must mirror vortex_setup.py defaults) -------------
RE_GAMMA = 530.0
SEPARATION = 1.0
CORE_RADIUS = 0.125
COLUMN_LENGTH = 50.0 * CORE_RADIUS
FIELD_SPACING = 0.30 * CORE_RADIUS
PROFILE_SPACING = 0.10 * CORE_RADIUS
COLUMN_SPACING = 0.80 * CORE_RADIUS

# z = 0 sampling plane for the single-vortex cases (mirrors domain_bounds[:4])
LATERAL_HALF_WIDTH = 7.0 * CORE_RADIUS

SNAP_STEP_RE = re.compile(r"vpm_(.+)_(\d{6})\.h5$")


# =============================================================
# Snapshot I/O
# =============================================================


def _header_particle_count(path: Path) -> int:
    import h5py

    with h5py.File(path, "r") as handle:
        return int(handle["solver"].attrs["number_of_particles"])


def _load_snapshot(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        solver_group = handle["solver"]
        particles = handle["particles"]
        n = int(solver_group.attrs["number_of_particles"])
        data = {
            "time": float(solver_group.attrs["flow_time"]),
            "step": int(solver_group.attrs["time_step"]),
            "time_step_size": float(solver_group.attrs["time_step_size"]),
            "position": np.asarray(particles["position"][:n], dtype=np.float64),
            "circulation": np.asarray(particles["circulation"][:n], dtype=np.float64),
            "radius": np.asarray(particles["radius"][:n], dtype=np.float64),
        }
        if "group_id" in particles:
            data["group_id"] = np.asarray(particles["group_id"][:n], dtype=np.int32)
        return data


def _find_snapshots(solution_dir: Path) -> dict[str, list[Path]]:
    """Group snapshots by case name, honouring nested (legacy) and flat layouts."""
    cases: dict[str, list[Path]] = {}
    for case_dir in sorted(solution_dir.glob("*/")):
        if not case_dir.is_dir():
            continue
        snaps = sorted(case_dir.glob("vpm_*.h5"))
        if snaps:
            cases[case_dir.name] = snaps
    for snap in sorted(solution_dir.glob("vpm_*.h5")):
        match = SNAP_STEP_RE.match(snap.name)
        if match:
            cases.setdefault(match.group(1), []).append(snap)
    for case in cases:
        cases[case] = sorted(cases[case])
    return cases


def _physics_of(case: str) -> str:
    return case.split("_", 1)[0]


def _scheme_of(case: str) -> str:
    return case.split("_", 1)[1]


def _case_complete(case: str, snaps: list[Path], samples_dir: Path) -> bool:
    """True when every plot artifact for this case already exists."""
    case_samples = samples_dir / case
    if not (case_samples / "flow_integrals.csv").is_file():
        return False
    if not (case_samples / "run_metadata.json").is_file():
        return False
    if _physics_of(case) == "vortex":
        final_step = snaps[-1].stem.split("_")[-1]
        return (
            (case_samples / f"{case}_x.csv").is_file()
            and (case_samples / f"{case}_y.csv").is_file()
            and (case_samples / f"{case}_z0_{final_step}.vts").is_file()
            and (case_samples / f"{case}_z0.pvd").is_file()
        )
    return (case_samples / "pair_diagnostics.csv").is_file()


# =============================================================
# Taichi field evaluation (mirrors the live SurfaceSampler / LineSampler)
# =============================================================


class _ParticleWrapper:
    """Minimal stand-in for the solver's ``Particles`` container."""

    def __init__(self, position, circulation, radius, velocity_background, count):
        self.position = position
        self.circulation = circulation
        self.radius = radius
        self.velocity_background = velocity_background
        self._count = count

    def __len__(self):
        return self._count


def _build_engine(max_particles: int):
    from source.solvers.VPM.physics.engine import PhysicsEngine

    ti.init(arch=ti.cpu, default_fp=ti.f32)
    engine = PhysicsEngine(
        particles_kernel="GAUSSIAN",
        max_particles=max_particles,
        max_targets=100000,
    )
    # Same velocity evaluation the solver is configured with.
    engine.configure_velocity("TREECODE", theta=0.7, multipole_order=3, sort_particle_targets=True)
    return engine


def _make_wrapper(data: dict):
    n = len(data["position"])
    pos = ti.Vector.field(3, ti.f32, shape=n)
    str_ = ti.Vector.field(3, ti.f32, shape=n)
    rad = ti.field(ti.f32, shape=n)
    bg = ti.Vector.field(3, ti.f32, shape=())
    bg[None] = [0.0, 0.0, 0.0]
    pos.from_numpy(data["position"].astype(np.float32))
    str_.from_numpy(data["circulation"].astype(np.float32))
    rad.from_numpy(data["radius"].astype(np.float32))
    return _ParticleWrapper(pos, str_, rad, bg, n)


def _induced_fields(engine, data: dict, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (velocity, vorticity) at ``targets`` induced by the snapshot."""
    wrapper = _make_wrapper(data)
    velocity = engine.compute_target_velocities(wrapper, targets, include_freestream=True)
    vorticity = engine.compute_target_vorticities(wrapper, targets)
    return np.asarray(velocity, dtype=np.float64), np.asarray(vorticity, dtype=np.float64)


# =============================================================
# Sampling geometry (mirrors SurfaceSampler / LineSampler)
# =============================================================


def _surface_targets() -> tuple[np.ndarray, tuple[int, int]]:
    c1 = np.arange(-LATERAL_HALF_WIDTH, LATERAL_HALF_WIDTH + FIELD_SPACING / 2, FIELD_SPACING)
    c2 = np.arange(-LATERAL_HALF_WIDTH, LATERAL_HALF_WIDTH + FIELD_SPACING / 2, FIELD_SPACING)
    c1_grid, c2_grid = np.meshgrid(c1, c2, indexing="ij")
    targets = np.column_stack([c1_grid.ravel(), c2_grid.ravel(), np.zeros(c1_grid.size)])
    return targets, c1_grid.shape


def _line_points(axis: str) -> np.ndarray:
    n_points = int(np.ceil(2.0 * LATERAL_HALF_WIDTH / PROFILE_SPACING)) + 1
    points = np.zeros((n_points, 3), dtype=np.float64)
    points[:, {"x": 0, "y": 1}[axis]] = np.linspace(
        -LATERAL_HALF_WIDTH, LATERAL_HALF_WIDTH, n_points
    )
    return points


# =============================================================
# Writers
# =============================================================


def _write_profile_csv(path: Path, flow_time: float, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        stream.write(f"# flow_time={flow_time}\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(SAMPLER_CSV_COLUMNS)
        for index in range(len(data["x"])):
            writer.writerow([data[column][index] for column in SAMPLER_CSV_COLUMNS])


def _field_dict(
    targets: np.ndarray, velocity: np.ndarray, vorticity: np.ndarray
) -> dict[str, np.ndarray]:
    n = len(targets)
    data = {
        "x": targets[:, 0],
        "y": targets[:, 1],
        "z": targets[:, 2],
        "Ux": velocity[:, 0],
        "Uy": velocity[:, 1],
        "Uz": velocity[:, 2],
        "omega_x": vorticity[:, 0],
        "omega_y": vorticity[:, 1],
        "omega_z": vorticity[:, 2],
    }
    for key in (
        "Sxx",
        "Sxy",
        "Sxz",
        "Syy",
        "Syz",
        "Szz",
        "dudx",
        "dudy",
        "dudz",
        "dvdx",
        "dvdy",
        "dvdz",
        "dwdx",
        "dwdy",
        "dwdz",
    ):
        data[key] = np.zeros(n, dtype=np.float64)
    return data


def _write_vts(path: Path, grid_shape: tuple[int, int], data: dict) -> None:
    import pyvista as pv

    ni, nj = grid_shape
    x_2d = data["x"].reshape(ni, nj)
    y_2d = data["y"].reshape(ni, nj)
    z_2d = data["z"].reshape(ni, nj)
    grid = pv.StructuredGrid(x_2d[:, :, None], y_2d[:, :, None], z_2d[:, :, None])

    def _fortran(key):
        return data[key].reshape(ni, nj).ravel(order="F")

    velocity = np.column_stack([_fortran("Ux"), _fortran("Uy"), _fortran("Uz")])
    vorticity = np.column_stack([_fortran("omega_x"), _fortran("omega_y"), _fortran("omega_z")])
    grid.point_data["Velocity"] = velocity
    grid.point_data["VelocityMagnitude"] = np.linalg.norm(velocity, axis=1)
    grid.point_data["Vorticity"] = vorticity
    grid.point_data["VorticityMagnitude"] = np.linalg.norm(vorticity, axis=1)
    grid.save(str(path), binary=True)


def _write_pvd(path: Path, name_prefix: str, entries: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time_val, filename in entries:
        lines.append(f'    <DataSet timestep="{time_val}" file="{filename}"/>')
    lines.append("  </Collection>")
    lines.append("</VTKFile>")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def _pair_moment(
    positions: np.ndarray,
    weights: np.ndarray,
    radii: np.ndarray,
    group_ids: np.ndarray,
    group: int,
) -> tuple[np.ndarray, float, float]:
    """Weighted in-plane centroid and field radius of one circulation group."""
    selected = group_ids == group
    group_weights = weights[selected]
    total = float(group_weights.sum())
    if total <= np.finfo(float).tiny:
        return np.full(2, np.nan), np.nan, 0.0
    group_positions = positions[selected]
    center = np.einsum("i,ij->j", group_weights, group_positions) / total
    radial_distance_squared = np.sum((group_positions - center) ** 2, axis=1)
    field_radius_squared = (
        np.dot(group_weights, radial_distance_squared + radii[selected] ** 2) / total
    )
    return center, float(field_radius_squared), total


def _pair_row(case: str, data: dict) -> list:
    """One pair_diagnostics row for a snapshot, mirroring PairDiagnosticsSampler."""
    positions = data["position"]
    strengths = data["circulation"]
    radii = data["radius"]
    group_ids = data["group_id"]
    weights = np.linalg.norm(strengths, axis=1)
    finite = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(weights)
        & np.isfinite(radii)
        & (weights > 0.0)
        & (radii > 0.0)
    )
    positions, weights, radii, group_ids = (
        positions[finite],
        weights[finite],
        radii[finite],
        group_ids[finite],
    )

    column_length = COLUMN_LENGTH
    circulation_0_full = float(weights[group_ids == 0].sum()) / column_length
    circulation_full = float(weights.sum()) / column_length

    slab_half_width = 1.5 * COLUMN_SPACING
    slab_center = 0.0
    central = np.abs(positions[:, 2] - slab_center) <= slab_half_width
    if np.count_nonzero(central) < 4:
        nearest = np.argsort(np.abs(positions[:, 2] - slab_center))[:4]
        central = np.zeros(len(positions), dtype=bool)
        central[nearest] = True

    positions = positions[central, :2]
    weights = weights[central]
    radii = radii[central]
    group_ids = group_ids[central]
    center_0, radius_squared_0, _ = _pair_moment(positions, weights, radii, group_ids, 0)

    if case == "dipole":
        return [
            *center_0,
            np.sqrt(radius_squared_0),
            np.nan,
            np.nan,
            np.nan,
            circulation_0_full,
            False,
            *center_0,
            np.nan,
            np.nan,
        ]

    center_1, _, _ = _pair_moment(positions, weights, radii, group_ids, 1)
    centers = np.array([center_0, center_1])
    separation = float(np.linalg.norm(center_0 - center_1))
    midpoint = centers.mean(axis=0)
    selected = np.isin(group_ids, (0, 1))
    pair_weights = weights[selected]
    pair_positions = positions[selected]
    pair_radii = radii[selected]
    total = float(pair_weights.sum())
    field_radius_squared = (
        np.dot(pair_weights, np.sum((pair_positions - midpoint) ** 2, axis=1) + pair_radii**2)
        / total
    )
    core_area = 0.5 * max(field_radius_squared - (separation / 2.0) ** 2, 0.0)
    angle = float(np.arctan2(center_0[1] - midpoint[1], center_0[0] - midpoint[0]))
    merged = not np.isfinite(separation) or not (
        0.05 * SEPARATION <= separation <= 1.15 * SEPARATION
    )
    return [
        *center_0,
        np.nan,
        separation,
        core_area,
        angle,
        circulation_full,
        merged,
        *centers.ravel(),
    ]


def _write_pair_diagnostics(path: Path, case: str, snaps: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(PAIR_CSV_COLUMNS)
        for snap in snaps:
            data = _load_snapshot(snap)
            writer.writerow([data["time"], data["step"], *_pair_row(case, data)])


# =============================================================
# Per-case extraction
# =============================================================


def _copy_first(candidates: list[Path], target: Path) -> bool:
    for source in candidates:
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            return True
    return False


def _extract_case(
    engine,
    case: str,
    snaps: list[Path],
    solution_dir: Path,
    samples_dir: Path,
) -> None:
    physics = _physics_of(case)
    scheme = _scheme_of(case)
    case_samples = samples_dir / case
    case_samples.mkdir(parents=True, exist_ok=True)

    flow_integrals = case_samples / "flow_integrals.csv"
    if not flow_integrals.is_file():
        if _copy_first(
            [
                solution_dir / case / "samples" / "flow_integrals.csv",
                solution_dir / case / "flow_integrals.csv",
            ],
            flow_integrals,
        ):
            print(f"  [extract] {case}: copied flow_integrals.csv")
        else:
            print(f"  [extract] {case}: no flow_integrals.csv found")

    metadata_path = case_samples / "run_metadata.json"
    if not metadata_path.is_file():
        if not _copy_first(
            [
                solution_dir / case / "run_metadata.json",
                solution_dir / case / "samples" / "run_metadata.json",
            ],
            metadata_path,
        ):
            last = _load_snapshot(snaps[-1])
            viscosity = 1.0 / RE_GAMMA
            spacing = 0.4 * CORE_RADIUS
            metadata = {
                "case": physics,
                "scheme": scheme,
                "circulations": (
                    (1.0,)
                    if physics == "vortex"
                    else (-1.0, 1.0)
                    if physics == "dipole"
                    else (1.0, 1.0)
                ),
                "viscosity": viscosity,
                "core_radius": CORE_RADIUS,
                "separation": SEPARATION,
                "column_half_length": COLUMN_LENGTH / 2.0,
                "total_time": last["time"],
                "time_step": last["time_step_size"],
                "in_plane_spacing": spacing,
                "column_spacing": COLUMN_SPACING,
                "particle_radius": 1.50 * spacing,
                "particle_replicas": 1,
                "diffusion_grid_spacing": spacing,
                "field_spacing": FIELD_SPACING,
                "sample_interval": 0.5,
                "raw_backup_interval": 2.5,
                "source": "extract_samples.py",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"  [extract] {case}: wrote run_metadata.json")

    if physics == "vortex":
        x_csv = case_samples / f"{case}_x.csv"
        y_csv = case_samples / f"{case}_y.csv"
        if not (x_csv.is_file() and y_csv.is_file()):
            final = _load_snapshot(snaps[-1])
            for axis, path in (("x", x_csv), ("y", y_csv)):
                if path.is_file():
                    continue
                velocity, vorticity = _induced_fields(engine, final, _line_points(axis))
                data = _field_dict(_line_points(axis), velocity, vorticity)
                _write_profile_csv(path, final["time"], data)
                print(f"  [extract] {case}: wrote {path.name} at t={final['time']:.3g}s")

        pvd = case_samples / f"{case}_z0.pvd"
        vts = case_samples / f"{case}_z0_{snaps[-1].stem.split('_')[-1]}.vts"
        if not (pvd.is_file() and vts.is_file()):
            final = _load_snapshot(snaps[-1])
            targets, grid_shape = _surface_targets()
            velocity, vorticity = _induced_fields(engine, final, targets)
            data = _field_dict(targets, velocity, vorticity)
            _write_vts(vts, grid_shape, data)
            _write_pvd(pvd, f"{case}_z0", [(final["time"], vts.name)])
            print(f"  [extract] {case}: wrote {vts.name} at t={final['time']:.3g}s")
    else:
        pair_csv = case_samples / "pair_diagnostics.csv"
        if not pair_csv.is_file():
            _write_pair_diagnostics(pair_csv, physics, snaps)
            print(f"  [extract] {case}: wrote pair_diagnostics.csv ({len(snaps)} steps)")


# =============================================================
# Entry point
# =============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=_TUTORIAL_DIR / "solution",
        help="directory holding the VPM snapshots",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=_TUTORIAL_DIR / "samples",
        help="directory where the plot scripts read samples",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="only regenerate one case (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    solution_dir = Path(args.solution_dir)
    samples_dir = Path(args.samples_dir)

    if not solution_dir.is_dir():
        print(f"  [extract] no solution directory: {solution_dir}")
        return 0

    cases = _find_snapshots(solution_dir)
    if args.case is not None:
        cases = {name: snaps for name, snaps in cases.items() if name == args.case}
    if not cases:
        print(f"  [extract] no VPM snapshots under {solution_dir}")
        return 0

    pending = {
        name: snaps for name, snaps in cases.items() if not _case_complete(name, snaps, samples_dir)
    }
    if not pending:
        print("  [extract] samples already up to date")
        return 0

    max_n = max(_header_particle_count(snap) for snaps in pending.values() for snap in snaps)
    engine = _build_engine(max(10000, max_n + 10000))
    print(f"  [extract] regenerating samples for {len(pending)} case(s), max {max_n} particles")

    for case in sorted(pending):
        _extract_case(engine, case, pending[case], solution_dir, samples_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
