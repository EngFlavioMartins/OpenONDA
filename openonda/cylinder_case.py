"""Validated parameters and reproducibility helpers for cylinder campaigns.

The tutorial setup files import this small module so the coupled and reference
cases cannot silently drift in Reynolds number, span, or mesh-family naming.
It deliberately contains no solver imports, which keeps campaign inspection
and manifest generation cheap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid


@dataclass(frozen=True, slots=True)
class CylinderCaseParameters:
    """Shared physical and campaign parameters for the Re=150 cylinder."""

    reynolds_number: float = 150.0
    diameter: float = 1.0
    freestream_velocity: float = 1.0
    resolved_span: float = 0.96
    coupled_wall_spacing: float = 0.04
    coupled_end_time: float = 100.0
    reference_end_time: float = 100.0
    reference_force_window: tuple[float, float] = (40.0, 100.0)

    def validate(self) -> CylinderCaseParameters:
        """Validate invariants shared by the coupled and reference setups."""
        if self.reynolds_number != 150.0:
            raise ValueError("the cylinder campaign is defined at Re=150")
        if self.diameter <= 0.0 or self.freestream_velocity <= 0.0:
            raise ValueError("diameter and freestream velocity must be positive")
        if self.resolved_span <= 0.0:
            raise ValueError("resolved span must be positive")
        if self.coupled_wall_spacing <= 0.0:
            raise ValueError("coupled wall spacing must be positive")
        start, end = self.reference_force_window
        if not 0.0 <= start < end <= self.reference_end_time:
            raise ValueError("reference force window must lie inside the reference horizon")
        return self

    @property
    def kinematic_viscosity(self) -> float:
        return self.freestream_velocity * self.diameter / self.reynolds_number

    def reference_grid(self) -> tuple[tuple[str, float], ...]:
        """Return the planned geometric reference family, coarse to fine."""
        return (
            ("grid_h008", 0.08),
            ("grid_h00565685", 0.0565685424949238),
            ("grid_h004", 0.04),
            ("grid_h00282843", 0.0282842712474619),
        )


DEFAULT_CYLINDER_CASE = CylinderCaseParameters().validate()

COUPLED_OVERRIDE_NAMES = frozenset(
    {
        "hxy",
        "span",
        "dz",
        "particle_spacing_ratio",
        "core_radius_ratio",
        "blend_width_ratio",
        "release_width_ratio",
        "exchange_dt",
        "cores",
        "compute_device",
        "particle_limit",
        "boundary_condition_mode",
        "transfer_method",
        "transfer_region_scale",
        "transfer_amplification_cap",
        "fvm_consistency_width",
        "interface_iterations",
        "interface_acceleration",
    }
)


def normalize_coupled_overrides(overrides: dict[str, object] | None) -> dict[str, object]:
    """Validate the small set of sensitivity knobs exposed by the campaign."""
    values = {} if overrides is None else dict(overrides)
    unknown = set(values) - COUPLED_OVERRIDE_NAMES
    if unknown:
        raise ValueError(f"unsupported coupled override(s): {sorted(unknown)}")
    normalized = dict(values)
    if "interface_acceleration" in normalized and normalized["interface_acceleration"] not in {
        "none",
        "aitken",
    }:
        raise ValueError("interface_acceleration must be 'none' or 'aitken'")
    if "transfer_region_scale" in normalized:
        normalized["transfer_region_scale"] = float(normalized["transfer_region_scale"])
        if normalized["transfer_region_scale"] <= 0.0:
            raise ValueError("transfer_region_scale must be positive")
    for name in ("transfer_amplification_cap", "fvm_consistency_width"):
        if name in normalized:
            normalized[name] = float(normalized[name])
            if normalized[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
    if "interface_iterations" in normalized:
        normalized["interface_iterations"] = int(normalized["interface_iterations"])
        if normalized["interface_iterations"] < 1:
            raise ValueError("interface_iterations must be positive")
    return normalized


def validate_coupled_geometry(
    *,
    hxy: float,
    span: float,
    dz: float,
    particle_spacing_ratio: float,
    core_radius_ratio: float,
    blend_width_ratio: float,
    release_width_ratio: float,
    exchange_dt: float,
    cores: int,
    particle_limit: int,
    end_time: float,
    fvm_time_step: float,
) -> None:
    """Validate geometry-factory inputs outside the tutorial learning surface."""
    if min(hxy, span, dz, particle_spacing_ratio, core_radius_ratio, exchange_dt) <= 0.0:
        raise ValueError("mesh, span, particle, core, and exchange spacings must be positive")
    if span <= hxy:
        raise ValueError("span must leave a nonempty FVM transfer region")
    if not 0.0 < release_width_ratio < blend_width_ratio:
        raise ValueError("release_width_ratio must lie strictly within blend_width_ratio")
    if cores < 1 or particle_limit < 1:
        raise ValueError("cores and particle_limit must be positive")
    fvm_substeps = round(exchange_dt / fvm_time_step)
    if (
        not math.isclose(fvm_substeps * fvm_time_step, exchange_dt, rel_tol=0.0, abs_tol=1.0e-12)
        or fvm_substeps < 1
    ):
        raise ValueError("exchange_dt must be a positive integer multiple of the FVM step")
    if end_time <= 0.0:
        raise ValueError("end_time must be positive")
    exchanges = round(end_time / exchange_dt)
    if not math.isclose(exchanges * exchange_dt, end_time, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("end_time must land on an accepted exchange boundary")


def validate_cylinder_authority(
    *, transfer_edge_x: float, radius: float, blend_width: float
) -> float:
    """Keep the whole immersed cylinder inside full FVM authority."""
    authority_edge = transfer_edge_x - blend_width
    if authority_edge <= radius + 1.0e-6:
        raise ValueError("authority ramp begins inside the cylinder wall; reduce blend_width_ratio")
    return authority_edge


def diff_hash(repository: Path) -> str:
    """Hash tracked and uncommitted changes without modifying the checkout."""
    if not (repository / ".git").exists():
        return "unknown"
    valid_repository = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if valid_repository.returncode != 0 or valid_repository.stdout.strip() != "true":
        return "unknown"
    result = subprocess.run(
        ["git", "diff", "HEAD", "--binary", "--no-ext-diff"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    # Keep generated and unrelated untracked artifacts out of provenance. The
    # manifest separately hashes the exact setup and geometry inputs.
    return hashlib.sha256(result.stdout).hexdigest()


def file_hash(path: Path) -> str:
    """Return a stable SHA-256 digest for a campaign input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_NUMERICAL_DEPENDENCIES = ("numpy", "scipy", "numba", "taichi", "mpi4py", "h5py", "gmsh", "vtk")


def _source_tree_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative, path in _source_tree_files(repository):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_tree_files(repository: Path) -> list[tuple[str, Path]]:
    files = []
    for prefix in ("openonda", "source"):
        root = repository / prefix
        if root.is_dir():
            files.extend(
                (path.relative_to(repository).as_posix(), path) for path in root.rglob("*.py")
            )
    return sorted(files)


def _source_tree_signature(repository: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (relative, path.stat().st_size, path.stat().st_mtime_ns)
        for relative, path in _source_tree_files(repository)
    )


@lru_cache(maxsize=8)
def _cached_software_fingerprint(
    repository_text: str, signature: tuple[tuple[str, int, int], ...]
) -> dict[str, object]:
    del signature
    repository = Path(repository_text)
    versions = {}
    for package in _NUMERICAL_DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    try:
        versions["openonda"] = metadata.version("openonda")
    except metadata.PackageNotFoundError:
        versions["openonda"] = "unknown"
    payload = {
        "schema": "openonda-software-fingerprint/1",
        "python": platform.python_version(),
        "versions": versions,
        "source_digest": _source_tree_digest(repository),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def software_fingerprint(repository: Path | None = None) -> dict[str, object]:
    """Return a cached identity for OpenONDA source and numerical dependencies."""
    root = Path(__file__).resolve().parents[1] if repository is None else Path(repository).resolve()
    return _cached_software_fingerprint(str(root), _source_tree_signature(root))


def hardware_snapshot() -> dict[str, object]:
    """Capture inexpensive host facts needed to interpret timings."""
    return {
        "hostname": platform.node(),
        "system": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": _physical_cpu_count(),
        "memory_available_bytes": _available_memory_bytes(),
        "vulkan_summary": _vulkan_summary(),
    }


def _physical_cpu_count() -> int | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        pairs = set()
        physical = None
        core = None
        for line in cpuinfo.read_text(encoding="utf-8").splitlines() + [""]:
            if line.startswith("physical id:"):
                physical = line.split(":", 1)[1].strip()
            elif line.startswith("core id:"):
                core = line.split(":", 1)[1].strip()
            elif not line.strip():
                if physical is not None and core is not None:
                    pairs.add((physical, core))
                physical = core = None
        if pairs:
            return len(pairs)
    topology = Path("/sys/devices/system/cpu")
    pairs = set()
    for cpu in topology.glob("cpu[0-9]*"):
        core = cpu / "topology/core_id"
        package = cpu / "topology/physical_package_id"
        if core.is_file() and package.is_file():
            pairs.add((package.read_text().strip(), core.read_text().strip()))
    if pairs:
        return len(pairs)
    result = subprocess.run(
        ["sysctl", "-n", "hw.physicalcpu"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return int(value) if value.isdigit() and int(value) > 0 else None


def _available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            return int(fields[1]) * 1024 if len(fields) >= 2 else None
    return None


def _vulkan_summary() -> str | None:
    result = subprocess.run(
        ["bash", "-lc", "env -u DISPLAY vulkaninfo --summary 2>/dev/null"],
        capture_output=True,
        text=True,
        check=False,
    )
    summary = result.stdout.strip()
    return summary[:4096] if summary else None


def new_run_directory(root: Path, label: str) -> Path:
    """Allocate a unique, non-destructive campaign directory."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{label}-{stamp}-{uuid.uuid4().hex[:8]}"


def write_manifest(
    path: Path, *, status: str, config: dict[str, object], inputs: tuple[Path, ...]
) -> None:
    """Atomically publish campaign provenance and lifecycle status."""
    repository = Path(__file__).resolve().parents[1]
    payload = {
        "schema": "openonda-cylinder-campaign/1",
        "status": status,
        "updated_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "hardware": hardware_snapshot(),
        "git_diff_hash": diff_hash(repository),
        "software_fingerprint": software_fingerprint(repository),
        "inputs": {str(item): file_hash(item) for item in inputs if item.is_file()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete_marker(path: Path) -> Path:
    """Return the marker used to distinguish complete runs from resumable ones."""
    return path / "COMPLETE"


def is_complete(path: Path) -> bool:
    """Require both a marker and a complete campaign manifest."""
    manifest = path / "campaign_manifest.json"
    if not complete_marker(path).is_file() or not manifest.is_file():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return payload.get("status") == "complete"


def write_complete_marker(path: Path) -> None:
    """Publish completion only after all outputs have been flushed."""
    complete_marker(path).write_text("complete\n", encoding="utf-8")


def as_config(parameters: CylinderCaseParameters = DEFAULT_CYLINDER_CASE) -> dict[str, object]:
    """Serialize shared parameters for manifests and review."""
    return asdict(parameters)


__all__ = [
    "CylinderCaseParameters",
    "DEFAULT_CYLINDER_CASE",
    "as_config",
    "complete_marker",
    "diff_hash",
    "file_hash",
    "hardware_snapshot",
    "is_complete",
    "new_run_directory",
    "normalize_coupled_overrides",
    "validate_coupled_geometry",
    "write_complete_marker",
    "write_manifest",
    "software_fingerprint",
]
