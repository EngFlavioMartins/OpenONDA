"""Parameter-study source information and run status utilities for verification tools."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid


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
    # Keep generated and unrelated untracked checkpoint_files out of source information. The
    # checkpoint_info separately hashes the exact setup and geometry inputs.
    return hashlib.sha256(result.stdout).hexdigest()


def file_hash(path: Path) -> str:
    """Return a stable SHA-256 digest for a parameter study input file."""
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
def _cached_software_versions_and_hash(
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
    record = {
        "schema": "openonda-software-versions-and-hash/2",
        "python": platform.python_version(),
        "versions": versions,
        "source_digest": _source_tree_digest(repository),
    }
    ordered_json = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return {**record, "digest": hashlib.sha256(ordered_json.encode("utf-8")).hexdigest()}


def software_versions_and_hash(repository: Path | None = None) -> dict[str, object]:
    """Return a cached software description for OpenONDA source and numerical dependencies."""
    root = Path(__file__).resolve().parents[3] if repository is None else Path(repository).resolve()
    return _cached_software_versions_and_hash(str(root), _source_tree_signature(root))


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
    """Allocate a unique, non-destructive parameter study directory."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{label}-{stamp}-{uuid.uuid4().hex[:8]}"


def write_metadata(
    path: Path, *, status: str, config: dict[str, object], inputs: tuple[Path, ...]
) -> None:
    """Atomically write parameter study source information and run_stages status."""
    repository = Path(__file__).resolve().parents[3]
    record = {
        "schema": "openonda-cylinder-parameter-study/2",
        "status": status,
        "updated_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "hardware": hardware_snapshot(),
        "git_diff_hash": diff_hash(repository),
        "software_versions_and_hash": software_versions_and_hash(repository),
        "inputs": {str(item): file_hash(item) for item in inputs if item.is_file()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete_marker(path: Path) -> Path:
    """Return the marker used to distinguish complete runs from resumable ones."""
    return path / "COMPLETE"


def is_complete(path: Path) -> bool:
    """Require both a marker and a complete parameter study checkpoint_info."""
    checkpoint_info = path / "study_metadata.json"
    if not complete_marker(path).is_file() or not checkpoint_info.is_file():
        return False
    record = json.loads(checkpoint_info.read_text(encoding="utf-8"))
    return record.get("status") == "complete"


def write_complete_marker(path: Path) -> None:
    """Record completion only after all outputs have been flushed."""
    complete_marker(path).write_text("complete\n", encoding="utf-8")
