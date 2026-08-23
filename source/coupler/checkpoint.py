"""Atomic restart checkpoints for coupled FVM--VPM state."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from source.schemas import SCHEMA_VERSION

CHECKPOINT_DIRECTORY = "checkpoints"
CHECKPOINT_FORMAT_VERSION = 8


def config_digest(config) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def artifact_digest(path: Path) -> str:
    """Hash one checkpoint artifact, including relative names for directories."""
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
    else:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(target: Path, artifact: str) -> Path:
    """Resolve a manifest artifact without allowing checkpoint path escape."""
    if not isinstance(artifact, str) or not artifact:
        raise ValueError("Coupled checkpoint artifact names must be non-empty strings")
    relative = Path(artifact)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe coupled checkpoint artifact path: {artifact!r}")
    target_resolved = target.resolve()
    resolved = (target / relative).resolve()
    if resolved != target_resolved and target_resolved not in resolved.parents:
        raise ValueError(f"Coupled checkpoint artifact escapes its directory: {artifact!r}")
    return resolved


def config_diff(stored: dict | None, current: dict) -> list[str]:
    """Return ``section.key: old -> new`` lines for a two-level config dict."""
    if not stored:
        return []
    lines: list[str] = []
    for section in sorted(set(stored) | set(current)):
        old_section, new_section = stored.get(section), current.get(section)
        if isinstance(old_section, dict) and isinstance(new_section, dict):
            for key in sorted(set(old_section) | set(new_section)):
                if old_section.get(key) != new_section.get(key):
                    lines.append(
                        f"{section}.{key}: {old_section.get(key)!r} -> {new_section.get(key)!r}"
                    )
        elif old_section != new_section:
            lines.append(f"{section}: {old_section!r} -> {new_section!r}")
    return lines


def save_coupled_state(coupler, directory, *, coupling_step: int | None = None) -> Path:
    """Write both solvers and the boundary-history state, committing the manifest last."""
    if coupler.fvm_solver is None:
        raise RuntimeError("Initialize the coupler before saving a checkpoint")

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    step = (
        int(coupling_step)
        if coupling_step is not None
        else int(coupler.fvm_solver.step // coupler.n_fvm_substeps)
    )
    suffix = f"{step:06d}"
    partitioned = coupler.fvm_solver.parallel.is_partitioned
    fvm_artifact = f"fvm_{suffix}" if partitioned else f"fvm_{suffix}.npz"

    # Partitioned FVM checkpoints are collective; every rank must enter first.
    coupler.fvm_solver.save_state(target / fvm_artifact)
    if not coupler._is_master:
        return target
    if coupler.vpm_solver is None:
        raise RuntimeError("Initialize the coupler before saving a checkpoint")

    coupler.vpm_solver.save_numerical_state(str(target / f"vpm_{suffix}"))
    boundary_artifact = f"vpm_boundary_condition_{suffix}.npz"
    boundary_temporary = target / f".{boundary_artifact}.tmp"
    try:
        with open(boundary_temporary, "wb") as stream:
            np.savez_compressed(
                stream,
                physical_field_schema_version=np.asarray(SCHEMA_VERSION),
                boundary_schema_version=np.asarray(1, dtype=np.int64),
                has_velocity=np.asarray(coupler._velocity_boundary_condition_old is not None),
                velocity=np.empty((0, 3))
                if coupler._velocity_boundary_condition_old is None
                else coupler._velocity_boundary_condition_old,
                has_normal_velocity=np.asarray(
                    coupler._normal_velocity_boundary_condition_old is not None
                ),
                normal_velocity=(
                    np.empty(0)
                    if coupler._normal_velocity_boundary_condition_old is None
                    else coupler._normal_velocity_boundary_condition_old
                ),
                has_tangential_gradient=np.asarray(
                    coupler._tangential_gradient_boundary_condition_old is not None
                ),
                tangential_gradient=(
                    np.empty((0, 3))
                    if coupler._tangential_gradient_boundary_condition_old is None
                    else coupler._tangential_gradient_boundary_condition_old
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(boundary_temporary, target / boundary_artifact)
    finally:
        boundary_temporary.unlink(missing_ok=True)

    manifest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "physical_field_schema_version": SCHEMA_VERSION,
        "kind": "openonda.coupled_checkpoint",
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": "fvm",
        "config_sha256": config_digest(coupler.setup),
        "config": coupler.setup.to_dict(),
        "coupling_step": step,
        "time": float(coupler.vpm_solver.time),
        "fvm_step": int(coupler.fvm_solver.step),
        "vpm_step": int(coupler.vpm_solver.step),
        "n_fvm_substeps": int(coupler.n_fvm_substeps),
        "artifacts": {
            "fvm": fvm_artifact,
            "vpm": f"vpm_{suffix}.h5",
            "vpm_xdmf": f"vpm_{suffix}.xdmf",
            "vpm_boundary_condition": boundary_artifact,
        },
    }
    manifest["artifact_sha256"] = {
        name: artifact_digest(target / artifact) for name, artifact in manifest["artifacts"].items()
    }
    manifest_temporary = target / "manifest.json.tmp"
    with manifest_temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(manifest_temporary, target / "manifest.json")

    keep = {"manifest.json", *manifest["artifacts"].values()}
    stale = {
        *target.glob("fvm_*"),
        *target.glob("vpm_*"),
        *target.glob("vpm_boundary_condition_*"),
    }
    for artifact in stale:
        if artifact.name in keep or not artifact.exists():
            continue
        if artifact.is_dir():
            shutil.rmtree(artifact)
        else:
            artifact.unlink()
    return target


def load_coupled_state(coupler, directory, *, comm=None) -> int:
    """Restore both solvers and the VPM boundary-history state."""
    if coupler.fvm_solver is None or (coupler._is_master and coupler.vpm_solver is None):
        raise RuntimeError("Initialize the coupler before loading a checkpoint")

    target = Path(directory)
    error: str | None = None
    manifest: dict | None = None
    artifacts: dict[str, str] = {}
    artifact_paths: dict[str, Path] = {}
    if coupler._is_master:
        try:
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        except OSError as exc:
            error = f"Cannot read coupled checkpoint manifest at {target}: {exc}"
        except json.JSONDecodeError as exc:
            error = f"Invalid coupled checkpoint manifest at {target}: {exc}"
        if error is None and not isinstance(manifest, dict):
            error = "Coupled checkpoint manifest must be a JSON object"
        if error is None:
            assert isinstance(manifest, dict)
            version = manifest.get("format_version")
            schema_version = manifest.get("physical_field_schema_version")
            if (
                version != CHECKPOINT_FORMAT_VERSION
                or manifest.get("backend") != "fvm"
                or schema_version != SCHEMA_VERSION
            ):
                error = "Unsupported coupled checkpoint format or backend"
            else:
                stored_artifacts = manifest.get("artifacts", {})
                if not isinstance(stored_artifacts, dict):
                    error = "Coupled checkpoint artifacts must be a mapping"
                else:
                    artifacts = dict(stored_artifacts)
            if error is None:
                manifest["artifacts"] = artifacts
                try:
                    artifact_paths = {
                        name: _resolve_artifact(target, artifact)
                        for name, artifact in artifacts.items()
                    }
                except ValueError as exc:
                    error = str(exc)
                    artifact_paths = {}
                artifact_hashes = manifest.get("artifact_sha256", {})
                if artifact_hashes and not isinstance(artifact_hashes, dict):
                    error = "Coupled checkpoint artifact_sha256 must be a mapping"
                if error is None and (
                    not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(artifacts)
                ):
                    error = (
                        "Coupled checkpoint format 8 requires one SHA-256 digest "
                        "for every declared artifact"
                    )
                if error is None and artifact_hashes:
                    for name, expected_hash in artifact_hashes.items():
                        artifact = artifacts.get(name)
                        if not artifact or name not in artifact_paths:
                            error = f"Coupled checkpoint manifest hashes unknown artifact {name!r}"
                            break
                        artifact_path = artifact_paths[name]
                        if (
                            not isinstance(expected_hash, str)
                            or len(expected_hash) != 64
                            or not artifact_path.exists()
                            or artifact_digest(artifact_path) != expected_hash
                        ):
                            error = f"Coupled checkpoint artifact hash mismatch: {artifact}"
                            break
            required_artifacts = [
                "fvm",
                "vpm",
                "vpm_boundary_condition",
            ]
            required_artifacts.append("vpm_xdmf")
            missing = [
                name
                for name in required_artifacts
                if not artifacts.get(name)
                or name not in artifact_paths
                or not artifact_paths[name].exists()
            ]
            if error is None and missing:
                error = f"Incomplete coupled checkpoint; missing: {', '.join(missing)}"
            elif error is None and manifest.get("config_sha256") != config_digest(coupler.setup):
                changes = config_diff(manifest.get("config"), coupler.setup.to_dict())
                detail = "\n  ".join(changes) if changes else "(checkpoint stored no config)"
                error = f"Coupled checkpoint configuration differs:\n  {detail}"
    if comm is not None and comm.Get_size() > 1:
        error, manifest = comm.bcast(
            (error, manifest) if coupler._is_master else None,
            root=0,
        )
    if error is not None:
        raise ValueError(error)
    assert manifest is not None
    artifacts = manifest["artifacts"]

    coupler.fvm_solver.load_state(target / artifacts["fvm"])
    expected_fvm_step = int(manifest["vpm_step"]) * coupler.n_fvm_substeps
    if coupler.fvm_solver.step != expected_fvm_step:
        raise ValueError(
            f"Coupled checkpoint time-step mismatch: FVM={coupler.fvm_solver.step}, "
            f"expected {expected_fvm_step} from VPM={manifest['vpm_step']}"
        )

    if coupler._is_master:
        assert coupler.vpm_solver is not None
        coupler.vpm_solver.load_numerical_state(str(target / artifacts["vpm"]))
        with np.load(target / artifacts["vpm_boundary_condition"], allow_pickle=False) as boundary:
            if (
                "physical_field_schema_version" not in boundary.files
                or str(boundary["physical_field_schema_version"]) != SCHEMA_VERSION
            ):
                raise ValueError(
                    "Coupled boundary checkpoint has an unsupported physical-field schema"
                )
            if (
                "boundary_schema_version" not in boundary.files
                or int(boundary["boundary_schema_version"]) != 1
            ):
                raise ValueError("Unsupported coupled boundary checkpoint schema")
            coupler._velocity_boundary_condition_old = (
                boundary["velocity"].copy() if bool(boundary["has_velocity"]) else None
            )
            coupler._normal_velocity_boundary_condition_old = (
                boundary["normal_velocity"].copy()
                if bool(boundary["has_normal_velocity"])
                else None
            )
            coupler._tangential_gradient_boundary_condition_old = (
                boundary["tangential_gradient"].copy()
                if bool(boundary["has_tangential_gradient"])
                else None
            )
            coupler._normal_velocity_boundary_condition = None
            coupler._tangential_gradient_boundary_condition = None
        if not np.isclose(coupler.fvm_solver.time, coupler.vpm_solver.time, rtol=0.0, atol=1e-12):
            error = f"Coupled checkpoint time mismatch: FVM={coupler.fvm_solver.time}, VPM={coupler.vpm_solver.time}"
    if comm is not None and comm.Get_size() > 1:
        error = comm.bcast(error if coupler._is_master else None, root=0)
    if error is not None:
        raise ValueError(error)
    return int(manifest["coupling_step"])


__all__ = [
    "CHECKPOINT_DIRECTORY",
    "CHECKPOINT_FORMAT_VERSION",
    "config_diff",
    "config_digest",
    "load_coupled_state",
    "save_coupled_state",
]
