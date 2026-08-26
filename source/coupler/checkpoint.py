"""Atomic restart checkpoints for coupled FVM--VPM state."""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import warnings

import numpy as np

from source.solvers.fvm.io.checkpoint import decode_state, encode_state

CHECKPOINT_DIRECTORY = "checkpoints"
CHECKPOINT_FORMAT_VERSION = 11

_VPM_OPERATIONAL_CONFIG_FIELDS = frozenset(
    {
        "time",
        "step",
        "logging_interval_steps",
        "timing_interval_steps",
        "checkpoint_interval_steps",
        "checkpoint_name",
        "checkpoint_directory",
        "sample_subdirectory",
        "export_flow_integrals",
        "export_discretization_health",
        "log_mode",
        "clean",
        "write_precision",
        "checkpoint_store_velocity_gradient",
        "debug_mode",
        "verbose",
    }
)


def config_mapping_digest(config: dict) -> str:
    """Hash an already-serialized configuration mapping."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def config_digest(config) -> str:
    return config_mapping_digest(config.to_dict())


def _vpm_numerical_config(vpm_setup) -> dict:
    """Return only VPM settings that can affect the continued solution."""
    serialized = vpm_setup.to_dict()
    if not isinstance(serialized, dict):
        raise TypeError("VPM checkpoint configuration must serialize to a mapping")
    return {
        key: value for key, value in serialized.items() if key not in _VPM_OPERATIONAL_CONFIG_FIELDS
    }


def _panel_numerical_config(panel_solver) -> dict | None:
    """Serialize the constructor-level panel choices, excluding diagnostics."""
    if panel_solver is None:
        return None
    force_config = getattr(panel_solver, "force_config", None)
    freestream_velocity = getattr(panel_solver, "freestream_velocity", None)
    return {
        "type": f"{type(panel_solver).__module__}.{type(panel_solver).__qualname__}",
        "max_n_panels": int(panel_solver.max_n_panels),
        "float_dtype": panel_solver.float_dtype,
        "linear_solver": panel_solver.linear_solver_name,
        "force_method": None if force_config is None else force_config.method,
        "boundary_condition_type": panel_solver.boundary_condition_type,
        "density": float(panel_solver.density),
        "freestream_velocity": (
            None
            if freestream_velocity is None
            else np.asarray(freestream_velocity, dtype=np.float64).tolist()
        ),
        "coupling_scope": panel_solver.coupling_scope,
        "raise_on_non_convergence": bool(panel_solver.raise_on_non_convergence),
        "residual_tolerance": panel_solver.residual_tolerance,
        "far_field_acceptance": float(panel_solver.far_field_acceptance),
        "far_field_min_panels": int(panel_solver.far_field_min_panels),
        "reuse_constrained_factorization": bool(panel_solver.reuse_constrained_factorization),
    }


def _checkpoint_config(coupler) -> dict:
    """Build the strict restart identity for all coupled numerical components."""
    if coupler.vpm_solver is None:
        raise RuntimeError("Initialize the coupler before checkpointing configuration")
    config = dict(coupler.setup.to_dict())
    config["vpm"] = _vpm_numerical_config(coupler.vpm_solver.setup)
    config["panel"] = _panel_numerical_config(coupler.vpm_solver.panel_solver)
    return config


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


def _config_differences(
    stored: object,
    current: object,
    *,
    prefix: str = "",
) -> list[tuple[str, object, object]]:
    """Return recursive leaf differences with stable dotted paths."""
    if isinstance(stored, dict) and isinstance(current, dict):
        differences: list[tuple[str, object, object]] = []
        for key in sorted(set(stored) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in stored:
                differences.append((path, "<missing>", current[key]))
            elif key not in current:
                differences.append((path, stored[key], "<missing>"))
            else:
                differences.extend(_config_differences(stored[key], current[key], prefix=path))
        return differences
    if stored != current:
        return [(prefix, stored, current)]
    return []


def config_diff(stored: dict | None, current: dict) -> list[str]:
    """Return recursive ``path: old -> new`` configuration differences."""
    if stored is None:
        return []
    return [
        f"{path}: {old!r} -> {new!r}" for path, old, new in _config_differences(stored, current)
    ]


def config_difference_paths(stored: dict | None, current: dict) -> set[str]:
    """Return the exact recursive configuration paths whose values differ."""
    if stored is None:
        return set()
    return {path for path, _old, _new in _config_differences(stored, current)}


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
    boundary_state = {
        "boundary_schema_version": np.asarray(3, dtype=np.int64),
        "has_velocity": np.asarray(coupler._velocity_boundary_condition_old is not None),
        "velocity": (
            np.empty((0, 3))
            if coupler._velocity_boundary_condition_old is None
            else coupler._velocity_boundary_condition_old
        ),
        "has_normal_velocity": np.asarray(
            coupler._normal_velocity_boundary_condition_old is not None
        ),
        "normal_velocity": (
            np.empty(0)
            if coupler._normal_velocity_boundary_condition_old is None
            else coupler._normal_velocity_boundary_condition_old
        ),
        "has_tangential_gradient": np.asarray(
            coupler._tangential_gradient_boundary_condition_old is not None
        ),
        "tangential_gradient": (
            np.empty((0, 3))
            if coupler._tangential_gradient_boundary_condition_old is None
            else coupler._tangential_gradient_boundary_condition_old
        ),
        "has_kinematic_pressure_gradient": np.asarray(
            coupler._kinematic_pressure_gradient_boundary_condition_old is not None
        ),
        "kinematic_pressure_gradient": (
            np.empty((0, 3))
            if coupler._kinematic_pressure_gradient_boundary_condition_old is None
            else coupler._kinematic_pressure_gradient_boundary_condition_old
        ),
        "has_pressure_velocity_snapshot": np.asarray(
            coupler._pressure_velocity_snapshot is not None
        ),
        "pressure_velocity_snapshot": (
            np.empty((0, 3))
            if coupler._pressure_velocity_snapshot is None
            else coupler._pressure_velocity_snapshot
        ),
    }
    try:
        with open(boundary_temporary, "wb") as stream:
            np.savez_compressed(stream, **encode_state(boundary_state))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(boundary_temporary, target / boundary_artifact)
    finally:
        boundary_temporary.unlink(missing_ok=True)

    checkpoint_config = _checkpoint_config(coupler)
    manifest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "openonda.coupled_checkpoint",
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": "fvm",
        "config_sha256": config_mapping_digest(checkpoint_config),
        "config": checkpoint_config,
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


def load_coupled_state(
    coupler,
    directory,
    *,
    comm=None,
    allowed_config_differences: Collection[str] = (),
) -> int:
    """Restore both solvers and the VPM boundary-history state.

    Configuration matching remains strict unless a caller explicitly names
    the exact paths allowed to differ for a controlled restart experiment.
    Artifact integrity and every unlisted configuration field remain strict.
    """
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
            expected_manifest_keys = {
                "format_version",
                "kind",
                "created_utc",
                "backend",
                "config_sha256",
                "config",
                "coupling_step",
                "time",
                "fvm_step",
                "vpm_step",
                "n_fvm_substeps",
                "artifacts",
                "artifact_sha256",
            }
            version = manifest.get("format_version")
            if (
                set(manifest) != expected_manifest_keys
                or version != CHECKPOINT_FORMAT_VERSION
                or manifest.get("kind") != "openonda.coupled_checkpoint"
                or manifest.get("backend") != "fvm"
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
                        f"Coupled checkpoint format {CHECKPOINT_FORMAT_VERSION} requires one "
                        "SHA-256 digest "
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
            elif error is None:
                stored_config = manifest.get("config")
                if not isinstance(stored_config, dict):
                    error = "Coupled checkpoint configuration must be a mapping"
                elif manifest.get("config_sha256") != config_mapping_digest(stored_config):
                    error = "Coupled checkpoint stored configuration hash mismatch"
                else:
                    current_config = _checkpoint_config(coupler)
                    if manifest.get("config_sha256") != config_mapping_digest(current_config):
                        changed_paths = config_difference_paths(stored_config, current_config)
                        unexpected = changed_paths - set(allowed_config_differences)
                        changes = config_diff(stored_config, current_config)
                        detail = "\n  ".join(changes) if changes else "(no structured diff)"
                        if unexpected:
                            error = (
                                "Coupled checkpoint configuration differs outside the explicit "
                                "allowlist:\n  "
                                + detail
                                + "\n  disallowed paths: "
                                + ", ".join(sorted(unexpected))
                            )
                        else:
                            warnings.warn(
                                "Loading a coupled checkpoint with explicitly allowed "
                                f"configuration differences: {', '.join(sorted(changed_paths))}",
                                RuntimeWarning,
                                stacklevel=2,
                            )
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
            expected_boundary_keys = {
                "boundary_schema_version",
                "has_velocity",
                "velocity",
                "has_normal_velocity",
                "normal_velocity",
                "has_tangential_gradient",
                "tangential_gradient",
                "has_kinematic_pressure_gradient",
                "kinematic_pressure_gradient",
                "has_pressure_velocity_snapshot",
                "pressure_velocity_snapshot",
                "storage_layout",
            }
            if set(boundary.files) != expected_boundary_keys:
                raise ValueError("Coupled boundary checkpoint has invalid fields")
            boundary_state = decode_state(
                {name: np.array(boundary[name], copy=True) for name in boundary.files}
            )
            if (
                "boundary_schema_version" not in boundary_state
                or int(boundary_state["boundary_schema_version"]) != 3
            ):
                raise ValueError("Unsupported coupled boundary checkpoint schema")
            coupler._velocity_boundary_condition_old = (
                boundary_state["velocity"].copy() if bool(boundary_state["has_velocity"]) else None
            )
            coupler._normal_velocity_boundary_condition_old = (
                boundary_state["normal_velocity"].copy()
                if bool(boundary_state["has_normal_velocity"])
                else None
            )
            coupler._tangential_gradient_boundary_condition_old = (
                boundary_state["tangential_gradient"].copy()
                if bool(boundary_state["has_tangential_gradient"])
                else None
            )
            coupler._kinematic_pressure_gradient_boundary_condition_old = (
                boundary_state["kinematic_pressure_gradient"].copy()
                if bool(boundary_state["has_kinematic_pressure_gradient"])
                else None
            )
            coupler._pressure_velocity_snapshot = (
                boundary_state["pressure_velocity_snapshot"].copy()
                if bool(boundary_state["has_pressure_velocity_snapshot"])
                else None
            )
            coupler._normal_velocity_boundary_condition = None
            coupler._tangential_gradient_boundary_condition = None
            coupler._kinematic_pressure_gradient_boundary_condition = None
        if not np.isclose(coupler.fvm_solver.time, coupler.vpm_solver.time, rtol=0.0, atol=1e-12):
            error = f"Coupled checkpoint time mismatch: FVM={coupler.fvm_solver.time}, VPM={coupler.vpm_solver.time}"
    if comm is not None and comm.Get_size() > 1:
        error = comm.bcast(error if coupler._is_master else None, root=0)
    if error is not None:
        raise ValueError(error)
    coupling_step = int(manifest["coupling_step"])
    if coupler.vorticity_transfer is None:
        raise RuntimeError("Coupled checkpoint load requires an initialized vorticity transfer")
    # A fresh run performs one initial synchronization plus one transfer after
    # every completed coupling interval.  Preserve that cadence so resumed
    # diagnostics and their cost remain identical to an uninterrupted run.
    coupler.vorticity_transfer.step = coupling_step + 1
    return coupling_step


__all__ = [
    "CHECKPOINT_DIRECTORY",
    "CHECKPOINT_FORMAT_VERSION",
    "config_diff",
    "config_difference_paths",
    "config_digest",
    "config_mapping_digest",
    "load_coupled_state",
    "save_coupled_state",
]
