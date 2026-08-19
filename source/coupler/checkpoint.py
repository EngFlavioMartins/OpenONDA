"""Atomic restart checkpoints for coupled FVM--VPM state."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

CHECKPOINT_DIRECTORY = "checkpoint"
CHECKPOINT_FORMAT_VERSION = 4


def config_digest(config) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


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
    if coupler.fvm is None:
        raise RuntimeError("Initialize the coupler before saving a checkpoint")

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    step = (
        int(coupling_step)
        if coupling_step is not None
        else int(coupler.fvm.step // coupler.fvm_substeps)
    )
    suffix = f"{step:06d}"
    partitioned = coupler.fvm.parallel.is_partitioned
    fvm_artifact = f"fvm_{suffix}" if partitioned else f"fvm_{suffix}.npz"

    # Partitioned FVM checkpoints are collective; every rank must enter first.
    coupler.fvm.save_state(target / fvm_artifact)
    if not coupler._is_master:
        return target
    if coupler.vpm is None:
        raise RuntimeError("Initialize the coupler before saving a checkpoint")

    coupler.vpm.save_numerical_state(str(target / f"vpm_{suffix}"))
    boundary_artifact = f"vpm_bc_{suffix}.npz"
    boundary_temporary = target / f".{boundary_artifact}.tmp"
    with open(boundary_temporary, "wb") as stream:
        np.savez_compressed(
            stream,
            u_present=np.asarray(coupler._u_bc_prev is not None),
            u=np.empty((0, 3)) if coupler._u_bc_prev is None else coupler._u_bc_prev,
            normal_velocity_present=np.asarray(coupler._normal_velocity_bc_prev is not None),
            normal_velocity=(
                np.empty(0)
                if coupler._normal_velocity_bc_prev is None
                else coupler._normal_velocity_bc_prev
            ),
            tangential_gradient_present=np.asarray(
                coupler._tangential_gradient_bc_prev is not None
            ),
            tangential_gradient=(
                np.empty((0, 3))
                if coupler._tangential_gradient_bc_prev is None
                else coupler._tangential_gradient_bc_prev
            ),
        )
    os.replace(boundary_temporary, target / boundary_artifact)

    manifest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "openonda.coupled_checkpoint",
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": "fvm",
        "config_sha256": config_digest(coupler.config),
        "config": coupler.config.to_dict(),
        "coupling_step": step,
        "flow_time": float(coupler.vpm.time),
        "fvm_time_step": int(coupler.fvm.step),
        "vpm_time_step": int(coupler.vpm.step),
        "n_fvm_substeps": int(coupler.fvm_substeps),
        "artifacts": {
            "fvm": fvm_artifact,
            "vpm": f"vpm_{suffix}.h5",
            "vpm_bc": boundary_artifact,
        },
    }
    manifest_temporary = target / "manifest.json.tmp"
    manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_temporary, target / "manifest.json")

    keep = {"manifest.json", *manifest["artifacts"].values(), f"vpm_{suffix}.xdmf"}
    stale = {*target.glob("fvm_*"), *target.glob("vpm_*"), *target.glob("vpm_bc_*")}
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
    if coupler.fvm is None or (coupler._is_master and coupler.vpm is None):
        raise RuntimeError("Initialize the coupler before loading a checkpoint")

    target = Path(directory)
    error: str | None = None
    manifest: dict | None = None
    artifacts: dict[str, str] = {}
    if coupler._is_master:
        try:
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        except OSError as exc:
            error = f"Cannot read coupled checkpoint manifest at {target}: {exc}"
        if error is None:
            assert manifest is not None
            if (
                manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION
                or manifest.get("backend") != "fvm"
            ):
                error = "Unsupported coupled checkpoint format or backend"
            else:
                artifacts = manifest.get("artifacts", {})
            missing = [
                name
                for name in ("fvm", "vpm", "vpm_bc")
                if not artifacts.get(name) or not (target / artifacts[name]).exists()
            ]
            if error is None and missing:
                error = f"Incomplete coupled checkpoint; missing: {', '.join(missing)}"
            elif error is None and manifest.get("config_sha256") != config_digest(coupler.config):
                changes = config_diff(manifest.get("config"), coupler.config.to_dict())
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

    coupler.fvm.load_state(target / artifacts["fvm"])
    expected_fvm_step = int(manifest["vpm_time_step"]) * coupler.fvm_substeps
    if coupler.fvm.step != expected_fvm_step:
        raise ValueError(
            f"Coupled checkpoint time-step mismatch: FVM={coupler.fvm.step}, "
            f"expected {expected_fvm_step} from VPM={manifest['vpm_time_step']}"
        )

    if coupler._is_master:
        assert coupler.vpm is not None
        coupler.vpm.load_numerical_state(str(target / artifacts["vpm"]))
        with np.load(target / artifacts["vpm_bc"], allow_pickle=False) as boundary:
            coupler._u_bc_prev = boundary["u"].copy() if bool(boundary["u_present"]) else None
            coupler._normal_velocity_bc_prev = (
                boundary["normal_velocity"].copy()
                if "normal_velocity_present" in boundary.files
                and bool(boundary["normal_velocity_present"])
                else None
            )
            coupler._tangential_gradient_bc_prev = (
                boundary["tangential_gradient"].copy()
                if "tangential_gradient_present" in boundary.files
                and bool(boundary["tangential_gradient_present"])
                else None
            )
            coupler._normal_velocity_bc_next = None
            coupler._tangential_gradient_bc_next = None
        if not np.isclose(coupler.fvm.time, coupler.vpm.time, rtol=0.0, atol=1e-12):
            error = (
                f"Coupled checkpoint time mismatch: FVM={coupler.fvm.time}, VPM={coupler.vpm.time}"
            )
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
