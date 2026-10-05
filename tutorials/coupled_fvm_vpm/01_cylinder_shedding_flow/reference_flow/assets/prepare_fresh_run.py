"""Archive the selected cylinder run before starting again at t=0."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import uuid

from openonda.runtime import detected_world_size


def archive_previous_run(case_dir: Path, *, reuse_mesh: bool = False) -> Path | None:
    """Move only the selected case's outputs; preserve its sibling cases."""
    if detected_world_size() != 1:
        raise RuntimeError("Run ./allrun.sh --fresh outside mpiexec; it launches MPI itself")
    case_dir = case_dir.resolve()
    paths = [case_dir / name for name in ("solution", "samples", "figures")]
    paths.extend(sorted(case_dir.glob("*.log")))
    paths = [path for path in paths if path.exists() or path.is_symlink()]
    if not paths:
        return None
    keep_mesh = False
    mesh_path = case_dir / "solution/fvm/mesh.npz"
    if reuse_mesh and mesh_path.is_file():
        from openonda.tutorial_runner import load_case_module

        setup = load_case_module(case_dir)
        *_, builder = setup.build_case()
        keep_mesh = setup.cached_mesh_matches_case(mesh_path, builder, require_identity=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = case_dir / "previous_runs" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    archive.mkdir(parents=True)
    moved = []
    try:
        for path in paths:
            path.rename(archive / path.name)
            moved.append(path.name)
    except OSError:
        # Restore the original layout if an output could not be moved.
        for name in reversed(moved):
            (archive / name).rename(case_dir / name)
        raise
    (archive / "archive_manifest.json").write_text(
        json.dumps(
            {
                "case": str(case_dir),
                "created_utc": stamp,
                "outputs": moved,
                "reused_compatible_mesh": keep_mesh,
            },
            indent=2,
        )
        + "\n"
    )
    if keep_mesh:
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive / "solution/fvm/mesh.npz", mesh_path)
    return archive
