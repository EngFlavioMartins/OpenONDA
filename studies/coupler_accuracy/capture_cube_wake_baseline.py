"""Preserve the interrupted cube experiment before controlled drift studies.

Uses independent APFS clones on macOS; neither the original case nor its
reference is modified. Native states remain local under ignored solution/
directories. Sampler data, metadata, hashes and the existing tutorial figures
can be committed without putting numerical checkpoints in Git.
"""

from __future__ import annotations

from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/02_cube_flow"
DESTINATION = ROOT / "studies/coupler_accuracy/results/cube-wake-drift-2026-09-15/baseline"


def main():
    """Clone once and record the exact bytes used by subsequent probes."""
    DESTINATION.mkdir(parents=True, exist_ok=False)
    directories = (
        ("solution", "solution/coupled"),
        ("constant", "solution/coupled_mesh"),
        ("samples", "samples/coupled"),
        ("reference_flow/solution/fine", "solution/reference_fine"),
        ("reference_flow/samples/fine", "samples/reference_fine"),
    )
    entries = []
    for relative, copied in directories:
        source = CASE / relative
        destination = DESTINATION / copied
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/bin/cp", "-cR", str(source), str(destination)], check=True)
        for original in sorted(source.rglob("*")):
            if not original.is_file():
                continue
            archived = destination / original.relative_to(source)
            with archived.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            entries.append(
                {
                    "source": str(original.relative_to(ROOT)),
                    "archive": str(archived.relative_to(DESTINATION)),
                    "size": archived.stat().st_size,
                    "sha256": digest,
                }
            )
        print(f"Preserved {relative}", flush=True)
    metadata = DESTINATION / "metadata"
    metadata.mkdir()
    for relative, name in (
        ("solution/fvm_metadata.json", "coupled_fvm.json"),
        ("solution/vpm_metadata.json", "coupled_vpm.json"),
        ("solution/backups/manifest.json", "coupled_checkpoint.json"),
        ("reference_flow/solution/fine/fvm_metadata.json", "reference_fine.json"),
    ):
        shutil.copy2(CASE / relative, metadata / name)
    telemetry = DESTINATION / "telemetry"
    telemetry.mkdir()
    for relative, name in (
        ("solution/coupled/diagnostics.jsonl", "coupled_fvm.jsonl.gz"),
        ("solution/coupled/coupler_diagnostics.jsonl", "coupler.jsonl.gz"),
        ("solution/reference_fine/diagnostics.jsonl", "reference_fine.jsonl.gz"),
    ):
        (telemetry / name).write_bytes(
            gzip.compress((DESTINATION / relative).read_bytes(), mtime=0)
        )
    manifest = {
        "captured_utc": datetime.now(UTC).isoformat(),
        "case": str(CASE.relative_to(ROOT)),
        "entries": entries,
        "native_states": "Local immutable clones, excluded from Git by the solution/ rule.",
        "source_identity": "Use the baseline commit and archived runtime metadata; metadata does not identify a run-time Git revision.",
    }
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Captured {len(entries)} files in {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
