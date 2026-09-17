"""Audit a published native VPM/VLM checkpoint prefix.

A published checkpoint owns three synchronized series below one solution
directory: native ``vpm/vpm_<step>.h5`` frames with their ``.xdmf`` templated
companions, and the root-level ``vlm.pvd`` time series of triangulated surface
frames below ``vlm/``. ``audit_checkpoint`` proves that every native backup up
to and including the audited step is present, paired with its XDMF/VTP
companions, and that all clocks agree.  Later publications are ignored so the
audit can run while the solver is writing its next outputs.

This helper is deliberately a shared owner: the Delta-wing lineage finalizer,
the flat-plate and delta-wing studies, and the checkpoint-audit qualification
tests all consume the same verification entry point.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import re
import sys
from typing import Final, cast

from defusedxml import ElementTree
import h5py
import numpy as np

from source.solution_layout import collection_path

# Detection tolerance must be below the dataset's own isclose margins so that a
# single ulp drift (for example 1.0 + 1e-12) is reported as a clock conflict.
CLOCK_ATOL: Final[float] = 1.0e-13

_NATIVE_RE: Final = re.compile(r"vpm_(\d{6})\.(?:h5|xdmf)")
_NATIVE_TMP_RE: Final = re.compile(r"vpm_(\d{6})\.(?:h5|xdmf)\.tmp")
_VTP_RE: Final = re.compile(r"vlm_(\d{6})\.vtp")


def _xdmf_time(path: Path) -> float:
    parsed = ElementTree.parse(str(path))
    values = [
        float(element.attrib["Value"])
        for element in parsed.iter()
        if element.tag.endswith("Time") and "Value" in element.attrib
    ]
    if len(values) != 1:
        raise ValueError(f"XDMF companion must carry exactly one Time value: {path}")
    return values[0]


def _vtp_time(path: Path) -> float:
    """Read the clock from ASCII, encoded or raw-appended native VTK files."""
    try:
        parsed = ElementTree.parse(str(path))
        values = []
        for element in parsed.iter():
            if element.tag.endswith("DataArray") and element.attrib.get("Name") == "TimeValue":
                if element.attrib.get("format", "ascii") != "ascii":
                    raise ValueError("encoded VTK field requires the VTK reader")
                if element.text is not None:
                    values.append(float(element.text))
    except (ElementTree.ParseError, ValueError):
        import pyvista as pv

        # Raw AppendedData is valid VTK but not parseable by an XML reader.
        frame = pv.read(path)
        values = np.asarray(frame.field_data.get("TimeValue", [])).reshape(-1).tolist()
    if len(values) != 1:
        raise ValueError(f"VTP frame must carry exactly one TimeValue: {path}")
    return values[0]


def _check_clock(value: float, expected: float, kind: str, path: Path) -> None:
    if not np.isfinite(value) or not np.isclose(value, expected, rtol=0.0, atol=CLOCK_ATOL):
        raise ValueError(
            f"{kind} time {value} is not the native VPM clock {expected} ({path.name})"
        )


def _classify(path: Path) -> tuple[str, int] | None:
    """Classify a native ``vpm_*`` file in the prefix directory.

    Returns a ``("checkpoint"|"unfinished", step)`` pair, ``None`` for files
    that are not checkpoint frames (temporal collections, unrelated resources),
    and raises for malformed numeric filenames below the ``vpm_*`` prefix.
    """
    name = path.name
    if not name.startswith("vpm_"):
        return None
    if name == "vpm_series_temporal.xdmf":
        return None
    if name.endswith(".tmp"):
        match = _NATIVE_TMP_RE.fullmatch(name)
        if match:
            return "unfinished", int(match.group(1))
        if name[4].isdigit():
            raise ValueError(f"invalid vpm series filename: {name}")
        return None
    match = _NATIVE_RE.fullmatch(name)
    if match:
        return "checkpoint", int(match.group(1))
    if name[4].isdigit():
        raise ValueError(f"invalid vpm series filename: {name}")
    return None


def _pvd_timestep(entries: Sequence[tuple[str, float]], candidate_step: int) -> float:
    for name, timestep in entries:
        frame_match = _VTP_RE.fullmatch(Path(name).name)
        if frame_match and int(frame_match.group(1)) == candidate_step:
            return timestep
    raise ValueError(f"vlm.pvd does not index step {candidate_step:06d}")


def audit_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    """Audit the published prefix through one native checkpoint.

    ``checkpoint_path`` is a native ``vpm_<step>.h5`` backup.  The audit scans
    its solution directory, proves the native/XDMF/VTP/pvd clocks agree through
    that step, and returns a machine-readable evidence snapshot.

    Raises:
        ValueError: If the prefix is incomplete, unpublished, or inconsistent
            at or before the audited step.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    frame_directory = checkpoint_path.parent
    if frame_directory.name != "vpm":
        raise ValueError("checkpoint must be stored below a solution/vpm directory")
    solution_directory = frame_directory.parent

    match = _NATIVE_RE.fullmatch(checkpoint_path.name)
    if not match:
        raise ValueError(f"invalid vpm series filename: {checkpoint_path.name}")
    step = int(match.group(1))

    with h5py.File(checkpoint_path, "r") as archive:
        solver = archive["solver"]
        native_step = int(solver.attrs["step"])
        time = float(solver.attrs["time"])
        identity = str(archive["solver/vlm"].attrs["identity"])
    if native_step != step:
        raise ValueError(
            f"native VPM step/time conflicts with filename {checkpoint_path.name}: "
            f"stored step {native_step}"
        )

    h5_files: dict[int, Path] = {}
    xdmf_files: dict[int, Path] = {}
    for path in sorted(frame_directory.iterdir()):
        classification = _classify(path)
        if classification is None:
            continue
        kind, candidate_step = classification
        if candidate_step > step:
            continue
        if kind == "unfinished":
            raise ValueError(f"native HDF5/XDMF prefix is not fully published: {path.name}")
        if path.suffix == ".h5":
            h5_files[candidate_step] = path
        else:
            xdmf_files[candidate_step] = path

    native_steps: set[int] = set()
    for candidate_step in sorted(h5_files.keys() | xdmf_files.keys()):
        has_h5 = candidate_step in h5_files
        has_xdmf = candidate_step in xdmf_files
        if not has_h5 or not has_xdmf:
            missing = "h5" if has_xdmf else "xdmf"
            raise ValueError(
                f"native HDF5/XDMF prefix is not fully published: "
                f"missing vpm_{candidate_step:06d}.{missing}"
            )
        path = h5_files[candidate_step]
        with h5py.File(path, "r") as archive:
            stored_step = int(archive["solver"].attrs["step"])
            stored_time = float(archive["solver"].attrs["time"])
        if stored_step != candidate_step:
            raise ValueError(
                f"native VPM step/time conflicts with filename {path.name}: "
                f"stored step {stored_step}"
            )
        _check_clock(stored_time, _xdmf_time(xdmf_files[candidate_step]), "native", path)
        native_steps.add(candidate_step)

    pvd = collection_path(solution_directory, "vlm")
    if not pvd.is_file():
        raise ValueError(f"missing vlm.pvd index below {solution_directory.name}")
    parsed = ElementTree.parse(str(pvd))
    entries: list[tuple[str, float]] = []
    for element in parsed.iter():
        if element.tag.endswith("DataSet"):
            entries.append((element.attrib["file"], float(element.attrib["timestep"])))
    pvd_counts: dict[int, int] = {}
    for name, _timestep in entries:
        frame_match = _VTP_RE.fullmatch(Path(name).name)
        if not frame_match:
            raise ValueError(f"invalid vlm series filename: {name}")
        candidate_step = int(frame_match.group(1))
        if candidate_step <= step:
            pvd_counts[candidate_step] = pvd_counts.get(candidate_step, 0) + 1

    frame_steps: set[int] = set()
    last_frame = None
    for candidate_step in sorted(pvd_counts):
        vtp = solution_directory / f"vlm/vlm_{candidate_step:06d}.vtp"
        if not vtp.is_file():
            raise ValueError(f"missing surface frame referenced by vlm.pvd: {vtp.name}")
        _check_clock(
            _vtp_time(vtp),
            _pvd_timestep(entries, candidate_step),
            "index",
            vtp,
        )
        frame_steps.add(candidate_step)
        last_frame = vtp.name

    extra = sorted(frame_steps - native_steps)
    missing = sorted(native_steps - frame_steps)
    if extra or missing:
        raise ValueError(
            f"native VPM backup steps (extra={extra}, missing={missing}) above {solution_directory}"
        )

    return {
        "step": step,
        "time": time,
        "vlm_identity": identity,
        "vtp_series": {
            "frames": sorted(frame_steps),
            "frame_count": len(frame_steps),
            "pvd_entries": len(pvd_counts),
            "native_checkpoint_count": len(native_steps),
            "last_frame": last_frame,
            "through_step": max(frame_steps) if frame_steps else step,
        },
    }


def main(arguments: list[str] | None = None) -> int:
    """Audit one or every native checkpoint below a solution directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="vpm_*.h5 checkpoint or solution directory")
    args = parser.parse_args(arguments)
    if not args.path.exists():
        parser.error(f"path does not exist: {args.path}")
    checkpoints = sorted(args.path.rglob("vpm_*.h5")) if args.path.is_dir() else [args.path]
    for checkpoint in checkpoints:
        evidence = audit_checkpoint(checkpoint)
        series = cast(dict[str, object], evidence["vtp_series"])
        from .logging import Logging

        Logging.section(
            "checkpoint audit",
            ("checkpoint", checkpoint.name),
            ("accepted step", evidence["step"]),
            ("physical time", evidence["time"], "s"),
            ("frames", series["frame_count"]),
            ("last frame", series["last_frame"]),
            ("VLM identity", evidence["vlm_identity"]),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
