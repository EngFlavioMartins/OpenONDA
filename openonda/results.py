"""Lossless, portable tutorial result bundles.

Bundles retain original file bytes; scientific status records the limitations of
those results and does not imply completion or numerical qualification.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from contextlib import suppress
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
import numpy as np

_ARCHIVE_PART_BYTES = 1024**3
_VTK_COLLECTION_SUFFIXES = {".pvd", ".pvtu", ".pvtp", ".pvts", ".pvtr", ".pvti", ".vtm"}


def _validate_vtk_references(relative: str, payload: bytes, files: set[str]) -> None:
    """Require every VTK collection dependency to travel in the same bundle."""
    try:
        collection = ElementTree.fromstring(payload)
    except (ElementTree.ParseError, DefusedXmlException) as error:
        raise ResultsError(f"Invalid VTK collection: {relative}: {error}") from error
    for element in collection.iter():
        for attribute in ("file", "Source"):
            if attribute not in element.attrib:
                continue
            reference = element.attrib[attribute]
            # Windows drive paths and URI references are not portable local files.
            if ":" in reference:
                raise ResultsError(f"Unsafe VTK reference in {relative}: {reference!r}")
            reference = _relative(reference)
            target = str(PurePosixPath(relative).parent / reference)
            if target not in files:
                raise ResultsError(f"Missing VTK reference in {relative}: {reference!r}")


class _HashingReader:
    def __init__(self, stream, digest, capture=False):
        self.stream = stream
        self.digest = digest
        self.blocks = [] if capture else None

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        if self.blocks is not None:
            self.blocks.append(data)
        return data


class ResultsError(ValueError):
    """A result bundle cannot safely be packed or restored."""


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(p in ("", ".", "..") for p in value.split("/"))
        or "\\" in value
    ):
        raise ResultsError(f"Unsafe case-relative path: {value!r}")
    return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_symlinks(root: Path, relative: str) -> Path:
    result = root
    for part in PurePosixPath(relative).parts:
        result = result / part
        if result.is_symlink():
            raise ResultsError(f"Symlink is not a portable result input: {result}")
    return result


def _restore_roots(files: Iterable[str], explicit: Iterable[str] | None) -> list[str]:
    files = list(files)
    if explicit is None:
        roots = set()
        for relative in files:
            parts = PurePosixPath(relative).parts
            index = next(
                (
                    i
                    for i, part in enumerate(parts[:-1])
                    if part in {"samples", "solution", "study_results"}
                ),
                0,
            )
            roots.add("/".join(parts[: index + 1]))
    else:
        roots = {_relative(root) for root in explicit}
    roots = sorted(roots)
    if not roots or any(a != b and b.startswith(a + "/") for a in roots for b in roots):
        raise ResultsError("Restore roots must be nonempty and must not overlap")
    if any(sum(path.startswith(root + "/") for root in roots) != 1 for path in files):
        raise ResultsError("Every result file must belong to exactly one restore root")
    if any(not any(path.startswith(root + "/") for path in files) for root in roots):
        raise ResultsError("Every restore root must contain a result file")
    return roots


def pack_results(
    source_root: Path,
    bundle_dir: Path,
    files: Iterable[str],
    *,
    scientific_status: str,
    provenance: dict,
    superseded: Iterable[str] = (),
    restore_roots: Iterable[str] | None = None,
) -> dict:
    """Pack an explicit set of case-relative regular files deterministically.

    ``superseded`` is an explicit exclusion list, supplied only after checking
    accepted continuation lineage. No inference from filenames is made here.
    """
    source_root = Path(source_root).absolute()
    bundle_dir = Path(bundle_dir)
    excluded = {_relative(p) for p in superseded}
    selected = sorted({_relative(p) for p in files} - excluded)
    if not selected or not scientific_status.strip() or not isinstance(provenance, dict):
        raise ResultsError("Files, scientific status and provenance are required")
    if any(len(PurePosixPath(p).parts) < 2 for p in selected):
        raise ResultsError("Result files must be inside case-relative directories")
    roots = _restore_roots(selected, restore_roots)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix=".results-pack-", dir=bundle_dir) as temporary:
        archive = Path(temporary) / "data.tar.gz"
        with (
            archive.open("wb") as raw,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as tar,
        ):
            for relative in selected:
                source = _no_symlinks(source_root, relative)
                if not source.is_file():
                    raise ResultsError(f"Missing regular result file: {source}")
                info = tarfile.TarInfo(relative)
                before = source.stat()
                info.size = before.st_size
                info.mode = 0o644
                digest = hashlib.sha256()
                with source.open("rb") as stream:
                    reader = _HashingReader(
                        stream,
                        digest,
                        capture=PurePosixPath(relative).suffix.lower() in _VTK_COLLECTION_SUFFIXES,
                    )
                    tar.addfile(info, reader)
                    if reader.blocks is not None:
                        _validate_vtk_references(relative, b"".join(reader.blocks), set(selected))
                after = source.stat()
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise ResultsError(f"Input changed while packing: {source}")
                records.append({"path": relative, "size": info.size, "sha256": digest.hexdigest()})
        manifest = {
            "schema_version": 1,
            "archive": "data.tar.gz",
            "archive_sha256": _sha256(archive),
            "scientific_status": scientific_status,
            "provenance": provenance,
            "excluded_superseded": sorted(excluded),
            "files": records,
            "restore_roots": roots,
        }
        outputs = [archive]
        if archive.stat().st_size > _ARCHIVE_PART_BYTES:
            outputs = []
            parts = []
            with archive.open("rb") as stream:
                index = 0
                while stream.tell() < archive.stat().st_size:
                    part = Path(temporary) / f"data.tar.gz.part{index:03d}"
                    size = 0
                    digest = hashlib.sha256()
                    with part.open("wb") as destination:
                        while size < _ARCHIVE_PART_BYTES:
                            block = stream.read(min(1024 * 1024, _ARCHIVE_PART_BYTES - size))
                            if not block:
                                break
                            destination.write(block)
                            digest.update(block)
                            size += len(block)
                    outputs.append(part)
                    parts.append({"name": part.name, "size": size, "sha256": digest.hexdigest()})
                    index += 1
            manifest["archive_parts"] = parts
        metadata = Path(temporary) / "manifest.json"
        metadata.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for output in outputs:
            os.replace(output, bundle_dir / output.name)
        os.replace(metadata, bundle_dir / metadata.name)
        current = {output.name for output in outputs}
        for previous in bundle_dir.glob("data.tar.gz*"):
            owned = previous.name == "data.tar.gz" or (
                previous.name.startswith("data.tar.gz.part")
                and previous.name.removeprefix("data.tar.gz.part").isdigit()
            )
            if owned and previous.name not in current:
                previous.unlink()
    return manifest


def _verify_archive_file(path: Path, sha256: str, size: int | None = None) -> None:
    if not path.is_file() or path.is_symlink():
        raise ResultsError(f"Missing regular result archive: {path}")
    with path.open("rb") as stream:
        pointer = stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1\n")
    if pointer:
        raise ResultsError(
            f"Result archive is a Git LFS pointer: {path}. Run git lfs pull from the checkout, then retry."
        )
    if size is not None and (not isinstance(size, int) or size <= 0 or path.stat().st_size != size):
        raise ResultsError(f"Result archive size mismatch: {path}")
    if _sha256(path) != sha256:
        raise ResultsError(f"Result archive SHA-256 mismatch: {path}")


def _ensure_archive_file(
    path: Path, sha256: str, size: int | None = None, url: str | None = None
) -> None:
    """Fetch an absent published archive, then verify it before caching it."""
    pointer = False
    if path.is_file() and not path.is_symlink():
        with path.open("rb") as stream:
            pointer = stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1\n")
    if url is None or (os.path.lexists(path) and not pointer):
        _verify_archive_file(path, sha256, size)
        return
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ResultsError(f"Result archive requires an HTTPS download URL: {url!r}")
    if not isinstance(size, int) or size <= 0:
        raise ResultsError(f"Published result archive requires a positive size: {path}")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(c not in "0123456789abcdef" for c in sha256)
    ):
        raise ResultsError(f"Published result archive requires a SHA-256 checksum: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tutorial results: {path.name} ({size:,} bytes)", flush=True)
    with tempfile.TemporaryDirectory(prefix=".results-download-", dir=path.parent) as temporary:
        download = Path(temporary) / path.name
        digest = hashlib.sha256()
        received = 0
        request = Request(url, headers={"User-Agent": "OpenONDA-results"})
        with urlopen(request, timeout=60) as response, download.open("wb") as destination:
            if urlparse(response.geturl()).scheme != "https":
                raise ResultsError("Result archive download redirected away from HTTPS")
            for block in iter(lambda: response.read(1024 * 1024), b""):
                received += len(block)
                if received > size:
                    raise ResultsError(f"Result archive size mismatch: {path}")
                destination.write(block)
                digest.update(block)
        if received != size or digest.hexdigest() != sha256:
            raise ResultsError(f"Result archive download checksum or size mismatch: {path}")
        # Another restore may have populated the cache during the download.
        if path.exists() and not pointer:
            _verify_archive_file(path, sha256, size)
        else:
            os.replace(download, path)


def find_bundle(case_dir: Path) -> Path:
    """Find the bundle colocated with a copied or cloned tutorial."""
    return case_dir.absolute() / "assets" / "results"


def restore_results(case_dir: Path, bundle_dir: Path | None = None) -> list[str]:
    """Verify a coherent bundle and restore it only when every root is absent.

    Any existing directory, file or symlink at a result root is preserved. A
    corrupt bundle is rejected before any directory is installed. Publication
    is atomic per directory; the group is reserved before publication.
    """
    case_dir = Path(case_dir).absolute()
    bundle_dir = Path(bundle_dir) if bundle_dir is not None else find_bundle(case_dir)
    metadata = bundle_dir / "manifest.json"
    if not metadata.is_file():
        return []
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("archive") != "data.tar.gz":
        raise ResultsError("Unsupported result manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ResultsError("Result manifest has no files")
    expected = {}
    for record in entries:
        relative = _relative(record["path"])
        if len(PurePosixPath(relative).parts) < 2 or relative in expected:
            raise ResultsError(f"Invalid or duplicate result path: {relative}")
        if not isinstance(record.get("size"), int) or record["size"] < 0:
            raise ResultsError(f"Invalid result size: {relative}")
        expected[relative] = record
    roots = _restore_roots(expected, manifest.get("restore_roots"))
    if any(os.path.lexists(case_dir / root) for root in roots):
        return []
    for root in roots:
        _no_symlinks(case_dir, root)
    archive = bundle_dir / manifest["archive"]
    parts = manifest.get("archive_parts")
    if parts is None:
        _ensure_archive_file(
            archive,
            manifest.get("archive_sha256"),
            manifest.get("archive_size"),
            manifest.get("archive_url"),
        )
    else:
        if not isinstance(parts, list) or not parts:
            raise ResultsError("Archive parts must be a nonempty list")
        for index, record in enumerate(parts):
            if record.get("name") != f"data.tar.gz.part{index:03d}":
                raise ResultsError("Archive parts must have consecutive canonical names")
            _ensure_archive_file(
                bundle_dir / record["name"],
                record.get("sha256"),
                record.get("size"),
                record.get("url"),
            )
    case_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".results-restore-", dir=case_dir) as temporary:
        stage = Path(temporary) / "contents"
        stage.mkdir()
        if parts is not None:
            archive = Path(temporary) / "data.tar.gz"
            with archive.open("wb") as destination:
                for record in parts:
                    with (bundle_dir / record["name"]).open("rb") as source:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
            if _sha256(archive) != manifest.get("archive_sha256"):
                raise ResultsError("Combined result archive SHA-256 mismatch")
        seen = set()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                relative = _relative(member.name)
                if not member.isfile() or relative not in expected or relative in seen:
                    raise ResultsError(f"Unexpected or unsafe archive member: {member.name}")
                record = expected[relative]
                if member.size != record["size"]:
                    raise ResultsError(f"Result size mismatch: {relative}")
                seen.add(relative)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with tar.extractfile(member) as source, target.open("wb") as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                        digest.update(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ResultsError(f"Result SHA-256 mismatch: {relative}")
        if seen != set(expected):
            raise ResultsError("Result archive is missing manifest files")
        for relative in expected:
            if PurePosixPath(relative).suffix.lower() in _VTK_COLLECTION_SUFFIXES:
                _validate_vtk_references(relative, (stage / relative).read_bytes(), seen)
        reserved = []
        restored = []
        try:
            for root in roots:
                target = _no_symlinks(case_dir, root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.mkdir()
                reserved.append(root)
            for root in roots:
                target = _no_symlinks(case_dir, root)
                os.replace(stage / root, target)
                restored.append(root)
        except BaseException:
            # Remove only the files this restore installed, and only if their
            # archived bytes remain intact. Concurrent new files are preserved.
            for root in reversed(restored):
                target = case_dir / root
                if target.is_symlink():
                    continue
                for relative, record in expected.items():
                    if relative.startswith(root + "/"):
                        try:
                            path = _no_symlinks(case_dir, relative)
                        except ResultsError:
                            continue
                        if (
                            path.is_file()
                            and not path.is_symlink()
                            and _sha256(path) == record["sha256"]
                        ):
                            path.unlink()
                for directory in sorted(
                    target.rglob("*"), key=lambda p: len(p.parts), reverse=True
                ):
                    if directory.is_dir() and not directory.is_symlink():
                        with suppress(OSError):
                            directory.rmdir()
            for root in reversed(reserved):
                with suppress(OSError):
                    (case_dir / root).rmdir()
            raise
        return restored


def read_json(path: str | Path) -> dict:
    """Read one recorded JSON object."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def read_json_lines(path: str | Path) -> list[dict]:
    """Read diagnostic records, allowing an unfinished final live-write line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise
    return records


def read_csv_columns(path: str | Path) -> dict[str, np.ndarray]:
    """Read a numeric CSV table without replacing invalid measured values."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Numeric CSV has no records: {path}")
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def read_numeric_table(path: str | Path, *, delimiter: str = ",") -> np.ndarray:
    """Read a finite headerless numeric table."""
    values = np.loadtxt(path, delimiter=delimiter)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"Numeric table must contain finite measured values: {path}")
    return values


def read_grouped_csv(path: str | Path, group_columns) -> dict:
    """Read numeric history columns grouped by their recorded entity."""
    columns = (group_columns,) if isinstance(group_columns, str) else tuple(group_columns)
    groups = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values = tuple(row[name] for name in columns)
            key = values[0] if len(values) == 1 else values
            group = groups.setdefault(key, {})
            for name, value in row.items():
                if name not in columns:
                    group.setdefault(name, []).append(float(value))
    if not groups:
        raise ValueError(f"Grouped CSV has no records: {path}")
    return {
        key: {name: np.asarray(values) for name, values in group.items()}
        for key, group in groups.items()
    }


def read_csv_table(path: str | Path) -> dict[str, np.ndarray]:
    """Read native named CSV fields, including a recorded single-frame clock."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        time_comment = re.fullmatch(r"#\s*time\s*=\s*([^\s]+)\s*", stream.readline())
    rows = np.atleast_1d(
        np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
            skip_header=1 if time_comment else 0,
        )
    )
    if rows.dtype.names is None or not rows.size:
        raise ValueError(f"Expected a nonempty named CSV table: {path}")
    table = {name: np.asarray(rows[name]) for name in rows.dtype.names}
    if "step" in table:
        steps = table["step"]
        if (
            not (np.issubdtype(steps.dtype, np.integer) or np.issubdtype(steps.dtype, np.floating))
            or not np.isfinite(steps).all()
            or np.any(steps < 0)
            or np.any(steps != np.floor(steps))
            or np.any(steps[1:] < steps[:-1])
        ):
            raise ValueError(
                f"CSV steps must be finite nonnegative integers in increasing order: {path}"
            )
    if "time" not in table and time_comment is not None:
        table["time"] = np.full(rows.size, float(time_comment.group(1)))
    return table


def read_history_table(path: str | Path) -> dict[str, np.ndarray]:
    """Read a single-entity accepted history with one row per physical time."""
    table = read_csv_table(path)
    times = table["time"]
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f"History times are duplicate or nonmonotonic: {path}")
    if "step" in table and np.any(table["step"][1:] <= table["step"][:-1]):
        raise ValueError(f"History steps are duplicate or nonmonotonic: {path}")
    return table


def history_window(table: dict, start: float, end: float, *, columns: tuple[str, ...]) -> dict:
    """Interpolate a fully covered physical interval from an accepted history."""
    time = np.asarray(table["time"], dtype=float)
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("History window requires finite increasing endpoints")
    if time.ndim != 1 or len(time) < 2 or np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("History window requires finite increasing recorded times")
    if time[0] > start or time[-1] < end:
        raise ValueError(f"History does not cover the requested window [{start}, {end}]")
    interior = (time > start) & (time < end)
    result = {"time": np.r_[start, time[interior], end]}
    for name in columns:
        values = np.asarray(table[name], dtype=float)
        if values.shape != time.shape or not np.isfinite(values).all():
            raise ValueError(f"History column {name!r} must match finite recorded times")
        result[name] = np.interp(result["time"], time, values)
    return result


def read_csv_frame(path: str | Path, time: float, *, coordinates: tuple[str, ...]) -> dict:
    """Select one recorded clock and admit each finite sample coordinate once."""
    from .saved_times import match_saved_times

    table = read_csv_table(path)
    times = np.unique(table["time"])
    match = match_saved_times([time], times)
    if not match.times:
        raise ValueError(f"No recorded CSV frame at t={time:g}: {path}")
    picked = times[match.indices[1][0]]
    frame = {name: values[table["time"] == picked] for name, values in table.items()}
    positions = np.column_stack([frame[name] for name in coordinates])
    if np.any(~np.isfinite(positions)) or len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError(f"Frame has non-finite or duplicate sample coordinates: {path}")
    order = np.lexsort(positions[:, ::-1].T)
    return {name: values[order] for name, values in frame.items()}


def read_pvd_frames(path: str | Path) -> list[tuple[float, Path]]:
    """Read a current VTK collection's saved physical times and frame paths."""
    from .saved_times import match_saved_times

    path = Path(path)
    frames = sorted(
        (float(item.attrib["timestep"]), path.parent / item.attrib["file"])
        for item in ElementTree.parse(path).iter("DataSet")
    )
    match_saved_times([time for time, _ in frames])
    return frames


def read_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Copy array data while owning the archive lifetime."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
    """Publish a complete text artifact while owning its file lifetime."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".results-write-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(content, encoding=encoding)
        os.replace(staged, path)


def write_json(path: str | Path, data: dict) -> None:
    """Publish complete recorded metadata atomically."""
    write_text(path, json.dumps(data, indent=2) + "\n")


def write_csv_table(path: str | Path, rows, *, columns) -> None:
    """Publish one complete named table while owning file lifetime."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".results-write-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        with staged.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)
        os.replace(staged, path)


def write_npz_arrays(path: str | Path, **arrays) -> None:
    """Publish an array archive while owning its temporary-file lifetime."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".results-write-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        np.savez_compressed(staged, **arrays)
        os.replace(staged, path)


def snapshot_vector_field(mesh, name: str):
    """Return a native field and the coordinates at its recorded association."""
    if name in mesh.cell_data:
        return mesh.cell_data[name], mesh.cell_centers().points
    if name in mesh.point_data:
        return mesh.point_data[name], mesh.points
    raise ValueError(f"Snapshot has no {name!r} field in cell or point data")


def snapshot_cell_field(mesh, name: str) -> np.ndarray:
    """Read one field at native cells, converting recorded point association."""
    if name in mesh.cell_data:
        return mesh.cell_data[name]
    if name in mesh.point_data:
        return mesh.point_data_to_cell_data().cell_data[name]
    raise ValueError(f"Snapshot has no {name!r} field in cell or point data")


def latest_fvm_frame(solution_directory: str | Path) -> Path:
    """Locate a saved native FVM frame or report the missing input."""
    from .plotting import latest_fvm_snapshot

    path = latest_fvm_snapshot(solution_directory)
    if path is None:
        raise FileNotFoundError(f"No saved FVM frames in {solution_directory}")
    return path


def matched_surface_vectors(source: dict, target: dict) -> np.ndarray:
    """Require the same recorded grid before comparing vector samples."""
    for name in ("x", "y"):
        left, right = source[name], target[name]
        tolerance = 8 * np.finfo(np.float32).eps * max(1, float(np.max(np.abs(right))))
        if left.shape != right.shape or not np.allclose(left, right, rtol=0, atol=tolerance):
            raise ValueError("Slice coordinates differ; sample fields on one common grid")
    values = np.array(source["velocity"], dtype=float, copy=True)
    values[~source["valid"]] = np.nan
    return values


def section_polygons(mesh, *, normal, origin, field: str):
    """Cut native cells and return each polygon with its unaveraged cell field."""
    if field not in mesh.cell_data:
        if field not in mesh.point_data:
            raise ValueError(f"Snapshot has no {field} field")
        mesh = mesh.point_data_to_cell_data()
    section = mesh.slice(normal=normal, origin=origin)
    values = section.cell_data[field]
    polygons = []
    cursor = 0
    while cursor < len(section.faces):
        count = int(section.faces[cursor])
        if count < 3:
            raise ValueError("Section contains a non-polygon cell")
        ids = section.faces[cursor + 1 : cursor + 1 + count]
        polygons.append(section.points[ids, :2])
        cursor += count + 1
    if len(polygons) != len(values):
        raise ValueError("Section polygon and field counts differ")
    return polygons, values


def planar_direction(velocity, *, axes: tuple[int, int]) -> np.ndarray:
    """Resolve a finite nonzero recorded direction in the selected plane."""
    vector = np.asarray(velocity, dtype=float)
    if vector.shape != (3,) or np.any(~np.isfinite(vector)):
        raise ValueError("Recorded velocity must be a finite three-component vector")
    normal = ({0, 1, 2} - set(axes)).pop()
    plane = vector[list(axes)]
    speed = np.linalg.norm(plane)
    if speed <= 0 or not np.isclose(vector[normal], 0):
        raise ValueError("Recorded velocity must lie in a nonzero selected plane")
    return plane / speed


def hexahedron_footprints(mesh):
    """Return the ordered lower-face x-y footprints of native hexahedra."""
    import pyvista as pv

    if mesh.n_cells == 0 or not np.all(mesh.celltypes == pv.CellType.HEXAHEDRON):
        raise ValueError("Snapshot must contain hexahedral cells")
    points = mesh.points[mesh.cells.reshape(-1, 9)[:, 1:]]
    lower = np.argsort(points[:, :, 2], axis=1)[:, :4]
    face = points[np.arange(mesh.n_cells)[:, None], lower, :2]
    centre = face.mean(axis=1, keepdims=True)
    angles = np.arctan2(face[:, :, 1] - centre[:, :, 1], face[:, :, 0] - centre[:, :, 0])
    return face[np.arange(mesh.n_cells)[:, None], np.argsort(angles, axis=1)]


def load_vpm_particles(path: str | Path) -> dict[str, np.ndarray]:
    """Load active particle arrays from a native VPM backup."""
    import h5py

    with h5py.File(path, "r") as handle:
        count = int(handle["solver"].attrs["n_particles_total"])
        particles = handle["particles"]
        return {
            name: np.asarray(particles[name][:count])
            for name in ("position", "vortex_strength", "core_radius")
        }


def read_surface_frame(path: str | Path) -> dict:
    """Read a structured native sampler frame with its coordinate ordering."""
    import pyvista as pv

    grid = pv.read(path)
    if not isinstance(grid, pv.StructuredGrid):
        raise ValueError("Surface frame must be a native structured grid")
    ni, nj, nk = grid.dimensions
    if ni < 2 or nj < 2 or nk != 1 or grid.n_points != ni * nj:
        raise ValueError("Surface frame must contain a two-dimensional sample grid")
    shape = (nj, ni)
    points = np.asarray(grid.points, dtype=float)
    if not np.all(np.isfinite(points)) or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Surface sample coordinates must be finite and distinct")
    if "vtkValidPointMask" in grid.cell_data and "vtkValidPointMask" not in grid.point_data:
        raise ValueError("Surface validity mask must be associated with sample points")
    valid = np.asarray(grid.point_data.get("vtkValidPointMask", np.ones(grid.n_points)))
    if valid.shape != (grid.n_points,) or not np.all(np.isin(valid, (0, 1))):
        raise ValueError("Surface validity mask must contain one binary value per point")
    valid = valid.astype(bool)

    def point_field(name, components):
        if name not in grid.point_data:
            raise ValueError(f"Surface field {name} must be associated with sample points")
        values = np.asarray(grid.point_data[name], dtype=float)
        expected = (grid.n_points, components) if components > 1 else (grid.n_points,)
        if values.shape != expected or not np.all(np.isfinite(values[valid])):
            raise ValueError(f"Surface field {name} must contain finite values on valid points")
        if name.endswith("_standard_error") and np.any(values[valid] < 0):
            raise ValueError(f"Surface field {name} must contain nonnegative standard errors")
        return values.reshape(*shape, components) if components > 1 else values.reshape(shape)

    velocity = point_field("velocity", 3)
    data = {
        "x": points[:, 0].reshape(shape),
        "y": points[:, 1].reshape(shape),
        "z": points[:, 2].reshape(shape),
        "velocity": velocity,
        "valid": valid.reshape(shape),
        **{f"velocity_{axis}": velocity[..., index] for index, axis in enumerate("xyz")},
    }
    for name in ("vorticity", "velocity_standard_error", "vorticity_standard_error"):
        if name in grid.point_data or name in grid.cell_data:
            vector = point_field(name, 3)
            data[name] = vector
            data.update({f"{name}_{axis}": vector[..., index] for index, axis in enumerate("xyz")})
    for name in ("velocity_gradient_yx", "velocity_gradient_yx_standard_error"):
        if name in grid.point_data or name in grid.cell_data:
            data[name] = point_field(name, 1)
    statistical = [name for name in data if name.endswith("_standard_error")]
    for name in statistical:
        if name.removesuffix("_standard_error") not in data:
            raise ValueError(f"Surface uncertainty {name} requires its measured field")
    metadata = ("ensemble_size", "confidence_multiplier")
    if statistical or any(name in grid.field_data for name in metadata):
        for name in metadata:
            if name not in grid.field_data or np.asarray(grid.field_data[name]).shape != (1,):
                raise ValueError("Surface ensemble metadata must record size and confidence")
            value = np.asarray(grid.field_data[name])[0]
            if not isinstance(value, (np.integer, np.floating)) or not np.isfinite(value):
                raise ValueError("Surface ensemble metadata must contain finite numeric values")
            data[name] = value.item()
        size, confidence = data["ensemble_size"], data["confidence_multiplier"]
        if size < 2 or size != int(size) or confidence <= 0:
            raise ValueError("Surface ensemble size and confidence must be valid positive values")
    return data


class NativeVelocity:
    """Sample archived velocity using its native centroids and affine probes."""

    def __init__(self, mesh_path: str | Path, *, k: int):
        from source.solvers.fvm.io.mesh_storage import load_native_mesh
        from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

        mesh = load_native_mesh(mesh_path)
        self.centres = compute_mesh_geometry(mesh, compute_lsq=False)["cell_centre"]
        self.k = k

    def sample(self, source: str | Path, queries: dict[str, np.ndarray], *, mask=None):
        import pyvista as pv

        from source.solvers.fvm.sampling.fields import _PointProbe

        grid = pv.read(source)
        data = grid.cell_data
        velocity = np.asarray(data["velocity"], dtype=float)
        keep = np.asarray(data.get("vtkGhostType", np.zeros(len(velocity)))) == 0
        ids = np.asarray(data.get("global_cell_id", np.arange(len(velocity))), dtype=int)[keep]
        if len(ids) != len(self.centres) or not np.array_equal(
            np.sort(ids), np.arange(len(self.centres))
        ):
            raise ValueError(f"Snapshot does not cover its native mesh exactly once: {source}")
        ordered = np.empty_like(velocity[keep])
        ordered[ids] = velocity[keep]
        if not np.all(np.isfinite(ordered)):
            raise ValueError(f"Non-finite archived velocity: {source}")
        owned_grid = grid.extract_cells(keep)
        result = {}
        for name, points in queries.items():
            inside = owned_grid.find_containing_cell(points) >= 0
            if mask is not None:
                inside &= mask(points)
            probe = _PointProbe(points, k=self.k, reconstruction="affine")
            values = probe._interpolate(ordered, self.centres)
            values[~inside] = np.nan
            result[name] = values
        return result


def read_velocity_profile_frames(
    path: str | Path,
    *,
    query_path: str | Path,
    collection: str | Path,
    mesh: str | Path,
    k: int,
    bounds: dict,
):
    """Read sampled velocity profiles or reconstruct their coincident native fields."""
    from .saved_times import match_saved_times

    path = Path(path)
    coordinates = tuple(f"position_{axis}" for axis in "xyz")
    if path.is_file():
        for time in np.unique(read_csv_table(path)["time"]):
            yield float(time), read_csv_frame(path, time, coordinates=coordinates), None
        return
    query_times = np.unique(read_csv_table(query_path)["time"])
    fields = read_pvd_frames(collection)
    matches = match_saved_times(query_times, [time for time, _ in fields])
    native = NativeVelocity(mesh, k=k)
    for query_index, field_index in zip(*matches.indices, strict=True):
        time = query_times[query_index]
        native_time, field = fields[field_index]
        queries = read_csv_frame(query_path, time, coordinates=coordinates)
        positions = np.column_stack([queries[name] for name in coordinates])
        keep = np.ones(len(positions), dtype=bool)
        for axis, name in enumerate("xyz"):
            keep &= positions[:, axis] >= bounds[f"{name}min"] - 1e-12
            keep &= positions[:, axis] <= bounds[f"{name}max"] + 1e-12
        positions = positions[keep]
        velocity = native.sample(field, {"profile": positions})["profile"]
        frame = {name: positions[:, index] for index, name in enumerate(coordinates)}
        frame["time"] = np.full(len(positions), native_time)
        frame.update({f"velocity_{axis}": velocity[:, index] for index, axis in enumerate("xyz")})
        yield (
            float(time),
            frame,
            {
                "method": f"native-centres-affine-k{k}",
                "native_time": native_time,
                "native_field": field,
                "point_count": len(positions),
            },
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    restore = subcommands.add_parser("restore", help="Restore absent tutorial result directories")
    restore.add_argument("case_dir", type=Path, nargs="?", default=Path.cwd())
    restore.add_argument(
        "--bundle", type=Path, help="Bundle directory (defaults to CASE_DIR/assets/results)"
    )
    args = parser.parse_args(argv)
    try:
        restored = restore_results(args.case_dir, args.bundle)
    except (
        ResultsError,
        OSError,
        tarfile.TarError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        parser.exit(1, f"Tutorial results: {error}\n")
    if restored:
        print("Restored tutorial results: " + ", ".join(restored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
