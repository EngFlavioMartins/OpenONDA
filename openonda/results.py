"""Lossless, portable tutorial result bundles.

Bundles retain original file bytes; scientific status records the limitations of
those results and does not imply completion or numerical qualification.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from contextlib import suppress
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

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
