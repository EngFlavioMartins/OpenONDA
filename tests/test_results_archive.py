"""Portable tutorial bundles preserve bytes and existing simulation results."""

import hashlib
import io
import json
import tarfile

import pytest

from openonda.results import ResultsError, pack_results, restore_results


def test_repacking_removes_only_obsolete_archive_parts(tmp_path, monkeypatch):
    import openonda.results as results

    source, output = bundle(tmp_path)
    unrelated = output / "data.tar.gz.notes"
    unrelated.write_text("keep")
    arguments = {"scientific_status": "partial", "provenance": {"revision": "test"}}
    files = ["samples/history.csv", "solution/state.bin"]
    monkeypatch.setattr(results, "_ARCHIVE_PART_BYTES", 100)
    pack_results(source, output, files, **arguments)
    assert not (output / "data.tar.gz").exists()
    assert list(output.glob("data.tar.gz.part*"))
    monkeypatch.setattr(results, "_ARCHIVE_PART_BYTES", 10000)
    pack_results(source, output, files, **arguments)
    assert (output / "data.tar.gz").is_file()
    assert not list(output.glob("data.tar.gz.part*"))
    assert unrelated.read_text() == "keep"


def bundle(tmp_path):
    source = tmp_path / "source"
    (source / "samples").mkdir(parents=True)
    (source / "solution").mkdir()
    (source / "samples" / "history.csv").write_bytes(b"time,value\r\n0,1\r\n")
    (source / "solution" / "state.bin").write_bytes(bytes(range(256)))
    output = tmp_path / "bundle"
    pack_results(
        source,
        output,
        ["samples/history.csv", "solution/state.bin"],
        scientific_status="partial",
        provenance={"revision": "test"},
    )
    return source, output


def test_roundtrip_and_existing_results(tmp_path):
    source, output = bundle(tmp_path)
    target = tmp_path / "relocated"
    (target / "solution").mkdir(parents=True)
    (target / "solution" / "local").write_bytes(b"local solver output")
    assert restore_results(target, output) == []
    assert not (target / "samples").exists()
    assert list((target / "solution").iterdir()) == [target / "solution/local"]
    assert restore_results(target, output) == []


def test_cli_restores_explicit_bundle_without_touching_source(tmp_path):
    from openonda.results import main

    source, output = bundle(tmp_path)
    target = tmp_path / "relocated"
    assert main(["restore", str(target), "--bundle", str(output)]) == 0
    for relative in ("samples/history.csv", "solution/state.bin"):
        assert (target / relative).read_bytes() == (source / relative).read_bytes()
    (target / "samples/history.csv").write_bytes(b"new local run")
    assert main(["restore", str(target), "--bundle", str(output)]) == 0
    assert (target / "samples/history.csv").read_bytes() == b"new local run"


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("pointer", [False, True])
def test_release_download_restore_and_offline_cache(tmp_path, monkeypatch, sharded, pointer):
    from openonda import results

    if sharded:
        monkeypatch.setattr(results, "_ARCHIVE_PART_BYTES", 100)
    source, output = bundle(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    records = manifest.get("archive_parts") or [{"name": "data.tar.gz"}]
    payloads = {}
    for record in records:
        path = output / record["name"]
        url = "https://example.org/" + path.name
        payloads[url] = path.read_bytes()
        if sharded:
            record["url"] = url
        else:
            manifest.update(archive_url=url, archive_size=path.stat().st_size)
        if pointer:
            path.write_text("version https://git-lfs.github.com/spec/v1\n")
        else:
            path.unlink()
    (output / "manifest.json").write_text(json.dumps(manifest))
    calls = []

    def fetch(request, timeout):
        calls.append(request.full_url)
        stream = io.BytesIO(payloads[request.full_url])
        stream.geturl = lambda: request.full_url
        return stream

    monkeypatch.setattr(results, "urlopen", fetch)
    target = tmp_path / "downloaded"
    assert restore_results(target, output) == manifest["restore_roots"]
    assert len(calls) == len(payloads)
    for relative in ("samples/history.csv", "solution/state.bin"):
        assert (target / relative).read_bytes() == (source / relative).read_bytes()
    calls.clear()
    assert restore_results(tmp_path / "offline", output) == manifest["restore_roots"]
    assert not calls


@pytest.mark.parametrize("damage", ["oversize", "truncated", "checksum", "disconnect", "redirect"])
def test_failed_download_leaves_no_cache_or_results(tmp_path, monkeypatch, damage):
    from openonda import results

    _, output = bundle(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    archive = output / "data.tar.gz"
    payload = archive.read_bytes()
    manifest.update(archive_url="https://example.org/data.tar.gz", archive_size=len(payload))
    (output / "manifest.json").write_text(json.dumps(manifest))
    archive.unlink()
    if damage == "oversize":
        payload += b"extra"
    elif damage == "truncated":
        payload = payload[:-1]
    elif damage == "checksum":
        payload = b"X" * len(payload)

    def fetch(request, timeout):
        if damage == "disconnect":
            raise OSError("connection interrupted")
        stream = io.BytesIO(payload)
        stream.geturl = lambda: (
            "http://example.org/file" if damage == "redirect" else request.full_url
        )
        return stream

    monkeypatch.setattr(results, "urlopen", fetch)
    target = tmp_path / "downloaded"
    with pytest.raises((ResultsError, OSError)):
        restore_results(target, output)
    assert not archive.exists()
    assert not target.exists()
    assert not list(output.glob(".results-download-*"))


def test_download_preserves_existing_local_results_and_corrupt_cache(tmp_path, monkeypatch):
    from openonda import results

    _, output = bundle(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    archive = output / "data.tar.gz"
    manifest.update(
        archive_url="https://example.org/data.tar.gz", archive_size=archive.stat().st_size
    )
    (output / "manifest.json").write_text(json.dumps(manifest))
    archive.write_bytes(b"corrupt local archive")
    monkeypatch.setattr(results, "urlopen", lambda *a, **kw: pytest.fail("Unexpected download"))
    target = tmp_path / "local"
    (target / "samples").mkdir(parents=True)
    assert restore_results(target, output) == []
    with pytest.raises(ResultsError, match="archive"):
        restore_results(tmp_path / "empty", output)
    assert archive.read_bytes() == b"corrupt local archive"


def test_deterministic_and_explicit_superseded(tmp_path):
    source, output = bundle(tmp_path)
    second = tmp_path / "second"
    pack_results(
        source,
        second,
        ["solution/state.bin", "samples/history.csv"],
        scientific_status="partial",
        provenance={"revision": "test"},
    )
    assert (output / "data.tar.gz").read_bytes() == (second / "data.tar.gz").read_bytes()
    pack_results(
        source,
        second,
        ["solution/state.bin", "samples/history.csv"],
        scientific_status="partial",
        provenance={},
        superseded=["solution/state.bin"],
    )
    assert [r["path"] for r in json.loads((second / "manifest.json").read_text())["files"]] == [
        "samples/history.csv"
    ]
    assert (source / "solution/state.bin").exists()


@pytest.mark.parametrize(
    "relative", ["../outside", "/outside", "samples/../outside", "samples\\outside"]
)
def test_pack_rejects_unsafe_path(tmp_path, relative):
    with pytest.raises(ResultsError, match="Unsafe"):
        pack_results(
            tmp_path, tmp_path / "bundle", [relative], scientific_status="partial", provenance={}
        )


def test_symlink_input_and_destination(tmp_path):
    source, output = bundle(tmp_path)
    (source / "samples/link").symlink_to(source / "solution/state.bin")
    with pytest.raises(ResultsError, match="Symlink"):
        pack_results(
            source, tmp_path / "bad", ["samples/link"], scientific_status="partial", provenance={}
        )
    target = tmp_path / "target"
    target.mkdir()
    (target / "samples").symlink_to(source / "samples")
    assert restore_results(target, output) == []
    assert (target / "samples").is_symlink()


def test_corruption_installs_nothing(tmp_path):
    _, output = bundle(tmp_path)
    archive = output / "data.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"corruption")
    target = tmp_path / "target"
    with pytest.raises(ResultsError, match="SHA-256"):
        restore_results(target, output)
    assert not target.exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "missing", "hash"])
def test_rejects_bad_members_before_installing(tmp_path, kind):
    _, output = bundle(tmp_path)
    archive = output / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        if kind != "missing":
            info = tarfile.TarInfo("../escape" if kind == "traversal" else "samples/history.csv")
            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "../../escape"
                tar.addfile(info)
            else:
                payload = b"time,value\r\n0,2\r\n"
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    metadata = output / "manifest.json"
    manifest = json.loads(metadata.read_text())
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(manifest))
    target = tmp_path / "target"
    with pytest.raises(ResultsError):
        restore_results(target, output)
    assert not (target / "samples").exists()
    assert not (target / "solution").exists()
    assert not (tmp_path / "escape").exists()


def test_lfs_pointer_help_and_absent_bundle(tmp_path):
    assert restore_results(tmp_path) == []
    _, output = bundle(tmp_path)
    (output / "data.tar.gz").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n"
    )
    with pytest.raises(ResultsError, match="git lfs pull"):
        restore_results(tmp_path / "target", output)


def test_complete_roundtrip(tmp_path):
    source, output = bundle(tmp_path)
    target = tmp_path / "target"
    assert restore_results(target, output) == ["samples", "solution"]
    for relative in ("samples/history.csv", "solution/state.bin"):
        assert (target / relative).read_bytes() == (source / relative).read_bytes()


def test_nested_reference_roots(tmp_path):
    source, output = bundle(tmp_path)
    relative = "reference_flow/samples/profiles.csv"
    (source / relative).parent.mkdir(parents=True)
    (source / relative).write_bytes(b"reference")
    manifest = pack_results(
        source,
        output,
        ["samples/history.csv", "solution/state.bin", relative],
        scientific_status="partial",
        provenance={},
    )
    assert manifest["restore_roots"] == ["reference_flow/samples", "samples", "solution"]
    target = tmp_path / "target"
    (target / "reference_flow").mkdir(parents=True)
    (target / "reference_flow/setup.py").write_text("source template")
    assert restore_results(target, output) == manifest["restore_roots"]
    assert (target / relative).read_bytes() == b"reference"
    assert (target / "reference_flow/setup.py").read_text() == "source template"


def test_same_length_rewrite_rejected(tmp_path, monkeypatch):
    from openonda import results

    source, _ = bundle(tmp_path)
    original = results._HashingReader.read

    def rewrite(reader, size=-1):
        data = original(reader, size)
        if data:
            path = source / "samples/history.csv"
            path.write_bytes(b"X" * path.stat().st_size)
        return data

    monkeypatch.setattr(results._HashingReader, "read", rewrite)
    with pytest.raises(ResultsError, match="Input changed"):
        pack_results(
            source,
            tmp_path / "changed",
            ["samples/history.csv"],
            scientific_status="partial",
            provenance={},
        )


def test_publication_conflict_rolls_back_own_roots(tmp_path, monkeypatch):
    from openonda import results

    _, output = bundle(tmp_path)
    target = tmp_path / "target"
    original = results.os.replace

    def conflict(source, destination):
        if destination == target / "solution":
            (destination / "concurrent.bin").write_bytes(b"solver")
        return original(source, destination)

    monkeypatch.setattr(results.os, "replace", conflict)
    with pytest.raises(OSError):
        restore_results(target, output)
    assert not (target / "samples").exists()
    assert (target / "solution/concurrent.bin").read_bytes() == b"solver"
    assert not (target / "solution/state.bin").exists()


@pytest.mark.parametrize("damage", [None, "corrupt", "missing", "pointer", "whole_hash"])
def test_sharded_bundle_roundtrip_and_rejection(tmp_path, monkeypatch, damage):
    from openonda import results

    monkeypatch.setattr(results, "_ARCHIVE_PART_BYTES", 64)
    source, output = bundle(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    parts = manifest["archive_parts"]
    assert len(parts) > 1
    assert not (output / "data.tar.gz").exists()
    assert all(record["size"] <= 64 for record in parts)
    part = output / parts[-1]["name"]
    if damage == "corrupt":
        original = part.read_bytes()
        part.write_bytes(bytes([original[0] ^ 255]) + original[1:])
    elif damage == "missing":
        part.unlink()
    elif damage == "pointer":
        part.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n")
    elif damage == "whole_hash":
        manifest["archive_sha256"] = "0" * 64
        (output / "manifest.json").write_text(json.dumps(manifest))
    target = tmp_path / "target"
    if damage:
        with pytest.raises(
            ResultsError, match="git lfs pull" if damage == "pointer" else "archive"
        ):
            restore_results(target, output)
        assert not (target / "samples").exists()
        assert not (target / "solution").exists()
    else:
        assert restore_results(target, output) == ["samples", "solution"]
        for relative in ("samples/history.csv", "solution/state.bin"):
            assert (target / relative).read_bytes() == (source / relative).read_bytes()


@pytest.mark.parametrize("suffix,attribute", [("pvd", "file"), ("pvtu", "Source"), ("vtm", "file")])
def test_vtk_collection_nested_roundtrip(tmp_path, suffix, attribute):
    source = tmp_path / "source"
    collection = f"samples/collection.{suffix}"
    child = "samples/nested/frame.vtu"
    (source / child).parent.mkdir(parents=True)
    (source / child).write_bytes(b"original VTK bytes")
    payload = f'<VTKFile><DataSet {attribute}="nested/frame.vtu"/></VTKFile>'.encode()
    (source / collection).write_bytes(payload)
    output = tmp_path / "bundle"
    pack_results(source, output, [collection, child], scientific_status="partial", provenance={})
    target = tmp_path / "target"
    assert restore_results(target, output) == ["samples"]
    assert (target / collection).read_bytes() == payload
    assert (target / child).read_bytes() == b"original VTK bytes"


@pytest.mark.parametrize(
    "reference",
    [
        "missing.vts",
        "/outside.vts",
        "../outside.vts",
        "nested/../../outside.vts",
        "C:/outside.vts",
        "nested\\outside.vts",
    ],
)
def test_vtk_pack_rejects_incomplete_or_unsafe_collection(tmp_path, reference):
    source, output = bundle(tmp_path)
    metadata_before = (output / "manifest.json").read_bytes()
    archive_before = (output / "data.tar.gz").read_bytes()
    (source / "samples/wake.pvd").write_text(f'<VTKFile><DataSet file="{reference}"/></VTKFile>')
    with pytest.raises(ResultsError, match="(Missing VTK|Unsafe)"):
        pack_results(
            source, output, ["samples/wake.pvd"], scientific_status="partial", provenance={}
        )
    assert (output / "manifest.json").read_bytes() == metadata_before
    assert (output / "data.tar.gz").read_bytes() == archive_before


def test_vtk_restore_reference_corruption_installs_nothing(tmp_path):
    _, output = bundle(tmp_path)
    archive = output / "data.tar.gz"
    payloads = {
        "samples/wake.pvd": b'<VTKFile><DataSet file="missing.vts"/></VTKFile>',
        "solution/state.bin": b"genuine bytes",
    }
    with tarfile.open(archive, "w:gz") as tar:
        for relative, payload in payloads.items():
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    metadata = output / "manifest.json"
    manifest = json.loads(metadata.read_text())
    manifest["files"] = [
        {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in payloads.items()
    ]
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(manifest))
    target = tmp_path / "target"
    with pytest.raises(ResultsError, match="Missing VTK"):
        restore_results(target, output)
    assert not (target / "samples").exists()
    assert not (target / "solution").exists()
    assert not list(target.glob(".results-restore-*"))


def test_vtk_collection_rejects_xml_entity_expansion(tmp_path):
    source, output = bundle(tmp_path)
    (source / "samples/wake.pvd").write_text(
        '<!DOCTYPE VTKFile [<!ENTITY local "frame.vts">]>'
        '<VTKFile><DataSet file="&local;"/></VTKFile>'
    )
    with pytest.raises(ResultsError, match="Invalid VTK collection"):
        pack_results(
            source, output, ["samples/wake.pvd"], scientific_status="partial", provenance={}
        )
