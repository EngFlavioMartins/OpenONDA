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
