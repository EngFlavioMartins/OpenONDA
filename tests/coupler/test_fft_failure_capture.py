"""Pure host tests: traceback capture never invokes device methods."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / "tests/support/cylinder/capture_fft_failure.py"
spec = importlib.util.spec_from_file_location("fft_failure_capture_under_test", PATH)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def failed_fft():
    namespace = {"np": np}
    exec(
        compile(
            "def rfft(self, array):\n    raise RuntimeError('CUFFT_INVALID_VALUE')\n",
            "/frozen/gaussian_mesh/runtime.py",
            "exec",
        ),
        namespace,
    )
    exec(
        compile(
            "def evaluate(self, source, query, source_only=True):\n"
            "    images=[(0, True)]\n    return rfft(plan, array)\n",
            "/frozen/gaussian_mesh/session.py",
            "exec",
        ),
        namespace,
    )
    namespace["array"] = SimpleNamespace(
        shape=(15, 15, 15),
        strides=(900, 60, 4),
        dtype=np.dtype("float32"),
        flags=SimpleNamespace(c_contiguous=True),
        data=SimpleNamespace(ptr=4096 + 15**3 * 4),
    )
    namespace["plan"] = SimpleNamespace(
        shape=(15, 15, 15),
        spectrum_shape=(15, 15, 8),
        work_bytes=42,
        max_plan_bytes=100,
        single_workspace=False,
        plan_builds=2,
        closed=True,
    )
    source = (
        np.zeros((3, 3), np.float32),
        np.ones((3, 3), np.float32),
        np.full(3, 0.04, np.float32),
    )
    query = np.full((2, 3), 0.5, np.float32)
    try:
        namespace["evaluate"](None, source, query)
    except RuntimeError as cause:
        try:
            raise ValueError("sampler failed") from cause
        except ValueError as error:
            return error, source, query


def test_captures_original_layout_and_exact_host_arrays(tmp_path):
    error, source, query = failed_fft()
    result = capture.save_fft_failure(error, tmp_path / "fft_failure")
    assert result["captured"] and result["host_inputs"]
    report = json.loads((tmp_path / "fft_failure.json").read_text())
    assert report["fft_frames"][0]["array"]["pointer_mod_8"] == 4
    assert report["fft_frames"][0]["shape"] == [15, 15, 15]
    assert report["source_only_primary"] is True
    with np.load(tmp_path / "fft_failure.npz", allow_pickle=False) as saved:
        for key, value in zip(
            ("source_position", "source_strength", "source_core", "query"),
            (*source, query),
            strict=True,
        ):
            np.testing.assert_array_equal(saved[key], value)
            assert saved[key].dtype == value.dtype
    assert all(value.flags.writeable for value in source)
    with pytest.raises(FileExistsError):
        capture.save_fft_failure(error, tmp_path / "fft_failure")


def test_non_fft_failure_creates_no_checkpoint_file(tmp_path):
    assert (
        capture.save_fft_failure(RuntimeError("unrelated"), tmp_path / "failure")["captured"]
        is False
    )
    assert not list(tmp_path.iterdir())


def test_context_cycle_is_bounded():
    error = RuntimeError("cycle")
    error.__context__ = error
    report, arrays = capture.collect_fft_failure(error)
    assert len(report["exceptions"]) == 1
    assert arrays == {}


def test_constructor_memory_validation_captures_host_only_evidence(tmp_path):
    namespace = {}
    exec(
        compile(
            "def __init__(self):\n    free, total = 100, 600\n"
            "    raise MemoryError('insufficient free device memory')\n",
            "/frozen/gaussian_mesh/fields.py",
            "exec",
        ),
        namespace,
    )
    owner = SimpleNamespace(
        shape=(17, 18, 19),
        fft_shape=(35, 36, 40),
        estimated_field_bytes=200,
        max_plan_bytes=25,
        max_correction_bytes=50,
        max_scratch_bytes=400,
        max_total_bytes=450,
    )
    try:
        namespace["__init__"](owner)
    except MemoryError as error:
        result = capture.save_fft_failure(error, tmp_path / "memory_failure")
    assert result["captured"] and not result["host_inputs"]
    report = json.loads((tmp_path / "memory_failure.json").read_text())
    frame = report["field_frames"][0]
    assert frame["free_device_bytes"] == 100 and frame["total_device_bytes"] == 600
    assert frame["estimated_field_bytes"] == 200 and frame["max_plan_bytes"] == 25
    assert report["fft_frames"] == []
