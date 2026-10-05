"""Allocation recovery preserves the finite operator and uncertain-owner guard."""

from copy import deepcopy

import pytest

from source.solvers.vpm.config.restart import (
    _configuration_mismatches,
    canonical_restart_configuration,
)
from source.solvers.vpm.physics.induction.gaussian_mesh.execution import PortableGaussianImageFields


@pytest.mark.parametrize("phase", ["construct", "prepare", "query"])
def test_clean_cuda_memory_failure_rebuilds_identical_host_request(monkeypatch, caplog, phase):
    from source.solvers.vpm.physics.induction.gaussian_mesh import (
        blocked_fields,
        fields,
        host_fields,
    )

    events = []
    failure = MemoryError("device memory exhausted")

    class CUDA:
        kind = "cuda"

        def __init__(self, *args, **kwargs):
            events.append((f"{self.kind}_construct", args, kwargs))
            if phase == "construct":
                raise failure

        def prepare(self, images):
            events.append((f"{self.kind}_prepare", images))
            if phase == "prepare":
                raise failure
            return {}

        def evaluate_prepared(self, targets):
            raise failure

        def close(self):
            events.append((f"{self.kind}_close",))

    class BlockedCUDA(CUDA):
        kind = "blocked"

    class Host:
        def __init__(self, *args, **kwargs):
            events.append(("host_construct", args, kwargs))

        def prepare(self, images):
            events.append(("host_prepare", images))
            return {}

        def evaluate_prepared(self, targets):
            return targets, "gradient", {}

        def close(self):
            events.append(("host_close",))

    monkeypatch.setattr(fields, "GaussianImageFields", CUDA)
    monkeypatch.setattr(blocked_fields, "GaussianBlockedCUDAFields", BlockedCUDA)
    monkeypatch.setattr(host_fields, "GaussianHostImageFields", Host)
    controls = {"spacing": .03, "tau": .12, "cutoff": .6, "dtype": "float32", "max_scratch_bytes": 123}
    owner = PortableGaussianImageFields("x", "gamma", "sigma", "targets",
                                        execution_backend="cupy_cuda", **controls)
    images = ((-1, True), (1, False))
    owner.prepare(images)
    u, j, diagnostic = owner.evaluate_prepared("query")
    assert (u, j) == ("query", "gradient")
    assert diagnostic["execution_backend"] == "cpu"
    assert diagnostic["memory_fallback"] == str(failure)
    assert "evaluating on CPU: device memory exhausted" in caplog.text
    assert len([record for record in caplog.records if record.name == "vpm"]) == 1
    original, rebuilt = events[0], next(e for e in events if e[0] == "host_construct")
    assert original[1:] == rebuilt[1:]
    assert events.index(("cuda_close",)) < events.index(rebuilt)
    blocked = next(e for e in events if e[0] == "blocked_construct")
    assert original[1:] == blocked[1:]
    assert events.index(("cuda_close",)) < events.index(blocked)
    assert events.index(("blocked_close",)) < events.index(rebuilt)
    assert ("host_prepare", images) in events
    owner.close()


@pytest.mark.parametrize("phase", ["construct", "prepare", "query"])
def test_cuda_memory_failure_subdivides_before_using_cpu(monkeypatch, caplog, phase):
    from source.solvers.vpm.physics.induction.gaussian_mesh import (
        blocked_fields,
        fields,
        host_fields,
    )

    events = []

    class CUDA:
        def __init__(self, *args, **kwargs):
            if phase == "construct":
                raise MemoryError("full FFT does not fit")

        def prepare(self, images):
            if phase == "prepare":
                raise MemoryError("full FFT does not fit")
            return {}

        def evaluate_prepared(self, targets):
            raise MemoryError("full FFT does not fit")

        def close(self):
            events.append("closed_full")

    class BlockedCUDA:
        def __init__(self, *args, **kwargs):
            assert events == ["closed_full"]
            events.append("built_blocks")

        def prepare(self, images):
            assert images == ((0, True),)
            return {}

        def evaluate_prepared(self, targets):
            return targets, "gradient", {"fft_blocks": 2}

        def close(self):
            events.append("closed_blocks")

    monkeypatch.setattr(fields, "GaussianImageFields", CUDA)
    monkeypatch.setattr(blocked_fields, "GaussianBlockedCUDAFields", BlockedCUDA)
    monkeypatch.setattr(host_fields, "GaussianHostImageFields",
                        lambda *a, **kw: pytest.fail("an admitted CUDA block must stay on CUDA"))
    owner = PortableGaussianImageFields(execution_backend="cupy_cuda")
    owner.prepare(((0, True),))
    u, j, report = owner.evaluate_prepared("query")
    assert (u, j) == ("query", "gradient")
    assert report["execution_backend"] == "cupy_cuda"
    assert report["fft_blocks"] == 2
    assert report["memory_fallback"] is None
    assert report["memory_subdivision"] == "full FFT does not fit"
    assert not [record for record in caplog.records if record.name == "vpm"]
    owner.close()
    assert events == ["closed_full", "built_blocks", "closed_blocks"]


@pytest.mark.parametrize("failed_strategy", ["full", "blocked"])
def test_uncertain_cuda_cleanup_cannot_allocate_a_host_replacement(monkeypatch, failed_strategy):
    from source.solvers.vpm.physics.induction.gaussian_mesh import (
        blocked_fields,
        fields,
        host_fields,
    )

    failure = MemoryError("original allocation failure")
    class CUDA:
        def __init__(self, *args, **kwargs):
            raise failure

        def close(self):
            if failed_strategy == "full":
                raise RuntimeError("failed stream drain")

    class BlockedCUDA(CUDA):
        def close(self):
            raise RuntimeError("failed stream drain")

    monkeypatch.setattr(fields, "GaussianImageFields", CUDA)
    monkeypatch.setattr(blocked_fields, "GaussianBlockedCUDAFields", BlockedCUDA)
    monkeypatch.setattr(host_fields, "GaussianHostImageFields",
                        lambda *a, **kw: pytest.fail("uncertain CUDA owner cannot be replaced"))
    with pytest.raises(MemoryError) as captured:
        PortableGaussianImageFields(execution_backend="cupy_cuda")
    assert captured.value is failure
    assert any("cleanup failed" in note for note in failure.__notes__)


def test_restart_execution_placement_is_portable_but_math_remains_strict():
    stored = {"induction": {"gaussian_mesh_policy": {
        "backend": "cupy_cuda", "tail_contract": "gaussian_interval_remainder_v1",
        "mesh": {"order": 10, "spacing_over_tau": .25}, "max_scratch_bytes": 100,
    }}}
    current = deepcopy(stored)
    current["induction"]["gaussian_mesh_policy"]["backend"] = "cpu"
    def differences():
        return _configuration_mismatches(canonical_restart_configuration(current),
                                         canonical_restart_configuration(stored))
    assert differences() == []
    current["induction"]["gaussian_mesh_policy"]["mesh"]["order"] = 8
    assert differences() == ["induction.gaussian_mesh_policy.mesh.order"]
    assert stored["induction"]["gaussian_mesh_policy"]["backend"] == "cupy_cuda"
