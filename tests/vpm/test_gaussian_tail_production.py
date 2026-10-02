"""Production extraction parity, owned value identity and fail-closed admission."""

from dataclasses import FrozenInstanceError, asdict
import gc
from pathlib import Path
import weakref

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_tail import (
    SourceValueMismatchError,
    _interval,
    certificate,
    prepare_tail_source,
    query_tail_bound,
    validate_source_values,
)
from tests.vpm._gaussian_tail_aabb_certificate import prepare_tail_source as prototype_prepare
from tests.vpm._gaussian_tail_aabb_certificate import query_tail_bound as prototype_query
from tests.vpm._gaussian_tail_coefficient_enclosure import (
    coefficient_enclosure as prototype_coefficient,
)
from tests.vpm.test_gaussian_image_tail_moments import cloud


@pytest.mark.parametrize("kind", ["random", "cancelled", "axial", "translated", "near_admission"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_extracted_bound_matches_qualified_prototype_exactly(kind, dtype):
    x, g, sigma, targets, zmin, zmax, _ = cloud(kind)
    x, g, sigma = (value.astype(dtype) for value in (x, g, sigma))
    source = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    control = prototype_prepare(x, g, sigma, z_min=zmin, z_max=zmax)
    for k in (64, 128):
        result = asdict(query_tail_bound(source, targets.min(axis=0), targets.max(axis=0), shells=k))
        expected = asdict(prototype_query(control, targets.min(axis=0), targets.max(axis=0), shells=k))
        # Production provenance retains original dtype/shape, unlike prototype.
        result["diagnostics"].pop("source_sha256")
        expected["diagnostics"].pop("source_sha256")
        assert result == expected
    validate_source_values(source, x.copy(), g.copy(), sigma.copy(), z_min=zmin, z_max=zmax)


@pytest.mark.parametrize("k", [1, 32, 64, 128, 256])
def test_coefficient_arithmetic_unchanged(k):
    assert asdict(_interval.coefficient_enclosure(k)) == asdict(prototype_coefficient(k))


@pytest.mark.parametrize("field", ["position", "strength", "radius"])
def test_each_source_value_change_rejects_reuse(field):
    x, g, sigma, _, zmin, zmax, _ = cloud("random")
    source = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    values = {"position": x, "strength": g, "radius": sigma}
    values[field].flat[0] = np.nextafter(values[field].flat[0], np.inf)
    with pytest.raises(SourceValueMismatchError):
        validate_source_values(source, x, g, sigma, z_min=zmin, z_max=zmax)


def test_original_dtype_shape_and_signed_zero_are_part_of_exact_identity():
    x = np.array([[0., .1, .25]], dtype=np.float32)
    g, sigma = np.ones((1, 3), dtype=np.float32), np.ones(1, dtype=np.float32)*.04
    source = prepare_tail_source(x, g, sigma, z_min=0., z_max=1.)
    assert source.source_identity.arrays[0].dtype == x.dtype.str
    assert source.source_identity.arrays[0].shape == (1, 3)
    with pytest.raises(SourceValueMismatchError):
        validate_source_values(source, x.astype(np.float64), g, sigma, z_min=0., z_max=1.)
    x[0, 0] = -0.
    with pytest.raises(SourceValueMismatchError):
        validate_source_values(source, x, g, sigma, z_min=0., z_max=1.)
    x[0, 0] = 0.
    with pytest.raises(SourceValueMismatchError):
        validate_source_values(source, x, g, sigma, z_min=-0., z_max=1.)
    with pytest.raises(ValueError):
        validate_source_values(source, x.reshape(3), g, sigma, z_min=0., z_max=1.)


def test_byte_equality_not_digest_equality_controls_admission(monkeypatch):
    x, g, sigma, _, zmin, zmax, _ = cloud("random")
    monkeypatch.setattr(certificate, "_identity_digest", lambda identity: "same-hash")
    source = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    g[0, 0] += .1
    other = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    assert source.source_sha256 == other.source_sha256
    with pytest.raises(SourceValueMismatchError):
        validate_source_values(source, x, g, sigma, z_min=zmin, z_max=zmax)


def test_snapshot_owns_bytes_and_does_not_retain_caller_arrays():
    x, g, sigma = np.zeros((5, 3)), np.ones((5, 3)), np.ones(5)*.04
    references = [weakref.ref(v) for v in (x, g, sigma)]
    byte_count = x.nbytes+g.nbytes+sigma.nbytes+16
    source = prepare_tail_source(x, g, sigma, z_min=-.5, z_max=.5)
    del x, g, sigma
    gc.collect()
    assert all(ref() is None for ref in references)
    assert source.source_identity.retained_bytes == byte_count
    assert all(type(item.payload) is bytes for item in source.source_identity.arrays)
    with pytest.raises(FrozenInstanceError):
        source.origin = (0., 0., 0.)


def test_stride_layout_does_not_change_logical_source_value_identity():
    x, g, sigma, _, zmin, zmax, _ = cloud("random")
    source = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    validate_source_values(source, np.asfortranarray(x), np.asfortranarray(g), sigma,
                           z_min=zmin, z_max=zmax)


@pytest.mark.parametrize("dtype", [np.int64, np.float16, np.complex128, object])
def test_unsupported_source_dtypes_fail_before_preparation(dtype):
    with pytest.raises(ValueError, match="binary32 or binary64"):
        prepare_tail_source(np.zeros((2, 3), dtype=dtype), np.ones((2, 3)), np.ones(2),
                            z_min=-.5, z_max=.5)


@pytest.mark.parametrize("count", [0, 3])
def test_zero_source_identity_and_zero_field_admission(count):
    x, g, sigma = np.zeros((count, 3)), np.zeros((count, 3)), np.ones(count)
    source = prepare_tail_source(x, g, sigma, z_min=-.5, z_max=.5)
    validate_source_values(source, x, g, sigma, z_min=-.5, z_max=.5)
    result = query_tail_bound(source, np.ones(3)*1e5, np.ones(3)*1e6, shells=1)
    assert result.velocity_upper == result.gradient_upper == 0.


def test_bad_floating_environment_fails_closed_before_source_or_query_work(monkeypatch):
    x, g, sigma = np.zeros((1, 3)), np.ones((1, 3)), np.ones(1)*.04
    source = prepare_tail_source(x, g, sigma, z_min=-.5, z_max=.5)
    def fail():
        raise RuntimeError("injected unsupported floating environment")
    monkeypatch.setattr(certificate, "_platform", fail)
    with pytest.raises(RuntimeError, match="floating environment"):
        prepare_tail_source(x, g, sigma, z_min=-.5, z_max=.5)
    with pytest.raises(RuntimeError, match="floating environment"):
        query_tail_bound(source, np.zeros(3), np.ones(3), shells=32)


def test_query_failure_does_not_damage_owned_snapshot():
    x, g, sigma, _, zmin, zmax, _ = cloud("random")
    source = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    before = query_tail_bound(source, np.zeros(3), np.ones(3), shells=64)
    with pytest.raises(ValueError, match="L1 extent"):
        query_tail_bound(source, np.ones(3)*100., np.ones(3)*101., shells=1)
    after = query_tail_bound(source, np.zeros(3), np.ones(3), shells=64)
    assert after == before


def test_production_package_has_no_test_or_prototype_imports():
    package = Path(certificate.__file__).parent
    for path in package.glob("*.py"):
        assert "from tests" not in path.read_text()
        assert "import tests" not in path.read_text()
