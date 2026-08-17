"""Verification of the band-limited FVM->VPM strength transfer.

Strengths stay under the cap, the representable band is carried exactly, and
the rest is reported.
"""

import numpy as np
import pytest

from source.coupler.core.helpers.continuous_overlap import (
    DEFAULT_TRANSFER_AMPLIFICATION_CAP,
    _gaussian_mollified_circulation,
    bandlimited_transfer,
    bounded_local_transfer,
    spectral_band_ratio,
    transfer_symbols,
)


def _plane_wave(shape, h, wavelength_cells, component=2, axis=0):
    coords = np.arange(shape[axis]) * h
    field = np.zeros((*shape, 3))
    wave = np.sin(2.0 * np.pi * coords / (wavelength_cells * h))
    view = [None, None, None]
    view[axis] = slice(None)
    field[..., component] = wave[tuple(slice(None) if i == axis else None for i in range(3))]
    return field


def test_symbols_respect_the_amplification_cap():
    shape = (32, 32, 32)
    for cap in (1.5, 2.0, 5.0):
        w, phi = transfer_symbols(shape, 0.1, sigma=0.1, amplification_cap=cap)
        assert w.max() <= cap * (1.0 + 1e-12)
        assert w.max() == pytest.approx(cap, rel=1e-3)
        assert phi.max() <= 1.0 + 1e-12
        assert phi.min() >= 0.0
        # k = 0 must be carried exactly, otherwise every hand-off leaks a fixed
        # fraction of the total circulation.
        assert phi.ravel()[0] == pytest.approx(1.0, abs=1e-12)


def test_amplification_cap_is_enforced_on_a_real_field():
    shape = (48, 48, 48)
    h = 0.05
    rng = np.random.default_rng(0)
    target = rng.normal(size=(*shape, 3)) * 1e-4
    for cap in (1.5, 2.0, 4.0):
        gamma, _, _ = bandlimited_transfer(
            target.reshape(-1, 3), shape, h, sigma=h, amplification_cap=cap
        )
        ratio = (
            np.linalg.norm(gamma, axis=1).max()
            / np.linalg.norm(target.reshape(-1, 3), axis=1).max()
        )
        assert ratio <= cap * 1.05


@pytest.mark.verification
@pytest.mark.parametrize("wavelength_cells", [8, 12, 16, 24])
def test_resolved_wavelengths_are_reproduced_to_round_off(wavelength_cells):
    """The particle field must reproduce the band it claims to carry."""
    shape = (64, 64, 64)
    h = 0.05
    target = _plane_wave(shape, h, wavelength_cells)
    gamma, inband, _ = bandlimited_transfer(
        target.reshape(-1, 3), shape, h, sigma=h, amplification_cap=4.0
    )
    mollified = _gaussian_mollified_circulation(gamma, shape, h, sigma=h).reshape(-1, 3)
    core = (slice(16, 48),) * 3
    left = mollified.reshape(*shape, 3)[core]
    right = inband.reshape(*shape, 3)[core]
    error = np.linalg.norm(left - right) / (np.linalg.norm(right) + 1e-30)
    assert error < 5e-3, f"in-band residual {error:.3e} at lambda={wavelength_cells}h"


@pytest.mark.verification
def test_out_of_band_fraction_is_a_monotone_resolution_diagnostic():
    """Finer structures must report a larger unrepresentable fraction."""
    shape = (64, 64, 64)
    h = 0.05
    previous = -1.0
    for wavelength_cells in (32, 16, 8, 4, 2):
        target = _plane_wave(shape, h, wavelength_cells)
        _, _, out_of_band = bandlimited_transfer(
            target.reshape(-1, 3), shape, h, sigma=h, amplification_cap=2.0
        )
        assert out_of_band >= previous - 1e-9, "out-of-band fraction must not decrease"
        previous = out_of_band
    # The grid scale is genuinely unrepresentable; that must be visible.
    assert previous > 0.3


def test_transfer_is_linear():
    shape = (24, 24, 24)
    h = 0.1
    rng = np.random.default_rng(5)
    a = rng.normal(size=(np.prod(shape), 3))
    b = rng.normal(size=(np.prod(shape), 3))
    ga, _, _ = bandlimited_transfer(a, shape, h, sigma=h)
    gb, _, _ = bandlimited_transfer(b, shape, h, sigma=h)
    gab, _, _ = bandlimited_transfer(2.0 * a - 3.0 * b, shape, h, sigma=h)
    np.testing.assert_allclose(gab, 2.0 * ga - 3.0 * gb, atol=1e-10)


def test_zero_target_transfers_to_zero():
    shape = (16, 16, 16)
    gamma, inband, out_of_band = bandlimited_transfer(
        np.zeros((np.prod(shape), 3)), shape, 0.1, sigma=0.1
    )
    assert not np.any(gamma)
    assert not np.any(inband)
    assert out_of_band == pytest.approx(0.0)


def test_default_cap_is_conservative():
    assert 1.0 < DEFAULT_TRANSFER_AMPLIFICATION_CAP <= 4.0


def test_local_transfer_preserves_a_purely_lagrangian_field():
    shape = (16, 16, 16)
    rng = np.random.default_rng(17)
    strength = rng.normal(size=(np.prod(shape), 3)) * 1.0e-4
    result, _, residual_pre, residual_post, _ = bounded_local_transfer(
        strength,
        np.zeros_like(strength),
        np.zeros(len(strength)),
        shape,
        0.1,
        sigma=0.1,
    )
    np.testing.assert_array_equal(result, strength)
    assert residual_pre == pytest.approx(0.0)
    assert residual_post == pytest.approx(0.0)


def test_local_transfer_does_not_fill_the_far_field():
    shape = (33, 33, 33)
    target = np.zeros((*shape, 3))
    centre = np.asarray(shape) // 2
    target[*centre, 2] = 1.0
    strength, _, residual_pre, residual_post, amplification = bounded_local_transfer(
        np.zeros((np.prod(shape), 3)),
        target.reshape(-1, 3),
        np.ones(np.prod(shape)),
        shape,
        0.1,
        sigma=0.1,
    )
    strength = strength.reshape(*shape, 3)
    coordinates = np.stack(np.meshgrid(*map(np.arange, shape), indexing="ij"), axis=-1)
    far = np.max(np.abs(coordinates - centre), axis=-1) > 4
    assert not np.any(strength[far])
    assert residual_post < residual_pre
    assert amplification <= DEFAULT_TRANSFER_AMPLIFICATION_CAP


def test_spectral_band_ratio_is_one_for_identical_fields():
    shape = (32, 32, 32)
    rng = np.random.default_rng(11)
    field = rng.normal(size=(np.prod(shape), 3))
    ratios = spectral_band_ratio(field, field, shape, 0.1)
    assert ratios
    for value in ratios.values():
        assert value == pytest.approx(1.0, abs=1e-9)


def test_spectral_band_ratio_detects_a_grid_scale_excess():
    """The scalar L1 flux ratio hid a 2.7x grid-scale excess; this must not."""
    shape = (32, 32, 32)
    h = 0.1
    # Both wavelengths land exactly on a lattice wavenumber (indices 2 and 10),
    # so there is no leakage between bands to confuse the assertion.
    coarse = _plane_wave(shape, h, 16.0).reshape(-1, 3)
    fine = _plane_wave(shape, h, 3.2).reshape(-1, 3)
    ratios = spectral_band_ratio(coarse + 2.0 * fine, coarse + fine, shape, h)
    assert ratios["2-4h"] == pytest.approx(2.0, rel=1e-6)
    assert ratios["8-16h"] == pytest.approx(1.0, rel=1e-6)
