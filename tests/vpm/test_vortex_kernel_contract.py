"""Qualification tests for the shared radial vortex-kernel contract."""

import inspect

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.config.case import Numerics
from source.solvers.vpm.kernels.base import make_device_vortex_kernels, make_vortex_kernel
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.fmm import FMMInduction
from source.solvers.vpm.physics.induction.treecode import TreecodeInduction

KERNELS = ("GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS")


@ti.data_oriented
class _DeviceKernelSampler:
    def __init__(self, name: str, count: int) -> None:
        functions = make_device_vortex_kernels(name, ti.f32)
        self.q_kernel = functions["q_"]
        self.zeta_kernel = functions["zeta_"]
        self.rho = ti.field(dtype=ti.f32, shape=count)
        self.q = ti.field(dtype=ti.f32, shape=count)
        self.zeta = ti.field(dtype=ti.f32, shape=count)

    @ti.kernel
    def evaluate(self):
        for index in self.rho:
            self.q[index] = self.q_kernel(self.rho[index])
            self.zeta[index] = self.zeta_kernel(self.rho[index])


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("induction_type", (DirectInduction, FMMInduction))
def test_direct_and_fmm_advertise_every_supported_kernel(name, induction_type):
    """Construction rejects no member of the shared radial-kernel family."""
    numerics = Numerics(particle_kernel=name, induction=induction_type(), verbose=False)
    assert numerics.particle_kernel == name


@pytest.mark.parametrize("name", KERNELS)
def test_treecode_rejects_unsupported_kernels_at_configuration_boundary(name):
    induction = TreecodeInduction()
    if name in induction.supported_kernels:
        Numerics(particle_kernel=name, induction=induction, verbose=False)
    else:
        with pytest.raises(ValueError, match="does not support"):
            Numerics(particle_kernel=name, induction=TreecodeInduction(), verbose=False)


@pytest.mark.parametrize("name", KERNELS)
def test_radial_kernel_has_regular_origin_and_singular_far_field_limit(name):
    kernel = make_vortex_kernel(name)
    rho = np.array([0.0, 1.0e-4, 0.25, 1.0, 20.0])

    assert kernel.q(rho)[0] == pytest.approx(0.0, abs=1.0e-15)
    assert np.all(np.isfinite(kernel.q(rho)))
    assert np.all(np.isfinite(kernel.zeta(rho)))
    assert kernel.q(rho)[-1] == pytest.approx(kernel.q_infinity, rel=2.0e-5)


@pytest.mark.parametrize("name", KERNELS)
def test_host_and_device_radial_functions_share_one_registry_contract(name):
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, offline_cache=False, cpu_max_num_threads=2)
    rho = np.array([0.0, 1.0e-4, 0.25, 1.0, 4.0, 20.0], dtype=np.float32)
    sampler = _DeviceKernelSampler(name, len(rho))
    sampler.rho.from_numpy(rho)
    sampler.evaluate()
    kernel = make_vortex_kernel(name)

    np.testing.assert_allclose(
        sampler.q.to_numpy(),
        kernel.q(rho),
        rtol=4.0e-5,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        sampler.zeta.to_numpy(),
        kernel.zeta(rho),
        rtol=4.0e-5,
        atol=2.0e-8,
    )


@pytest.mark.parametrize("name", KERNELS)
def test_pair_radius_kernel_preserves_the_conservative_two_particle_rate(name):
    kernel = make_vortex_kernel(name)
    displacement = np.array([0.7, -0.3, 0.2])
    target_strength = np.array([0.2, 0.4, -0.1])
    source_strength = np.array([-0.3, 0.1, 0.5])

    target_rate = kernel.transposed_rate_pair(
        displacement, target_strength, source_strength, 0.2, 0.3
    )
    source_rate = kernel.transposed_rate_pair(
        -displacement, source_strength, target_strength, 0.3, 0.2
    )

    np.testing.assert_allclose(target_rate + source_rate, 0.0, rtol=0.0, atol=2.0e-14)


def test_kernel_gradient_matches_a_finite_difference_of_pair_velocity():
    kernel = make_vortex_kernel("GAUSSIAN")
    displacement = np.array([0.7, -0.3, 0.2])
    source_strength = np.array([0.2, 0.4, -0.1])
    epsilon = 1.0e-6
    finite_difference = np.column_stack(
        [
            (
                kernel.velocity_pair(
                    displacement + epsilon * np.eye(3)[axis], source_strength, 0.2, 0.3
                )
                - kernel.velocity_pair(
                    displacement - epsilon * np.eye(3)[axis], source_strength, 0.2, 0.3
                )
            )
            / (2.0 * epsilon)
            for axis in range(3)
        ]
    )

    np.testing.assert_allclose(
        kernel.gradient_pair(displacement, source_strength, 0.2, 0.3),
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_induction_backends_declare_strength_rate_semantics_explicitly():
    direct = DirectInduction()
    treecode = TreecodeInduction()
    fmm = FMMInduction()

    assert direct.strength_rate_mode == "PAIRWISE_TRANSPOSED"
    assert treecode.strength_rate_mode == "HIERARCHICAL_GRADIENT"
    assert fmm.strength_rate_mode == "HIERARCHICAL_GRADIENT"
    assert treecode.diagnostics["strength_rate_mode"] == "HIERARCHICAL_GRADIENT"
    assert fmm.diagnostics.strength_rate_mode == "HIERARCHICAL_GRADIENT"


@pytest.mark.parametrize("induction_type", (DirectInduction, TreecodeInduction, FMMInduction))
def test_public_induction_constructors_expose_no_tuning_arguments(induction_type):
    assert not inspect.signature(induction_type).parameters
