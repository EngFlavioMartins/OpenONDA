import pytest
import taichi as ti

from source.solvers.VPM import VPMSetup, VPMSolver
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig
from source.solvers.VPM.core import solver as solver_module
from source.solvers.VPM.runtime.backend import reset_taichi_backend


def _pretend_vulkan_on_cpu(monkeypatch):
    """Initialize Taichi on CPU while making VPMSolver exercise Vulkan policy."""

    def _fake_initialize_taichi_backend(*args, **kwargs):
        ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32, offline_cache=False)
        return "VULKAN"

    monkeypatch.setattr(solver_module, "initialize_taichi_backend", _fake_initialize_taichi_backend)


def _grid_diffusion_config(**kwargs) -> VPMSetup:
    return VPMSetup(
        time_step_size=0.01,
        compute_device="VULKAN",
        stretching=StretchingConfig.disabled(),
        advection=AdvectionConfig(scheme="NONE"),
        viscous=ViscousConfig.dvh(particle_spacing=0.25, padding=0.0, kinematic_viscosity=1.0e-3),
        checkpoint_interval_steps=0,
        logging_interval_steps=0,
        **kwargs,
    )


def test_vulkan_grid_diffusion_requires_domain_bounds(monkeypatch, tmp_path):
    reset_taichi_backend()
    _pretend_vulkan_on_cpu(monkeypatch)
    try:
        with pytest.raises(ValueError, match="requires domain_bounds"):
            VPMSolver(_grid_diffusion_config(checkpoint_directory=str(tmp_path)))
    finally:
        reset_taichi_backend()


def test_vulkan_grid_diffusion_preallocates_fixed_domain_grid(monkeypatch, tmp_path):
    reset_taichi_backend()
    _pretend_vulkan_on_cpu(monkeypatch)
    try:
        solver = VPMSolver(
            _grid_diffusion_config(
                checkpoint_directory=str(tmp_path),
                domain_bounds=[-0.5, 0.5, -0.25, 0.25, -0.25, 0.25],
            )
        )
        assert solver.compute_device == "VULKAN"
        assert solver.physics._require_fixed_grid_allocation is True
        assert solver.physics._grid_shape == (5, 5, 5)
    finally:
        reset_taichi_backend()


def test_cpu_grid_diffusion_does_not_preallocate_the_removal_domain(tmp_path):
    """Only Vulkan needs the fixed grid; CPU allocates around live particles."""
    reset_taichi_backend()
    try:
        solver = VPMSolver(
            VPMSetup(
                time_step_size=0.01,
                compute_device="CPU",
                stretching=StretchingConfig.disabled(),
                advection=AdvectionConfig(scheme="NONE"),
                viscous=ViscousConfig.dvh(
                    particle_spacing=0.25, padding=3.0, kinematic_viscosity=1.0e-3
                ),
                domain_bounds=[-2.0, 10.0, -2.0, 2.0, -2.0, 2.0],
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
            )
        )
        assert solver.compute_device == "CPU"
        assert solver.physics._require_fixed_grid_allocation is False
        assert solver.physics._grid_a is None
    finally:
        reset_taichi_backend()
