import pytest
import taichi as ti

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.backend import reset_taichi_backend
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig
from source.solvers.VPM.core import solver as solver_module


def _pretend_vulkan_on_cpu(monkeypatch):
    """Initialize Taichi on CPU while making Solver exercise Vulkan policy."""

    def _fake_initialize_taichi_backend(*args, **kwargs):
        ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32, offline_cache=False)
        return "VULKAN"

    monkeypatch.setattr(solver_module, "initialize_taichi_backend", _fake_initialize_taichi_backend)


def _grid_diffusion_config(**kwargs) -> VPMSetup:
    return VPMSetup(
        time_step_size=0.01,
        processing_unit="VULKAN",
        stretching=StretchingConfig.disabled(),
        advection=AdvectionConfig(scheme="NONE"),
        viscous=ViscousConfig.dvh(particle_spacing=0.25, padding=0.0, viscosity=1.0e-3),
        backup_frequency=0,
        logging_frequency=0,
        **kwargs,
    )


def test_vulkan_grid_diffusion_requires_domain_bounds(monkeypatch, tmp_path):
    reset_taichi_backend()
    _pretend_vulkan_on_cpu(monkeypatch)
    try:
        with pytest.raises(ValueError, match="requires vpm_domain_bounds"):
            Solver(_grid_diffusion_config(backup_directory=str(tmp_path)))
    finally:
        reset_taichi_backend()


def test_vulkan_grid_diffusion_preallocates_fixed_domain_grid(monkeypatch, tmp_path):
    reset_taichi_backend()
    _pretend_vulkan_on_cpu(monkeypatch)
    try:
        solver = Solver(
            _grid_diffusion_config(
                backup_directory=str(tmp_path),
                vpm_domain_bounds=[-0.5, 0.5, -0.25, 0.25, -0.25, 0.25],
            )
        )
        assert solver.processing_unit == "VULKAN"
        assert solver.physics._require_fixed_grid_allocation is True
        assert solver.physics._grid_shape == (5, 5, 5)
    finally:
        reset_taichi_backend()


def test_cpu_grid_diffusion_does_not_preallocate_the_removal_domain(tmp_path):
    """Only Vulkan needs the fixed grid; CPU allocates around live particles."""
    reset_taichi_backend()
    try:
        solver = Solver(
            VPMSetup(
                time_step_size=0.01,
                processing_unit="CPU",
                stretching=StretchingConfig.disabled(),
                advection=AdvectionConfig(scheme="NONE"),
                viscous=ViscousConfig.dvh(particle_spacing=0.25, padding=3.0, viscosity=1.0e-3),
                vpm_domain_bounds=[-2.0, 10.0, -2.0, 2.0, -2.0, 2.0],
                backup_frequency=0,
                logging_frequency=0,
                backup_directory=str(tmp_path),
            )
        )
        assert solver.processing_unit == "CPU"
        assert solver.physics._require_fixed_grid_allocation is False
        assert solver.physics._grid_a is None
    finally:
        reset_taichi_backend()
