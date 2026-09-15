"""Native VLM continuation rejects changed physics or generated geometry."""

from dataclasses import replace

from _flat_plate_geometry import create_flat_plate
import h5py
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.solver.restart import (
    validate_vlm_restart,
    write_vlm_restart,
)
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver


def test_restart_identity_requires_current_physics_and_geometry(tmp_path):
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    try:
        plate = create_flat_plate(chord=1, span=1, n_chordwise_panels=1, n_spanwise_panels=1)
        setup = VLMSetup(surfaces=(VLMSurfaceSetup(plate),), dtype="f64")
        source = VLMSolver(setup)
        source.generate_mesh()
        with h5py.File(tmp_path / "vlm.h5", "w") as archive:
            group = archive.create_group("vlm")
            write_vlm_restart(source, group)
            validate_vlm_restart(source, group)
            for changed in (
                replace(setup, density=1.1),
                replace(setup, boundary_response="responsive"),
                replace(setup, surfaces=(replace(setup.surfaces[0], group_id=1),)),
                replace(setup, surfaces=(replace(setup.surfaces[0], translation=(0.1, 0, 0)),)),
            ):
                target = VLMSolver(changed)
                target.generate_mesh()
                with pytest.raises(ValueError, match="configuration"):
                    validate_vlm_restart(target, group)
    finally:
        ti.reset()
