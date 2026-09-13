"""Legacy lagged checkpoints may migrate declarations only with an exact old hash."""

from dataclasses import replace

from _flat_plate_geometry import create_flat_plate
import h5py
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.solver.restart import (
    _identity_controls,
    _identity_digest,
    validate_vlm_restart,
    write_vlm_restart,
)
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver


def test_pre_declaration_identity_requires_exact_original_physics_and_geometry(tmp_path):
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    try:
        plate = create_flat_plate(chord=1, span=1, n_chordwise_panels=1, n_spanwise_panels=1)
        setup = VLMSetup(surfaces=(VLMSurfaceSetup(plate),), dtype="f64")
        source = VLMSolver(setup)
        source.generate_mesh()
        # Reconstruct the historical serializer, retaining every old physics key.
        controls = _identity_controls(source)
        for key in ("field_contract", "boundary_response", "surface_event_policy"):
            del controls[key]
        legacy_hash = _identity_digest(source, controls)
        output = {
            "logging_interval_steps": 1,
            "sample_surface_forces": True,
            "surface_sample_forces": [None],
        }
        with h5py.File(tmp_path / "legacy.h5", "w") as archive:
            group = archive.create_group("vlm")
            write_vlm_restart(source, group)
            group.attrs["version"] = 6
            group.attrs["identity"] = legacy_hash
            del group.attrs["physics_identity"]
            for key in ("area", "relative_velocity", "bound_relative_velocity"):
                del group[key]
            result = validate_vlm_restart(source, group, legacy_output_controls=output)
            assert result["kind"] == "legacy_lagged_field_declaration"
            assert result["source_identity"] == legacy_hash
            for changed in (
                replace(setup, density=1.1),
                replace(setup, boundary_response="responsive"),
                replace(setup, surfaces=(replace(setup.surfaces[0], group_id=1),)),
                replace(setup, surfaces=(replace(setup.surfaces[0], translation=(0.1, 0, 0)),)),
            ):
                target = VLMSolver(changed)
                target.generate_mesh()
                with pytest.raises(ValueError, match="configuration"):
                    validate_vlm_restart(target, group, legacy_output_controls=output)
            with pytest.raises(ValueError, match="configuration"):
                validate_vlm_restart(source, group)
    finally:
        ti.reset()
