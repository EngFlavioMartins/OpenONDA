"""Device-free admission contracts for the isolated native GBD profiler."""

import copy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def asset():
    directory = (
        Path(__file__).resolve().parents[2]
        / "tests/support/cylinder"
    )
    sys.path.insert(0, str(directory))
    spec = spec_from_file_location(
        "diffusion_checkpoint_profile_test", directory / "profile_diffusion_checkpoint.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint():
    return {
        "manifest": {
            "config": {
                "vpm": {
                    "precision": "f32",
                    "axisymmetric_no_swirl_axis": None,
                    "turbulence": {"flow_model": "INVISCID"},
                    "induction": {"method": "SLIP_SLAB"},
                    "viscous": {
                        "scheme": "GBD",
                        "gbd_grid_spacing": 0.04,
                        "kinematic_viscosity": 0.02,
                    },
                }
            }
        },
        "fvm_manifest": {"kinematic_viscosity": 0.02},
    }


def test_config_rejects_unqualified_diffusion_contracts():
    module = asset()
    accepted = checkpoint()
    assert module.validate_gbd_configuration(accepted)["precision"] == "f32"
    for group, name, value in (
        ("viscous", "scheme", "DVH"),
        ("viscous", "gbd_grid_spacing", 0),
        ("viscous", "kinematic_viscosity", 0.01),
        ("turbulence", "flow_model", "LES"),
        ("induction", "method", "PLANAR"),
    ):
        changed = copy.deepcopy(accepted)
        changed["manifest"]["config"]["vpm"][group][name] = value
        with pytest.raises(ValueError):
            module.validate_gbd_configuration(changed)


def test_wall_adapter_preserves_rank_face_and_patch_order():
    module = asset()
    mesh = {
        "n_faces": 4,
        "vertex_position": np.arange(18).reshape(6, 3),
        "faces": [
            np.array([0, 1, 2]),
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            np.array([3, 4, 5]),
        ],
        "boundary": [
            {"name": "outer", "start_face": 0, "n_faces": 1},
            {"name": "wall", "start_face": 1, "n_faces": 3},
        ],
    }
    setup = SimpleNamespace(boundaries=[SimpleNamespace(name="wall", mesh_type="wall")])
    seen = []

    class Interface:
        @staticmethod
        def get_wall_surface_triangles(owner):
            seen.append(owner)
            result = []
            for patch in owner.boundaries:
                if patch["name"] == "wall":
                    for index in range(patch["start_face"], patch["start_face"] + patch["n_faces"]):
                        result.append(
                            owner.mesh_data["vertex_position"][owner.mesh_data["faces"][index]]
                        )
            return np.asarray(result).reshape(-1, 3, 3)

    ranks = [{"global_face_id": np.array([0, 2])}, {"global_face_id": np.array([1, 3])}]
    result = module.rank_wall_triangles(mesh, ranks, setup, Interface)
    expected = np.array([mesh["vertex_position"][mesh["faces"][index]] for index in (2, 1, 3)])
    np.testing.assert_array_equal(result, expected)
    assert seen[0].parallel is None
    with pytest.raises(ValueError, match="missing or multiply owned"):
        module.rank_wall_triangles(mesh, ranks + ranks[:1], setup, Interface)
    with pytest.raises(ValueError, match="unique ascending"):
        module.rank_wall_triangles(mesh, [{"global_face_id": np.array([2, 1])}], setup, Interface)


def test_result_digest_tracks_dtype_shape_and_values():
    module = asset()
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert module.array_digest(value) == module.array_digest(value.copy())
    assert module.array_digest(value) != module.array_digest(value.reshape(6))
    assert module.array_digest(value) != module.array_digest(value.astype(np.float64))
    summary = module.summarize_result({"position": value, "vortex_strength": value})
    assert summary["particles"] == 2
    np.testing.assert_array_equal(summary["vortex_strength_net"], [3, 5, 7])
    with pytest.raises(ValueError, match="Nonfinite"):
        module.summarize_result({"position": value, "vortex_strength": value * np.nan})


def test_cache_observation_is_read_only_and_json_serializable():
    import json

    module = asset()
    physics = SimpleNamespace(
        _body_mask_host=np.zeros((3, 4, 5), dtype=bool),
        _grid_shape=(7, 8, 9),
        _body_mask_cache_key=(123, ("wall",), (0.0, 0.0, 0.0), 0.04, 3, 4, 5, (-0.48, 0.48)),
        _body_geometry_revision=("wall",),
    )
    mask = physics._body_mask_host
    result = module.geometry_cache_state(physics)
    assert result["active_body_mask_shape"] == [3, 4, 5]
    assert result["allocated_grid_shape"] == [7, 8, 9]
    assert physics._body_mask_host is mask
    json.dumps(result)
