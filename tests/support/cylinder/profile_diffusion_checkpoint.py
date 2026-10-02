"""Read-only native-input GBD microbenchmark; never advances a saved solution.

The inputs are an accepted checkpoint, NOT the post-RK cloud on which the next
production diffusion acts. No induction, FVM solver, native restart bypass,
sampler, or solver output manager is constructed. Mesh topology, rank-ordered
wall triangles, lattice phase and GBD settings are verified against the native
checkpoint before private device workspaces are allocated.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from compare_coupled_checkpoints import artifact_digest, load_checkpoint


def runtime_environment():
    from threadpoolctl import threadpool_info
    from vtkmodules.vtkCommonCore import vtkSMPTools, vtkVersion

    return {
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VTK_SMP_BACKEND_IN_USE",
                "VTK_SMP_MAX_THREADS",
            )
        },
        "loaded_threadpools": threadpool_info(),
        "vtk_version": vtkVersion.GetVTKVersion(),
        "vtk_smp_backend": vtkSMPTools.GetBackend(),
        "vtk_smp_estimated_threads": vtkSMPTools.GetEstimatedNumberOfThreads(),
    }


def source_identity(root):
    names = (
        "source/solvers/vpm/physics/base.py",
        "source/solvers/vpm/physics/engine.py",
        "source/solvers/vpm/physics/diffusion/grid.py",
        "source/solvers/vpm/particles/container.py",
        "source/coupler/geometry.py",
        "source/solvers/fvm/coupling/coupler_interface.py",
        "source/solvers/fvm/mesh/geometry.py",
    )
    return {name: artifact_digest(root / name) for name in names}


def validate_gbd_configuration(checkpoint):
    config = checkpoint["manifest"]["config"]["vpm"]
    viscous = config["viscous"]
    if viscous["scheme"] != "GBD":
        raise ValueError("This isolated operator profile requires native GBD")
    if config["precision"] not in {"f32", "f64"}:
        raise ValueError("Unsupported native particle precision")
    if config.get("axisymmetric_no_swirl_axis") is not None:
        raise ValueError("Axisymmetric orbit handling needs a separate qualified harness")
    if viscous["gbd_grid_spacing"] is None or viscous["gbd_grid_spacing"] <= 0:
        raise ValueError("GBD requires a positive native grid spacing")
    if config["turbulence"]["flow_model"] == "LES":
        raise ValueError("This harness does not reconstruct pre-diffusion LES updates")
    if config["induction"]["method"] != "SLIP_SLAB":
        raise ValueError("This harness currently admits only the native slip-slab contract")
    if not np.isclose(
        float(viscous["kinematic_viscosity"]),
        float(checkpoint["fvm_manifest"]["kinematic_viscosity"]),
        rtol=0,
        atol=0,
    ):
        raise ValueError("Native FVM and GBD viscosities disagree")
    return config


def rank_wall_triangles(mesh, rank_states, setup, interface_mixin):
    """Invoke the real getter in native MPI rank/face order, without MPI."""
    walls = {entry.name for entry in setup.boundaries if entry.mesh_type == "wall"}
    expected_ids = np.concatenate(
        [
            np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
            for patch in mesh["boundary"]
            if patch["name"] in walls
        ]
    )
    if not len(expected_ids):
        raise ValueError("Native body-fitted wall geometry is required")
    result, gathered_ids = [], []
    for state in rank_states:
        ids = np.asarray(state["global_face_id"], dtype=np.int64)
        if ids.ndim != 1 or np.any(np.diff(ids) <= 0):
            raise ValueError("Rank face IDs are not unique ascending native IDs")
        if np.any(ids < 0) or np.any(ids >= mesh["n_faces"]):
            raise ValueError("Rank face IDs lie outside the admitted mesh")
        boundaries = []
        for patch in mesh["boundary"]:
            first = int(patch["start_face"])
            last = first + int(patch["n_faces"])
            start, end = np.searchsorted(ids, [first, last])
            if end > start:
                boundaries.append(dict(patch, start_face=int(start), n_faces=int(end - start)))
                if patch["name"] in walls:
                    gathered_ids.extend(ids[start:end].tolist())
        adapter = SimpleNamespace(
            setup=setup,
            boundaries=boundaries,
            mesh_data={
                "vertex_position": mesh["vertex_position"],
                "faces": [mesh["faces"][index] for index in ids],
            },
            parallel=None,
            _root_view=lambda values, **_: values,
        )
        result.append(interface_mixin.get_wall_surface_triangles(adapter))
    if not np.array_equal(np.sort(gathered_ids), np.sort(expected_ids)):
        raise ValueError("Partitioned wall faces are missing or multiply owned")
    return np.concatenate(result, axis=0)


def load_inputs(root, checkpoint_path, mesh_path, case_setup):
    """Host-only admission; importing Taichi modules does not initialize them."""
    sys.path.insert(0, str(root))
    from source.coupler.geometry import SolidBoundary, TriangulatedWall
    from source.solvers.fvm.coupling.coupler_interface import CouplerInterfaceMixin
    from source.solvers.fvm.io.backup import config_hash, mesh_hash
    from source.solvers.fvm.io.mesh_storage import load_native_mesh
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
    from source.solvers.fvm.factory import _runtime_setup

    checkpoint = load_checkpoint(checkpoint_path)
    config = validate_gbd_configuration(checkpoint)
    spec = importlib.util.spec_from_file_location("gbd_profile_case_setup", case_setup)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fvm_setup, _, _, _ = module.build_case()
    # The ordinary factory resolves e.g. four requested cores to partitioned
    # PETSc before recording the numerical fingerprint. Use that pure resolver,
    # not an ad-hoc change to the hash or a constructed solver.
    fvm_setup = _runtime_setup(fvm_setup)
    if config_hash(fvm_setup) != checkpoint["fvm_manifest"]["config_hash"]:
        raise ValueError("Case FVM setup does not match native numerical configuration")
    mesh_digest = artifact_digest(mesh_path)
    mesh = load_native_mesh(mesh_path)
    if mesh_hash(mesh) != checkpoint["fvm_manifest"]["mesh_hash"]:
        raise ValueError("Native mesh topology fingerprint mismatch")
    # Reuse native face/cell geometry, without allocating FVM solution fields.
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    wall_names = {entry.name for entry in fvm_setup.boundaries if entry.mesh_type == "wall"}
    outer_faces = np.concatenate(
        [
            geometry["face_centre"][patch["start_face"] : patch["start_face"] + patch["n_faces"]]
            for patch in mesh["boundary"]
            if patch["name"] not in wall_names
        ]
    )
    bounds = np.stack((outer_faces.min(axis=0), outer_faces.max(axis=0)), axis=1).reshape(6)
    triangles = rank_wall_triangles(mesh, checkpoint["ranks"], fvm_setup, CouplerInterfaceMixin)
    wall = TriangulatedWall(triangles, bounds)
    revisions = checkpoint["manifest"]["config"]["solid_geometry"]["wall_revisions"]
    if revisions != [wall.revision]:
        raise ValueError("Rank-ordered wall geometry revision differs from native checkpoint")
    boundary = SolidBoundary((wall,))
    anchor = geometry["cell_centre"][: mesh["n_cells"]].min(axis=0)
    anchor[2] = config["induction"]["z_min"] + 0.5 * config["viscous"]["particle_spacing"]
    saved_anchor = np.asarray(checkpoint["manifest"]["config"]["transfer_lattice"]["anchor"])
    if not np.array_equal(anchor, saved_anchor):
        raise ValueError("Native geometry-derived lattice phase differs from saved anchor")
    for name, imported in tuple(sys.modules.items()):
        if name.startswith(("source.", "openonda.")) and getattr(imported, "__file__", None):
            if not Path(imported.__file__).resolve().is_relative_to(root):
                raise ValueError(f"Mixed numerical source roots: {imported.__file__}")
    if artifact_digest(mesh_path) != mesh_digest:
        raise ValueError("Mesh changed during host preflight")
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "artifact_sha256": checkpoint["manifest"]["artifact_sha256"],
        "config_sha256": checkpoint["manifest"]["config_sha256"],
        "mesh": str(mesh_path),
        "mesh_sha256": mesh_digest,
        "mesh_topology_sha256": checkpoint["fvm_manifest"]["mesh_hash"],
        "setup": str(case_setup),
        "setup_sha256": artifact_digest(case_setup),
        "fvm_config_hash": checkpoint["fvm_manifest"]["config_hash"],
        "fvm_bounds": bounds.tolist(),
        "wall_revision": wall.revision,
        "wall_triangle_count": len(triangles),
        "lattice_anchor": anchor.tolist(),
        "particles": len(checkpoint["particle"]["position"]),
        "time": float(checkpoint["manifest"]["time"]),
        "configuration": config,
    }
    return checkpoint, boundary, anchor, metadata


def array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def summarize_result(result):
    if result is None or not len(result["position"]):
        raise ValueError("Unexpected empty native-input diffusion result")
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("Nonfinite diffusion output")
    strength = np.asarray(result["vortex_strength"], dtype=np.float64)
    return {
        "particles": len(result["position"]),
        "field_sha256": {name: array_digest(value) for name, value in result.items()},
        "vortex_strength_l1": float(np.linalg.norm(strength, axis=1).sum()),
        "vortex_strength_net": strength.sum(axis=0).tolist(),
    }


def geometry_cache_state(physics):
    """Read cache identity without classifying points or transferring grid data."""
    mask = physics._body_mask_host
    return {
        "allocated_grid_shape": list(physics._grid_shape),
        "active_body_mask_shape": None if mask is None else list(mask.shape),
        "body_mask_key": physics._body_mask_cache_key,
        "body_geometry_revision": physics._body_geometry_revision,
    }


def run_profile(checkpoint, boundary, anchor, *, repeats, rebuild_geometry_cache=False):
    import taichi as ti

    from source.solvers.vpm.particles.container import Particles
    from source.solvers.vpm.physics.engine import PhysicsEngine
    if __package__:
        from .profile_solver_components import profile_grid_diffusion
    else:
        from profile_solver_components import profile_grid_diffusion

    config = checkpoint["manifest"]["config"]["vpm"]
    viscous = config["viscous"]
    arrays = checkpoint["particle"]
    dtype = ti.f32 if config["precision"] == "f32" else ti.f64
    ti.init(arch=ti.cuda, default_fp=dtype, cpu_max_num_threads=2, offline_cache=False)
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("GPU qualification requires CUDA, not a silent fallback")
    started = time.perf_counter()
    physics = PhysicsEngine(
        particle_kernel=config["particle_kernel"],
        max_n_particles=config["max_n_particles"],
        accumulator_dtype=dtype,
    )
    particles = Particles(config["max_n_particles"], float_dtype=config["precision"])
    copied = (
        "position",
        "velocity",
        "vortex_strength",
        "core_radius",
        "particle_volume",
        "kinematic_viscosity",
        "eddy_viscosity",
        "group_id",
        "zone_id",
    )
    particles.replace_from_numpy(**{name: arrays[name] for name in copied})
    for name in ("vorticity", "effective_viscosity"):
        particles.set_field(name, arrays[name])
    physics.core_radius_ratio = float(viscous["core_radius_ratio"])
    physics._slip_slab_bounds = (config["induction"]["z_min"], config["induction"]["z_max"])
    physics.require_fixed_grid_allocation(True)
    physics.configure_max_grid_extent(
        config["domain_bounds"], viscous["gbd_grid_spacing"], viscous["gbd_domain_padding"]
    )
    physics.configure_body_classifier(
        lambda points: boundary.contains(points, include_boundary=False),
        revision=boundary.revision,
        query_bounds=boundary.bounds,
        blocks_segments=boundary.blocks_segments,
    )
    physics.configure_grid_lattice_anchor(anchor, viscous["gbd_grid_spacing"])
    ti.sync()
    report = {
        "allocation_and_upload_seconds": time.perf_counter() - started,
        "grid_shape": list(physics._grid_shape),
        "fixed_grid_min": physics._fixed_grid_min.tolist(),
        "device": "CUDA",
        "measurements": [],
    }
    result = None
    for index in range(repeats + 1):
        # The operator returns new arrays; they are NEVER installed as the next
        # input. Every cold/warm call receives the same private native fields.
        measured = {}
        # This option changes only disposable private geometry-cache validity,
        # never input fields or geometry. It separates JIT warm-up from the cost
        # of rebuilding a mask after active-grid bounds change in a real wake.
        if index and rebuild_geometry_cache:
            physics._body_mask_cache_key = None
        cache_before = geometry_cache_state(physics)
        ti.sync()
        started = time.perf_counter()
        with profile_grid_diffusion(physics, measured):
            result = physics.gbd_diffusion(
                particles,
                time_step_size=config["time_step_size"],
                particle_spacing=viscous["gbd_grid_spacing"],
                kinematic_viscosity=viscous["kinematic_viscosity"],
                domain_padding=viscous["gbd_domain_padding"],
                regen_threshold=viscous["gbd_threshold"],
                regen_threshold_mode=viscous["gbd_threshold_mode"],
                effective_viscosity=None,
                remeshing_kernel=viscous["gbd_remeshing_kernel"],
            )
        ti.sync()
        elapsed = time.perf_counter() - started
        cache_after = geometry_cache_state(physics)
        for name in copied + ("vorticity", "effective_viscosity"):
            values = getattr(particles, name).to_numpy()[: len(arrays["position"])]
            if not np.array_equal(values, arrays[name]):
                raise RuntimeError(f"Private diffusion mutated input particle field: {name}")
        report["measurements"].append(
            {
                "kind": "cold_operator" if index == 0 else "warm_operator",
                "repeat": index,
                "seconds": elapsed,
                "component_timings": measured,
                "component_timings_are_nested_not_additive": True,
                "geometry_cache_before": cache_before,
                "geometry_cache_after": cache_after,
                "geometry_cache_invalidated_for_this_repeat": bool(
                    index and rebuild_geometry_cache
                ),
                "body_mask_cache_hit_observed": (
                    cache_after["body_mask_key"] is not None
                    and cache_before["body_mask_key"] == cache_after["body_mask_key"]
                    and measured.get("gbd._prepare_body_links", {}).get("calls", 0) == 0
                ),
                "input_fields_unchanged": True,
                "result": summarize_result(result),
                "moment_recovery": physics._last_gbd_moment_recovery,
                "wall_transfer": physics._last_gbd_wall_transfer,
                "diffusion_substeps": physics._last_gbd_diffusion_substeps,
            }
        )
    return report, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Native coupled backup directory"
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--case-setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New ordinary solution prefix")
    parser.add_argument("--repeats", type=int, default=2, help="Warm repeats after one cold call")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--rebuild-geometry-cache-each-warm",
        action="store_true",
        help="Invalidate only the private body-mask key before warm calls to measure warm cache rebuilding",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Require at least one warm repeat")
    root, checkpoint_path, mesh_path, setup_path = (
        path.resolve() for path in (args.source_root, args.checkpoint, args.mesh, args.case_setup)
    )
    output = args.output.resolve()
    ordinary_solution = (Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow") / "solution"
    if output.parent != ordinary_solution or any(
        output.with_suffix(suffix).exists() for suffix in (".json", ".npz")
    ):
        raise ValueError("Require an unused prefix in the ordinary case solution directory")
    source_hashes = source_identity(root)
    started = time.perf_counter()
    checkpoint, boundary, anchor, metadata = load_inputs(
        root, checkpoint_path, mesh_path, setup_path
    )
    report = {
        "status": "host_preflight_complete",
        "scope": "Isolated GBD at accepted native checkpoint; NOT post-RK diffusion or solution continuation",
        "source_root": str(root),
        "source_hashes": source_hashes,
        "input": metadata,
        "host_preflight_seconds": time.perf_counter() - started,
        "gpu_initialized": False,
        "production_state_modified": False,
        "runtime_environment": runtime_environment(),
        "cold_warm_note": "Warm operator workspaces and fixed-geometry caches are reused; input cloud is identical on every call. Fine timers synchronize kernels and are diagnostic, not uninstrumented production timings.",
    }
    with output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2)
    if args.preflight_only:
        return
    try:
        report["gpu_initialized"] = None  # unknown if initialization raises
        measurements, result = run_profile(
            checkpoint,
            boundary,
            anchor,
            repeats=args.repeats,
            rebuild_geometry_cache=args.rebuild_geometry_cache_each_warm,
        )
        if source_identity(root) != source_hashes:
            raise RuntimeError("Numerical source changed during profiling")
        if artifact_digest(mesh_path) != metadata["mesh_sha256"]:
            raise RuntimeError("Mesh changed during profiling")
        if artifact_digest(setup_path) != metadata["setup_sha256"]:
            raise RuntimeError("Setup changed during profiling")
        for name, expected in metadata["artifact_sha256"].items():
            path = checkpoint_path / checkpoint["manifest"]["artifacts"][name]
            if artifact_digest(path) != expected:
                raise RuntimeError(f"Native input artifact changed during profiling: {name}")
        with output.with_suffix(".npz").open("xb") as stream:
            np.savez_compressed(stream, **result)
        report.update(measurements, status="complete", gpu_initialized=True)
        report["result_npz_sha256"] = artifact_digest(output.with_suffix(".npz"))
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
