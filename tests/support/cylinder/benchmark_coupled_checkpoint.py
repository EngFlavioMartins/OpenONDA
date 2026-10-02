"""Bounded native restart qualification in the ordinary tutorial directory.

This uses the unchanged local case factory, strict backup admission and normal
output reconciliation. Existing observations must be preserved before replay.
An immutable source root permits a control replay alongside implementation work.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import importlib.util
import json
from pathlib import Path
import sys
import time


_FENV_MODULE = "source.solvers.vpm.numerics._fenv"
_POLICY_PATH = "vpm.induction.gaussian_mesh_policy"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("an exact lowercase SHA-256 digest is required")
    return value


def _case_assets(case):
    return {
        Path(__file__).resolve(), case / "setup.py", case / "assets/sampling.py",
        Path(__file__).with_name("capture_induction_reuse.py"), Path(__file__).with_name("capture_interface_traces.py"),
        Path(__file__).with_name("profile_solver_components.py"),
        Path(__file__).with_name("capture_fft_failure.py"),
    }


def _install_fenv_extension(path):
    """One explicit optional compiled module, never a general import bypass."""
    if path is None:
        return None
    path = path.resolve(strict=True)
    if not path.is_file() or not any(path.name == "_fenv"+suffix for suffix in EXTENSION_SUFFIXES):
        raise ValueError("--fenv-extension must name a compiled _fenv extension")
    identity = {"module": _FENV_MODULE, "path": str(path), "sha256": _digest(path)}
    existing = sys.modules.get(_FENV_MODULE)
    if existing is not None:
        if not getattr(existing, "__file__", None) or Path(existing.__file__).resolve() != path:
            raise RuntimeError("another floating-environment extension is already imported")
        return identity
    spec = importlib.util.spec_from_file_location(_FENV_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for method in ("capture", "enter_default", "restore", "round_to_nearest"):
        if not callable(getattr(module, method, None)):
            raise RuntimeError("the selected _fenv extension lacks required capabilities")
    if _digest(path) != identity["sha256"]:
        raise RuntimeError("floating-environment extension changed during import")
    sys.modules[_FENV_MODULE] = module
    return identity


def _loaded_source_hashes(root, case, external_fenv=None):
    """Record the actual imported implementation, not only its directory name."""
    paths = _case_assets(case)
    for name, imported in tuple(sys.modules.items()):
        if (name in ("source", "openonda") or name.startswith(("source.", "openonda."))) and getattr(imported, "__file__", None):
            path = Path(imported.__file__).resolve()
            if not path.is_relative_to(root):
                if (external_fenv is None or name != _FENV_MODULE
                        or str(path) != external_fenv["path"]
                        or _digest(path) != external_fenv["sha256"]):
                    raise RuntimeError(f"Benchmark imported source outside its selected root: {path}")
            paths.add(path)
    return {str(path): _digest(path) for path in sorted(paths)}


def _source_hashes(root, case, external_fenv=None):
    """Include lazy numerical sources before any solver initialization."""
    paths = _case_assets(case)
    for package in (root / "source", root / "openonda"):
        if package.is_dir():
            for path in package.rglob("*"):
                if path.is_file() and (path.suffix in (".py", ".c", ".cpp", ".h")
                                      or any(path.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES)):
                    resolved = path.resolve()
                    if not resolved.is_relative_to(root):
                        raise RuntimeError("selected source inventory escapes its root: " + str(path))
                    paths.add(resolved)
    if external_fenv is not None:
        paths.add(Path(external_fenv["path"]))
    result = {str(path): _digest(path) for path in sorted(paths)}
    loaded = _loaded_source_hashes(root, case, external_fenv)
    if any(result.get(path) != value for path, value in loaded.items()):
        raise RuntimeError("an imported source is absent from the pre-initialization inventory")
    return result


def _with_gaussian_mesh(case):
    """Replace frozen case values; never modify the authored tutorial setup."""
    from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabPolicy
    from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction

    old = case.numerics.induction
    if type(old) is not SlipSlabInduction or old.physics is not None:
        raise ValueError("qualification mesh opt-in requires an unbound standard SlipSlabInduction")
    policy = GaussianSlabPolicy()
    if old.gaussian_mesh_policy is not None and old.gaussian_mesh_policy != policy:
        raise ValueError("qualification cannot overwrite an authored nondefault mesh policy")
    induction = SlipSlabInduction(old.base.build(), z_min=old.z_min, z_max=old.z_max,
        tail_tolerance=old.tail_tolerance, max_shells=old.max_shells,
        velocity_scale=old.velocity_scale, gradient_scale=old.gradient_scale,
        gaussian_mesh_policy=policy)
    return replace(case, numerics=replace(case.numerics, induction=induction))


def _mesh_restart_admission(checkpoint, current_numerics, expected_manifest_sha256=None):
    """Read-only evidence and ONE exact permission; native run rechecks all."""
    from source.coupler.backup import (
        BACKUP_FORMAT_VERSION,
        _resolve_artifact,
        artifact_digest,
        config_mapping_digest,
    )
    from source.solvers.vpm.config.fingerprint import numerical_configuration
    from source.solvers.vpm.config.restart_changes import (
        MISSING_CONFIGURATION_VALUE,
        admit_configuration_changes,
    )

    manifest_path = checkpoint / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha != _sha256(expected_manifest_sha256):
        raise ValueError("source coupled manifest SHA-256 does not match the explicit expectation")
    manifest = json.loads(manifest_bytes)
    if (not isinstance(manifest, dict) or manifest.get("format_version") != BACKUP_FORMAT_VERSION
            or manifest.get("kind") != "openonda.coupled_backup" or manifest.get("backend") != "fvm"):
        raise ValueError("unsupported coupled backup format for mesh qualification")
    config = manifest.get("config")
    if not isinstance(config, dict) or manifest.get("config_sha256") != config_mapping_digest(config):
        raise ValueError("source coupled configuration SHA-256 mismatch")
    artifacts, hashes = manifest.get("artifacts"), manifest.get("artifact_sha256")
    required = {"fvm", "vpm", "vpm_vtu", "vpm_boundary_condition"}
    if (not isinstance(artifacts, dict) or not isinstance(hashes, dict)
            or set(artifacts) != set(hashes) or not required <= set(artifacts)):
        raise ValueError("complete authenticated coupled artifacts required")
    authenticated = {}
    for name, relative in artifacts.items():
        path = _resolve_artifact(checkpoint, relative)
        if not path.exists() or artifact_digest(path) != _sha256(hashes[name]):
            raise ValueError("source coupled artifact SHA-256 mismatch: " + str(name))
        authenticated[name] = {"path": str(path), "sha256": hashes[name]}
    stored, current = config.get("vpm"), numerical_configuration(current_numerics)
    if not isinstance(stored, dict) or not isinstance(stored.get("induction"), dict):
        raise ValueError("stored VPM induction configuration required")
    canonical_policy = current["induction"].get("gaussian_mesh_policy")
    if not isinstance(canonical_policy, dict):
        raise ValueError("mesh opt-in is missing from the current numerical fingerprint")
    key = "induction.gaussian_mesh_policy"
    missing = "gaussian_mesh_policy" not in stored["induction"]
    if missing and expected_manifest_sha256 is None:
        raise ValueError("old-to-mesh transition requires --expected-manifest-sha256")
    allowed = (key,) if missing else ()
    expectations = {key: (MISSING_CONFIGURATION_VALUE, canonical_policy)} if missing else None
    changes = admit_configuration_changes(current, stored,
        allowed_config_differences=allowed, expected_config_differences=expectations)
    kwargs = {}
    if missing:
        kwargs = {"restart_allowed_config_differences": (_POLICY_PATH,),
                  "restart_expected_config_differences": {
                      _POLICY_PATH: (MISSING_CONFIGURATION_VALUE, canonical_policy)}}
    evidence = {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
        "expected_manifest_sha256": expected_manifest_sha256,
        "source_configuration_sha256": manifest["config_sha256"],
        "source_vpm_configuration_sha256": config_mapping_digest(stored),
        "current_vpm_configuration_sha256": config_mapping_digest(current),
        "source_artifacts": authenticated,
        "permissions": [{**change, "path": "vpm."+change["path"]} for change in changes],
        "native_admission": "public run performs unchanged full coupled and VPM preflight before restoration"}
    if _digest(manifest_path) != manifest_sha:
        raise RuntimeError("source manifest changed during read-only admission")
    return kwargs, evidence


def _assert_checkpoint_evidence(evidence):
    """Reject changed committed inputs; never rewrite manifests or artifacts."""
    from source.coupler.backup import artifact_digest

    if _digest(Path(evidence["manifest_path"])) != evidence["manifest_sha256"]:
        raise RuntimeError("authenticated source manifest changed")
    for artifact in evidence["source_artifacts"].values():
        if artifact_digest(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError("authenticated source artifact changed: " + artifact["path"])


def _collective_read(comm, action):
    result, error = None, None
    if comm is None or comm.Get_rank() == 0:
        try:
            result = action()
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    if comm is not None and comm.Get_size() > 1:
        error, result = comm.bcast((error, result), root=0)
    if error is not None:
        raise ValueError("Qualification read-only admission failed: " + error)
    return result


def _induction_reuse_statistics(coupler):
    """Copy operational counters without touching the certified backend."""
    stage_rhs = coupler.vpm_solver.stage_rhs
    snapshot = getattr(stage_rhs, "induction_reuse_statistics", None)
    if snapshot is not None:
        return asdict(snapshot)
    cache = getattr(stage_rhs, "induction_reuse", None)
    return None if cache is None else asdict(cache.statistics)


def _induction_geometry_statistics(coupler):
    """Read host-side scratch counters without wrapping certified operators."""
    induction = coupler.vpm_solver.induction
    diagnostics = getattr(getattr(induction, "base", induction), "diagnostics", None)
    if diagnostics is None or not hasattr(diagnostics, "image_target_geometry_restores"):
        return None
    return {
        name: getattr(diagnostics, name)
        for name in (
            "image_target_geometry_builds", "image_target_geometry_restores",
            "image_geometry_allocations", "image_geometry_fallback_scopes",
            "image_geometry_bytes", "peak_image_geometry_bytes",
        )
    }


def _collective_output_preflight(comm, case, report_path, trace_prefix, steps):
    """Check shared output admission once, before a fast master can publish.

    Rechecking existence independently on each rank races the master's first
    report/trace write. Every rank consumes the same root decision instead.
    """
    error = None
    if comm is None or comm.Get_rank() == 0:
        try:
            if steps < 1 or report_path.exists():
                raise ValueError("Require a positive step cap and a new report path")
            if report_path.parent != case / "solution":
                raise ValueError(
                    "Qualification reports must stay in the ordinary solution directory"
                )
            if trace_prefix is not None:
                if trace_prefix.parent != case / "solution":
                    raise ValueError(
                        "Interface traces must stay in the ordinary solution directory"
                    )
                if any(trace_prefix.parent.glob(trace_prefix.name + "-step*.npz")):
                    raise FileExistsError("Interface trace prefix already has recorded exchanges")
        except Exception as exc:
            error = repr(exc)
    if comm is not None and comm.Get_size() > 1:
        error = comm.bcast(error, root=0)
    if error is not None:
        raise ValueError("Qualification output admission failed: " + error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gaussian-mesh", action="store_true",
        help="Explicit qualification-only GaussianSlabPolicy opt-in; authored case is unchanged")
    parser.add_argument("--expected-manifest-sha256",
        help="Exact source manifest SHA-256, required for old-to-mesh structural transition")
    parser.add_argument("--fenv-extension", type=Path,
        help="Optional explicit compiled _fenv module; no other external source imports allowed")
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--fft-failure-prefix", type=Path,
                        help="Optional new solution/ prefix for read-only FFT traceback/input evidence")
    parser.add_argument("--ordinary-initialization", action="store_true",
                        help="Match the ordinary launcher initialization before strict native loading")
    parser.add_argument(
        "--no-gbd-detail", action="store_true",
        help="Keep physics method identities unchanged; retain outer stepper phase timers",
    )
    parser.add_argument(
        "--trace-induction-reuse", action="store_true",
        help="Record per-request cache eligibility without extra device reads or backend patches",
    )
    parser.add_argument(
        "--disable-interface-prediction", action="store_true",
        help="Qualification control: use the original raw interface initial guess every exchange",
    )
    parser.add_argument(
        "--interface-trace-prefix",
        type=Path,
        help="Optional new solution/ prefix for per-exchange copied interface trace NPZ files",
    )
    args = parser.parse_args()
    if args.expected_manifest_sha256 is not None:
        if not args.gaussian_mesh:
            parser.error("--expected-manifest-sha256 requires --gaussian-mesh")
        _sha256(args.expected_manifest_sha256)
    root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    report_path = args.report.resolve()
    case = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
    trace_prefix = args.interface_trace_prefix.resolve() if args.interface_trace_prefix else None
    from mpi4py import MPI

    _collective_output_preflight(MPI.COMM_WORLD, case, report_path, trace_prefix, args.steps)
    if args.fft_failure_prefix is not None:
        failure_prefix = args.fft_failure_prefix.resolve()
        def admit_failure_evidence():
            if (failure_prefix.parent != case / "solution"
                    or any(failure_prefix.with_suffix(suffix).exists() for suffix in (".json", ".npz"))):
                raise ValueError("new ordinary solution/ FFT evidence paths required")
        _collective_read(MPI.COMM_WORLD, admit_failure_evidence)
    sys.path.insert(0, str(root))
    external_fenv = _install_fenv_extension(args.fenv_extension)
    import source.coupler.solver as driver_module
    from source.coupler import create_coupler

    spec = importlib.util.spec_from_file_location("qualification_case", case / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setup, particles, coupling, _ = module.build_case()
    restart_keywords, restart_evidence = {}, None
    if args.gaussian_mesh:
        particles = _with_gaussian_mesh(particles)
        restart_keywords, restart_evidence = _collective_read(MPI.COMM_WORLD,
            lambda: _mesh_restart_admission(checkpoint, particles.numerics,
                                            args.expected_manifest_sha256))
    mesh = case / "solution/fvm/mesh.npz"
    if not mesh.is_file():
        raise FileNotFoundError("Qualification requires the existing ordinary case mesh")
    record = {
        "source_root": str(root),
        "checkpoint": str(checkpoint),
        "max_coupling_steps": args.steps,
        "exchanges": [],
        "status": "initializing",
        "component_timings": {},
        "component_timings_are_nested_not_additive": True,
        "interface_traces": [],
        "qualification_controls": {
            "interface_prediction_disabled": args.disable_interface_prediction,
            "gbd_detail_disabled": args.no_gbd_detail,
            "gaussian_mesh_explicit_opt_in": args.gaussian_mesh,
            "ordinary_initialization": args.ordinary_initialization,
        },
        "restart_admission": restart_evidence,
        "external_fenv": external_fenv,
    }
    started = time.perf_counter()
    # Snapshot the complete selected implementation now, including numerical
    # modules imported only after lazy solver initialization or first sampling.
    source_inventory = _source_hashes(root, case, external_fenv)
    record["source_hashes_before_initialization"] = source_inventory
    original_advance = driver_module.FVMVPMCoupler._advance_vpm
    original_record = driver_module.record_step
    exchange_started = None
    component_start = {}

    def advance(owner, *positional, **keywords):
        nonlocal exchange_started, component_start
        exchange_started = time.perf_counter()
        component_start = {name: dict(values) for name, values in record["component_timings"].items()}
        return original_advance(owner, *positional, **keywords)

    def report_step(owner, step, clock, *positional, **keywords):
        result = original_record(owner, step, clock, *positional, **keywords)
        if owner._is_master:
            record["exchanges"].append(
                {
                    "step": step,
                    "time": clock,
                    "wall_seconds_through_scheduled_backup": time.perf_counter() - exchange_started,
                    "induction_reuse": _induction_reuse_statistics(owner),
                    "induction_geometry": _induction_geometry_statistics(owner),
                    "body_geometry": getattr(
                        owner.vpm_solver.physics, "body_geometry_cache_diagnostics", None
                    ),
                    # Nested timers remain non-additive; these deltas avoid
                    # attributing cold compilation from earlier exchanges to
                    # the warm accepted step being measured.
                    "component_timings": {
                        name: {
                            key: value - component_start.get(name, {}).get(key, 0)
                            for key, value in values.items()
                        }
                        for name, values in record["component_timings"].items()
                    },
                }
            )
            report_path.write_text(json.dumps(record, indent=2) + "\n")
        return result

    driver_module.FVMVPMCoupler._advance_vpm = advance
    driver_module.record_step = report_step
    with create_coupler(setup, particles, coupling, mesh=mesh, case_dir=case) as solver:
        if args.disable_interface_prediction:
            solver.interface_predictor.enabled = False
        loaded_at_construction = _loaded_source_hashes(root, case, external_fenv)
        if solver._is_master:
            record["construction_seconds"] = time.perf_counter() - started
            record["source_hashes_at_construction"] = loaded_at_construction
            record["status"] = "strict-restart"
            report_path.write_text(json.dumps(record, indent=2) + "\n")
        try:
            if args.ordinary_initialization:
                from openonda.cylinder_campaign import initialize_cylinder_perturbation

                solver.initialize()
                induction = particles.numerics.induction
                initialize_cylinder_perturbation(solver.fvm_solver, induction.z_max-induction.z_min)
            if args.profile_components or args.trace_induction_reuse:
                solver.initialize()
            if args.profile_components:
                from profile_solver_components import profile_components

                # The coupled factory is lazy: ownership and vpm_solver are
                # bound collectively by initialize(), not __enter__(). The
                # normal run() performs the same idempotent initialization.
                component_context = profile_components(
                    solver, record["component_timings"], gbd_detail=not args.no_gbd_detail
                )
            else:
                component_context = nullcontext()
            if trace_prefix is not None:
                from capture_interface_traces import capture_interface_traces

                trace_context = capture_interface_traces(
                    solver, trace_prefix, record["interface_traces"], max_exchanges=args.steps
                )
            else:
                trace_context = nullcontext()
            if args.trace_induction_reuse:
                from capture_induction_reuse import capture_induction_reuse_requests

                reuse_context = capture_induction_reuse_requests(
                    solver, record.setdefault("induction_reuse_requests", [])
                )
            else:
                reuse_context = nullcontext()
            with component_context, trace_context, reuse_context:
                if restart_evidence is not None:
                    _collective_read(MPI.COMM_WORLD, lambda: _assert_checkpoint_evidence(restart_evidence))
                final = solver.run(
                    restart_from=checkpoint, max_coupling_steps=args.steps, backup_at_stop=True,
                    **restart_keywords,
                )
            if solver._is_master:
                record["source_hashes_at_completion"] = _loaded_source_hashes(root, case, external_fenv)
                completed_inventory = _source_hashes(root, case, external_fenv)
                record["source_inventory_at_completion"] = completed_inventory
                changed = [
                    name for name in sorted(set(source_inventory) | set(completed_inventory))
                    if source_inventory.get(name) != completed_inventory.get(name)
                ]
                record["loaded_source_files_changed_during_run"] = changed
                if changed:
                    raise RuntimeError("Qualification source changed during execution: " + repr(changed))
                if restart_evidence is not None:
                    _assert_checkpoint_evidence(restart_evidence)
                    record["source_checkpoint_unchanged"] = True
                record.update(
                    status="complete",
                    final_step=final,
                    induction_reuse=_induction_reuse_statistics(solver),
                    induction_geometry=_induction_geometry_statistics(solver),
                    body_geometry=getattr(
                        solver.vpm_solver.physics, "body_geometry_cache_diagnostics", None
                    ),
                    wall_seconds_including_initialization_and_final_backup=time.perf_counter()
                    - started,
                )
                report_path.write_text(json.dumps(record, indent=2) + "\n")
        except BaseException as exc:
            if solver._is_master:
                if args.fft_failure_prefix is not None:
                    try:
                        from capture_fft_failure import save_fft_failure
                        record["fft_failure_evidence"] = save_fft_failure(exc, failure_prefix)
                    except BaseException as capture_error:
                        record["fft_failure_evidence"] = {"captured": False, "error": repr(capture_error)}
                        exc.add_note("FFT evidence capture failed: " + repr(capture_error))
                record.update(status="failed", error=repr(exc))
                report_path.write_text(json.dumps(record, indent=2) + "\n")
            raise


if __name__ == "__main__":
    main()
