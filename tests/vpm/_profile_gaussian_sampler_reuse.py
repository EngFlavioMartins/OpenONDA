"""Paired read-only sampler-coordinate replay on one authenticated checkpoint.

This measures the coherent particle-induced u/J request sequence, not complete
output dispatch: CSV/VTK publication, freestream and body callbacks are absent.
Every run includes a fresh session, all owner replacements and final close.
The old-box control changes only owner-domain admission at the same location
as the new check. Exact source identity, role and per-query certificates still
run inside the identical session implementation.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_identity(value):
    array = np.ascontiguousarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def difference(current, reference):
    if current.shape != reference.shape or not (np.isfinite(current).all()
                                                and np.isfinite(reference).all()):
        raise ValueError("finite identical field shapes required")
    delta = current.astype(np.float64)-reference.astype(np.float64)
    norm = float(np.linalg.norm(reference.astype(np.float64)))
    return {"relative_l2": float(np.linalg.norm(delta))/norm if norm else None,
            "max_component": float(np.abs(delta).max(initial=0)),
            "max_point_norm": float(np.linalg.norm(delta.reshape(len(delta), -1), axis=1).max(initial=0)),
            "bitwise_equal": current.dtype == reference.dtype and current.tobytes() == reference.tobytes()}


class InitialQueryBoxOwner:
    """Qualification-only proxy reproducing the former owner-domain predicate.

    Session source and role admission happen BEFORE this method. Closing the
    proxy closes the real owner. No cached fields, source state or certificates
    are fabricated; a false predicate goes through the ordinary close/rebuild.
    """

    def __init__(self, owner, initial_query):
        self.owner = owner
        query = np.asarray(initial_query)
        self.lower, self.upper = query.min(axis=0).copy(), query.max(axis=0).copy()

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def can_evaluate_targets(self, query):
        return bool(np.all(np.min(query, axis=0) >= self.lower)
                    and np.all(np.max(query, axis=0) <= self.upper))


@contextmanager
def owner_domain_policy(session_module, mode, ledger):
    """Instrument construction only; restore the factory even after failure."""
    if mode not in ("old_query_box", "logical_domain"):
        raise ValueError("unknown owner-domain policy")
    original = session_module._new_field_owner

    def construct(*args, **kwargs):
        owner = original(*args, **kwargs)
        ledger["successful_owner_builds"] += 1
        return InitialQueryBoxOwner(owner, args[3]) if mode == "old_query_box" else owner

    session_module._new_field_owner = construct
    try:
        yield
    finally:
        session_module._new_field_owner = original


def run_sequence(session_module, controls, policy, sources, requests, *, mode, synchronize):
    """Include session creation, every evaluate/transfer and final owner close."""
    ledger = {"successful_owner_builds": 0}
    arrays, records = {}, []
    session = None
    synchronize()
    started = perf_counter()
    with owner_domain_policy(session_module, mode, ledger):
        try:
            session = session_module.GaussianSlabFieldSession(**controls, policy=policy)
            constructed = perf_counter()
            for request in requests:
                before = perf_counter()
                u, j, diagnostic = session.evaluate(*sources, request["points"], source_only=True)
                synchronize()
                elapsed = perf_counter()-before
                records.append({"name": request["name"], "seconds": elapsed,
                                "diagnostics": diagnostic})
                arrays[request["name"]+"_velocity"] = u
                arrays[request["name"]+"_gradient"] = j
        finally:
            before_close = perf_counter()
            if session is not None:
                session.close()
            synchronize()
            close_seconds = perf_counter()-before_close
    return {"mode": mode, "complete_sequence_seconds": perf_counter()-started,
            "session_construction_seconds": constructed-started,
            "final_close_seconds": close_seconds, **ledger, "requests": records}, arrays


def prepare_requests(samplers, *, step, time, dt):
    result = []
    for sampler in samplers:
        if sampler.schedule.is_final_only or not sampler.schedule.is_due(step, time, dt):
            continue
        raw = np.asarray(sampler.line_points if hasattr(sampler, "line_points") else sampler.grid_points)
        # Production PhysicsBase uploads these sampler coordinates to f32
        # target fields. Preserve exactly those represented coordinates.
        points = np.array(raw, dtype=np.float32, order="C", copy=True)
        points.flags.writeable = False
        result.append({"name": sampler.file_name, "points": points,
                       "constructed_coordinates": array_identity(raw),
                       "backend_coordinates": array_identity(points)})
    if not result or len({item["name"] for item in result}) != len(result):
        raise ValueError("nonempty uniquely named configured request sequence required")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "samples", "prior-benchmark", "vpm-metadata",
                 "surface-frame", "fenv-extension", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--schedule-steps", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--admission-only", action="store_true",
                        help="authenticate checkpoint/request identities without importing CuPy or evaluating fields")
    args = parser.parse_args()
    if args.repeats not in (1, 2, 3) or any(step <= 0 for step in args.schedule_steps):
        raise ValueError("one to three repeats and positive explicit scheduling steps required")
    root = Path(__file__).resolve().parents[2]
    case = root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
    output, archive = args.output.resolve(), args.output.resolve().with_suffix(".npz")
    if output.parent != case/"solution" or output.suffix != ".json" or output.exists() or archive.exists():
        raise ValueError("new ordinary-solution evidence paths required")
    comparison_path = root / "tests/support/cylinder/compare_coupled_checkpoints.py"
    compare = load_module("sampler_reuse_checkpoint_admission", comparison_path)
    checkpoint = compare.load_checkpoint(args.checkpoint)
    manifest = checkpoint["manifest"]
    if checkpoint["manifest_sha256"] != args.expected_manifest_sha256:
        raise ValueError("checkpoint manifest differs from explicit expected identity")
    prior = json.loads(args.prior_benchmark.read_text())
    phase_path = case/"assets/sampling.py"
    for key in ("source_hashes_before_initialization", "source_hashes_at_completion"):
        if prior.get(key, {}).get(str(phase_path)) != digest(phase_path):
            raise ValueError("configured sampler factory differs from preserved native run")
    if (prior.get("status") != "complete"
            or prior.get("final_step") != manifest["vpm_step"]
            or not prior.get("exchanges")
            or prior["exchanges"][-1].get("step") != manifest["vpm_step"]
            or prior["exchanges"][-1].get("time") != manifest["time"]):
        raise ValueError("completed preserved native benchmark required")
    # Native modules are imported only after the requested compiled FENV bridge.
    extension = args.fenv_extension.resolve(strict=True)
    load_module("source.solvers.vpm.numerics._fenv", extension)
    phase = load_module("sampler_reuse_configured_phase", phase_path)
    from source.solvers.vpm.io.manifest import _sampler_identity
    from source.solvers.vpm.physics.induction.gaussian_mesh import session as session_module

    samplers = phase.vpm_samplers()
    frame = args.surface_frame.resolve(strict=True)
    if frame.parent != args.samples.resolve(strict=True):
        raise ValueError("surface geometry must come from the explicit preserved samples directory")
    saved_samplers = json.loads(args.vpm_metadata.read_text())["configuration"]["samplers"]["items"]
    if len(saved_samplers) != len(samplers):
        raise ValueError("saved and configured sampler counts differ")
    for sampler, saved in zip(samplers, saved_samplers, strict=True):
        if hasattr(sampler, "prepare_existing_vtk"):
            sampler.prepare_existing_vtk(frame)
        actual = _sampler_identity(sampler)
        # An equivalent later VTS frame changes provenance, not coordinates.
        # Compare every other stored construction/schedule field exactly.
        if ({k: v for k, v in actual.items() if k != "grid_resume_admission"}
                != {k: v for k, v in saved.items() if k != "grid_resume_admission"}):
            raise ValueError(f"saved sampler identity differs: {sampler.file_name}")
    config = manifest["config"]["vpm"]
    induction = config["induction"]
    policy = session_module.GaussianSlabPolicy()
    if (config["precision"] != "f32" or config["particle_kernel"] != "GAUSSIAN"
            or induction["method"] != "SLIP_SLAB"
            or induction["gaussian_mesh_policy"] != asdict(policy)):
        raise ValueError("qualification requires saved unchanged default Gaussian slab policy/f32")
    controls = {key: induction[key] for key in ("z_min", "z_max", "tail_tolerance", "max_shells",
                                               "velocity_scale", "gradient_scale")}
    controls["dtype"] = "float32"
    particles = checkpoint["particle"]
    sources = tuple(particles[name] for name in ("position", "vortex_strength", "core_radius"))
    source_identity = [array_identity(value) for value in sources]
    dt = float(config["time_step_size"])
    sequences = [(step, prepare_requests(samplers, step=step, time=step*dt, dt=dt))
                 for step in args.schedule_steps]
    input_paths = (args.prior_benchmark, args.vpm_metadata, frame, extension,
                   args.checkpoint/"manifest.json")
    input_hashes = {str(path.resolve()): digest(path) for path in input_paths}
    paths = set((root/"source").rglob("*.py")) | set((root/"openonda").rglob("*.py"))
    paths.update((phase_path, comparison_path, Path(__file__).resolve(), extension,
                  root/"source/solvers/vpm/numerics/_fenv.c"))
    source_hashes = {str(path): digest(path) for path in sorted(paths)}
    report = {"status": "running", "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
              "source_step": manifest["vpm_step"], "source_time": manifest["time"],
              "configuration_sha256": manifest["config_sha256"],
              "artifact_sha256": manifest["artifact_sha256"], "source_arrays": source_identity,
              "source_hashes": source_hashes, "input_hashes": input_hashes,
              "policy": asdict(policy), "sequences": [], "comparisons": [],
              "scope": "coherent induced sampler u/J only; complete session creation/replacement/close; no output writes, freestream/body callbacks or health-owner close",
              "control": "same-source/same-role session checks unchanged; only initial-query-box versus logical-stencil owner admission differs",
              "accuracy_scope": "all requested fields compared; direct/plane physics qualification belongs to separate small-cloud domain tests and public native direct-oracle driver"}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    if args.admission_only:
        report["status"] = "admitted_inputs_only"
        report["sequences"] = [{"schedule_step": step,
            "label": "actual_saved_step_request_order" if step == manifest["vpm_step"]
            else "counterfactual_request_order_on_saved_source_state",
            "requests": [{k: v for k, v in item.items() if k != "points"} for item in requests]}
            for step, requests in sequences]
        publish()
        return

    arrays = {}
    try:
        import cupy as cp
        synchronize = cp.cuda.get_current_stream().synchronize
        for step, requests in sequences:
            context = {"schedule_step": step, "schedule_time": step*dt,
                       "label": "actual_saved_step_request_order" if step == manifest["vpm_step"]
                       else "counterfactual_request_order_on_saved_source_state",
                       "requests": [{k: v for k, v in item.items() if k != "points"} for item in requests],
                       "runs": []}
            report["sequences"].append(context)
            for item in requests:
                arrays[f"step{step}_{item['name']}_position"] = item["points"]
            for repeat in range(args.repeats):
                # AB/BA reduces a systematic ordering bias; source arrays remain
                # unchanged and every mode starts a newly constructed session.
                modes = ("old_query_box", "logical_domain") if repeat % 2 == 0 else ("logical_domain", "old_query_box")
                values = {}
                for mode in modes:
                    record, fields = run_sequence(session_module, controls, policy, sources,
                                                  requests, mode=mode, synchronize=synchronize)
                    record["repeat"] = repeat
                    context["runs"].append(record)
                    values[mode] = fields
                    arrays.update({f"step{step}_repeat{repeat}_{mode}_{key}": value for key, value in fields.items()})
                    print(json.dumps({"step": step, "repeat": repeat, "mode": mode,
                                      "seconds": record["complete_sequence_seconds"],
                                      "owner_builds": record["successful_owner_builds"]}), flush=True)
                    publish()
                report["comparisons"].append({"schedule_step": step, "repeat": repeat,
                    "all_requested_fields": {key: difference(value, values["old_query_box"][key])
                                             for key, value in values["logical_domain"].items()}})
                if [array_identity(value) for value in sources] != source_identity:
                    raise RuntimeError("source arrays changed during read-only replay")
        final_checkpoint = compare.load_checkpoint(args.checkpoint)
        if (final_checkpoint["manifest_sha256"] != checkpoint["manifest_sha256"]
                or final_checkpoint["manifest"]["artifact_sha256"] != manifest["artifact_sha256"]):
            raise RuntimeError("checkpoint changed during qualification")
        if any(digest(path) != value for path, value in input_hashes.items()):
            raise RuntimeError("qualification input changed")
        final_hashes = {path: digest(path) for path in source_hashes}
        if final_hashes != source_hashes:
            raise RuntimeError("numerical source changed during qualification")
        np.savez(archive, **arrays)
        report.update(status="completed", output_archive=str(archive),
                      output_archive_sha256=digest(archive), source_hashes_after=final_hashes)
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
