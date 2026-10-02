"""Opt-in, read-only interface trace capture for bounded checkpoint benchmarks.

Only one exchange's arrays are retained at a time. This is instrumentation,
not a predictor, restart cache, or a production solver feature.
"""

from contextlib import contextmanager
import importlib
import json
from pathlib import Path

import numpy as np


def _trace(owner, *, old=False, velocity=None):
    suffix = "_old" if old else ""
    if velocity is None:
        velocity = getattr(owner, "_velocity_boundary_condition" + suffix)
    return {
        "velocity": np.array(velocity, copy=True),
        "normal_velocity": np.array(
            getattr(owner, "_normal_velocity_boundary_condition" + suffix), copy=True
        ),
        "tangential_gradient": np.array(
            getattr(owner, "_tangential_gradient_boundary_condition" + suffix), copy=True
        ),
    }


def _clocks(owner):
    return {
        name: {
            "step": int(getattr(getattr(owner, name + "_solver"), "step")),
            "time": float(getattr(getattr(owner, name + "_solver"), "time")),
        }
        for name in ("fvm", "vpm")
    }


@contextmanager
def capture_interface_traces(owner, prefix, reports, *, max_exchanges, iteration_module=None):
    """Capture explicit trial inputs/outputs without changing solver state.

    All ranks may enter, but only ``owner`` on the VPM-owning master captures
    arrays or writes. ``reports`` receives one JSON-compatible entry per saved
    exchange. Existing artifacts are never replaced. The context patches only
    the benchmark process's module bindings and restores them even on errors.
    """
    if isinstance(max_exchanges, bool) or not isinstance(max_exchanges, int) or max_exchanges < 1:
        raise ValueError("Trace capture requires a positive bounded exchange count")
    prefix = Path(prefix)
    if not prefix.name or not prefix.parent.is_dir():
        raise ValueError("Trace prefix requires an existing output directory")
    module = iteration_module or importlib.import_module("source.coupler.interface_iteration")
    originals = {
        name: getattr(module, name)
        for name in (
            "advance_iterated_interface",
            "advance_fvm",
            "update_boundary_history_after_replacement",
        )
    }
    active = None
    count = 0

    def append_trace(label, trace, *, trial=None):
        event = {"sequence": len(active["metadata"]["events"]), "label": label}
        if trial is not None:
            event["trial"] = trial
        event["clocks"] = _clocks(owner)
        event["arrays"] = {}
        for field, value in trace.items():
            key = f"event_{event['sequence']:03d}_{field}"
            active["arrays"][key] = value
            event["arrays"][field] = key
        active["metadata"]["events"].append(event)

    def advance_fvm(current, *args, **kwargs):
        if active is not None and current is owner:
            # Original signature ends with old and candidate velocity. The
            # matching normal/gradient fields are already installed on owner.
            candidate = kwargs.get("velocity_boundary_condition")
            if candidate is None:
                candidate = args[-1]
            active["trial"] += 1
            append_trace("trial_input", _trace(owner, velocity=candidate), trial=active["trial"])
        return originals["advance_fvm"](current, *args, **kwargs)

    def refresh(current, *args, **kwargs):
        result = originals["update_boundary_history_after_replacement"](current, *args, **kwargs)
        if active is not None and current is owner:
            append_trace("trial_output", _trace(owner, old=True), trial=active["trial"])
        return result

    def iterate(current, geometry, next_velocity):
        nonlocal active, count
        if current is not owner:
            return originals["advance_iterated_interface"](current, geometry, next_velocity)
        comm = getattr(getattr(owner.fvm_solver, "parallel", None), "comm", None)
        collective = comm is not None and comm.Get_size() > 1
        stream = None
        admission_error = None
        if owner._is_master:
            try:
                if active is not None:
                    raise RuntimeError("Nested interface capture is unsupported")
                if count >= max_exchanges:
                    raise RuntimeError("Interface trace capture exceeded its exchange cap")
                clock = _clocks(owner)
                path = prefix.with_name(f"{prefix.name}-step{clock['vpm']['step']:06d}.npz")
                # Reserve before the trial. Mode xb protects against a
                # conflicting writer appearing after benchmark preflight.
                stream = path.open("xb")
                count += 1
                active = {
                    "arrays": {},
                    "trial": 0,
                    "metadata": {
                        "schema_version": 2,
                        "status": "running",
                        "entry_clocks": clock,
                        "events": [],
                        "accepted_sweep": None,
                        "description": "Copied physical traces, including any explicitly reported seed probe; capture makes no state changes",
                    },
                }
                for name, values in zip(
                    ("face_centre", "face_normal", "face_area"), geometry, strict=True
                ):
                    active["arrays"][name] = np.array(values, copy=True)
                append_trace("old_physical_endpoint", _trace(owner, old=True))
                append_trace("raw_predictor", _trace(owner, velocity=next_velocity))
            except BaseException as exc:
                admission_error = repr(exc)
                if stream is not None:
                    stream.close()
                active = None
        if collective:
            admission_error = comm.bcast(admission_error, root=0)
        if admission_error is not None:
            raise RuntimeError("Interface trace capture admission failed: " + admission_error)
        solve_failed = False
        save_error = None
        try:
            result = originals["advance_iterated_interface"](current, geometry, next_velocity)
            if owner._is_master:
                diagnostics = owner._last_interface_iteration_diagnostics
                active["metadata"]["interface_iteration"] = diagnostics
                active["metadata"]["accepted_sweep"] = diagnostics["accepted_sweep"]
                append_trace("accepted_endpoint", _trace(owner, old=True))
                active["metadata"]["status"] = "complete"
        except BaseException as exc:
            solve_failed = True
            if owner._is_master:
                active["metadata"].update(status="failed", error=repr(exc))
            raise
        finally:
            if owner._is_master:
                metadata = active["metadata"]
                metadata["exit_clocks"] = _clocks(owner)
                try:
                    np.savez(
                        stream, metadata_json=np.asarray(json.dumps(metadata)), **active["arrays"]
                    )
                    stream.flush()
                    reports.append(
                        {
                            "path": str(path.resolve()),
                            "status": metadata["status"],
                            "step": clock["vpm"]["step"],
                            "time": clock["vpm"]["time"],
                            "trials": active["trial"],
                            "accepted_sweep": metadata["accepted_sweep"],
                        }
                    )
                except BaseException as exc:
                    save_error = repr(exc)
                    if solve_failed:
                        reports.append(
                            {
                                "path": str(path.resolve()),
                                "status": "save_failed",
                                "error": save_error,
                            }
                        )
                finally:
                    stream.close()
                    active = None
        # A master-only output failure must not leave other ranks entering
        # the next solver collective. Preserve the original solve exception
        # without introducing an extra collective into its failure path.
        if collective:
            save_error = comm.bcast(save_error, root=0)
        if save_error is not None:
            raise RuntimeError("Interface trace capture write failed: " + save_error)
        return result

    module.advance_iterated_interface = iterate
    module.advance_fvm = advance_fvm
    module.update_boundary_history_after_replacement = refresh
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)
