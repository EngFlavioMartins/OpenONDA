"""Read-only traceback evidence for an opt-in, failed native qualification.

No device operation is retried and no array is copied back from the GPU. The
exception's live frames retain the admitted host sources/query and FFT layout,
even after the ordinary owner cleanup. This is diagnostic capture, not restart
state; the coupled committed checkpoint remains the only restart authority.
"""

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def _exceptions(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ if error.__cause__ is not None else error.__context__


def _array_layout(array):
    pointer = getattr(getattr(array, "data", None), "ptr", None)
    return {
        "shape": list(array.shape), "strides": list(array.strides),
        "dtype": str(array.dtype), "c_contiguous": bool(array.flags.c_contiguous),
        "pointer": None if pointer is None else int(pointer),
        "pointer_mod_8": None if pointer is None else int(pointer) % 8,
        "pointer_mod_16": None if pointer is None else int(pointer) % 16,
    }


def collect_fft_failure(error):
    report = {"scope": "failed-operation layout and unchanged host inputs; NOT a restart checkpoint",
              "exceptions": [], "fft_frames": [], "field_frames": []}
    arrays = {}
    for exception in _exceptions(error):
        report["exceptions"].append({"type": type(exception).__name__, "message": str(exception)})
        trace = exception.__traceback__
        while trace is not None:
            frame = trace.tb_frame
            name, path = frame.f_code.co_name, Path(frame.f_code.co_filename)
            local = frame.f_locals
            if path.parent.name == "gaussian_mesh":
                owner = local.get("self")
                if path.name == "runtime.py" and name in ("rfft", "irfft"):
                    entry = {"file": str(path), "function": name, "line": trace.tb_lineno}
                    for key in ("array", "output", "result"):
                        if key in local:
                            entry[key] = _array_layout(local[key])
                    for key in ("shape", "spectrum_shape", "work_bytes", "max_plan_bytes",
                                "single_workspace", "plan_builds", "closed"):
                        value = getattr(owner, key, None)
                        entry[key] = list(value) if isinstance(value, tuple) else value
                    report["fft_frames"].append(entry)
                elif path.name == "fields.py" and name in ("__init__", "_stream_fields", "_all_channel_fields"):
                    entry = {"file": str(path), "function": name, "line": trace.tb_lineno,
                             "slot": local.get("slot"), "channel": local.get("channel")}
                    for key in ("shape", "fft_shape", "volume", "spacing", "tau", "cutoff",
                                "order", "source_only_primary"):
                        value = getattr(owner, key, None)
                        entry[key] = list(value) if isinstance(value, tuple) else value
                    plan = getattr(owner, "execution_plan", None)
                    entry["execution_plan"] = asdict(plan) if is_dataclass(plan) else repr(plan)
                    if name == "__init__":
                        for key in ("free", "total"):
                            value = local.get(key)
                            entry[key + "_device_bytes"] = None if value is None else int(value)
                        for key in ("estimated_payload_bytes", "max_scratch_bytes", "max_plan_bytes",
                                    "max_correction_bytes", "max_total_bytes"):
                            entry[key] = getattr(owner, key, None)
                    report["field_frames"].append(entry)
                elif path.name == "session.py" and name == "evaluate" and not arrays:
                    source = local.get("source")
                    query = local.get("query")
                    if (isinstance(source, tuple) and len(source) == 3
                            and all(isinstance(value, np.ndarray) for value in (*source, query))):
                        arrays = {key: np.array(value, copy=True, order="C") for key, value in zip(
                            ("source_position", "source_strength", "source_core", "query"),
                            (*source, query), strict=True)}
                        report["source_only_primary"] = local.get("source_only")
                        report["images"] = local.get("images")
            trace = trace.tb_next
    return report, arrays


def save_fft_failure(error, prefix):
    prefix = Path(prefix)
    metadata, archive = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
    if metadata.exists() or archive.exists():
        raise FileExistsError("FFT failure evidence already exists")
    report, arrays = collect_fft_failure(error)
    if not report["fft_frames"] and not report["field_frames"]:
        return {"captured": False, "reason": "no Gaussian FFT or field-construction traceback frame"}
    # Validate JSON serialization before creating either artifact.
    json.dumps(report, allow_nan=False)
    if arrays:
        with archive.open("xb") as stream:
            np.savez(stream, **arrays)
        report["host_input_archive"] = str(archive)
        report["host_input_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with metadata.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"captured": True, "report": str(metadata), "host_inputs": bool(arrays)}
