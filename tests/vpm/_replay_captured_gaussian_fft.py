"""Replay captured Gaussian FFT host inputs; no solver or particle advance.

The capture is diagnostic input, never restart authority. Each invocation
imports exactly one explicitly selected source tree, allowing the failed
archived implementation and the repaired implementation to run in isolated
processes. Mathematical grids, descriptors and resource caps are unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--z-min", required=True, type=float)
    parser.add_argument("--z-max", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expect-invalid-value", action="store_true")
    parser.add_argument("--aligned-copy-oracle", action="store_true",
                        help="qualification-only archived FFT adapter; private cap unchanged")
    args = parser.parse_args()
    if args.expect_invalid_value and args.aligned_copy_oracle:
        raise ValueError("failure and oracle modes are mutually exclusive")
    root = Path(__file__).resolve().parents[2]
    solution = root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output = args.output.resolve()
    archive = output.with_suffix(".npz")
    if output.parent != solution or output.suffix != ".json" or output.exists() or archive.exists():
        raise ValueError("fresh evidence required in ordinary solution directory")
    source_root = args.source_root.resolve(strict=True)
    if "source" in sys.modules or not (source_root/"source/__init__.py").is_file():
        raise ValueError("isolated, explicit source package required")
    capture = json.loads(args.capture.read_bytes())
    input_path = Path(capture["host_input_archive"])
    if digest(input_path) != capture["host_input_sha256"]:
        raise ValueError("captured host inputs do not match the recorded digest")
    field_frames = capture["field_frames"]
    if len(field_frames) != 1 or len(capture["fft_frames"]) != 1:
        raise ValueError("exactly one failed Gaussian FFT operation required")
    frame = field_frames[0]
    fft_frame = capture["fft_frames"][0]
    if (frame["function"] not in ("_stream_fields", "_all_channel_fields")
            or fft_frame["function"] != "rfft"):
        raise ValueError("unsupported captured operation")
    images = tuple((int(index), odd) for index, odd in capture["images"])
    if any(type(odd) is not bool for _, odd in images):
        raise ValueError("explicit captured reflection parity required")
    with np.load(input_path, allow_pickle=False) as saved:
        x, gamma, sigma, query = (saved[name].copy() for name in
            ("source_position", "source_strength", "source_core", "query"))
    hashes = {str(path): digest(path) for path in sorted((source_root/"source").rglob("*.py"))}
    hashes[str(Path(__file__).resolve())] = digest(__file__)
    report = {"status": "running", "scope": "captured host field replay; NOT a restart checkpoint",
              "capture": str(args.capture.resolve()), "capture_sha256": digest(args.capture),
              "input_sha256": digest(input_path), "source_hashes": hashes,
              "source_root": str(source_root), "source_count": len(x), "query_count": len(query),
              "captured_fft": fft_frame, "captured_field": frame,
              "expected_invalid_value": args.expect_invalid_value,
              "aligned_copy_oracle": args.aligned_copy_oracle}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    owner = None
    started = time.perf_counter()
    try:
        sys.path.insert(0, str(source_root))
        import cupy as cp

        from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
        from source.solvers.vpm.physics.induction.gaussian_mesh.runtime import FFTPlanPair

        oracle_copies = {"count": 0, "bytes": 0}
        if args.aligned_copy_oracle:
            original_rfft = FFTPlanPair.rfft

            def aligned_copy_rfft(plan, array):
                # Deliberately independent of the production staging path:
                # use a temporary byte-identical aligned allocation, leaving
                # the immutable archived source and its private cap unchanged.
                if array.data.ptr % plan.complex_dtype.itemsize:
                    with plan.owner.allocation_scope():
                        array = cp.array(array, copy=True, order="C")
                    oracle_copies["count"] += 1
                    oracle_copies["bytes"] += array.nbytes
                return original_rfft(plan, array)

            FFTPlanPair.rfft = aligned_copy_rfft

        loaded = Path(sys.modules[GaussianImageFields.__module__].__file__).resolve()
        if not loaded.is_relative_to(source_root/"source"):
            raise RuntimeError("mixed source import before replay")
        owner = GaussianImageFields(x, gamma, sigma, query,
            zmin=args.z_min, zmax=args.z_max, tau=frame["tau"], spacing=frame["spacing"],
            cutoff=frame["cutoff"], order=frame["order"], dtype=fft_frame["array"]["dtype"],
            correction_dtype=fft_frame["array"]["dtype"], max_images=len(images),
            source_only_primary=frame["source_only_primary"])
        report["constructor_seconds"] = time.perf_counter()-started
        report["actual_shape"], report["actual_fft_shape"] = owner.shape, owner.fft_shape
        if list(owner.shape) != frame["shape"] or list(owner.fft_shape) != frame["fft_shape"]:
            raise AssertionError("captured mathematical grid or FFT shape changed")
        record = owner.prepare(images)
        velocity, gradient, query_record = owner.evaluate_prepared(query)
        host_u, host_j = cp.asnumpy(velocity), cp.asnumpy(gradient)
        del velocity, gradient
        if not np.isfinite(host_u).all() or not np.isfinite(host_j).all():
            raise AssertionError("replay returned nonfinite fields")
        if args.expect_invalid_value:
            raise AssertionError("archived failure unexpectedly did not reproduce")
        with archive.open("xb") as stream:
            np.savez(stream, position=query, velocity=host_u, gradient=host_j)
        report.update(status="complete", prepare=record, query=query_record,
                      output_sha256=digest(archive), oracle_alignment_copies=oracle_copies)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      exception_notes=getattr(error, "__notes__", []))
        if args.expect_invalid_value and type(error).__name__ == "CuFFTError" and "CUFFT_INVALID_VALUE" in str(error):
            report["status"] = "expected_failure_reproduced"
        else:
            raise
    finally:
        if owner is not None:
            try:
                owner.close()
            except BaseException as error:
                report.update(status="failed", cleanup_error=repr(error))
        report["complete_seconds_including_close"] = time.perf_counter()-started
        report["sources_changed"] = [path for path, before in hashes.items() if digest(path) != before]
        report["inputs_unchanged"] = digest(input_path) == capture["host_input_sha256"]
        report["mixed_source_imports"] = [name for name, module in sys.modules.items()
            if (name == "source" or name.startswith("source."))
            and getattr(module, "__file__", None)
            and not Path(module.__file__).resolve().is_relative_to(source_root/"source")]
        if report["sources_changed"] or report["mixed_source_imports"] or not report["inputs_unchanged"]:
            report["status"] = "failed"
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({key: report.get(key) for key in
            ("status", "source_count", "query_count", "actual_fft_shape", "complete_seconds_including_close", "error")}))
    if report["status"] not in ("complete", "expected_failure_reproduced"):
        raise RuntimeError("captured-field replay evidence failed")


if __name__ == "__main__":
    main()
