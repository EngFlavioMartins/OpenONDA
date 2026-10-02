"""Read-only CuFFT channel-alignment reproduction against a frozen runtime."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_path = args.runtime.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("fresh reproduction evidence path required")
    before = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("frozen_gaussian_fft_runtime", runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    import cupy as cp

    report = {"status": "running", "runtime_path": str(runtime_path), "runtime_sha256": before,
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cupy_version": cp.__version__, "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
              "driver_version": cp.cuda.runtime.driverGetVersion(), "cases": []}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        for dtype in (np.float32, np.float64):
            for shape in ((5, 7, 9), (5, 7, 10), (9, 15, 21)):
                owner = runtime.DeviceOwner(32*1024**2)
                plan = None
                with owner.allocation_scope():
                    slab = cp.arange(3*math.prod(shape), dtype=dtype).reshape((3, *shape))/dtype(64)
                try:
                    plan = runtime.FFTPlanPair(owner, shape, dtype, 8*1024**2, single_workspace=True)
                    for slot in range(3):
                        view = slab[slot]
                        with owner.allocation_scope():
                            aligned = cp.array(view, copy=True, order="C")
                        np.testing.assert_array_equal(cp.asnumpy(view), cp.asnumpy(aligned))
                        exact = np.fft.rfftn(cp.asnumpy(view))
                        case = {"shape": list(shape), "volume": math.prod(shape),
                                "dtype": np.dtype(dtype).name, "slot": slot,
                                "view_pointer": view.data.ptr, "view_offset": view.data.ptr-slab.data.ptr,
                                "view_mod_complex_itemsize": view.data.ptr % (2*np.dtype(dtype).itemsize),
                                "c_contiguous": bool(view.flags.c_contiguous),
                                "aligned_pointer": aligned.data.ptr,
                                "plan_shape": list(plan.shape), "spectrum_shape": list(plan.spectrum_shape),
                                "plan_work_bytes": plan.work_bytes, "results": {}}
                        for label, array in (("channel_view", view), ("aligned_copy", aligned)):
                            try:
                                value = plan.rfft(array)
                                owner.stream.synchronize()
                                host = cp.asnumpy(value)
                                case["results"][label] = {"success": True,
                                    "max_absolute_error": float(np.max(np.abs(host-exact))),
                                    "relative_l2_error": float(np.linalg.norm(host-exact)/np.linalg.norm(exact))}
                                del value
                            except cp.cuda.cufft.CuFFTError as error:
                                case["results"][label] = {"success": False,
                                    "error_type": type(error).__name__, "message": str(error)}
                        report["cases"].append(case)
                        print(json.dumps(case), flush=True)
                        del aligned, view
                        publish()
                finally:
                    if plan is not None:
                        plan.close()
                    del slab
                    owner.close()
        if hashlib.sha256(runtime_path.read_bytes()).hexdigest() != before:
            raise RuntimeError("frozen runtime source changed")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
