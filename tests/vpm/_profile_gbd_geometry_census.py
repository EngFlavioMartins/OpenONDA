"""Read-only exact-axis reuse census on accepted native checkpoint clouds.

Accepted clouds are NOT post-RK production GBD inputs. This census demonstrates
available native coordinate overlap, not the actual next-step cache hit rate or
timing. No geometry classification, Taichi runtime, solver or GPU is invoked.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from tests.vpm._gbd_body_host_oracle import legacy_host_type
from tests.vpm._gbd_geometry_cache_prototype import _matching_axis


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def census(manifest_path, particles, source_root):
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]["vpm"]
    viscous, induction = config["viscous"], config["induction"]
    if viscous["scheme"] != "GBD" or induction["method"] != "SLIP_SLAB":
        raise ValueError("Census requires the explicitly reconstructed GBD/slab contract")
    h, padding = viscous["gbd_grid_spacing"], viscous["gbd_domain_padding"]
    slab = induction["z_min"], induction["z_max"]
    host = legacy_host_type(source_root)((1, 1, 1), None, None, slab)
    # Invoke the actual strict, f32-guarded production stage-count body.
    substeps, _ = host._explicit_diffusion_substep_count(
        viscous["kinematic_viscosity"], config["time_step_size"], h)
    halo = max(3, substeps + 1) * h
    host.configure_max_grid_extent(config["domain_bounds"], h, padding)
    host.configure_grid_lattice_anchor(manifest["config"]["transfer_lattice"]["anchor"], h)
    records, previous_axes32, previous_axes64 = [], None, None
    for path in particles:
        initial_digest = digest(path)
        with h5py.File(path, "r") as saved:
            native_config = json.loads(saved["solver"].attrs["numerical_configuration"])
            if native_config != config:
                raise ValueError("Particle snapshot numerical configuration differs")
            position = saved["particles/position"][:]
            step, time = int(saved["solver"].attrs["step"]), float(saved["solver"].attrs["time"])
            if not np.all(np.isfinite(position)):
                raise ValueError("Nonfinite particle positions")
        origin, shape = host._lattice_aligned_bounds(position, h, padding,
                                                    required_z_bounds=(slab[0] - halo, slab[1] + halo))
        axes64 = tuple(origin[d] + np.arange(n, dtype=np.int64) * h for d, n in enumerate(shape))
        axes32 = tuple(axis.astype(np.float32) for axis in axes64)
        record = {"path": str(path), "sha256": initial_digest, "step": step, "time": time,
                  "particles": len(position), "origin": origin.tolist(), "shape": list(shape),
                  "nodes": math.prod(shape), "comparison": None}
        if previous_axes32 is not None:
            counts32 = [len(_matching_axis(a, b)[0]) for a, b in zip(previous_axes32, axes32, strict=True)]
            matches64 = [_matching_axis(a, b) for a, b in zip(previous_axes64, axes64, strict=True)]
            link_upper = 0
            for axis in range(3):
                counts = [len(new) for new, _ in matches64]
                new, old = matches64[axis]
                counts[axis] = int(np.count_nonzero((new + 1 < len(axes64[axis]))
                                                   & (old + 1 < len(previous_axes64[axis]))))
                link_upper += math.prod(counts)
            possible_links = sum((shape[d] - 1) * math.prod(shape[:d] + shape[d + 1:]) for d in range(3))
            record["comparison"] = {
                "membership_matching_axes": counts32, "membership_reusable_nodes": math.prod(counts32),
                "membership_reuse_fraction": math.prod(counts32) / math.prod(shape),
                "link_matching_axes": [len(new) for new, _ in matches64],
                "link_reuse_upper_bound_before_fluid_gate": link_upper,
                "all_possible_links_before_fluid_gate": possible_links,
                "link_reuse_upper_fraction": link_upper / possible_links,
            }
        if digest(path) != initial_digest:
            raise ValueError("Input snapshot changed during census")
        records.append(record)
        previous_axes32, previous_axes64 = axes32, axes64
    return {"scope": __doc__, "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
            "grid_source_sha256": digest(source_root / "source/solvers/vpm/physics/diffusion/grid.py"),
            "spacing": h, "fixed_origin": host._fixed_grid_min.tolist(), "records": records,
            "production_state_modified": False, "gpu_initialized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--particles", type=Path, nargs="+", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = census(args.manifest, args.particles, args.source_root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
