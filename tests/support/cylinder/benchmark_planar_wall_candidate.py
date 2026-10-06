"""Matched frozen native planar GBD timing, without physical flow advance.

Alternate the supplied baseline source and candidate repository on identical
saved particles and unchanged physical parameters, threshold and capacity.
This isolates host GBD cost, excluding FVM, GPU induction, MPI and checkpoint
I/O. The first two trials are warmup; five trials are reported for each model.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import h5py
import numpy as np


def benchmark(repository, directory, baseline_source):
    """Save a matched host timing report with exact input and source hashes."""
    sys.path.insert(0, str(repository))
    spec = importlib.util.spec_from_file_location(
        "wall_audit", Path(__file__).with_name("audit_saved_wall_circulation.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from source.solvers.vpm.physics.diffusion import planar
    from source.solvers.vpm.physics.diffusion.planar import planar_gbd

    if (
        Path(planar.__file__).resolve()
        != (repository / "source/solvers/vpm/physics/diffusion/planar.py").resolve()
    ):
        raise RuntimeError("Planar numerical source does not belong to the supplied repository")

    spec = importlib.util.spec_from_file_location("baseline_planar", baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    metadata_path = directory / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    state_path = metadata_path.parent / metadata["checkpoint_files"]["fvm"]
    vpm_path = metadata_path.parent / metadata["checkpoint_files"]["vpm"]
    input_paths = (metadata_path, state_path, vpm_path, directory / "coupled_mesh.npz")
    hashes = {str(path): helper.digest(path) for path in input_paths}
    _, _, _, _, boundary, wall, _ = helper.load_donor_geometry(directory, state_path)
    configuration = metadata["config"]["vpm"]
    config = SimpleNamespace(**configuration["viscous"])
    with h5py.File(vpm_path) as stored:
        position = np.asarray(stored["particles/position"])
        strength = np.asarray(stored["particles/vortex_strength"])
    particles = SimpleNamespace(
        n_particles_total=len(position),
        position_cpu=lambda: position,
        vortex_strength_cpu=lambda: strength,
    )
    arguments = {
        "dt": configuration["time_step_size"],
        "anchor": metadata["config"]["transfer_lattice"]["anchor"],
        "core_radius_ratio": config.core_radius_ratio,
        "max_particles": configuration["max_n_particles"],
        "solid_at": lambda p: boundary.contains(p, include_boundary=False),
    }
    backends = {
        name: SimpleNamespace(planar_span=configuration["induction"]["planar_span"], plane_z=0)
        for name in ("baseline", "candidate")
    }
    warm = {"baseline": [], "candidate": []}
    counts = {}
    for repeat in range(7):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for name in order:
            operation = baseline.planar_gbd if name == "baseline" else planar_gbd
            wall_arguments = (
                {}
                if name == "baseline"
                else {
                    "wall_crosses": boundary.blocks_segments,
                    "geometry_key": ("frozen-verified-wall", wall.revision),
                }
            )
            start = perf_counter()
            result, _ = operation(particles, config, backends[name], **arguments, **wall_arguments)
            elapsed = perf_counter() - start
            if repeat >= 2:
                warm[name].append(elapsed)
            counts[name] = len(result["position"])
    report = {
        "physical_time": metadata["time"],
        "physical_time_advanced": 0,
        "input_sha256": hashes,
        "timing_scope": "Matched host GBD operator on identical frozen 46 s particles, physical parameters, capacity and unchanged threshold. Alternate operator order; exclude first two warmup trials. Candidate adds exact wall links and wall-visible conservative scatter. Does not measure FVM, induction GPU, MPI, I/O or a whole exchange.",
        "warm_seconds": warm,
        "warm_mean_seconds": {name: float(np.mean(times)) for name, times in warm.items()},
        "output_particle_counts": counts,
        "baseline_sha256": helper.digest(Path(baseline.__file__)),
        "candidate_sha256": helper.digest(
            repository / "source/solvers/vpm/physics/diffusion/planar.py"
        ),
    }
    report["candidate_over_baseline_mean"] = (
        report["warm_mean_seconds"]["candidate"] / report["warm_mean_seconds"]["baseline"]
    )
    if any(helper.digest(path) != hashes[str(path)] for path in input_paths):
        raise RuntimeError("Frozen checkpoint inputs changed during the timing comparison")
    destination = directory / "candidate_planar_wall_timing.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    arguments = parser.parse_args()
    print(benchmark(arguments.repository, arguments.directory, arguments.baseline_source))
