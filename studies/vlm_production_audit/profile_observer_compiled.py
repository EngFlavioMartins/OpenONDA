"""Bounded host-profile replay of 16 recorded rotor particles, for cost attribution."""

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import h5py
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from tests._tutorial_helpers import load_tutorial_module


def main():
    """Profile classification, not a new physical trajectory or full solver step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=Path(__file__).stem)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError("tag must be a simple filename")
    output = Path(__file__).with_name(args.tag)
    if output.with_suffix(".json").exists():
        raise FileExistsError("Preserve existing replay evidence; use a fresh --tag")
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=2)
    try:
        setup = load_tutorial_module("vpm/rotor_flow")
        vlm = VLMSolver(setup.build_case().numerics.vlm)
        vlm.generate_mesh()
        with h5py.File(Path(__file__).with_name("rotor_smoke") / "solution/vpm_000003.h5") as f:
            arrays = {
                name: f[f"particles/{name}"][:16]
                for name in ("position", "velocity", "vortex_strength", "core_radius", "group_id")
            }

        class Cloud:
            n_particles_total = 16

            def position_cpu(self, **kwargs):
                return arrays["position"]

            def vortex_strength_cpu(self, **kwargs):
                return arrays["vortex_strength"]

            def core_radius_cpu(self, **kwargs):
                return arrays["core_radius"]

            def group_id_cpu(self, **kwargs):
                return arrays["group_id"]

        vlm._snapshot_pre_transport_positions(Cloud(), time=0.012, time_step_size=0.006)
        # Synthetic straight backtracking isolates observer cost. It is not
        # evidence about actual accepted particle crossings.
        vlm._pre_transport_position = arrays["position"] - 0.006 * arrays["velocity"]
        vlm.observe_surface_interaction(Cloud())  # compile/warm outside timing
        vlm._snapshot_pre_transport_positions(Cloud(), time=0.012, time_step_size=0.006)
        vlm._pre_transport_position = arrays["position"] - 0.006 * arrays["velocity"]
        start = time.perf_counter()
        profiler = cProfile.Profile()
        result = profiler.runcall(vlm.observe_surface_interaction, Cloud())
        seconds = time.perf_counter() - start
        with output.with_suffix(".txt").open("w") as out:
            pstats.Stats(profiler, stream=out).sort_stats("cumtime").print_stats(25)
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "scope": "16-particle synthetic replay of recorded rotor geometry/cores",
                    "panels": vlm.lattice.n_panels,
                    "events": result["n_events"],
                    "seconds": seconds,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        ti.reset()


if __name__ == "__main__":
    main()
