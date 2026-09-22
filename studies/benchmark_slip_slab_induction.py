import argparse
import json
import time

import numpy as np
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.fmm import FMMInduction
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction

parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=["cpu", "vulkan"], required=True)
parser.add_argument("--sources", type=int, default=1000)
parser.add_argument("--targets", type=int, default=1000)
parser.add_argument("--capacity", type=int, required=True)
args = parser.parse_args()
ti.init(
    arch=ti.cpu if args.arch == "cpu" else ti.vulkan,
    default_fp=ti.f32,
    offline_cache=False,
    cpu_max_num_threads=4,
)
rng = np.random.default_rng(101)
p = rng.uniform((-1, -1, -0.42), (1, 1, 0.42), size=(args.sources, 3)).astype("f4")
g = rng.normal(0, 0.005, size=(args.sources, 3)).astype("f4")
g -= g.mean(axis=0)
t = rng.uniform((-1, -1, -0.48), (1, 1, 0.48), size=(args.targets, 3)).astype("f4")
physics = PhysicsBase("GAUSSIAN", args.sources, ti.f32, max_evaluation_points=args.capacity)
slab = SlipSlabInduction(
    FMMInduction(), z_min=-0.48, z_max=0.48, tail_tolerance=1e-4, max_shells=129
).bind(physics)
x = ti.Vector.field(3, ti.f32, shape=args.sources)
x.from_numpy(p)
w = ti.Vector.field(3, ti.f32, shape=args.sources)
w.from_numpy(g)
r = ti.field(ti.f32, shape=args.sources)
r.fill(0.08)
y = ti.Vector.field(3, ti.f32, shape=args.targets)
y.from_numpy(t)
u = ti.Vector.field(3, ti.f32, shape=args.targets)
j = ti.Matrix.field(3, 3, ti.f32, shape=args.targets)
records = []
for _run in range(2):
    started = time.perf_counter()
    error = None
    try:
        slab.evaluate_targets(
            target_position=y,
            source_position=x,
            source_vortex_strength=w,
            source_core_radius=r,
            target_velocity=u,
            target_velocity_gradient=j,
            target_count=args.targets,
            source_count=args.sources,
            include_freestream=False,
            background_velocity=physics._zero_velocity,
        )
    except RuntimeError as exc:
        error = str(exc)
    records.append(
        {"elapsed_seconds": time.perf_counter() - started, "tail": slab.last_tail, "error": error}
    )
    if error:
        break
print(
    json.dumps(
        {
            "arch": args.arch,
            "sources": args.sources,
            "targets": args.targets,
            "capacity": args.capacity,
            "runs": records,
            "device_memory_estimate_bytes": slab.base.diagnostics.device_memory_estimate_bytes,
        }
    )
)
