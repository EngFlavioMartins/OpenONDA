"""Compare the former atomic image accumulation with per-target reduction.

Run on an otherwise idle CPU with ``PYTHONPATH=. python
studies/benchmark_slip_slab_accumulation.py``. Both implementations consume
the same frozen image fields or source/target cloud in one Taichi process.
Compilation and first-call cost are excluded from the median measurements.
"""

import json
import statistics
from time import perf_counter

import numpy as np
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.fmm import FMMInduction
import source.solvers.vpm.physics.induction.slip_slab as slab_module
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction, _add_reflected_results


@ti.kernel
def old_accumulate(
    image_velocity: ti.template(),
    image_gradient: ti.template(),
    odd_flags: ti.template(),
    velocity: ti.template(),
    gradient: ti.template(),
    shell_velocity: ti.template(),
    shell_gradient: ti.template(),
    points: ti.i32,
    images: ti.i32,
):
    for i in range(points * images):
        point = i % points
        image = i // points
        v = image_velocity[i]
        j = image_gradient[i]
        if odd_flags[image]:
            v[2] = -v[2]
            for a, b in ti.static(ti.ndrange(3, 3)):
                if (a == 2) != (b == 2):
                    j[a, b] = -j[a, b]
        for a in ti.static(range(3)):
            ti.atomic_add(velocity[point][a], v[a])
            ti.atomic_add(shell_velocity[point][a], v[a])
        for a, b in ti.static(ti.ndrange(3, 3)):
            ti.atomic_add(gradient[point][a, b], j[a, b])
            ti.atomic_add(shell_gradient[point][a, b], j[a, b])


ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=4)
points, images = 2000, 32
rng = np.random.default_rng(5)
iv = ti.Vector.field(3, ti.f32, shape=points * images)
ij = ti.Matrix.field(3, 3, ti.f32, shape=points * images)
iv.from_numpy(rng.normal(size=(points * images, 3)).astype("f4"))
ij.from_numpy(rng.normal(size=(points * images, 3, 3)).astype("f4"))
odd = ti.field(ti.i32, shape=64)
odd.from_numpy(np.arange(64, dtype=np.int32) % 2)
shift = ti.field(ti.f32, shape=64)
strength = ti.Vector.field(3, ti.f32, shape=points)
v = ti.Vector.field(3, ti.f32, shape=points)
j = ti.Matrix.field(3, 3, ti.f32, shape=points)
sv = ti.Vector.field(3, ti.f32, shape=points)
sj = ti.Matrix.field(3, 3, ti.f32, shape=points)
rate = ti.Vector.field(3, ti.f32, shape=points)


def new():
    _add_reflected_results(
        iv, ij, shift, odd, strength, v, rate, j, sv, sj, 0, points, images, 0, 0, True, True, False
    )


def old():
    old_accumulate(iv, ij, odd, v, j, sv, sj, points, images)


def measure(call):
    call()
    times = []
    for _ in range(20):
        v.fill(0)
        j.fill(0)
        sv.fill(0)
        sj.fill(0)
        start = perf_counter()
        call()
        ti.sync()
        times.append(perf_counter() - start)
    return statistics.median(times), v.to_numpy(), j.to_numpy()


old_time, old_v, old_j = measure(old)
new_time, new_v, new_j = measure(new)
print(
    json.dumps(
        {
            "case": "accumulator",
            "points": points,
            "images": images,
            "old_median_seconds": old_time,
            "new_median_seconds": new_time,
            "ratio": old_time / new_time,
            "max_velocity_delta": float(np.max(np.abs(old_v - new_v))),
            "max_gradient_delta": float(np.max(np.abs(old_j - new_j))),
        }
    )
)


@ti.kernel
def old_full(
    image_velocity: ti.template(),
    image_gradient: ti.template(),
    shifts: ti.template(),
    odd_flags: ti.template(),
    strength: ti.template(),
    velocity: ti.template(),
    rate: ti.template(),
    gradient: ti.template(),
    shell_velocity: ti.template(),
    shell_gradient: ti.template(),
    start: ti.i32,
    points_per_image: ti.i32,
    image_count: ti.i32,
    mode: ti.i32,
    rate_enabled: ti.i32,
    has_velocity: ti.template(),
    has_gradient: ti.template(),
    is_stage: ti.template(),
):
    for i in range(points_per_image * image_count):
        point = start + i % points_per_image
        image = i // points_per_image
        v = image_velocity[i]
        jac = image_gradient[i]
        if odd_flags[image]:
            v[2] = -v[2]
            for a, b in ti.static(ti.ndrange(3, 3)):
                if (a == 2) != (b == 2):
                    jac[a, b] = -jac[a, b]
        if ti.static(has_velocity):
            for a in ti.static(range(3)):
                ti.atomic_add(velocity[point][a], v[a])
        for a, b in ti.static(ti.ndrange(3, 3)):
            ti.atomic_add(shell_gradient[point][a, b], jac[a, b])
            if ti.static(has_gradient):
                ti.atomic_add(gradient[point][a, b], jac[a, b])
        for a in ti.static(range(3)):
            ti.atomic_add(shell_velocity[point][a], v[a])


sources = targets = 1000
p = rng.uniform((-1, -1, -0.42), (1, 1, 0.42), (sources, 3)).astype("f4")
g = rng.normal(0, 0.005, (sources, 3)).astype("f4")
g -= g.mean(axis=0)
t = rng.uniform((-1, -1, -0.48), (1, 1, 0.48), (targets, 3)).astype("f4")
physics = PhysicsBase("GAUSSIAN", sources, ti.f32, max_evaluation_points=32768)
slab = SlipSlabInduction(
    FMMInduction(), z_min=-0.48, z_max=0.48, tail_tolerance=1e-4, max_shells=129
).bind(physics)
x = ti.Vector.field(3, ti.f32, shape=sources)
w = ti.Vector.field(3, ti.f32, shape=sources)
r = ti.field(ti.f32, shape=sources)
y = ti.Vector.field(3, ti.f32, shape=targets)
u = ti.Vector.field(3, ti.f32, shape=targets)
grad = ti.Matrix.field(3, 3, ti.f32, shape=targets)
x.from_numpy(p)
w.from_numpy(g)
r.fill(0.08)
y.from_numpy(t)


def evaluate():
    slab.evaluate_targets(
        target_position=y,
        source_position=x,
        source_vortex_strength=w,
        source_core_radius=r,
        target_velocity=u,
        target_velocity_gradient=grad,
        target_count=targets,
        source_count=sources,
        include_freestream=False,
        background_velocity=physics._zero_velocity,
    )


full_results = {}
for name, kernel in [("old", old_full), ("new", _add_reflected_results)]:
    slab_module._add_reflected_results = kernel
    evaluate()
    times = []
    for _ in range(3):
        start = perf_counter()
        evaluate()
        ti.sync()
        times.append(perf_counter() - start)
    full_results[name] = (
        statistics.median(times),
        u.to_numpy(),
        grad.to_numpy(),
        slab.last_tail.copy(),
    )
old_elapsed, old_u, old_grad, old_tail = full_results["old"]
new_elapsed, new_u, new_grad, new_tail = full_results["new"]
print(
    json.dumps(
        {
            "case": "full_slab_targets",
            "sources": sources,
            "targets": targets,
            "old_median_seconds": old_elapsed,
            "new_median_seconds": new_elapsed,
            "ratio": old_elapsed / new_elapsed,
            "max_velocity_delta": float(np.max(np.abs(old_u - new_u))),
            "max_gradient_delta": float(np.max(np.abs(old_grad - new_grad))),
            "old_tail": {k: v for k, v in old_tail.items() if k != "seconds"},
            "new_tail": {k: v for k, v in new_tail.items() if k != "seconds"},
        }
    )
)
