"""Current reflected-FMM fields used by geometry ownership tests."""

import numpy as np
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.fmm.device import FMMInduction
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction


class Harness:
    def __init__(self, kernel):
        self.n = 11
        rng = np.random.default_rng(8061)
        self.x = ti.Vector.field(3, ti.f32, shape=self.n)
        self.gamma = ti.Vector.field(3, ti.f32, shape=self.n)
        self.radius = ti.field(ti.f32, shape=self.n)
        self.velocity = ti.Vector.field(3, ti.f32, shape=self.n)
        self.gradient = ti.Matrix.field(3, 3, ti.f32, shape=self.n)
        self.rate = ti.Vector.field(3, ti.f32, shape=self.n)
        self.positions = rng.uniform(-0.35, 0.35, (self.n, 3)).astype(np.float32)
        self.x.from_numpy(self.positions)
        self.gamma.from_numpy(rng.normal(0, 1e-7, (self.n, 3)).astype(np.float32))
        self.radius.from_numpy(rng.uniform(0.04, 0.13, self.n).astype(np.float32))
        physics = PhysicsBase(kernel, self.n, ti.f32, max_evaluation_points=4)
        self.slab = SlipSlabInduction(
            FMMInduction(), z_min=-0.48, z_max=0.48, tail_tolerance=1e-4, max_shells=3
        ).bind(physics)

    def run(self):
        self.slab.evaluate_stage(
            position=self.x,
            vortex_strength=self.gamma,
            core_radius=self.radius,
            count=self.n,
            velocity_out=self.velocity,
            velocity_gradient_out=self.gradient,
            vortex_strength_rate_out=self.rate,
        )
        observations = {
            key: self.slab.last_tail[key]
            for key in (
                "shell",
                "block_start",
                "relative",
                "velocity",
                "gradient",
                "target_batches",
            )
        }
        return [
            self.velocity.to_numpy(),
            self.gradient.to_numpy(),
            self.rate.to_numpy(),
        ], observations

    def close(self):
        self.slab.base._release_target_workspace()
        self.slab.base.workspace.destroy()
