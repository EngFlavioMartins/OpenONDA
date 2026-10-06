"""Bounded native planar CUDA sums checked against independent CPU induction.

Allocates at most one source batch and one face-target array on the device.
Particle positions, strengths and radii retain their admitted storage dtype;
velocity/Jacobian sums use native float64 accumulators. No solver is created.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tests.support.cylinder.audit_saved_wall_circulation import gaussian_velocity_and_gradient


class PlanarImageInduction:
    """Project frozen particle images on a bounded native CUDA workspace."""

    def __init__(self, points, sigma, span, output_dtype, *, source_batch_size=32768):
        import taichi as ti

        from source.solvers.vpm.physics.induction.planar import PlanarInduction

        dtype = np.dtype(output_dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("Image sources must retain native float32/64 storage")
        if not isinstance(source_batch_size, int) or not 1 <= source_batch_size <= 65536:
            raise ValueError("Planar source batch must be bounded by65536")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            raise ValueError("Planar targets must be a nonempty(N,3) array")
        ti.init(
            arch=ti.cuda,
            default_fp=ti.f64,
            offline_cache=False,
            cpu_max_num_threads=2,
            device_memory_fraction=0.02,
        )
        if ti.cfg.arch != ti.cuda:
            raise RuntimeError("The requested CUDA image projection is unavailable")
        self.ti, self.dtype, self.batch = ti, dtype, source_batch_size
        self.sigma, self.span = dtype.type(sigma), float(span)
        source_type = ti.f32 if dtype == np.dtype(np.float32) else ti.f64
        self.source = ti.Vector.field(3, source_type, shape=self.batch)
        self.strength = ti.Vector.field(3, source_type, shape=self.batch)
        self.radius = ti.field(source_type, shape=self.batch)
        self.target = ti.Vector.field(3, ti.f64, shape=len(points))
        self.velocity = ti.Vector.field(3, ti.f64, shape=len(points))
        self.gradient = ti.Matrix.field(3, 3, ti.f64, shape=len(points))
        self.points = points.copy()
        self.target.from_numpy(points)
        self.radius.from_numpy(np.full(self.batch, self.sigma, dtype=dtype))
        self.induction = PlanarInduction(span=span).bind(
            SimpleNamespace(accumulator_dtype=ti.f64, max_n_particles=self.batch)
        )
        self.validation = None

    def evaluate(self, positions, strengths, *, check_cpu=False):
        positions = np.asarray(positions)
        strengths = np.asarray(strengths)
        if positions.dtype != self.dtype or strengths.dtype != self.dtype:
            raise ValueError("CUDA image projection cannot change source storage precision")
        if positions.shape != strengths.shape or positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("Image positions and strengths must share shape(N,3)")
        self.induction.validate_source_arrays(positions, strengths)
        velocity = np.zeros_like(self.points)
        gradient = np.zeros((len(self.points), 3, 3))
        for start in range(0, len(positions), self.batch):
            count = min(self.batch, len(positions) - start)
            source_buffer = np.zeros((self.batch, 3), dtype=self.dtype)
            strength_buffer = np.zeros_like(source_buffer)
            source_buffer[:count] = positions[start : start + count]
            strength_buffer[:count] = strengths[start : start + count]
            self.source.from_numpy(source_buffer)
            self.strength.from_numpy(strength_buffer)
            self.induction.evaluate_targets(
                target_position=self.target,
                source_position=self.source,
                source_vortex_strength=self.strength,
                source_core_radius=self.radius,
                target_velocity=self.velocity,
                target_velocity_gradient=self.gradient,
                target_count=len(self.points),
                source_count=count,
                include_freestream=False,
                background_velocity=(0.0, 0.0, 0.0),
            )
            velocity += self.velocity.to_numpy()
            gradient += self.gradient.to_numpy()
        if check_cpu:
            expected_velocity, expected_gradient = gaussian_velocity_and_gradient(
                self.points,
                positions.astype(float),
                strengths[:, 2].astype(float) / self.span,
                float(self.sigma),
            )
            differences = {
                "maximum_velocity_absolute_error": float(
                    np.max(np.abs(velocity - expected_velocity))
                ),
                "maximum_jacobian_absolute_error": float(
                    np.max(np.abs(gradient - expected_gradient))
                ),
            }
            if (
                differences["maximum_velocity_absolute_error"] > 2e-6
                or differences["maximum_jacobian_absolute_error"] > 2e-6
            ):
                raise ValueError(
                    "Bounded native CUDA image sum differs from independent CPU reference"
                )
            self.validation = differences | {
                "absolute_error_bound": 2e-6,
                "source_count": len(positions),
                "source_batch_size": self.batch,
                "source_storage_dtype": str(self.dtype),
                "accumulator_dtype": "float64",
            }
        return velocity, gradient
