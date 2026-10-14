"""Sub-grid diffusion by particle strength exchange.

Core spreading applies an eddy viscosity by growing each particle core with its
own viscosity. Neighbouring cores then differ in size, which degrades the
particle quadrature exactly where the sub-grid model acts. Particle strength
exchange (PSE; Degond & Mas-Gallic 1989) instead diffuses the strengths between
particles and leaves every core unchanged:

    d(alpha_p)/dt = sum_q nu_pq (V_p alpha_q - V_q alpha_p) eta_eps(x_p - x_q) / eps^2,

with ``nu_pq`` the mean eddy viscosity of the pair, ``eps^2`` the mean squared
core radius of the pair and the Gaussian ``eta_eps = 4 pi^(-3/2) exp(-r^2/eps^2)
/ eps^3``, whose second moment makes the sum approximate div(nu grad omega)
(Cottet & Koumoutsakos 2000, Sec. 5.2). Each pair term is antisymmetric, so total
strength is conserved to round-off.
"""

import math
from typing import TYPE_CHECKING

import taichi as ti

if TYPE_CHECKING:
    from ..particles.container import Particles

_SUPPORT_RADII = 4.0  # exp(-16) ~ 1e-7: beyond 4 eps a pair adds nothing in f32
_MAX_CELLS = 4_000_000
_STABLE_RATE = 0.5  # explicit Euler: dt * sum_q nu V eta / eps^2 <= 1/2 per sub-step


@ti.data_oriented
class StrengthExchange:
    """Apply per-particle eddy viscosity by conservative strength exchange."""

    def __init__(self, max_n_particles: int, dtype=ti.f32) -> None:
        self.dtype = dtype
        self.capacity = max_n_particles
        self._rate = ti.Vector.field(3, dtype=dtype, shape=max_n_particles)
        self._order = ti.field(dtype=ti.i32, shape=max_n_particles)
        self._cell = ti.field(dtype=ti.i32, shape=max_n_particles)
        self._lower = ti.Vector.field(3, dtype=dtype, shape=())
        self._upper = ti.Vector.field(3, dtype=dtype, shape=())
        self._extrema = ti.field(dtype=dtype, shape=3)  # max nu_t, max core, min core
        self._cell_capacity = 0
        self._cell_start = None
        self._cell_count = None

    def apply(self, particles: "Particles", time_step_size: float) -> None:
        """Advance the strengths by ``time_step_size`` of sub-grid diffusion."""
        count = len(particles)
        if count < 2 or time_step_size <= 0.0:
            return
        self._bounds(particles.position, particles.eddy_viscosity, particles.core_radius, count)
        max_viscosity, max_core, min_core = (float(v) for v in self._extrema.to_numpy())
        if max_viscosity <= 0.0:
            return
        lower = self._lower[None].to_numpy()
        extent = self._upper[None].to_numpy() - lower
        cell_size = _SUPPORT_RADII * max_core
        shape = [max(1, math.ceil(e / cell_size) + 1) for e in extent]
        while shape[0] * shape[1] * shape[2] > _MAX_CELLS:
            cell_size *= 1.5
            shape = [max(1, math.ceil(e / cell_size) + 1) for e in extent]
        cells = shape[0] * shape[1] * shape[2]
        if cells > self._cell_capacity:
            self._cell_capacity = int(1.25 * cells)
            self._cell_start = ti.field(dtype=ti.i32, shape=self._cell_capacity + 1)
            self._cell_count = ti.field(dtype=ti.i32, shape=self._cell_capacity)
        self._bin(
            particles.position,
            self._cell_start,
            self._cell_count,
            ti.Vector(lower),
            cell_size,
            *shape,
            count,
        )

        # The exchange rate of a particle is at most 4 nu_max / eps_min^2 times its strength.
        stiffness = 4.0 * max_viscosity / max(min_core, 1e-30) ** 2
        n_substeps = max(1, math.ceil(time_step_size * stiffness / _STABLE_RATE))
        substep = time_step_size / n_substeps
        for _ in range(n_substeps):
            self._exchange(
                particles.position,
                particles.vortex_strength,
                particles.core_radius,
                particles.particle_volume,
                particles.eddy_viscosity,
                self._cell_start,
                ti.Vector(lower),
                cell_size,
                *shape,
                count,
            )
            self._advance(particles.vortex_strength, substep, count)
        particles.touch_state()

    @ti.kernel
    def _bounds(
        self,
        position: ti.template(),
        eddy_viscosity: ti.template(),
        core_radius: ti.template(),
        count: ti.i32,
    ):
        self._lower[None] = position[0]
        self._upper[None] = position[0]
        self._extrema[0] = 0.0
        self._extrema[1] = core_radius[0]
        self._extrema[2] = core_radius[0]
        for p in range(count):
            for k in ti.static(range(3)):
                ti.atomic_min(self._lower[None][k], position[p][k])
                ti.atomic_max(self._upper[None][k], position[p][k])
            ti.atomic_max(self._extrema[0], eddy_viscosity[p])
            ti.atomic_max(self._extrema[1], core_radius[p])
            ti.atomic_min(self._extrema[2], core_radius[p])

    @ti.func
    def _cell_of(self, x, lower, cell_size, nx: ti.i32, ny: ti.i32, nz: ti.i32):
        c = ti.cast(ti.floor((x - lower) / cell_size), ti.i32)
        return ti.Vector(
            [
                ti.min(ti.max(c[0], 0), nx - 1),
                ti.min(ti.max(c[1], 0), ny - 1),
                ti.min(ti.max(c[2], 0), nz - 1),
            ]
        )

    @ti.kernel
    def _bin(
        self,
        position: ti.template(),
        cell_start: ti.template(),
        cell_count: ti.template(),
        lower: ti.types.vector(3, ti.f32),
        cell_size: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        count: ti.i32,
    ):
        for c in range(nx * ny * nz):
            cell_count[c] = 0
        for p in range(count):
            c = self._cell_of(position[p], ti.cast(lower, self.dtype), cell_size, nx, ny, nz)
            index = (c[0] * ny + c[1]) * nz + c[2]
            self._cell[p] = index
            ti.atomic_add(cell_count[index], 1)
        ti.loop_config(serialize=True)
        for c in range(nx * ny * nz + 1):
            if c == 0:
                cell_start[0] = 0
            else:
                cell_start[c] = cell_start[c - 1] + cell_count[c - 1]
        for c in range(nx * ny * nz):
            cell_count[c] = 0
        for p in range(count):
            index = self._cell[p]
            slot = ti.atomic_add(cell_count[index], 1)
            self._order[cell_start[index] + slot] = p

    @ti.kernel
    def _exchange(
        self,
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        particle_volume: ti.template(),
        eddy_viscosity: ti.template(),
        cell_start: ti.template(),
        lower: ti.types.vector(3, ti.f32),
        cell_size: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        count: ti.i32,
    ):
        normalization = 4.0 / ti.pow(math.pi, 1.5)
        for p in range(count):
            xp = position[p]
            ap = vortex_strength[p]
            vp = particle_volume[p]
            sp2 = core_radius[p] * core_radius[p]
            nup = eddy_viscosity[p]
            rate = ti.Vector([0.0, 0.0, 0.0], dt=self.dtype)
            c = self._cell_of(xp, ti.cast(lower, self.dtype), cell_size, nx, ny, nz)
            for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                ci, cj, ck = c[0] + i, c[1] + j, c[2] + k
                if 0 <= ci < nx and 0 <= cj < ny and 0 <= ck < nz:
                    index = (ci * ny + cj) * nz + ck
                    for slot in range(cell_start[index], cell_start[index + 1]):
                        q = self._order[slot]
                        viscosity = 0.5 * (nup + eddy_viscosity[q])
                        if q != p and viscosity > 0.0:
                            eps2 = 0.5 * (sp2 + core_radius[q] * core_radius[q])
                            d = xp - position[q]
                            r2 = d.dot(d)
                            if r2 < _SUPPORT_RADII**2 * eps2:
                                eta = normalization * ti.exp(-r2 / eps2) / (eps2 * ti.sqrt(eps2))
                                rate += (
                                    viscosity
                                    * eta
                                    / eps2
                                    * (vp * vortex_strength[q] - particle_volume[q] * ap)
                                )
            self._rate[p] = rate

    @ti.kernel
    def _advance(self, vortex_strength: ti.template(), time_step_size: ti.f32, count: ti.i32):
        for p in range(count):
            vortex_strength[p] += time_step_size * self._rate[p]
