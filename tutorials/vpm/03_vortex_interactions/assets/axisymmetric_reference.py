"""Independent axisymmetric Navier--Stokes diagnostic, not a VPM replacement.

Evolves q=omega_theta/r by centered differences and SSPRK3, with a sparse
streamfunction solve at each stage. An impulse-dipole far boundary approximates
unbounded flow; domain, grid and timestep controls are required before using
this diagnostic as a reference. No instability, LES or stabilization is added.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.ndimage import maximum_filter
from scipy.fft import dst, idst


class MeridionalPoisson:
    """Separable inverse of the same discrete Dirichlet streamfunction matrix."""

    def __init__(self, n, radial_interior, spacing):
        self.shape = (n, len(radial_interior))
        h = spacing
        eigen = 4 * np.sin(np.arange(1, n + 1) * np.pi / (2 * (n + 1))) ** 2 / h**2
        self.lower = (-1 - h / (2 * radial_interior[1:])) / h**2
        upper = (-1 + h / (2 * radial_interior[:-1])) / h**2
        self.inverse = np.empty((len(radial_interior), n))
        self.upper = np.empty((len(radial_interior) - 1, n))
        diagonal = 2 / h**2 + eigen
        self.inverse[0] = 1 / diagonal
        for j in range(len(radial_interior) - 1):
            self.upper[j] = upper[j] * self.inverse[j]
            self.inverse[j + 1] = 1 / (diagonal - self.lower[j] * self.upper[j])

    def solve(self, right_hand_side):
        work = dst(
            np.asarray(right_hand_side).reshape(self.shape), type=1, axis=0, norm="ortho"
        ).T.copy()
        work[0] *= self.inverse[0]
        for j in range(1, len(work)):
            work[j] = (work[j] - self.lower[j - 1] * work[j - 1]) * self.inverse[j]
        for j in range(len(work) - 2, -1, -1):
            work[j] -= self.upper[j] * work[j + 1]
        return idst(work.T, type=1, axis=0, norm="ortho").ravel()


def run(args):
    started = time.perf_counter()
    h = args.spacing
    x = np.arange(-args.margin, 10 + args.margin + h / 2, h)
    r = np.arange(0, args.radius + h / 2, h)
    nx, nr = len(x), len(r)
    xx, rr = np.meshgrid(x, r, indexing="ij")
    omega = sum(100 * np.exp(-((xx - c) ** 2 + (rr - 1) ** 2) / 0.01) for c in [-0.5, 0.5])
    q = np.divide(omega, rr, out=np.zeros_like(omega), where=rr > 0)
    q[:, 0] = q[:, 1]
    ri = r[1:-1]
    m = nr - 2
    n = nx - 2
    dx = sparse.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]) / h**2
    # -psi_rr + psi_r/r; interior radial row uses its own radius.
    dr = sparse.diags(
        [(-1 - h / (2 * ri[1:])) / h**2, 2 * np.ones(m) / h**2, (-1 + h / (2 * ri[:-1])) / h**2],
        [-1, 0, 1],
    )
    poisson = (
        splu((sparse.kron(dx, sparse.eye(m)) + sparse.kron(sparse.eye(n), dr)).tocsc())
        if args.poisson == "sparse"
        else MeridionalPoisson(n, ri, h)
    )
    nu = np.pi / args.reynolds
    initial_impulse = float(np.pi * np.sum(rr**3 * q) * h * h)

    def rhs(q):
        omega = rr * q
        impulse_weights = rr**2 * omega
        centre = float(np.sum(xx * impulse_weights) / np.sum(impulse_weights))
        distance = ((xx - centre) ** 2 + rr**2) ** 1.5
        boundary = np.divide(
            initial_impulse * rr**2, 4 * np.pi * distance, out=np.zeros_like(q), where=distance > 0
        )
        b = rr[1:-1, 1:-1] * omega[1:-1, 1:-1]
        b = b.copy()
        b[0, :] += boundary[0, 1:-1] / h**2
        b[-1, :] += boundary[-1, 1:-1] / h**2
        b[:, -1] += (1 - h / (2 * ri[-1])) * boundary[1:-1, -1] / h**2
        psi = boundary
        psi[:, 0] = 0
        psi[1:-1, 1:-1] = poisson.solve(b.ravel()).reshape(n, m)
        ux = np.zeros_like(q)
        ur = np.zeros_like(q)
        ux[:, 1:] = np.gradient(psi, h, axis=1, edge_order=2)[:, 1:] / r[None, 1:]
        ux[:, 0] = 2 * psi[:, 1] / h**2
        ur[:, 1:] = -np.gradient(psi, h, axis=0, edge_order=2)[:, 1:] / r[None, 1:]
        qx = np.gradient(q, h, axis=0, edge_order=2)
        qr = np.gradient(q, h, axis=1, edge_order=2)
        qr[:, 0] = 0
        lap = np.zeros_like(q)
        lap[1:-1, 1:-1] = (
            q[2:, 1:-1] + q[:-2, 1:-1] + q[1:-1, 2:] + q[1:-1, :-2] - 4 * q[1:-1, 1:-1]
        ) / h**2 + 3 * qr[1:-1, 1:-1] / ri
        lap[1:-1, 0] = (
            q[2:, 0] + q[:-2, 0] - 2 * q[1:-1, 0] + 8 * (q[1:-1, 1] - q[1:-1, 0])
        ) / h**2
        value = -ux * qx - ur * qr + nu * lap
        value[[0, -1], :] = 0
        value[:, -1] = 0
        return value, max(float(np.max(abs(ux))), float(np.max(abs(ur))))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(args.steps + 1):
        if step % args.sample == 0 or step == args.steps:
            field = rr * q
            peaks = np.argwhere(
                (field == maximum_filter(field, size=5)) & (field > 0.15 * field.max())
            )
            # Grid peaks are retained as observations; no labels forced after merger.
            for rank, (i, j) in enumerate(sorted(peaks, key=lambda ij: -field[tuple(ij)])):
                rows.append(
                    dict(
                        step=step,
                        time=step * args.dt,
                        x=x[i],
                        radius=r[j],
                        vorticity=field[i, j],
                        peak_rank=rank,
                        n_peaks=len(peaks),
                        impulse=np.pi * np.sum(rr**2 * field) * h * h,
                        minimum_vorticity=field.min(),
                    )
                )
            np.savez_compressed(
                output / f"{step:06d}.npz", x=x, r=r, vorticity=field, time=step * args.dt
            )
            pd.DataFrame(rows).to_csv(output / "peaks.csv", index=False)
            print(
                step,
                round(step * args.dt, 4),
                "peaks",
                [(round(x[i], 3), round(r[j], 3)) for i, j in peaks],
                "min/max",
                float(field.min() / field.max()),
                "elapsed",
                round(time.perf_counter() - started, 1),
                flush=True,
            )
        if step == args.steps:
            break
        k1, speed = rhs(q)
        if speed * args.dt / h > 0.6:
            raise RuntimeError(f"Control CFL too large: {speed * args.dt / h}")
        a = q + args.dt * k1
        b = 0.75 * q + 0.25 * (a + args.dt * rhs(a)[0])
        q = q / 3 + 2 * (b + args.dt * rhs(b)[0]) / 3
        if not np.isfinite(q).all():
            raise RuntimeError("nonfinite control")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spacing", type=float, default=0.03)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--reynolds", type=float, default=3000)
    p.add_argument("--margin", type=float, default=3.0)
    p.add_argument("--radius", type=float, default=4.0)
    p.add_argument("--output", required=True)
    p.add_argument("--poisson", choices=("separable", "sparse"), default="separable")
    run(p.parse_args())
