"""
fvm_fringe.py  --  build and update the fringe-relaxation fields for the FVM.

The coupler calls this to push two volume fields into the Eulerian backend:

  * `lambdaRelax`  (scalar field, set ONCE): the relaxation rate, 0 in the FVM
    core and ramping smoothly to `lambda_max` at the numerical boundary, over the
    buffer band.  This is the FVM-side complement of the VPM-side authority weight
    eta -- where eta -> 0 (VPM takes over), lambda -> max (FVM defers to VPM).

  * `Utarget`  (vector field, set EVERY coupling step): the VPM velocity
    sampled at the FVM cell centres, i.e. the field the fringe relaxes toward.

Both are looked up by name inside the backend's momentum source: the OFW wrapper
reads them via relaxationSource.H (fvOptions); the native FVM via
``registered_fields`` in ``Solver._fringe_source``.
"""

from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger("coupler")


def build_lambda(
    cell_centres: np.ndarray,
    fvm_box: tuple[float, float, float, float, float, float],
    buffer_thickness: float,
    lambda_max: float,
    dead_zone: float = 0.0,
) -> np.ndarray:
    """Static FVM relaxation rate complementary to particle authority ``eta``.

    The old profile vanished both at the numerical boundary and in the core.
    Together with ``eta=0`` in the dead zone this left several cell layers in
    which neither representation controlled the solution -- precisely the
    undamped cavity that generated the observed interface reflections.

    This profile is the actual partition-of-authority complement:

        lambda/lambda_max = 1 - eta

    It is therefore constant at full strength in the dead zone, falls with a
    C1 cosine ramp through the overlap, and is exactly zero in the FVM core::

        lambda = lambda_max                                  d <= dead_zone
        lambda = lambda_max * 0.5 * (1 + cos(pi*s))          dead_zone < d < B
        lambda = 0                                           d >= B

    Boundary velocity and the adjacent volume forcing now approach the same
    VPM target continuously; there is no unowned strip and no abrupt source
    turn-on one or more cells inside the domain.
    """
    x = np.atleast_2d(cell_centres)
    xmin, xmax, ymin, ymax, zmin, zmax = fvm_box
    lo = np.array([xmin, ymin, zmin])
    hi = np.array([xmax, ymax, zmax])
    d_in = np.minimum(x - lo, hi - x).min(axis=1)  # signed distance inward to nearest face

    dz = max(dead_zone, 0.0)
    buf_width = max(buffer_thickness - dz, 1e-30)
    lam = np.zeros(len(d_in), dtype=np.float64)
    lam[d_in <= dz] = lambda_max
    active = (d_in > dz) & (d_in < buffer_thickness)
    if active.any():
        s = (d_in[active] - dz) / buf_width
        lam[active] = lambda_max * 0.5 * (1.0 + np.cos(np.pi * s))

    return lam


def lambda_max_from_scales(
    u_char: float, buffer_thickness: float, dt: float, A: float = 4.0
) -> float:
    """Pick lambda_max so the integrated damping a structure sees while convecting
    through the band, ~ lambda_max * L_buffer / u_char, is O(A) (a few).  Capped so
    the implicit time-constant is not absurdly stiffer than needed.

    A in [2, 10] works; start at 4.  Larger A -> stronger matching but stiffer;
    too small leaves residual reflection.  Because the term is implicit
    (fvm::Sp), stiffness is safe, but keep tau = 1/lambda_max >~ dt for accuracy.
    """
    lam_conv = A * u_char / max(buffer_thickness, 1e-12)
    lam_cap = 1.0 / max(dt, 1e-12)  # tau ~ dt floor
    return float(min(lam_conv, lam_cap))


class FringeFields:
    """Owns lambda (static) and pushes Utarget each step."""

    def __init__(self, cfg, vpm, ofw, *, coupling_dt: float):
        self.cfg = cfg
        self.vpm = vpm
        self.ofw = ofw
        self.cc = np.asarray(ofw.get_cell_center_coordinates(), float).reshape(-1, 3)

        u_char = float(np.linalg.norm(cfg.U_inf))
        lam_max = lambda_max_from_scales(
            u_char, cfg.buffer_thickness, coupling_dt, A=cfg.fringe_strength
        )
        dead_zone = float(cfg.dead_zone_h) * float(cfg.h)
        self.lam = build_lambda(
            self.cc,
            cfg.fvm_box,
            cfg.buffer_thickness,
            lam_max,
            dead_zone=dead_zone,
        )

        self.ofw.set_cell_scalar_field("lambdaRelax", np.ascontiguousarray(self.lam))

        # Report the strength that was actually built.  The damping a structure
        # sees crossing the band is exp(-∫λ dd / u_char), and with the
        # dead-zone-plus-cosine profile ∫λ dd = λ_max·(dead_zone + B/2 -
        # dead_zone/2) — NOT λ_max·B as the `A` heuristic assumes.  Without this
        # line the effective fringe strength is invisible in the log, which makes
        # an A/B on `fringe_strength` unverifiable after the fact.
        band_integral = lam_max * (dead_zone + 0.5 * (cfg.buffer_thickness - dead_zone))
        attenuation = float(np.exp(-band_integral / max(u_char, 1e-30)))
        n_band = int(np.count_nonzero(self.lam > 0.0))
        _logger.info(
            "[Fringe] A=%.3g  λ_max=%.3e 1/s (τ=%.3e s, τ/dt=%.1f)  band=%.3f m "
            "(dead_zone=%.3f)  cells in band=%d  transit attenuation=%.3f",
            float(cfg.fringe_strength),
            lam_max,
            1.0 / max(lam_max, 1e-30),
            (1.0 / max(lam_max, 1e-30)) / max(coupling_dt, 1e-30),
            float(cfg.buffer_thickness),
            dead_zone,
            n_band,
            attenuation,
        )

    def update_target(self) -> None:
        """Sample the VPM velocity at the FVM cell centres and push Utarget.

        Only cells with lambda>0 (the band) actually use it, so to save work we
        evaluate VPM only there and leave the core as freestream (unused).
        """
        band = self.lam > 0.0
        Ut = np.tile(self.cfg.U_inf, (len(self.cc), 1)).astype(float)
        if band.any():
            # full VPM field (freestream + ALL particles) at band cell centres:
            # in the buffer VPM is authoritative, so relaxing FVM -> VPM enforces
            # agreement exactly where eta -> 0.
            Ut[band] = self.vpm.compute_target_velocities(
                self.cc[band], include_freestream=True, zone_mask=None
            )
        self.ofw.set_cell_vector_field(
            "Utarget",
            np.ascontiguousarray(Ut[:, 0]),
            np.ascontiguousarray(Ut[:, 1]),
            np.ascontiguousarray(Ut[:, 2]),
        )
