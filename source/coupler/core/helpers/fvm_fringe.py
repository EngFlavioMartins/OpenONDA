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

import numpy as np


def build_lambda(
    cell_centres: np.ndarray,
    fvm_box: tuple[float, float, float, float, float, float],
    buffer_thickness: float,
    lambda_max: float,
    dead_zone: float = 0.0,
) -> np.ndarray:
    """Static relaxation rate per cell, respecting the hand-off dead zone.

    The fringe is the FVM-side complement of the VPM-side η weight: it forces
    the FVM velocity toward the VPM field in the transition band.  However, the
    dead zone (the thin strip at every FVM face where η = 0) must be excluded
    from the fringe as well: VPM particles in the dead zone carry stale
    (Lagrangian, uncorrected) vorticity and are not forced toward the FVM, so
    forcing the FVM toward their velocity would introduce a spurious one-way
    coupling at exactly the faces where the two descriptions should be
    decoupled.

    The corrected profile uses a sin² ramp restricted to the buffer zone
    (dead_zone < d < buffer_thickness):

        λ = 0                                    d ≤ dead_zone  (dead zone)
        λ = λ_max · sin²(π·s),  s=(d−dz)/(B−dz)  dead_zone < d < buffer_thickness
        λ = 0                                    d ≥ buffer_thickness (FVM core)

    This profile is C¹ everywhere (zero value AND zero slope at both ends of
    the ramp), peaks at the midpoint of the buffer zone, and vanishes in both
    the dead zone and the core — matching the region where η is actively
    transitioning (0 < η < 1) and particle corrections are meaningful.
    """
    x = np.atleast_2d(cell_centres)
    xmin, xmax, ymin, ymax, zmin, zmax = fvm_box
    lo = np.array([xmin, ymin, zmin])
    hi = np.array([xmax, ymax, zmax])
    d_in = np.minimum(x - lo, hi - x).min(axis=1)  # signed distance inward to nearest face

    lam = np.zeros(len(d_in), dtype=np.float64)

    dz = max(dead_zone, 0.0)
    buf_width = max(buffer_thickness - dz, 1e-30)
    active = (d_in > dz) & (d_in < buffer_thickness)
    if active.any():
        s = (d_in[active] - dz) / buf_width
        # sin² is C¹, zero at s=0 (dead-zone edge) and s=1 (core boundary),
        # peaks at s=0.5 (midpoint of the transition band).
        lam[active] = lambda_max * np.sin(np.pi * s) ** 2

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

    def __init__(self, cfg, vpm, ofw):
        self.cfg = cfg
        self.vpm = vpm
        self.ofw = ofw
        self.cc = np.asarray(ofw.get_cell_center_coordinates(), float).reshape(-1, 3)

        u_char = float(np.linalg.norm(cfg.U_inf))
        lam_max = lambda_max_from_scales(
            u_char, cfg.buffer_thickness, cfg.dt, A=cfg.fringe_strength
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
