"""
fvm_fringe.py  --  build and update the fringe-relaxation fields for the FVM.

The coupler calls this to push two volume fields into OpenFOAM each run:

  * `lambdaRelax`  (volScalarField, set ONCE): the relaxation rate, 0 in the FVM
    core and ramping smoothly to `lambda_max` at the numerical boundary, over the
    buffer band.  This is the FVM-side complement of the VPM-side authority weight
    eta -- where eta -> 0 (VPM takes over), lambda -> max (FVM defers to VPM).

  * `Utarget`  (volVectorField, set EVERY coupling step): the VPM velocity sampled
    at the FVM cell centres, i.e. the field the fringe relaxes toward.

Both are looked up by name inside the solver (relaxationSource.H / fvOptions).
"""

from __future__ import annotations

import numpy as np


def build_lambda(
    cell_centres: np.ndarray,
    fvm_box: tuple[float, float, float, float, float, float],
    buffer_thickness: float,
    lambda_max: float,
) -> np.ndarray:
    """Static relaxation rate per cell.

    lambda(d) = lambda_max * 0.5 * (1 + cos(pi * s)),   s = clip(d / L_buffer, 0, 1)
    where d is the distance from the cell centre INWARD to the nearest box face.
    lambda = lambda_max at the face (d=0), 0 once d >= buffer_thickness (the core).
    Raised cosine => C^1 smooth => no impedance jump => no reflection.
    """
    x = np.atleast_2d(cell_centres)
    xmin, xmax, ymin, ymax, zmin, zmax = fvm_box
    lo = np.array([xmin, ymin, zmin])
    hi = np.array([xmax, ymax, zmax])
    d_in = np.minimum(x - lo, hi - x).min(axis=1)  # distance to nearest face
    s = np.clip(d_in / max(buffer_thickness, 1e-12), 0.0, 1.0)
    lam = lambda_max * 0.5 * (1.0 + np.cos(np.pi * s))
    lam[d_in >= buffer_thickness] = 0.0  # core: FVM is free
    lam[d_in < 0.0] = lambda_max  # (shouldn't happen) outside
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
            u_char, cfg.buffer_thickness, cfg.dt, A=getattr(cfg, "fringe_strength", 4.0)
        )
        self.lam = build_lambda(self.cc, cfg.fvm_box, cfg.buffer_thickness, lam_max)

        # set the static lambda once
        self.ofw.set_cell_scalar_field("lambdaRelax", np.ascontiguousarray(self.lam))

    def update_target(self) -> None:
        """Sample the VPM velocity at the FVM cell centres and push as Utarget.

        Only cells with lambda>0 (the band) actually use it, so to save work we
        evaluate VPM only there and leave the core as freestream (unused)."""
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
