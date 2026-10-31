"""Boundary data of the outer FVM boundary, evaluated from the particles."""

from __future__ import annotations

import logging
import time

import numpy as np

from openonda.core.simulation.parallel import collective_phase

from .reporting import boundary_induction_summary, format_coupler_log

logger = logging.getLogger("openonda.coupler")

_MAX_BOUNDARY_FLUX_TOLERANCE = 1.0e-2
_BOUNDARY_FLUX_QUADRATURE_FACTOR = 2.0

INFLOW_TRANSITION_FRACTION = 0.1
"""Normal outflow speed, as a fraction of the reference speed, above which a
coupling face is purely mixed; inflow faces always take the particle velocity."""


def inflow_transition_speed(coupler, face_velocity: np.ndarray) -> float:
    """Return the normal speed (m/s) over which coupling faces switch from Dirichlet to mixed.

    The reference speed is the larger of the freestream speed and the RMS boundary
    speed, so that cases without a freestream still have a meaningful scale. The
    root rank's value is broadcast, keeping every rank's face weights identical.
    """
    value = None
    if coupler._is_root:
        velocity = np.asarray(face_velocity, dtype=np.float64).reshape(-1, 3)
        rms = (
            float(np.sqrt(np.mean(np.einsum("ij,ij->i", velocity, velocity))))
            if len(velocity)
            else 0.0
        )
        reference = max(float(np.linalg.norm(coupler.freestream_velocity)), rms)
        value = INFLOW_TRANSITION_FRACTION * max(reference, 1.0e-12)
    comm = getattr(coupler, "_comm", None)
    if comm is not None and comm.Get_size() > 1:
        value = comm.bcast(value, root=0)
    return float(value)


def _record_boundary_induction(coupler, phase: str, summary: dict) -> None:
    if phase == "initial":
        coupler._last_boundary_induction_diagnostics = {"initial": summary}
        return
    diagnostics = getattr(coupler, "_last_boundary_induction_diagnostics", None)
    if diagnostics is None:
        diagnostics = coupler._last_boundary_induction_diagnostics = {}
    refresh = diagnostics.setdefault("refresh", {"calls": 0, "execution_backends": {}})
    refresh["calls"] += 1
    backend = summary["execution_backend"] or "unspecified"
    counts = refresh["execution_backends"]
    counts[backend] = counts.get(backend, 0) + 1
    for name, value in summary.items():
        if name.endswith("_seconds"):
            refresh[name] = refresh.get(name, 0.0) + value
    refresh["last"] = summary


def outflow_axis_sign(freestream_velocity: np.ndarray) -> tuple[int, float]:
    """Return the coordinate axis and sign most closely aligned with the free stream."""
    velocity = np.asarray(freestream_velocity, dtype=np.float64).reshape(-1)
    if velocity.size != 3 or not np.any(velocity != 0.0):
        return 0, 1.0
    axis = int(np.argmax(np.abs(velocity)))
    return axis, float(np.sign(velocity[axis]))


def _log_outflow_velocity(
    face_centre: np.ndarray,
    velocity: np.ndarray,
    *,
    freestream_velocity: np.ndarray,
    fvm_domain_bounds: np.ndarray,
) -> None:
    """Log the streamwise velocity on the downstream face of the outer FVM boundary."""
    axis, sign = outflow_axis_sign(freestream_velocity)
    freestream_speed = float(np.linalg.norm(freestream_velocity)) + 1.0e-30
    face_lo, face_hi = fvm_domain_bounds[2 * axis], fvm_domain_bounds[2 * axis + 1]
    if sign >= 0.0:
        mask = face_centre[:, axis] >= face_hi - 1.0e-6
    else:
        mask = face_centre[:, axis] <= face_lo + 1.0e-6
    if not mask.any():
        return
    streamwise_velocity = velocity[mask] @ (np.asarray(freestream_velocity) / freestream_speed)
    face_name = f"{'xyz'[axis]}{'+' if sign >= 0.0 else '-'}"
    logger.info(
        format_coupler_log(
            f"vpm target at outflow, face {face_name}",
            ("faces", f"{int(mask.sum()):,}"),
            (
                "streamwise velocity, min",
                f"{streamwise_velocity.min() / freestream_speed:.3f}",
                "U_inf",
            ),
            (
                "streamwise velocity, mean",
                f"{streamwise_velocity.mean() / freestream_speed:.3f}",
                "U_inf",
            ),
            (
                "streamwise velocity, max",
                f"{streamwise_velocity.max() / freestream_speed:.3f}",
                "U_inf",
            ),
        )
    )


COUPLING_PATCH_SKEW_LIMIT = 0.05
"""Largest owner-to-face tangential offset, relative to the normal distance, of a
coupling-grade outer patch."""


def report_coupling_patch_quality(
    coupler, face_normal: np.ndarray, owner_offset: np.ndarray
) -> dict[str, float] | None:
    """Log the outer patch's geometric quality and warn if it is not coupling-grade.

    A coupling-grade patch has planar faces and axis-aligned owner cells, so the
    boundary closure does not extrapolate across a skewed cell and the
    particle-mesh transfer sees a planar interface. ``CartesianMesher`` produces
    one when the box faces are declared as coupling patches. The direction-aware
    boundary condition is stable without it, but accuracy degrades.
    """
    if not coupler._is_root or not len(face_normal):
        return None
    normal = np.asarray(face_normal, dtype=np.float64).reshape(-1, 3)
    offset = np.asarray(owner_offset, dtype=np.float64).reshape(-1, 3)
    distance = np.einsum("ij,ij->i", offset, normal)
    tangential = np.linalg.norm(offset - distance[:, None] * normal, axis=1)
    ratio = tangential / np.maximum(distance, 1.0e-300)
    tilt = 1.0 - np.max(np.abs(normal), axis=1)
    worst = int(np.argmax(ratio))
    quality = {
        "max_tangential_offset_ratio": float(ratio[worst]),
        "max_normal_tilt": float(tilt.max()),
        "normal_distance_ratio": float(distance.min() / distance.max()),
    }
    coupler._coupling_patch_quality = quality
    logger.info(
        format_coupler_log(
            "outer boundary cells",
            (
                "owner offset, tangential / normal, max",
                f"{quality['max_tangential_offset_ratio']:.3e}",
            ),
            ("normal tilt from box axis, max", f"{quality['max_normal_tilt']:.3e}"),
            ("normal distance, min / max", f"{quality['normal_distance_ratio']:.3f}"),
        )
    )
    if quality["max_tangential_offset_ratio"] > COUPLING_PATCH_SKEW_LIMIT:
        logger.warning(
            format_coupler_log(
                "outer boundary cells are not coupling-grade",
                (
                    "owner offset, tangential / normal, max",
                    f"{quality['max_tangential_offset_ratio']:.3f} (limit {COUPLING_PATCH_SKEW_LIMIT:g})",
                ),
                ("normal tilt from box axis, max", f"{quality['max_normal_tilt']:.3e}"),
                (
                    "remedy",
                    "mesh the box with BoxPatches(..., coupling=(name,)) or BoxPatches.coupled(name)",
                ),
            )
        )
    return quality


def boundary_flux_tolerance(particle_spacing: float, fvm_domain_bounds: np.ndarray) -> float:
    """Return a bounded, second-order tolerance for discrete boundary flux.

    The face-centre boundary data form a second-order surface quadrature, so the
    normalized net flux through the outer FVM boundary scales as ``(h/L_FVM)²``.
    Its leading coefficient depends on the boundary velocity field; a factor of
    two allows that coefficient to exceed one while the one-percent ceiling
    still stops the run when the particle field has a substantial net flux.
    """
    spacing = float(particle_spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("particle_spacing must be finite and positive")
    bounds = np.asarray(fvm_domain_bounds, dtype=np.float64).reshape(6)
    extent = bounds[1::2] - bounds[::2]
    length = float(np.min(extent))
    if length <= 0.0:
        raise ValueError("the FVM domain extents must be positive")
    second_order = _BOUNDARY_FLUX_QUADRATURE_FACTOR * (spacing / length) ** 2
    return float(
        max(
            4096.0 * np.finfo(float).eps,
            min(second_order, _MAX_BOUNDARY_FLUX_TOLERANCE),
        )
    )


def tangential_velocity_normal_gradient(
    target_velocity_gradient: np.ndarray, face_normal: np.ndarray
) -> np.ndarray:
    r"""Return the normal gradient of the tangential velocity, ``g_t = P_t G n``, on each face.

    The VPM velocity gradient uses ``G[i,j] = d(u_i)/d(x_j)``, hence ``du/dn = G n``,
    and ``P_t = J - n nᵀ`` removes its normal component.
    """
    normal = np.asarray(face_normal, dtype=np.float64).reshape(-1, 3)
    jacobian = np.asarray(target_velocity_gradient, dtype=np.float64)
    if jacobian.size != 9 * len(normal):
        raise ValueError("VPM velocity-gradient count does not match boundary faces")
    jacobian = jacobian.reshape(-1, 3, 3)
    if not np.all(np.isfinite(jacobian)):
        raise RuntimeError("VPM target-gradient evaluation returned non-finite data")
    normal_velocity_gradient = np.einsum("fij,fj->fi", jacobian, normal)
    return (
        normal_velocity_gradient
        - np.einsum("fi,fi->f", normal_velocity_gradient, normal)[:, np.newaxis] * normal
    )


def evaluate_vpm_velocity(
    vpm,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    face_area: np.ndarray,
    *,
    freestream_velocity: np.ndarray,
    fvm_domain_bounds: np.ndarray,
    particle_spacing: float,
    evaluated_velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate the particle velocity on the outer FVM boundary and correct its net flux.

    Only a net volume flux within the tolerance of :func:`boundary_flux_tolerance`
    is corrected, by a uniform normal velocity. Face coordinates are in metres,
    velocities in m/s, normals are unit vectors and areas are in m². The returned
    face velocity has shape ``(N, 3)``.
    """
    normal = np.asarray(face_normal, dtype=np.float64).reshape(-1, 3)
    areas = np.asarray(face_area, dtype=np.float64).reshape(-1)
    if evaluated_velocity is None:
        evaluated_velocity = vpm.compute_velocity_at_points(
            face_centre,
            include_freestream=True,
            zone_mask=None,
            include_body=True,
        )
    velocity = np.asarray(evaluated_velocity, dtype=np.float64).reshape(-1, 3)
    if len(velocity) != len(face_centre):
        raise ValueError(
            "evaluated VPM boundary-condition velocity count does not match boundary faces"
        )

    raw_flux = 0.0
    raw_relative = 0.0
    correction = 0.0
    corrected_flux = 0.0
    tolerance = boundary_flux_tolerance(particle_spacing, fvm_domain_bounds)
    total_area = float(np.sum(areas))
    if areas.size:
        raw_flux = float(np.dot(np.einsum("ij,ij->i", velocity, normal), areas))
        if total_area > 0.0:
            freestream_speed = float(np.linalg.norm(freestream_velocity))
            velocity_scale = max(
                freestream_speed,
                float(np.sqrt(np.mean(np.einsum("ij,ij->i", velocity, velocity)))),
                np.finfo(float).tiny,
            )
            scale = velocity_scale * total_area
            raw_relative = abs(raw_flux) / scale
            if raw_relative > tolerance:
                raise RuntimeError(
                    "The particle velocity has a significant net flux through the outer FVM "
                    f"boundary: |integral(u.n dA)|/(U_scale A_FVM)={raw_relative:.3e}, "
                    f"tolerance={tolerance:.3e}. The flux correction would hide an error "
                    "of the particle velocity."
                )
            correction = raw_flux / total_area
            velocity = velocity - correction * normal
        corrected_flux = float(np.dot(np.einsum("ij,ij->i", velocity, normal), areas))
        logger.info(
            format_coupler_log(
                "vpm boundary flux",
                ("particles", f"{int(vpm.particles.n_particles_total):,}"),
                ("raw flux", f"{raw_flux:.3e}", "m^3/s"),
                ("relative to boundary velocity scale", f"{raw_relative:.3e}"),
                ("acceptance limit", f"{tolerance:.3e}"),
                ("correction, normal velocity", f"{correction:.3e}", "m/s"),
                ("residual flux", f"{corrected_flux:.3e}", "m^3/s"),
            )
        )

    _log_outflow_velocity(
        face_centre,
        velocity,
        freestream_velocity=np.asarray(freestream_velocity, dtype=np.float64),
        fvm_domain_bounds=np.asarray(fvm_domain_bounds, dtype=np.float64),
    )
    diagnostics = {
        "raw_mismatch": abs(raw_flux),
        "raw_relative": raw_relative,
        "acceptance_limit": tolerance,
        "applied_correction": abs(correction),
        "corrected_mismatch": abs(corrected_flux),
    }
    return velocity, diagnostics


def evaluate_vpm_boundary(
    coupler,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    face_area: np.ndarray,
):
    """Evaluate predictor boundary data at the end of the shared time step."""
    boundary_condition_wall_time = time.perf_counter()
    vpm_boundary_condition_velocity = None
    normal_gradient: np.ndarray | None = None
    with collective_phase(coupler._comm, "VPM boundary evaluation"):
        if coupler._is_root:
            assert coupler.vpm_solver is not None
            query_started = time.perf_counter()
            vpm_boundary_condition_velocity, normal_gradient = (
                coupler.vpm_solver.compute_velocity_and_tangential_normal_gradient_at_points(
                    face_centre, face_normal, particle_spacing=coupler.vpm_particle_spacing
                )
            )
            _record_boundary_induction(
                coupler,
                "initial",
                boundary_induction_summary(
                    coupler.vpm_solver,
                    target_count=len(face_centre),
                    wall_seconds=time.perf_counter() - query_started,
                ),
            )
            vpm_boundary_condition_velocity = np.asarray(
                vpm_boundary_condition_velocity, dtype=np.float64
            ).reshape(-1, 3)
            if vpm_boundary_condition_velocity.shape != face_centre.shape:
                raise RuntimeError(
                    "VPM target evaluation returned an invalid shape: "
                    f"expected {face_centre.shape}, got {vpm_boundary_condition_velocity.shape}"
                )
            if not np.all(np.isfinite(vpm_boundary_condition_velocity)):
                raise RuntimeError("VPM target evaluation returned non-finite velocities")
            freestream_speed = float(np.linalg.norm(coupler.freestream_velocity))
            if (
                len(face_centre) > 0
                and freestream_speed > 0.0
                and float(np.max(np.linalg.norm(vpm_boundary_condition_velocity, axis=1)))
                <= 1.0e-6 * freestream_speed
            ):
                raise RuntimeError(
                    "VPM target evaluation returned an identically zero field despite a "
                    "nonzero freestream; aborting before the corrupted VPM boundary-condition data reaches the FVM"
                )
        if coupler._is_root:
            assert coupler.fvm_domain_bounds is not None
            velocity_boundary_condition, coupler._last_vpm_boundary_condition_flux_diagnostics = (
                evaluate_vpm_velocity(
                    coupler.vpm_solver,
                    face_centre,
                    face_normal,
                    face_area,
                    freestream_velocity=coupler.freestream_velocity,
                    fvm_domain_bounds=coupler.fvm_domain_bounds,
                    particle_spacing=coupler.vpm_particle_spacing,
                    evaluated_velocity=vpm_boundary_condition_velocity,
                )
            )
            # The FVM takes inertial-frame velocities on its moving mesh.
            velocity_boundary_condition = coupler.inertial_boundary_velocity(
                velocity_boundary_condition
            )
            if coupler._velocity_boundary_condition_old is None:
                coupler._velocity_boundary_condition_old = velocity_boundary_condition.copy()
            assert normal_gradient is not None
            normal_velocity_boundary_condition = np.einsum(
                "ij,ij->i", velocity_boundary_condition, face_normal
            )
            normal_gradient_boundary_condition = normal_gradient
            coupler._normal_velocity_boundary_condition = normal_velocity_boundary_condition
            coupler._normal_gradient_boundary_condition = normal_gradient_boundary_condition
            if coupler._normal_velocity_boundary_condition_old is None:
                coupler._normal_velocity_boundary_condition_old = (
                    normal_velocity_boundary_condition.copy()
                )
            if coupler._normal_gradient_boundary_condition_old is None:
                coupler._normal_gradient_boundary_condition_old = (
                    normal_gradient_boundary_condition.copy()
                )
        else:
            velocity_boundary_condition = np.zeros_like(face_centre)
            if coupler._velocity_boundary_condition_old is None:
                coupler._velocity_boundary_condition_old = np.zeros_like(face_centre)
            coupler._normal_velocity_boundary_condition = np.zeros(
                len(face_centre), dtype=np.float64
            )
            coupler._normal_gradient_boundary_condition = np.zeros_like(face_centre)
            if coupler._normal_velocity_boundary_condition_old is None:
                coupler._normal_velocity_boundary_condition_old = (
                    coupler._normal_velocity_boundary_condition.copy()
                )
            if coupler._normal_gradient_boundary_condition_old is None:
                coupler._normal_gradient_boundary_condition_old = (
                    coupler._normal_gradient_boundary_condition.copy()
                )
    boundary_condition_wall_time = time.perf_counter() - boundary_condition_wall_time
    return (
        coupler._velocity_boundary_condition_old,
        velocity_boundary_condition,
        boundary_condition_wall_time,
    )


def initialize_vpm_boundary_history(
    coupler,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    face_area: np.ndarray,
) -> None:
    """Evaluate the start-of-interval boundary data ``b^n`` before the first VPM advance."""
    needs_boundary_history = coupler._velocity_boundary_condition_old is None
    # Every rank enters the same boundary evaluation after restart.
    comm = coupler._comm
    if comm is not None and comm.Get_size() > 1:
        needs_boundary_history = comm.bcast(
            needs_boundary_history if coupler._is_root else None, root=0
        )
    if not needs_boundary_history:
        return
    evaluate_vpm_boundary(coupler, face_centre, face_normal, face_area)
    logger.info(format_coupler_log("coupling history, initial time level stored"))


def advance_fvm(
    coupler,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    face_area: np.ndarray,
    velocity_boundary_condition: np.ndarray,
) -> float:
    """Advance integer FVM substeps with boundary data interpolated over the VPM interval."""
    fvm_wall_time_start = time.perf_counter()
    previous_velocity = coupler._velocity_boundary_condition_old
    previous_normal_velocity = coupler._normal_velocity_boundary_condition_old
    previous_normal_gradient = coupler._normal_gradient_boundary_condition_old
    next_normal_velocity = coupler._normal_velocity_boundary_condition
    next_normal_gradient = coupler._normal_gradient_boundary_condition
    if any(
        values is None
        for values in (
            previous_velocity,
            previous_normal_velocity,
            previous_normal_gradient,
            next_normal_velocity,
            next_normal_gradient,
        )
    ):
        raise RuntimeError("FVM sub-cycling received incomplete VPM boundary data")

    fvm = coupler.fvm_solver
    interval_start_time = fvm.time
    interval_end_time = coupler.coupling_time(coupler.step + 1)
    interval_duration = interval_end_time - interval_start_time
    end_tolerance = max(1.0e-14, abs(interval_end_time) * 1.0e-12)
    substeps = 0
    while interval_end_time - fvm.time > end_tolerance:
        proposed_step = fvm.select_next_time_step_size(
            additional_event_schedules=coupler._coupling_event_schedules(),
            stop_time=interval_end_time,
        )
        remaining = interval_end_time - fvm.time
        accepted_time = (
            interval_end_time
            if remaining <= proposed_step + end_tolerance
            else fvm.time + proposed_step
        )
        coupler.fvm_time_step_size = accepted_time - fvm.time
        fraction = (accepted_time - interval_start_time) / interval_duration
        interpolated_velocity = (
            1.0 - fraction
        ) * previous_velocity + fraction * velocity_boundary_condition
        coupler._normal_velocity_boundary_condition = (
            1.0 - fraction
        ) * previous_normal_velocity + fraction * next_normal_velocity
        coupler._normal_gradient_boundary_condition = (
            1.0 - fraction
        ) * previous_normal_gradient + fraction * next_normal_gradient
        advance_fvm_step(
            coupler,
            coupler.config.outer_boundary_patch,
            interpolated_velocity,
            accepted_time=accepted_time,
        )
        substeps += 1
    coupler.n_fvm_substeps = substeps
    _record_fvm_boundary_data(
        coupler,
        face_centre,
        face_normal,
        velocity_boundary_condition,
    )
    if coupler._is_root:
        coupler._velocity_boundary_condition_old = velocity_boundary_condition
        coupler._normal_velocity_boundary_condition_old = next_normal_velocity
        coupler._normal_gradient_boundary_condition_old = next_normal_gradient
    else:
        coupler._velocity_boundary_condition_old = velocity_boundary_condition
        coupler._normal_velocity_boundary_condition_old = next_normal_velocity
        coupler._normal_gradient_boundary_condition_old = next_normal_gradient
    return time.perf_counter() - fvm_wall_time_start


def update_boundary_history_after_replacement(
    coupler,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    face_area: np.ndarray,
) -> None:
    """Evaluate the start-of-interval boundary data from the particles after the transfer.

    Otherwise each coupling interval would start from the predictor data. The FVM
    is not solved again.
    """
    if coupler._is_root:
        assert coupler.vpm_solver is not None
        normal_gradient: np.ndarray | None = None
        query_started = time.perf_counter()
        corrected_boundary, normal_gradient = (
            coupler.vpm_solver.compute_velocity_and_tangential_normal_gradient_at_points(
                face_centre, face_normal, particle_spacing=coupler.vpm_particle_spacing
            )
        )
        _record_boundary_induction(
            coupler,
            "refresh",
            boundary_induction_summary(
                coupler.vpm_solver,
                target_count=len(face_centre),
                wall_seconds=time.perf_counter() - query_started,
            ),
        )
        corrected_boundary = np.asarray(corrected_boundary, dtype=np.float64).reshape(-1, 3)
        if corrected_boundary.shape != face_centre.shape or not np.all(
            np.isfinite(corrected_boundary)
        ):
            raise RuntimeError(
                "VPM boundary-condition resynchronisation returned invalid velocities"
            )

        assert coupler.fvm_domain_bounds is not None
        corrected_boundary, coupler._last_vpm_boundary_condition_flux_diagnostics = (
            evaluate_vpm_velocity(
                coupler.vpm_solver,
                face_centre,
                face_normal,
                face_area,
                freestream_velocity=coupler.freestream_velocity,
                fvm_domain_bounds=coupler.fvm_domain_bounds,
                particle_spacing=coupler.vpm_particle_spacing,
                evaluated_velocity=corrected_boundary,
            )
        )
        corrected_boundary = coupler.inertial_boundary_velocity(corrected_boundary)
        freestream_speed = float(np.linalg.norm(coupler.freestream_velocity)) + 1e-30
        drift = (
            float(
                np.max(
                    np.linalg.norm(
                        corrected_boundary - coupler._velocity_boundary_condition_old, axis=1
                    )
                )
            )
            / freestream_speed
            if coupler._velocity_boundary_condition_old is not None and len(face_centre)
            else 0.0
        )
        coupler._velocity_boundary_condition_old = corrected_boundary
        assert normal_gradient is not None
        coupler._normal_velocity_boundary_condition_old = np.einsum(
            "ij,ij->i", corrected_boundary, face_normal
        )
        coupler._normal_gradient_boundary_condition_old = normal_gradient
        logger.info(
            format_coupler_log(
                "boundary update, post transfer",
                ("velocity difference, max", f"{drift:.3e}", "U_inf"),
            )
        )


def _record_fvm_boundary_data(
    coupler,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    vpm_target_velocity: np.ndarray,
) -> None:
    """Record the FVM face velocity on the outer FVM boundary and its mismatch to the data."""
    assert coupler.fvm_solver is not None
    face_velocity = coupler.fvm_solver.gather_boundary_face_velocity(
        coupler.config.outer_boundary_patch
    )
    failure = None
    if coupler._is_root:
        failure = _record_root_boundary_data(
            coupler, face_centre, face_normal, vpm_target_velocity, face_velocity
        )
    comm = getattr(coupler, "_comm", None)
    if comm is not None and comm.Get_size() > 1:
        failure = comm.bcast(failure, root=0)
    if failure is not None:
        raise BoundaryMismatchError(failure)


class BoundaryMismatchError(RuntimeError):
    """The FVM boundary velocity departed from the particle data by an unphysical amount."""


def _mismatch_location(
    face_centre: np.ndarray, face_normal: np.ndarray, difference: np.ndarray, index: int
) -> str:
    normal = face_normal[index]
    axis = int(np.argmax(np.abs(normal)))
    side = ("x", "y", "z")[axis] + ("max" if normal[axis] > 0 else "min")
    return (
        f"face at {np.array2string(face_centre[index], precision=4)} m ({side} side), "
        f"velocity difference {np.array2string(difference[index], precision=4)} m/s"
    )


def _check_boundary_mismatch(
    coupler,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    difference: np.ndarray,
    magnitude: np.ndarray,
    reference_speed: float,
) -> str | None:
    """Warn on, or report a stop for, an unphysical boundary mismatch."""
    config = coupler.config
    if not len(magnitude):
        return None
    index = int(np.argmax(magnitude))
    ratio = float(magnitude[index] / reference_speed)
    limit = config.boundary_mismatch_limit
    if limit is not None and ratio > limit:
        return (
            f"The FVM velocity on the outer boundary differs from the particle velocity by "
            f"{ratio:.3g} times the reference speed {reference_speed:.4g} m/s (limit {limit:g}) "
            f"at t = {coupler.time:.6g} s: "
            + _mismatch_location(face_centre, face_normal, difference, index)
            + ". The FVM boundary closure or the boundary data are unstable there; check the "
            "outer-patch cells (mesh the box with coupling patches) and the coupling interval."
        )
    warning = config.boundary_mismatch_warning
    if warning is not None and ratio > warning:
        # One warning per doubling keeps a slowly drifting case readable.
        level = int(np.floor(np.log2(ratio / warning)))
        if level > getattr(coupler, "_boundary_mismatch_warning_level", -1):
            coupler._boundary_mismatch_warning_level = level
            logger.warning(
                format_coupler_log(
                    "boundary mismatch",
                    ("velocity mismatch, max", f"{ratio:.3e}", "U_ref"),
                    ("location", _mismatch_location(face_centre, face_normal, difference, index)),
                )
            )
    return None


def _record_root_boundary_data(
    coupler,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    vpm_target_velocity: np.ndarray,
    face_velocity: np.ndarray,
) -> str | None:
    difference = np.asarray(face_velocity) - np.asarray(vpm_target_velocity)
    speed = float(np.linalg.norm(coupler.freestream_velocity)) + 1.0e-30
    magnitude = np.linalg.norm(difference, axis=1)
    target = np.asarray(vpm_target_velocity, dtype=np.float64).reshape(-1, 3)
    reference_speed = max(
        float(np.linalg.norm(coupler.freestream_velocity)),
        float(np.sqrt(np.mean(np.einsum("ij,ij->i", target, target)))) if len(target) else 0.0,
        1.0e-12,
    )
    normal_mismatch = np.abs(np.einsum("ij,ij->i", difference, face_normal))
    axis, sign = outflow_axis_sign(coupler.freestream_velocity)
    assert coupler.fvm_domain_bounds is not None
    boundary_coordinate = coupler.fvm_domain_bounds[2 * axis + (1 if sign >= 0.0 else 0)]
    if sign >= 0.0:
        outflow = face_centre[:, axis] >= boundary_coordinate - 1.0e-6
    else:
        outflow = face_centre[:, axis] <= boundary_coordinate + 1.0e-6
    outflow_magnitude = magnitude[outflow]
    diagnostics = {
        "mean_velocity_mismatch": float(np.mean(magnitude) / speed) if len(magnitude) else 0.0,
        "maximum_velocity_mismatch": float(np.max(magnitude) / speed) if len(magnitude) else 0.0,
        "maximum_normal_velocity_mismatch": (
            float(np.max(normal_mismatch) / speed) if len(normal_mismatch) else 0.0
        ),
        "mean_outflow_velocity_mismatch": (
            float(np.mean(outflow_magnitude) / speed) if len(outflow_magnitude) else 0.0
        ),
        "maximum_outflow_velocity_mismatch": (
            float(np.max(outflow_magnitude) / speed) if len(outflow_magnitude) else 0.0
        ),
    }
    coupler._last_fvm_boundary_data_diagnostics = diagnostics
    logger.info(
        format_coupler_log(
            "fvm boundary data",
            ("velocity mismatch, mean", f"{diagnostics['mean_velocity_mismatch']:.3e}", "U_inf"),
            (
                "velocity mismatch, max",
                f"{diagnostics['maximum_velocity_mismatch']:.3e}",
                "U_inf",
            ),
            (
                "normal mismatch, max",
                f"{diagnostics['maximum_normal_velocity_mismatch']:.3e}",
                "U_inf",
            ),
            (
                "outflow mismatch, mean",
                f"{diagnostics['mean_outflow_velocity_mismatch']:.3e}",
                "U_inf",
            ),
            (
                "outflow mismatch, max",
                f"{diagnostics['maximum_outflow_velocity_mismatch']:.3e}",
                "U_inf",
            ),
        )
    )
    return _check_boundary_mismatch(
        coupler, face_centre, face_normal, difference, magnitude, reference_speed
    )


def apply_fvm_boundary(
    coupler,
    patch: str,
    prescribed_velocity: np.ndarray,
    normal_velocity: np.ndarray | None = None,
    normal_gradient: np.ndarray | None = None,
    *,
    accepted_time: float,
) -> None:
    """Apply the boundary data to the outer FVM boundary and advance one FVM step to ``accepted_time``."""
    assert coupler.fvm_solver is not None
    freestream_velocity = np.asarray(coupler.freestream_velocity, dtype=np.float64)
    freestream_speed = float(np.linalg.norm(freestream_velocity)) + 1e-30
    prescribed_velocity = np.ascontiguousarray(prescribed_velocity, dtype=np.float64)
    if normal_velocity is None or normal_gradient is None:
        raise RuntimeError(
            "The mixed boundary condition requires the normal velocity and the normal "
            "gradient of the tangential velocity"
        )
    # Direction-aware coupling condition: the particle velocity on inflow faces,
    # the mixed condition on outflow faces (see ``mixed_velocity_boundary``).
    coupler.fvm_solver.set_mixed_velocity_boundary_condition(
        np.ascontiguousarray(normal_velocity, dtype=np.float64),
        np.ascontiguousarray(normal_gradient, dtype=np.float64),
        patch,
        inflow_velocity=prescribed_velocity,
        transition_speed=inflow_transition_speed(coupler, prescribed_velocity),
    )
    coupler.fvm_solver.set_flux_consistent_pressure_boundary_condition(patch)
    # With coupling iterations the FVM writes its output once the boundary data are accepted.
    coupler.fvm_solver.advance(
        defer_output=coupler.config.max_coupling_iterations > 1,
        time=accepted_time,
        time_step_size=coupler.fvm_time_step_size,
    )

    if prescribed_velocity.shape[0] > 0:
        streamwise = prescribed_velocity @ (freestream_velocity / freestream_speed)
        logger.info(
            format_coupler_log(
                f"vpm target, fvm step {int(coupler.fvm_solver.step):,}",
                ("streamwise velocity, min", f"{streamwise.min() / freestream_speed:.3f}", "U_inf"),
                (
                    "streamwise velocity, mean",
                    f"{streamwise.mean() / freestream_speed:.3f}",
                    "U_inf",
                ),
                ("streamwise velocity, max", f"{streamwise.max() / freestream_speed:.3f}", "U_inf"),
            )
        )


def advance_fvm_step(
    coupler,
    patch: str,
    velocity_boundary_condition: np.ndarray,
    *,
    accepted_time: float | None = None,
) -> None:
    """Advance one shared FVM--VPM step using the VPM data at its end time."""
    normal_velocity = coupler._normal_velocity_boundary_condition
    normal_gradient = coupler._normal_gradient_boundary_condition
    if normal_velocity is None or normal_gradient is None:
        raise RuntimeError("The FVM step requires complete VPM boundary data")
    apply_fvm_boundary(
        coupler,
        patch,
        velocity_boundary_condition,
        normal_velocity=normal_velocity,
        normal_gradient=normal_gradient,
        accepted_time=(
            coupler.time + coupler.fvm_time_step_size if accepted_time is None else accepted_time
        ),
    )
