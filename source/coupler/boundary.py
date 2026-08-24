"""Numerical operations for the VPM trace imposed on the FVM boundary."""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger("coupler")


def outflow_axis_sign(freestream_velocity: np.ndarray) -> tuple[int, float]:
    """Return the box axis and sign most closely aligned with the freestream."""
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
    fvm_box: np.ndarray,
) -> None:
    """Log the streamwise velocity on the downstream face of the FVM box."""
    axis, sign = outflow_axis_sign(freestream_velocity)
    freestream_speed = float(np.linalg.norm(freestream_velocity)) + 1.0e-30
    face_lo, face_hi = fvm_box[2 * axis], fvm_box[2 * axis + 1]
    if sign >= 0.0:
        mask = face_centre[:, axis] >= face_hi - 1.0e-6
    else:
        mask = face_centre[:, axis] <= face_lo + 1.0e-6
    if not mask.any():
        return
    streamwise_velocity = velocity[mask] @ (np.asarray(freestream_velocity) / freestream_speed)
    logger.info(
        "[Coupler][BoundaryOutflow] axis=%s sign=%+d n_faces=%d "
        "min_streamwise_velocity_ratio=%.3f mean_streamwise_velocity_ratio=%.3f "
        "max_streamwise_velocity_ratio=%.3f",
        "xyz"[axis],
        int(sign),
        int(mask.sum()),
        streamwise_velocity.min() / freestream_speed,
        streamwise_velocity.mean() / freestream_speed,
        streamwise_velocity.max() / freestream_speed,
    )


def boundary_flux_tolerance(particle_spacing: float, fvm_box: np.ndarray) -> float:
    """Second-order trace allowance used for the dimensionless flux residual."""
    bounds = np.asarray(fvm_box, dtype=np.float64).reshape(6)
    extent = bounds[1::2] - bounds[::2]
    length = float(np.min(extent))
    if length <= 0.0:
        raise ValueError("FVM box extents must be positive")
    second_order = (particle_spacing / length) ** 2
    return float(max(4096.0 * np.finfo(float).eps, min(second_order, 1.0e-3)))


def tangential_normal_velocity_gradient(
    target_velocity_gradient: np.ndarray, face_normal: np.ndarray
) -> np.ndarray:
    r"""Return the tangential component of ``du/dn`` on each boundary face.

    The VPM Jacobian uses ``J[i,j] = d(u_i)/d(x_j)``, hence ``du/dn = J n``.
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
    fvm_box: np.ndarray,
    particle_spacing: float,
    evaluated_velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate the VPM trace and correct only a discretization-scale flux residual.

    Face coordinates are in metres, velocities in m/s, normal are unit vectors,
    and areas are in m². The returned trace has shape ``(N, 3)``.
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
    tolerance = boundary_flux_tolerance(particle_spacing, fvm_box)
    total_area = float(np.sum(areas))
    if areas.size:
        raw_flux = float(np.dot(np.einsum("ij,ij->i", velocity, normal), areas))
        if total_area > 0.0:
            freestream_speed = float(np.linalg.norm(freestream_velocity))
            scale = (
                max(
                    freestream_speed,
                    float(np.sqrt(np.mean(np.einsum("ij,ij->i", velocity, velocity)))),
                    np.finfo(float).tiny,
                )
                * total_area
            )
            raw_relative = abs(raw_flux) / scale
            if raw_relative > tolerance:
                raise RuntimeError(
                    "VPM boundary trace has a physically significant net flux: "
                    f"|integral(velocity.n dA)|/(reference_velocity reference_area)={raw_relative:.3e}, "
                    f"acceptance limit={tolerance:.3e}. Refusing to hide the "
                    "upstream boundary-field error with a projection."
                )
            correction = raw_flux / total_area
            velocity = velocity - correction * normal
        corrected_flux = float(np.dot(np.einsum("ij,ij->i", velocity, normal), areas))
        logger.info(
            "[Coupler][BoundaryFlux] n_particles=%d integrated_flux_m3_s=%.3e "
            "integrated_flux_ratio=%.3e acceptance_limit_ratio=%.3e "
            "normal_velocity_correction_m_s=%.3e corrected_integrated_flux_m3_s=%.3e",
            int(vpm.particles.n_particles_total),
            raw_flux,
            raw_relative,
            tolerance,
            correction,
            corrected_flux,
        )

    _log_outflow_velocity(
        face_centre,
        velocity,
        freestream_velocity=np.asarray(freestream_velocity, dtype=np.float64),
        fvm_box=np.asarray(fvm_box, dtype=np.float64),
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
    """Construct the next VPM boundary-condition trace."""
    vpm_boundary_condition_velocity = None
    tangential_normal_gradient: np.ndarray | None = None
    if coupler._is_master:
        assert coupler.vpm_solver is not None
        if coupler.setup.boundary_condition_mode == "vorticity_mixed":
            vpm_boundary_condition_velocity, tangential_normal_gradient = (
                coupler.vpm_solver.compute_velocity_and_tangential_normal_gradient_at_points(
                    face_centre, face_normal, particle_spacing=coupler.setup.vpm_particle_spacing
                )
            )
            vpm_boundary_condition_velocity = np.asarray(
                vpm_boundary_condition_velocity, dtype=np.float64
            ).reshape(-1, 3)
        else:
            vpm_boundary_condition_velocity = np.asarray(
                coupler.vpm_solver.compute_velocity_at_points(
                    face_centre,
                    include_freestream=True,
                    zone_mask=None,
                    include_body=True,
                ),
                dtype=np.float64,
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

    boundary_condition_wall_time = time.perf_counter()
    if coupler._is_master:
        if coupler.setup.boundary_condition_mode == "pressure_gradient":
            assert coupler.vpm_solver is not None
            assert coupler.density is not None
            assert coupler.kinematic_viscosity is not None
            assert coupler.vpm_time_step_size is not None
            pressure_result, pressure_velocity = (
                coupler.vpm_solver.compute_pressure_gradient_at_points(
                    face_centre,
                    density=coupler.density,
                    kinematic_viscosity=coupler.kinematic_viscosity,
                    include_viscous=False,
                    include_temporal=coupler._pressure_velocity_snapshot is not None,
                    include_freestream=True,
                    include_body=True,
                    particle_spacing=coupler.setup.vpm_particle_spacing,
                    temporal_method="eulerian",
                    velocity_previous=coupler._pressure_velocity_snapshot,
                    time_step_size=coupler.vpm_time_step_size,
                    return_velocity=True,
                    treecode_theta=0.3,
                )
            )
            pressure_gradient = (
                np.asarray(pressure_result["pressure_gradient"], dtype=np.float64).reshape(-1, 3)
                / coupler.density
            )
            pressure_velocity = np.asarray(pressure_velocity, dtype=np.float64).reshape(-1, 3)
            if pressure_gradient.shape != face_centre.shape or not np.all(
                np.isfinite(pressure_gradient)
            ):
                raise RuntimeError(
                    "VPM pressure-gradient VPM boundary condition returned invalid data"
                )
            if pressure_velocity.shape != face_centre.shape or not np.all(
                np.isfinite(pressure_velocity)
            ):
                raise RuntimeError(
                    "VPM pressure-gradient VPM boundary condition returned invalid velocity data"
                )
            # Form velocity and pressure-gradient Cauchy data from one
            # body-complete velocity field.
            vpm_boundary_condition_velocity = pressure_velocity
            pressure_norm = np.linalg.norm(pressure_gradient, axis=1)
            logger.info(
                "[Coupler][BoundaryPressureGradient] rms_m_s2=%.3e max_m_s2=%.3e temporal_term=%s",
                float(np.sqrt(np.mean(pressure_norm**2))) if len(pressure_norm) else 0.0,
                float(np.max(pressure_norm)) if len(pressure_norm) else 0.0,
                coupler._pressure_velocity_snapshot is not None,
            )
            coupler._pressure_velocity_snapshot = pressure_velocity.copy()
            coupler._kinematic_pressure_gradient_boundary_condition = pressure_gradient
            if coupler._kinematic_pressure_gradient_boundary_condition_old is None:
                coupler._kinematic_pressure_gradient_boundary_condition_old = (
                    pressure_gradient.copy()
                )
        assert coupler.fvm_box is not None
        velocity_boundary_condition, coupler._last_vpm_boundary_condition_flux_diagnostics = (
            evaluate_vpm_velocity(
                coupler.vpm_solver,
                face_centre,
                face_normal,
                face_area,
                freestream_velocity=coupler.freestream_velocity,
                fvm_box=coupler.fvm_box,
                particle_spacing=coupler.setup.vpm_particle_spacing,
                evaluated_velocity=vpm_boundary_condition_velocity,
            )
        )
        if coupler._velocity_boundary_condition_old is None:
            coupler._velocity_boundary_condition_old = velocity_boundary_condition.copy()
        if coupler.setup.boundary_condition_mode == "vorticity_mixed":
            assert tangential_normal_gradient is not None
            normal_velocity_boundary_condition = np.einsum(
                "ij,ij->i", velocity_boundary_condition, face_normal
            )
            tangential_gradient_boundary_condition = tangential_normal_gradient
            coupler._normal_velocity_boundary_condition = normal_velocity_boundary_condition
            coupler._tangential_gradient_boundary_condition = tangential_gradient_boundary_condition
            if coupler._normal_velocity_boundary_condition_old is None:
                coupler._normal_velocity_boundary_condition_old = (
                    normal_velocity_boundary_condition.copy()
                )
            if coupler._tangential_gradient_boundary_condition_old is None:
                coupler._tangential_gradient_boundary_condition_old = (
                    tangential_gradient_boundary_condition.copy()
                )
    else:
        velocity_boundary_condition = np.zeros_like(face_centre)
        if coupler._velocity_boundary_condition_old is None:
            coupler._velocity_boundary_condition_old = np.zeros_like(face_centre)
        if coupler.setup.boundary_condition_mode == "pressure_gradient":
            coupler._kinematic_pressure_gradient_boundary_condition = np.zeros_like(face_centre)
            if coupler._kinematic_pressure_gradient_boundary_condition_old is None:
                coupler._kinematic_pressure_gradient_boundary_condition_old = np.zeros_like(
                    face_centre
                )
        elif coupler.setup.boundary_condition_mode == "vorticity_mixed":
            coupler._normal_velocity_boundary_condition = np.zeros(
                len(face_centre), dtype=np.float64
            )
            coupler._tangential_gradient_boundary_condition = np.zeros_like(face_centre)
            if coupler._normal_velocity_boundary_condition_old is None:
                coupler._normal_velocity_boundary_condition_old = (
                    coupler._normal_velocity_boundary_condition.copy()
                )
            if coupler._tangential_gradient_boundary_condition_old is None:
                coupler._tangential_gradient_boundary_condition_old = (
                    coupler._tangential_gradient_boundary_condition.copy()
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
    """Evaluate the physical ``t_n`` trace before the first VPM advance."""
    if coupler._velocity_boundary_condition_old is not None:
        return
    evaluate_vpm_boundary(coupler, face_centre, face_normal, face_area)
    logger.info("[Coupler][BoundaryHistory] time_level=initial")


def advance_fvm(
    coupler,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    face_area: np.ndarray,
    velocity_boundary_condition_old: np.ndarray,
    velocity_boundary_condition: np.ndarray,
) -> float:
    """Run FVM sub-cycles and refresh its velocity snapshot."""
    fvm_wall_time_start = time.perf_counter()
    advance_fvm_substeps(
        coupler,
        coupler.setup.coupling_patch,
        face_centre,
        face_normal,
        face_area,
        velocity_boundary_condition_old,
        velocity_boundary_condition,
        coupler._kinematic_pressure_gradient_boundary_condition_old,
        coupler._kinematic_pressure_gradient_boundary_condition,
        coupler._normal_velocity_boundary_condition_old,
        coupler._normal_velocity_boundary_condition,
        coupler._tangential_gradient_boundary_condition_old,
        coupler._tangential_gradient_boundary_condition,
    )
    if coupler._is_master:
        coupler._velocity_boundary_condition_old = velocity_boundary_condition
        if coupler._kinematic_pressure_gradient_boundary_condition is not None:
            coupler._kinematic_pressure_gradient_boundary_condition_old = (
                coupler._kinematic_pressure_gradient_boundary_condition
            )
        if coupler._normal_velocity_boundary_condition is not None:
            coupler._normal_velocity_boundary_condition_old = (
                coupler._normal_velocity_boundary_condition
            )
        if coupler._tangential_gradient_boundary_condition is not None:
            coupler._tangential_gradient_boundary_condition_old = (
                coupler._tangential_gradient_boundary_condition
            )
    return time.perf_counter() - fvm_wall_time_start


def resynchronize_vpm_boundary(
    coupler,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    face_area: np.ndarray,
) -> None:
    """Re-evaluate the VPM boundary-condition trace from the corrected particle field.

    Otherwise each interval starts from a stale prediction. Not a Picard
    sweep: the FVM is not re-solved.
    """
    if not coupler.setup.is_boundary_condition_resynchronized_after_transfer:
        return
    if not coupler._is_master:
        return

    assert coupler.vpm_solver is not None
    tangential_normal_gradient: np.ndarray | None = None
    if coupler.setup.boundary_condition_mode == "vorticity_mixed":
        corrected_boundary, tangential_normal_gradient = (
            coupler.vpm_solver.compute_velocity_and_tangential_normal_gradient_at_points(
                face_centre, face_normal, particle_spacing=coupler.setup.vpm_particle_spacing
            )
        )
        corrected_boundary = np.asarray(corrected_boundary, dtype=np.float64).reshape(-1, 3)
    else:
        corrected_boundary = np.asarray(
            coupler.vpm_solver.compute_velocity_at_points(
                face_centre, include_freestream=True, zone_mask=None, include_body=True
            ),
            dtype=np.float64,
        ).reshape(-1, 3)
    if corrected_boundary.shape != face_centre.shape or not np.all(np.isfinite(corrected_boundary)):
        raise RuntimeError("VPM boundary-condition resynchronisation returned invalid velocities")

    assert coupler.fvm_box is not None
    corrected_boundary, coupler._last_vpm_boundary_condition_flux_diagnostics = (
        evaluate_vpm_velocity(
            coupler.vpm_solver,
            face_centre,
            face_normal,
            face_area,
            freestream_velocity=coupler.freestream_velocity,
            fvm_box=coupler.fvm_box,
            particle_spacing=coupler.setup.vpm_particle_spacing,
            evaluated_velocity=corrected_boundary,
        )
    )
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
    if coupler.setup.boundary_condition_mode == "pressure_gradient":
        # The FVM-to-VPM transfer replaces the particle representation at
        # fixed physical time. Refresh the Eulerian pressure history so
        # that the next backward difference does not interpret that
        # representation jump as a physical temporal acceleration.
        coupler._pressure_velocity_snapshot = corrected_boundary.copy()
    elif coupler.setup.boundary_condition_mode == "vorticity_mixed":
        assert tangential_normal_gradient is not None
        coupler._normal_velocity_boundary_condition_old = np.einsum(
            "ij,ij->i", corrected_boundary, face_normal
        )
        coupler._tangential_gradient_boundary_condition_old = tangential_normal_gradient
    logger.info(
        "[Coupler][BoundaryUpdate] post_transfer_velocity_difference_magnitude_max_over_freestream_speed=%.3e",
        drift,
    )


def apply_fvm_boundary(
    coupler,
    patch: str,
    prescribed_velocity: np.ndarray,
    pressure_gradient: np.ndarray | None = None,
    normal_velocity: np.ndarray | None = None,
    tangential_gradient: np.ndarray | None = None,
) -> None:
    """Apply the configured VPM boundary condition trace and advance one FVM step."""
    assert coupler.fvm_solver is not None
    freestream_velocity = np.asarray(coupler.setup.freestream_velocity, dtype=np.float64)
    freestream_speed = float(np.linalg.norm(freestream_velocity)) + 1e-30
    boundary_mode = coupler.setup.boundary_condition_mode
    prescribed_velocity = np.ascontiguousarray(prescribed_velocity, dtype=np.float64)
    if boundary_mode == "vorticity_mixed":
        if normal_velocity is None or tangential_gradient is None:
            raise RuntimeError(
                "vorticity_mixed VPM boundary-condition mode requires normal velocity and "
                "tangential-gradient data"
            )
        coupler.fvm_solver.set_normal_velocity_tangential_gradient_boundary_condition(
            np.ascontiguousarray(normal_velocity, dtype=np.float64),
            np.ascontiguousarray(tangential_gradient, dtype=np.float64),
            patch,
        )
        coupler.fvm_solver.set_flux_consistent_pressure_boundary_condition(patch)
    elif boundary_mode == "characteristic":
        coupler.fvm_solver.set_freestream_velocity_boundary_condition_vec(
            prescribed_velocity, patch
        )
        coupler.fvm_solver.set_freestream_pressure_boundary_condition(patch, value=0.0)
    elif boundary_mode == "directional_outflow":
        coupler.fvm_solver.set_directional_freestream_velocity_boundary_condition_vec(
            prescribed_velocity, patch, coupler.setup.freestream_velocity
        )
        coupler.fvm_solver.set_directional_freestream_pressure_boundary_condition(patch, value=0.0)
    elif boundary_mode == "pressure_gradient":
        if pressure_gradient is None:
            raise RuntimeError(
                "pressure_gradient VPM boundary-condition mode requires pressure-gradient data"
            )
        coupler.fvm_solver.set_dirichlet_velocity_boundary_condition_vec(prescribed_velocity, patch)
        coupler.fvm_solver.set_neumann_pressure_boundary_condition(pressure_gradient, patch)
    else:
        coupler.fvm_solver.set_dirichlet_velocity_boundary_condition_vec(prescribed_velocity, patch)

    step_time_step_size = coupler.fvm_solver.time_step_size
    step_wall_time_start = time.perf_counter()
    coupler.fvm_solver.logger.step_begin(
        coupler.fvm_solver.step + 1,
        coupler.fvm_solver.time + step_time_step_size,
        step_time_step_size,
    )

    coupler.fvm_solver.solve_pimple()

    time_config = coupler.fvm_solver.setup.time
    coupler.fvm_solver.logger.courant_info(
        coupler.fvm_solver.max_courant_number,
        time_config.max_courant_number if time_config.adjust_time_step else None,
    )

    coupler.fvm_solver.advance_time()
    coupler.fvm_solver.logger.step_end(time.perf_counter() - step_wall_time_start)

    if prescribed_velocity.shape[0] > 0:
        streamwise = prescribed_velocity @ (freestream_velocity / freestream_speed)
        logger.info(
            "[Coupler][FVMSubstep] fvm_step=%d min_streamwise_velocity_ratio=%.3f "
            "mean_streamwise_velocity_ratio=%.3f max_streamwise_velocity_ratio=%.3f",
            int(coupler.fvm_solver.step),
            streamwise.min() / freestream_speed,
            streamwise.mean() / freestream_speed,
            streamwise.max() / freestream_speed,
        )
    y_plus = coupler.fvm_solver.last_y_plus
    if y_plus:
        for name, stats in y_plus.items():
            coupler.fvm_solver.logger.message(
                f"[FVM][Wall] step={coupler.fvm_solver.step} patch={name} "
                f"min_y_plus={stats['min']:.3f} mean_y_plus={stats['avg']:.3f} "
                f"max_y_plus={stats['max']:.3f}"
            )


def advance_fvm_substeps(
    coupler,
    patch: str,
    face_centre: np.ndarray,
    face_normal: np.ndarray,
    face_area: np.ndarray,
    previous_velocity: np.ndarray,
    next_velocity: np.ndarray,
    previous_kinematic_pressure_gradient: np.ndarray | None = None,
    next_kinematic_pressure_gradient: np.ndarray | None = None,
    previous_normal_velocity: np.ndarray | None = None,
    next_normal_velocity: np.ndarray | None = None,
    previous_tangential_gradient: np.ndarray | None = None,
    next_tangential_gradient: np.ndarray | None = None,
) -> None:
    """Advance FVM substeps with interpolated VPM boundary condition data."""
    n_substeps = coupler.n_fvm_substeps
    freestream_speed = float(np.linalg.norm(coupler.freestream_velocity)) + 1e-30
    if n_substeps > 1 and next_velocity.shape[0] > 0:
        boundary_velocity_difference_ratio = (
            float(np.max(np.linalg.norm(next_velocity - previous_velocity, axis=1)))
            / freestream_speed
        )
        is_large_boundary_velocity_difference = boundary_velocity_difference_ratio > 0.5
        logger.log(
            logging.WARNING if is_large_boundary_velocity_difference else logging.INFO,
            "[Coupler][TimeInterpolation] severity=%s n_fvm_substeps=%d "
            "fvm_time_step_size_s=%.3e "
            "boundary_velocity_difference_magnitude_max_over_freestream_speed_ratio=%.3f "
            "warning_limit_ratio=%.3f",
            "warning" if is_large_boundary_velocity_difference else "info",
            n_substeps,
            coupler.fvm_time_step_size,
            boundary_velocity_difference_ratio,
            0.5,
        )

    for substep in range(n_substeps):
        alpha = (substep + 1) / n_substeps
        interpolated_velocity = (1.0 - alpha) * previous_velocity + alpha * next_velocity
        pressure_gradient = None
        if (
            previous_kinematic_pressure_gradient is not None
            and next_kinematic_pressure_gradient is not None
        ):
            pressure_gradient = (
                1.0 - alpha
            ) * previous_kinematic_pressure_gradient + alpha * next_kinematic_pressure_gradient
        normal_velocity = None
        interpolated_tangential_gradient = None
        if coupler.setup.boundary_condition_mode == "vorticity_mixed":
            if (
                previous_normal_velocity is None
                or next_normal_velocity is None
                or previous_tangential_gradient is None
                or next_tangential_gradient is None
            ):
                raise RuntimeError("vorticity_mixed subcycling received an incomplete trace")
            normal_velocity = (
                1.0 - alpha
            ) * previous_normal_velocity + alpha * next_normal_velocity
            interpolated_tangential_gradient = (
                1.0 - alpha
            ) * previous_tangential_gradient + alpha * next_tangential_gradient
        apply_fvm_boundary(
            coupler,
            patch,
            interpolated_velocity,
            pressure_gradient,
            normal_velocity=normal_velocity,
            tangential_gradient=interpolated_tangential_gradient,
        )
