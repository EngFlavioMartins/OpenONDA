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


def _log_outflow_deficit(
    face_centers: np.ndarray,
    velocity: np.ndarray,
    *,
    freestream_velocity: np.ndarray,
    fvm_box: np.ndarray,
) -> None:
    """Log the streamwise velocity on the downstream face of the FVM box."""
    axis, sign = outflow_axis_sign(freestream_velocity)
    u_mag = float(np.linalg.norm(freestream_velocity)) + 1.0e-30
    face_lo, face_hi = fvm_box[2 * axis], fvm_box[2 * axis + 1]
    if sign >= 0.0:
        mask = face_centers[:, axis] >= face_hi - 1.0e-6
    else:
        mask = face_centers[:, axis] <= face_lo + 1.0e-6
    if not mask.any():
        return
    u_stream = velocity[mask] @ (np.asarray(freestream_velocity) / u_mag)
    logger.info(
        "     [VPM-BC deficit outflow axis=%d sign=%+d] u_s/U∞ min=%.3f "
        "mean=%.3f max=%.3f  n_face=%d",
        axis,
        int(sign),
        u_stream.min() / u_mag,
        u_stream.mean() / u_mag,
        u_stream.max() / u_mag,
        int(mask.sum()),
    )


def project_solenoidal_velocity(
    velocity: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
) -> np.ndarray:
    """Apply the minimum-L2 normal correction required for zero net flux."""
    normals = np.asarray(face_normals, dtype=np.float64).reshape(-1, 3)
    areas = np.asarray(face_areas, dtype=np.float64).reshape(-1)
    if areas.size == 0:
        return velocity
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        return velocity
    flux = float(np.dot(np.einsum("ij,ij->i", velocity, normals), areas))
    return velocity - (flux / total_area) * normals


def project_normal_velocity(normal_velocity: np.ndarray, face_areas: np.ndarray) -> np.ndarray:
    """Remove the area-weighted flux residual from a scalar normal trace."""
    values = np.asarray(normal_velocity, dtype=np.float64).reshape(-1)
    areas = np.asarray(face_areas, dtype=np.float64).reshape(-1)
    if values.shape != areas.shape:
        raise ValueError("normal velocity and face area counts differ")
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        return values
    return values - float(np.dot(values, areas)) / total_area


def tangential_normal_velocity_gradient(
    target_velocity_gradient: np.ndarray, face_normals: np.ndarray
) -> np.ndarray:
    r"""Return the tangential component of ``du/dn`` on each boundary face.

    The VPM Jacobian uses ``J[i,j] = d(u_i)/d(x_j)``, hence ``du/dn = J n``.
    """
    normals = np.asarray(face_normals, dtype=np.float64).reshape(-1, 3)
    jacobian = np.asarray(target_velocity_gradient, dtype=np.float64)
    if jacobian.size != 9 * len(normals):
        raise ValueError("VPM velocity-gradient count does not match boundary faces")
    jacobian = jacobian.reshape(-1, 3, 3)
    if not np.all(np.isfinite(jacobian)):
        raise RuntimeError("VPM target-gradient evaluation returned non-finite data")
    d_u_dn = np.einsum("fij,fj->fi", jacobian, normals)
    return d_u_dn - np.einsum("fi,fi->f", d_u_dn, normals)[:, np.newaxis] * normals


def evaluate_vpm_velocity(
    vpm,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    *,
    freestream_velocity: np.ndarray,
    fvm_box: np.ndarray,
    evaluated_velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate the body-complete VPM velocity and enforce zero boundary flux.

    Face coordinates are in metres, velocities in m/s, normals are unit vectors,
    and areas are in m². The returned trace has shape ``(N, 3)``.
    """
    normals = np.asarray(face_normals, dtype=np.float64).reshape(-1, 3)
    areas = np.asarray(face_areas, dtype=np.float64).reshape(-1)
    logger.info("     [VPM-BC] particles=%d", vpm.particles.n_particles)
    if evaluated_velocity is None:
        evaluated_velocity = vpm.compute_target_velocities(
            face_centers,
            include_freestream=True,
            zone_mask=None,
            include_body=True,
        )
    velocity = np.asarray(evaluated_velocity, dtype=np.float64).reshape(-1, 3)
    if len(velocity) != len(face_centers):
        raise ValueError("evaluated VPM-BC velocity count does not match boundary faces")

    raw_flux = 0.0
    correction = 0.0
    corrected_flux = 0.0
    total_area = float(np.sum(areas))
    if areas.size:
        raw_flux = float(np.dot(np.einsum("ij,ij->i", velocity, normals), areas))
        if total_area > 0.0:
            correction = raw_flux / total_area
            velocity = velocity - correction * normals
        corrected_flux = float(np.dot(np.einsum("ij,ij->i", velocity, normals), areas))
        freestream_velocity_mag = float(np.linalg.norm(freestream_velocity)) + 1.0e-30
        logger.info(
            "     [VPM-BC] Flux residual: raw=%.3e m³/s (%.2e×U∞A)  "
            "post-projection=%.3e m³/s (%.2e×U∞A)  δu_n=%.3e m/s",
            raw_flux,
            abs(raw_flux) / (freestream_velocity_mag * total_area + 1.0e-30),
            corrected_flux,
            abs(corrected_flux) / (freestream_velocity_mag * total_area + 1.0e-30),
            correction,
        )

    _log_outflow_deficit(
        face_centers,
        velocity,
        freestream_velocity=np.asarray(freestream_velocity, dtype=np.float64),
        fvm_box=np.asarray(fvm_box, dtype=np.float64),
    )
    diagnostics = {
        "raw_mismatch": abs(raw_flux),
        "applied_correction": abs(correction),
        "corrected_mismatch": abs(corrected_flux),
    }
    return velocity, diagnostics


def evaluate_vpm_boundary(
    coupler,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
):
    """Update blending-zone data and construct the next VPM boundary condition trace."""
    assert coupler.blending_zone is not None
    t_blending = time.perf_counter()
    vpm_bc_velocity = None
    tangential_normal_gradient: np.ndarray | None = None
    if coupler._is_master:
        assert coupler.vpm_solver is not None
        blend_cell_centres = coupler.blending_zone.active_cell_centres
        n_blend_cells = len(blend_cell_centres)
        if coupler.setup.vpm_bc_mode == "vorticity_mixed":
            # The blending field needs velocity only, whereas the boundary
            # faces need both velocity and du/dn.  Keeping these target sets
            # separate avoids gradients over every blending cell while the
            # face query shares one VPM tree traversal.
            blend_velocity = np.asarray(
                coupler.vpm_solver.compute_target_velocities(
                    blend_cell_centres,
                    include_freestream=True,
                    zone_mask=None,
                    include_body=True,
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            vpm_bc_velocity, tangential_normal_gradient = (
                coupler.vpm_solver.compute_complete_target_velocity_and_tangential_normal_gradient(
                    face_centers, face_normals, particle_spacing=coupler.setup.vpm_particle_spacing
                )
            )
            vpm_bc_velocity = np.asarray(vpm_bc_velocity, dtype=np.float64).reshape(-1, 3)
            target_velocity = np.concatenate((blend_velocity, vpm_bc_velocity), axis=0)
        else:
            target_points = np.concatenate((blend_cell_centres, face_centers), axis=0)
            target_velocity = np.asarray(
                coupler.vpm_solver.compute_target_velocities(
                    target_points,
                    include_freestream=True,
                    zone_mask=None,
                    include_body=True,
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
        expected_targets = n_blend_cells + len(face_centers)
        if target_velocity.shape != (expected_targets, 3):
            raise RuntimeError(
                "VPM target evaluation returned an invalid shape: "
                f"expected {(expected_targets, 3)}, got {target_velocity.shape}"
            )
        if not np.all(np.isfinite(target_velocity)):
            raise RuntimeError("VPM target evaluation returned non-finite velocities")
        freestream_speed = float(np.linalg.norm(coupler.freestream_velocity))
        if (
            expected_targets > 0
            and freestream_speed > 0.0
            and float(np.max(np.linalg.norm(target_velocity, axis=1))) <= 1.0e-6 * freestream_speed
        ):
            raise RuntimeError(
                "VPM target evaluation returned an identically zero field despite a "
                "nonzero freestream; aborting before the corrupted VPM-BC data reaches the FVM"
            )
        coupler.blending_zone.update_target(target_velocity[:n_blend_cells])
        vpm_bc_velocity = target_velocity[n_blend_cells:]
    else:
        coupler.blending_zone.update_target()
    t_blending = time.perf_counter() - t_blending

    t_vpm_bc = time.perf_counter()
    if coupler._is_master:
        if coupler.setup.vpm_bc_mode == "pressure_gradient":
            assert coupler.vpm_solver is not None
            assert coupler.density is not None
            assert coupler.kinematic_viscosity is not None
            assert coupler.vpm_time_step_size is not None
            pressure_result, pressure_velocity = (
                coupler.vpm_solver.compute_target_pressure_gradients(
                    face_centers,
                    density=coupler.density,
                    nu=coupler.kinematic_viscosity,
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
                np.asarray(pressure_result["grad_p"], dtype=np.float64).reshape(-1, 3)
                / coupler.density
            )
            pressure_velocity = np.asarray(pressure_velocity, dtype=np.float64).reshape(-1, 3)
            if pressure_gradient.shape != face_centers.shape or not np.all(
                np.isfinite(pressure_gradient)
            ):
                raise RuntimeError("VPM pressure-gradient VPM BC returned invalid data")
            if pressure_velocity.shape != face_centers.shape or not np.all(
                np.isfinite(pressure_velocity)
            ):
                raise RuntimeError("VPM pressure-gradient VPM BC returned invalid velocity data")
            # Form velocity and pressure-gradient Cauchy data from one
            # body-complete velocity field.
            vpm_bc_velocity = pressure_velocity
            pressure_norm = np.linalg.norm(pressure_gradient, axis=1)
            logger.info(
                "     [VPM-BC pressure] |∇(p/ρ)| rms=%.3e max=%.3e m/s²  temporal=%s",
                float(np.sqrt(np.mean(pressure_norm**2))) if len(pressure_norm) else 0.0,
                float(np.max(pressure_norm)) if len(pressure_norm) else 0.0,
                coupler._pressure_velocity_snapshot is not None,
            )
            coupler._pressure_velocity_snapshot = pressure_velocity.copy()
            coupler._pressure_gradient_bc_next = pressure_gradient
            if coupler._pressure_gradient_bc_prev is None:
                coupler._pressure_gradient_bc_prev = pressure_gradient.copy()
        assert coupler.fvm_box is not None
        u_bc_next, coupler._last_vpm_bc_flux_diagnostics = evaluate_vpm_velocity(
            coupler.vpm_solver,
            face_centers,
            face_normals,
            face_areas,
            freestream_velocity=coupler.freestream_velocity,
            fvm_box=coupler.fvm_box,
            evaluated_velocity=vpm_bc_velocity,
        )
        if coupler._velocity_bc_prev is None:
            coupler._velocity_bc_prev = u_bc_next.copy()
        if coupler.setup.vpm_bc_mode == "vorticity_mixed":
            assert tangential_normal_gradient is not None
            normal_next = project_normal_velocity(
                np.einsum("ij,ij->i", u_bc_next, face_normals), face_areas
            )
            tangential_next = tangential_normal_gradient
            coupler._normal_velocity_bc_next = normal_next
            coupler._tangential_gradient_bc_next = tangential_next
            if coupler._normal_velocity_bc_prev is None:
                coupler._normal_velocity_bc_prev = normal_next.copy()
            if coupler._tangential_gradient_bc_prev is None:
                coupler._tangential_gradient_bc_prev = tangential_next.copy()
    else:
        u_bc_next = np.zeros_like(face_centers)
        if coupler._velocity_bc_prev is None:
            coupler._velocity_bc_prev = np.zeros_like(face_centers)
        if coupler.setup.vpm_bc_mode == "pressure_gradient":
            coupler._pressure_gradient_bc_next = np.zeros_like(face_centers)
            if coupler._pressure_gradient_bc_prev is None:
                coupler._pressure_gradient_bc_prev = np.zeros_like(face_centers)
        elif coupler.setup.vpm_bc_mode == "vorticity_mixed":
            coupler._normal_velocity_bc_next = np.zeros(len(face_centers), dtype=np.float64)
            coupler._tangential_gradient_bc_next = np.zeros_like(face_centers)
            if coupler._normal_velocity_bc_prev is None:
                coupler._normal_velocity_bc_prev = coupler._normal_velocity_bc_next.copy()
            if coupler._tangential_gradient_bc_prev is None:
                coupler._tangential_gradient_bc_prev = coupler._tangential_gradient_bc_next.copy()
    t_vpm_bc = time.perf_counter() - t_vpm_bc
    return (
        coupler._velocity_bc_prev,
        u_bc_next,
        t_blending,
        t_vpm_bc,
    )


def advance_fvm(
    coupler,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    u_bc_prev: np.ndarray,
    u_bc_next: np.ndarray,
) -> float:
    """Run FVM sub-cycles and refresh its velocity snapshot."""
    t_fvm = time.perf_counter()
    advance_fvm_substeps(
        coupler,
        coupler.setup.bc_patch_name,
        face_centers,
        face_normals,
        face_areas,
        u_bc_prev,
        u_bc_next,
        coupler._pressure_gradient_bc_prev,
        coupler._pressure_gradient_bc_next,
        coupler._normal_velocity_bc_prev,
        coupler._normal_velocity_bc_next,
        coupler._tangential_gradient_bc_prev,
        coupler._tangential_gradient_bc_next,
    )
    if coupler._is_master:
        coupler._velocity_bc_prev = u_bc_next
        if coupler._pressure_gradient_bc_next is not None:
            coupler._pressure_gradient_bc_prev = coupler._pressure_gradient_bc_next
        if coupler._normal_velocity_bc_next is not None:
            coupler._normal_velocity_bc_prev = coupler._normal_velocity_bc_next
        if coupler._tangential_gradient_bc_next is not None:
            coupler._tangential_gradient_bc_prev = coupler._tangential_gradient_bc_next
    return time.perf_counter() - t_fvm


def resynchronize_vpm_boundary(
    coupler,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
) -> None:
    """Re-evaluate the VPM-BC trace from the corrected particle field.

    Otherwise each interval starts from a stale prediction. Not a Picard
    sweep: the FVM is not re-solved.
    """
    if not coupler.setup.bc_resync_after_transfer:
        return
    assert coupler.blending_zone is not None
    if not coupler._is_master:
        coupler.blending_zone.update_endpoint()
        return

    assert coupler.vpm_solver is not None
    blend_cell_centres = coupler.blending_zone.active_cell_centres
    n_blend_cells = len(blend_cell_centres)
    tangential_normal_gradient: np.ndarray | None = None
    if coupler.setup.vpm_bc_mode == "vorticity_mixed":
        blend_velocity = np.asarray(
            coupler.vpm_solver.compute_target_velocities(
                blend_cell_centres,
                include_freestream=True,
                zone_mask=None,
                include_body=True,
            ),
            dtype=np.float64,
        ).reshape(-1, 3)
        corrected_boundary, tangential_normal_gradient = (
            coupler.vpm_solver.compute_complete_target_velocity_and_tangential_normal_gradient(
                face_centers, face_normals, particle_spacing=coupler.setup.vpm_particle_spacing
            )
        )
        corrected = np.concatenate(
            (blend_velocity, np.asarray(corrected_boundary, dtype=np.float64).reshape(-1, 3)),
            axis=0,
        )
    else:
        targets = np.concatenate((blend_cell_centres, face_centers), axis=0)
        corrected = np.asarray(
            coupler.vpm_solver.compute_target_velocities(
                targets, include_freestream=True, zone_mask=None, include_body=True
            ),
            dtype=np.float64,
        ).reshape(-1, 3)
    if corrected.shape != (n_blend_cells + len(face_centers), 3) or not np.all(
        np.isfinite(corrected)
    ):
        raise RuntimeError("VPM-BC resynchronisation returned invalid velocities")

    coupler.blending_zone.update_endpoint(corrected[:n_blend_cells])
    corrected_boundary = corrected[n_blend_cells:]
    if coupler.setup.vpm_bc_mode == "vorticity_mixed":
        corrected_boundary = project_solenoidal_velocity(
            corrected_boundary, face_normals, face_areas
        )
    freestream_velocity_mag = float(np.linalg.norm(coupler.freestream_velocity)) + 1e-30
    drift = (
        float(np.max(np.linalg.norm(corrected_boundary - coupler._velocity_bc_prev, axis=1)))
        / freestream_velocity_mag
        if coupler._velocity_bc_prev is not None and len(face_centers)
        else 0.0
    )
    coupler._velocity_bc_prev = corrected_boundary
    if coupler.setup.vpm_bc_mode == "pressure_gradient":
        # The FVM-to-VPM transfer replaces the particle representation at
        # fixed physical time. Refresh the Eulerian pressure history so
        # that the next backward difference does not interpret that
        # representation jump as a physical temporal acceleration.
        coupler._pressure_velocity_snapshot = corrected_boundary.copy()
    elif coupler.setup.vpm_bc_mode == "vorticity_mixed":
        assert tangential_normal_gradient is not None
        coupler._normal_velocity_bc_prev = project_normal_velocity(
            np.einsum("ij,ij->i", corrected_boundary, face_normals), face_areas
        )
        coupler._tangential_gradient_bc_prev = tangential_normal_gradient
    logger.info("     [Resync] VPM-BC endpoint moved max|du|/Uinf=%.3e", drift)


def apply_fvm_boundary(
    coupler,
    patch: str,
    u_target: np.ndarray,
    pressure_gradient: np.ndarray | None = None,
    normal_velocity: np.ndarray | None = None,
    tangential_gradient: np.ndarray | None = None,
) -> None:
    """Apply the configured VPM boundary condition trace and advance one FVM step."""
    assert coupler.fvm_solver is not None
    freestream_velocity_mag = float(np.linalg.norm(coupler.setup.freestream_velocity)) + 1e-30
    boundary_mode = coupler.setup.vpm_bc_mode
    u_target = np.ascontiguousarray(u_target, dtype=np.float64)
    if boundary_mode == "vorticity_mixed":
        if normal_velocity is None or tangential_gradient is None:
            raise RuntimeError(
                "vorticity_mixed VPM-BC mode requires normal velocity and tangential-gradient data"
            )
        coupler.fvm_solver.set_normal_velocity_tangential_gradient_boundary_condition(
            np.ascontiguousarray(normal_velocity, dtype=np.float64),
            np.ascontiguousarray(tangential_gradient, dtype=np.float64),
            patch,
        )
        coupler.fvm_solver.set_flux_consistent_pressure_boundary_condition(patch)
        boundary_description = "normal VPM velocity / tangential VPM gradient / fixedFluxPressure"
    elif boundary_mode == "characteristic":
        coupler.fvm_solver.set_freestream_velocity_boundary_condition_vec(u_target, patch)
        coupler.fvm_solver.set_freestream_pressure_boundary_condition(patch, value=0.0)
        boundary_description = "characteristic U/p"
    elif boundary_mode == "directional_outflow":
        coupler.fvm_solver.set_directional_freestream_velocity_boundary_condition_vec(
            u_target, patch, coupler.setup.freestream_velocity
        )
        coupler.fvm_solver.set_directional_freestream_pressure_boundary_condition(patch, value=0.0)
        boundary_description = "directional-outflow mixed U/p"
    elif boundary_mode == "pressure_gradient":
        if pressure_gradient is None:
            raise RuntimeError("pressure_gradient VPM-BC mode requires pressure-gradient data")
        coupler.fvm_solver.set_dirichlet_velocity_boundary_condition_vec(u_target, patch)
        coupler.fvm_solver.set_neumann_pressure_boundary_condition(pressure_gradient, patch)
        boundary_description = "Dirichlet U / VPM pressure gradient"
    else:
        coupler.fvm_solver.set_dirichlet_velocity_boundary_condition_vec(u_target, patch)
        boundary_description = "Dirichlet U / fixedFluxPressure"

    step_time_step_size = coupler.fvm_solver.time_step_size
    step_t0 = time.perf_counter()
    coupler.fvm_solver.logger.step_begin(
        coupler.fvm_solver.step + 1,
        coupler.fvm_solver.time + step_time_step_size,
        step_time_step_size,
    )

    coupler.fvm_solver.solve_pimple()

    cfg_time = coupler.fvm_solver.setup.time
    coupler.fvm_solver.logger.courant_info(
        coupler.fvm_solver.cfl_max, cfg_time.max_cfl if cfg_time.adjust_time_step else None
    )

    coupler.fvm_solver.advance_time()
    coupler.fvm_solver.logger.step_end(time.perf_counter() - step_t0)

    if u_target.shape[0] > 0:
        logger.info(
            "     [FVM] solved with %s  u_x/U∞ face[min=%.2f max=%.2f]",
            boundary_description,
            u_target[:, 0].min() / freestream_velocity_mag,
            u_target[:, 0].max() / freestream_velocity_mag,
        )
    yplus = coupler.fvm_solver.last_yplus
    if yplus:
        parts = [
            f"{name}: y+ min={s['min']:.2f} max={s['max']:.2f} avg={s['avg']:.2f}"
            for name, s in yplus.items()
        ]
        logger.info("     [FVM wall] %s", " | ".join(parts))


def advance_fvm_substeps(
    coupler,
    patch: str,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    u_prev: np.ndarray,
    u_next: np.ndarray,
    pressure_gradient_prev: np.ndarray | None = None,
    pressure_gradient_next: np.ndarray | None = None,
    normal_velocity_prev: np.ndarray | None = None,
    normal_velocity_next: np.ndarray | None = None,
    tangential_gradient_prev: np.ndarray | None = None,
    tangential_gradient_next: np.ndarray | None = None,
) -> None:
    """Advance FVM substeps with interpolated VPM boundary condition data."""
    n_substeps = coupler.fvm_substeps
    freestream_velocity_mag = float(np.linalg.norm(coupler.freestream_velocity)) + 1e-30
    if n_substeps > 1 and u_next.shape[0] > 0:
        dU = float(np.max(np.linalg.norm(u_next - u_prev, axis=1))) / freestream_velocity_mag
        big = dU > 0.5
        logger.log(
            logging.WARNING if big else logging.INFO,
            "     [Sub-cycle] %d×fvm_dt=%.3e s  VPM-BC Δ max|Δu|/U∞=%.3f%s",
            n_substeps,
            coupler.fvm_time_step_size,
            dU,
            "  (large — lower the time step)" if big else "",
        )

    assert coupler.blending_zone is not None
    for substep in range(n_substeps):
        alpha = (substep + 1) / n_substeps
        coupler.blending_zone.push_target(alpha)
        u_bc = (1.0 - alpha) * u_prev + alpha * u_next
        u_bc = project_solenoidal_velocity(u_bc, face_normals, face_areas)
        pressure_gradient = None
        if pressure_gradient_prev is not None and pressure_gradient_next is not None:
            pressure_gradient = (
                1.0 - alpha
            ) * pressure_gradient_prev + alpha * pressure_gradient_next
        normal_velocity = None
        tangential_gradient = None
        if coupler.setup.vpm_bc_mode == "vorticity_mixed":
            if (
                normal_velocity_prev is None
                or normal_velocity_next is None
                or tangential_gradient_prev is None
                or tangential_gradient_next is None
            ):
                raise RuntimeError("vorticity_mixed subcycling received an incomplete trace")
            normal_velocity = project_normal_velocity(
                (1.0 - alpha) * normal_velocity_prev + alpha * normal_velocity_next,
                face_areas,
            )
            tangential_gradient = (
                1.0 - alpha
            ) * tangential_gradient_prev + alpha * tangential_gradient_next
        apply_fvm_boundary(
            coupler,
            patch,
            u_bc,
            pressure_gradient,
            normal_velocity=normal_velocity,
            tangential_gradient=tangential_gradient,
        )
