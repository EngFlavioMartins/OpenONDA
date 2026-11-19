"""Fixed-point (Picard) iterations of boundary data within one VPM interval."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import TypeAlias, TypedDict

import numpy as np

from openonda.core.parallel import collective_phase
from openonda.fvm.io.backup import capture_restart_state, restore_restart_state

from .boundary import reference_speed
from .reporting import domain_records
from .vorticity_transfer import _particle_state_snapshot, _restore_particle_state

logger = logging.getLogger("openonda.coupler")
_BOUNDARY_DATA_FIELDS = (
    "velocity_boundary_condition",
    "normal_velocity_boundary_condition",
    "normal_gradient_boundary_condition",
)

BoundaryData: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]
"""Face velocity in m/s, normal velocity in m/s and normal gradient of the tangential velocity in 1/s."""


class CouplingIterationRow(TypedDict, total=False):
    """Area-weighted RMS change of the boundary data over one coupling iteration.

    A one-domain run has its residuals at the top level, several domains below
    ``domains``; ``scaled_residual`` and ``converged`` cover all of them.
    """

    iteration: int
    normal_velocity_residual: float
    normal_gradient_residual: float
    scaled_residual: float
    converged: bool
    domains: dict[str, dict[str, float | bool]]


def _read_boundary_data(coupler, *, old=False, velocity=None) -> BoundaryData:
    suffix = "_old" if old else ""
    if old:
        velocity = coupler._velocity_boundary_condition_old
    return (
        np.asarray(velocity).copy(),
        np.asarray(getattr(coupler, "_" + _BOUNDARY_DATA_FIELDS[1] + suffix)).copy(),
        np.asarray(getattr(coupler, "_" + _BOUNDARY_DATA_FIELDS[2] + suffix)).copy(),
    )


def _write_boundary_data(coupler, values: BoundaryData, *, old=False) -> None:
    suffix = "_old" if old else ""
    for name, value in zip(_BOUNDARY_DATA_FIELDS, values, strict=True):
        if old or name != _BOUNDARY_DATA_FIELDS[0]:
            setattr(coupler, "_" + name + suffix, value.copy())


def _first_estimate(coupler, predictor: BoundaryData, face_normal: np.ndarray) -> BoundaryData:
    """Add the expected boundary-data correction of this interval to the VPM predictor data.

    The vorticity transfer and the FVM response change the boundary data at a
    rate ``r = c / Δt`` that varies slowly from interval to interval, while the
    intervals themselves differ (output times shorten some). The estimate for the
    next interval ``Δt`` is ``Δt r_k`` with ``boundary_data_extrapolation="constant"``,
    and with ``"linear"`` and two accepted corrections the rate extrapolated
    linearly in time to the middle of the next interval,
    ``Δt [r_k + (r_k − r_(k−1)) (Δt + Δt_k) / (Δt_k + Δt_(k−1))]``, which is
    ``2 c_k − c_(k−1)`` for equal intervals. The normal velocity is recomputed from
    the corrected face velocity.
    """
    correction = coupler.boundary_data_correction
    if correction is None:
        return predictor
    interval = float(coupler.vpm_time_step_size)
    rate = tuple(c / coupler.boundary_data_correction_interval for c in correction)
    previous = coupler.previous_boundary_data_correction
    if coupler.config.boundary_data_extrapolation == "linear" and previous is not None:
        previous_interval = coupler.previous_boundary_data_correction_interval
        previous_rate = tuple(c / previous_interval for c in previous)
        reach = (interval + coupler.boundary_data_correction_interval) / (
            coupler.boundary_data_correction_interval + previous_interval
        )
        rate = tuple(r + (r - p) * reach for r, p in zip(rate, previous_rate, strict=True))
    correction = tuple(interval * r for r in rate)
    velocity = predictor[0] + correction[0]
    return velocity, np.einsum("ij,ij->i", velocity, face_normal), predictor[2] + correction[2]


def coupling_tolerances(coupler, face_velocity: np.ndarray) -> tuple[float, float]:
    """Return the absolute tolerances of the normal-velocity (m/s) and normal-gradient (1/s) residuals.

    The configured tolerances are relative: to the reference speed ``U`` of the
    coupled boundary (:func:`reference_speed`) and to ``U / h`` with ``h`` the
    particle spacing, so one value fits every case.
    """
    speed = reference_speed(coupler, face_velocity)
    spacing = float(coupler.vpm_particle_spacing)
    return (
        coupler.config.normal_velocity_tolerance * speed,
        coupler.config.normal_gradient_tolerance * speed / spacing,
    )


def _residual_row(domain, updated: BoundaryData, estimate: BoundaryData, tolerance) -> dict:
    """Area-weighted RMS change of one domain's boundary data over a coupling iteration."""
    face_area = domain._face_geometry[2]
    normal = float(np.sqrt(np.average((updated[1] - estimate[1]) ** 2, weights=face_area)))
    gradient = float(
        np.sqrt(np.average(np.sum((updated[2] - estimate[2]) ** 2, axis=1), weights=face_area))
    )
    return {
        "normal_velocity_residual": normal,
        "normal_gradient_residual": gradient,
        "scaled_residual": max(normal / tolerance[0], gradient / tolerance[1]),
        "converged": bool(normal <= tolerance[0] and gradient <= tolerance[1]),
    }


def advance_coupling_iterations(coupler, next_velocities):
    """Repeat the FVM intervals and the vorticity transfers until the boundary data converge.

    The VPM advances once to the shared step's end-time predictor. Each coupling
    iteration restarts every FVM domain and the advected particles from the same
    state, solves the FVM substeps of every domain with its end-time estimate
    ``b^(m)`` of the boundary data, transfers the FVM vorticity of every domain
    and then evaluates the boundary data of every domain from the corrected
    particles. Those data are the next estimates. The iteration converges when
    the residuals of every domain meet their tolerances. The first estimate is
    :func:`_first_estimate`. Only the accepted iteration writes FVM output. MPI
    ranks share the root's residuals and stopping decision.

    Parameters
    ----------
    coupler : CoupledSolver
        The coupled solver; its domains hold the boundary data.
    next_velocities : sequence of numpy.ndarray
        Predictor face velocity of each domain, in m/s.

    Returns
    -------
    tuple
        Accepted transfer results (one per domain), FVM seconds including the
        accepted output, and transfer seconds including state capture and
        restoration.
    """
    domains = coupler.domains
    vpm = coupler.vpm_solver
    comm = coupler._comm
    started = perf_counter()
    with collective_phase(comm, "coupling-iteration state capture"):
        fvm_start = [capture_restart_state(domain.fvm_solver) for domain in domains]
        particles_start = (
            _particle_state_snapshot(vpm, slot="coupler-predictor") if coupler._is_root else None
        )
        old = [_read_boundary_data(domain, old=True) for domain in domains]
        predictor = [
            _read_boundary_data(domain, velocity=next_velocity)
            for domain, next_velocity in zip(domains, next_velocities, strict=True)
        ]
        estimate = [
            _first_estimate(domain, values, domain._face_geometry[1])
            for domain, values in zip(domains, predictor, strict=True)
        ]
    first_estimate = (
        "vpm_predictor"
        if all(e is p for e, p in zip(estimate, predictor, strict=True))
        else "previous_correction"
    )
    transfer_steps = [domain.vorticity_transfer.step for domain in domains]
    phase_seconds = {
        "state_capture": perf_counter() - started,
        "state_restore": 0.0,
        "boundary_refresh": 0.0,
        "accepted_output": 0.0,
    }
    fvm_seconds = transfer_seconds = 0.0
    rows: list[CouplingIterationRow] = []
    tolerances = [
        coupling_tolerances(domain, values[0])
        for domain, values in zip(domains, predictor, strict=True)
    ]
    results = []
    imposed = estimate
    for iteration in range(1, coupler.config.max_coupling_iterations + 1):
        started = perf_counter()
        if iteration > 1:
            for domain, state in zip(domains, fvm_start, strict=True):
                restore_restart_state(domain.fvm_solver, state)
            if coupler._is_root:
                _restore_particle_state(vpm, particles_start)
            for domain, step, values in zip(domains, transfer_steps, old, strict=True):
                domain.vorticity_transfer.step = step
                _write_boundary_data(domain, values, old=True)
        for domain, values in zip(domains, estimate, strict=True):
            _write_boundary_data(domain, values)
        phase_seconds["state_restore"] += perf_counter() - started
        for domain, values in zip(domains, estimate, strict=True):
            fvm_seconds += domain.advance_fvm(values[0])
        started = perf_counter()
        results, _ = coupler.transfer_vorticity_to_vpm()
        transfer_seconds += perf_counter() - started
        phase_seconds["boundary_refresh"] += sum(
            domain.refresh_boundary_data() for domain in domains
        )
        row: CouplingIterationRow | None = None
        imposed = estimate
        if coupler._is_root:
            updated = [_read_boundary_data(domain, old=True) for domain in domains]
            domain_rows = {
                domain.name: _residual_row(domain, new, guess, tolerance)
                for domain, new, guess, tolerance in zip(
                    domains, updated, estimate, tolerances, strict=True
                )
            }
            row = {
                "iteration": iteration,
                "scaled_residual": max(item["scaled_residual"] for item in domain_rows.values()),
                "converged": all(item["converged"] for item in domain_rows.values()),
                **domain_records(coupler, domain_rows),
            }
            estimate = updated
        if comm is not None and comm.Get_size() > 1:
            row = comm.bcast(row, root=0)
        assert row is not None
        rows.append(row)
        if row["converged"]:
            break
    converged = rows[-1]["converged"]
    if not converged:
        if coupler.config.stop_on_unconverged_coupling:
            raise RuntimeError(_unconverged_message(coupler, rows[-1], tolerances))
        if coupler._is_root:
            # The FVM ended the interval on the data it was given, so the next interval
            # starts from them rather than from the newer evaluation.
            for domain, values in zip(domains, imposed, strict=True):
                _write_boundary_data(domain, values, old=True)
    # The accepted end-of-interval data are those evaluated in the last iteration.
    for domain, accepted, first in zip(domains, estimate, predictor, strict=True):
        domain.previous_boundary_data_correction = (
            domain.boundary_data_correction if converged else None
        )
        domain.previous_boundary_data_correction_interval = domain.boundary_data_correction_interval
        domain.boundary_data_correction_interval = float(coupler.vpm_time_step_size)
        domain.boundary_data_correction = (
            tuple(a - f for a, f in zip(accepted, first, strict=True)) if converged else None
        )
    started = perf_counter()
    for domain in domains:
        domain.fvm_solver.write_step_output()
    phase_seconds["accepted_output"] = perf_counter() - started
    fvm_seconds += phase_seconds["accepted_output"]
    transfer_seconds += phase_seconds["state_capture"] + phase_seconds["state_restore"]
    coupler._last_coupling_iteration_diagnostics = {
        "iterations": len(rows),
        "converged": converged,
        "first_estimate": first_estimate,
        "residuals": rows,
        "phase_seconds": phase_seconds,
    }
    if coupler._is_root:
        for domain in domains:
            last = rows[-1] if len(domains) == 1 else rows[-1]["domains"][domain.name]
            logger.log(
                logging.INFO if converged else logging.WARNING,
                "coupler  coupling iterations%s | iterations=%d | converged=%s | "
                "normal velocity=%.3e | normal gradient=%.3e",
                "" if len(domains) == 1 else f", {domain.name}",
                len(rows),
                converged,
                last["normal_velocity_residual"],
                last["normal_gradient_residual"],
            )
    return results, fvm_seconds, transfer_seconds


def _unconverged_message(coupler, row, tolerances) -> str:
    """The stop message of unconverged coupling iterations, for the domain furthest from its tolerance."""
    domains = coupler.domains
    records = [row] if len(domains) == 1 else [row["domains"][domain.name] for domain in domains]
    index = max(range(len(domains)), key=lambda item: records[item]["scaled_residual"])
    record, tolerance = records[index], tolerances[index]
    where = "" if len(domains) == 1 else f" in the FVM domain {domains[index].name!r}"
    return (
        f"the coupling iterations did not converge in {row['iteration']} iterations at "
        f"t = {coupler.coupling_time(coupler.step + 1):.6g} s{where}: normal-velocity residual "
        f"{record['normal_velocity_residual']:.3e} m/s (tolerance {tolerance[0]:.3e}), "
        f"normal-gradient residual {record['normal_gradient_residual']:.3e} 1/s "
        f"(tolerance {tolerance[1]:.3e})"
    )
