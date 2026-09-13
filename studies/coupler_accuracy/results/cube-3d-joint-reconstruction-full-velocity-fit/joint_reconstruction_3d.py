#!/usr/bin/env python3
"""Frozen native 3D cube: regularized joint velocity/vorticity reconstruction.

Uses the saved projected-renewal failure as an immutable source/basis oracle.
All outer particles remain fixed. The original fit and verification cells stay
disjoint, and the body-complete velocity operator includes the production
108-triangle Neumann panel response with its zero-source-flux constraint.
Candidates are studied offline; no solver checkpoint or production gate is
modified. The Gaussian sum and curl of Biot--Savart velocity are measured as
distinct fields.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.spatial import cKDTree
from scipy.special import gammainc
import taichi as ti

import openonda.fvm as fvm
import openonda.vpm as vpm
from source.coupler.renewal_projection import (
    gaussian_velocity_operator,
    geometric_renewal_mask,
    sparse_gaussian_vorticity_basis,
)
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.vpm.boundary_elements.panels.kernels.induced_velocity import (
    compute_source_induced_velocity_kernel,
)
from source.solvers.vpm.boundary_elements.panels.solver.linear_solvers import (
    EqualityConstrainedLeastSquaresFactorization,
)
from studies.coupler_accuracy.cube_boundary_oracle import (
    CASE,
    ROOT,
    field_rms,
    hash_file,
    setup_for,
)


def gaussian_velocity_curl_operator(target, position, radius):
    """Analytical curl of the 3D Gaussian Biot--Savart operator, in float64.

    For u = Gamma x r f(r), curl(u) = [(zeta-f) I +
    (3 f-zeta) r r^T/r²] Gamma. At r=0 this is (2/3) zeta(0) Gamma,
    whereas the raw Gaussian vorticity sum is zeta(0) Gamma. The radial
    coefficient uses a Taylor series near zero to avoid cancellation.
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    radius = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(position),))
    if not all(np.all(np.isfinite(x)) for x in (target, position, radius)) or np.any(radius <= 0):
        raise ValueError("Positions must be finite and Gaussian radii positive")
    delta = target[:, None] - position[None]
    r2 = np.sum(delta**2, axis=-1)
    q2 = r2 / radius[None]**2
    peak = 1 / (np.pi**1.5 * radius[None]**3)
    zeta = peak * np.exp(-q2)
    f = np.divide(gammainc(1.5, q2), 4 * np.pi * r2**1.5,
                  out=np.zeros_like(r2), where=r2 > 0)
    small = q2 < 1e-4
    series = peak * (1 / 3 - q2 / 5 + q2**2 / 14 - q2**3 / 54 + q2**4 / 264)
    f[small] = series[small]
    radial = np.divide(3 * f - zeta, r2, out=np.zeros_like(r2), where=r2 > 0)
    radial_series = peak / radius[None]**2 * (2 / 5 - 2 * q2 / 7 + q2**2 / 9 - q2**3 / 33)
    radial[small] = radial_series[small]
    blocks = (zeta - f)[..., None, None] * np.eye(3)
    blocks += radial[..., None, None] * delta[..., :, None] * delta[..., None, :]
    return blocks.transpose(0, 2, 1, 3).reshape(3 * len(target), 3 * len(position))


class CubePanelResponse:
    """Linear map through the real f64 Neumann panel kernels and constraint."""

    def __init__(self):
        if ti.lang.impl.get_runtime().prog is None:
            ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
        self.panel = vpm.PanelSolver(max_n_panels=128, float_dtype="f64", linear_solver="SCIPY",
                                     boundary_condition_type="NEUMANN", density=1,
                                     freestream_velocity=np.array([1, 0, 0]),
                                     coupling_scope="vpm_boundary_condition")
        self.panel.add_surface("cube", str(CASE / "assets/cube.stl"), reference_area=1)
        self.panel.initialize()
        lattice = self.panel.lattice
        self.count = lattice.n_panels
        assert self.count == 108
        self.vertices = lattice.vertex_position.to_numpy()[:self.count]
        self.normal = lattice.normal.to_numpy()[:self.count]
        self.centres = lattice.panel_centre.to_numpy()[:self.count]
        self.area = lattice.area.to_numpy()[:self.count]
        a = self.panel.aerodynamic_influence_coefficient.to_numpy()[:self.count, :self.count]
        constraints = self.panel._neumann_constraints(self.area, self.count)
        self.factor = EqualityConstrainedLeastSquaresFactorization.factorize(a, constraints)
        self.response = np.column_stack([self.factor.solve(e) for e in np.eye(self.count)])
        np.testing.assert_allclose(constraints @ self.response, 0, rtol=0, atol=3e-14)

    def source_velocity(self, target):
        target = np.ascontiguousarray(target, dtype=np.float64)
        result = np.empty((3 * len(target), self.count))
        for j in range(self.count):
            amplitude = np.eye(self.count)[j]
            values = np.zeros_like(target)
            compute_source_induced_velocity_kernel(self.vertices, self.normal, amplitude, target, values)
            result[:, j] = values.ravel()
        return result

    def velocity_operator(self, target, position, radius):
        source = self.source_velocity(target)
        normal_incidence = gaussian_velocity_operator(self.centres, position, radius, normal=self.normal)
        direct = gaussian_velocity_operator(target, position, radius)
        response = source @ self.response
        complete = direct - response @ normal_incidence
        freestream = np.tile([1., 0., 0.], (len(target), 1)).ravel()
        background = freestream - response @ self.normal[:, 0]
        return complete, background


def regularized_projection(omega_operator, velocity_operator, target_omega, target_velocity,
                           prior, omega_weights, velocity_weights, *, velocity_weight,
                           prior_weight=0.05, max_strength_l1_ratio=2.0, budget_method="segment"):
    """Fit an absolute donor-anchored state, then enforce a finite L1 budget.

    The regularization anchor is immutable donor data, not the previously
    fitted candidate. The one-dimensional shortening of its correction keeps
    the L1 coefficient budget; no component clipping or gate relaxation occurs.
    """
    n = prior.size
    prior_flat = prior.ravel()
    scale = np.linalg.norm(prior_flat)
    if scale <= 0:
        raise ValueError("This experiment requires a nonzero donor anchor")
    if prior_weight <= 0 or velocity_weight < 0 or max_strength_l1_ratio < 1:
        raise ValueError("Require positive prior penalty, nonnegative velocity weight, and L1 ratio >= 1")
    ow = np.repeat(omega_weights, 3)
    vw = np.repeat(velocity_weights, 3) * velocity_weight
    p = prior_weight / scale

    def omega_apply(x):
        if omega_operator.shape[1] == len(prior):
            return np.asarray(omega_operator @ x.reshape(-1, 3)).ravel()
        return omega_operator @ x

    def omega_transpose(y):
        if omega_operator.shape[1] == len(prior):
            return np.asarray(omega_operator.T @ y.reshape(-1, 3)).ravel()
        return omega_operator.T @ y

    m_omega, m_velocity = len(ow), len(vw)

    def matvec(x):
        velocity = velocity_operator @ x if velocity_weight else np.zeros(m_velocity)
        return np.r_[ow * omega_apply(x), vw * velocity, p * x]

    def rmatvec(y):
        velocity = (velocity_operator.T @ (vw * y[m_omega:m_omega + m_velocity])
                    if velocity_weight else np.zeros(n))
        return omega_transpose(ow * y[:m_omega]) + velocity + p * y[m_omega + m_velocity:]

    operator = LinearOperator((m_omega + m_velocity + n, n), matvec=matvec, rmatvec=rmatvec, dtype=float)
    rhs = np.r_[ow * (target_omega.ravel() - omega_apply(prior_flat)),
                vw * (target_velocity.ravel() - velocity_operator @ prior_flat), np.zeros(n)]
    answer = lsmr(operator, rhs, atol=1e-9, btol=1e-9, maxiter=1000)
    delta = answer[0].reshape(-1, 3)
    budget = max_strength_l1_ratio * np.linalg.norm(prior, axis=1).sum()
    extra = {}
    alpha = 1.0
    if np.linalg.norm(prior + delta, axis=1).sum() > budget:
        lo, hi = 0.0, 1.0
        for _ in range(55):
            mid = (lo + hi) / 2
            if np.linalg.norm(prior + mid * delta, axis=1).sum() <= budget:
                lo = mid
            else:
                hi = mid
        alpha = lo
    strength = prior + alpha * delta
    if budget_method == "projected" and alpha < 1:
        strength, extra = constrained_least_squares(operator, rhs, prior, budget, strength)
    elif budget_method not in {"segment", "projected"}:
        raise ValueError("Unknown coefficient budget method")
    return strength, {"lsmr_stop": int(answer[1]), "lsmr_iterations": int(answer[2]),
                      "lsmr_residual_norm": float(answer[3]),
                      "lsmr_normal_residual_norm": float(answer[4]),
                      "lsmr_condition_estimate": float(answer[6]),
                      "prior_weight": prior_weight, "velocity_weight": velocity_weight,
                      "l1_budget_ratio": max_strength_l1_ratio, "budget_method": budget_method,
                      "segment_correction_fraction": alpha, **extra}


def project_strength_budget(strength, budget):
    """Euclidean projection onto sum_p ||Gamma_p||_2 <= budget."""
    lengths = np.linalg.norm(strength, axis=1)
    if lengths.sum() <= budget:
        return strength.copy()
    order = np.sort(lengths)[::-1]
    threshold_candidates = (np.cumsum(order) - budget) / np.arange(1, len(order) + 1)
    active = np.flatnonzero(order > threshold_candidates)[-1]
    threshold = threshold_candidates[active]
    scale = np.divide(np.maximum(lengths - threshold, 0), lengths,
                      out=np.zeros_like(lengths), where=lengths > 0)
    return strength * scale[:, None]


def constrained_least_squares(operator, rhs, prior, budget, initial, max_iterations=4000):
    """Monotone accelerated projected gradient with checked backtracking.

    The fit is convex. A projected-gradient mapping measures stationarity;
    fitting residual alone is not a convergence check at an active budget.
    """
    prior_flat = prior.ravel()
    x = initial.ravel() - prior_flat
    y, momentum = x.copy(), 1.0
    probe = np.random.default_rng(610).normal(size=len(x))
    probe /= np.linalg.norm(probe)
    for _ in range(50):
        probe = operator.rmatvec(operator.matvec(probe))
        probe /= np.linalg.norm(probe)
    lipschitz = 1.05 * float(probe @ operator.rmatvec(operator.matvec(probe)))
    residual = operator.matvec(x) - rhs
    value = 0.5 * residual @ residual
    initial_value = float(value)
    gradient_scale = max(float(np.linalg.norm(operator.rmatvec(rhs))), 1e-30)
    relative_mapping = np.inf
    converged = False
    for iteration in range(1, max_iterations + 1):
        ry = operator.matvec(y) - rhs
        gradient = operator.rmatvec(ry)
        fy = 0.5 * ry @ ry
        while True:
            trial = (project_strength_budget((prior_flat + y - gradient / lipschitz).reshape(-1, 3), budget).ravel()
                     - prior_flat)
            step = trial - y
            rt = operator.matvec(trial) - rhs
            ft = 0.5 * rt @ rt
            upper = fy + gradient @ step + 0.5 * lipschitz * (step @ step)
            if ft <= upper + 1e-14 * max(1, abs(fy)):
                break
            lipschitz *= 2
        if ft > value + 1e-14 * max(1, abs(value)):
            # Restart an extrapolation that increases the objective.
            y, momentum = x.copy(), 1.0
            continue
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        y = trial + (momentum - 1) / next_momentum * (trial - x)
        x, value, momentum = trial, ft, next_momentum
        if iteration % 10 == 0:
            gx = operator.rmatvec(rt)
            projected = project_strength_budget((prior_flat + x - gx / lipschitz).reshape(-1, 3), budget).ravel()
            relative_mapping = float(lipschitz * np.linalg.norm(prior_flat + x - projected) / gradient_scale)
            if relative_mapping < 1e-8:
                converged = True
                break
    return (prior_flat + x).reshape(-1, 3), {
        "constrained_converged": converged, "constrained_iterations": iteration,
        "constrained_initial_objective": initial_value, "constrained_objective": float(value),
        "constrained_relative_gradient_mapping": relative_mapping,
        "constrained_lipschitz": lipschitz}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    sources = [hash_file(p) for p in (Path(__file__), args.failure, args.oracle / "initial-cell-fields.npz",
                                     args.oracle / "full-native-mesh.npz",
                                     CASE / "assets/cube.stl",
                                     ROOT / "source/coupler/renewal_projection.py",
                                     ROOT / "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
                                     ROOT / "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py",
                                     ROOT / "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
                                     ROOT / "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py")]
    with np.load(args.failure, allow_pickle=False) as data:
        failure = {k: data[k].copy() for k in data.files}
    with np.load(args.oracle / "initial-cell-fields.npz", allow_pickle=False) as data:
        centres, velocity, pressure = (data[k].copy() for k in ("centres", "velocity", "pressure"))
        physical_time = float(data["physical_time"])
    mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    with fvm.create_fvm_solver(setup_for(mesh, "donor", 0.01, 1), mesh=copy.deepcopy(mesh),
                               case_dir=output / "donor") as full:
        full.set_initial_state(velocity, pressure)
        omega, volume = full.get_vorticity_field().copy(), full.get_cell_volume().copy()
    tree = cKDTree(centres)
    distance, small_ids = tree.query(failure["fvm_position"])
    np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(velocity[small_ids], failure["fvm_velocity"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(omega[small_ids], failure["fvm_vorticity"], rtol=0, atol=1e-12)
    h = float(failure["particle_spacing"])
    retained = np.linalg.norm(omega, axis=1) >= 0.02
    initial_position = centres[retained].astype(np.float32).astype(float)
    initial_strength = (omega * volume[:, None])[retained].astype(np.float32).astype(float)
    preserved = ~geometric_renewal_mask(initial_position, failure["renewal_bounds"], particle_spacing=h)
    outer_position, outer_strength = initial_position[preserved], initial_strength[preserved]
    position, radius, prior = (failure[k] for k in ("solve_position", "solve_radius", "solve_prior"))
    # Verify the retained part of the failed basis really is this exact seed.
    prior_nonzero = np.linalg.norm(prior, axis=1) > 0
    distance, original_ids = cKDTree(initial_position).query(position[prior_nonzero])
    np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(prior[prior_nonzero], initial_strength[original_ids], rtol=0, atol=1e-12)

    fit, held = failure["fit_position"], failure["verification_position"]
    _, fit_ids = tree.query(fit)
    _, held_ids = tree.query(held)
    assert not set(fit_ids) & set(held_ids)
    rng = np.random.default_rng(20260914)
    velocity_rows = np.sort(rng.choice(len(fit), min(args.velocity_points, len(fit)), replace=False))
    vf, vh = fit[velocity_rows], held
    fit_volume, held_volume = volume[fit_ids], volume[held_ids]
    omega_scale = field_rms(omega[fit_ids], fit_volume)
    weights_omega = np.sqrt(fit_volume / fit_volume.sum()) / omega_scale
    weights_velocity = np.sqrt(fit_volume[velocity_rows] / fit_volume[velocity_rows].sum())
    panel = CubePanelResponse()
    print(json.dumps({"event": "basis_ready", "renewable_particles": len(position),
                      "preserved_particles": len(outer_position), "fit_points": len(fit),
                      "held_points": len(held), "velocity_points": len(vf)}), flush=True)
    gfit = sparse_gaussian_vorticity_basis(fit, position, radius)
    gheld = sparse_gaussian_vorticity_basis(held, position, radius)
    outer_gfit = sparse_gaussian_vorticity_basis(fit, outer_position, h) @ outer_strength
    outer_gheld = sparse_gaussian_vorticity_basis(held, outer_position, h) @ outer_strength
    cfit = gaussian_velocity_curl_operator(fit, position, radius)
    cheld = gaussian_velocity_curl_operator(held, position, radius)
    outer_cfit = (gaussian_velocity_curl_operator(fit, outer_position, h) @ outer_strength.ravel()).reshape(-1, 3)
    outer_cheld = (gaussian_velocity_curl_operator(held, outer_position, h) @ outer_strength.ravel()).reshape(-1, 3)
    ufit, background_fit = panel.velocity_operator(vf, position, radius)
    uheld, background_held = panel.velocity_operator(vh, position, radius)
    outer_ufit, check_background = panel.velocity_operator(vf, outer_position, h)
    np.testing.assert_allclose(background_fit, check_background, rtol=0, atol=1e-13)
    outer_uheld, _ = panel.velocity_operator(vh, outer_position, h)
    background_fit += outer_ufit @ outer_strength.ravel()
    background_held += outer_uheld @ outer_strength.ravel()
    target_velocity = velocity[fit_ids[velocity_rows]] - background_fit.reshape(-1, 3)
    prior_l1 = np.linalg.norm(prior, axis=1).sum()
    records, fields = [], {}

    def measure(name, strength, diagnostics):
        fitted_velocity = (ufit @ strength.ravel() + background_fit).reshape(-1, 3)
        raw = gheld @ strength + outer_gheld
        physical = (cheld @ strength.ravel()).reshape(-1, 3) + outer_cheld
        actual_velocity = (uheld @ strength.ravel() + background_held).reshape(-1, 3)
        near = np.max(np.abs(held), axis=1) < 1
        record = {"name": name, **diagnostics,
                  "fit_velocity_rms_over_Uinf": field_rms(fitted_velocity - velocity[fit_ids[velocity_rows]], fit_volume[velocity_rows]),
                  "fit_raw_vorticity_relative_error": field_rms(gfit @ strength + outer_gfit - omega[fit_ids], fit_volume) / omega_scale,
                  "fit_velocity_curl_relative_error": field_rms((cfit @ strength.ravel()).reshape(-1, 3) + outer_cfit - omega[fit_ids], fit_volume) / omega_scale,
                  "held_raw_vorticity_unweighted_relative_error": float(np.linalg.norm(raw - omega[held_ids]) / np.linalg.norm(omega[held_ids])),
                  "held_raw_vorticity_relative_error": field_rms(raw - omega[held_ids], held_volume) / field_rms(omega[held_ids], held_volume),
                  "held_velocity_curl_relative_error": field_rms(physical - omega[held_ids], held_volume) / field_rms(omega[held_ids], held_volume),
                  "held_raw_vs_velocity_curl_relative_difference": field_rms(raw - physical, held_volume) / field_rms(omega[held_ids], held_volume),
                  "held_velocity_rms_over_Uinf": field_rms(actual_velocity - velocity[held_ids], held_volume),
                  "held_near_velocity_rms_over_Uinf": field_rms((actual_velocity - velocity[held_ids])[near], held_volume[near]),
                  "renewable_strength_l1": float(np.linalg.norm(strength, axis=1).sum()),
                  "renewable_strength_l1_over_prior": float(np.linalg.norm(strength, axis=1).sum() / prior_l1),
                  "maximum_strength": float(np.linalg.norm(strength, axis=1).max()),
                  "coefficient_relative_change": float(np.linalg.norm(strength - prior) / np.linalg.norm(prior)),
                  "net_strength_change": (strength - prior).sum(axis=0).tolist(),
                  "elapsed_seconds": time.perf_counter() - started}
        records.append(record)
        fields[name + "__strength"] = strength
        fields[name + "__velocity"] = actual_velocity
        fields[name + "__raw_vorticity"] = raw
        fields[name + "__velocity_curl"] = physical
        print(json.dumps(record), flush=True)
        (output / "history.json").write_text(json.dumps(records, indent=2) + "\n")

    measure("donor_volume_vorticity", prior, {})
    measure("original_unregularized_fit", failure["solve_strength"], {})
    np.testing.assert_allclose(records[-1]["held_raw_vorticity_unweighted_relative_error"],
                               float(failure["verification_error"]), rtol=1e-7)
    cases = [
        ("regularized_raw_vorticity", gfit, outer_gfit, 0, 1),
        ("joint_raw_vorticity_velocity", gfit, outer_gfit, 5, 1),
        ("joint_velocity_curl_velocity", cfit, outer_cfit, 5, 1),
    ]
    if args.include_velocity_only:
        cases.append(("velocity_only", gfit, outer_gfit, 5, 0))
    for name, operator, outer, velocity_weight, omega_weight in cases:
        strength, diagnostics = regularized_projection(
            operator, ufit, omega[fit_ids] - outer, target_velocity, prior,
            omega_weight * weights_omega, weights_velocity, velocity_weight=velocity_weight,
            budget_method=args.budget_method)
        diagnostics["omega_weight"] = omega_weight
        measure(name, strength, diagnostics)
    np.savez_compressed(output / "held-fields.npz", position=held, volume=held_volume,
                        fvm_velocity=velocity[held_ids], fvm_vorticity=omega[held_ids],
                        renewable_position=position, radius=radius, **fields)
    report = {"schema": "openonda-frozen-joint-3d-reconstruction/1", "status": "complete",
              "spatial_dimensions": 3, "physical_time": physical_time,
              "renewal_bounds": failure["renewal_bounds"].tolist(), "particle_spacing": h,
              "renewable_particles": len(position), "preserved_particles": len(outer_position),
              "fit_points": len(fit), "held_points": len(held), "velocity_fit_points": len(vf),
              "error_weighting": "Cell-volume-weighted vector RMS unless explicitly marked unweighted",
              "budget_method": args.budget_method,
              "panel_count": panel.count, "precision": "f64 direct kernels and constrained panel response",
              "results": records, "sources": sources,
              "limitations": ["Frozen coarse native 3D cube only, not evolved hybrid acceptance.",
                              "Regularized candidates share an immutable donor anchor; no production state is changed.",
                              "The L1 bound limits coefficient growth but does not enforce circulation/impulse constraints.",
                              "All targets are native cell samples; subcell/continuum convergence remains separate.",
                              "Dense direct operators are an offline accuracy oracle, not a production scalability solution."]}
    (output / "joint-reconstruction-3d.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, default=ROOT / "studies/coupler_accuracy/results/cube-3d-frozen-projected-guard/hybrid/renewal_projection_failure_oracle.npz")
    parser.add_argument("--oracle", type=Path, default=ROOT / "studies/coupler_accuracy/results/cube-3d-oracle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--velocity-points", type=int, default=384)
    parser.add_argument("--budget-method", choices=("segment", "projected"), default="segment")
    parser.add_argument("--include-velocity-only", action="store_true")
    run(parser.parse_args())
