"""Budget one saved cube stage using the actual uniform-core particle cloud.

The three resolved terms use the cloud's self-induced velocity plus the saved
unit freestream. Native-minus-self motion/stretching is retained separately;
it includes the body contribution and native induction discrepancy. This avoids
silently extending a body potential through a solid in a Helmholtz identity.

Run: python studies/coupler_accuracy/cube_represented_evolution_budget.py
     --audit DIRECTORY --output NEW_DIRECTORY
No solver is constructed or advanced. Fourier workers are limited to two.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import TypedDict, cast

from cube_wake_nullspace_probe import direct_velocity_rate
from cube_wake_operator_audit import reflection_asymmetry
from cube_wake_particle_probe import direct_gaussian, rms
from numba import set_num_threads
import numpy as np
from numpy.typing import NDArray
from scipy import fft
from threadpoolctl import threadpool_limits

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


class RateFields(TypedDict):
    """Saved shape-(392,3) velocity-rate contributions in m/s^2."""

    native_free_space_direct: FloatArray
    native_periodic: FloatArray
    self_euler: FloatArray
    self_longitudinal: FloatArray
    self_finite_core: FloatArray
    native_minus_self: FloatArray
    closure: FloatArray


class PeriodicCloud:
    """One padded Fourier quadrature for a saved common-core Gaussian cloud.

    Positions are snapped only to their existing GBD lattice, with that error
    reported against the saved float32 coordinates. Arrays use C ordering in
    x,y,z; vector components are last. The periodic curl projector removes the
    zero mode, which is retained in the velocity-invisible part of vorticity.
    """

    def __init__(
        self,
        position: FloatArray,
        points: FloatArray,
        spacing: float,
        lengths: tuple[float, float, float],
        core_radius: float,
    ) -> None:
        self.spacing = spacing
        self.shape = tuple(round(length / spacing) for length in lengths)
        if int(np.prod(self.shape)) > 14_000_000:
            raise ValueError("bounded Fourier study exceeds its 14-million-node limit")
        centre = np.array([2.34, 0.0, 0.0])
        self.origin = 0.03 + spacing * np.rint((centre - np.asarray(lengths) / 2 - 0.03) / spacing)
        index = np.rint((position - self.origin) / spacing).astype(int)
        self.snapped = self.origin + spacing * index
        if np.max(np.abs(position - self.snapped)) > 3e-7:
            raise ValueError("saved cloud is not the expected float32 GBD lattice")
        if np.any(index < 0) or np.any(index >= np.asarray(self.shape)):
            raise ValueError("Fourier box does not contain the complete saved cloud")
        self.source_index = tuple(index.T)
        shift = points[0] - self.origin - spacing * np.rint((points[0] - self.origin) / spacing)
        query_index = np.rint((points - self.origin - shift) / spacing).astype(int)
        np.testing.assert_allclose(
            self.origin + shift + spacing * query_index, points, rtol=0, atol=1e-12
        )
        self.query_index = tuple((query_index % np.asarray(self.shape)).T)
        self.wave = (
            (2 * np.pi * fft.fftfreq(self.shape[0], spacing))[:, None, None],
            (2 * np.pi * fft.fftfreq(self.shape[1], spacing))[None, :, None],
            (2 * np.pi * fft.rfftfreq(self.shape[2], spacing))[None, None, :],
        )
        squared = self.wave[0] ** 2 + self.wave[1] ** 2 + self.wave[2] ** 2
        self.inverse_squared = np.zeros_like(squared)
        np.divide(1, squared, out=self.inverse_squared, where=squared > 0)
        self.gaussian = np.exp(-0.25 * core_radius**2 * squared)
        self.phase = np.exp(1j * sum(k * shift[j] for j, k in enumerate(self.wave)))
        self.spectral_shape = squared.shape

    def transform(self, field: FloatArray) -> ComplexArray:
        """Return the scalar 3D real Fourier transform with two workers."""
        return fft.rfftn(field, workers=2)

    def invert(self, spectrum: ComplexArray) -> FloatArray:
        """Return a float64 scalar field on the declared periodic quadrature."""
        return fft.irfftn(spectrum, s=self.shape, workers=2)

    def deposit(self, coefficient: FloatArray) -> ComplexArray:
        """Deposit one scalar particle coefficient divided by quadrature volume."""
        field = np.zeros(self.shape, dtype=np.float64)
        np.add.at(field, self.source_index, coefficient / self.spacing**3)
        return self.transform(field)

    def vector_spectrum(self) -> ComplexArray:
        """Allocate three complex Fourier components with no initialized values."""
        return np.empty((*self.spectral_shape, 3), dtype=np.complex128)

    def vector_field(self, spectrum: ComplexArray) -> FloatArray:
        """Reconstruct the three independent real components."""
        field = np.empty((*self.shape, 3), dtype=np.float64)
        for component in range(3):
            field[..., component] = self.invert(spectrum[..., component])
        return field

    def sample(self, spectrum: ComplexArray) -> FloatArray:
        """Evaluate a vector spectrum at the exact common-shift diagnostic probes."""
        return np.column_stack(
            [self.invert(spectrum[..., j] * self.phase)[self.query_index] for j in range(3)]
        )

    def induce(self, spectrum: ComplexArray) -> ComplexArray:
        """Apply periodic Biot-Savart, with its mean velocity set to zero."""
        velocity = self.vector_spectrum()
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            velocity[..., i] = (
                1j
                * self.inverse_squared
                * (self.wave[j] * spectrum[..., k] - self.wave[k] * spectrum[..., j])
            )
        return velocity

    def project(self, spectrum: ComplexArray) -> ComplexArray:
        """Project a vector spectrum in place onto periodic curl fields."""
        longitudinal = sum(self.wave[j] * spectrum[..., j] for j in range(3))
        longitudinal *= self.inverse_squared
        for j in range(3):
            spectrum[..., j] -= self.wave[j] * longitudinal
        spectrum[0, 0, 0] = 0.0
        return spectrum

    def particle_derivative(
        self,
        strength: FloatArray,
        velocity: FloatArray,
        strength_rate: FloatArray,
    ) -> ComplexArray:
        """Differentiate the moving Gaussian sum, including all centre motion.

        In Fourier space: wdot_hat=G_hat*(Gamma_dot_hat-i*k_j*(Gamma*U_j)_hat).
        Core radii are fixed during this inviscid instantaneous stage.
        """
        derivative = self.vector_spectrum()
        for i in range(3):
            value = self.deposit(strength_rate[:, i])
            for j in range(3):
                value -= 1j * self.wave[j] * self.deposit(strength[:, i] * velocity[:, j])
            derivative[..., i] = self.gaussian * value
        return derivative


def _one_case(
    fields: dict[str, FloatArray],
    sigma: float,
    spacing: float,
    lengths: tuple[float, float, float],
    direct_rate: FloatArray,
    selected_self: tuple[FloatArray, FloatArray],
    coefficient_potential_path: Path | None = None,
) -> tuple[dict, RateFields]:
    """Compute the actual-cloud self-flow budget and keep native extra rates separate."""
    started = time.perf_counter()
    position, strength, points = (
        fields[k] for k in ("particle_position", "particle_strength", "points")
    )
    grid = PeriodicCloud(position, points, spacing, lengths, sigma)
    native_derivative = grid.particle_derivative(
        strength, fields["stage_velocity"], fields["stage_strength_rate"]
    )
    native_hat = grid.induce(native_derivative)
    native_rate = grid.sample(native_hat)
    del native_derivative, native_hat

    w_hat = grid.vector_spectrum()
    for i in range(3):
        w_hat[..., i] = grid.gaussian * grid.deposit(strength[:, i])
    u_hat = grid.induce(w_hat)
    q_hat = grid.project(w_hat.copy())
    potential_record = None
    if coefficient_potential_path is not None:
        # Q a = grad(psi_a) + mean(a), with w=G*a. Keeping a scalar
        # spectrum is sufficient to recover both coefficient projections.
        potential = np.zeros(grid.spectral_shape, dtype=np.complex128)
        for j in range(3):
            potential += grid.wave[j] * w_hat[..., j]
        potential *= -1j * grid.inverse_squared / grid.gaussian
        coefficient_potential_path.parent.mkdir(parents=True, exist_ok=True)
        with coefficient_potential_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                coefficient_scalar_potential_spectrum=potential,
                coefficient_mean_density=strength.sum(axis=0) / np.prod(lengths),
                shape=np.asarray(grid.shape),
                origin=grid.origin,
                spacing=np.asarray(spacing),
                core_radius=np.asarray(sigma),
            )
        potential_record = {
            "path": str(coefficient_potential_path.resolve()),
            "sha256": hashlib.sha256(coefficient_potential_path.read_bytes()).hexdigest(),
            "meaning": "Periodic coefficient-density projection: a_L_hat=i*k*psi_hat plus mean(a); a_T=a-a_L. Truncating this field as free-space particles is not velocity-preserving without an independent check.",
        }
        del potential
    validation = {
        "represented_vorticity_probe_error_rms": rms(grid.sample(w_hat) - fields["accepted_omega"]),
        "velocity_curl_probe_error_rms": rms(grid.sample(q_hat) - fields["accepted_curl"]),
        "self_velocity_probe_error_rms": rms(grid.sample(u_hat) - fields["accepted_velocity"]),
        "native_velocity_rate_probe_error_rms": rms(native_rate - direct_rate),
        "source_snap_max_m": float(np.max(np.abs(position - grid.snapped))),
    }
    w = grid.vector_field(w_hat)
    q = grid.vector_field(q_hat)
    del q_hat
    u = grid.vector_field(u_hat)
    u[..., 0] += 1.0
    self_particle_velocity = u[grid.source_index]
    selected = fields["selected"].astype(int)
    validation["self_velocity_selected_particle_error_rms"] = rms(
        self_particle_velocity[selected] - selected_self[0] - [1.0, 0.0, 0.0]
    )

    lie_hat, longitudinal_hat = grid.vector_spectrum(), grid.vector_spectrum()
    self_strength_rate = np.zeros_like(strength)
    selected_gradient = np.empty((len(selected), 3, 3))
    for i in range(3):
        lie = np.zeros(grid.shape)
        longitudinal = np.zeros(grid.shape)
        for j in range(3):
            gradient = grid.invert(1j * grid.wave[i] * u_hat[..., j])
            self_strength_rate[:, i] += gradient[grid.source_index] * strength[:, j]
            selected_gradient[:, j, i] = gradient[grid.source_index][selected]
            longitudinal += 2 * gradient * (w[..., j] - q[..., j])
            lie += gradient * w[..., j]
            del gradient
            lie -= u[..., j] * grid.invert(1j * grid.wave[j] * w_hat[..., i])
        lie_hat[..., i] = grid.transform(lie)
        longitudinal_hat[..., i] = grid.transform(longitudinal)
        del lie, longitudinal
    validation["self_gradient_selected_particle_error_rms"] = rms(
        selected_gradient - selected_self[1]
    )

    euler_hat = grid.vector_spectrum()
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        euler_hat[..., i] = grid.transform(u[..., j] * q[..., k] - u[..., k] * q[..., j])
    del w, q, u, w_hat, u_hat
    euler_rate = grid.sample(grid.project(euler_hat))
    del euler_hat
    induced_longitudinal = grid.induce(longitudinal_hat)
    longitudinal_rate = grid.sample(induced_longitudinal)
    del longitudinal_hat, induced_longitudinal

    self_derivative = grid.particle_derivative(strength, self_particle_velocity, self_strength_rate)
    self_hat = grid.induce(self_derivative)
    self_rate = grid.sample(self_hat)
    del self_hat
    # Independently form R_core = wdot_particle - Lw, with Lw differentiated
    # above. Closing E+longitudinal+R then tests the product-rule quadrature.
    self_derivative -= lie_hat
    del lie_hat
    core_hat = grid.induce(self_derivative)
    core_rate = grid.sample(core_hat)
    del self_derivative, core_hat
    native_extra = native_rate - self_rate
    closure = euler_rate + longitudinal_rate + core_rate + native_extra - native_rate
    validation["resolved_product_rule_closure_rms"] = rms(closure)

    rates: RateFields = {
        "native_free_space_direct": direct_rate,
        "native_periodic": native_rate,
        "self_euler": euler_rate,
        "self_longitudinal": longitudinal_rate,
        "self_finite_core": core_rate,
        "native_minus_self": native_extra,
        "closure": closure,
    }
    regions = {}
    asymmetry = reflection_asymmetry(points, fields["accepted_velocity"])
    for name, mask in {
        "authority_ramp": points[:, 0] < 1.25,
        "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
        "outer_wake": points[:, 0] > 1.62,
    }.items():
        regions[name] = {}
        for label, rate_value in rates.items():
            rate = cast(FloatArray, rate_value)
            odd = reflection_asymmetry(points, rate)
            regions[name][label] = {
                "velocity_rate_rms": rms(rate[mask]),
                "reflection_energy_derivative": float(
                    2 * np.mean(np.sum(asymmetry[mask] * odd[mask], axis=1))
                ),
            }
    record = {
        "quadrature_spacing_m": spacing,
        "periodic_lengths_m": lengths,
        "grid_shape": grid.shape,
        "validation": validation,
        "regions": regions,
        "coefficient_projection": potential_record,
        "wall_seconds_not_a_benchmark": time.perf_counter() - started,
    }
    return record, rates


def main() -> None:
    """Read one t6 state and write independently refined instantaneous budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    field_path = args.audit / "fields_t6.npz"
    particle_path = args.audit / "particles_accepted_t6.npz"
    with np.load(field_path) as data:
        fields = {key: data[key].astype(np.float64) for key in data.files}
    with np.load(particle_path) as data:
        radius = data["core_radius"]
        np.testing.assert_array_equal(data["position"], fields["particle_position"])
    np.testing.assert_array_equal(radius, np.full(len(radius), radius[0]))
    set_num_threads(2)
    p, g = fields["particle_position"], fields["particle_strength"]
    direct_rate = direct_velocity_rate(
        fields["points"], p, g, radius, fields["stage_velocity"], fields["stage_strength_rate"]
    )
    selected_self = direct_gaussian(p[fields["selected"].astype(int)], p, g, radius)
    report = {
        "scope": __doc__,
        "time_s": 6.0,
        "particle_count": len(p),
        "core_radius_m": float(radius[0]),
        "native_stage_array_precision": "saved float32 values, accumulated here in float64",
        "input_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (field_path, particle_path)
        },
        "cases": [],
    }
    for label, spacing, lengths in (
        ("base", 0.03, (7.2, 3.6, 3.6)),
        ("larger_domain", 0.03, (9.6, 4.8, 4.8)),
        ("finer_quadrature", 0.02, (7.2, 3.6, 3.6)),
        ("largest_domain", 0.03, (10.8, 5.4, 5.4)),
    ):
        with threadpool_limits(limits=2):
            record, rates = _one_case(
                fields,
                float(radius[0]),
                spacing,
                lengths,
                direct_rate,
                selected_self,
                coefficient_potential_path=(
                    args.output / "coefficient_longitudinal_potential.npz"
                    if label == "base"
                    else None
                ),
            )
        record["label"] = label
        report["cases"].append(record)
        np.savez_compressed(args.output / f"{label}.npz", points=fields["points"], **rates)
        (args.output / "budget.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
