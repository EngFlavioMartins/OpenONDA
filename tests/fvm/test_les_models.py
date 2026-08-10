"""Verification of the LES subgrid models via their defining physics.

Linear velocity fields make the cell gradient exact (with exact boundary
ghosts), so the eddy viscosity is an analytic check rather than a smoke test:

* **uniform flow** (∇u = 0): every model gives ν_t = 0.
* **pure shear** u = (y, 0, 0): constant Smagorinsky spuriously activates,
  whereas WALE and sigma both vanish (correct laminar behaviour).
* **solid-body rotation** u = (−y, x, 0): Smagorinsky vanishes (S = 0) and
  sigma vanishes (a 2D flow), but WALE does not — confirming sigma's stronger
  "zero ν_t for any 2D flow" property.
* **dynamic Smagorinsky**: finite, non-negative, and zero on uniform flow.
"""

import numpy as np
import pytest

from source.solvers.FVM.config.types import TurbulenceConfig
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.turbulence import (
    WALE,
    DynamicSmagorinsky,
    EquilibriumSmagorinsky,
    Sigma,
    Smagorinsky,
    create_model,
)
from source.solvers.FVM.turbulence.les_models import _wale_operator

from ._structured_mesh import structured_box


def _field_on_mesh(mesh, geo, fn):
    """Velocity array (interior + boundary ghosts) sampled from analytic fn(x,y,z)."""
    for patch in mesh["boundary"]:
        patch["bc_type_U"] = "fixedValue"
    n_elem = mesh["n_elements"]
    n_int = mesh["n_interior_faces"]
    cc, fc = geo["element_centroids"], geo["face_centroids"]
    U = np.zeros((n_elem + mesh["n_faces"] - n_int, 3))
    U[:n_elem] = fn(cc[:, 0], cc[:, 1], cc[:, 2])
    for b in mesh["boundary"]:
        for j in range(b["nFaces"]):
            fi = b["startFace"] + j
            gi = n_elem + (fi - n_int)
            U[gi] = fn(np.array([fc[fi, 0]]), np.array([fc[fi, 1]]), np.array([fc[fi, 2]])).ravel()
    return U


def _uniform(x, y, z):
    return np.column_stack([np.full_like(x, 2.0), np.full_like(x, -1.0), np.full_like(x, 0.5)])


def _pure_shear(x, y, z):
    return np.column_stack([y, np.zeros_like(x), np.zeros_like(x)])


def _solid_rotation(x, y, z):
    return np.column_stack([-y, x, np.zeros_like(x)])


def _diagonal_strain(x, y, z):
    return np.column_stack([2.0 * x, -0.5 * y, 0.25 * z])


class TestLESModels:
    def setup_method(self):
        self.mesh = structured_box(8, 8, 8)
        self.geo = compute_mesh_geometry(self.mesh)
        self.models = {
            "smag": Smagorinsky(self.mesh, self.geo, Cs=0.17),
            "equilibrium_smag": EquilibriumSmagorinsky(self.mesh, self.geo),
            "wale": WALE(self.mesh, self.geo),
            "sigma": Sigma(self.mesh, self.geo),
            "dyn": DynamicSmagorinsky(self.mesh, self.geo),
        }

    def _nut(self, key, fn):
        U = _field_on_mesh(self.mesh, self.geo, fn)
        return self.models[key].compute_nut(U, self.mesh, self.geo)

    def test_uniform_flow_zero_nut(self):
        for key in self.models:
            nut = self._nut(key, _uniform)
            assert np.max(np.abs(nut)) < 1e-12, f"{key} nut nonzero for uniform flow"

    def test_pure_shear_wale_and_sigma_vanish(self):
        smag = self._nut("smag", _pure_shear)
        wale = self._nut("wale", _pure_shear)
        sigma = self._nut("sigma", _pure_shear)
        assert np.max(smag) > 1e-5, "Smagorinsky should activate in pure shear"
        assert np.max(wale) < 1e-12, "WALE must vanish in pure shear"
        assert np.max(sigma) < 1e-12, "sigma must vanish in pure shear"

    def test_equilibrium_smagorinsky_matches_incompressible_reduction(self):
        model = self.models["equilibrium_smag"]
        nut = self._nut("equilibrium_smag", _pure_shear)
        delta = self.geo["element_volumes"] ** (1.0 / 3.0)
        expected = model.equivalent_Cs**2 * delta**2
        np.testing.assert_allclose(nut, expected, rtol=1e-12, atol=1e-14)

    def test_equilibrium_smagorinsky_uses_full_algebraic_energy_equation(self):
        model = self.models["equilibrium_smag"]
        velocity = _field_on_mesh(self.mesh, self.geo, _diagonal_strain)
        k_sgs = model.compute_sgs_kinetic_energy(velocity)

        diagonal = np.array([2.0, -0.5, 0.25])
        trace = float(np.sum(diagonal))
        dev = diagonal - trace / 3.0
        contraction = float(np.dot(dev, diagonal))
        delta = self.geo["element_volumes"] ** (1.0 / 3.0)
        a = model.Ce / delta
        b = (2.0 / 3.0) * trace
        c = 2.0 * model.Ck * delta * contraction
        expected = np.square((-b + np.sqrt(b * b + 4.0 * a * c)) / (2.0 * a))
        np.testing.assert_allclose(k_sgs, expected, rtol=1e-12, atol=1e-14)

    def test_solid_rotation_sigma_vanishes_wale_does_not(self):
        smag = self._nut("smag", _solid_rotation)
        wale = self._nut("wale", _solid_rotation)
        sigma = self._nut("sigma", _solid_rotation)
        # Smagorinsky: S = 0 for antisymmetric ∇u ⇒ no ν_t.
        assert np.max(smag) < 1e-12, "Smagorinsky should be zero for solid rotation (S=0)"
        # sigma vanishes for any 2D flow (incl. solid rotation).
        assert np.max(sigma) < 1e-12, "sigma must vanish for solid rotation"
        # WALE does NOT vanish for rotation — the discriminator vs sigma.
        assert np.max(wale) > 1e-6, "WALE should be nonzero for solid rotation"

    def test_dynamic_smagorinsky_finite_nonneg(self):
        nut = self._nut("dyn", _pure_shear)
        assert np.all(np.isfinite(nut)) and np.all(nut >= 0.0)
        assert np.max(np.abs(self._nut("dyn", _uniform))) < 1e-12

    @pytest.mark.parametrize("model", ["smag", "equilibrium_smag", "wale", "sigma", "dyn"])
    def test_nonfinite_velocity_is_not_silently_deactivated(self, model):
        velocity = _field_on_mesh(self.mesh, self.geo, _uniform)
        velocity[0, 0] = np.nan
        with pytest.raises(FloatingPointError, match="non-finite|invalid"):
            self.models[model].compute_nut(velocity, self.mesh, self.geo)

    @pytest.mark.parametrize(
        ("config", "attribute"),
        [(TurbulenceConfig.wale(0.0), "Cw"), (TurbulenceConfig.sigma(0.0), "Csigma")],
    )
    def test_explicit_zero_coefficient_is_preserved(self, config, attribute):
        model = create_model(config, self.mesh, self.geo)
        assert getattr(model, attribute) == 0.0


def test_wale_operator_has_cubic_near_wall_scaling():
    distance = np.logspace(-5, -2, 16)
    gradient = np.zeros((len(distance), 3, 3))
    gradient[:, 0, 1] = 1.0
    gradient[:, 1, 0] = distance
    operator = _wale_operator(gradient)
    slope = np.polyfit(np.log(distance), np.log(operator), 1)[0]
    assert slope == pytest.approx(3.0, abs=0.02)
