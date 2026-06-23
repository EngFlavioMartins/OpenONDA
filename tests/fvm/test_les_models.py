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

from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.turbulence import DynamicSmagorinsky, Sigma, Smagorinsky, WALE

from ._structured_mesh import structured_box


def _field_on_mesh(mesh, geo, fn):
    """Velocity array (interior + boundary ghosts) sampled from analytic fn(x,y,z)."""
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


class TestLESModels:
    def setup_method(self):
        self.mesh = structured_box(8, 8, 8)
        self.geo = compute_mesh_geometry(self.mesh)
        self.models = {
            "smag": Smagorinsky(self.mesh, self.geo, Cs=0.17),
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
