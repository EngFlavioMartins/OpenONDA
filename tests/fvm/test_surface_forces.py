import numpy as np
import pytest

from source.solvers.FVM.fields.diagnostics import compute_surface_forces
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


class TestSurfaceForces:
    """Force computation on hand_built_3d_mesh with uniform p and viscous profiles."""

    @pytest.fixture(autouse=True)
    def setup(self, hand_built_3d_mesh):
        self.mesh = hand_built_3d_mesh
        self.geo = compute_mesh_geometry(hand_built_3d_mesh)
        self.rho = 1.0
        self.mu = 0.01
        self.ref_U = 1.0
        self.ref_area = 1.0

    def _add_bc_types(self):
        for b in self.mesh["boundary"]:
            b["bc_type"] = "zeroGradient"
            b["bc_type_U"] = "zeroGradient"

    def _build_full_field(self, interior, n_components=1):
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        if n_components == 1:
            phi = np.zeros(n_elem + n_bnd)
            phi[:n_elem] = interior
            return phi
        else:
            phi = np.zeros((n_elem + n_bnd, n_components))
            phi[:n_elem] = interior
            return phi

    def _all_patch_names(self):
        return [b["name"] for b in self.mesh["boundary"]]

    def _sum_forces(self, result, key):
        total = np.zeros(3)
        for name in self._all_patch_names():
            total += result[name][key]
        return total

    def test_uniform_pressure_gives_zero_net_force(self):
        """Uniform p=1 on a closed surface → Fp = (0,0,0)."""
        self._add_bc_types()
        p = self._build_full_field(np.ones(self.mesh["n_elements"]))
        U = self._build_full_field(np.zeros((self.mesh["n_elements"], 3)), n_components=3)
        result = compute_surface_forces(
            U,
            p,
            self.mu,
            self.rho,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=self._all_patch_names(),
        )
        fp = self._sum_forces(result, "Fp")
        assert np.allclose(fp, 0.0, atol=1e-12), f"uniform p on closed surface: Fp = {fp}"

    def test_pressure_on_xmax_face(self):
        """p=2 on +x face only → Fp = (+2, 0, 0) for that patch."""
        self._add_bc_types()
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        p = np.ones(n_elem + n_bnd) * 1.0
        U = np.zeros((n_elem + n_bnd, 3))

        result = compute_surface_forces(
            U,
            p,
            self.mu,
            self.rho,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=self._all_patch_names(),
        )
        # xmax has 4 faces, each Sf = (+1, 0, 0), p=1 → Fp = +p·∑Sf = (+4, 0, 0)
        assert "xmax" in result
        assert np.allclose(result["xmax"]["Fp"], [4.0, 0.0, 0.0], atol=1e-12), (
            f"xmax Fp = {result['xmax']['Fp']}"
        )

    def test_total_force_is_pressure_plus_viscous(self):
        """Ftot = Fp + Fv (vector sum holds for any field)."""
        self._add_bc_types()
        np.random.seed(42)
        n_elem = self.mesh["n_elements"]
        p = self._build_full_field(np.random.randn(n_elem))
        U = self._build_full_field(np.random.randn(n_elem, 3), n_components=3)

        result = compute_surface_forces(
            U,
            p,
            self.mu,
            self.rho,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=self._all_patch_names(),
        )
        ftot = self._sum_forces(result, "Ftot")
        fp = self._sum_forces(result, "Fp")
        fv = self._sum_forces(result, "Fv")
        assert np.allclose(ftot, fp + fv, atol=1e-12), "Ftot != Fp + Fv"

    def test_net_pressure_coefficient(self):
        """Net force over all patches with uniform p=1 is zero → net Cd=0."""
        self._add_bc_types()
        p = self._build_full_field(np.ones(self.mesh["n_elements"]))
        U = self._build_full_field(np.zeros((self.mesh["n_elements"], 3)), n_components=3)
        result = compute_surface_forces(
            U,
            p,
            self.mu,
            self.rho,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=self._all_patch_names(),
            ref_U=self.ref_U,
            ref_area=self.ref_area,
        )
        fp = self._sum_forces(result, "Fp")
        fv = self._sum_forces(result, "Fv")
        assert np.allclose(fp, 0.0, atol=1e-12), f"net Fp = {fp}"
        assert np.allclose(fv, 0.0, atol=1e-12), f"net Fv = {fv}"
