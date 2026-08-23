import numpy as np
import pytest

from source.solvers.fvm.fields.diagnostics import compute_surface_face_loads, compute_surface_forces
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry


class TestSurfaceForces:
    """Force computation on hand_built_3d_mesh with uniform p and viscous profiles."""

    @pytest.fixture(autouse=True)
    def setup(self, hand_built_3d_mesh):
        self.mesh = hand_built_3d_mesh
        self.geo = compute_mesh_geometry(hand_built_3d_mesh)
        self.density = 1.0
        self.dynamic_viscosity = 0.01
        self.reference_velocity = 1.0
        self.reference_area = 1.0

    def _add_bc_types(self):
        for b in self.mesh["boundary"]:
            b["boundary_condition_type"] = "zeroGradient"
            b["velocity_type"] = "zeroGradient"

    def _build_full_field(self, interior, n_components=1):
        n_elem = self.mesh["n_cells"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        if n_components == 1:
            field_values = np.zeros(n_elem + n_bnd)
            field_values[:n_elem] = interior
            return field_values
        else:
            field_values = np.zeros((n_elem + n_bnd, n_components))
            field_values[:n_elem] = interior
            return field_values

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
        n_total = self.mesh["n_cells"] + (self.mesh["n_faces"] - self.mesh["n_interior_faces"])
        p = np.ones(n_total)
        velocity = self._build_full_field(np.zeros((self.mesh["n_cells"], 3)), n_components=3)
        result = compute_surface_forces(
            velocity,
            p,
            self.dynamic_viscosity,
            self.density,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=self._all_patch_names(),
        )
        fp = self._sum_forces(result, "pressure_force")
        assert np.allclose(fp, 0.0, atol=1e-12), f"uniform p on closed surface: Fp = {fp}"

    def test_pressure_on_xmax_face(self):
        """p=2 on +x face only → Fp = (+2, 0, 0) for that patch."""
        self._add_bc_types()
        n_elem = self.mesh["n_cells"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        p = np.ones(n_elem + n_bnd) * 1.0
        velocity = np.zeros((n_elem + n_bnd, 3))

        result = compute_surface_forces(
            velocity,
            p,
            self.dynamic_viscosity,
            self.density,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=self._all_patch_names(),
        )
        # xmax has 4 faces, each Sf = (+1, 0, 0), p=1 → Fp = +p·∑Sf = (+4, 0, 0)
        assert "xmax" in result
        assert np.allclose(result["xmax"]["pressure_force"], [4.0, 0.0, 0.0], atol=1e-12), (
            f"xmax pressure_force = {result['xmax']['pressure_force']}"
        )

    def test_pressure_uses_boundary_face_value_and_kinematic_density(self):
        """Kinematic pressure is sampled at the face and multiplied by density."""
        self._add_bc_types()
        n_elem = self.mesh["n_cells"]
        n_int = self.mesh["n_interior_faces"]
        n_bnd = self.mesh["n_faces"] - n_int
        p = np.ones(n_elem + n_bnd)
        velocity = np.zeros((n_elem + n_bnd, 3))
        xmax = next(b for b in self.mesh["boundary"] if b["name"] == "xmax")
        b_start = n_elem + xmax["start_face"] - n_int
        p[b_start : b_start + xmax["n_faces"]] = 2.0

        result = compute_surface_forces(
            velocity,
            p,
            0.0,
            3.0,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=["xmax"],
        )

        # Four unit-area faces: density * p_face * area = 3 * 2 * 4.
        np.testing.assert_allclose(result["xmax"]["pressure_force"], [24.0, 0.0, 0.0], atol=1e-12)

    def test_wall_shear_matches_boundary_diffusion_flux(self):
        """Wall traction uses the prescribed face-normal velocity gradient."""
        self._add_bc_types()
        n_elem = self.mesh["n_cells"]
        n_int = self.mesh["n_interior_faces"]
        n_bnd = self.mesh["n_faces"] - n_int
        p = np.zeros(n_elem + n_bnd)
        velocity = np.zeros((n_elem + n_bnd, 3))
        xmax = next(b for b in self.mesh["boundary"] if b["name"] == "xmax")
        b_start = n_elem + xmax["start_face"] - n_int
        velocity[b_start : b_start + xmax["n_faces"], 1] = 1.0

        result = compute_surface_forces(
            velocity,
            p,
            self.dynamic_viscosity,
            self.density,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=["xmax"],
        )

        faces = np.arange(xmax["start_face"], xmax["start_face"] + xmax["n_faces"])
        expected_shear = -self.dynamic_viscosity * np.sum(
            self.geo["face_area"][faces] / self.geo["wall_distance"][faces]
        )
        np.testing.assert_allclose(
            result["xmax"]["viscous_force"], [0.0, expected_shear, 0.0], atol=1e-12
        )

    def test_total_force_is_pressure_plus_viscous(self):
        """Ftot = Fp + Fv (vector sum holds for any field)."""
        self._add_bc_types()
        np.random.seed(42)
        n_elem = self.mesh["n_cells"]
        p = self._build_full_field(np.random.randn(n_elem))
        velocity = self._build_full_field(np.random.randn(n_elem, 3), n_components=3)

        result = compute_surface_forces(
            velocity,
            p,
            self.dynamic_viscosity,
            self.density,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=self._all_patch_names(),
        )
        ftot = self._sum_forces(result, "total_force")
        fp = self._sum_forces(result, "pressure_force")
        fv = self._sum_forces(result, "viscous_force")
        assert np.allclose(ftot, fp + fv, atol=1e-12), "Ftot != Fp + Fv"

    def test_face_load_api_integrates_to_surface_force(self):
        """The public face data used by validation matches the force total."""
        self._add_bc_types()
        n_elem = self.mesh["n_cells"]
        p = self._build_full_field(np.linspace(-1.0, 1.0, n_elem))
        velocity = self._build_full_field(np.ones((n_elem, 3)), n_components=3)
        names = self._all_patch_names()
        faces = compute_surface_face_loads(
            velocity,
            p,
            self.dynamic_viscosity,
            self.density,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            names,
        )
        total = compute_surface_forces(
            velocity,
            p,
            self.dynamic_viscosity,
            self.density,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            names,
        )
        for name in names:
            np.testing.assert_allclose(
                faces[name]["pressure_force"].sum(axis=0), total[name]["pressure_force"]
            )
            np.testing.assert_allclose(
                faces[name]["viscous_force"].sum(axis=0), total[name]["viscous_force"]
            )

    def test_net_pressure_coefficient(self):
        """Net force over all patches with uniform p=1 is zero → net Cd=0."""
        self._add_bc_types()
        p = self._build_full_field(np.ones(self.mesh["n_cells"]))
        velocity = self._build_full_field(np.zeros((self.mesh["n_cells"], 3)), n_components=3)
        result = compute_surface_forces(
            velocity,
            p,
            self.dynamic_viscosity,
            self.density,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=self._all_patch_names(),
            reference_velocity=self.reference_velocity,
            reference_area=self.reference_area,
        )
        fp = self._sum_forces(result, "pressure_force")
        fv = self._sum_forces(result, "viscous_force")
        assert np.allclose(fp, 0.0, atol=1e-12), f"net Fp = {fp}"
        assert np.allclose(fv, 0.0, atol=1e-12), f"net Fv = {fv}"

    def test_moment_coefficient_uses_face_moment(self):
        self._add_bc_types()
        n_elem = self.mesh["n_cells"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        velocity = np.zeros((n_elem + n_bnd, 3))
        p = np.ones(n_elem + n_bnd)
        centre = np.array([0.0, 0.25, 0.0])

        result = compute_surface_forces(
            velocity,
            p,
            0.0,
            self.density,
            self.mesh,
            self.geo,
            self.mesh["boundary"],
            patch_names=["xmax"],
            reference_velocity=2.0,
            reference_area=3.0,
            reference_length=4.0,
            moment_centre=centre,
        )["xmax"]

        boundary = next(item for item in self.mesh["boundary"] if item["name"] == "xmax")
        faces = np.arange(boundary["start_face"], boundary["start_face"] + boundary["n_faces"])
        expected_moment = np.sum(
            np.cross(
                self.geo["face_centre"][faces] - centre,
                self.geo["face_area_vector"][faces],
            ),
            axis=0,
        )
        np.testing.assert_allclose(result["moment"], expected_moment)
        denominator = 0.5 * self.density * 2.0**2 * 3.0 * 4.0
        assert result["coeffs"]["pitching_moment_coefficient"] == pytest.approx(
            expected_moment[2] / denominator
        )
