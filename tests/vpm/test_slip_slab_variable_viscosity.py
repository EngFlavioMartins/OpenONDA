"""Grid diffusion with variable eddy viscosity between free-slip slab planes."""

import numpy as np

from openonda import vpm
from openonda.vpm.config.case import Numerics
from openonda.vpm.config.turbulence import TurbulenceConfig
from openonda.vpm.config.viscous import ViscousConfig
from openonda.vpm.physics.induction.direct import DirectInduction
from openonda.vpm.physics.induction.slip_slab import SlipSlabInduction

H = 0.08
NU = 1.0 / 150.0
DT = 0.05  # two explicit stages at the largest viscosity, within the three-cell halo


def _gbd_solver(directory, z_bounds, induction, capacity):
    return vpm.VPMSolver(
        vpm.VPMCase(
            directory=directory,
            backup=vpm.Backup(interval_steps=0),
            numerics=vpm.Numerics(
                time_step_size=DT,
                compute_device="CPU",
                precision="f32",
                max_n_particles=capacity,
                domain_bounds=(-0.8, 0.8, -0.8, 0.8, *z_bounds),
                induction=induction,
                viscous=vpm.ViscousConfig.gbd(
                    particle_spacing=H, padding=3.0, kinematic_viscosity=NU
                ),
                verbose=False,
            ),
        )
    )


def _diffuse(solver, position, strength, viscosity):
    count = len(position)
    solver.add_vortex_particles(
        position, np.zeros((count, 3)), strength, np.full(count, H), np.full(count, H**3)
    )
    return solver.physics.gbd_diffusion(
        solver.particles,
        time_step_size=DT,
        particle_spacing=H,
        kinematic_viscosity=NU,
        regen_threshold=1.0e-30,
        regen_threshold_mode="absolute",
        effective_viscosity=np.asarray(viscosity, dtype=np.float32),
    )


def test_slab_accepts_les_with_grid_diffusion():
    slab = SlipSlabInduction(DirectInduction(), z_min=-0.5 * H, z_max=0.5 * H)
    Numerics(
        induction=slab,
        viscous=ViscousConfig.gbd(particle_spacing=H, kinematic_viscosity=NU),
        turbulence=TurbulenceConfig(model="LES_SMAGORINSKY"),
    )


def test_one_layer_variable_viscosity_is_the_two_dimensional_operator(tmp_path):
    planes = (-0.5 * H, 0.5 * H)
    solver = _gbd_solver(
        tmp_path,
        planes,
        SlipSlabInduction(DirectInduction(), z_min=planes[0], z_max=planes[1]),
        capacity=1024,
    )
    try:
        # Axial Gaussian vortex on the single layer; the eddy viscosity peaks
        # in the core and is uniform on the outer rings of the particle block.
        x, y = np.meshgrid(H * np.arange(-4, 5), H * np.arange(-4, 5), indexing="ij")
        radius = np.hypot(x, y).ravel()
        omega = np.exp(-((radius / (2 * H)) ** 2))
        bump = np.where(radius < 3 * H, np.cos(np.pi * radius / (6 * H)) ** 2, 0.0)
        viscosity = NU * (1.0 + bump)
        count = omega.size
        position = np.column_stack((x.ravel(), y.ravel(), np.zeros(count)))
        strength = np.zeros((count, 3))
        strength[:, 2] = omega * H**3
        result = _diffuse(solver, position, strength, viscosity)
        stages = solver.physics.last_gbd_diffusion_substeps
        assert stages == 2

        # The same explicit stages of div(nu grad omega) on the active plane,
        # with arithmetic face viscosities and zero flux at the active edge
        grid_min, shape = solver.physics._lattice_aligned_bounds(position, H, 3.0)
        index = np.rint((position[:, :2] - grid_min[:2]) / H).astype(int)
        nu = np.full(shape[:2], NU)
        nu[index[:, 0], index[:, 1]] = viscosity
        reference = np.zeros(shape[:2])
        reference[index[:, 0], index[:, 1]] = strength[:, 2]
        nu_padded = np.pad(nu, 1, mode="edge")
        for _ in range(stages):
            padded = np.pad(reference, 1, mode="edge")
            flux = np.zeros_like(reference)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rows = slice(1 + di, padded.shape[0] - 1 + di)
                cols = slice(1 + dj, padded.shape[1] - 1 + dj)
                flux += 0.5 * (nu + nu_padded[rows, cols]) * (padded[rows, cols] - reference)
            reference = reference + DT / stages * flux / H**2

        nodes = np.rint((result["position"][:, :2] - grid_min[:2]) / H).astype(int)
        np.testing.assert_allclose(result["position"][:, 2], 0.0, atol=1e-7)
        np.testing.assert_allclose(result["vortex_strength"][:, :2], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            result["vortex_strength"][:, 2],
            reference[nodes[:, 0], nodes[:, 1]],
            rtol=1e-5,
            atol=1e-6 * H**3,
        )
        assert len(result["position"]) > count
    finally:
        solver.close()


def _fold(z, z_min, z_max):
    """Physical position and reflection count of a point under the slab images."""
    reflections = 0
    while z < z_min or z > z_max:
        z = 2.0 * z_min - z if z < z_min else 2.0 * z_max - z
        reflections += 1
    return z, reflections


def test_thick_slab_variable_viscosity_matches_explicit_images(tmp_path):
    planes = (-0.2, 0.2)  # five layers between half-node planes
    layers = H * np.arange(-2, 3)
    x, y = np.meshgrid(H * np.arange(-2, 3), H * np.arange(-2, 3), indexing="ij")
    rng = np.random.default_rng(11)
    layer_strength = rng.uniform(-1.0, 1.0, size=(len(layers), x.size, 3)) * H**3
    layer_viscosity = NU * (1.0 + 0.25 * np.arange(len(layers)))

    def cloud(z_values):
        position, strength, viscosity = [], [], []
        for z in z_values:
            physical, reflections = _fold(z, *planes)
            layer = int(np.argmin(np.abs(layers - physical)))
            parity = np.array([-1.0, -1.0, 1.0]) if reflections % 2 else np.ones(3)
            position.append(np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z))))
            strength.append(layer_strength[layer] * parity)
            viscosity.append(np.full(x.size, layer_viscosity[layer]))
        return np.concatenate(position), np.concatenate(strength), np.concatenate(viscosity)

    slab_solver = _gbd_solver(
        tmp_path / "slab",
        planes,
        SlipSlabInduction(DirectInduction(), z_min=planes[0], z_max=planes[1]),
        capacity=4096,
    )
    try:
        slab = _diffuse(slab_solver, *cloud(layers))
        assert slab_solver.physics.last_gbd_diffusion_substeps == 2
    finally:
        slab_solver.close()

    # Explicit images two periods beyond each plane, diffused without a slab
    images = H * np.arange(-12, 13)
    free_solver = _gbd_solver(
        tmp_path / "free", (images[0], images[-1]), DirectInduction(), capacity=16384
    )
    try:
        free = _diffuse(free_solver, *cloud(images))
    finally:
        free_solver.close()

    lookup = {
        tuple(np.rint(p / H).astype(int)): s
        for p, s in zip(free["position"], free["vortex_strength"], strict=True)
    }
    expected = np.array([lookup[tuple(np.rint(p / H).astype(int))] for p in slab["position"]])
    assert np.all(np.abs(slab["position"][:, 2]) < planes[1])
    np.testing.assert_allclose(
        slab["vortex_strength"], expected, rtol=0.0, atol=2e-6 * np.abs(expected).max()
    )
