"""Slip-slab images of fields deposited on a Cartesian lattice.

Free-slip planes at z_min and z_max reflect vorticity as an axial vector: an
image across a plane keeps the spanwise component and reverses the in-plane
ones, and two reflections translate by twice the slab width. A nodal scalar
such as the eddy viscosity is even under every image. Grid diffusion and
the vorticity transfer both need the field of every image, for slabs of any width
down to a single lattice layer.
"""

import numpy as np


def slab_image_lattice(
    origin_z: float, spacing: float, planes: tuple[float, float]
) -> tuple[int, int]:
    """Lower-plane node index and image period of a slab on a z lattice.

    Nodes sit at ``origin_z + k * spacing``. The planes must lie on nodes or
    half nodes, so every image maps nodes onto nodes. Returns ``(lower,
    period)`` in node units: a reflection maps node k to ``lower - k``, and
    the images repeat every ``period`` nodes.
    """
    lower = 2.0 * (float(planes[0]) - float(origin_z)) / float(spacing)
    upper = 2.0 * (float(planes[1]) - float(origin_z)) / float(spacing)
    lower_index, upper_index = round(lower), round(upper)
    if (
        max(abs(lower - lower_index), abs(upper - upper_index)) > 1.0e-4
        or upper_index <= lower_index
    ):
        raise ValueError("slip planes must be increasing and lie on lattice nodes or half nodes")
    return int(lower_index), int(upper_index - lower_index)


def _image_maps(
    nz: int, origin_z: float, spacing: float, planes: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Direct and reflected image maps between the ``nz`` nodes of a z lattice.

    Entry (k, m) of the direct map is one when a translation by whole periods
    maps node m onto node k, and of the reflected map when an odd number of
    reflections does: ``k - m`` or ``k + m - lower`` is a multiple of the period.
    """
    lower, period = slab_image_lattice(origin_z, spacing, planes)
    node = np.arange(nz)
    direct = ((node[:, None] - node[None, :]) % period == 0).astype(np.float64)
    reflected = ((node[:, None] + node[None, :] - lower) % period == 0).astype(np.float64)
    return direct, reflected


def add_slab_images(
    deposit: np.ndarray, origin_z: float, spacing: float, planes: tuple[float, float]
) -> np.ndarray:
    """Return the field of all slab images of a vorticity ``deposit``, shape (..., nz, 3).

    Node k receives the deposit of node m once for each image that maps m onto
    k: directly when k - m is a multiple of the period, and reflected when
    k + m - lower is. A node on a plane is its own reflection, so its spanwise
    component doubles and its in-plane components cancel. The lattice must
    hold the complete support of every physical deposit.
    """
    direct, reflected = _image_maps(deposit.shape[-2], origin_z, spacing, planes)
    images = np.empty(deposit.shape, dtype=np.float64)
    images[..., :2] = np.einsum("km,...mc->...kc", direct - reflected, deposit[..., :2])
    images[..., 2] = np.einsum("km,...m->...k", direct + reflected, deposit[..., 2])
    return images


def physical_slab_share(
    nz: int, origin_z: float, spacing: float, planes: tuple[float, float]
) -> np.ndarray:
    """Share of each z node that belongs to the physical slab, shape (nz,).

    One between the planes, zero beyond them, and one half on a node-aligned
    plane, whose reflection maps the node onto itself and so counts it twice.
    """
    slab_image_lattice(origin_z, spacing, planes)
    z = float(origin_z) + float(spacing) * np.arange(nz)
    tolerance = 1.0e-4 * float(spacing)
    share = ((z > planes[0] - tolerance) & (z < planes[1] + tolerance)).astype(np.float64)
    share[np.abs(z - planes[0]) < tolerance] = 0.5
    share[np.abs(z - planes[1]) < tolerance] = 0.5
    return share


def even_slab_field(
    values: np.ndarray, origin_z: float, spacing: float, planes: tuple[float, float]
) -> np.ndarray:
    """Extend a span-even nodal scalar from the physical slab to every node.

    ``values`` has shape (..., nz); only nodes between the planes are read.
    Every other node takes the value of the physical node that one of the
    slab images maps onto it, so the result is even under every reflection.
    """
    direct, reflected = _image_maps(values.shape[-1], origin_z, spacing, planes)
    physical = values * physical_slab_share(values.shape[-1], origin_z, spacing, planes)
    return np.einsum("km,...m->...k", direct + reflected, physical)


# An image across a slip plane keeps the spanwise strength and reverses the
# in-plane components; vorticity is an axial vector.
AXIAL_PARITY = np.array([-1.0, -1.0, 1.0])


def fold_into_slab(
    position: np.ndarray, vortex_strength: np.ndarray, planes: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Replace every particle outside the slab by its image inside the slab.

    A particle carried across a slip plane, by a random walk or a split, is
    the same field as its image entering from the other side. The image lies
    an odd or even number of reflections away; an odd one reverses the
    in-plane strength. Particles inside the slab are returned unchanged.
    Returns float64 copies of ``position`` and ``vortex_strength``, shape (N, 3).
    """
    z_min, z_max = float(planes[0]), float(planes[1])
    width = z_max - z_min
    position = np.array(position, dtype=np.float64, copy=True).reshape(-1, 3)
    vortex_strength = np.array(vortex_strength, dtype=np.float64, copy=True).reshape(-1, 3)
    outside = np.flatnonzero((position[:, 2] < z_min) | (position[:, 2] > z_max))
    phase = np.mod(position[outside, 2] - z_min, 2.0 * width)
    odd = phase > width
    position[outside, 2] = np.where(odd, z_min + 2.0 * width - phase, z_min + phase)
    vortex_strength[outside[odd]] *= AXIAL_PARITY
    return position, vortex_strength


def slab_period_images(
    position: np.ndarray, vortex_strength: np.ndarray, planes: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Particles and their reflections across the lower plane: one image period.

    The reflections lie in [z_min - width, z_min], which a z-periodic lattice of
    period 2 * width identifies with [z_max, z_max + width]. A field deposited
    from both on that lattice is the field of every slab image. Returns
    arrays of shape (2N, 3), physical particles first.
    """
    position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3)
    reflected = position.copy()
    reflected[:, 2] = 2.0 * float(planes[0]) - reflected[:, 2]
    return (
        np.concatenate((position, reflected)),
        np.concatenate((vortex_strength, vortex_strength * AXIAL_PARITY)),
    )


def slab_period_lattice(spacing: float, planes: tuple[float, float]) -> tuple[float, int]:
    """Origin and node count of a z lattice spanning one image period.

    Nodes sit at ``z_min + (k + 1/2) * spacing`` for k = 0 .. count - 1, so the
    lower plane lies on a half node and a slab one spacing wide holds a
    single layer of cell centres, as on the grid-diffusion lattice. The slab
    width must be a multiple of half the spacing.
    """
    count = 2.0 * (float(planes[1]) - float(planes[0])) / float(spacing)
    if round(count) < 1 or abs(count - round(count)) > 1.0e-6 * max(1.0, count):
        raise ValueError(
            f"slip-slab width {float(planes[1]) - float(planes[0]):.6g} m must be a multiple "
            f"of half the lattice spacing {float(spacing):.6g} m"
        )
    return float(planes[0]) + 0.5 * float(spacing), int(round(count))


def slab_image_particles(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    planes: tuple[float, float],
    reach: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every image of the particles whose centre lies within ``reach`` of the slab.

    The particles themselves are not included. Returns the image positions,
    strengths, and the index of the particle each image belongs to, so that
    per-particle properties such as core radii can follow.
    """
    position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3)
    z_min, z_max = float(planes[0]), float(planes[1])
    period = 2.0 * (z_max - z_min)
    shells = int(np.ceil((float(reach) + period) / period))
    reflected = 2.0 * z_min - position[:, 2]
    positions, strengths, owners = [], [], []
    for shift in range(-shells, shells + 1):
        for odd, base in ((False, position[:, 2]), (True, reflected)):
            if shift == 0 and not odd:
                continue
            z = base + shift * period
            near = np.flatnonzero((z >= z_min - reach) & (z <= z_max + reach))
            if not len(near):
                continue
            image = position[near].copy()
            image[:, 2] = z[near]
            positions.append(image)
            strengths.append(vortex_strength[near] * (AXIAL_PARITY if odd else 1.0))
            owners.append(near)
    if not positions:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=np.int64)
    return np.concatenate(positions), np.concatenate(strengths), np.concatenate(owners)
