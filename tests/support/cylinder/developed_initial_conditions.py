"""Initialize a domain-size control from a saved developed physical field.

This is a new initial-value problem at time zero, not a restart. Native clocks
and initial-state setters are used without altering checkpoint metadata.
The old FVM field is retained inside its domain and particle velocity supplies
new downstream cells. Initial pressure outside that domain uses Bernoulli;
the subsequent pressure correction determines the evolved pressure.
"""

import json
from xml.etree import ElementTree

import h5py
import numpy as np
from scipy.spatial import cKDTree

from source.coupler.backup import checkpoint_path_hash

from .audit_saved_wall_circulation import digest, load_donor_geometry


def initialize_saved_volume(owner, directory, physical_age):
    """Start a new native initial-value problem from simultaneous saved fields.

    This uses ordinary FVM volume output rather than constructing restart
    metadata. Native BDF history starts at zero; a matched baseline qualifies
    the resulting initialization transient before any wake inference.
    """
    import pyvista as pv

    collection = directory / "solution/fvm.pvd"
    entries = ElementTree.parse(collection).getroot().findall("Collection/DataSet")
    entry = next(
        item for item in entries if abs(float(item.attrib["timestep"]) - physical_age) < 1e-8
    )
    volume_path = directory / "solution" / entry.attrib["file"]
    particle_path = directory / "solution/vpm" / f"vpm_{round(physical_age / 0.04):06d}.h5"
    mesh_path = directory / "solution/fvm/mesh.npz"
    paths = (collection, volume_path, particle_path, mesh_path)
    hashes = {str(path): digest(path) for path in paths}
    volume = pv.read(volume_path)
    flow = owner.fvm_solver
    count = flow.mesh_data["n_cells"]
    if volume.n_cells != count:
        raise ValueError("Saved-volume initial conditions require the same FVM mesh")
    np.testing.assert_allclose(
        volume.cell_data["cell_volume"], flow.geo_data["cell_volume"][:count], rtol=1e-6
    )
    np.testing.assert_allclose(volume.points, flow.mesh_data["vertex_position"], atol=1e-6, rtol=0)
    with h5py.File(particle_path) as saved:
        if abs(float(saved["solver"].attrs["time"]) - physical_age) > 1e-8:
            raise ValueError("Saved FVM and particle fields are not simultaneous")
        fields = {
            name: np.asarray(saved["particles/" + name])
            for name in (
                "position",
                "velocity",
                "vortex_strength",
                "core_radius",
                "particle_volume",
                "kinematic_viscosity",
                "group_id",
                "zone_id",
            )
        }
    owner.vpm_solver.replace_vortex_particles(**fields, report_removal=False)
    flow.set_initial_state(
        np.asarray(volume.cell_data["velocity"], dtype=float),
        np.asarray(volume.cell_data["kinematic_pressure"], dtype=float),
    )
    if any(digest(path) != hashes[str(path)] for path in paths):
        raise RuntimeError("Saved-volume physical source changed")
    return {
        "source_physical_age_s": physical_age,
        "new_solver_time_s": flow.time,
        "source_sha256": hashes,
        "maximum_source_particle_x_m": float(fields["position"][:, 0].max()),
        "interpretation": "New initial-value problem, not a native restart. FVM BDF histories reset through set_initial_state; matched baseline qualification required.",
    }


def initialize_developed_flow(owner, snapshot):
    metadata_path = snapshot / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    flow_path = metadata_path.parent / metadata["checkpoint_files"]["fvm"]
    particle_path = metadata_path.parent / metadata["checkpoint_files"]["vpm"]
    paths = (metadata_path, flow_path, particle_path, snapshot / "coupled_mesh.npz")
    hashes = {str(path): digest(path) for path in paths}
    for name, path in (("fvm", flow_path), ("vpm", particle_path)):
        if checkpoint_path_hash(path) != metadata["file_sha256"][name]:
            raise ValueError("Developed initial-condition checkpoint differs: " + name)
    mesh, geometry, state, gradient, _boundary, _wall, trace = load_donor_geometry(
        snapshot, flow_path
    )
    with h5py.File(particle_path) as saved:
        fields = {
            name: np.asarray(saved["particles/" + name])
            for name in (
                "position",
                "velocity",
                "vortex_strength",
                "core_radius",
                "particle_volume",
                "kinematic_viscosity",
                "group_id",
                "zone_id",
            )
        }
    owner.vpm_solver.replace_vortex_particles(**fields, report_removal=False)
    flow = owner.fvm_solver
    points = flow.geo_data["cell_centre"][: flow.mesh_data["n_cells"]]
    velocity = owner.vpm_solver.compute_velocity_at_points(points)
    pressure = 0.5 * (1.0 - np.sum(velocity**2, axis=1))
    vertices = mesh["vertex_position"]
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    interior = np.all((points >= lower) & (points <= upper), axis=1)
    velocity[interior] = trace.prepare(points[interior]).sample(
        state["velocity"][: mesh["n_cells"]], gradient
    )
    distance, nearest = cKDTree(geometry["cell_centre"]).query(points[interior])
    pressure[interior] = state["kinematic_pressure"][nearest]
    flow.set_initial_state(velocity, pressure)
    if any(digest(path) != hashes[str(path)] for path in paths):
        raise RuntimeError("Developed initial-condition source changed")
    return {
        "source_physical_age_s": metadata["time"],
        "new_solver_time_s": flow.time,
        "source_sha256": hashes,
        "fvm_cells_initialized_from_saved_fvm": int(interior.sum()),
        "fvm_cells_initialized_from_particle_velocity": int((~interior).sum()),
        "maximum_saved_domain_donor_distance_m": float(distance.max()),
        "pressure_initialization": "Saved FVM pressure inside old domain; Bernoulli outside",
        "interpretation": "New initial-value problem. Reset time histories cause a short initialization transient; qualification excludes it.",
    }
