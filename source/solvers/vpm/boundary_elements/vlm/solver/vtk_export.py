"""One surface representation for live VLM samples and saved-state visualization."""

from pathlib import Path

import numpy as np

from source.vtk_output import write_vtk_dataset

# Optional fields are omitted when visualizing an older numerical checkpoint.
# In particular, absent unsteady loads must never be presented as measured zeros.
CELL_FIELDS = (
    "area",
    "normal",
    "is_trailing_edge",
    "is_leading_edge",
    "wing_id",
    "segment_id",
    "is_mirrored",
    "circulation",
    "cumulative_circulation",
    "bound_vortex_velocity",
    "bound_external_velocity",
    "bound_relative_velocity",
    "velocity",
    "kinematic_velocity",
    "relative_velocity",
    "panel_force",
    "unsteady_panel_force",
    "panel_moment_correction",
    "unsteady_pressure_jump_coefficient",
)


def write_lattice_vtk(fields, filename, *, reference_speed, time, force_density=None):
    """Write saved or live arrays without advancing motion or recomputing loads.

    Panel corners retain their stored precision. ``velocity`` and
    ``bound_vortex_velocity`` are inertial fluid velocities; the corresponding
    ``relative_velocity`` fields subtract the prescribed body motion. Circulation
    is in [m^2/s], and ``panel_force`` is the force in newtons at the solve
    density when ``force_density`` is supplied. The legacy
    ``pressure_jump_coefficient`` cell array is retained solely for reader
    compatibility: it is a circulation-based lifting-surface proxy, not a
    pressure reconstruction, and it is not in ``CELL_FIELDS`` or selected as
    the active scalar. New readers should use the explicit
    ``circulation_pressure_jump_proxy`` or ``panel_normal_load_coefficient``
    fields. All fields can be selected with ParaView's Color By control.
    """
    from vtk import vtkCellArray, vtkPoints, vtkPolyData
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    corners = np.asarray(fields["panel_corner_position"])
    if corners.ndim != 3 or corners.shape[1:] != (4, 3) or not np.isfinite(corners).all():
        raise ValueError("VLM surface corners must be finite with shape (panels, 4, 3)")
    if not np.isfinite(time) or not np.isfinite(reference_speed):
        raise ValueError("VLM surface time and reference speed must be finite")
    count = len(corners)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(corners.reshape(-1, 3)), deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(count + 1, dtype=np.int64) * 4, deep=True),
        numpy_to_vtkIdTypeArray(np.arange(count * 4, dtype=np.int64), deep=True),
    )
    surface = vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(cells)

    def add(data, name, values):
        array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        array.SetName(name)
        data.AddArray(array)

    computed_area = None
    if "area" not in fields:
        # Area is geometry-derived, so a legacy checkpoint can still be
        # re-exported deterministically.  Keep the validated local array for
        # dimensional normal-load coefficients instead of only adding it to
        # VTK cell data below.
        computed_area = 0.5 * np.linalg.norm(
            np.cross(
                corners[:, 2] - corners[:, 0],
                corners[:, 1] - corners[:, 3],
            ),
            axis=1,
        )
        if not np.isfinite(computed_area).all():
            raise ValueError("Invalid geometry-derived VLM panel area")

    cell_data = surface.GetCellData()
    for name in CELL_FIELDS:
        if name in fields:
            values = np.asarray(fields[name])
            if values.shape not in ((count,), (count, 3)) or not np.isfinite(values).all():
                raise ValueError(f"Invalid saved VLM field {name}")
            add(cell_data, name, values)
    add(cell_data, "panel_id", np.arange(count, dtype=np.int32))
    add(cell_data, "panel_centre", corners.mean(axis=1))
    chord = np.linalg.norm(
        0.5 * (corners[:, 3] + corners[:, 2] - corners[:, 0] - corners[:, 1]),
        axis=1,
    )
    add(cell_data, "panel_chord", chord)
    if computed_area is not None:
        add(cell_data, "area", computed_area)
    if "vortex_point_position" in fields:
        vortex = np.asarray(fields["vortex_point_position"])
        add(cell_data, "bound_vortex_leg", vortex[:, 2] - vortex[:, 1])
    if "circulation" in fields:
        denominator = reference_speed * chord
        pressure = np.divide(
            2.0 * fields["circulation"],
            denominator,
            out=np.zeros(count, dtype=corners.dtype),
            where=denominator > 1e-15,
        )
        if "unsteady_pressure_jump_coefficient" in fields:
            pressure += fields["unsteady_pressure_jump_coefficient"]
        add(cell_data, "pressure_jump_coefficient", pressure)
        add(cell_data, "circulation_pressure_jump_proxy", pressure)
        cell_data.SetActiveScalars("circulation")
    if "pressure_coefficient" in fields:
        speed_pressure_coefficient = np.asarray(fields["pressure_coefficient"])
        if (
            speed_pressure_coefficient.shape != (count,)
            or not np.isfinite(speed_pressure_coefficient).all()
        ):
            raise ValueError("Invalid saved VLM pressure coefficient")
        add(cell_data, "speed_pressure_coefficient", speed_pressure_coefficient)
    if force_density is not None:
        force_density = float(force_density)
        if not np.isfinite(force_density) or force_density <= 0.0:
            raise ValueError("VLM force density must be finite and positive")
        area = np.asarray(fields["area"]) if "area" in fields else computed_area
        if {"panel_force", "normal"} <= set(fields) and area is not None:
            denominator = 0.5 * force_density * reference_speed**2 * area
            normal_load = np.divide(
                np.sum(np.asarray(fields["panel_force"]) * np.asarray(fields["normal"]), axis=1),
                denominator,
                out=np.zeros(count, dtype=corners.dtype),
                where=denominator > 1e-15,
            )
            add(cell_data, "panel_normal_load_coefficient", normal_load)
    # VTK requires the exact metadata key TimeValue to recover physical time
    # from standalone VTP file series.  It is not a model field; the native
    # API and scientific arrays keep their snake_case names.
    for name in ("time", "TimeValue"):
        add(surface.GetFieldData(), name, np.array([time], dtype=np.float64))
    from vtk import vtkStringArray

    metadata = {
        "velocity_frame": "inertial fluid velocity at collocation points [m/s]",
        "relative_velocity_definition": "relative_velocity = velocity - kinematic_velocity [m/s]",
        "bound_velocity_frame": "inertial fluid velocity at bound-vortex midpoints [m/s]",
        "bound_relative_velocity_definition": (
            "bound_relative_velocity = bound_vortex_velocity - bound_kinematic_velocity [m/s]"
        ),
        "cell_association": "all VLM fields are cell-centred; panel_id is lattice order",
        "pressure_jump_coefficient_definition": (
            "legacy circulation proxy 2*circulation/(reference_speed*panel_chord) "
            "+ unsteady contribution when present"
        ),
        "pressure_jump_coefficient_status": (
            "compatibility-only cell-data alias; not in CELL_FIELDS and not the default scalar"
        ),
        "circulation_pressure_jump_proxy_definition": (
            "same legacy circulation proxy under an explicit non-pressure name"
        ),
        "speed_pressure_coefficient_definition": (
            "Bernoulli speed coefficient 1-|velocity|^2/reference_speed^2 at collocation points; not a pressure jump"
        ),
    }
    if force_density is not None:
        add(surface.GetFieldData(), "force_density", np.array([force_density], dtype=np.float64))
        metadata["panel_force_definition"] = "panel_force is dimensional force [N] at force_density"
        metadata["panel_normal_load_coefficient_definition"] = (
            "dot(panel_force, normal)/(0.5*force_density*reference_speed^2*area)"
        )
    else:
        metadata["panel_force_definition"] = (
            "panel_force units are preserved from input; force density unavailable"
        )
    for name, value in metadata.items():
        array = vtkStringArray()
        array.SetName(name)
        array.InsertNextValue(value)
        surface.GetFieldData().AddArray(array)
    return write_vtk_dataset(surface, Path(filename))
