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
    "velocity",
    "kinematic_velocity",
    "panel_force",
    "unsteady_panel_force",
    "panel_moment_correction",
    "unsteady_pressure_jump_coefficient",
)


def write_lattice_vtk(fields, filename, *, reference_speed, time):
    """Write saved or live arrays without advancing motion or recomputing loads.

    Panel corners retain their stored precision. Cell fields retain the solver's
    conventions, including force per density [m^4/s^2] and circulation [m^2/s].
    ``pressure_jump_coefficient`` includes the stored unsteady contribution when
    present. All fields can be selected with ParaView's Color By control.
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
    if "area" not in fields:
        # Same diagonal-cross-product definition used by the native mesh.
        area = 0.5 * np.linalg.norm(
            np.cross(
                corners[:, 2] - corners[:, 0],
                corners[:, 1] - corners[:, 3],
            ),
            axis=1,
        )
        add(cell_data, "area", area)
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
        cell_data.SetActiveScalars("circulation")
    # TimeValue is recognized by VTK readers; time preserves the native API.
    for name in ("time", "TimeValue"):
        add(surface.GetFieldData(), name, np.array([time], dtype=np.float64))
    return write_vtk_dataset(surface, Path(filename))
