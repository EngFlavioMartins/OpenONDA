"""Small filter-only ParaView qualification; no views, rendering or real outputs."""

import ast
from pathlib import Path
import tempfile

from paraview import servermanager
from paraview.simple import ExtractTimeSteps, GetAnimationScene, GetTimeKeeper, PVDReader

# Execute the actual checkout-state helper loading prelude, stopping before
# importing ParaView/creating a view. This must work without OpenONDA on sys.path.
root = Path(__file__).resolve().parents[2]
for case in ("02_cube_flow",):
    state = root / "tutorials/coupled_fvm_vpm" / case / "assets/paraview_state.py"
    tree = ast.parse(state.read_text())
    prelude = []
    for statement in tree.body:
        if isinstance(statement, ast.Import) and any(
            item.name == "paraview" for item in statement.names
        ):
            break
        prelude.append(statement)
    namespace = {"__file__": str(state)}
    exec(compile(ast.Module(body=prelude, type_ignores=[]), str(state), "exec"), namespace)
    assert namespace["match_saved_times"]([0, 4], [0, 4]).times == (0, 4)
match_saved_times = namespace["match_saved_times"]
read_pvd_times = namespace["read_pvd_times"]
same_saved_time = namespace["_saved_time_helpers"]["same_saved_time"]


def collection(directory, name, times):
    rows = []
    for index, time in enumerate(times):
        frame = directory / f"{name}_{index}.vtp"
        frame.write_text(
            '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian"><PolyData>'
            '<Piece NumberOfPoints="1" NumberOfVerts="0" NumberOfLines="0" '
            'NumberOfStrips="0" NumberOfPolys="0"><PointData>'
            f'<DataArray type="Float64" Name="saved_time" format="ascii">{time}</DataArray>'
            '</PointData><Points><DataArray type="Float64" NumberOfComponents="3" '
            'format="ascii">0 0 0</DataArray></Points></Piece></PolyData></VTKFile>'
        )
        rows.append(f'<DataSet timestep="{time}" file="{frame.name}"/>')
    path = directory / f"{name}.pvd"
    path.write_text(
        '<VTKFile type="Collection"><Collection>' + "".join(rows) + "</Collection></VTKFile>"
    )
    return path


with tempfile.TemporaryDirectory(prefix="openonda-pv-clocks-") as temporary:
    directory = Path(temporary)
    paths = (
        collection(directory, "vpm", range(9)),
        collection(directory, "fvm", [0, 4.00000000000001, 7.99999999999999]),
    )
    matched = match_saved_times(*(read_pvd_times(path) for path in paths))
    readers = [PVDReader(FileName=str(path)) for path in paths]
    selected = []
    for reader, indices in zip(readers, matched.indices, strict=True):
        proxy = ExtractTimeSteps(Input=reader)
        proxy.SelectionMode = "Select Time Steps"
        proxy.TimeStepIndices = list(indices)
        proxy.ApproximationMode = "Nearest Time Step"
        proxy.UpdatePipelineInformation()
        selected.append(proxy)
    keeper = GetTimeKeeper()
    keeper.SuppressedTimeSources = [*readers, selected[1]]
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    scene.PlayMode = "Snap To TimeSteps"
    assert tuple(keeper.TimestepValues) == matched.times, (keeper.TimestepValues, matched.times)
    for time in matched.times:
        scene.AnimationTime = time
        for proxy in selected:
            proxy.UpdatePipeline(time)
            saved = servermanager.Fetch(proxy).GetPointData().GetArray("saved_time").GetValue(0)
            assert same_saved_time(saved, time), (time, saved)
    print(f"FILTER_ONLY_PASS: actual common clocks {matched.times}; no rendering or real outputs")
