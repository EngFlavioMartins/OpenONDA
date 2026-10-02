"""Pure saved-clock matching; no solver or ParaView runtime required."""

import pytest

from openonda.saved_times import match_saved_times, read_pvd_times


def test_different_cadences_match_only_saved_frames():
    result = match_saved_times([0, 1, 2, 3, 4, 5, 8], [0, 4, 8])
    assert result.times == (0, 4, 8)
    assert result.indices == ((0, 4, 6), (0, 1, 2))


def test_accumulated_clock_roundoff_is_not_an_output_time_shift():
    result = match_saved_times([0, 11, 12], [0, 10.999999999999, 12.000000000001])
    assert result.times == (0, 11, 12)
    assert match_saved_times([11], [11.00000004]).times == ()


@pytest.mark.parametrize("times", [[1, 1], [2, 1], [1, 1 + 1e-11], [float('nan')], [-1]])
def test_invalid_or_ambiguous_clocks_are_rejected(times):
    with pytest.raises(ValueError):
        match_saved_times(times)


def test_one_query_cannot_choose_between_two_close_saved_states():
    with pytest.raises(ValueError, match="Ambiguous"):
        match_saved_times([1], [1 - 8e-11, 1 + 8e-11])


def test_empty_sources_do_not_invent_states():
    assert match_saved_times([], [0]).times == ()
    assert match_saved_times().indices == ()


def test_pvd_attribute_order_and_saved_frame_existence(tmp_path):
    path = tmp_path / "fields.pvd"
    (tmp_path / "a.vtu").touch()
    (tmp_path / "b.vtu").touch()
    path.write_text('<VTKFile><Collection><DataSet file="b.vtu" timestep="4"/>'
                    '<DataSet timestep="0" file="a.vtu"/></Collection></VTKFile>')
    assert read_pvd_times(path) == (0, 4)
    (tmp_path / "b.vtu").unlink()
    with pytest.raises(FileNotFoundError, match="Missing saved frame"):
        read_pvd_times(path)


def test_parallel_frame_requires_every_piece(tmp_path):
    collection = tmp_path / "fvm.pvd"
    collection.write_text('<VTKFile><Collection><DataSet timestep="1" file="fvm.pvtu"/>'
                          '</Collection></VTKFile>')
    (tmp_path / "fvm.pvtu").write_text('<VTKFile><PUnstructuredGrid><Piece Source="rank0.vtu"/>'
                                     '<Piece Source="rank1.vtu"/></PUnstructuredGrid></VTKFile>')
    (tmp_path / "rank0.vtu").touch()
    with pytest.raises(FileNotFoundError, match="rank1.vtu"):
        read_pvd_times(collection)
    (tmp_path / "rank1.vtu").touch()
    assert read_pvd_times(collection) == (1,)
