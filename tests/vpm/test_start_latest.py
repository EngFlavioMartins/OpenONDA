"""Automatic continuation preserves numerical state and scientific histories."""

from dataclasses import replace

import h5py
import numpy as np
import pandas as pd
import pytest

from openonda import vpm
from tests.vpm._flat_plate_geometry import create_flat_plate
from tests.vpm.test_backup_storage import _add_counter_rotating_pair, _case


def _build(directory, with_vlm):
    case = _case(directory)
    if with_vlm:
        vlm = vpm.VLMSetup(
            surfaces=(
                vpm.VLMSurfaceSetup(
                    create_flat_plate(
                        chord=1,
                        span=2,
                        angle_of_attack_degrees=5,
                        n_chordwise_panels=1,
                        n_spanwise_panels=2,
                    )
                ),
            ),
            freestream_velocity=(1, 0, 0),
            kinematic_viscosity=0.01,
        )
        case = replace(case, numerics=replace(case.numerics, vlm=vlm))
    return replace(
        case,
        run=vpm.RunPlan(steps=3),
        backup=vpm.Backup(interval_steps=2),
        samplers=vpm.Samplers(
            samples=(vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(1), initial=True),)
        ),
    )


def _state(directory):
    with h5py.File(directory / "solution/vpm/vpm_000003.h5") as archive:
        values = {}

        def collect(name, item):
            if isinstance(item, h5py.Dataset):
                values[name] = item[()]

        archive.visititems(collect)
    return values


@pytest.mark.parametrize("with_vlm", [False, True])
def test_latest_replays_unsaved_tail_and_completed_rerun_is_idle(tmp_path, with_vlm):
    reference = vpm.VPMSolver(_build(tmp_path / "reference", with_vlm))
    _add_counter_rotating_pair(reference)
    reference.run(start_from="latest")
    expected = _state(tmp_path / "reference")

    directory = tmp_path / "continued"
    interrupted = vpm.VPMSolver(_build(directory, with_vlm))
    _add_counter_rotating_pair(interrupted)
    advance = interrupted.advance

    def fail_after_unsaved_output():
        advance()
        if interrupted.step == 3:
            raise RuntimeError("interrupted after samples")

    interrupted.advance = fail_after_unsaved_output
    with pytest.raises(RuntimeError, match="interrupted after samples"):
        interrupted.run(start_from="latest")
    log = (directory / "solution/vpm.log").read_text()
    assert not (directory / "solution/vpm/vpm_000003.h5").exists()
    # An unfinished newer checkpoint must never win discovery.
    (directory / "solution/vpm/vpm_999999.h5.tmp").write_text("unfinished")
    # A committed checkpoint can precede a sampler write. Recover that event
    # as well as discarding the future event at step three.
    table = directory / "samples/flow_integrals.csv"
    rows = pd.read_csv(table)
    rows[rows["step"] != 2].to_csv(table, index=False)
    resumed = vpm.VPMSolver(_build(directory, with_vlm))
    resumed.run(start_from="latest")
    assert resumed.step == 3
    for name, values in _state(directory).items():
        if values.dtype.kind in "fiu":
            np.testing.assert_allclose(values, expected[name], rtol=1e-6, atol=1e-8)
    assert (directory / "solution/vpm.log").read_text().startswith(log)
    for path in (directory / "samples").glob("*.csv"):
        actual = pd.read_csv(path)
        wanted = pd.read_csv(tmp_path / "reference/samples" / path.name)
        assert actual["time"].tolist() == wanted["time"].tolist(), path
    histories = {path: path.read_bytes() for path in (directory / "samples").glob("*.csv")}
    backup = directory / "solution/vpm/vpm_000003.h5"
    stamp = backup.stat().st_mtime_ns
    done = vpm.VPMSolver(_build(directory, with_vlm))
    done.run(start_from="latest")
    assert done.step == 3
    assert backup.stat().st_mtime_ns == stamp
    assert all(path.read_bytes() == data for path, data in histories.items())
    assert list((directory / "samples/restart-branches").glob("before-*"))


def test_latest_does_not_fall_back_from_corrupt_committed_backup(tmp_path):
    directory = tmp_path / "solution/vpm"
    directory.mkdir(parents=True)
    (directory / "vpm_000002.h5").write_text("corrupt committed state")
    solver = vpm.VPMSolver(_build(tmp_path, False))
    with pytest.raises((ValueError, OSError)):
        solver.run(start_from="latest")
    assert solver.step == 0
