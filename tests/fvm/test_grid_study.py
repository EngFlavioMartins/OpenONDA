"""Grid-study force metrics and FVM case checks name what they measure."""

import numpy as np
import pytest

from openonda import fvm
from openonda.core.simulation.forcing import VelocityBoundary, VelocityRamp
from openonda.fvm.grid_study import _force_metrics


def test_force_metrics_report_the_lift_frequency_in_hertz(tmp_path):
    time = np.linspace(0.0, 10.0, 2001)
    lift = 0.3 * np.sin(2.0 * np.pi * 2.0 * time)
    path = tmp_path / "forces_history.csv"
    rows = np.column_stack((time, np.full_like(time, 1.2), lift, np.zeros_like(time)))
    header = "time,drag_coefficient,lift_coefficient,side_force_coefficient"
    np.savetxt(path, rows, delimiter=",", header=header, comments="")

    metrics = _force_metrics(path, 0.0, 10.0)

    assert "strouhal" not in metrics
    assert metrics["lift_frequency"] == pytest.approx(2.0, rel=0.01)
    assert metrics["mean_drag"] == pytest.approx(1.2)


def test_velocity_history_on_an_unknown_patch_lists_the_configured_patches():
    x = np.linspace(0.0, 1.0, 3)
    mesh = fvm.mesher.box_mesh_3d(x, x, np.array([0.0, 0.5]), empty_spanwise=True)
    ramp = VelocityRamp(
        initial=(0.0, 0.0, 0.0), final=(1.0, 0.0, 0.0), start_time=0.0, end_time=1.0
    )
    with pytest.raises(ValueError, match=r"unconfigured patches \['inflow'\]; configured patches"):
        fvm.FVMCase(
            name="box",
            run=fvm.RunPlan(end_time=1.0),
            mesh=mesh,
            boundaries=(fvm.BoundaryConfig.inlet("inlet", (1.0, 0.0, 0.0)),),
            velocity_boundaries=(VelocityBoundary(patches=("inflow",), velocity=ramp),),
        )
