"""Small regression checks for Delta sparse-to-dense sample merging."""

from pathlib import Path

import pandas as pd
import pytest

from tutorials.vpm.delta_wing.assets._delta_wing_plots import _validate_duplicate_csv_rows
from tutorials.vpm.delta_wing.assets.render_delta_wing_gif import (
    _require_dense_native_source,
    _require_unique_presentation_sources,
    select_frames,
)


def test_duplicate_sample_clock_is_stricter_than_physical_f32_tolerance():
    sparse = pd.DataFrame(
        {
            "step": [1600],
            "surface": ["front_wing"],
            "time": [4.0],
            "force_z": [1.0],
            "source_segment": ["sparse"],
            "source_samples_directory": ["sparse"],
        }
    )
    dense = sparse.copy()
    dense["time"] = 4.00001
    dense["source_segment"] = "dense"
    dense["source_samples_directory"] = "dense"

    # 10 microseconds is inside the generic rtol=5e-6 physical tolerance at
    # t=4, but it is not the same accepted solver clock.
    with pytest.raises(ValueError, match="conflicts in time"):
        _validate_duplicate_csv_rows("vlm_surface_forces.csv", [sparse, dense], ["step", "surface"])


@pytest.mark.parametrize("time", [(float("nan"), float("nan")), ("4.0", "4.0")])
def test_duplicate_sample_clock_must_be_finite_numeric(time):
    sparse = pd.DataFrame(
        {
            "step": [1600],
            "surface": ["front_wing"],
            "time": [time[0]],
            "force_z": [1.0],
            "source_segment": ["sparse"],
            "source_samples_directory": ["sparse"],
        }
    )
    dense = sparse.copy()
    dense["time"] = time[1]
    dense["source_segment"] = "dense"
    dense["source_samples_directory"] = "dense"

    with pytest.raises(ValueError, match="conflicts in time"):
        _validate_duplicate_csv_rows("vlm_surface_forces.csv", [sparse, dense], ["step", "surface"])


def test_sparse_animation_holds_native_frames_at_every_presentation_target():
    records = [
        (0.0, "vpm_000000.h5", "sparse"),
        (1.0, "vpm_000100.h5", "sparse"),
    ]
    targets, selected = select_frames(records)

    assert len(targets) == 31
    assert len(selected) == 31
    assert selected[0] == records[0]
    assert selected[-1] == records[-1]


def test_public_animation_rejects_sparse_native_source():
    records = [
        (0.0, "vpm_000000.h5", "sparse"),
        (1.0, "vpm_000100.h5", "sparse"),
    ]
    with pytest.raises(ValueError, match="native backup gaps"):
        _require_dense_native_source(records)


def test_public_animation_rejects_repeated_native_presentation_state():
    records = [(0.0, Path("vpm_000000.h5"), "dense"), (0.025, Path("vpm_000010.h5"), "dense")]
    with pytest.raises(ValueError, match="repeat a native state"):
        _require_unique_presentation_sources([records[0], records[0], records[1]])
