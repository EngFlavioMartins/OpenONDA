from pathlib import Path

import numpy as np

from source.coupler.diagnostics.recirculation import (
    compare_series,
    first_exceedance,
    load_centerline_series,
    recirculation_metrics,
)


def test_recirc_metrics_interpolates_reattachment():
    x = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    ux = np.array([0.1, -0.4, -0.2, 0.1, 0.5])

    metrics = recirculation_metrics(x, ux, time=1.0, probe_x=1.45)

    assert metrics.min_ux == -0.4
    assert metrics.x_min_ux == 0.75
    assert np.isclose(metrics.reattachment_x, 1.1666666666666667)
    assert np.isclose(metrics.recirculation_length, 0.6666666666666667)
    assert np.isclose(metrics.ux_probe, 0.42)


def _write_centerline(root: Path, time: float, ux_values: list[float]) -> None:
    time_dir = root / f"{time:g}"
    time_dir.mkdir(parents=True)
    x = [0.5, 0.75, 1.0, 1.25, 1.5]
    rows = ["x,p,U_0,U_1,U_2"]
    rows.extend(f"{xi},0,{ui},0,0" for xi, ui in zip(x, ux_values))
    (time_dir / "centerline_p_U.csv").write_text("\n".join(rows) + "\n")


def test_compare_series_detects_first_reattachment_exceedance(tmp_path):
    hybrid_dir = tmp_path / "hybrid"
    reference_dir = tmp_path / "reference"
    _write_centerline(hybrid_dir, 1.0, [0.1, -0.4, -0.2, 0.1, 0.5])
    _write_centerline(reference_dir, 1.0, [0.1, -0.4, -0.2, 0.1, 0.5])
    _write_centerline(hybrid_dir, 2.0, [0.1, -0.4, -0.2, 0.1, 0.5])
    _write_centerline(reference_dir, 2.0, [0.1, -0.4, -0.3, -0.2, 0.1])

    hybrid = load_centerline_series(hybrid_dir)
    reference = load_centerline_series(reference_dir)
    comparisons = compare_series(hybrid, reference, start_time=1.0)

    first = first_exceedance(comparisons, "reattachment_error", 0.1)

    assert first is not None
    assert first.time == 2.0
    assert np.isclose(first.hybrid.reattachment_x, 1.1666666666666667)
    assert np.isclose(first.reference.reattachment_x, 1.4166666666666667)
