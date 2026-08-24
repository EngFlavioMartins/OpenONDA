from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from source.coupler.reporting import format_coupler_log
from source.coupler.vorticity_transfer import _transfer_log_record


def test_format_coupler_log_keeps_one_searchable_header() -> None:
    record = format_coupler_log(
        "Example",
        "step 12",
        "first quantity  1.000e+00 m/s",
        "second quantity 2.000e+00 m^3/s",
    )

    assert record.splitlines() == [
        "[Coupler][Example] step 12",
        "  first quantity  1.000e+00 m/s",
        "  second quantity 2.000e+00 m^3/s",
    ]


def test_transfer_log_is_scannable_and_retains_all_diagnostics() -> None:
    result = SimpleNamespace(
        n_existing_particles=68_049,
        n_updated_particles=68_045,
        n_added_particles=395_203,
        n_total_particles=463_252,
        n_support_nodes=463_248,
        correction_vortex_strength_l1=1.834,
        correction_vortex_strength_net=np.array([3.924e-4, 0.0, 0.0]),
        diagnostics_evaluated=False,
    )

    record = _transfer_log_record(11, result)

    assert record.count("[Coupler][Transfer]") == 1
    assert "existing 68,049 | updated 68,045 | added 395,203 | total 463,252" in record
    assert "463,248 lattice nodes" in record
    assert "1.834e+00 m^3/s | net magnitude 3.924e-04 m^3/s" in record
    assert "relative vorticity L2 not evaluated" in record
    assert "n_existing_particles=" not in record
    assert max(map(len, record.splitlines())) <= 110
