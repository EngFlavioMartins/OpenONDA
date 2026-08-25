from __future__ import annotations

import numpy as np

from source.coupler.reporting import format_coupler_log
from source.coupler.vorticity_transfer import TransferResult, _transfer_log_record


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
    result = TransferResult(
        n_particles_before=463_252,
        n_particles_retained=68_049,
        n_particles_removed=395_203,
        n_particles_blended=12_345,
        n_particles_injected=401_002,
        n_particles_after=469_051,
        injected_vortex_strength_l1=1.834,
        injected_vortex_strength_net=np.array([3.924e-4, 0.0, 0.0]),
        replaced_vortex_strength_l1=1.812,
        replaced_vortex_strength_net=np.array([3.824e-4, 0.0, 0.0]),
        state_change_vortex_strength_net=np.array([1.0e-5, 0.0, 0.0]),
        eta_blending_enabled=True,
    )

    record = _transfer_log_record(11, result)

    assert record.count("[Coupler][StateReplacement]") == 1
    assert "eta blend on" in record
    assert "removed 395,203 | blended 12,345 | injected 401,002" in record
    assert "replaced L1 1.812e+00 | injected L1 1.834e+00 m^3/s" in record
    assert "net state change 1.000e-05 m^3/s" in record
    assert "n_particles_before=" not in record


def test_projected_renewal_log_reports_field_gates_and_runtime_gbd_guard() -> None:
    result = TransferResult(
        n_particles_before=120,
        n_particles_retained=120,
        n_particles_removed=0,
        n_particles_blended=96,
        n_particles_injected=4,
        n_particles_after=124,
        injected_vortex_strength_l1=2.0e-4,
        transfer_method="projected_gbd_renewal",
        projection_vorticity_relative_error=4.0e-4,
        projection_velocity_relative_error=7.0e-5,
        projection_condition_number=12.5,
        selective_support_births=4,
        renewal_guard_width=0.09375,
        renewal_diffusion_substeps=1,
    )

    record = _transfer_log_record(3, result)

    assert "omega error 4.000e-04 | normal velocity error 7.000e-05" in record
    assert "GBD guard  0.09375 m | diffusion substeps 1 | selective births 4" in record
