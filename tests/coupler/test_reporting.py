from __future__ import annotations

import numpy as np

from source.coupler.reporting import format_coupler_log, format_coupler_step
from source.coupler.vorticity_transfer import TransferResult, _transfer_log_record


def test_format_coupler_log_aligns_values_on_one_column() -> None:
    record = format_coupler_log(
        "example",
        ("first quantity", "1.000e+00", "m/s"),
        ("second quantity", "2.000e+00", "m^3/s"),
        ("mode", "impulsive"),
    )

    lines = record.splitlines()
    assert lines[0] == "coupler  example"
    assert lines[1:] == [
        f"{'':14}{'first quantity':<30}{'1.000e+00':>12}  m/s",
        f"{'':14}{'second quantity':<30}{'2.000e+00':>12}  m^3/s",
        f"{'':14}{'mode':<30}{'impulsive':>12}",
    ]
    # Every value ends on the same column, so the numbers line up when scanned.
    assert lines[1].index("1.000e+00") + len("1.000e+00") == 56
    assert lines[2].index("2.000e+00") + len("2.000e+00") == 56


def test_format_coupler_step_opens_a_banner_with_the_physical_time() -> None:
    banner = format_coupler_step(7, 2000, 0.07)

    assert banner.splitlines() == [
        "",
        "=" * 78,
        " coupling step 7 of 2,000                             physical time 0.070000 s",
        "=" * 78,
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
    lines = record.splitlines()

    assert lines[0] == "coupler  state replacement, step 11"
    assert all(line.startswith(" " * 14) for line in lines[1:])
    assert "blend, eta" in record and "on" in record
    assert "particles, removed" in record and "395,203" in record
    assert "particles, blended" in record and "12,345" in record
    assert "particles, injected" in record and "401,002" in record
    assert "1.812e+00" in record and "1.834e+00" in record
    assert "vortex strength, net change" in record and "1.000e-05" in record
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

    assert "projection error, omega" in record and "4.000e-04" in record
    assert "projection error, normal velocity" in record and "7.000e-05" in record
    assert "gbd guard width" in record and "0.09375" in record
    assert "gbd diffusion substeps" in record
    assert "selective support births" in record
