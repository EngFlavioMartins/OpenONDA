from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from source.coupler.reporting import compute_diagnostics
from source.coupler.vorticity_transfer import TransferResult, _transfer_log_record


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

    record = str(_transfer_log_record(3, result)).lower()

    assert "projection error, omega" in record and "4.000e-04" in record
    assert "projection error, normal velocity" in record and "7.0000e-05" in record
    assert "gbd guard width" in record and "0.09375" in record
    assert "gbd diffusion substeps" in record
    assert "selective support births" in record


def test_buffered_renewal_serializes_raw_applied_and_corrected_closure() -> None:
    result = TransferResult(
        n_particles_before=12,
        n_particles_retained=3,
        n_particles_removed=9,
        n_particles_blended=9,
        n_particles_injected=10,
        n_particles_after=13,
        injected_vortex_strength_l1=1.0,
        transfer_method="buffered_m4_renewal",
        coalesced_outer_particles=1,
        renewal_raw_vortex_strength_error=0.4,
        renewal_applied_vortex_strength_correction=0.4,
        renewal_conservation_error=2.0e-14,
        renewal_vortex_strength_tolerance=4.0e-8,
        renewal_raw_linear_impulse_error=0.3,
        renewal_applied_linear_impulse_correction=0.3,
        renewal_linear_impulse_error=3.0e-14,
        renewal_linear_impulse_tolerance=6.0e-8,
        renewal_applied_particle_strength_fraction=0.047,
    )
    record = str(_transfer_log_record(3, result)).lower()
    assert "renewal closure, raw strength mismatch" in record
    assert "renewal closure, particle-strength correction" in record
    assert "fvm map, net error" not in record
    assert "fvm map, first-moment error" not in record

    coupler = SimpleNamespace(
        _last_transfer_result=None,
        _last_vpm_boundary_condition_flux_diagnostics={
            "raw_mismatch": 0.0,
            "raw_relative": 0.0,
            "acceptance_limit": 0.0,
            "applied_correction": 0.0,
            "corrected_mismatch": 0.0,
        },
        vorticity_transfer=None,
        vpm_solver=SimpleNamespace(
            physics=SimpleNamespace(
                induction=SimpleNamespace(last_tail={"shell": 16, "relative": 2.5e-5}),
                last_gbd_moment_recovery={
                    "applied": True,
                    "nonzero_node_count": 120,
                    "retained_node_count": 100,
                    "pruned_node_count": 20,
                    "support_augmented_node_count": 3,
                    "correction_fraction": 0.012,
                    "normalized_vortex_strength_residual": 1.0e-8,
                    "normalized_linear_impulse_residual": 2.0e-8,
                    "normalized_angular_impulse_residual": 3.0e-8,
                },
                last_solid_projection={
                    "stage_count": 2,
                    "accepted_count": 1,
                    "stage_displacement_l1": 0.004,
                    "accepted_displacement_l1": 0.001,
                    "stage_impulse_change": np.array([1.0e-5, 0.0, 0.0]),
                    "accepted_impulse_change": np.array([0.0, 2.0e-5, 0.0]),
                },
                last_gbd_wall_transfer={
                    "wall_adjacent_particles": 6,
                    "excluded_signed_weight_l1": 0.03,
                    "fluid_correction_l1": 0.04,
                },
            )
        ),
        n_fvm_substeps=2,
    )

    diagnostics = compute_diagnostics(coupler, result)
    transfer = diagnostics["transfer"]

    assert transfer["coalesced_outer_particles"] == 1
    assert transfer["renewal_raw_vortex_strength_error"] == 0.4
    assert transfer["renewal_applied_vortex_strength_correction"] == 0.4
    assert transfer["renewal_conservation_error"] == 2.0e-14
    assert transfer["renewal_vortex_strength_tolerance"] == 4.0e-8
    assert transfer["renewal_raw_linear_impulse_error"] == 0.3
    assert transfer["renewal_applied_linear_impulse_correction"] == 0.3
    assert transfer["renewal_linear_impulse_error"] == 3.0e-14
    assert transfer["renewal_linear_impulse_tolerance"] == 6.0e-8
    assert transfer["renewal_applied_particle_strength_fraction"] == 0.047
    assert transfer["fvm_mapping_vortex_strength_error"] is None
    assert transfer["fvm_mapping_first_moment_error"] is None
    assert diagnostics["gbd_moment_recovery"] == {
        "applied": True,
        "nonzero_node_count": 120,
        "retained_node_count": 100,
        "pruned_node_count": 20,
        "support_augmented_node_count": 3,
        "correction_fraction": 0.012,
        "normalized_vortex_strength_residual": 1.0e-8,
        "normalized_linear_impulse_residual": 2.0e-8,
        "normalized_angular_impulse_residual": 3.0e-8,
    }
    assert diagnostics["last_induction_image_call"] == {"shell": 16.0, "relative": 2.5e-5}
    assert diagnostics["solid_projection"]["stage_count"] == 2
    assert diagnostics["solid_projection"]["accepted_impulse_change"] == [0.0, 2.0e-5, 0.0]
    assert diagnostics["gbd_wall_transfer"]["wall_adjacent_particles"] == 6
