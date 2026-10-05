"""Matched planar cylinder phase comparison; not a grid qualification."""

from dataclasses import asdict
from pathlib import Path

from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
H = 0.04
module = load_case_module(CASE)
sampling = module
SPAN = module.FVM_RESOLVED_SPAN
DT = module.FVM_TIME_STEP_SIZE
EXCHANGE = module.VPM_TIME_STEP_SIZE
END = module.END_TIME
FORCES = sampling.FORCES
PROFILES = sampling.PROFILES
SLICES = sampling.SLICES
BACKUPS = sampling.BACKUPS
VOLUMES = sampling.VOLUMES

phase_lines = sampling.phase_lines
field_lines = sampling.field_lines


def steps(period, time_step):
    count = round(period / time_step)
    if count < 1 or abs(count * time_step - period) > 1e-10:
        raise ValueError("Verification period must resolve on the accepted clock")
    return count


def reference_case(end=END, cores=6):
    reference = load_case_module(CASE / "reference_flow")
    return reference.build_case("phase_h004", H, end_time=end, cores=cores)


def coupled_case(end=END, cores=4, device="CUDA"):
    return module.build_case(
        end_time=end,
        overrides={
            "hxy": H,
            "span": SPAN,
            "particle_spacing_ratio": 1.0,
            "cores": cores,
            "compute_device": device,
            "exchange_dt": EXCHANGE,
        },
    )


def comparison_settings():
    reference = load_case_module(CASE / "reference_flow")
    a, am = reference_case()
    b, particles, coupling, bm = coupled_case()
    from source.solvers.fvm.sampling.base import sampler_to_dict

    assert asdict(a.schemes) == asdict(b.schemes)
    assert asdict(a.pimple) == asdict(b.pimple)
    assert asdict(a.linear) == asdict(b.linear)
    assert asdict(a.transport) == asdict(b.transport)
    assert asdict(a.turbulence) == asdict(b.turbulence)
    assert a.time.time_step_size == b.time.time_step_size == DT
    assert a.time.adjustment is b.time.adjustment is None
    assert SPAN == reference.SPAN == 1.0
    assert am.levels == bm.levels == (-0.5 * SPAN, 0.5 * SPAN)
    assert particles.numerics.induction.planar_span == SPAN
    assert particles.numerics.induction.plane_z == 0.0
    assert particles.numerics.viscous.particle_spacing == H
    for flow in (a, b):
        boundaries = {patch.name: patch for patch in flow.boundaries}
        for name, other in (("zmin", "zmax"), ("zmax", "zmin")):
            patch = boundaries[name]
            assert patch.velocity_type == patch.pressure_type == "cyclic"
            assert patch.neighbour_patch == other
        force = next(sample for sample in flow.samplers if sample.file_name == "forces_history")
        assert force.reference_area == SPAN * module.DIAMETER
        assert force.reference_velocity == 1.0
    assert (
        tuple(a.initial_velocity)
        == tuple(b.initial_velocity)
        == tuple(module.STARTUP_FREESTREAM_VELOCITY)
    )
    assert reference.STARTUP_DURATION == module.STARTUP_DURATION
    assert reference.STARTUP_TRANSITION_DURATION == module.STARTUP_TRANSITION_DURATION
    assert tuple(reference.STARTUP_FREESTREAM_VELOCITY) == tuple(module.STARTUP_FREESTREAM_VELOCITY)
    assert tuple(reference.VELOCITY) == tuple(module.FREESTREAM_VELOCITY)
    common = {s.name: sampler_to_dict(s) for s in a.samplers if s.name != "midspan"}
    for sampler in b.samplers:
        if sampler.name != "midspan":
            assert sampler_to_dict(sampler) == common[sampler.name]
    return {
        "h": H,
        "span": SPAN,
        "span_layers": 1,
        "particle_span_layers": 1,
        "spanwise_boundary": "periodic",
        "fvm_dt": DT,
        "exchange_dt": EXCHANGE,
        "end_time": END,
        "force_phase_interval": FORCES,
        "profile_interval": PROFILES,
        "slice_interval": SLICES,
        "backup_interval": BACKUPS,
        "volume_interval": VOLUMES,
        "reference_domain": am.domain.bounds,
        "coupled_domain": bm.domain.bounds,
        "schemes": asdict(a.schemes),
        "pimple": asdict(a.pimple),
        "linear": asdict(a.linear),
        "interface": coupling.to_dict(),
        "reference_samplers": [sampler_to_dict(s) for s in a.samplers],
        "coupled_fvm_samplers": [sampler_to_dict(s) for s in b.samplers],
        "status": "provisional planar working mesh; no grid-independence claim",
        "initial_condition": "same compact divergence-free XY curl on the startup background; w=0",
        "startup": {
            "duration": module.STARTUP_DURATION,
            "transition_duration": module.STARTUP_TRANSITION_DURATION,
            "freestream_velocity": list(module.STARTUP_FREESTREAM_VELOCITY),
            "steady_freestream_velocity": list(module.FREESTREAM_VELOCITY),
        },
        "phase_analysis": "Separate startup phase offset, period error and accumulating phase drift; never shift simulation clocks.",
    }
