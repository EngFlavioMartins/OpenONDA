"""Historical matched cylinder phase comparison; not a grid qualification."""

from dataclasses import asdict
from pathlib import Path

from openonda.tutorial_runner import load_case_module
from openonda.tutorial_support import cylinder_sampling as sampling

CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
H = 0.04
module = load_case_module(CASE)
SPAN = module.FVM_RESOLVED_SPAN
DT = module.FVM_TIME_STEP_SIZE
EXCHANGE = module.VPM_TIME_STEP_SIZE
END = module.END_TIME
FORCES = sampling.FORCES
PROFILES = sampling.PROFILES
SLICES = sampling.SLICES
BACKUPS = sampling.BACKUPS
VOLUMES = sampling.VOLUMES

steps = sampling.steps
phase_lines = sampling.phase_lines
field_lines = sampling.field_lines


def reference_case(end=END, cores=6):
    reference = load_case_module(CASE / "reference_flow")
    return reference.build_case("phase_h004", H, end_time=end, cores=cores)


def coupled_case(end=END, cores=4, device="CUDA"):
    return module.build_case(
        end_time=end,
        overrides={
            "hxy": H,
            "dz": H,
            "span": SPAN,
            "particle_spacing_ratio": 1.0,
            "cores": cores,
            "compute_device": device,
            "exchange_dt": EXCHANGE,
        },
    )


def contract():
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
    assert len(am.levels) == len(bm.levels) == 25
    assert particles.numerics.viscous.particle_spacing == H
    assert tuple(a.initial_velocity) == tuple(b.initial_velocity) == tuple(
        module.STARTUP_FREESTREAM_VELOCITY
    )
    assert reference.STARTUP_DURATION == module.STARTUP_DURATION
    assert reference.STARTUP_TRANSITION_DURATION == module.STARTUP_TRANSITION_DURATION
    assert tuple(reference.STARTUP_FREESTREAM_VELOCITY) == tuple(
        module.STARTUP_FREESTREAM_VELOCITY
    )
    assert tuple(reference.VELOCITY) == tuple(module.FREESTREAM_VELOCITY)
    common = {s.name: sampler_to_dict(s) for s in a.samplers if s.name != "midspan"}
    for sampler in b.samplers:
        if sampler.name != "midspan":
            assert sampler_to_dict(sampler) == common[sampler.name]
    return {
        "h": H,
        "span": SPAN,
        "span_layers": 24,
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
        "status": "provisional working mesh; no grid-independence claim",
        "initial_condition": "same compact divergence-free curl on the startup background",
        "startup": {
            "duration": module.STARTUP_DURATION,
            "transition_duration": module.STARTUP_TRANSITION_DURATION,
            "freestream_velocity": list(module.STARTUP_FREESTREAM_VELOCITY),
            "steady_freestream_velocity": list(module.FREESTREAM_VELOCITY),
        },
        "phase_analysis": "Separate startup phase offset, period error and accumulating phase drift; never shift simulation clocks.",
    }
