"""Historical matched cylinder phase comparison; not a grid qualification."""

from dataclasses import asdict, replace
from pathlib import Path

import openonda.fvm as fvm
import openonda.vpm as vpm
from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
sampling = load_case_module(CASE, "assets.sampling")
H = 0.04
SPAN = sampling.SPAN
DT = sampling.DT
EXCHANGE = sampling.EXCHANGE
END = sampling.END
FORCES = sampling.FORCES
PROFILES = sampling.PROFILES
SLICES = sampling.SLICES
BACKUPS = sampling.BACKUPS
VOLUMES = sampling.VOLUMES

steps = sampling.steps
phase_lines = sampling.phase_lines
field_lines = sampling.field_lines
_sampling = sampling._sampling
fvm_samplers = sampling.fvm_samplers
vpm_samplers = sampling.vpm_samplers
matched_fvm = sampling.configure_fvm


def reference_case(end=END, cores=6):
    module = load_case_module(CASE / "reference_flow")
    setup, mesh = module.build_case("phase_h004", H, end_time=end, cores=cores)
    return matched_fvm(setup, True, end), mesh


def coupled_case(end=END, cores=4, device="CUDA"):
    module = load_case_module(CASE)
    # This historical comparison factory explicitly retains the legacy image
    # operator on either device; the ordinary setup selects its mesh policy.
    setup, particles, coupling, mesh = module.build_case(
        end_time=end, gaussian_mesh_policy=None, overrides=dict(
        hxy=H, dz=H, span=SPAN, particle_spacing_ratio=1., cores=cores,
        compute_device=device, exchange_dt=EXCHANGE))
    samples = vpm_samplers()
    particles = replace(particles, samplers=vpm.Samplers(samples=tuple(samples)))
    coupling = replace(coupling, backup_interval_steps=steps(BACKUPS, EXCHANGE))
    return matched_fvm(setup, False, end), particles, coupling, mesh


def contract():
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
    common = {s.name: sampler_to_dict(s) for s in a.samplers if s.name != "midspan"}
    for sampler in b.samplers:
        if sampler.name != "midspan":
            assert sampler_to_dict(sampler) == common[sampler.name]
    return dict(h=H, span=SPAN, span_layers=24, fvm_dt=DT, exchange_dt=EXCHANGE,
        end_time=END, force_phase_interval=FORCES, profile_interval=PROFILES,
        slice_interval=SLICES, backup_interval=BACKUPS, volume_interval=VOLUMES,
        reference_domain=am.domain.bounds, coupled_domain=bm.domain.bounds,
        schemes=asdict(a.schemes), pimple=asdict(a.pimple),
        linear=asdict(a.linear), interface=coupling.to_dict(),
        reference_samplers=[sampler_to_dict(s) for s in a.samplers],
        coupled_fvm_samplers=[sampler_to_dict(s) for s in b.samplers],
        status="provisional working mesh; no grid-independence claim",
        initial_condition="same openonda.cylinder_campaign.cylinder_initial_velocity",
        phase_analysis="Separate startup phase offset, period error and accumulating phase drift; never shift simulation clocks.")
