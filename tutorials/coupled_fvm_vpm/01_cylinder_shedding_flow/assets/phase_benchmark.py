"""Matched, provisional cylinder phase benchmark; not a grid qualification.

Both FVM regions use h=.04, dz=.04, dt=.008, BDF2 and identical PIMPLE
controls. The coupled outer region uses hp=.04 and exchange dt=.04. No
reference signal enters either solver. Output clocks and probes are physical.
"""
from dataclasses import asdict, replace
import math
from pathlib import Path

import openonda.fvm as fvm
import openonda.vpm as vpm
from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[1]
H = 0.04
SPAN = 0.96
DT = 0.008
EXCHANGE = 0.04
END = 100.0
FORCES = 0.04
PROFILES = 0.2
SLICES = 0.4
BACKUPS = 1.0
VOLUMES = 4.0


def steps(interval, dt=DT):
    count = round(interval / dt)
    if count < 1 or abs(count * dt - interval) > 1e-10:
        raise ValueError("Every output must land on a physical accepted time")
    return count


def phase_lines(outer=False):
    """Identical coordinates in the reference and corresponding coupled region."""
    lo, hi, count = (1.6, 8.0, 65) if outer else (0.6, 1.4, 9)
    region = "wake" if outer else "near"
    return [(f"phase_{region}_{side}", [lo, y, 0.], [hi, y, 0.], count)
            for side, y in (("upper", .6), ("lower", -.6))]


def field_lines(reference):
    lines = [("centreline", [.6, 0., 0.], [1.4, 0., 0.], 11)]
    for label, z in (("lower", -SPAN/4), ("middle", 0.), ("upper", SPAN/4)):
        lines.append((f"span_{label}", [1., -1.2, z], [1., 1.2, z], 31))
    if reference:
        for x in (2, 4):
            lines.append((f"transverse_x{x}", [x, -2., 0.], [x, 2., 0.], 51))
    return lines


def fvm_samplers(reference):
    fast = fvm.RunSchedule(every_n_steps=steps(FORCES))
    slow = fvm.RunSchedule(every_n_steps=steps(PROFILES))
    samplers = [fvm.ForceSampler(patch_names=["cylinder"], reference_velocity=1.,
        reference_area=SPAN, reference_length=1., file_name="forces_history", schedule=fast)]
    for name, start, end, count in phase_lines() + (phase_lines(True) if reference else []):
        samplers.append(fvm.LineSampler(start=start, end=end, n_points=count, k=12,
            reconstruction="affine", file_name=name, schedule=fast))
    for name, start, end, count in field_lines(reference):
        samplers.append(fvm.LineSampler(start=start, end=end, n_points=count, k=12,
            reconstruction="affine", file_name=name, schedule=slow))
    samplers.append(fvm.SurfaceSampler(point=[0., 0., 0.], normal=[0., 0., 1.],
        bounds=[-1.6, 8. if reference else 1.6, -2. if reference else -1.6,
                2. if reference else 1.6], spacing=.1, file_name="midspan",
        schedule=fvm.RunSchedule(every_n_steps=steps(SLICES)),
        body_bounds=[-.5, .5, -.5, .5, -6., 6.], body_geometry="cylinder_z"))
    return tuple(samplers)


def matched_fvm(setup, reference, end=END):
    return replace(setup, samplers=fvm_samplers(reference),
        schemes=fvm.DiscretizationConfig(convection_scheme="limitedLinear",
                                        gradient_scheme="lsq", time_scheme="backward"),
        pimple=fvm.PimpleControl(n_outer_correctors=2, n_correctors=2,
                                velocity_relaxation=.7, pressure_relaxation=.3),
        time=fvm.TimeConfig(time_step_size=DT, end_time=end,
            output_schedule=fvm.RunSchedule(every_n_steps=steps(VOLUMES))),
        backup=fvm.BackupConfig(schedule=(fvm.RunSchedule(every_n_steps=steps(BACKUPS))
                                         if reference else None), write_at_end=reference))


def reference_case(end=END, cores=6):
    module = load_case_module(CASE / "reference_flow")
    setup, mesh = module.build_case("phase_h004", H, end_time=end, cores=cores)
    return matched_fvm(setup, True, end), mesh


def coupled_case(end=END, cores=4, device="CPU"):
    module = load_case_module(CASE)
    setup, particles, coupling, mesh = module.build_case(end_time=end, overrides=dict(
        hxy=H, dz=H, span=SPAN, particle_spacing_ratio=1., cores=cores,
        compute_device=device, exchange_dt=EXCHANGE))
    samples = []
    for name, start, finish, count in phase_lines() + phase_lines(True):
        samples.append(vpm.LineSampler(start=start, end=finish,
            spacing=math.dist(start, finish)/(count-1)*(1+1e-12), file_name="vpm_" + name,
            schedule=vpm.EverySteps(steps(FORCES, EXCHANGE))))
    for x in (2, 4):
        samples.append(vpm.LineSampler(start=[x, -2., 0.], end=[x, 2., 0.],
            spacing=.08*(1+1e-12), file_name=f"vpm_transverse_x{x}",
            schedule=vpm.EverySteps(steps(PROFILES, EXCHANGE))))
    samples.append(vpm.SurfaceSampler(point=[0., 0., 0.], normal=[0., 0., 1.],
        bounds=[1.6, 8., -2., 2.], spacing=.1, file_name="vpm_midspan",
        include_derivatives=False, schedule=vpm.EverySteps(steps(SLICES, EXCHANGE))))
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
