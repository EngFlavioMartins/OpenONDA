from types import SimpleNamespace

from source.solvers.vpm.turbulence.turbulence import ParticlesLES


def test_zero_smagorinsky_skips_viscosity_kernels():
    model = SimpleNamespace(
        smagorinsky_coefficient=0.0,
        compute=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    les = object.__new__(ParticlesLES)
    les.model = model

    les.compute(object())

    assert les.min_eddy_viscosity == 0.0
    assert les.max_eddy_viscosity == 0.0
    assert les.min_eddy_viscosity_ratio == 0.0
    assert les.max_eddy_viscosity_ratio == 0.0
