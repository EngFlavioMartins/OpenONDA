# VPM and VLM configuration

VPM and VLM cases use one immutable setup graph. Runtime solvers do not accept
configuration keyword overrides and do not expose configuration update methods.

```python
from source.solvers.VPM import (
    Solver,
    StabilizationConfig,
    VLMSetup,
    VLMMeshSetup,
    VLMSurfaceSetup,
    VPMSetup,
)

setup = VPMSetup(
    time_step_size=0.01,
    vlm=VLMSetup(
        surfaces=(
            VLMSurfaceSetup(
                "assets/wing.json",
                name="wing",
                kinematics=wing_motion,
            ),
        ),
        mesh=VLMMeshSetup.geometric(ratio=3.0, region="end"),
        density=1.225,
        viscosity=1.5e-5,
    ),
    stabilization=StabilizationConfig.bounded_domain(
        (-2.0, 20.0, -2.0, 2.0, -2.0, 2.0)
    ),
)

solver = Solver(setup=setup)
```

The main rules are:

- Use `VPMSetup`; there is no second solver-configuration alias.
- Pass the complete setup as `Solver(setup=...)`. Constructor overrides and
  `update_config()` are intentionally unavailable.
- Declare VLM surfaces, transforms, kinematics, mesh distribution, force
  policy, and fluid properties in `VLMSetup`. The VPM solver creates the
  runtime VLM solver after initializing Taichi.
- Use `StabilizationConfig.bounded_domain()` only as a particle-retention
  policy for finite wake or coupled domains. It does not alter particle
  strengths or core sizes. Configuration dataclasses are frozen.
- Use `AUTO`, `CPU`, `CUDA`, `VULKAN`, or `METAL` for the compute backend.
- Sampler persistence has no format switch. Surface samplers write VTS/PVD
  time series; line and point samplers append to one CSV table with
  `flow_time` and `time_step` columns.
- Put end-of-run field snapshots in `final_samplers` and call
  `solver.execute_final_samplers()` after the time loop.

Defaults select the standard production path: RK3 advection and stretching,
core-spreading viscosity, DNS turbulence, treecode velocity for supported
kernels, automatic accelerator selection, and geometry-derived VLM capacity.
