# Storage and visualization policy

OpenONDA separates compute precision from write precision. The write choices
are `f16`, `f32`, and `f64`; the default is `f32`.

`f16` rounds floating fields to half precision and writes float32 arrays. VTK
XML and XDMF readers used by ParaView do not reliably support native float16,
while rounded float32 remains compact under the configured compression and is
portable across ParaView readers.

For FVM visualization, configure `fvm.OutputConfig(precision="f16" | "f32" |
"f64", compression="lz4" | "zlib" | "none")`. VTU output is appended raw
VTK XML. FVM restart checkpoints are always lossless; their storage layout is
an implementation detail and not a visualization format.

VPM backups and logging have one configuration object:

```python
backup=vpm.Backup(
    interval_steps=25,
    directory="solution",
    log_directory="solution",
)
```

Both directories default to `solution`; specify either only when it differs.
Backups use the fixed names `vpm_STEP.h5` and `vpm_STEP.xdmf`, gzip level 4,
and byte shuffling. The schema preserves position, velocity, vorticity, vortex
strength, core radius, volume, viscosities, group ID, and zone ID. Velocity
gradient, strain rate, and vector magnitudes are derived fields and are never
stored. A non-potential restore recomputes the gradient.

Sampling likewise has one configuration object. Sampler objects are passed
positionally and the optional directory is placed inside the constructor:

```python
samplers=vpm.Samplers(
    vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(10)),
    directory="run_a",
)
```

## Sampling budgets

Storage reductions must not remove the resolution a plot or diagnostic needs.
Lamb--Oseen surface fields use 12.5 points across the initial vortex-core
diameter and 53 frames over each 30-second experiment.  For the paired-vortex
cases, the sampled plane stops three final core radii beyond the tracked
vortices; the larger solver domain is unchanged.  This removes only far-field
samples whose vorticity is below the useful diagnostic range.

The vortex-ring tutorial keeps its 0.1-second scalar-diagnostic cadence and
0.5-second particle-backup cadence. The former supplies the local speed
fit used by its plots, and the latter gives 120 ParaView frames over the run.
Those backups use the fixed schema above.
