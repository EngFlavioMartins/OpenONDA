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

For VPM, configure `vpm.VPMSetup(write_precision="f16" | "f32" | "f64",
checkpoint_store_velocity_gradient=True | False)`. VPM checkpoints are HDF5
with gzip level 4 and byte shuffling, plus an XDMF descriptor. They preserve
the particle fields consumed by tutorial post-processing and ParaView:
position, velocity, vorticity, vortex strength, core radius, volume,
viscosities, group ID, and zone ID.

`checkpoint_store_velocity_gradient=True` is the default and retains the
gradient for restart continuity and ParaView. Setting it to `False` omits that
large derived field; a non-potential VPM restore recomputes the gradient.
Strain rate and vector magnitudes are derived fields and are not stored.
ParaView can calculate magnitudes directly, and tutorial plotters compute
needed magnitudes from the retained vectors.

## Sampling budgets

Storage reductions must not remove the resolution a plot or diagnostic needs.
Lamb--Oseen surface fields use 12.5 points across the initial vortex-core
diameter and 53 frames over each 30-second experiment.  For the paired-vortex
cases, the sampled plane stops three final core radii beyond the tracked
vortices; the larger solver domain is unchanged.  This removes only far-field
samples whose vorticity is below the useful diagnostic range.

The vortex-ring tutorial keeps its 0.1-second scalar-diagnostic cadence and
0.5-second particle-checkpoint cadence.  The former supplies the local speed
fit used by its plots, and the latter gives 120 ParaView frames over the run.
Those checkpoints omit velocity gradients because the tutorial consumes only
the canonical particle fields and can recompute the gradient on restart.
