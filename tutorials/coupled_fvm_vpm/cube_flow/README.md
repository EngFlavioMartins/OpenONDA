# Native FVM–VPM cube flow

This Re=1000 cube case uses OpenONDA's adaptive Cartesian FVM mesher and the
FVM–VPM coupler. It has no external mesher or repository-path requirement.

After installing OpenONDA, run the production case with:

```sh
./allrun.sh
./allplot.sh
```

`allrun.sh` removes the previous generated outputs before starting a new run.
The preserved baseline samples remain in `samples_archive/`; the fully
meshed reference samples are in `reference_flow/samples/`. Run
`./allplot.sh pdf` when PDF output is required; PNG is the default.

The production case uses four partitioned FVM ranks. Its coupled MPI smoke
test includes the body-wall geometry gathers used during transfer setup.

Production inputs use the wall-commensurate particle spacing `h=0.03125`, an
FVM time-step size of `0.005 s`, and a VPM/coupling time-step size of `0.010 s`.
The coupler therefore advances two FVM substeps per VPM step. All force, line,
and surface samplers use the single `SAMPLING_INTERVAL_TIME` defined in
`cube_flow_setup.py`; FVM and VPM backups are written every `0.5 s`. The
fully meshed reference uses the same `0.005 s` FVM time step and `0.050 s`
sampling interval. Matching the transient time discretization is required for
the reference comparison. The coupled and reference FVMs use the same pressure
corrector counts. Particle diffusion, global thresholding, and population
control are performed only by VPM GBD.

The FVM-to-VPM hand-off is absolute overlap replacement. Fluid-cell particles
carry `Gamma = cell_volume * FVM_vorticity`; particles inside the transfer box
are replaced each coupling interval and particles outside are left untouched.
`ETA_BLEND_WIDTH = 0` selects the validated hard replacement. A positive value
enables the C1 `eta` state blend at the transfer-box faces. The coupler does not
run an additional remeshing pass: the VPM's normal GBD step regenerates the
particle cloud. Boundary-only panel strengths are refreshed against the current
particle state before their harmonic velocity is used on the FVM boundary.
