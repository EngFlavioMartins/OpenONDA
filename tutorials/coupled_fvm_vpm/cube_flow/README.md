# Native FVM–VPM cube flow

This Re=1000 cube case uses OpenONDA's adaptive Cartesian FVM mesher and the
FVM–VPM coupler. It has no external mesher or repository-path requirement.

After installing OpenONDA, run the configured cube case with:

```sh
./allrun.sh
./allplot.sh
```

`allrun.sh` has no run modes or command-line flags. It runs the `t=20` case
defined in `cube_flow_setup.py`, checks numerical integrity for the complete
run, and applies the strict drag/profile gate wherever its current reference
overlaps (the default cache ends at `t=0.10`). Point `OPENONDA_CUBE_REFERENCE` to a
current full-horizon FVM reference to gate all sample times.
The run removes the previous generated outputs before starting.
The preserved baseline samples remain in `samples_archive/`; the archived
full-horizon reference samples are in `reference_flow/samples/`. Run
`./allplot.sh pdf` when PDF output is required; PNG is the default.

The case uses four partitioned FVM ranks. Its coupled MPI initialization
includes the body-wall geometry gathers used during transfer setup.

The inputs use the wall-commensurate particle spacing `h=0.03125`, an
FVM time-step size of `0.005 s`, and a VPM/coupling time-step size of `0.010 s`.
The coupler therefore advances two FVM substeps per VPM step. All force, line,
and surface samplers use the single `SAMPLING_INTERVAL_TIME` defined in
`cube_flow_setup.py`; FVM and VPM backups are written every `0.5 s`. The
fully meshed reference uses the same `0.005 s` FVM time step and `0.050 s`
sampling interval. FVM visualization and the atomic coupled restart checkpoint
are written every `0.5 s`; the native VPM checkpoint writer is disabled because
it would capture the pre-replacement state. Matching the transient time discretization is required for
the reference comparison. The coupled and reference FVMs use the same pressure
corrector counts. `VPM_VISCOUS_SCHEME` selects CS, RWM, DVH, GBD, or NONE;
this case uses GBD.

The FVM-to-VPM hand-off is an absolute state replacement on the regular VPM
lattice. The coupler maps `Gamma = cell_volume * FVM_vorticity` and the current
replaceable VPM circulation to that same lattice with complete M4' support,
then applies `Gamma_new = eta*Gamma_FVM + (1-eta)*Gamma_VPM`. The C1 transition
band is three particle spacings wide. A lattice Poisson correction removes only
the cross-divergence introduced by the varying `eta`; it preserves net
circulation and is zero when the two states match. Particles in VPM authority remain
persistent; coincident release nodes are merged without creating duplicates.
This mapping is independent of the selected VPM viscous scheme. Boundary-only
panel strengths are refreshed against the current particle state before their
harmonic velocity is used on the FVM boundary.
