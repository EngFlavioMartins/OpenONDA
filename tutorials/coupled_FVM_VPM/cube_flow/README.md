# Native FVM–VPM cube flow

This Re=1000 cube case uses OpenONDA's adaptive Cartesian FVM mesher and the
FVM–VPM coupler. It has no external mesher or repository-path requirement.

After installing OpenONDA, run the production case with:

```sh
./allrun.sh
./allplot.sh
```

`allrun.sh` moves any existing `samples/` into a timestamped `run_backups/`
entry before cleaning. The preserved baseline hybrid samples remain in
`samples_backup/`; the fully meshed reference samples are in
`reference_flow/samples/`.

The production case uses four partitioned FVM ranks. Its coupled MPI smoke
test includes the body-wall geometry gathers used during transfer setup.

Production inputs use the wall-commensurate particle spacing `h=0.03125`, FVM
time step `0.01`, and VPM/coupling interval `0.03` (three FVM substeps). The
nearest complete endpoint to 20 s is `20.01 s`. FVM volumes and VPM/coupled
restart states are scheduled every 1 s; lines/forces are sampled every 0.05 s
and slices every 0.10 s, matching `reference_flow`. Particle diffusion, global
thresholding, and population control are performed only by VPM GBD.
