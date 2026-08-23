# Native FVM–VPM cube flow

This Re=1000 cube case uses OpenONDA's adaptive Cartesian FVM mesher and the
FVM–VPM coupler. It has no external mesher or repository-path requirement.

After installing OpenONDA, run the production case with:

```sh
./allrun.sh
./allplot.sh
```

`allrun.sh` moves the previous `samples/`, `solution/`, and `figures/` into one
timestamped `run_backups/` entry before cleaning, together with the setup file
that produced them. Plot an archived run with
`./allplot.sh png run_backups/<timestamp>`. The preserved baseline hybrid
samples remain in `samples_backup/`; the fully meshed reference samples are in
`reference_flow/samples/`.

The production case uses four partitioned FVM ranks. Its coupled MPI smoke
test includes the body-wall geometry gathers used during transfer setup.

Production inputs use the wall-commensurate particle spacing `h=0.03125`, FVM
time step `0.01`, and VPM/coupling interval `0.03` (three FVM substeps). The
end time is rounded to its nearest complete coupling step. Shared samples and
backups are configured only in
`cube_flow_timing.py`: edit the two time steps and the three desired output
intervals there. The file rounds each interval to the closest coupling step
and derives the matching FVM step count, so the coupled FVM, VPM, and
`reference_flow` always write the same accepted states. After changing it,
rerun both `reference_flow/allrun.sh` and `./allrun.sh` before plotting. The
coupled and reference FVMs use the same pressure corrector counts. Particle
diffusion, global thresholding, and population control are performed only by
VPM GBD.
