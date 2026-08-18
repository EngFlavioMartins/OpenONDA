# Native FVM–VPM cube flow

This Re=1000 cube case uses OpenONDA's adaptive Cartesian FVM mesher and the
FVM–VPM coupler. It has no external mesher or repository-path requirement.

After installing OpenONDA, run the production case with:

```sh
./allrun.sh
./allplot.sh
```

For a short installation, meshing, panel, FVM, VPM, coupling, and conservation
qualification, run:

```sh
OPENONDA_SMOKE=1 ./allrun.sh
```

`OPENONDA_T_END` provides an explicit run-length override. The tutorial uses
the qualified serial FVM-VPM path; partitioned FVM-VPM coupling remains
unqualified until it has a collective regression test.
