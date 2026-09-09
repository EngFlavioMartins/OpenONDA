# Uniform coupled flow

A small CPU example of native FVM–VPM coupling: two particle steps each contain
three finite-volume steps. The exact velocity is `(1, 0, 0)` everywhere. With
zero vorticity, the particle population remains empty. This tests coupling,
subcycling, output and backup construction; it is not a wake-accuracy benchmark.

After installing OpenONDA, run `./allrun.sh` from this folder or launch
`openonda tutorial run coupled_fvm_vpm/uniform_flow` from any directory.
The solvers write their standard metadata and coupled backups. Exact-solution
checks belong to the coupling regression tests. `./allclean.sh` removes this
case's generated output.
