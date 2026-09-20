# Flow past a cylinder at Re = 150

A body-fitted finite-volume region resolves the cylinder wall and transfers
vorticity to the surrounding vortex-particle domain. The diameter and
freestream speed are 1 m and 1 m/s, giving kinematic viscosity 1/150 m²/s.
The FVM force coefficients use a one-diameter resolved span. Future coupled
runs end at 100 s, retaining volume fields and coupled checkpoints every 0.24 s
(six 0.04 s coupling steps). This preserves the tested time discretization; an
exact 0.25 s event would fall between accepted coupled states.

`setup.py` shows the coupled configuration with a finite-span panel body.
Use the installed OpenONDA environment and inspect its physical inputs before
running `python setup.py`. `allrun.sh` first removes that tutorial's generated
output, then starts this configuration.

The [fully meshed reference](reference_flow/README.md) specifies a geometric
XY refinement family and independent span/time controls. Its slip span
boundaries describe a span-invariant cylinder flow without endcaps.

The optional [planar coupling study](../../../studies/panel_removal/CYLINDER_EXPERIMENT.md)
uses an explicitly infinite-span VPM model with no panel solver and a larger
FVM region containing the wake-formation region. It is a separate numerical
assessment run from the repository study directory. Its force, wake-profile,
span-invariance and shedding-frequency checks must be assessed before claiming
agreement with the fully meshed reference. This planar model does not represent
finite cylinder ends or three-dimensional wake instabilities.

See [portable runs](../../../studies/panel_removal/PORTABLE_RUNS.md) for external
output directories and restart requirements. Existing running jobs retain their
submitted horizons and output cadence.
