# Flow past a cylinder at Re = 150

A body-fitted finite-volume region resolves the cylinder wall and transfers
vorticity to the surrounding vortex-particle domain. The diameter and
freestream speed are 1 m and 1 m/s, giving kinematic viscosity 1/150 m²/s.
The compact FVM box is
`[-1.48, 1.48] × [-1.48, 1.48] × [-0.48, 0.48] m`. It uses the reference
fine spacing `h = 0.04 m` uniformly in every direction, giving 24 cells across
the resolved span. There are no mesh-refinement or coarsening regions. Force
coefficients use the resolved 0.96 m span.

Coupled runs end at 100 s, retaining volume fields and coupled checkpoints
every 0.24 s (six 0.04 s coupling steps). This preserves the tested time
discretization; an exact 0.25 s event would fall between accepted coupled
states.

`setup.py` shows the coupled configuration. Use the installed OpenONDA
environment and inspect its physical inputs before running `python setup.py`.
`allrun.sh` first removes that tutorial's generated output, then starts this
configuration.

The [fully meshed reference](reference_flow/README.md) specifies a four-level
geometric XY refinement family. Its slip span boundaries describe a
span-invariant cylinder flow without endcaps.
