# Rejected per-face mixed outflow gate

This production-resolution 0.05 s gate tested commit `67d06a9` on the compact
FVM box.  The merged patch used the intended per-face pairing: extrapolated
velocity and fixed pressure only on the downstream face, with nonuniform VPM-BC
velocity and momentum-compatible fixed-flux pressure on the other five faces.

The discrete operator passed its unit and PIMPLE tests, but the physical gate
failed: the first handoff injected 175,942 particles and Cd was 0.2117.  This
nearly matches the 175,960-particle zero-gradient-pressure gate, showing that
the startup artifact is dominated by the abrupt downstream velocity switch,
not by the VPM-BC-face pressure formula.

A future experiment should retain Dirichlet coupling initially and introduce
the downstream convective component continuously only after a physical wake
has reached the interface.  The production `allrun.sh` remains Dirichlet.

