# Rejected directional-outflow pressure pairing

This production-resolution 0.05 s gate tested commit `aeb04b7` on the compact
FVM box.  The velocity switch was fixed geometrically to the downstream face,
but the existing `freestream` pressure partner imposed zero-gradient pressure
on the other five faces.

The run was stable but rejected immediately: the first handoff injected
175,960 particles versus 56,982 for the accepted compact Dirichlet baseline,
and Cd was distorted to 0.3960.  This is much smaller than the 397,999-particle
failure produced by flux-based switching on all six faces, confirming that the
geometric velocity mask works.  The next gate retains `fixedFluxPressure` on
the merged patch and changes only the downstream velocity treatment.

