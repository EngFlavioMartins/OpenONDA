# Rejected directional-outflow / fixed-flux pairing

This production-resolution 0.05 s gate tested commit `c610a36` on the compact
FVM box.  Only the downstream velocity was extrapolated, while
`fixedFluxPressure` remained active on the complete merged coupling patch.

The pairing is incompatible: the first handoff injected 335,538 particles and
Cd rose to 16.8778.  Together with the preceding zero-gradient-pressure gate,
this demonstrates that a usable downstream condition must switch velocity and
pressure on the same geometric face while retaining fixed-value velocity and
fixed-flux pressure on the other five faces.

