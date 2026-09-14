This first prefix analysis is superseded by
`../long-wake-prefix-through-four-qualified/long-wake-prefix-verification-3d.json`.

The first analysis assembled the composite hybrid profile in an array copied
from the float32 VPM query, inadvertently rounding the float64 FVM samples.
The corrected verifier explicitly creates a float64 composite and checks that
its FVM entries remain bitwise equal to the saved FVM profile. Across all
composite metrics the largest change is `2.8405821243804308e-8` in normalized
velocity units. The force records, comparison history, separate FVM and VPM
profile metrics, and direct-kernel checks are unchanged.

The first analysis and its frozen source remain intact for provenance. Use
`verify_accepted_wake_prefix_3d.py` for subsequent prefix verification.
