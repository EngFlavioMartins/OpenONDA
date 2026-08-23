"""Exact shared output cadence for the cube-flow comparison.

Edit the two solver time steps and the three desired output intervals below.
They are rounded once to a whole coupling step, so every derived output time
remains a state accepted by the coupled FVM, VPM, and reference FVM runs.
"""

from __future__ import annotations

import math


# Solver clocks
FVM_TIME_STEP_SIZE = 0.01
VPM_TIME_STEP_SIZE = 0.03
REQUESTED_END_TIME = 20.0

# Desired output cadence in seconds. Actual cadence is the closest coupling step.
LINE_SAMPLE_TARGET_INTERVAL_TIME = 0.05
SLICE_SAMPLE_TARGET_INTERVAL_TIME = 0.10
BACKUP_TARGET_INTERVAL_TIME = 1.00

if FVM_TIME_STEP_SIZE <= 0.0 or VPM_TIME_STEP_SIZE <= 0.0:
    raise ValueError("FVM_TIME_STEP_SIZE and VPM_TIME_STEP_SIZE must be positive")


def _coupling_steps_for(name: str, target_interval: float) -> int:
    if target_interval <= 0.0:
        raise ValueError(f"{name} must be positive")
    return max(1, math.floor(target_interval / VPM_TIME_STEP_SIZE + 0.5))


LINE_SAMPLE_EVERY_COUPLING_STEPS = _coupling_steps_for(
    "LINE_SAMPLE_TARGET_INTERVAL_TIME", LINE_SAMPLE_TARGET_INTERVAL_TIME
)
SLICE_SAMPLE_EVERY_COUPLING_STEPS = _coupling_steps_for(
    "SLICE_SAMPLE_TARGET_INTERVAL_TIME", SLICE_SAMPLE_TARGET_INTERVAL_TIME
)
BACKUP_EVERY_COUPLING_STEPS = _coupling_steps_for(
    "BACKUP_TARGET_INTERVAL_TIME", BACKUP_TARGET_INTERVAL_TIME
)

_substep_ratio = VPM_TIME_STEP_SIZE / FVM_TIME_STEP_SIZE
FVM_STEPS_PER_COUPLING_STEP = round(_substep_ratio)
if FVM_STEPS_PER_COUPLING_STEP < 1 or not math.isclose(
    _substep_ratio, FVM_STEPS_PER_COUPLING_STEP, rel_tol=0.0, abs_tol=1e-12
):
    raise ValueError(
        "VPM_TIME_STEP_SIZE must be an integer multiple of FVM_TIME_STEP_SIZE "
        "for exact shared samples"
    )

LINE_SAMPLE_EVERY_FVM_STEPS = LINE_SAMPLE_EVERY_COUPLING_STEPS * FVM_STEPS_PER_COUPLING_STEP
SLICE_SAMPLE_EVERY_FVM_STEPS = SLICE_SAMPLE_EVERY_COUPLING_STEPS * FVM_STEPS_PER_COUPLING_STEP
BACKUP_EVERY_FVM_STEPS = BACKUP_EVERY_COUPLING_STEPS * FVM_STEPS_PER_COUPLING_STEP

COUPLING_STEPS = round(REQUESTED_END_TIME / VPM_TIME_STEP_SIZE)
END_TIME = COUPLING_STEPS * VPM_TIME_STEP_SIZE

# Compatibility/readability values for plot labels and reports.
LINE_SAMPLE_INTERVAL_TIME = LINE_SAMPLE_EVERY_COUPLING_STEPS * VPM_TIME_STEP_SIZE
SLICE_SAMPLE_INTERVAL_TIME = SLICE_SAMPLE_EVERY_COUPLING_STEPS * VPM_TIME_STEP_SIZE
BACKUP_INTERVAL_TIME = BACKUP_EVERY_COUPLING_STEPS * VPM_TIME_STEP_SIZE
