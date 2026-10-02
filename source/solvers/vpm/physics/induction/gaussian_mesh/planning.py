"""Shape-only finite-field execution plans; never an accuracy admission.

V is the padded real-grid volume, S the RFFT half-spectrum and L the retained
logical grid. The full path keeps six source spectra, twelve result spectra,
one temporary spectrum plus spare FFT capacity, nine radial grids and one
inverse grid. Streaming retains either six or three source spectra, B result
spectra, a temporary spectrum and two spare spectra, K partially fused radial
grids and one inverse grid. All paths retain the SAME twelve logical outputs.

The particle/stencil estimate deliberately remains conservative, including
both source families and all admitted initial queries even when temporaries
are not simultaneously live. Plan workspaces have a separate unchanged cap.
Pool fragmentation and cuFFT/context allocations still need runtime guards;
this arithmetic does not claim that every otherwise admitted CUDA allocation
will succeed. No numerical parameter participates in execution-plan choice.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FieldExecutionPlan:
    mode: str
    output_batch: int
    source_families: int
    kernel_channels: int
    payload_bytes: int
    metadata_bytes: int
    radial_grid_passes: int
    kernel_forward_transforms: int
    inverse_transforms: int


def required_channels(columns):
    """Unique Gaussian derivative channels used by explicit curl columns."""
    derivatives = (((0, -1),), ((1, -1),), ((2, -1),),
                   ((0, 0),), ((0, 1), (1, 0)), ((0, 2), (2, 0)),
                   ((1, 1),), ((1, 2), (2, 1)), ((2, 2),))
    needed = []
    for channel, pairs in enumerate(derivatives):
        for axis, derivative in pairs:
            for row, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
                column = row if derivative < 0 else 3+3*row+derivative
                if axis in (a, b) and column in columns and channel not in needed:
                    needed.append(channel)
    return tuple(needed)


def field_execution_plan(shape, fft_shape, source_count, target_count, order,
                         itemsize, pool_cap, plan_cap):
    v, logical = math.prod(fft_shape), math.prod(shape)
    spectrum = fft_shape[0]*fft_shape[1]*(fft_shape[2]//2+1)
    metadata = ((2*source_count+target_count)*(12+3*order*itemsize)
                +(source_count+target_count)*24+target_count*12*itemsize)
    compact = 12*logical*itemsize
    full = (40*spectrum+10*v)*itemsize+compact+metadata
    available = pool_cap-plan_cap
    if full <= available:
        return FieldExecutionPlan("all_channels", 12, 2, 9, full, metadata, 2, 18, 12)
    # The expensive work is the radial finite-image sum, not just the number
    # of result spectra. Jointly choose output and radial-channel batches.
    # Count radial grid passes first, then channel FFTs and inverses; these
    # counts are deterministic conservative two-family work, not a time claim.
    candidates = []
    for families in (2, 1):
        for batch in range(12, 0, -1):
            spectra = 3*families+batch+3
            counts = [len(required_channels(range(first, min(first+batch, 12))))
                      for first in range(0, 12, batch)]
            for kernels in range(1, 10):
                payload = (2*spectra*spectrum+(kernels+1)*v)*itemsize+compact+metadata
                if payload <= available:
                    passes = 2*sum(math.ceil(count/kernels) for count in counts)
                    transforms, inverse = 2*sum(counts), 12 if families == 2 else 24
                    plan = FieldExecutionPlan("streamed", batch, families, kernels,
                                              payload, metadata, passes, transforms, inverse)
                    candidates.append(((passes, transforms, inverse, -families, payload), plan))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    raise MemoryError("finite field payload exceeds smooth pool admission even with streaming")


def device_field_execution_plan(shape, fft_shape, source_count, target_count, order,
                                itemsize, pool_cap, plan_cap, correction_cap, free_bytes):
    """Select exact execution batches within both policy and live-device bounds.

    Other owners (including the primary particle operator) legitimately occupy
    the same device. A configured maximum is not available memory. Reserve the
    complete unchanged correction cap; FFT work remains inside the smooth pool
    and its separate plan cap. Smaller batches change storage, not the grid,
    precision, physical sources, images, or mathematical error budgets.
    """
    effective_cap = min(pool_cap, free_bytes-correction_cap)
    try:
        plan = field_execution_plan(shape, fft_shape, source_count, target_count,
                                    order, itemsize, effective_cap, plan_cap)
    except MemoryError as error:
        raise MemoryError("insufficient free device memory for exact Gaussian field execution: "
                          f"free={free_bytes}, correction_reserve={correction_cap}, "
                          f"smooth_cap={pool_cap}, available_smooth={effective_cap}, "
                          f"plan_cap={plan_cap}, fft_shape={fft_shape}") from error
    return plan, effective_cap


def query_batch_size(target_count, order, itemsize, correction_itemsize, *,
                     smooth_cap, smooth_used, correction_cap, correction_used,
                     device_available):
    """Fit fresh query scratch around retained fields and one complete output.

    Queries can grow after the original field was built. Their coordinates,
    stencils and local corrections need not coexist for the entire query.
    The public device output does, and is admitted before any batch runs.
    The caller supplies current owned-pool usage and driver free memory plus
    its own unused cached blocks; unrelated cached allocations are excluded.
    Fixed reserves cover allocation rounding and scalar/reduction scratch.
    Allocation failures can still require a smaller runtime batch.
    """
    if target_count == 0:
        return 0
    output = ((12*target_count*itemsize+511)//512)*512
    smooth_fixed, correction_fixed = 4096, 8192
    # Includes the host-coordinate transfer, complete stencil, and a bounded
    # isfinite check. Correction input/results and their finite check use
    # their independently selected arithmetic precision.
    smooth_per_target = 24+12+3*order*itemsize+12
    correction_per_target = 24+12*correction_itemsize+12
    smooth = smooth_cap-smooth_used-output-smooth_fixed
    correction = correction_cap-correction_used-correction_fixed
    device = device_available-output-smooth_fixed-correction_fixed
    batch = min(target_count, smooth//smooth_per_target,
                correction//correction_per_target,
                device//(smooth_per_target+correction_per_target))
    if batch < 1:
        raise MemoryError("insufficient memory for complete Gaussian query output and one exact batch: "
                          f"targets={target_count}, output_bytes={output}, "
                          f"smooth_available={smooth}, correction_available={correction}, "
                          f"device_available={device}")
    return int(batch)
