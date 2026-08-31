"""Field samplers for VPM output.

``SurfaceSampler`` and ``LineSampler`` sample VPM-induced fields (velocity,
vorticity, strain, velocity gradients) onto fixed grid points or line points.
The canonical CSV schema is ``SAMPLER_CSV_COLUMNS``; ``resolve_samples_dir``
fixed the output root from a backup directory.
"""

from .field_samplers import (
    SAMPLER_CSV_COLUMNS,
    LineSampler,
    SurfaceSampler,
    resolve_samples_dir,
    sampler_csv_columns,
)
from .schedule import SamplingSchedule

__all__ = [
    "SAMPLER_CSV_COLUMNS",
    "LineSampler",
    "SamplingSchedule",
    "SurfaceSampler",
    "resolve_samples_dir",
    "sampler_csv_columns",
]
