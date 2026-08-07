"""FVM sampling package: forces, line/surface field samplers and post-processing.

All samplers share the :class:`~.base.Sampler` abstraction and write
exclusively to ``<case_root>/samples/``.  Offline post-processing
(:class:`~.postprocess.PostProcess`) drives the same samplers over archived
snapshots, so live and archived output are byte-for-byte comparable.
"""

from .base import SAMPLER_CSV_COLUMNS, Sampler, SamplingSchedule, samples_dir
from .executor import FVMSamplerExecutor
from .fields import LineSampler, SurfaceSampler
from .forces import ForceSampler, IBMForceSampler, YPlusSampler
from .postprocess import PostProcess, SnapshotContext

__all__ = [
    "SAMPLER_CSV_COLUMNS",
    "ForceSampler",
    "FVMSamplerExecutor",
    "IBMForceSampler",
    "LineSampler",
    "PostProcess",
    "Sampler",
    "SamplingSchedule",
    "SnapshotContext",
    "SurfaceSampler",
    "YPlusSampler",
    "samples_dir",
]
