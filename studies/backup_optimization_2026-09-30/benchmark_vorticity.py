"""Compare unchanged direct density sum with the new hierarchy on a saved cloud."""
import os
import sys
from pathlib import Path
from time import perf_counter
import hashlib
import json
import resource

import h5py
import numpy as np
import taichi as ti

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'frozen'))
from source.solvers.vpm.kernels.gaussian import create_gaussian_kernels
from source.solvers.vpm.numerics.kernels_common import _make_compute_vorticities_kernel
from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode

checkpoint = Path('/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/05_delta_wing/solution/vpm/vpm_002780.h5')
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
with h5py.File(checkpoint, 'r') as handle:
    arrays = {name: handle['particles'][name][:] for name in ('position', 'vortex_strength', 'core_radius')}
count = len(arrays['position'])
ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=2, offline_cache_file_path=str(root / 'ti-cache'))
position = ti.Vector.field(3, ti.f32, shape=count)
strength = ti.Vector.field(3, ti.f32, shape=count)
radius = ti.field(ti.f32, shape=count)
output = ti.Vector.field(3, ti.f32, shape=count)
for field, name in ((position, 'position'), (strength, 'vortex_strength'), (radius, 'core_radius')):
    field.from_numpy(arrays[name])
direct = _make_compute_vorticities_kernel(create_gaussian_kernels(ti.f32)['zeta_'])
tree = TaichiTreecode(max_n_particles=count, max_nodes=2*count, kernel_type='GAUSSIAN', hierarchy_only=True, max_evaluation_points=4096)
report = {'checkpoint': str(checkpoint), 'sha256': digest, 'particles': count, 'cpu_threads': 2, 'seconds': {}}
def measure(label, function):
    start = perf_counter()
    function()
    ti.sync()
    report['seconds'][label] = perf_counter() - start
    print(label, report['seconds'][label], flush=True)
def hierarchical():
    tree.build(position, strength, radius, count)
    tree.compute_gaussian_particle_vorticity(output, count)
measure('direct_compile_and_128_targets', lambda: direct(position, strength, radius, output, 128))
measure('direct_full', lambda: direct(position, strength, radius, output, count))
expected = output.to_numpy()
measure('hierarchy_cold', hierarchical)
measure('hierarchy_warm', hierarchical)
actual = output.to_numpy()
error = actual.astype(float) - expected
report['relative_l2_error'] = float(np.linalg.norm(error) / np.linalg.norm(expected))
report['max_absolute_error'] = float(abs(error).max())
report['reference_max_absolute'] = float(abs(expected).max())
report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
report['sha256_after'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert report['relative_l2_error'] < 1e-5
assert report['sha256_after'] == digest
(root / 'vorticity_results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
