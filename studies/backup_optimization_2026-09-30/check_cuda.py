import json
from pathlib import Path
import sys
import numpy as np
import taichi as ti
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'frozen'))
from source.solvers.vpm.kernels.gaussian import create_gaussian_kernels
from source.solvers.vpm.numerics.kernels_common import _make_compute_vorticities_kernel
from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode

ti.init(arch=ti.cuda,enable_fallback=False,default_fp=ti.f32,device_memory_GB=0.25,offline_cache_file_path=str(root/'cuda-cache'))
count=3073
rng=np.random.default_rng(114)
position=ti.Vector.field(3,ti.f32,shape=count)
strength=ti.Vector.field(3,ti.f32,shape=count)
radius=ti.field(ti.f32,shape=count)
output=ti.Vector.field(3,ti.f32,shape=count)
xyz=rng.uniform(-1,1,(count,3)).astype('f4');xyz[:2]=0
radii=rng.uniform(0.005,0.07,count).astype('f4');radii[:3]=[0.02,0.2,0.5]
position.from_numpy(xyz);strength.from_numpy(rng.normal(size=(count,3)).astype('f4')*0.001);radius.from_numpy(radii)
direct=_make_compute_vorticities_kernel(create_gaussian_kernels(ti.f32)['zeta_'])
direct(position,strength,radius,output,count)
expected=output.to_numpy()
tree=TaichiTreecode(max_n_particles=count,max_nodes=2*count,kernel_type='GAUSSIAN',hierarchy_only=True,max_evaluation_points=1024,device_sort_only=True)
tree.build(position,strength,radius,count)
tree.compute_gaussian_particle_vorticity(output,count)
actual=output.to_numpy()
np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=2e-5)
report={'device':str(ti.lang.impl.current_cfg().arch),'particles':count,'relative_l2_error':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),'max_absolute_error':float(np.max(abs(actual-expected)))}
(root/'cuda_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
