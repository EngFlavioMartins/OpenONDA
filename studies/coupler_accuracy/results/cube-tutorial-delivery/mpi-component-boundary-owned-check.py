"""Small 3D MPI lifecycle check; not the delivered-resolution accuracy run."""
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import faulthandler
import numpy as np

root = Path('/Users/flaviomartins/OpenONDA')
case_file = root/'tutorials/coupled_fvm_vpm/02_cube_flow/setup.py'
spec = spec_from_file_location('cube_case', case_file)
case = module_from_spec(spec)
spec.loader.exec_module(case)
case.FVM_MESH = case.msh.coupling_box_mesh(case.FVM_BOX, .125, hole_box=case.CUBE_BOUNDS)
case.COUPLER_SETUP = replace(case.COUPLER_SETUP, transfer_region_bounds=(-1.,1.,-1.,1.,-1.,1.), eta_blend_width=.375)
viscous = case.vpm.ViscousConfig.gbd(kinematic_viscosity=.001, particle_spacing=.0625,
    core_radius_ratio=1., padding=5., threshold_mode='absolute', threshold=.02*.0625**3, max_nodes=100000)
case.VPM_CASE = replace(case.VPM_CASE, numerics=replace(case.VPM_CASE.numerics,
    viscous=viscous, max_n_particles=100000, max_evaluation_points=100000))
m = case.FVM_MESH
patch = m['boundary'][0]
outer = np.unique(m['owners'][patch['start_face']:patch['start_face']+patch['n_faces']])
keep = np.ones(m['n_cells'], dtype=bool)
keep[outer] = False
order = np.r_[np.flatnonzero(keep), outer]
inverse = np.empty_like(order)
inverse[order] = np.arange(len(order))
m['owners'] = inverse[m['owners']].astype(np.int32)
m['neighbours'] = inverse[m['neighbours']].astype(np.int32)
for key in ('cell_vertex_indices', 'cell_type_code'):
    m[key] = m[key][order]
faulthandler.dump_traceback_later(180, exit=True)
case.main(output=root/'studies/coupler_accuracy/results/cube-tutorial-delivery/mpi-component-boundary-owned',
          max_coupling_steps=2, backup_at_stop=True)

faulthandler.cancel_dump_traceback_later()
