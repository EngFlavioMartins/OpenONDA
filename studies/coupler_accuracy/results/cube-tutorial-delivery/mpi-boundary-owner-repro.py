"""Small 3D reproducer: only the final partition owns coupling faces."""
from dataclasses import replace
import faulthandler
from pathlib import Path
import runpy
import sys

import numpy as np
from openonda.runtime import RunConfig

root = Path('/Users/flaviomartins/OpenONDA')
audit = root/'studies/coupler_accuracy/results/cube-tutorial-delivery'
case = runpy.run_path(str(root/'tutorials/coupled_fvm_vpm/02_cube_flow/setup.py'))
RunConfig(cpu_cores=4, parallel_mode='mpi').ensure_runtime(sys.argv[0])
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
trace = (audit/f'mpi-boundary-owner-rank-{rank}.log').open('w')
faulthandler.enable(trace)

def mesh():
    m = case['msh'].coupling_box_mesh(case['FVM_BOX'], .125, hole_box=case['CUBE_BOUNDS'])
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
    return m

solver = case['fvm'].create_fvm_solver(
    replace(case['FVM_SETUP'], samplers=()), case_dir=audit/'mpi-boundary-owner', mesh=mesh)
normals = solver.get_boundary_face_normal('numericalBoundary')
solver.set_normal_velocity_tangential_gradient_boundary_condition(
    normals[:, 0], np.zeros_like(normals), 'numericalBoundary')
solver.set_flux_consistent_pressure_boundary_condition('numericalBoundary')
faulthandler.dump_traceback_later(30, exit=True, file=trace)
solver.solve_pimple()
solver.advance_time()
faulthandler.cancel_dump_traceback_later()
print(f'RANK {rank} PASS step={solver.step}', flush=True)
solver.close()
trace.close()
