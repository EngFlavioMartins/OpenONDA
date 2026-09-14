"""Small 3D MPI lifecycle check; not the delivered-resolution accuracy run."""
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
case.main(output=root/'studies/coupler_accuracy/results/cube-tutorial-delivery/mpi-component',
          max_coupling_steps=2, backup_at_stop=True)
