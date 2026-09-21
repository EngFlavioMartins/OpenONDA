# FVM tests and limitations

Run the FVM, coupling, and public-API checks from the repository root:

```bash
python -m pytest tests/fvm tests/coupler tests/test_public_api.py
```

Some tests need MPI/PETSc or optional meshing tools. See the
[test guide](../../tests/README.md) for test selection and prerequisites.

## Focused checks

| Question | Test file |
| --- | --- |
| Does a saved state restart with the same fields and time history? | `tests/fvm/test_restart_and_diagnostics.py` |
| Does time-step adjustment respect CFL and output times? | `tests/fvm/test_time_step_control.py` |
| Are gradients accurate on manufactured fields? | `tests/fvm/test_manufactured_gradient_qualification.py` |
| Is curl evaluated with the correct component convention? | `tests/fvm/test_vorticity_analytic.py` |
| Are non-orthogonal pressure corrections applied consistently? | `tests/fvm/test_nonorthogonal_pressure_correction.py` |
| Does FVM-to-VPM interpolation preserve affine fields and converge under refinement? | `tests/coupler/test_interpolation_qualification.py` |
| Do renewal and pruning preserve their circulation and impulse budgets? | `tests/coupler/test_stable_renewal.py` |
| Do interface sweeps restore the predictor without advancing time twice? | `tests/coupler/test_interface_iteration.py` |

These checks exercise components and small cases. A new flow still needs its
own grid/time-step study, convergence of iterative solves, and an appropriate
analytical, experimental, or independently computed reference. Passing the
test suite alone does not establish those results.

## Current limits

The default FVM path is CPU, float64, with NumPy/SciPy. Alternative operator
backends are not necessarily faster; benchmark the complete case. MPI/PETSc
supports replicated and partitioned meshes, but periodic patches currently
need the replicated layout. All ranks must enter collective solves, field
queries, output, and restart calls in the same order.

Dynamic/ALE meshes, moving immersed bodies, compressible flow, and multiphase
flow are not implemented. GPU acceleration of VPM does not imply GPU FVM.
See the [solver guide](../fvm.md) for the configuration and field conventions.
