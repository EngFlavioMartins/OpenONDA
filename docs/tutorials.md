# Running tutorials

Install OpenONDA first, then use `openonda tutorial list` to see the catalog.
The command copies installed templates into a writable workspace so that
editing a case or running a solver never changes `site-packages`.

```bash
openonda tutorial run fvm/taylor_green --workspace ./first-flow
openonda tutorial create vpm/vortex_ring ./ring-workspace
```

The second command only creates a case. Read its README and edit its parameters
before running a large campaign. Recreating an existing case is rejected to
protect user edits; `run` reuses an existing workspace.

## Running a local copy

After installation, open a case directory and run Python directly:

```bash
cd ring-workspace/tutorials/vpm/02_vortex_ring
python setup.py
```

Edit the inputs in `setup.py` to define your experiment; for a short ring check,
reduce `N_STEPS` there before running. The shell files expose three simple
actions:

```bash
./allrun.sh    # Run the simulations listed in the file.
./allplot.sh   # Plot their results as PNG (the default in every tutorial).
./allplot.sh pdf  # Export PDF figures instead; explicit png is also accepted.
./allclean.sh  # Remove generated output when you choose to.
```

Most run scripts leave existing outputs in place. A case that requires fresh
outputs, such as vortex interactions, calls `allclean.sh` explicitly at the
start of `allrun.sh`. Running does not invoke plotting or result validation.
Every setup has a default case. The reference defaults are the cylinder's
medium grid (`dx=0.04`) and the cube's fine grid (`dx=0.06`). Variant and grid
arguments remain available for comparisons. Every shell launcher resolves its
own case directory, so it also works when invoked from elsewhere.
No interpreter variables or module-runner commands are required.

FVM construction uses `FVMSetup.cores` to start its runtime, limit numerical
thread pools, build the mesh once, and own logging and outputs. Coupled cases
pass the FVM setup and VPM case directly to `coupler.create_coupler`; the library
constructs one VPM instance and closes both solvers. Immersed bodies can be
passed to the factory, which selects their compatible replicated FVM layout.
Periodic cases also use this layout until partitioned periodic adjacency is
supported; their linear systems still solve across the configured MPI ranks.
No tutorial checks MPI ranks or launches MPI itself.
The optional `cores` count is the only parallel setting needed in a case;
MPI launch commands, communicators and PETSc execution modes stay in the
library. An allocated cluster task count alone does not imply MPI is running.
An explicitly serial `spsolve` request resolves to distributed GMRES under MPI,
with zero relative tolerance and absolute tolerance at most `1e-10`; the runtime
logs that choice. Existing iterative solver settings and discretization schemes
are preserved. Per-face boundary profiles are mapped to their owning partitions.

For a custom analysis, `solver.evaluate(function, ...)` supplies a detached
`AnalysisSnapshot` containing every global cell and physical boundary face.
The function runs once and can write its analysis files; its result is shared
with the running application. `solver.write_csv(...)` also has one writer.
Normal solver steps do not gather these fields unless an analysis requests them.

The installed CLI performs the same separate actions from any directory:

```bash
openonda tutorial plot fvm/taylor_green --workspace ./first-flow
openonda tutorial clean fvm/taylor_green --workspace ./first-flow
```

Small comparisons expose only their meaningful variant arguments. For example:

```bash
# In the Lamb–Oseen case:
python setup.py vortex CS
python assets/rwm_ensemble.py vortex --number-of-realizations 10 --converge

# In the vortex-interactions case:
python setup.py baseline
python setup.py stretching_viscosity
python setup.py p_moments
```

The vortex-interactions comparison has one LES baseline and two stabilization
methods. Historical individual research trials remain in `assets/`.

Validators are explicit, for example `python assets/postprocess.py --available`
in the ring case. They and the plotters read the solvers' `vpm_metadata.json`
or `fvm_metadata.json`, rather than a second tutorial metadata file.

See the [tutorial style guide](development/tutorial_style.md) when adding a case.

## VPM induction and stretching

Choose the induction backend and stretching formulation independently. All
three backends accept `stretching_scheme="direct"`, `"transposed"`, or `"mixed"`:

```python
from openonda import vpm

vpm.DirectInduction(stretching_scheme="transposed")
vpm.TreecodeInduction(stretching_scheme="transposed")
vpm.FMMInduction(stretching_scheme="transposed")

numerics = vpm.Numerics(induction=vpm.FMMInduction(stretching_scheme="mixed"))
```

For the velocity Jacobian `J` and particle strength `Γ`, direct stretching is
`J @ Γ`, transposed is `J.T @ Γ`, and mixed is `0.5 * (J + J.T) @ Γ`.
The chosen backend evaluates that same formulation: direct summation over
particle pairs, a treecode traversal, or FMM expansions and near-field pairs.
Accelerated results approximate the same equations to their numerical accuracy.

The default remains transposed for every backend. Output records the backend
and `stretching_scheme` separately, with uppercase values. Restart checks
include the formulation; older direct/FMM checkpoints without this field retain
their implicit transposed meaning.

## Catalog and scope

All catalog cases were inspected for source-root and machine-specific paths,
materialized with their inputs, and checked for shell syntax and local module
imports. This is a portability check, not evidence that every default research
campaign converges. Full physical qualification can require long runs.

| Family | Cases | Runtime notes |
| --- | --- | --- |
| FVM | `cartesian_mesher`, `airfoil_flow`, `boundary_layer`, `cube_flow`, `cylinder_ibm`, `step_profile`, `taylor_green` | Native meshers and solver. Taylor–Green is the short first example. |
| VPM/VLM | `delta_wing`, `flat_plate`, `lamb_oseen_vortex`, `quadcopter`, `rotor_flow`, `surface_interaction`, `vortex_interactions`, `vortex_ring` | CPU or supported GPU backends; geometry inputs are included. OpenVSP is optional for regenerating rotor inputs. The surface-interaction case is a real VPM vortex-ring/tandem-VLM run with owner-emitted loads, leakage, events, and restart/refinement study tables; its scope remains inviscid attached-flow self-consistency. |
| Hybrid | `cube_flow`, `cylinder_shedding_flow`, `naca4412_flow` | Small-to-medium CPU interoperability examples. Body-flow campaigns have separate accuracy and resolution requirements. |
| Hybrid references | `cube_flow/reference_flow`, `cylinder_shedding_flow/reference_flow` | Standalone FVM reference studies; each owns its geometry input. |

Names are prefixed with `fvm/`, `vpm/` or `coupled_fvm_vpm/` as appropriate.
Nested reference studies are available independently. Plotting support and its
font are installed under `openonda.plotting`; cases do not load files from a
repository `docs/` directory.

Core installation does not include external OpenFOAM/cfMesh executables,
OpenVSP, or ParaView. See [optional tool requirements](installation.md).
The [FVM qualification report](validation/fvm_qualification.md) records the
current executable contract and verification limits. The broader documentation
scope and terminology are summarized in [the documentation audit](documentation_audit.md).
