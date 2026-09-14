# Tutorial runtime audit — 14 September 2026

The tutorial collection now delegates MPI launch, thread limits, mesh ownership,
logging, analyses and output to the solver library. This is an execution and
output cleanup, not a new accuracy or performance benchmark of the cube flow.
No production simulation was run or existing simulation result changed during
this audit. Tests used temporary workspaces and small numerical cases.

## User entry points

All 21 `setup.py` entry points have default arguments. With OpenONDA installed
in the active Python environment, run `python setup.py` in the case directory.
All 21 `allrun.sh` and 18 `allplot.sh` scripts resolve their own directory and
call Python directly. They contain no MPI commands, interpreter selection,
import-path exports or thread/workspace tuning. PNG remains the plotting
default; `./allplot.sh pdf` forwards PDF selection to the figure generators.
Existing case-specific cleaning and scientific run variants remain explicit.

Runtime rank checks have also been removed from the older cube diagnostic and
trial scripts. Optional ParaView/TeX executable discovery belongs to the
library, rather than platform-specific paths in tutorial assets.

## Library changes

- FVM builds and exports its input mesh once. Both partitioned and replicated
  runs receive the appropriate mesh representation. The public `FVMCase`
  constructor also starts its runtime internally.
- The configuration-based coupled factory constructs VPM only on its owner
  and closes the coupled resources on success or failure. All coupled tutorials
  use this construction path.
- Mesh, owner-only particle work, boundary evaluation, transfer, application
  analyses and reporting exchange local failures before their next collective
  operation. Tested failures terminate both ranks instead of leaving a worker
  blocked. This does not promise recovery from failures inside MPI itself.
- `FVMSolver.evaluate` supplies complete, detached global fields to a callback
  that runs once. `write_csv` writes application tables once. Airfoil pressure,
  boundary-layer profiles, step reattachment and Taylor–Green histories use
  these APIs. Normal numerical steps do not gather a full analysis snapshot.
- Per-face velocity profiles map from global patch order onto local faces.
  The step tutorial no longer needs to know the partition layout.
- Periodic boundaries and native immersed-body interpolation select the
  supported replicated PETSc layout internally. These cases retain a complete
  mesh on each rank; partitioned periodic adjacency is not yet implemented.
- A serial `spsolve` selection resolves to MPI GMRES with absolute tolerance
  at most `1e-10` and zero relative tolerance. Existing iterative selections,
  their tolerances, and spatial/time discretization are preserved. The resolved
  choices are recorded by the solver.
- Scheduled terminal FVM snapshots and FVM/VPM backups are not written again
  by automatic finalization for the same state. Explicit manual save requests
  still write. Final health-limit sampling skips already-written events.
- MPI BLAS/Numba limits and the particle owner's CPU budget are internal.
  Separate PETSc equation workspaces no longer require a tutorial export.

The cube's mesh spacing, domain, numerical schemes and plotting comparisons
were not changed by this runtime cleanup.

## Verification

| Check | Result and scope |
| --- | --- |
| `tests/tutorials/test_plain_entrypoints.py` | All 21 defaults reach solver construction with isolated Python, no `PYTHONPATH`, and paths containing spaces. This intercepts construction and does not run full tutorial simulations. |
| Shell-launcher test in the same file | Every run/plot launcher works from an unrelated directory, forwards default/explicit PNG/PDF choices, and stops on a failed Python command. The solver/plot commands and cleaners are stubbed. |
| `tests/fvm/test_application_runtime_mpi.py` | Six real two-rank cases pass: partitioned, replicated, public case, immersed body, mesh failure and analysis failure. A 64-cell 3D mesh is built once; global analyses, one CSV writer, one log/index and one terminal backup are checked. |
| `tests/coupler/test_factory_mpi.py` | Four real two-rank cases pass: coupled advancement, VPM construction failure, accepted-state health failure and application callback failure. Checks one VPM owner, one callback execution, shared results and resource cleanup. |
| `tests/tutorials/test_parallel_fvm_analyses.py` | Tiny copied Taylor–Green and step cases pass in serial and MPI. All exported CSV columns agree at `atol=1e-7`, `rtol=1e-6`, using equally tightened linear tolerances in the test only. |
| Lifecycle/runtime/configuration regression group | 71 tests pass across VPM lifecycle, FVM restart/diagnostics, coupled factory/cube setup and thread runtime. Includes actual CPU VPM backup write counts. |
| Static checks | Selected changed files pass Ruff; all 156 tutorial Python files parse; changed runtime modules compile; `git diff --check` passes. Fresh isolated imports resolve the working checkout. |

The broader mesh/configuration run had 45 passes and one separate failure:
`test_geometry_independence_acceptance_matrix[two_disjoint_bodies]` raises
`ValueError: Box wall face has no unambiguous Cartesian surface plane` in the
unchanged Cartesian mesher. That geometry failure remains unresolved; the
complete project suite is not claimed to pass.

Full-duration numerical stability, reference agreement and throughput were not
remeasured. This audit does not supersede the earlier cube accuracy or timing
measurements.

## FVM MPI recheck — 15 September 2026

Inspected all 12 setups that use FVM: seven standalone tutorials, three coupled
tutorials and both standalone reference cases. Their setup files, shell
launchers and assets contain no MPI imports, launcher commands, communicator
or rank checks, explicit MPI/PETSc execution selection, or runtime exports.
`cores` remains the optional CPU allocation; the library resolves execution.

Found and fixed one shared-runtime edge case: `SLURM_NTASKS` describes a resource
allocation, which must not be mistaken for an already-launched MPI world.
FVM and the launch helper now use one detector based on MPI launcher variables.
The real four-rank check inherits a simulated 64-task allocation and verifies
that the library launches four workers and writes one mesh, log/index, analysis
table and scheduled terminal backup. This is a local Open MPI check, not a
qualification of every cluster scheduler.

Revalidation completed with 17 passes:

- `tests/test_runtime.py`: four tests.
- `tests/fvm/test_application_runtime_mpi.py`: seven real MPI cases, including
  the new four-rank launch.
- `tests/coupler/test_factory_mpi.py`: four real MPI cases.
- The tutorial MPI-encapsulation guard and shell-launcher check: two tests.

The new encapsulation guard checks FVM and coupled tutorial assets as well as
their entry points, so moving a workaround into an asset does not bypass it.
Selected Ruff checks and `git diff --check` also pass. All numerical checks
use small temporary cases; no production simulation or saved result was changed.

## Commit audit — 15 September 2026

The final focused regression selection covered 122 runtime, lifecycle, particle
state, time-step, coupled factory, cube comparison and VPM tutorial tests. The
first pass had 121 successes and one outdated rotor launcher text assertion;
that assertion was updated for the case-local directory change and passed on
recheck. All 23 default-entry-point, shell-launcher and FVM MPI-ownership tests
also passed. These entry-point checks stop at solver construction; they do not
run the full tutorial simulations. The real two/four-rank evidence is recorded
in the preceding section.

Changed Python files pass Ruff lint and formatting. The configured Vulture
check and Bandit medium/high-severity check also pass. Generated comparison
CSV line endings now use LF without changing any numeric values. Finder
metadata remains ignored even inside versioned figure directories.

The collection-wide legacy `test_tutorial_style.py` still fails because the
pending rotor setup contains explicit CLI/output validation. The same blanket
rule also conflicts with the existing surface-interaction result validation;
its shell whitelist predates the optional cube mesh-preserving cleaner. These
pre-existing style-policy discrepancies and the separate two-body mesher
failure above are not evidence of an MPI launch failure. They remain open;
the complete project suite is not claimed to pass.
