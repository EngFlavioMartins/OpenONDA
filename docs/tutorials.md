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

A case owns its `setup.py`, shell launchers, compact inputs under `assets/`, and
generated output. You may move the case directory elsewhere after creation.
With the installed Python environment active:

```bash
cd ring-workspace/tutorials/vpm/vortex_ring
bash allrun.sh --steps 2
```

For individual package-style modules, use the local runner:

```bash
python -m openonda.tutorial_runner . setup --variant dns_direct --steps 2
python -m openonda.tutorial_runner . assets.postprocess --available
```

This loads the edited local case, including its relative imports. It also
works with an absolute case-directory argument from another working directory.
`python setup.py` works for the directly executable setups. Running
`python -m tutorials.…` addresses installed templates and is unsuitable for
editing a materialized case. No package directory needs to be added to Python's
search path. Bash is needed for shell campaigns; Python modules can be run
individually. `openonda tutorial` keeps the CLI's interpreter for subprocesses.

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
| VPM/VLM | `delta_wing`, `flat_plate`, `lamb_oseen_vortex`, `quadcopter`, `rotor_flow`, `vortex_interactions`, `vortex_ring` | CPU or supported GPU backends; geometry inputs are included. OpenVSP is optional for regenerating rotor inputs. |
| Hybrid | `uniform_flow`, `cube_flow`, `cylinder_shedding_flow`, `naca4412_flow` | Uniform flow is a small CPU interoperability example. Body-flow campaigns have separate accuracy and resolution requirements. |
| Hybrid references | `cube_flow/reference_flow`, `cylinder_shedding_flow/reference_flow` | Standalone FVM reference studies; each owns its geometry input. |

Names are prefixed with `fvm/`, `vpm/` or `coupled_fvm_vpm/` as appropriate.
Nested reference studies are available independently. Plotting support and its
font are installed under `openonda.plotting`; cases do not load files from a
repository `docs/` directory.

Core installation does not include external OpenFOAM/cfMesh executables,
OpenVSP, or ParaView. See [optional tool requirements](installation.md).
The [repository audit](../repository_audit.md) records executed examples,
measured errors and verification limits.
