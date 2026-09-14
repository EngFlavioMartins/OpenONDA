# Writing an OpenONDA tutorial

A tutorial is executable documentation. Start with the physical inputs, then
show geometry and resolution, numerical choices, output, construction, and run.
Use `tutorials/vpm/01_lamb_oseen_vortex/setup.py` and
`tutorials/vpm/02_vortex_ring/setup.py` as VPM examples. The
[code guidelines](code_guidelines.md) explain their physics-first structure.

- Group related inputs and add short unit comments. Prefer the names used by
  the public API (`time_step_size`, `kinematic_viscosity`, `particle_spacing`).
- Show derived physical quantities. Omit explicit constructor defaults that
  do not explain the experiment.
- Trust the authored inputs. Solver constructors own input validation;
  `assets/validate_results.py` or tests own expected-result checks. Keep
  `try/except`, assertions, custom CLI validation, and environment checks out
  of `setup.py`.
- Keep physical branching when it explains distinct models or motion. A few
  comparison choices may use `argparse`, with defaults so `python setup.py`
  runs a case; other inputs are edited in Python.
  Keep one active `setup.py` per tutorial; place archival research tools in
  `assets/` when they are still needed for historical reproduction.
- Use the installed public API. Never alter `sys.path` or `PYTHONPATH`.
  Ordinary imports come first: standard library, third party, OpenONDA. A
  case-local `case_package(Path(__file__).parent)` declaration supports local
  relative asset imports for both direct execution and installed workspaces.
- Solver factories own MPI launch, thread limits, logging, and file ownership.
  Pass a mesh builder/configuration to the FVM factory instead of building it
  first. Use the configuration-based coupled factory rather than constructing
  VPM on each process. Custom global field analyses use `solver.evaluate`;
  tables use `solver.write_csv`. Never put rank checks in tutorial code.
  `tests/tutorials/test_plain_entrypoints.py` guards FVM setups, launchers and
  assets against MPI imports, rank checks and runtime environment workarounds.
- Let the solvers write `vpm_metadata.json` and `fvm_metadata.json`. Analysis
  reads those files, including each run's recorded parameters; never generate
  a second tutorial configuration or metadata file.
- Keep geometry generators, analytical comparisons, diagnostics, figures,
  reference data, and validators in `assets/`. Remove unnecessary machinery;
  do not hide it in a helper framework.

The case root normally contains `setup.py`, `allrun.sh`, `allplot.sh`,
`allclean.sh`, and `assets/`. Add a README or another physical setup only when
it helps explain the case. Generated solution, samples, and figures are not
inputs or installation resources.

Anchor each shell launcher to its own directory and list direct Python commands:

```bash
#!/bin/bash -e
cd -- "$(dirname -- "$0")"
python setup.py --variant dns_direct
python setup.py --variant dns_transposed
```

The shebang stops on command failure. Do not add loops, functions, interpreter
variables, logging, or plotting to `allrun.sh`. If the case needs a clean start,
call `allclean.sh` explicitly before the run commands. `allplot.sh` plots native
sampled data; postprocessing those samples is appropriate, while duplicating a
solver sampler or checkpoint extractor is not. Validation is an explicit Python
command. `allclean.sh` anchors deletion to its own directory and only removes
generated outputs.
`openonda tutorial run`, `plot`, and `clean` provide the corresponding actions
from any directory using the active installation.

Figures follow the Lamb–Oseen tutorial and `openonda.plotting.set_thesis_style()`:
Palatino/Pagella text, NewPX mathematics, and the shared font sizes and colors.
Use the shared figure-size presets, or explicit dimensions no wider than
12.5 cm. Stack panels when necessary to keep labels readable at that size.
Save with the shared DPI and `bbox_inches=None` so cropping does not change
the specified physical width. Check both PNGs and rendered PDFs; LaTeX labels
need escaped percent signs and mathematical degree symbols (`^\circ`).

Before merging a tutorial change, compare the resolved physical and numerical
configuration, run meaningful numerical checks, and execute a copied case
outside the repository with the installed package and no `PYTHONPATH`. Exercise
plotting and cleaning there. State whether evidence covers the complete
experiment or a reduced integration check. The collection-wide style guard is
`tests/tutorials/test_tutorial_style.py`; numerical correctness belongs in the
normal solver and tutorial tests.
