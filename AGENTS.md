# OpenONDA — instructions for contributors and AI agents

**Read this document before changing code, tutorials, tests, plots or documentation.**
This is the single development authority. Current explicit user instructions take
precedence; API manuals and case READMEs do not create policy exceptions. Apply
these rules without asking the user to repeat them.

Existing code may violate these rules. Correct issues within scope.

## 1. Non-negotiable rules

- **Physics first:** understand equations, units, assumptions and acceptance criteria.
- **One owner:** configuration declares; solvers own state/clocks; operators compute;
  framework I/O samples, writes and reports.
- **Useful hover documentation:** accurate units, shapes, conventions, effects and examples.
- **One current implementation:** delete replaced code and every obsolete consumer.
  No deprecation period, aliases or compatibility layer.
- **Minimal tutorials:** show the experiment, not infrastructure or development history.
- **Central logging:** use the owner and shared formatter; no scattered prints/rank checks.
- **Thesis figures:** PNG default, PDF option; prescribed colours, fonts, size and margins.
- **Evidence before claims:** a smoke test is not a full run; completion is not convergence.

## 2. Working procedure

Read interfaces, implementations, callers and tests with `rg`. Preserve unrelated
work; coordinate overlaps. Identify physics, owners, schemas and verification.
Tool configuration: [pyproject.toml](pyproject.toml).

Act autonomously within scope; ask only for missing physics or irreversible choices.
Respect user-run simulations. Apply section 13; report evidence, limits and reruns.

## 3. Repository map and ownership

| Location | Responsibility |
| --- | --- |
| [openonda/](openonda/) | Installed API, CLI, runtime and plotting; no duplicated physics. |
| [source/solvers/vpm/](source/solvers/vpm/) | Particles, numerics and attached VLM/panel models. |
| [source/solvers/fvm/](source/solvers/fvm/) | Finite-volume config, mesh, fields, assembly, solve and I/O. |
| [source/coupler/](source/coupler/) | FVM–VPM synchronization, domains, transfers, exchange and output. |
| [source/simulation/](source/simulation/), [openonda/runtime.py](openonda/runtime.py) | Shared paths, execution and parallel runtime. |
| [tutorials/](tutorials/) | Small physical experiments using the installed API. |
| [studies/](studies/) | Numerical assessments/evidence; no replaced implementations. |
| [tests/](tests/) | Independent numerical, lifecycle, interoperability and output checks. |
| [scripts/](scripts/) | Installation and repository tooling, not hidden tutorial infrastructure. |
| [docs/](docs/) | Current API/method/usage references. Development rules belong only in `AGENTS.md`. |

| Concern | Owner and boundary |
| --- | --- |
| Inputs | Typed case/configuration validates and normalizes once; kernels consume validated intent. |
| Accepted fields/clocks/histories | Running solver; trial/RK state stays distinct. |
| VPM lifecycle | [VPMSolver](source/solvers/vpm/core/solver.py): composition, run, health, diagnostics, output. |
| Evolution order | [EvolutionStepper](source/solvers/vpm/core/evolution.py): orders phases and delegates numerics. |
| Operators | VPM `physics/`, `numerics/`, `stabilization/`; FVM `assemble/`, `schemes/`, `solve/`. |
| Particle state | [Particles](source/solvers/vpm/particles/container.py): device authority; revisions invalidate derived/host caches. |
| FVM fields | [FieldState](source/solvers/fvm/core/state.py): owned/ghost cells, faces and histories. |
| Coupled advancement | [FVMVPMCoupler](source/coupler/solver.py): macro clock/exchange; each solver retains its numerics. |
| MPI/construction | Runtime and [factory](source/coupler/factory.py): owner-only VPM, collective decisions/failures. |
| Sampling/backup | Framework schedules/[OutputManager](source/solvers/vpm/io/sampler.py); no sampler secretly invokes another. |
| Attached VLM | [VLM solver](source/solvers/vpm/boundary_elements/vlm/solver/vlm_solver.py): lattice/circulation/loading; backups share the VPM clock. |
| Status/metadata/files | Solver/coupler I/O. Plotters read native records and never manufacture completion. |
| Presentation | Logging owners (section 8), [openonda.plotting](openonda/plotting.py) (section 10). |

Dependencies flow toward small state/numerical interfaces. Operators receive
explicit inputs/protocols; they do not import tutorials or whole solvers to inspect
unrelated state or print. Prefer composition and focused modules.

## 4. Python style, naming and public interfaces

- Follow Ruff settings: 100-character target lines, four spaces, double-quoted
  strings and grouped/sorted standard-library, third-party and project imports.
  Tool exclusions do not exempt tutorials from the conventions in this document.
- Use `PascalCase` for classes, `snake_case` for modules/functions/attributes and
  `UPPER_SNAKE_CASE` for physical/configuration constants. Prefix implementation
  details with `_`; export only intentional public names.
- Name a class for its physical object or responsibility: `VPMCase`, `StageState`,
  `FlowIntegralsSampler`. Use verbs for actions: `evaluate_stage`, `write_backup`.
  Prefer `time_step_size`, `vortex_strength`, `kinematic_viscosity`, `core_radius`
  and `particle_spacing` over cryptic abbreviations or generic “data” objects.
  Short mathematical symbols are appropriate inside a documented local equation.
- Keep backend and formulation independent: induction backend and stretching
  scheme are separate choices, not one misleading combined method name.
- Keep one canonical spelling across constructor arguments, attributes, CLI
  choices, metadata, backup fields, result directories, CSV headers and legends.
  Update producers and consumers together when renaming. Do not add synonym maps.
- Type public Python boundaries/nontrivial helpers. Prefer specific types and
  small dataclasses/protocols over broad `Any`, dictionaries or `getattr` chains.
  Preserve Taichi annotations (section 6).
- Keep immutable configuration separate from mutable runtime state. Use frozen
  dataclasses where appropriate, `default_factory` for owned defaults and explicit
  keyword-only options when positional calls would be ambiguous.
- Give functions one responsibility; split at physical/ownership boundaries.
- Comments explain equations, units, conventions or non-obvious decisions.
  Remove commented-out code, patch diaries and redundant narration; retain legal attribution.
- Imports must not launch simulations, initialize/reset a device, alter working
  directories, redirect process streams or change user environment settings.
  Put those effects in the owning lifecycle, with reliable cleanup.

## 4.1. Avoid AI coding mannerisms

Apply these checks to ChatGPT/Codex, Claude Code and human edits alike. Judge
observable behavior, not presumed authorship; a search hit is a lead, not a defect.

| Pattern to reject | Required correction |
| --- | --- |
| Unnecessary abstractions | Abstractions need demonstrated ownership/reuse: no one-use registries, service locators, generic managers or forwarding facades. |
| Hidden input errors | No `getattr(owner, "required_field", 0)` or catch-and-continue for required state. Validate type, finiteness and range at its boundary. |
| Layered workarounds | Fix the cause in its owner; remove the displaced path. No wrapper, monkey patch or second configuration authority. |
| Renamed compatibility code | “Canonicalizers”, “read adapters” and “migration helpers” violate section 7 when they preserve replaced interfaces. |
| False success | No zeros, clipping, silent backend switches or relaxed checks to hide failed/missing science. Preserve failure evidence. |
| Duplicate state | Reuse owned fields/clocks/revisions. Caches need complete keys, invalidation and failure/retry behavior. |
| Duplicated code | Find existing operators; consolidate identical logic. Check callers/exports, then delete uncalled helpers/dependencies. |
| Decorative code | No emoji banners, marketing, patch diaries or assignment narration. Use physical names; explain equations/decisions. |
| Inaccurate documentation | Verify units, precision, signatures, examples and performance claims. No invented API or promised capability. |
| Tests that hide defects | Independent mathematics/behavior, including failures. No implementation mirrors, broad skips, warning suppression or weaker assertions. |
| Unrelated changes | No sweeping reformatting, dependencies or expensive reruns without a task-specific reason. Finish affected consumers. |

Abstract protocols, MPI exception broadcasts, transaction rollback and scientific
alternatives can be necessary. Trace behavior before deleting. Keep useful hover
details: units, assumptions and examples are not filler.

## 5. Docstrings must work under mouse hover

Document the actual exported class, constructor, function and property so `help()`,
editor hover and `openonda api ...` expose the explanation. A README or wrapper
is not a substitute. Use relevant NumPy-style sections, without empty headings:

| Section | Required information when applicable |
| --- | --- |
| Summary | Physical/computational purpose, when to use it, principal assumption. |
| `Parameters` / `Attributes` | Signature names, types, defaults, allowed/special values, units, shapes and axis order. |
| `Returns` / `Yields` | Meaning, type, shape, units, precision, ownership, copy/shared storage. |
| `Raises` | Actual invalid-input, unsupported-combination and runtime failures. |
| `Notes` | Equation, frame, signs, normalization, model limits, method, relevant complexity. |
| `Side Effects` | Mutation, allocation, cache invalidation, files, MPI collectives, cleanup/lifecycle. |
| `Examples` | Correct runnable use with imports/defined values and output or observable effect. |
| `See Also` / `References` | Related API or identifiable primary numerical reference. |

State normalization; fluid/body-relative velocity; kinematic/dimensional pressure;
VLM circulation [m²/s] versus particle vector strength [m³/s]; local/world frame;
degrees/radians; tensor indices, normal orientation and cell/group indexing.

Document active/allocated extent, ghosts, contiguity and dtype where relevant.
Frozen dataclasses do not make contained arrays immutable; explain borrowing.
Nontrivial private helpers and kernels need docstrings too; a trivial getter can
use one precise sentence with units. Update class/constructor docs together.

This standalone example illustrates the required level of useful detail:

```python
import numpy as np
from numpy.typing import ArrayLike


def linear_impulse_per_density(
    position: ArrayLike, vortex_strength: ArrayLike
) -> np.ndarray:
    """Compute particle linear impulse divided by constant fluid density.

    Parameters
    ----------
    position : array_like, shape (N, 3)
        Cartesian particle centres in m, relative to the chosen fixed origin.
    vortex_strength : array_like, shape (N, 3)
        Particle vectors Gamma = omega * volume, in m³/s, in the same frame.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Independent float64 vector I/rho = 0.5 * sum(x cross Gamma), in m⁴/s.
        An empty cloud returns the zero vector.

    Raises
    ------
    ValueError
        If the arrays do not have matching (N, 3) shapes or contain NaN/Inf.

    Notes
    -----
    This discrete vortex impulse assumes a localized vorticity field. It
    excludes bound-surface contributions and is not a surface-force integral.
    The result depends on the origin when net vector strength is nonzero.
    Inputs are read only; conversion may allocate temporary float64 storage.

    Examples
    --------
    >>> position = np.array([[1.0, 0.0, 0.0]])
    >>> strength = np.array([[0.0, 2.0, 0.0]])
    >>> linear_impulse_per_density(position, strength)
    array([0., 0., 1.])
    """
    position = np.asarray(position, dtype=np.float64)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape (N, 3)")
    if vortex_strength.shape != position.shape:
        raise ValueError("vortex_strength must match position shape")
    if not np.isfinite(position).all() or not np.isfinite(vortex_strength).all():
        raise ValueError("particle arrays must be finite")
    return 0.5 * np.cross(position, vortex_strength).sum(axis=0)
```

Inspect [StageState/StageRates](source/solvers/vpm/physics/induction/base.py),
[FieldState](source/solvers/fvm/core/state.py),
[VLMSurfaceSetup](source/solvers/vpm/boundary_elements/vlm/config.py) and
[Particles](source/solvers/vpm/particles/container.py); this guide governs completeness.

## 6. Taichi and numerical implementation

- Keep bulk arithmetic on the owning backend; Python owns configuration,
  validation and lifecycle. Use Taichi for Taichi numerical paths. Retain intentional
  NumPy/SciPy/PETSc/Numba backends; do not rewrite unrelated FVM linear algebra.
- Use `@ti.kernel` for host-dispatched kernels, `@ti.func` for device helpers and
  `@ti.data_oriented` for field/kernel-owning classes. Respect the pinned DSL.
- Preserve `ti.template()`, `ti.types.ndarray(...)` and configured scalar/vector
  annotations. Do not replace them with Python types or postponed strings merely
  to satisfy static checking. **Never rebind a kernel argument**; use a local.
  Writing an explicitly mutable output field is not argument rebinding.
- Document dimensions, active counts and read/write roles. Pass active counts
  explicitly: allocated `field.shape` is not population. Test after removal/shrink
  with nonzero unused storage. Check capacity before mutation; reuse workspaces.
  No Python callbacks, NumPy operations or logging inside device loops.
- Initialize scratch/reduction outputs; use race-safe reductions/atomics for shared
  writes. Never depend on parallel iteration order or inactive/uninitialized values.
- Honour compute/accumulation precision in locals, fields and host buffers,
  separately from export precision. Check real f64 execution for narrowing warnings.
  Never silently lower precision or substitute a method/device to hide failure.
- Runtime code owns initialization, leases, threads and teardown. No tutorial or
  unrelated-operator `ti.init()`/`ti.reset()`; closing one solver must not invalidate
  another. Avoid unnecessary downloads/synchronization; invalidate host/derived
  caches after mutations through the particle container's state revision.
- RK rates consume the supplied temporary state and stage time. Position and
  strength share a tableau; equal times can represent distinct stages, so time
  alone is not a stage key. Never substitute cached accepted-state fields.
- Keep diffusion, turbulence, stretching and stabilization explicit. Preserve
  required moments, group labels and histories through splitting/remeshing/renewal.
  Do not disguise consistency loss with clipping, damping or relaxed limits.
- Verify actual kernel execution and numerical behavior on affected supported
  backends; compilation/type checking alone is insufficient.
- Avoid heap arrays in native parallel inner loops; measure allocations and verify numerical equivalence.

## 7. No deprecated, obsolete or compatibility code

**A replacement is incomplete while the replaced implementation is still present.**
Do not add or retain deprecation warnings, aliases, compatibility imports, legacy
readers, superseded-schema branches, synonym maps, fallback implementations or
migration scaffolding to keep replaced code working. Fix the supported method or
remove the nonworking feature and its references; never ship a placeholder as a
working feature. A supported numerical alternative is not obsolete merely because
another method exists; document its distinct purpose.

For each replacement or removal, finish the whole change:

1. Identify every producer, caller, import/export, serialized key and consumer.
2. Update current code, configurations, tutorials, scripts and output producers to
   the single canonical interface. Update current result/figure naming consistently.
3. Delete the old implementation and its helpers, flags, dependencies and files.
   Remove comments, examples, links and tests whose only purpose is preserving it.
   Preserve independent mathematical tests by expressing them through the current API.
4. Do not move dead code into `assets/`, `studies/`, `archive/`, frozen source trees,
   `_old`, `_new`, `_v2` or a compatibility package. Git history already preserves
   development history. Active documentation describes only the current design.
5. Search the maintained source, tests, scripts and docs for the removed symbols
   and paths, then exercise current consumers. A wrapper around the old code is
   not a completed replacement.

Reject incompatible saved schemas explicitly; state which runs need regeneration.
Do not relabel old results as new-method results. Research measurements, references
and required provenance are not obsolete executable code: do not erase/rewrite them
as incidental style cleanup. Keep data-removal scope explicit and preserve attribution.

## 8. Logging, MPI and errors

Use the existing logging boundary, not another `logger.py` or arbitrary root-logger
configuration:

| Context | Owner |
| --- | --- |
| Common layout/rows/units | [source/log_style.py](source/log_style.py) |
| VPM | `Logging` in [source/solvers/vpm/io/logging.py](source/solvers/vpm/io/logging.py) |
| FVM | `solver.logger` from [source/solvers/fvm/io/logging.py](source/solvers/fvm/io/logging.py) |
| Coupler | Its logger/handlers and [source/coupler/reporting.py](source/coupler/reporting.py) |
| Low-level physics | [PhysicsEventObserver](source/solvers/vpm/physics/events.py), connected to logging by its owner |

- Use the shared 88-column FVM-style blocks: uppercase sections, one scalar or
  short vector per row, units, wrapped long paths/labels. Keep startup, meshing,
  LES, diagnostics, stabilization, transfer, backup, profiling and failure output
  consistent. Distinguish elapsed run time from step wall time.
- Modules submit labelled host values to their owner; only the shared formatter
  controls presentation. No local padding, banners, direct prints, root-logger
  configuration, independent file sinks or competing progress counters. Explicit
  CLI help and machine-readable JSON retain their documented formats.
- Decide routine cadence before rendering. Bound pending records to the current
  step; retain no particle arrays or unbounded log history. Emit complete blocks.
  Reuse sampled measurements: logger properties must not trigger field downloads,
  kernels, reductions or collectives. Optional statistics follow sampling cadence;
  numerical acceptance checks keep their independent frequency and thresholds.
- Warnings/errors bypass routine cadence immediately. Identify component, step,
  actual failure and measured/allowed values with sufficient precision. Publish
  step summaries only after acceptance; preserve original errors and exit status.
- Rank zero/owner writes shared files; disabled worker sinks do no formatting.
  Keep rank checks in owners. Every rank makes the same schedule decision before
  optional collective statistics. Broadcast failures before another collective.
- Review all output producers after changes. Keep the output-ownership and lazy
  rendering tests passing; measure skipped/reporting overhead without full reruns.
  Close handlers and restore owned streams. Never suppress scientific failures or
  change sampling/physics to improve the console appearance.

## 9. Tutorials and user-facing scripts

A tutorial shows the physical problem, meaningful controls and solver call directly.
Inspect [Lamb–Oseen](tutorials/vpm/01_lamb_oseen_vortex/setup.py) and
[vortex ring](tutorials/vpm/02_vortex_ring/setup.py) for physical reading order.

Typical layout:

```text
case/
  setup.py                 # one current physical setup
  allrun.sh                # explicit run commands
  allplot.sh               # PNG default; optional PDF
  allclean.sh              # case-local generated-output cleanup
  assets/                  # geometry, references, analysis, plotting, validation
  solution/                # generated native states and metadata
  samples/                 # generated scientific measurements
  figures/                 # generated thesis-style figures
```

- Start `setup.py` with the experiment and short usage docstring. Order its code:
  physical inputs with units; geometry/resolution; derived quantities; numerical
  model choices; output schedules; case construction; run. Show equations such
  as `KINEMATIC_VISCOSITY = RING_STRENGTH / REYNOLDS_NUMBER` directly.
- Keep one active setup. Use simple `argparse` only for meaningful comparison
  choices, with a runnable default. Keep physical branching next to its inputs.
  Do not add generic recipe registries, irrelevant flags or independent status
  machinery. Omit constructor defaults that do not explain the experiment.
- Constructors validate input. Tests and `assets/validate_results.py` assess
  scientific results. Keep `try/except`, `raise`, assertions, environment checks,
  custom CLI validation, MPI controls and output repair out of `setup.py`.
- Use installed `openonda.fvm`, `openonda.vpm`, `openonda.coupler` and
  `openonda.fvm.mesher`. New standalone cases use the current `FVMCase`/`VPMCase`
  interfaces. If a construction boundary needs extending, fix it in the library;
  do not introduce another tutorial-side compatibility adapter.
- Pass mesh builders/configuration to framework construction instead of building
  shared meshes on every rank. Use configuration-based coupled construction so
  the factory creates VPM only on its owning process. Use `solver.evaluate` for
  custom global FVM analysis and `solver.write_csv` for framework-owned tables.
- Never alter `sys.path`/`PYTHONPATH`, choose an interpreter in a launcher, set
  thread/MPI environment variables, call `mpiexec`, or change shell startup files
  to make a case work. Fix packaging/runtime in its owner.
- A local relative import uses the existing package registration:

  ```python
  from pathlib import Path
  from openonda.tutorial_runner import case_package

  __package__ = case_package(Path(__file__).parent)
  from .assets.geometry import build_geometry
  ```

  This assumes a real `assets/geometry.py`; do not invent an unused helper.
  Never reach into another tutorial or assume the repository working directory.
  Library/tooling subprocesses use `sys.executable`.

Shell scripts are direct commands, anchored to their case. For example:

```bash
#!/bin/bash -e
cd -- "$(dirname -- "$0")"
./allclean.sh
python setup.py baseline
python setup.py selective_eddy_viscosity
```

Use only variants that the actual setup exposes. Call `allclean.sh` only when a
fresh run is intended. Do not add loops over `cases.txt`, shell functions, status
accumulators, interpreter variables, `tee`, environment exports or plotting to
`allrun.sh`. The shebang propagates command failure.

Name every figure script `plot_<figure>.py`; it may generate only
`<figure>.png`, `<figure>.pdf`, and non-figure data for that figure. Each
generated figure has exactly one plotting script. A tutorial may have one shared
`assets/postprocess.py` module for data loading and analysis used by its plotting
scripts; do not add parallel audit, runner, projection, or combined multi-figure
postprocessing scripts. Put generated tables and manifests under
`figures/auxiliary/`, never beside the top-level figures.

```bash
#!/bin/bash -e
cd -- "$(dirname -- "$0")"
python assets/plot_results.py --format "${1:-png}"
```

That `allplot.sh` example assumes a matching current plotter. Keep validation an
explicit Python command; necessary plot-input checks belong in the plotting code.
`allclean.sh` may remove only generated output below its own case, never input
assets, reference measurements or another case's data.

## 10. Figures: mandatory defaults and visual verification

These rules apply to **every plot**, including previews, diagnostic figures,
partial results, study figures and animation frames where applicable.

| Property | Required behavior |
| --- | --- |
| Format | `./allplot.sh` and Python plotters default to PNG. `./allplot.sh pdf` and `--format pdf` select PDF. Both retain identical data and styling. |
| Canvas | Default width **12.5 cm**; never exceed it without explicit user direction. Use physical dimensions, not a later scale-to-fit operation. |
| Typography | **10.95 pt** for all visible text, including ticks, legend, titles and colourbars. Use `set_thesis_style()` with the NewPX/Palatino thesis setup; no silent font substitution. |
| Colours | Use named colours from `openonda.plotting.COLORS`, matched to the thesis. No local invented palette or arbitrary Matplotlib cycle. |
| Horizontal margins | Measure outer y-axis text at final font size. Put its leftmost edge **5.5 pt** inside the canvas. If the required left plot margin is `p`, use right plot edge **`1-p`**. |
| Vertical layout | After horizontal placement, choose compact height and `hspace`/`wspace`. Aim for a readable small thesis page; split crowded panels into separate figures before shrinking text. |
| Export | Shared `DEFAULT_DPI` (400), `bbox_inches=None`; preserve the fixed canvas. Never apply tight cropping or relayout after the final margin adjustment. |

Named thesis colours: `TUDdark` navy `#0C2340`, `TUDcyan` teal `#0E8A85`,
`VPMpurple` `#5C3D9B`, `FVMorange` aubergine `#772953`, `RefGray` `#6E8898`,
`DarkText` `#2E3D46`, `AccentGreen` `#2B7A4E`, `AccentRed` `#9C2F50`.
`FVMorange` means thesis aubergine. Use shared named entries, not copied local hex
values. Keep method colours consistent; reinforce with markers/line styles.
Use thesis-based sequential scales for magnitudes and centred diverging scales
for signed fields; use common scales across comparable panels.

Established comparisons: vortex-interactions baseline uses `TUDdark`, selective
eddy viscosity `VPMpurple`, Pedrizzetti relaxation `TUDcyan`, particle splitting
`AccentGreen`, and LBM references `RefGray`. Delta-wing front/rear use
`TUDcyan`/`VPMpurple`. Preserve these assignments.

Starting heights: 7 cm for one history, 8.3 cm for two rows, 10.5 cm for three,
12 cm for six compact panels. Always measure and inspect the actual figure.

Implementation order:

1. Call `set_thesis_style()` before creating artists. Use `CM` for dimensions.
2. Plot real source data; label units and normalization. Prefer short mathematical
   labels whose symbols are defined in the caption. Keep internal task names,
   provenance paths and debug text outside the thesis plotting area.
3. Use `centered_subplots_adjust`, then `fit_thesis_y_label_margins` for ordinary
   grids. For equal-aspect maps/colourbars, measure with `thesis_y_label_margin`,
   derive height from the available width and place the axes explicitly. Include
   colourbars when checking the complete plotting area's symmetry.
4. Fix height, panel spacing, legends and shared labels. Recheck margins after
   resizing or changing ticks. Do not run `tight_layout` or constrained layout
   after this manual adjustment. A save helper that performs another layout pass
   must not undo it; use a direct fixed-canvas save when necessary.
5. Call `validate_thesis_figure`, save with the shared DPI and `bbox_inches=None`,
   and close the figure. PDF export must retain the same physical dimensions.
6. Inspect the PNG **and a rendered PDF** at a readable size. Check text overlap,
   clipping, empty space, legend readability, colourbars, measured symmetry and
   PDF page dimensions. Automated checks alone do not replace visual review.

State averaging weights and coverage; label partial data. No incomplete cycle
means or interpolated images presented as saved states. Keep equal physical aspect
for geometry/fields. Escape LaTeX percent signs; use `$^\circ$` for degrees.
Missing fonts/data must fail usefully, never produce fabricated output.

## 11. Scientific output and restart

Validate the baseline against its reference before comparing methods. Match
inputs, resolution, time step and cadence; change one method at a time. Check VPM
CFL, divergence, alignment, energy/enstrophy and core resolution. Numerical
survival alone is not physical agreement.

- Solvers own native `vpm_metadata.json`/`fvm_metadata.json`; coupled I/O owns its
  execution record. Read the executed configuration, not today's setup or a second
  tutorial metadata authority. Keep reference inputs separate from generated data.
- Reuse native samplers. Do not duplicate them or reconstruct fields to bypass
  missing output. Native readers for restart/visualization remain appropriate.
- Restart restores required clock, fields, histories, settings, labels and random
  state, with current numerical-identity checks. Never fabricate missing history.
- One writer per stream, ordered steps/times, genuine conflict rejection and atomic
  publication. Publish cache keys and derived values together only after success;
  test failure followed by an identical retry. Preserve the primary exception.
- Completion requires native lifecycle, requested horizon and output coverage.
  A backup preceding a failed sampler does not certify complete output.
- Never relax reference values, tolerances, health limits or horizons to obtain a
  pass. Justify model changes; keep comparison conditions and seeds reproducible.
- Do not commit caches, build products or incidental simulation output as source.
  Curated research/reference data needs provenance and purpose; exclude generated
  results from distributed tutorial inputs.

## 12. Verification commands and evidence

Use `python -m pip install -e ".[dev]"`. Start with affected tests; broaden only
when justified. No repeated expensive passing campaigns. Verify independent
mathematics/behavior, not cosmetic logs, private layouts or tuning constants.
Verify trivial documentation edits without creating a new test suite.

Repository checks (limit file arguments while iterating):

```bash
ruff check source tests tutorials scripts openonda
ruff format --check source tests tutorials scripts openonda
python -m compileall -q source tests tutorials openonda
python -m pytest tests -m "not qualification and not slow and not gpu"
```

| Changed area | Required relevant evidence |
| --- | --- |
| Numerical method | Analytical/manufactured solution, convergence order, moments/conservation, precision/resolution limits. |
| Taichi | Real kernel execution, argument-immutability check, affected backend parity. |
| FVM/coupler Python | `pyrefly check` in the configured scope; coupled/collective failure and restart tests when affected. |
| State/I/O | Round-trip required histories/state, atomic failure, ordered output and one writer. |
| Tutorial | Relevant `tests/tutorials/test_tutorial_style.py` and `test_plain_entrypoints.py`; bounded construction/run, plot and clean from a copied case outside the checkout, without path overrides. |
| Plot | Section 10's real-source PNG/PDF review; averaging/source-selection tests if changed. |
| Packaging | `python -m build`; inspect wheel/sdist, install the wheel in a fresh environment and run `python -I -m openonda.verify_install --require-site-packages` outside the checkout. Verify editable installation separately. |
| Documentation | Live symbols, valid links/anchors, executable examples and consistent rules. |

For numerical qualification when warranted:

```bash
python -m pytest tests -m "qualification and not gpu" --numerical-report=numerical-results.json
```

GPU/MPI/external checks need their runtimes. Report failed/unavailable checks and
whether evidence covers a small test, replay or full horizon. See the
[test index](tests/README.md). Fix pre-commit failures deliberately.

## 13. Agent completion checklist

Satisfy relevant items before reporting completion; an audit reports open findings.

- [ ] Cause/requirement investigated; correct owner/current API; unrelated work preserved.
- [ ] Section 4.1 checked: no unnecessary machinery, duplicate physics or misleading claims.
- [ ] Hover docs match signature, units, shapes, frames, ownership, effects and examples.
- [ ] Accepted/stage state, precision, MPI, cache invalidation and cadence remain correct.
- [ ] Replaced code, aliases, adapters, obsolete tests/names/dependencies and references removed.
- [ ] Tutorials show physics; launchers use direct commands without runtime/environment machinery.
- [ ] Owning logger used; primary failures and exit status preserved.
- [ ] Figures meet section 10, including PNG default, PDF option and visual review of both.
- [ ] Meaningful checks passed; partial/unavailable/failing evidence disclosed. No weakened
      tolerances, health limits or horizons.
- [ ] Examples, links, schemas and output names agree; no competing guidelines or patch diaries.
- [ ] Final response states changes, evidence, limits and reruns without overstating results.
