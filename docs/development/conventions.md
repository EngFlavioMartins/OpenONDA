# Code conventions

OpenONDA's FVM and VPM solvers are two parts of one program. They follow the
same conventions, so a reader who knows one solver can read the other. The
coupler, tutorials and tests follow them too.

## Cases and construction

Each solver is built from one immutable case object and constructed
directly. There are no factory functions and no second public configuration.

```python
fvm.FVMSolver(fvm.FVMCase(...))
vpm.VPMSolver(vpm.VPMCase(...))
coupling.CoupledSolver(fvm_case, vpm_case, coupling.CouplerConfig(...))
```

Every solver is a context manager and has `close(*, failure=None)`.

Both case objects share their common fields, in this order:

| Field | Type | Contents |
| --- | --- | --- |
| `name` | `str` | Case name, used in file names and metadata |
| `directory` | path | Case root; results go to `solution/` and `samples/` below it |
| `variant` | `str` or `None` | Optional sub-directory of `solution/` and `samples/` for one variant of a case |
| `numerics` | `Numerics` | Every setting that changes the computed solution, including `time_step_size` |
| `initial_conditions` | solver specific | Initial fields or particles |
| `run` | `RunPlan` | `end_time`, `initial_output`, `final_output`, run-control switches |
| `output` | `Output` | Visualization frames: `schedule`, `precision`, `compression`, ... |
| `samplers` | `Samplers` | `Samplers(samples=(...))` |
| `backup` | `Backup` | `Backup(schedule=None, write_at_end=True)` |
| `logging` | `LoggingConfig` | `mode`, `schedule`, `console`, `detailed_timing` |

Solver-specific fields (the FVM mesh, boundaries and mesh motion; the VPM
lifting surfaces) follow the common ones.

The components of a case are bare nouns: `Numerics`, `RunPlan`, `Output`,
`Samplers`, `Backup`. A group of related numerical options inside them is a
frozen dataclass named after its topic with the suffix `Config`
(`ViscousConfig`, `LinearSolverConfig`, `PimpleConfig`, `VLMConfig`). A group
of acceptance thresholds ends in `Limits`. The suffix `Setup` is not used, and
no object has two names. Every configuration dataclass is
`@dataclass(frozen=True, slots=True)`.

String options are lowercase: `"cuda"`, `"gaussian"`, `"gbd"`, `"pimple"`,
`"raise"`.

## Schedules

Every cadence is one of the shared schedules in `openonda.core.schedules`,
exported by both solvers:

- `EverySteps(interval)` for a number of accepted steps,
- `EveryTime(interval)` for a physical-time interval in s,
- `FinalOnly()` for the final state.

Samplers, visualization frames, backups and log lines take a `schedule`
argument. A schedule is never due at step 0: the initial state is written by
the initial output event, controlled by `RunPlan.initial_output` and each
sampler's `initial` switch.

## Samplers

Sampler arguments are ordered as: geometry (positional), solver-specific
numerics, then the keyword-only output arguments
`*, file_name=None, schedule=EverySteps(1), initial=True`. A line sampler takes
exactly one of `n_points` or `spacing`. Samplers write with `write_csv` and
`write_vtk`. Field samples share the base columns in
`openonda.core.sampling.FIELD_SAMPLE_COLUMNS`. Every sample CSV is keyed by
physical time: `time` is its first column and there is no `step` column.

## Solver lifecycle

| Method | Meaning |
| --- | --- |
| `run(*, start_from=None)` | Run the case to `RunPlan.end_time` |
| `start_from(selection="latest")` | Select the initial state or a backup |
| `advance(*, defer_output=False, time=None)` | Advance one accepted step |
| `write_initial_output()` | Write the initial frame and samples of a fresh run |
| `write_step_output()` | Write output deferred by `advance(defer_output=True)` |
| `write_final_output()` | Write the final frame and final-only samples |
| `save_backup(path=None)` / `load_backup(path, ...)` | Restart files |
| `write_run_metadata(*, status)` | `solution/<solver>_metadata.json` |
| `flush_output()` | Finish pending file output |

Run status values are the `openonda.core.status.RunStatus` members, for
example `completed`, `failed`, `unstable`.

## Clock

A solver commits `time = previous time + time_step_size` at each accepted
step. A driver that owns the clock, such as the FVM--VPM coupler, imposes the
accepted time instead (`advance(time=...)`), so both solvers write identical
times.

## Logging and profiling

Both solvers log through the standard library: `logging.getLogger("openonda.fvm")`,
`"openonda.vpm"` and `"openonda.coupler"`, formatted by `openonda.core.log_style`
and written to `solution/fvm.log`, `solution/vpm.log` and `solution/coupler.log`.
Timing uses `openonda.core.profiling.Profiler` and its `section(name)` context
manager.

## Names

- Physical and numerical quantities are named in full CFD terms:
  `kinematic_viscosity`, `time_step_size`, `vortex_strength`,
  `particle_spacing`. Units are SI and appear in docstrings, not names.
- Counts start with `n_` (`n_particles`, `n_cells`, `n_ranks`); maxima and
  minima start with `max_` and `min_`.
- Wall-clock durations end in `_seconds`; physical times end in `_time`.
- Methods that perform a costly evaluation start with `compute_`; methods
  that gather distributed data start with `gather_`; methods that write files
  start with `write_`. Cheap local state is a property.
- Boolean flags are adjectives or participles without an `is_` prefix on
  private members: `_closed`, `_run_started`.
- Private members start with one underscore; other packages use only public
  members of a solver.
- `__all__` lists are sorted, and imports sit at the top of a module.
- Modules start with a short docstring; there are no author or copyright
  banners in modules.
