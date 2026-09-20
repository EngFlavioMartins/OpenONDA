# Panel-free coupling: measured results and campaign status

The cube passes the declared 4 s force/profile comparison. The infinite-span
cylinder has encouraging startup agreement through 2 s and a verified restart
through 2.20 s. Mature shedding, reference grid independence and total runtime
savings are not yet established. The panel solver remains available.

These measurements were assessed on 20 September 2026. Current execution state
must be checked from native output and process identity; a launch record or
checkpoint alone is not proof of completion.

## Cube comparison

All 80 interface steps through 4 s converged. The comparison uses common saved
physical times, without force scaling or phase shifts. The full identity,
coverage and acceptance checks are in [cube_gate.json](cube_gate.json).

| Quantity, 2–4 s | Without panels | Saved panel run | Fully meshed fine reference |
| --- | ---: | ---: | ---: |
| Mean Cd | 0.936764 | 0.934304 | 0.924175 |
| Cd history RMS error relative to reference | 1.481% | 1.198% | — |
| FVM centreline RMS error / U∞ | 2.731% | 2.704% | — |
| FVM off-axis profile RMS error / U∞ | 3.305% | 2.998% | — |

Removing panels changes mean Cd by 0.263% and the Cd history by 0.364% RMS.
FVM centreline and off-axis profile changes are 0.094% and 0.752% of U∞.
The independent 1–4 s window and every-snapshot checks also pass. VPM wake-only
changes over 1–4 s are 0.201% and 0.243% of U∞.

The initial force impulse differs more strongly. Agreement after 1 s does not
establish agreement throughout startup or mature-flow statistics. The panel-free
run permits six interface sweeps and usually requires four; the saved panel run
usually required two or three under the same stopping tolerances. Comparison
with that saved run is not a controlled end-to-end performance benchmark.

[Force histories](cube_comparison.png) and [wake comparison](cube_comparison_wake.png)
use the recorded data. Profile companions and captions are stored alongside them.

## Cylinder startup comparison

[cylinder_startup.json](cylinder_startup.json) compares the qualified trajectory
with the historical fully meshed fine reference over 0.2–2 s. Mean Cd is
1.388162 versus 1.408302, a 1.430% difference.

| Mean vector-profile RMS error / U∞ | FVM | VPM |
| --- | ---: | ---: |
| Centreline | 0.317% | 0.328% |
| Transverse x/D=1 | 0.492% | 0.564% |
| Transverse x/D=2 | 0.378% | 0.404% |
| Transverse x/D=4 | 0.330% | 0.334% |

The 50 interface steps through 2 s converge. Startup lift is small but differs
between the runs; this window contains no complete shedding cycles. The report
therefore remains provisional with its mature gate false. The
[force histories](cylinder_startup.png), VPM companion and separate profile
figures show the same saved times, without fitted shifts.

A discontinuity in planar pruning/conservation recovery caused the diagnostic
pilot to fail interface convergence at 1.48 s. Continuous magnitude-weighted
recovery resolves that step in three sweeps with normal/gradient residuals
2.82e-7 and 1.88e-7, below unchanged 1e-5 tolerances. The native mesh and cutoffs
are unchanged; see [renewal qualification](planar_renewal_qualification.json)
and the [method definition](planar_model.md#continuous-planar-renewal).
The pilot remains unqualified and its results are excluded from admission.

The startup JSON preserves a failed original span-probe flag. The
[checkpoint audit](cylinder_span_probe_audit.json) reproduces that affine
probe's 0.002401 variation even on exactly span-invariant input. Actual raw-cell
stack deviation is 1.37e-5, below the unchanged 1e-3 limit. Nearest-cell span
sampling removes the interpolation artifact without changing equations,
velocity-profile stencils or saved measurements. After strict restart, all 55
steps through 2.20 s converge; the span sample's largest component range is
6.81e-6 U∞. See [restart qualification](cylinder_restart_qualification.json).

## Running campaigns

| Assessment | Output directory | Requested horizon |
| --- | --- | ---: |
| Cube without panels | `runs/cube_no_panel_converged` | 20 s |
| Cylinder without panels | `runs/cylinder_no_panel_converged` | 160 s |
| Cube reference | `02_cube_flow/reference_flow/campaigns/geometric_r15` | 120 s per case |
| Cylinder reference | `01_cylinder_shedding_flow/reference_flow/campaigns/geometric_xy_v1` | 160 s per case |

Reference paths are relative to `tutorials/coupled_fvm_vpm`. Coupled paths are
relative to this study. Each launch has `process.json` and `launcher.log`.
Historical samples and checkpoints remain intact. New-run horizons and denser
visualization schedules are documented in [portable runs](PORTABLE_RUNS.md);
the table records the original active launches, which are not reconfigured by
editing setup files.

The cube reference uses four spatial grids at r=1.5 and a half-step control;
see its [campaign specification](../../tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/GRID_CAMPAIGN.md).
The cylinder uses four XY grids at r=√2 with independent span-resolution,
span-width and half-step controls; see its
[reference specification](../../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/README.md).
Larger cube grids wait for the recorded coupled experiments to release memory.

The cylinder completion stage is attached to the continued experiment and its
reference campaign. It writes
`runs/cylinder_no_panel_converged/mature_comparison/status.json` and evaluates
80–160 s after both required results complete. See the
[experiment protocol](CYLINDER_EXPERIMENT.md) for invocation and acceptance rules.

## Remaining qualification

Grid independence requires stationary statistics, decreasing spatial differences,
observed order/GCI, agreement with the fourth grid and small time/span-control
errors. A geometric sequence alone is insufficient. Longer windows or additional
resolution studies are required if these checks fail.

The cylinder's particle spacing, coupling interval and FVM domain size also need
independent sensitivity checks before optimizing cost. Its explicitly planar
operator does not model finite-cylinder ends or three-dimensional wake modes.
A lower panel-evaluation cost does not establish lower total simulation cost.
