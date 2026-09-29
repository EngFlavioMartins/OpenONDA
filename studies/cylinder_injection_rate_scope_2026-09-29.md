# Cylinder injection-rate sensitivity: present algorithm and minimum evidence

The original request is an accuracy–cost sensitivity of injection/renewal. In the
implemented buffered-overlap algorithm, **each accepted exchange renews once**;
provisional interface sweeps replace a trial cloud but are not accepted physical
injections. Consequently changing the exchange interval from 0.04 to 0.02 or
0.08 D/U changes the accepted renewal frequency from 25 to 50 or 12.5 events
per D/U. This is a real sensitivity of the implemented injection rate, but also
changes VPM stepping, FVM boundary exchange and interpolation lag. It must be
reported as a **combined exchange/renewal-clock** sensitivity. An independent
held-cloud control would answer a different question and requires new ownership,
conservation, interface-residual and restart semantics; it is not a prerequisite
to report the implemented algorithm honestly.

The existing matched short cohort gives the initial screen at hxy=dz=hp=0.08D,
span 0.96D, four FVM ranks and common t=0.4D/U. Whole invocations were 418.1,
300.1 and 239.7 s for dt=0.02, 0.04 and 0.08; final particles were 4,887,
4,228 and 3,660; endpoint Cd was 1.7642, 1.5382 and 1.4584. Interface
convergence was 17/20, 2/10 and 0/5 accepted intervals, respectively.
Those transient endpoints neither rank stationary accuracy nor qualify the
0.04/0.08 interface setting. The 0.08 result is a rejected candidate under the
current convergence gate, not an apparent cheap winner. Source:
[cylinder_final_short_cohort.json](cylinder_final_short_cohort.json).

## Minimum paired experiment before a recommendation

Use **dt=0.02 and 0.04** as the first two levels because 0.08 already failed
all five short interface checks. Keep geometry, mesh, hp/core/blend/buffer,
physical viscosity, induction accuracy, FVM substep limit, interface tolerances,
force normalization, output cadence and reference selection identical. A common
qualified interface-iteration policy must pass on *both* clocks before long
runs; the Aitken pilot improved baseline convergence from 2/10 to 10/10 without
measured speedup (10.451 versus 10.416 s warm intervals), but it has not yet
qualified the 0.02 case. Start both from equivalent native initial states and
preserve all scientific health limits. If 0.08 is reconsidered, first pass its
short convergence/conservation gate under the same policy; it is optional for
the minimum comparison.

Screen the pair through a common bounded transient window with per-step
accepted-renewal count, provisional sweep count, net released strength,
replaced/retained particles, circulation/impulse budgets, interface residuals,
FVM/VPM time per phase, peak RSS and full-process wall time. This screen checks
safety and cost only. For an accuracy decision, run both and a spatially/time-
qualified fully meshed reference to the **same physical horizon**, provisionally
tU/D=100, and choose a common stationary statistics start no earlier than 40.
Require at least ten complete shedding periods in the selected window; extend
the horizon if that test fails. Re-equilibrate after any control change. Compare
time-weighted mean Cd, lift RMS, St, velocity profiles and spanwise variation
with cycle/block uncertainty; phase-align only where the metric requires it.
Use the plan's provisional 2%/5%/2%/3% drag/lift/St/profile criteria only when
reference uncertainty and grid/time/domain/span gates support them. Report cost
per physical flow unit and cold complete wall time separately. The 12-hour
allowance is per finest complete case, not a license to shorten observations.
At t=100, the two clocks imply 5,000 and 2,500 accepted renewals; 0.08 would
imply 1,250 if eventually qualified.

## Mesh/runtime decision now

No production mesh is yet qualified. The common-span reference h=0.08D mesh
has 42,624 cells and a 0.244 s warm median at six ranks only through t=1;
h=0.0565685D has 114,104 cells and a 0.708 s warm median only through t=0.16.
These are provisional lower-cost sizing candidates, not developed-wake cost or
accuracy certificates. The currently running h=0.04D, 302,832-cell reference
was read **without modifying the independent process** at step 2,825,
t=22.60, elapsed 5:15:21; its latest step was 7.52 s, Courant 0.321,
continuity error 9.80e-14 and Cd 1.0501. A t=100 trajectory at fixed dt=0.008
has 12,500 steps and a gross 12-hour allowance of 3.456 s/step including
startup. Its current progress already exceeds that budget, and no stationary
window has been reached; neither its eventual runtime nor accuracy is known.
The step-2,825 snapshot alone is not a forecast or a failed final qualification.

The **provisional next paired candidate** is the existing 0.96D slip-slab span,
reference outer domain [-8,24]D × [-10,10]D on h=0.08D with six ranks, and
coupled hxy=dz=hp=0.08D on the existing approximately ±1.48D inner box with
four FVM ranks. This reuses measured meshes and the same physical comparison;
it is selected for *qualification work*, not endorsed production accuracy.
Retain the h=0.0565685D reference as an intermediate spatial check and test a
larger coupled box independently before any domain claim. The reference h=0.08
startup warm median alone would imply ~0.85 h for 12,500 fixed 0.008 steps;
meshing, developed-wake growth, output and drift are omitted, so this is only
a size lower bound. The current h=0.04 reference snapshot demonstrates that a
fine nominal mesh can be materially more expensive than startup scaling.

The corrected coupled h=0.08D warm-start optimization measured 10.416 s per
0.04 interval through t=0.4, whereas the final short cohort's last interval
was 16.56 s. The h=0.064D coupled pilot measured ~16.3–17.0 s early warm
intervals with 36,555 cells and 5.54 GiB peak RSS; it preceded numerical
fixes. Multiplying any of these startup rates by 2,500 or 5,000 does not certify
a 12-hour developed-wake run. Choose a provisional affordable mesh family only
after independent hxy/dz, span and domain tests, paired developed-wake replay,
full wall-clock/memory measurements, and a conservative upper runtime estimate.
If h=0.04D cannot meet the budget, disclose its rejection and qualify a coarser
family rather than silently changing the observation horizon or tolerances.

Sources: [original plan](cylinder_3d_accuracy_performance_plan.md),
[execution report](cylinder_execution_report.md),
[current evidence](cylinder_current_evidence.md), and the live reference
`phase-20260929/logs/reference/console.log` snapshot (2026-09-29, step 2,825).
