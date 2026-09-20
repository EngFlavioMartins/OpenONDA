# Historical cylinder reference assessment

The completed five-grid campaign recorded below predates the geometric
`allrun.sh` sequence. It reached 100 s for every case. The `dense` force
history has a final uninterrupted restart segment from 63.34 s to 100 s;
therefore every case was compared over the final half of the common valid
interval, **81.67 <= t <= 100 s**. The table reports time-weighted force
statistics and the time-mean centreline-velocity error relative to `dense`.
The stated `h/D` values are the cylinder cell-size targets supplied through
`--dx`, rather than an inferred global mesh size.

| Case | Target h/D | Fluid cells | Mean Cd | RMS Cl | Centreline relative L2 error vs. dense |
|---|---:|---:|---:|---:|---:|
| `very_coarse` | 0.080 | 24,472 | 1.360481 | 0.334980 | 5.52% |
| `coarse` | 0.070 | 30,848 | 1.345987 | 0.323610 | 5.66% |
| `medium` | 0.060 | 50,850 | 1.364525 | 0.351658 | 2.56% |
| `fine` | 0.050 | 73,880 | 1.371434 | 0.363261 | 1.35% |
| `dense` | 0.040 | 156,828 | 1.371475 | 0.372193 | reference |

`fine` is a candidate in this historical family when the primary quantities
are mean drag and time-mean wake flow: its mean drag differs from `dense` by
only **0.003%** over this window, while using 47.1% of the dense-grid fluid
cells (a 2.12x reduction).
This is a practical grid-selection criterion, not a claim that every unsteady
quantity is fully converged. From `fine` to `dense`, RMS drag changes by 3.78%,
RMS lift by 2.40%, and the centreline mean velocity by 1.35% in relative L2
norm. Use `dense`, or perform a further refinement, when those unsteady
quantities require approximately 1% grid uncertainty.

There are two limitations to retain with this result. First, the mean-drag
difference between `fine` and `dense` becomes 0.124% when the shorter 90--100
s window is used; the 0.35% (`fine`) and 0.57% (`dense`) drag half-window
drifts show that longer averaging would strengthen the conclusion. Second, the
three finest target-spacing ratios are 1.20 and 1.25, and the coarser-grid
values are non-monotone. The assumptions for a Richardson/GCI estimate are
therefore not met. The controlled campaign uses a fixed span mesh, exact sqrt(2) nominal XY
refinement, and an 80-second common statistical window. Retain further cycles
if the batch variation still exceeds the finest-grid differences.

The source records remain in the cylinder reference tutorial's `samples/` and `solution/` directories.
