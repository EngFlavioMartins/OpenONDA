# Unsteady FVM–VPM benchmark: sphere at Re = 300

The coupling benchmark for an unsteady flow with a sub-1 % target.

```sh
cd referenceFlow && ./allrun.sh && cd ..
./allrun.sh
./allplot.sh
```

## Why a sphere, and not a quasi-2D cylinder

A vortex-particle method represents vorticity as a finite set of blobs with no
images and no periodicity. It can therefore only represent a field whose vortex
lines **close inside the particle cloud**.

A spanwise-uniform ("quasi-2D") cylinder wake violates that by construction: its
vortex lines run straight out through the spanwise faces. A straight vortex tube
of span `L` induces only `a/sqrt(a² + r²)` of the two-dimensional value at
distance `r` (with `a = L/2`):

| span | r = 0.5 D | r = 1 D | r = 2 D | r = 3 D |
|---|---|---|---|---|
| 2 D | 0.894 | **0.707** | **0.447** | **0.316** |
| 8 D | 0.992 | 0.970 | 0.894 | 0.800 |
| 20 D | 0.999 | 0.995 | 0.981 | 0.958 |

A 1 % deficit at r = 3 D needs a span of ≈ 42 D. So in the 2 D-span
`cylinderFlow` case the donor boundary condition is 30–70 % too weak wherever the
wake matters, and **no coupling scheme can repair it** — it is a mismatch between
the flow being modelled and what the method can represent. The reference makes it
worse either way: slip spanwise walls act as mirror images and make the reference
effectively infinite-span, while `fixedValue` walls make it neither.

A sphere at Re = 300 sheds a planar-symmetric train of hairpin vortices. It is
genuinely unsteady and periodic — so mean drag, lift amplitude and Strouhal
number all converge and 1 % is a meaningful target — and its vortex lines close.

The coupler now reports `vortex_line_closure` (mean `|ω·n| / |ω|`) on each
hand-off box face every step and warns above 0.25. Watch it: it is the check
that tells you whether the case is representable at all.

## Configuration

| | hybrid | reference |
|---|---|---|
| near-field spacing | h/D = 1/16 | h/D = 1/16 (identical) |
| domain | FVM box x ∈ [−2, 4.5], y,z ∈ [−2, 2] | x ∈ [−8, 20], y,z ∈ [−8, 8], graded |
| cells | ≈ 4.3 × 10⁵ | ≈ 1.6 × 10⁶ |
| outer treatment | VPM particles, unbounded | mesh + real outlet |
| body | IBM sphere, 804 markers | identical |
| scheme, dt, SGS | limitedLinear, backward, dt = 0.02, laminar | identical |

`h/D = 1/16` is not arbitrary. `tests/coupler/test_handoff_convergence.py`
measures the hand-off error against an analytic Lamb–Oseen core and shows it
needs about **four lattice cells per vortex-core radius** for 1 %:

| cells per core radius | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| transfer error | 23 % | 5.3 % | 2.2 % | **1.2 %** | 0.52 % | 0.29 % |

The hairpin cores two diameters downstream are ≈ 0.25 D, so h ≤ D/16.

The hand-off interface sits at x = 4 D, well downstream of the recirculation
bubble (which closes near x = 1.6 D). An interface inside reversed flow breaks
every sizing rule the transfer uses; the coupler warns if the mean outward
normal velocity on the outflow face is not positive.

## What the comparison reports

`allplot.sh` writes `samples/comparison_metrics.json` and `figures/forces.png`
with the three quantities that actually converge for a periodic flow:

* mean `Cd` over `tU/D ≥ 40`,
* lift RMS,
* Strouhal number (parabolically interpolated, since the raw FFT bin spacing is
  coarser than the 1 % target),

plus the lift phase lag and the correlation at that lag. Instantaneous pointwise
field differences are **not** the metric: two runs of the same periodic flow
drift in phase, and a pointwise difference then measures the phase, not the
coupling.

Literature anchor (Johnson & Patel, *JFM* **378** (1999) 19–70): Cd = 0.656,
St = 0.137, mean Cl = 0.069. That is a sanity check only — the benchmark proper
is hybrid versus the matched sibling.

## Cost

Serial, on a development laptop: the reference is the expensive half
(≈ 1.6 × 10⁶ cells, 5000 steps) and is an overnight run. The hybrid is roughly a
quarter of that plus the VPM. Both `allrun.sh` files take no arguments — edit
their explicit `OPENONDA_*` block so a study is version-controlled.

For a quick end-to-end check of meshing, IBM, coupling and output:

```sh
OPENONDA_SMOKE=1 ./allrun.sh
```

Before spending the overnight run, make the cheap gates pass:

```sh
pytest tests/coupler -m verification -s
```

They exercise the hand-off alone against analytic fields — no FVM, no chaos —
and print the sizing tables above.
