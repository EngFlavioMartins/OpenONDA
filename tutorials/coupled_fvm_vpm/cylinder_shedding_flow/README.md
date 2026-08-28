# Cut-cell FVM–VPM cylinder-shedding benchmark

This tutorial compares a fully meshed circular-cylinder FVM calculation with
a matched FVM–VPM calculation. Both calculations use the same watertight STL,
Reynolds number, near-body Cartesian lattice, physical sampling times, and
antisymmetric disturbance. The near-body cylinder is represented by a
conformal wall generated with OpenONDA's built-in cfMesh-derived adaptive
Cartesian mesher. **No immersed-boundary model (IBM) is used.**

The benchmark is intended to expose small coupling errors in drag, lift,
shedding frequency, and wake velocity. It must first establish a converged
fully meshed reference; only then is the coupled result judged against it.

## Physical case

| Quantity | Value |
|---|---:|
| Diameter, `D` | `1` |
| Solved cylinder span | `4D`, from `z=-2D` to `z=+2D` |
| STL extent | `12D`; remote caps remain outside the solved span |
| Freestream | `(1, 0, 0)` |
| Density | `1` |
| Reynolds number | `150` |
| Kinematic viscosity | `1/150` |
| Force reference area | `D L = 4` |
| Production horizon | `60 D/U_inf` |
| Force/probe period | `0.1 D/U_inf` |
| Full-field/checkpoint period | `2 D/U_inf` |

`Re=150` is high enough to produce a clean laminar von Kármán street while
remaining below the approximately `Re=188.5` onset of the first secondary
three-dimensional instability of an infinite cylinder. The spanwise slip
planes remove finite-cylinder cap forces from the solved segment. Published
infinite/periodic-cylinder values are still sanity bounds rather than
replacement truth; the converged fully meshed run is the quantitative
reference for this tutorial.

The versioned geometry is
[`assets/cylinder_long.stl`](assets/cylinder_long.stl). It contains 1,280
outward-oriented triangles and can be regenerated deterministically:

```bash
python assets/generate_cylinder_stl.py
```

## Verification matrix

The reference mesh uses a `0.5D` background grid and fixed nested refinement
boxes around the shear layers and wake. The ratio-two family is:

| Grid | Surface `h/D` | Shear layer `h/D` | Near wake `h/D` | Downstream wake `h/D` | `dt U/D` | Cell cap |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 1/8 | 1/4 | 1/4 | 1/2 | 0.02 | 100,000 |
| G0 | 1/16 | 1/8 | 1/8 | 1/4 | 0.01 | 150,000 |
| G1 | 1/32 | 1/16 | 1/8 | 1/4 | 0.005 | 350,000 |
| G2 | 1/64 | 1/32 | 1/16 | 1/8 | 0.0025 | 700,000 |

The fine wall spacing is applied only to STL-intersecting cells and its 2:1
transition band. It is not propagated through the domain. Likewise, the shear
layers, near wake, and downstream street use progressively coarser dedicated
boxes before returning to the `0.5D` far field. Thus G1 resolves the no-slip
wall with `h_w=D/32` while avoiding a globally fine Cartesian mesh; G2 checks
that result with `h_w=D/64`. G0 is the coarse convergence member, not the final
wall-resolution authority. `assets/audit_mesh_geometry.py` writes `mesh.vtu`
with `refinementLevel` and `wallAdjacent` cell arrays for direct inspection.

The production matrix contains `G0`, `G1`, and `G2`, plus a `G1` half-time-step
case and a `G1` enlarged-domain case. The baseline domain is
`[-8,20] x [-8,8] x [-2,2] D`; the larger domain is
`[-10,25] x [-12,12] x [-2,2] D`. Near-body refinement boxes remain fixed in
the domain study.

The reference gate uses the last 30 convective units (never earlier than
`t=30`) and requires the G1-to-G2, G1-to-half-`dt`, and G1-to-large-domain
changes to satisfy:

- mean drag coefficient and lift harmonic amplitude: `< 1%`;
- Strouhal number: `< 0.5%`;
- finite fields, converged linear solves, controlled CFL and continuity;
- valid cut-cell topology, positive volume, correct wall normals, and wall
  area within `1%` of the analytic circular side-wall area.

Every reference run also aborts if the instantaneous CFL exceeds `1.5`.
This caught an initially proposed G0 step of `0.02`; the production family
above keeps the same conservative `dt/h_surface=0.16` as the stable smoke
case. Both FVM paths use the bounded second-order `limitedLinear` TVD
convection scheme and least-squares gradients, which are the robust choices
for the non-orthogonal wall-adjacent polyhedra.

The coupled comparison then reports the same quantities plus lift RMS,
phase/frequency alignment, centerline and transverse velocity errors,
spanwise coherence, circulation transfer, population pruning, and hand-off
boundary leakage. Sub-1% coupled force/frequency agreement is the target, not
an assumed result.

## Running the case

The directory structure deliberately matches `cube_flow/`. The conventional
full-domain calculation writes `reference_flow/solution/` and
`reference_flow/samples/`. The coupled small-domain calculation writes the
root `solution/` and `samples/` directories. No additional run-directory
layer is used.

First audit and smoke-test the conformal mesh and reference sampler path:

```bash
OPENONDA_GRID=smoke python assets/audit_mesh_geometry.py
OPENONDA_SMOKE=1 OPENONDA_GRID=smoke ./reference_flow/allrun.sh
```

Run the conventional full-domain reference:

```bash
OPENONDA_GRID=g1 ./reference_flow/allrun.sh
```

The reference script cleans only `reference_flow/solution/` and
`reference_flow/samples/`; it never touches the coupled root output. Mesh,
time-step, and domain sensitivity calculations are run one at a time through
the same conventional case:

```bash
OPENONDA_GRID=g1 OPENONDA_DOMAIN=baseline OPENONDA_DT_SCALE=1 ./reference_flow/allrun.sh
OPENONDA_GRID=g1 OPENONDA_DOMAIN=large OPENONDA_DT_SCALE=1 ./reference_flow/allrun.sh
OPENONDA_GRID=g1 OPENONDA_DOMAIN=baseline OPENONDA_DT_SCALE=0.5 ./reference_flow/allrun.sh
```

After selecting the reference resolution, run the matched coupled calculation:

```bash
OPENONDA_GRID=g1 ./allrun.sh
./allplot.sh png
./allplot.sh pdf
```

The strict numerical comparison can be rerun directly:

```bash
python assets/analyse_coupled_benchmark.py
```

For a bounded integration test, set `OPENONDA_SMOKE=1`. Optional
`OPENONDA_MAX_STEPS` and `OPENONDA_MAX_COUPLING_STEPS` bounds are available for
developer checks. Production runs intentionally require one FVM core and use a
memory-availability gate. The independent documentation, static checks, and
post-processing may run concurrently, but large mesh builds and solvers should
remain serial on a 14-GiB workstation.

Compressed FVM volume fields and complete coupled restart checkpoints are
written every `2 D/U_inf`; smoke or bounded runs also write their terminal
state. Override the cadence with `OPENONDA_FIELD_OUTPUT_INTERVAL` when a denser
inspection sequence is needed.

## Outputs

| Directory | Contents |
|---|---|
| `reference_flow/solution/` | conventional full-domain FVM fields and diagnostics |
| `reference_flow/samples/` | conventional FVM forces, probes, profiles, and slices |
| `solution/` | coupled small-domain FVM–VPM fields, checkpoints, and diagnostics |
| `samples/` | coupled forces, probes, profiles, and slices |
| `figures/` | fixed-width PNG/PDF figures using the project publication style |

All figures are exactly 12.5 cm wide and use uniform 10-point DejaVu Serif
text for titles, labels, ticks, and legends, matching the Lamb–Oseen tutorial.
The scripts use explicit layouts and do not use tight bounding-box cropping,
so output dimensions remain comparable across figures.

## Literature

The downloadable papers and durable publisher/author links are indexed in
[`references/README.md`](references/README.md). Validated local copies of the
Henderson & Barkley secondary-instability note and Barkley & Henderson's
Floquet study are included in that directory.

The principal references are Williamson (1996), Henderson & Barkley (1996),
Karniadakis & Triantafyllou (1989), Posdziech & Grundmann (2007), and Barkley &
Henderson (1996). Please cite the DOI/publisher records listed in the reference
index rather than an unofficial mirror.
