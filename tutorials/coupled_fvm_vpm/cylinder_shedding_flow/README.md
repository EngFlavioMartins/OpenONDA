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

| Grid | Surface `h/D` | First wall cell `h_1/D` | Wall layers | Shear / near wake | Downstream wake | Far field | `dt U/D` | Qualified cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1/8 | 1/64 | 4 | `2h` | `4h` | `4h` | 0.005 | developer only |
| G0 | 1/8 | 1/64 | 6 | `2h` | `4h` | `4h` | 0.004 | 22,448 |
| G1 | 1/16 | 1/128 | 8 | `2h` | `4h` | `8h` | 0.002 | 37,712 |
| G2 | 1/32 | 1/256 | 10 | `2h` | `4h` | `16h` | 0.001 | 82,880 |

The cylinder has a fitted O-grid boundary layer stitched into the adaptive
Cartesian far field. The first-cell height is confined to the wall-normal
layers and is not propagated through the domain. Likewise, the shear layers,
near wake, and downstream street use progressively coarser dedicated boxes
before returning through balanced 2:1 transitions to the `0.5D` far field.
This anisotropic hierarchy is intentional: the small first-cell height
resolves the wall-normal gradient, while isotropic `2h` and `4h` cells track
separation and the vortex street without exporting wall resolution throughout
the domain. G1 therefore reaches `h_1=D/128`
without a globally fine mesh; G2 checks it at `D/256`. G0 is the coarse
convergence member, not the final wall-resolution authority.
`assets/audit_mesh_geometry.py` writes `mesh.vtu` with `refinementLevel` and
`wallAdjacent` cell arrays for direct inspection.

All three production meshes passed strict topology and geometry checks. Their
maximum non-orthogonality is below `43.1 deg`, maximum skewness below `0.479`,
and circular wall-area error below `0.09%`. The full `4D` span uses eight
native hexahedral slabs; comparison against sixteen slabs changed drag by at
most `0.614%`, mean streamwise wake velocity by `0.00191%`, and mean
transverse wake velocity by only `7.9e-7 U_inf`. The reconstructed in-plane
velocity range across either span was below `2.3e-6 U_inf`.

The production matrix contains `G0`, `G1`, and `G2`, plus a `G1` half-time-step
case and a `G1` enlarged-domain case. The baseline domain is
`[-8,20] x [-8,8] x [-2,2] D`; the larger domain is
`[-10,25] x [-12,12] x [-2,2] D`. Near-body refinement boxes remain fixed in
the domain study.

The reference gate uses the last 30 convective units (never earlier than
`t=30`) and requires the G1-to-G2, G1-to-half-`dt`, and G1-to-large-domain
changes to satisfy:

- mean drag coefficient and lift harmonic amplitude: `< 1%`;
- drag peak-to-peak amplitude: `< 2%`;
- Strouhal number: `< 0.5%`;
- finite fields, converged linear solves, controlled CFL and continuity;
- valid cut-cell topology, positive volume, correct wall normals, and wall
area within `1%` of the analytic circular side-wall area.

For cost control, the saturated G0-to-G1 comparison is also used as a
preliminary mesh-sizing bracket. Differences of `2%` or less in Strouhal
number, mean drag, drag peak-to-peak amplitude, and lift amplitude indicate
that the planned ratio-two G2 refinement is a credible final verification
mesh. This bracket guides sizing only: it does not replace the G0/G1/G2,
half-time-step, and large-domain independence gate above.

`assets/save_verification_case.py` preserves each statistically ready variant's
force history, performance history, metadata, diagnostics, and sample audit in
`reference_flow/solution/verification/` without replacing existing evidence.
The final `grid_independence` figure plots `St`, mean drag, drag peak-to-peak
amplitude, lift harmonic amplitude, measured solver wall time, and actual cell
count against the surface resolution `D/h`.

Every reference run also aborts if the instantaneous CFL exceeds `1.5`.
A G1 production-seed trial at `dt=0.004` failed this gate at `t=0.328`
(`CFL=2.814`); `dt=0.002` remains below `CFL=0.12` through the developing
wake and is the accepted production step pending the final half-step
comparison. Both FVM paths use the bounded second-order `limitedLinear` TVD
convection scheme and least-squares gradients, which are robust on the
non-orthogonal wall-adjacent polyhedra.

The coupled comparison then reports the same quantities plus lift RMS,
phase/frequency alignment, centerline and transverse velocity errors,
spanwise coherence, circulation transfer, population pruning, and hand-off
boundary leakage. Sub-1% coupled force/frequency agreement is the target, not
an assumed result.

Field probes use a distance-weighted, local affine reconstruction, which is
exact for constant and linear fields. The spanwise FVM line samples the eight
extruded slab centres directly; this avoids manufacturing spanwise modulation
by applying different nearest-cell stencils between slab centres. Midspan VTK
slices carry a `vtkValidPointMask` for the true circular cross-section, not its
square bounding box.

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

The production reference defaults to six MPI ranks. On the qualifying laptop,
the most sustainable affinity uses six efficiency cores:

```bash
OPENONDA_GRID=g1 \
OPENONDA_FVM_CORES=6 \
OPENONDA_RANK_CPUS=12,13,14,15,16,17 \
./reference_flow/allrun.sh
```

The case reads the stable Linux `TCPU` sensor when available and leaves a
complete checkpoint before thermally pausing. `OPENONDA_MAX_CPU_TEMP_C` and
`OPENONDA_RESUME_CPU_TEMP_C` control that workstation safeguard; it does not
change the flow equations or accepted time step.

The reference script cleans generated files only from `reference_flow/solution/`
and `reference_flow/samples/`; it preserves the small
`reference_flow/solution/verification/` histories and never touches the
coupled root output. Mesh, time-step, and domain sensitivity calculations are
run one at a time through the same conventional case:

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
developer checks. Production runs use a memory-availability gate. Independent
documentation, static checks, and lightweight post-processing may run during a
solver, but multiple large mesh builds or flow solvers should not compete on a
14-GiB workstation.

Compressed FVM volume fields and complete coupled restart checkpoints are
written every `2 D/U_inf`; smoke or bounded runs also write their terminal
state. Override the cadence with `OPENONDA_FIELD_OUTPUT_INTERVAL` when a denser
inspection sequence is needed.

After an interrupted restart, trim only uncheckpointed tail records with:

```bash
python assets/prune_restart_tail.py reference_flow
```

Reference-only force and solver diagnostics are generated with:

```bash
python assets/analyse_reference.py --require-ready reference_flow
python assets/audit_reference_samples.py reference_flow
```

The completed G0, G2, half-time-step, and large-domain force histories are
retained under `reference_flow/solution/verification/` with the simple names
used by `allvalidate.sh`. The selected G1 history remains canonical at
`reference_flow/samples/forces_history.csv`. Once those histories reach the
common horizon, validate without cleaning or launching any solver:

```bash
./allvalidate.sh reference
```

After the coupled root case is complete, `./allvalidate.sh all` applies both
the reference-independence gate and the coupled comparison. Validation never
deletes or replaces solution data.

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
