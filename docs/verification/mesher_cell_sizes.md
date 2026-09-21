# Cube reference mesher: cell-size investigation

Investigated on 2026-09-13 using the current working tree and the saved meshes
in `tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/solution/`.

## Diagnosis

The background-dependent size jumps occur in mesh generation, before the
flow solver is constructed. They follow cfMesh's octree sizing rules. Two
OpenONDA defects made this difficult to diagnose: box sizes were reported
with the surface rounding rule, and time-step visualization omitted the
cell geometry that was already available in the separate mesh backup.

Let `H = max_cell_size`. The available nominal octree edges are
`H, H/2, H/4, ...`:

- Boundary and patch requests select the first edge **at or below** the target.
- Box requests select the first edge **strictly below** the target, with
  cfMesh's small floating-point tolerance. An exact dyadic ratio adds a level.

This is verified against upstream
[`objectRefinement::calculateAdditionalRefLevels`](https://github.com/Unofficial-Extend-Project-Mirror/cfMesh/blob/master/meshLibrary/utilities/octrees/meshOctree/refinementControls/objectRefinement/objectRefinement.C#L108).
The implementation compares the requested size with the current size using
`<=` and halves the current size each pass. OpenONDA's
`object_additional_level()` uses the same conversion. The
[cfMesh user guide](https://cfmesh.com/wp-content/uploads/2015/09/User_Guide-cfMesh_v1.1.pdf)
also describes refinement levels relative to the maximum cell size.

The cube requests are fixed at `nearBody=3*dx`, `wake=6*dx`, and `cube=dx` in
the following sweep. The entries are nominal sizes divided by `dx`:

| `H / dx` | Near-body box | Wake box | Cube patch |
| ---: | ---: | ---: | ---: |
| 8 | 2 | 4 | 1 |
| 11.99 | 2.9975 | 5.995 | 0.749375 |
| 12 | 1.5 | 3 | 0.75 |
| 12.01 | 1.50125 | 3.0025 | 0.750625 |
| 16 | 2 | 4 | 1 |

These values were checked against fresh template meshes at `dx=0.25`, using
the reference STL, domain, boxes, and patch request. Each box contained cells
at its reported nominal size and no nominally coarser cells whose centres
were inside that box. A full production build at `H=12*dx` also passed its
normal mesh validation and published the corrected report.

Crossing from `11.99*dx` to `12*dx` nearly halves the box spacings. A factor
of two in an undistorted cell edge corresponds to a factor of eight in
volume. This is a discontinuity of the octree level selection, not evidence
of a time-integration instability.

Changing `H` can also change the root cube and lattice placement. Intersecting
leaves are refined, overlapping requests select the finer level, and 2:1
face/edge/corner balancing extends refinement outside the requested region.
Projection, wrapper insertion, and mesh optimization subsequently change
physical geometry. A nominal octree edge is therefore not an exact final
edge length, wall-normal spacing, or cube root of cell volume.

## The saved cases use different mesh settings

The following values were read from each saved `mesh.npz`; the domain bounds
were checked against the actual vertices. They are not inferred from the
current `setup.py` or the directory names.

| Saved case | Requested cube size | Background | Nominal cube size | Fluid cells | Domain |
| --- | ---: | ---: | ---: | ---: | --- |
| `very_coarse` | 0.22 | 0.5 | 0.125 | 13,560 | `[-5,10] × [-5,5] × [-5,5]` |
| `coarse` | 0.20 | 2.4 | 0.15 | 5,216 | `[-7.5,15] × [-7.5,7.5] × [-7.5,7.5]` |

Thus the saved `coarse` case has fewer cells and a coarser nominal cube
lattice despite its smaller requested cube size. The background and domain
were also changed, so these two outputs do not form a controlled refinement
study. These were local output files, not files shipped with the repository.
The measurements describe the September 13 inputs, not the current grid family;
see the [current reference setup](../../tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/README.md).

The old `coarse` metadata incorrectly reported `nearBody=0.6` and `wake=1.2`.
Their strict box conversions are actually `0.3` and `0.6`. The repaired
report uses the production conversion and records `resolved_box_sizes`.
Existing saved metadata and simulation output have not been rewritten.

## Output and reporting repairs

The mesh backup and solver visualization now share these cell fields:

| Field | Meaning | Units |
| --- | --- | --- |
| `cell_volume` | Physical volume used by the FVM solver | m³ |
| `cell_equivalent_size` | Cube root of physical volume; not a maximum edge | m |
| `cell_size` | Nominal octree edge, when present in the mesh | m |
| `refinement_level` | Absolute level counted from the root cube | dimensionless |
| `boundary_layer_index` | Layer provenance, when available | dimensionless |

The fields survive synchronous and asynchronous output, partition
localization, owned-only pieces, ghost-cell pieces, and the PVTU collection.
Geometry arrays remain cell data when flow fields are interpolated to points.
A localized mesh containing all global cells as owned cells plus halos still
needs local indices for its compact visualization mesh.

`GenerationReport.sizes` uses strict box conversion and inclusive
patch/boundary conversion. Its `level` counts additional levels from `H`,
unlike the absolute per-cell `refinement_level`.
`mesher.effective_cell_size(target, strict=True)` previews a box request;
the default previews a patch/boundary request.

## Using this in ParaView and subsequent studies

For existing results, open the separate `mesh.vtu` and inspect **Cell Data**
using cell hovering or selection.
It already contains `cell_volume`, `cell_size`, and `refinement_level` for
these cases. Alternatively, apply ParaView's **Cell Size** filter to an old
time-step mesh to calculate its geometric volume; see the
[ParaView Cell Size documentation](https://www.paraview.org/paraview-docs/v5.11.0/python/paraview.simple.CellSize.html).
Newly written time-step
VTU/PVTU files include the geometry arrays directly.

For an exact nominal cube spacing of `dx`, select a compatible background
such as `8*dx` or `16*dx`. The existing box requests then resolve to `2*dx`
and `4*dx`. For an exact desired box spacing `s` on that lattice, request a
strict upper target between `s` and `2*s`, for example `1.5*s`; requesting
exactly `s` causes another subdivision. Retain a fixed domain and deliberate
background-to-target ratios across a grid study, and compare recorded
resolved sizes and cell volumes rather than directory labels alone.

The current tutorial settings and flow equations were retained. The work
qualifies mesh sizing and visualization; it does not rerun a transient flow
or claim convergence of those differently configured saved simulations.

## Validation

The focused regression run passed 39 tests, including five fresh reference
templates, a complete reference mesher build, native cfMesh octree tests,
serial and asynchronous solver output, both partition ghost policies,
ParaView/PyVista PVTU readback, and independent VTK volume comparisons.

```bash
python -m pytest tests/fvm/test_mesh_size_output.py \
    tests/mesh_parity/test_cfmesh_size_reporting.py \
    tests/mesh_parity/test_cfmesh_octree.py tests/test_storage_output.py
```

Ruff checks and formatting checks passed for the changed implementation and
new test files. `git diff --check` passed.

A further 34 existing factory, restart, configuration, and mesh-contract tests
passed (73 distinct passing tests in total). The output/template subset also
passed again after the final optional-metadata handling adjustment.

```bash
python -m pytest tests/fvm/test_restart_and_diagnostics.py \
    tests/fvm/test_cartesian_config.py tests/fvm/test_mesh_contracts.py \
    -k 'not geometry_independence and not repeated_cartesian and not section_extrusion'
```

## Follow-up: patch-assignment overflow at `dx=0.17`

The reported `cfmesh_template.py` warning was reproduced by promoting
`RuntimeWarning` to an exception during the reference mesh's patch assignment.
Projected face centres can lie exactly on a candidate surface, making their
squared distance zero. The normal-alignment score used
`sqrt(max_distance_squared / max(distance_squared, tiny)) * alignment`.
Dividing by the binary64 `tiny` floor can overflow before the square root,
even when the final score fits in binary64. A perpendicular normal can then
produce `inf * 0 = NaN`, interfering with patch selection.

The implementation now takes the two square roots before dividing. This
preserves the score, distance floor, and candidate-order tie when all
distances are zero, without the overflowing intermediate ratio. It requires
no change to the domain bounds or refinement algorithm.

Validation for this follow-up:

- Decimal arithmetic at 100-digit precision verifies the scores for zero,
  subnormal, ordinary, and maximum finite binary64 squared distances.
- Complete reference meshes at `dx=0.17` and `dx=0.25` pass with runtime
  warnings and NumPy overflow/invalid/divide errors treated as failures.
- Additional full meshes at `dx=0.16` and `dx=0.18` pass under the same policy.
- The focused run passes 26 tests; the non-slow mesh-parity run passes 56.
  The union is 58 distinct tests, including the two complete-mesh regressions.
- The active Conda installation was patched with the same one-file change,
  preserving its other installed code and backing up the original module.
  An isolated Python process, using the actual reference `setup.py` and the
  installed package, builds `dx=0.17` without runtime warnings: 6,404 cells,
  20,960 faces, all seven named patches, and positive cell volumes. The
  requested external bounds remain `[-7.5,15] × [-7.5,7.5] × [-7.5,7.5]`.

```bash
python -m pytest tests/mesh_parity/test_cfmesh_patch_assignment.py \
    tests/mesh_parity/test_cfmesh_size_reporting.py
python -m pytest tests/mesh_parity -m 'not slow' -W error::RuntimeWarning
```
