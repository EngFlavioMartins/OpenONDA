# Historical plot-input audit — 2026-09-29

## Scope and evidence standard

This is a read-only audit of local Git history. It treats only versioned solver
outputs and their direct metadata as inputs. Rendered figures are not evidence
of recoverable numerical data. No LFS object was fetched or extracted.

## Absent data

No historical Git tree contains a `solution/`, `samples/`, `assets/results/`
bundle, native VTK/CSV output, or equivalent numerical archive for any of:

- `tutorials/fvm/airfoil_flow`
- `tutorials/fvm/cube_flow`
- `tutorials/fvm/cylinder_ibm`
- `tutorials/coupled_fvm_vpm/03_naca4412_flow`

Commit `20e9577c` only added `python -m openonda.results restore` calls to
these tutorials' `allplot.sh` scripts. No corresponding result manifest or
archive exists locally in Git history. Their current plotters require the
missing solver directories and samples, so they cannot be reproduced from
historical figures or from a missing bundle.

The full cylinder campaign likewise has no separately tracked historical
campaign output. Its current working-tree results are not historical Git
inputs and must not be used as if they were.

## Locally recoverable cylinder reference CSV and metadata

The only historical numerical candidate is the reference-only cylinder archive
under:

`tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/samples/`

It was introduced in `68156df59515d201ba0f984aca4effc70f127e31` (2026-09-13),
changed in `24cc275ab2caf426c2e017455509286cd7f567fe`, and removed from the
tree in `b16fc96e86303468dc086fc29e0c4434183cde1c` (2026-09-20). The normal Git
objects retain `grid_run.json`, force and probe histories, and centreline and
transverse-profile CSVs for `very_coarse`, `coarse`, `medium`, and `fine`.

The recorded end time is `59.99999999998448` s for all four grids. Their
recorded wall spacings and cell counts are:

| Grid | h (m) | Cell count |
| --- | ---: | ---: |
| very_coarse | 0.06 | 22,680 |
| coarse | 0.05 | 31,405 |
| medium | 0.04 | 67,284 |
| fine | 0.03 | 151,740 |

The PVD metadata records field times from 0 through approximately 60 s at a
0.5 s cadence. The old reference README describes statistics and profiles over
the common final-half window, hence at most a 30--60 s historical window is
supported by these data. It cannot cover the current 40--100 s qualification
window.

## Missing LFS/native fields

The archived midspan VTS fields are Git-LFS pointers, not native VTS payloads
in this checkout. `git lfs ls-files --all -l` marks the historical cylinder VTS
objects with `-`, indicating that their payloads are absent locally. The
13-September tree declares 645 sample files and 280,488,965 bytes; these are
tree/LFS metadata, not proof that the native fields can be read here. A
network-authorized LFS retrieval would be required before field plots could be
recreated.

## Physics and campaign compatibility

The `68156df` setup is a reference-only, body-fitted FVM calculation. It uses
`D = 1`, `U = 1`, `nu = 1/150`, and therefore Re=150. Its outer box is
`[-8, 24] x [-10, 10] x [-0.5, 0.5]`, a 1.0-D span. It uses inlet/outlet in x,
slip conditions at `ymin/ymax` and `zmin/zmax`, and a cylinder wall. The span
is explicitly extruded with `ceil(1/(4h)) + 1` levels; the z planes are slip,
not periodic. The archived setup runs to 60 s with nominal dt=0.001, adaptive
cap 0.004, force samples every 0.02 s, profiles every 0.1 s, and slices every
0.5 s.

The current Re=150 resolved-3D-slip-span campaign uses a 0.96-D physical slab
(`z = +/-0.48D`), 100-s reference and coupled horizons, and a 40--100-s force
statistics window. The coupled configuration additionally uses VPM and
`SlipSlabInduction`; it is not the old reference-only setup. The old archive
therefore may support provenance-labelled, reference-only inspection within
its own 0--60 s clock, but it must not be substituted into current resolved-3D
campaign plots, comparisons, statistics, or qualification gates.
