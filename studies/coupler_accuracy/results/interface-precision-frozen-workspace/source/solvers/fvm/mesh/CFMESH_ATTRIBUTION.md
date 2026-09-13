<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# cfMesh attribution and current-scope note

OpenONDA's `CartesianMesher` is an independent, solver-native
implementation inspired by the openly documented **cfMesh** Cartesian
workflow. The principal cfMesh developer is Dr. Franjo Juretić and the
original copyright holder is Creative Fields. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the source links and the
Phase 0 provenance boundary.

The typed implementation lives in `mesh/cartesian/` and is the production
path used by the migrated tutorials. The former adaptive monolith and
cylinder-specific O-grid class are no longer defined or exported from the
solver. General Cartesian construction is exposed only through
`openonda.fvm.mesher.CartesianMesher`.

The implementation must distinguish architectural inspiration from any
source translated or adapted directly. No cfMesh source has been copied or
translated in Phase 0. It must not describe silent partial snapping, Gmsh
delegation, or cylinder-only layers as a cfMesh robustness behavior. The exact
upstream study commit and file-level provenance must be added before direct
translation, if any, is introduced.
