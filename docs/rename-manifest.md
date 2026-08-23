# Canonical filesystem and stored-artifact manifest

This is the final layout after the one-way nomenclature conversion. Only the
canonical paths and schemas below are supported.

## Package roots

| Canonical path | Status |
| --- | --- |
| `source/solvers/fvm/` | complete |
| `source/solvers/vpm/` | complete |
| `tutorials/fvm/` | complete |
| `tutorials/vpm/` | complete |
| `tutorials/coupled_fvm_vpm/` | complete |

## Tutorial paths

| Canonical path | Status |
| --- | --- |
| `tutorials/vpm/lamb_oseen_vortex/` | complete |
| `tutorials/vpm/vortex_ring/` | complete |
| `tutorials/vpm/vortex_interactions/` | complete |
| `tutorials/vpm/quadcopter/` | complete |
| `tutorials/vpm/rotor_flow/` | complete |
| `tutorials/vpm/delta_wing/` | complete |
| `tutorials/vpm/flat_plate/` | complete |
| `tutorials/coupled_fvm_vpm/naca4412_flow/` | complete |
| `tutorials/coupled_fvm_vpm/cylinder_shedding_flow/` | complete |
| `tutorials/coupled_fvm_vpm/cube_flow/reference_flow/` | complete |

All setup files, plotting launchers, and helper scripts within these roots use
lower snake case. Line-sample filenames use `centreline`.

## Archive and checkpoint paths

| Canonical path | Meaning | Status |
| --- | --- | --- |
| `tutorials/coupled_fvm_vpm/cube_flow/run_archives/20260822T193932Z/` | tracked historical run | complete |
| `tutorials/coupled_fvm_vpm/cube_flow/run_archives/20260822T203309Z/` | tracked historical run | complete |
| `tutorials/coupled_fvm_vpm/cube_flow/samples_archive/` | archived sample output | complete |
| `tutorials/coupled_fvm_vpm/cube_flow/solution/checkpoints/` | restartable coupled state | complete |

## Stored schemas

| Artifact | Canonical contract | Status |
| --- | --- | --- |
| FVM serial NPZ | format 6 plus `physical_field_schema_version` | complete |
| FVM partitioned rank NPZ | full state names; partition manifest format 4 | complete |
| VPM HDF5 | checkpoint format 6.0; exact `particles/particle_volume` and other canonical particle datasets plus solver attributes | complete |
| VPM XDMF | lower snake-case attributes plus schema information | complete |
| Coupled manifest | format 8; four named and hashed artifacts | complete |
| Coupled boundary NPZ | boundary schema 1; explicit `has_*` fields | complete |
| VTK-family output | canonical array names plus schema field data | complete |
| CSV/JSON/JSONL output | full canonical headers and keys | complete |
| ParaView state | canonical physical array identifiers | complete |

The repository contains no alternate path tree, compatibility schema, or
conversion utility.
