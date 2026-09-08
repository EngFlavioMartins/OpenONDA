# Test suite

The suite protects independently meaningful numerical and behavioral contracts.
Choose a tier rather than running every expensive qualification during an edit:

```bash
python -m pytest tests -m "not qualification and not slow and not gpu"
python -m pytest tests -m "qualification and not gpu" --numerical-report=numerical-results.json
python -m pytest tests -m "not gpu"
```

GPU qualifications need a suitable backend. MPI/external OpenFOAM/cfMesh
comparisons need their separate runtimes; they are not prerequisites for
installing or using the native serial solvers. CPU tests retain analytical
solutions, spatial/temporal convergence, moment conservation, restart fidelity,
invalid geometry, solver interoperability, mutation rollback and output integrity.
No numerical tolerance was relaxed during this cleanup. The measured slow
generic-mesh builds are marked `slow`; CI has a separate slow CPU job so these
checks and restart tests are still executed.

## Maintenance decisions

The table accounts for all 98 test modules present at the start of the September
2026 repository audit. CONSOLIDATE retains meaningful assertions in the named
module or replaces snapshots with a broader behavioral contract. REMOVE means
the original file was removed; its reason identifies redundant protection.
KEEP modules preserve their independent numerical/error scenarios. Helpers and
reference fixtures remain alongside the relevant tests.

| Original module | Decision | Protection or replacement |
| --- | --- | --- |
| `coupler/test_common_m4_viscous_lifecycle.py` | KEEP | common m4 api is scheme agnostic and same state is idempotent; gbd and common m4 are stable across repeated physical lifecycles; related numerical and failure cases. |
| `coupler/test_coupled_backup.py` | KEEP | post renewal particle history is published outside the rolling backup; config difference paths are recursive and distinguish missing from none; related numerical and failure cases. |
| `coupler/test_cube_flow_setup.py` | CONSOLIDATE | Keep timing resolution and scientific acceptance/restart gates; replace a long fixed-tuning/absence snapshot with physical-time alignment invariants. |
| `coupler/test_cube_reference_grid_study.py` | KEEP | three completed runs create numbers and plot |
| `coupler/test_flux_handoff.py` | KEEP | vorticity transport flux has the conservative viscous sign; nonzero diffusive flux is included in the emitted circulation budget; related numerical and failure cases. |
| `coupler/test_flux_handoff_vpm_integration.py` | KEEP | external flux pair is injected then advected only by the real vpm; oblique external flux retains tangential phase before free vpm advection; related numerical and failure cases. |
| `coupler/test_fvm_consistency_band.py` | KEEP | consistency rate is c1 and confined to outer buffer; consistency rate is transit scaled and time step capped; related numerical and failure cases. |
| `coupler/test_fvm_vpm_smoke.py` | KEEP | coupled fvm vpm two steps; coupling step limit rejects invalid values |
| `coupler/test_gbd_projected_renewal.py` | KEEP | post gbd geometric authority absorbs lattice roundoff; exact current basis never triggers blanket support births; related numerical and failure cases. |
| `coupler/test_gbd_recovery.py` | CONSOLIDATE | `coupler/test_stable_renewal.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `coupler/test_interpolation_qualification.py` | KEEP | interpolation is affine exact and second order on graded meshes |
| `coupler/test_lattice_transfer.py` | CONSOLIDATE | Quadratic reproduction at 10,000 random phases already checks partition and first moment on complete support. |
| `coupler/test_physical_coupling.py` | KEEP | eta zero width is hard ownership and positive width is c1 blend; hard replacement removes inner particles injects cell circulation and preserves outer; related numerical and failure cases. |
| `coupler/test_reporting.py` | CONSOLIDATE | Remove formatting snapshots; keep numerical field gates and closure serialization. |
| `coupler/test_stable_renewal.py` | CONSOLIDATE | Same M4 weights, aligned scatter, and buffer bound covered in lattice_transfer and physical_coupling. Keep production wrapper, prune closure, float32 and repeated renewal tests. |
| `fvm/test_cartesian_config.py` | CONSOLIDATE | Generic surface/configuration builds protect geometry independence without policing import text. |
| `fvm/test_cartesian_extrusion.py` | CONSOLIDATE | `fvm/test_mesh_contracts.py`. Retain assertions together in the behavior-focused module. |
| `fvm/test_cartesian_mesher_phase0.py` | CONSOLIDATE | `fvm/test_mesh_contracts.py`. Replace development-stage filenames, lexical bans, and duplicate API/immutability checks with five geometry builds, topology and positive-volume validation; invalid surfaces retained. |
| `fvm/test_cartesian_surface_recovery.py` | CONSOLIDATE | `fvm/test_mesh_contracts.py`. Retain assertions together in the behavior-focused module. |
| `fvm/test_cube_reference_flow.py` | REMOVE | Consolidated in tutorials/test_reference_cases.py: local geometry, requested refinement, solver boundaries/output ownership. Removed exact four-file interfaces, import-name bans, campaign stdout and fixed recipe snapshots. |
| `fvm/test_cylinder_mesh_entry.py` | REMOVE | Consolidated in tutorials/test_reference_cases.py: local geometry, requested refinement, solver boundaries/output ownership. Removed exact four-file interfaces, import-name bans, campaign stdout and fixed recipe snapshots. |
| `fvm/test_cylinder_reference_asset.py` | KEEP | tracked cylinder surface crosses span with caps outside domain |
| `fvm/test_cylinder_reference_cartesian.py` | REMOVE | Consolidated in tutorials/test_reference_cases.py: local geometry, requested refinement, solver boundaries/output ownership. Removed exact four-file interfaces, import-name bans, campaign stdout and fixed recipe snapshots. |
| `fvm/test_cylinder_reference_tools.py` | REMOVE | Consolidated in tutorials/test_reference_cases.py: local geometry, requested refinement, solver boundaries/output ownership. Removed exact four-file interfaces, import-name bans, campaign stdout and fixed recipe snapshots. |
| `fvm/test_cylinder_study_campaign.py` | REMOVE | Consolidated in tutorials/test_reference_cases.py: local geometry, requested refinement, solver boundaries/output ownership. Removed exact four-file interfaces, import-name bans, campaign stdout and fixed recipe snapshots. |
| `fvm/test_geometry_chunks.py` | CONSOLIDATE | `fvm/test_mesh_contracts.py`. Retain assertions together in the behavior-focused module. |
| `fvm/test_logging.py` | CONSOLIDATE | Keep stdout destination isolation; cosmetic shared block layout is not a solver contract. |
| `fvm/test_manufactured_gradient_qualification.py` | KEEP | lsq gradient has second order spatial convergence |
| `fvm/test_matrix_workspace_boundary_layout.py` | KEEP | interior only partition uses an interior only workspace; interior only partition still enters boundary flux reductions |
| `fvm/test_mesh_validation.py` | CONSOLIDATE | `fvm/test_mesh_contracts.py`. Retain assertions together in the behavior-focused module. |
| `fvm/test_mixed_velocity_boundary.py` | KEEP | mixed boundary exactly reconstructs divergence free linear fields; tangential gradient has no normal component |
| `fvm/test_nonorthogonal_pressure_correction.py` | KEEP | linear pressure field cancels rhie chow flux on a nonorthogonal face; linear pressure field cancels boundary rhie chow flux; related numerical and failure cases. |
| `fvm/test_openfoam_poly_mesh.py` | KEEP | import counts cells that appear only as neighbours; export sorts internal faces and reverses swapped owners |
| `fvm/test_restart_and_diagnostics.py` | KEEP | restart restores backward time history; run manifest serializes sampler configuration; related numerical and failure cases. |
| `fvm/test_time_step_control.py` | KEEP | maximum courant control reduces immediately and limits growth; maximum courant control applies the configured step ceiling; related numerical and failure cases. |
| `mesh_parity/test_cfmesh_mesh_optimisation.py` | KEEP | quality scans accept an orthogonal unit cube; bad face scan detects an inward boundary face; related numerical and failure cases. |
| `mesh_parity/test_cfmesh_octree.py` | KEEP | exact dyadic object request uses native strict bound; sparse lookup matches dense queries and morton order; related numerical and failure cases. |
| `mesh_parity/test_native_check.py` | KEEP | checkmesh failed summary overrides zero exit status; checkmesh mesh ok summary passes; related numerical and failure cases. |
| `test_installed_tutorials.py` | CONSOLIDATE | General catalog materialization and actual local runner cover portability; removed campaign strings and fixed numeric presets, retain full-time diffusion workspace bounds. |
| `test_lamb_oseen_rwm_statistics.py` | KEEP | column projection recovers one gaussian blob and circulation; merging pair requires peak saddle contrast above ensemble noise; related numerical and failure cases. |
| `test_public_api.py` | CONSOLIDATE | Replace the 150-line exact export allowlist with resolvable exports, essential interfaces and real case construction. |
| `test_storage_output.py` | KEEP | write precision preserves integers and makes paraview safe float16; surface sampler writes compact paraview readable vts; related numerical and failure cases. |
| `tutorials/test_axisymmetric_reference.py` | KEEP | separable poisson residual |
| `tutorials/test_lamb_oseen_launcher.py` | KEEP | campaign phase logs preserve exit status |
| `tutorials/test_output_schemas.py` | KEEP | cube plot metadata accepts only supported coupling schemas; lamb oseen surface reader round trips the sampler schema; related numerical and failure cases. |
| `tutorials/test_vortex_core_agreement.py` | KEEP | core identity survives overtaking and peak rank changes; uniform distance score weights rings equally and keeps radius error; related numerical and failure cases. |
| `tutorials/test_vortex_core_sections.py` | KEEP | plane sampler round trip preserves curl orientation and physical time; setup and study share initial periodic and final plane sampling; related numerical and failure cases. |
| `tutorials/test_vortex_interaction_study.py` | KEEP | initial gaussian tail does not amplify the core peak; fixed core diffusion control preserves the physical viscosity; related numerical and failure cases. |
| `tutorials/test_vortex_interactions.py` | KEEP | ring pair is a translated symmetric toroidal cloud; all cases share the transposed les rk3 baseline; related numerical and failure cases. |
| `tutorials/test_vortex_ring_launcher.py` | KEEP | vortex ring campaign uses current variants and preserves exit status; vortex ring plot campaign runs all modules and stops on failure |
| `vpm/test_axisymmetric_field.py` | KEEP | half plane circulation matches independent numerical integral; angular integral matches independent cartesian gaussian quadrature; related numerical and failure cases. |
| `vpm/test_backup_storage.py` | KEEP | vpm backup has one fixed restart schema; vpm restart preserves compute precision and freestream; related numerical and failure cases. |
| `vpm/test_boundary_element_state_refresh.py` | KEEP | panel refresh resolves current state without advancing history; vpm refresh applies to full and boundary only panel scopes; related numerical and failure cases. |
| `vpm/test_case_lifecycle.py` | CONSOLIDATE | Duplicate public-name/signature checks removed; actual construction, rejection, lifecycle and failure paths retained. Kernel rejection covered by vortex kernel contract. |
| `vpm/test_core_numerical_qualification.py` | KEEP | unbounded fft energy converges to direct particle integrals; gaussian biot savart velocity and gradient match the closed form; related numerical and failure cases. |
| `vpm/test_core_spreading_projection.py` | KEEP | core spreading skips subprecision moment correction; coupled update keeps symmetric core spreading without subcycling |
| `vpm/test_coupled_runge_kutta.py` | KEEP | tableaux retain the three coupled schemes; every rk stage uses one common position strength state; related numerical and failure cases. |
| `vpm/test_dvh_contract.py` | KEEP | dvh rejects nonuniform effective viscosity; dvh zero viscosity is an identity for off lattice particles; related numerical and failure cases. |
| `vpm/test_evolution_transaction.py` | KEEP | failed physical phase does not commit solver clock; failed physical phase makes solver terminally invalid |
| `vpm/test_flow_integral_backend.py` | KEEP | large gaussian cloud uses the fourier integral backend; energy rate is defined between continuity preserving measurements; related numerical and failure cases. |
| `vpm/test_fmm_device.py` | KEEP | fmm workspace estimate is linear in capacity; particle capacity warning is emitted at eighty percent; related numerical and failure cases. |
| `vpm/test_fmm_hierarchy.py` | KEEP | fmm tree owns deterministic stage geometry and core metadata; multipole and local translations preserve leading coefficients; related numerical and failure cases. |
| `vpm/test_free_space_integrals.py` | KEEP | single blob has exact unbounded energy and viscous power; free space energy does not change when the fft box grows; related numerical and failure cases. |
| `vpm/test_gaussian_core_remeshing.py` | KEEP | grid projection removes gradient preserves solenoidal field and mean; old checkpoint default remains compatible but cannot enable projection; related numerical and failure cases. |
| `vpm/test_gbd_high_order_remeshing.py` | KEEP | six point weights reproduce polynomials through degree five; device scatter matches independent cardinal polynomial deposition; related numerical and failure cases. |
| `vpm/test_gbd_prune_conservation.py` | KEEP | prune recovery preserves vortex strength and impulse moments; no prune recovery is an exact noop; related numerical and failure cases. |
| `vpm/test_global_regeneration_threshold.py` | KEEP | regeneration threshold modes return one cloud wide value; non global regeneration threshold mode is rejected; related numerical and failure cases. |
| `vpm/test_health_limits.py` | KEEP | misalignment uses current curl independently of backup vorticity; health limits enforce finite state and cfl after field refresh; related numerical and failure cases. |
| `vpm/test_import_side_effects.py` | KEEP | vpm import does not redirect streams or change traceback limit |
| `vpm/test_induction_contract.py` | REMOVE | Both cases test only Python value objects or the test-defined spy itself, without calling production induction. stage_rhs and coupled_runge_kutta exercise production stage-state propagation. |
| `vpm/test_logging_cadence.py` | CONSOLIDATE | Remove cosmetic whitespace and text-layout contracts. Preserve warning visibility, accepted-step truth, throttling and final status. |
| `vpm/test_near_core_tree_targets.py` | KEEP | gaussian target velocity and jacobian have the correct origin limit |
| `vpm/test_output_configuration.py` | CONSOLIDATE | `vpm/test_output_contracts.py`. Historical absence/signature checks duplicate API coverage; serialization and resume are exercised by backup tests. |
| `vpm/test_output_directories.py` | CONSOLIDATE | `vpm/test_output_contracts.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_output_manager.py` | CONSOLIDATE | `vpm/test_output_contracts.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_panel_diagnostic_scheduling.py` | KEEP | diagnostic is off by default; diagnostic runs only on its schedule; related numerical and failure cases. |
| `vpm/test_panel_far_field.py` | KEEP | far field matches direct sum beyond acceptance radius; below threshold uses exact path; related numerical and failure cases. |
| `vpm/test_panel_linear_solver_convergence.py` | CONSOLIDATE | Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_panel_moving_qualification.py` | KEEP | translating sphere is galilean invariant; vpm incident field is distinct from body velocity and matches freestream; related numerical and failure cases. |
| `vpm/test_panel_multibody.py` | KEEP | appended bodies keep geometry ranges and group ids; each body can move independently; related numerical and failure cases. |
| `vpm/test_panel_particle_coupling.py` | KEEP | every active particle is deflected; deflection does not depend on particle ordering; related numerical and failure cases. |
| `vpm/test_panel_solver_memory_guard.py` | CONSOLIDATE | `vpm/test_panel_linear_solver_convergence.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_panel_solver_sphere_analytic.py` | KEEP | no penetration holds on the surface; surface speed matches the analytic sphere solution; related numerical and failure cases. |
| `vpm/test_panel_stl.py` | CONSOLIDATE | `vpm/test_panel_geometry.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_panel_stl_audit.py` | CONSOLIDATE | `vpm/test_panel_geometry.py`. Historical heuristic parity adds no protection beyond closed/concave body orientation, winding, and invalid topology cases. |
| `vpm/test_particle_cache_revision.py` | CONSOLIDATE | `vpm/test_particle_state.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_particle_initialization.py` | KEEP | rectangular distribution preserves spacing and sigma over h; single widnall mode has the requested centreline and slope; related numerical and failure cases. |
| `vpm/test_pressure_state.py` | CONSOLIDATE | `vpm/test_particle_state.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_realignment_cache.py` | CONSOLIDATE | `vpm/test_particle_state.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_sampler_execution.py` | CONSOLIDATE | `vpm/test_output_contracts.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_sampling_schedule.py` | CONSOLIDATE | `vpm/test_output_contracts.py`. Replace five selected timing examples with one independent cadence/offset predicate over 13,527 combinations; retain invalid schedule checks. |
| `vpm/test_stabilization_schedules.py` | KEEP | combined stabilization schedule is representable; pedrizzetti relaxation stops at end step; related numerical and failure cases. |
| `vpm/test_stage_rhs.py` | KEEP | stage rhs passes stage time and state to induction and external provider; callable stage contribution receives time and all coupled stage arrays; related numerical and failure cases. |
| `vpm/test_state_strict_schema.py` | CONSOLIDATE | `vpm/test_particle_state.py`. Keep invalid restart clock validation; valid clock round-trip is already covered by backup storage. |
| `vpm/test_taichi_kernels.py` | KEEP | taichi kernels do not rebind arguments |
| `vpm/test_turbulence_orchestration.py` | KEEP | zero smagorinsky skips viscosity kernels; parallel les statistics use active particles and reset each call |
| `vpm/test_vlm_mesh.py` | CONSOLIDATE | `vpm/test_panel_geometry.py`. Retain assertions in the shared behavior-focused module; no numerical cases removed. |
| `vpm/test_vortex_kernel_contract.py` | CONSOLIDATE | Remove constructor/signature snapshots; analytic kernels and actual tensor contractions retained. |

## Added and strengthened protection

- All tutorial templates materialize with no generated result trees; shell
  launchers parse successfully. Nested reference cases build configuration
  after moving away from their parent tutorials and use local geometry.
- A subprocess in an unrelated directory loads a user-edited local module with
  relative imports and preserves its nonzero exit status.
- Five generic surface geometries now undergo topology validation, positive
  finite-volume checks and geometry quality validation after meshing.
- Sampling offsets and cadence are compared with an independent predicate over
  13,527 combinations, replacing five individual examples.
- The public API test constructs real cases and checks resolvable exports,
  replacing a duplicated list of roughly 140 names.

- Process isolation checks now include Numba compilation after runtime
  configuration, protecting FVM-to-VPM interoperability in one process.

## Remaining qualification gaps

A package install smoke cannot certify long turbulent wakes or vortex-merging
statistics. Full production-horizon body-flow comparisons, hardware-specific
GPU checks and multi-rank restart/parity qualification remain separate work.
The tiny uniform hybrid tutorial intentionally has no emitted vorticity;
nonzero transfer dynamics remain protected by the flux-handoff and Gaussian
projection integration/qualification tests. Source-archive and wheel contents
must be checked whenever new asset types or generated directories are added.
Test packages have explicit namespace markers; neither tests nor the standalone
mesh-acceptance runner insert repository paths into Python imports.

See [repository_audit.md](../repository_audit.md) for measured suite size,
installation and execution results, and observed failures.
