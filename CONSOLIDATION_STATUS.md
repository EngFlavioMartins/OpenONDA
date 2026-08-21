
---

## 4. Auditor corrective pass (post `6806e66`)

Commits: `366d953` (stale `_last_V_wake` guard + rename artifacts/docstrings + regression tests) → `f13508b` (`extra="forbid"` state models + strict-schema tests) → `8cde03f` (`density`/`reference_speed` params; fixed scalar speed passed as VLM reference-velocity **vector** in KJ force path; dead kernel arg removed) → `b980abc` (pruned/excluded diagnostics renamed to vortex-strength terminology, incl. serialized JSON keys and tutorial readers; |Γ| comments → |α|) → `90b98ec` (VPM-owned ν consistency guard for attached VLM + tests; `flow_models.py` public factories renamed `viscosity→kinematic_viscosity`, typo `max_diturb_modes→max_disturb_modes`; stale tutorial kwargs fixed incl. broken `add_vortex_particles` calls in lambOseen/rings/vortexRing tutorials).

All fast gates green after each commit. Known leftover: `initialize_single_mode_toroidal_ring(viscosity=…)` definition site not located in source tree (vendored?); call sites left consistent with it.
