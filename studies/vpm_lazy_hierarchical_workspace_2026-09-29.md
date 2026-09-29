# Hierarchical VPM scratch allocation

The declared `max_n_particles` remains a hard particle ceiling. FMM previously allocated its multipole, interaction-list, LBVH, and traversal fields for that ceiling during `bind`, before any source particles existed. At a ceiling of one million sources, the field-payload estimator reports 2,356,185,088 bytes for a 4,096-target batch, excluding Taichi runtime and driver overhead. The new one-source FMM scratch estimate is 799,528 bytes. The source treecode likewise previously allocated its declared ceiling at its first query, including for a tiny cloud.

FMM now binds one-source scratch and grows geometrically when a stage or target query has more active sources. The LBVH treecode starts at at most 8,192 sources (also capped by the declared ceiling) and grows geometrically from actual demand. Both workspaces own their fields in disposable `ti.FieldsBuilder` trees; on growth, the old scratch is synchronized and destroyed before new fields are created. Source-tree reuse keys are invalidated. Target batch storage remains capped by its independent internal traversal batch size, so a small source cloud can still serve many targets efficiently. No public configuration or physical arithmetic changed.

Native checks completed:

- CPU: FMM growth `1→2→5` with bit-identical repeated two-source outputs, prior FMM and LBVH SNode trees destroyed, hard-cap rejection before allocation, no-source target output, and deliberate near-list overflow. Focused tests passed. The full FMM/core/slip suite reached 60 passing cases; its one failure was a test spy attached to the initial one-source tree before expected growth. Prewarming to the measured four-source capacity preserved its one-build assertion and the isolated test passed on rerun.
- CPU: `tests/vpm/test_target_workspace_batched.py` passed all five tests, including a source-tree growth from 8,192 to 16,384 slots, field release, direct-vortex numerical comparison, and independent large target batching.
- CPU: `test_fmm_restart_and_repeated_run_match_over_twenty_accepted_steps` passed uninterrupted, repeated, and 10+10 backup/restart state parity.
- CUDA: tiny private 2% Taichi pool checks passed FMM `2→5→2` stage evaluation and LBVH `8,192→8,193` capacity replacement with finite/repeated outputs. These validate field lifetime only; production FMM still does not advertise CUDA.

The physical 80,958-source DVH restart and production GPU memory savings remain to be measured separately. The tiny CUDA checks do not qualify FMM far-field accuracy, large interaction queues, CUDA restart replay, or production CUDA support.
