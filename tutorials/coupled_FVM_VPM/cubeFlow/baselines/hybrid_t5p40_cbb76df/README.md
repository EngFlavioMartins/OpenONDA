# Hybrid cube-flow baseline at t = 5.40

This directory preserves the sampled output from the interrupted production
FVM-VPM cube-flow run stopped on 2026-08-15 before the coupling investigation.

- Safety checkpoint: `cbb76df776797147d92ef15e53d619f2c1694bb9`
- Last complete sampled FVM frame: `t = 5.40`, step 540
- Last complete coupling/FVM diagnostic: `t = 5.45`, coupling step 109
- Sample files: 117
- The `samples/` copy was verified byte-for-byte against the live case with
  `diff -qr` immediately after copying.
- `diagnostics/` contains the run metadata, FVM and coupling JSONL histories,
  and the FVM, VPM, and coupler logs needed to audit the baseline.

The cubeFlow `allclean.sh` script does not remove `baselines/`.
