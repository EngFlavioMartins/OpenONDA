# Reproducible tutorial results

## Storage and use

Each archived case stores `assets/results/manifest.json` in Git and a lossless
`data.tar.gz` (automatically split for large new archives) in Git LFS. The manifest records source revision, scientific status,
saved solver clocks and SHA-256 hashes of every file. Current samples are retained
without temporal or spatial decimation. Unreferenced superseded restart histories
remain local; accepted continuation segments must remain in any published archive.

Install Git LFS before cloning, install OpenONDA and the documented rendering
dependencies, then run the case's `./allplot.sh`. The first command restores missing
result directories. It verifies the archive and every member before publishing
directories. Existing local result directories prevent restoration of the entire
group: archived samples are never combined with a different local solution. To
compare an archive with a local run, use a separate clone.

The numerical pip package excludes these large archives. Installed tutorial
templates remain small and produce their own results through `allrun.sh`.
`allcontinue.sh` retains local output and resumes native backups; plotting archive
restoration is confined to `allplot.sh`. Plot archives contain the saved states
needed by their figures, not necessarily every state required for a new simulation.

Git stores small LFS pointers rather than multi-gigabyte binary history. Frozen
archives should be published at meaningful result milestones, never after each
solver step. LFS still stores each changed archive version in full; its remote
storage and transfer allowances must cover published snapshots. See
[GitHub's LFS storage model](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).
No history rewriting or deletion is required by this scheme.

The archive packer is `openonda.results.pack_results`: it accepts a source
directory, bundle destination, explicit case-relative files, scientific status
and provenance. It rejects symlinks, unsafe paths and files changed during packing.
Freeze files from running simulations before packing. Record metadata first and
include only accepted checkpoints through that recorded clock. Live sample tails
remain stored; plots select accepted history where their validation requires it.

## Coverage and verification

The nine frozen bundles contain 4,550 files: 6.56 GB before compression and
4.82 GB stored in LFS. Their Git pointers total 1,202 bytes; inspectable checksum
manifests are tracked separately. Sizes below use decimal MB.

| Case | Archive | Scientific status |
| --- | ---: | --- |
| Lamb–Oseen | 1,526.6 MB | Complete deterministic cases and accepted ensembles |
| Vortex ring | 3.6 MB | Transposed variants complete; two health-stop histories |
| Vortex interactions | 312.9 MB | Selective viscosity complete; three health-stop histories |
| Flat plate | 88.6 MB | All twenty cases complete |
| Delta wing | 1,115.8 MB | Partial snapshot; recorded solver time 6.025 s |
| Rotor | 1,589.0 MB | Partial snapshot; recorded solver time 6.552 s |
| Quadcopter | 32.8 MB | Partial history ending on a strain health check |
| Coupled cube and reference | 153.2 MB | 88 matched states through 22 s; forces through 22.15 s |
| Taylor–Green | 0.1 MB | Default run completed to 0.05 s |

Archive provenance identifies the code revision used when packaging; original run
configuration and state remain in the exact saved metadata/checkpoints. Packaging
does not retroactively establish an unrecorded simulation source revision.

The active delta-wing and rotor simulations were left running during capture. Their archives are
explicitly partial snapshots, not claims that the requested final time was reached.
The vortex-ring, interaction and quadcopter health-stop states retain that status.
Missing production results are not replaced by synthetic data.

The installed wheel restored all nine bundles into a separate disk-backed
workspace, checking all 4,550 file hashes. The installed archive implementation
matches the source byte-for-byte. The 62-test installed-package/tutorial suite
passed; the final archive suite passes 23 tests, including sharding, corruption,
local-run preservation, and removal of obsolete archive parts. The bounded
solver-policy audit also passed 41 wall-retry, backend and restart tests.

The relocated vortex-ring launcher completed with ten figures. Cube validation
checked all 88 cached states; all four figure families rendered at the final
matched state at their normal 400 DPI. Full cube rendering produces hundreds of
figures and was not run to completion. Other relocated suites are recorded as
they finish; interrupted runs are not counted as successful verification.

After restoration, the complete Taylor–Green, quadcopter and vortex-interaction
launchers also passed (one, three and four PNGs respectively). Flat plate rendered
seven analytical figures but its scene command could not discover `pvpython` in
the restarted verification environment. That rendering dependency remains required;
the archive's data checks passed. Further plotting checks were stopped after the
host memory incident below, rather than reported as complete.

### Verification memory incident

The kernel recorded a global out-of-memory event on 29 September at 13:37:
ChatGPT was killed, followed by the rotor solver process (PID 298126). Concurrent
verification processes and RAM-backed `/tmp` scratch files added avoidable memory
pressure. The verification environment was moved to disk under the ignored
`build/` directory, obsolete scratch copies were removed, and remaining plotting
jobs were stopped. `/tmp` usage fell from about 3.3 GB to 0.5 GB. No user solver
process was deliberately stopped or restarted.

The rotor's latest surviving native checkpoint is `vpm_001096.h5`, with recorded
time 6.576 s. The published candidate archive remains the earlier, internally
consistent 6.552 s snapshot. Future verification on this host must use disk-backed
scratch space and run serially without competing with production simulations.

Implementation code is committed as `20e9577c`. The nine dataset archives and
manifests remain staged: automatic approval review requires explicit confirmation
of their 4.82 GB inclusion before their commit. Nothing has been pushed or uploaded.

## Implementation checklist

- [x] Agree lossless retention of current samples and separate superseded history.
- [x] Implement deterministic archives, checksums, safe restoration and local-run preservation.
- [x] Integrate restoration as one command in each plotting launcher.
- [x] Keep large result archives outside pip distributions and installed templates.
- [x] Fix rotor plotting of newer live CSV tails without modifying data.
- [x] Produce and plot the default Taylor–Green result.
- [x] Build and restore checksum-verified archives of all available scientific results.
- [ ] Verify plotting from a separate Git clone with an installed wheel.
- [ ] Commit the verified implementation and LFS pointers.
- [ ] Publish the commit and associated LFS objects before another machine clones from GitHub.

## Earlier requests retained

- [x] CPython 3.11 installation, README animation and outside-checkout package verification.
- [x] Remove machine-specific import paths and tutorial byte/RSS/soft-limit knobs found in the audit.
- [x] Use the shared thesis fonts, palette and validated figure layout.
- [x] Native latest-backup continuation and fresh-run cleanup contracts have regression coverage.
- [x] General wall-correction retry, output reconciliation and backend policy tests pass.
- [x] Cylinder geometry and prepared interpolation improvements have implementation evidence in the
  [execution report](../../studies/cylinder_execution_report.md).
- [ ] Qualify the original CUDA DVH/treecode OOM on the affected GPU; CPU/mocked tests do not establish this.
- [ ] Complete rotor, delta-wing and coupled-cube runs and investigate remaining numerical health stops.
- [ ] Obtain missing airfoil, boundary-layer, FVM cube, cylinder IBM, step-profile and coupled-NACA results.
- [ ] Complete the Re=150 three-dimensional cylinder grid/span and injection/exchange sensitivity campaign;
  retain the [original numerical plan](../../studies/cylinder_3d_accuracy_performance_plan.md), including
  measured runtime qualification of the finest case against the 12-hour target.
- [ ] Qualify any future reduced output cadence against spectra, transient peaks and plotting convergence.

The archive mechanism resolves transport and path portability. It does not resolve
unfinished simulations or establish grid independence. Those numerical requirements
remain explicitly open until supported by completed runs and comparisons.
