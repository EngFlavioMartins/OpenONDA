# Reproducible tutorial results

## Storage and use

The [30 September sample snapshot](../../studies/samples_snapshot_2026-09-30/README.md)
additionally preserves every current tutorial sample file, including study
histories and superseded branches. It restores into a separate destination using
`python -m openonda.results restore DESTINATION --bundle BUNDLE_DIRECTORY`.
The per-case bundles below remain matched plotting snapshots.

Each archived case stores `assets/results/manifest.json` in Git and a lossless
`data.tar.gz` (automatically split for large new archives) as versioned GitHub
release assets. Missing payloads download automatically on restoration and are
verified against the manifest before being cached locally. The manifest records source revision, scientific status,
saved solver clocks and SHA-256 hashes of every file. Current samples are retained
without temporal or spatial decimation. Unreferenced superseded restart histories
remain local; accepted continuation segments must remain in any published archive.

Clone the `development` branch, install OpenONDA and the documented rendering
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

Git stores checksum manifests rather than multi-gigabyte payloads. Frozen
archives should be published at meaningful result milestones, never after each
solver step. On 30 September, GitHub rejected LFS publication because the
repository exceeded its LFS budget. The current checkout therefore uses
[versioned release assets](https://github.com/EngFlavioMartins/OpenONDA/releases/tag/tutorial-results-2026-09-30),
and has no LFS pointers that could make cloning fail. Existing commits retain
their original identities and historical LFS pointers. The superseded delta-wing
archive from the outgoing history is also mirrored in the data release.
GitHub supports [large binary release assets](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases);
all uploaded assets are below its 2 GiB per-file limit. No history rewrite or
account billing change is required.

The archive packer is `openonda.results.pack_results`: it accepts a source
directory, bundle destination, explicit case-relative files, scientific status
and provenance. It rejects symlinks, unsafe paths and files changed during packing.
Freeze files from running simulations before packing. Record metadata first and
include only accepted checkpoints through that recorded clock. Live sample tails
remain stored; plots select accepted history where their validation requires it.

## Earlier archive coverage and verification

The measurements below describe the original LFS packaging and its local tests.
The same payload bytes are now distributed by the release mechanism above.

The twelve frozen bundles contain 4,614 files: 6,644,037,070 bytes before
compression and 4,838,433,580 bytes stored in LFS. Git tracks small LFS pointers
and inspectable checksum manifests. Sizes below use decimal MB.

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
| Boundary layer | 1.4 MB | Default run completed to 8 s; Blasius errors remain above targets |
| Step profile | 0.9 MB | Default run completed to 12 s; reattachment 3.59h, not grid-qualified |
| IBM cylinder | 13.3 MB | Default run completed to 60 s; mean drag in cited band, recirculation length outside band |

Archive provenance identifies the code revision used when packaging; original run
configuration and state remain in the exact saved metadata/checkpoints. Packaging
does not retroactively establish an unrecorded simulation source revision.

The active delta-wing and rotor simulations were left running during capture. Their archives are
explicitly partial snapshots, not claims that the requested final time was reached.
The vortex-ring, interaction and quadcopter health-stop states retain that status.
Missing production results are not replaced by synthetic data.

The installed wheel restored the original nine bundles into a separate
disk-backed workspace and verified every member hash. The later delta-wing
archive repair retains every existing file unchanged and adds three genuine
native fields required by its PVD collection. Both archive creation and
restoration now reject missing VTK dependencies. The current combined archive,
launcher, input-reader and thesis-style regression suite passes 99 tests.

Complete relocated plotting launchers have passed for vortex ring (ten figures),
vortex interactions (four), quadcopter (three), flat plate (seven analytical
figures and its ParaView scene), Taylor–Green (one), and delta wing (five).
The full coupled-cube launcher returned zero and all 353 PNGs decoded:
four families of 88 frames plus coupling diagnostics. Its paired cached fields
cover t=0.25–22 s; raw force histories retain their own sampling times.
Rendering does not establish statistical convergence of these histories.

Boundary-layer and step defaults were executed unchanged with the installed
solver, then archived. Ordinary `allplot.sh` restored them from a fresh Git export
and reproduced all four original image hashes. Their strict thesis geometry
checks and visual inspections passed. The boundary-layer comparisons remain
outside their stated accuracy targets; no scientific acceptance limit changed.
See [the recorded executions](../../studies/fvm_small_cases_archive_verification_2026-09-29.json).

The corrected rotor’s complete plotting launcher returned zero under a detached
supervisor. All five analytical PNGs and all 274 GIF frames decoded; representative
finite performance/field plots and an animation frame passed visual inspection.
This renders archived accepted output and does not establish full-horizon rotor
stability. See [the rotor execution record](../../studies/rotor_archive_plot_verification_2026-09-29.json).
Lamb–Oseen’s complete launcher returned zero; all seven PNGs decoded and
representative analytical and particle figures passed visual inspection. See
[the Lamb execution record](../../studies/lamb_archive_plot_verification_2026-09-29.json).
The completed IBM cylinder result is archived without reducing the force or
field histories. Its `allplot.sh` restored and rendered the genuine saved data;
the corrected field plots use actual stretched mesh cells. Mean Cd=1.7681 is
inside its cited band, but recirculation length 2.137D is outside 1.55–1.70D,
so the case remains scientifically unqualified. The cube reference's separate
plotting launcher also restores the parent bundle; its available 15–30 s
histories have zero complete force cycles and cannot qualify grid statistics.
No missing production inputs have been fabricated for the airfoil, FVM cube,
coupled NACA or full cylinder studies.

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

Implementation code is committed as `20e9577c`. After explicit approval of the
4.82 GB dataset, the nine archives and manifests were committed as `51f9ebf9`.
Nothing has been pushed or uploaded.

A separate disk-backed local clone of `51f9ebf9` retrieved all nine LFS payloads
automatically. Streaming SHA-256 verification matched every archive against its
manifest: 4,822,718,770 bytes in total. The clone shares ordinary Git objects with
the local source repository, but has its own checkout and hydrated LFS payloads.
This verifies local clone transport; remote availability still requires publishing
the commit and LFS objects.

From that clone, Taylor–Green's unmodified `./allplot.sh` restored its archived
solution and generated `figures/taylor_green_decay.png` successfully using the
installed wheel in a normally activated Python 3.11 environment. Imports resolved
to `site-packages`; no source-path overrides were used. The regenerated figure was
also visually inspected. Subsequent plotting coverage is recorded above.

## Implementation checklist

- [x] Agree lossless retention of current samples and separate superseded history.
- [x] Implement deterministic archives, checksums, safe restoration and local-run preservation.
- [x] Integrate restoration as one command in each plotting launcher.
- [x] Keep large result archives outside pip distributions and installed templates.
- [x] Fix rotor plotting of newer live CSV tails without modifying data.
- [x] Produce and plot the default Taylor–Green result.
- [x] Build and restore checksum-verified archives of all available scientific results.
- [x] Verify a complete plotting launcher from a separate Git clone with an installed wheel.
- [x] Complete plotting verification for every archived tutorial from a separate Git clone or export.
- [x] Verify automatic LFS hydration and archive checksums in a separate local clone.
- [x] Commit the verified implementation and LFS pointers.
- [x] Publish the commits and result payloads for other machines: release assets
  replaced LFS distribution on 30 September, preserving every original commit
  ID. A normal fresh clone and complete sample download passed verification;
  see the [publication record](../../studies/samples_snapshot_2026-09-30/verification.json).

## Earlier requests retained

- [x] CPython 3.11 installation, README animation and outside-checkout package verification.
- [x] Remove machine-specific import paths and tutorial byte/RSS/soft-limit knobs found in the audit.
- [x] Use the shared thesis fonts, palette and validated figure layout.
- [x] Native latest-backup continuation and fresh-run cleanup contracts have regression coverage.
- [x] General wall-correction retry, output reconciliation and backend policy tests pass.
- [x] Cylinder geometry and prepared interpolation improvements have implementation evidence in the
  [execution report](../../studies/cylinder_execution_report.md).
- [x] Qualify the original CUDA DVH/treecode OOM on the affected GPU: 80,958 particles restored and eight repeated evaluations held at 690 MiB with matching velocities. See [the CUDA record](../../studies/cuda_dvh_restart_2026-09-29.json).
- [ ] Complete rotor, delta-wing and coupled-cube runs and investigate remaining numerical health stops.
- [x] Obtain and archive genuine completed boundary-layer and step-profile defaults.
- [ ] Obtain missing airfoil, FVM cube and coupled-NACA results; the IBM cylinder is archived, but its recirculation-length discrepancy still needs investigation.
- [ ] Complete the Re=150 three-dimensional cylinder grid/span and injection/exchange sensitivity campaign;
  retain the [original numerical plan](../../studies/cylinder_3d_accuracy_performance_plan.md), including
  measured runtime qualification of the finest case against the 12-hour target.
- [ ] Qualify any future reduced output cadence against spectra, transient peaks and plotting convergence.

The archive mechanism resolves transport and path portability. It does not resolve
unfinished simulations or establish grid independence. Those numerical requirements
remain explicitly open until supported by completed runs and comparisons.

### Final installed-package recheck

The complete flat-plate `allplot.sh` now exits successfully in the fresh local
clone, including its ParaView scene. The verification environment explicitly
provides the documented renderer on PATH. No tutorial import paths were altered.
The rendered wake figure was inspected.
