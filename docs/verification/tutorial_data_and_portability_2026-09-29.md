# Tutorial data and portability audit — 29 September 2026

## Status and scope

The audited local `samples/` trees contain approximately **5.04 GB in 3,915
files** (decimal units). No tutorial sample files are currently tracked in Git.
The existing ignore rules exclude both sample directories and scientific data
extensions. A fresh clone therefore cannot reproduce all tutorial figures.
Some figures additionally require ignored solution snapshots and metadata.

This audit preserves the simulation data and sampling frequencies. The user
requested a joint decision before changing retention. The code commit does
not add 5 GB to Git, enable LFS, delete restart histories, or declare the
fresh-clone plotting requirement complete. Unrelated cylinder campaign work
and newly generated figures are outside this commit.

## Data requiring a retention decision

The inventory includes superseded restart histories and covers the existing
non-campaign tutorial sample trees. Files can continue to grow during runs.

| Case | Files | Size | Main cost |
| --- | ---: | ---: | --- |
| VPM 01 Lamb–Oseen | 1,127 | 1,842 MB | 1,030 sampled VTK grids, mostly ensemble fields |
| VPM 02 vortex ring | 12 | 8.6 MB | Diagnostic CSV histories; already modest |
| VPM 03 vortex interactions | 200 | 336 MB | 188 sampled VTK grids |
| VPM 04 flat plate | 100 | 410 MB | Repeated chordwise and spanwise loading tables |
| VPM 05 delta wing | 730 | 1,506 MB | Two current chordwise tables total 671 MB; 705 MB superseded histories |
| VPM 06 rotor | 237 | 635 MB | Current chordwise tables total 258 MB; 276 MB superseded histories |
| VPM 07 quadcopter | 73 | 87 MB | Loading tables and sampled planes |
| Coupled cube | 1,424 | 145 MB | Profile histories, fields and comparison cache |
| Coupled cube reference | 12 | 74 MB | Profile histories |

Superseded histories account for **982 MB**, leaving about **4.06 GB** of
current samples. Superseded files must be distinguished from any accepted
continuation segments referenced by a run's lineage manifest before archival.

Read-only gzip trials on representative files gave:

| Input | Original | Compressed | Interpretation |
| --- | ---: | ---: | --- |
| Delta-wing front chordwise CSV | 335.3 MB | 88.2 MB | Large lossless reduction |
| Rotor blade 0 chordwise CSV | 86.0 MB | 23.8 MB | Large lossless reduction |
| Flat-plate chordwise CSV | 20.9 MB | 4.8 MB | Large lossless reduction |
| Coupled cube centreline CSV | 37.4 MB | 14.6 MB | Useful lossless reduction |
| Lamb–Oseen ensemble VTS | 6.08 MB | 4.39 MB | Base64 XML overhead remains |
| Vortex-interaction VTS | 1.86 MB | 1.74 MB | Already effectively compressed |

The Lamb–Oseen ensemble writer now uses the project's existing compressed
raw VTK writer for future output. A structured-grid round trip retained the
points, arrays and ensemble metadata exactly. Existing data were not rewritten.
Git also compresses objects: gzip file-size savings alone are not estimates of
repository-history savings. Repeatedly committing evolving binary archives
can still make history expensive.

### Proposed options — not applied

1. Preserve all current samples losslessly, keep unreferenced superseded
   histories local, and design a portable archive/read path before adding data.
   Avoid including repeated static panel identifiers and coordinates in every
   stored loading frame. This requires reader, continuation and export tests.
2. Preserve all histories with Git LFS. This requires agreement on storage and
   transfer costs and changes to the README's current LFS-skip clone workflow.
3. First qualify lower sampling rates with spectra and plot-convergence checks,
   then agree which data to retain. Keep restart backup frequency separate from
   plot sampling: scene renderers currently consume some full particle backups.

### Sampling candidates, requiring numerical qualification

Nyquist requires a sampling rate above twice the highest frequency of interest;
it is not a sufficient accuracy target for transient peaks or harmonics. Start
with 10–20 samples per shortest period of interest, check spectra and integrated
loads against the original histories, and apply appropriate anti-aliasing before
decimation. No proposed rate below is an approved solver change.

| Case | Observed cadence | Candidate investigation |
| --- | --- | --- |
| Delta wing | Loading every 0.0025 s, 400 samples per 1 Hz heave cycle; 288 rows/frame | Test 0.025 s loading output (40/cycle), retaining force history at solver cadence |
| Rotor | Loading every 0.006 s; about 43 samples/blade-passage period; 132 rows/frame | Compare 0.012 s and 0.024 s; the existing 0.06 s plane cadence resolves only about 4.3 points/blade period and is unsuitable as a blanket loading cadence |
| Quadcopter | Loading every 0.00015625 s; 48 samples/blade-passage period; 48 rows/frame | Test every three steps (16/blade period) only after obtaining a stable reference run |
| Flat plate | Every 0.0125 s, only about 9.6 samples during the 0.12 s ramp | Preserve transient cadence; assess the later nearly steady interval separately |
| Lamb–Oseen/interactions | Spatial grids dominate | Qualify field cadence against merger/core motion; do not infer bandwidth from file count |

The vortex-ring plot's `n_skip` changes marker spacing while retaining all line
data. It is not evidence that the diagnostic histories are oversampled.
Animation display thinning also does not establish that numerical checkpoint
particles or spatial field samples can be discarded.

## Fresh-clone plotting dependencies

| Tutorial | Inputs still needing a portable data archive |
| --- | --- |
| VPM 01 Lamb–Oseen | Sample PVD/VTS series, deterministic and ensemble CSV diagnostics, run metadata; initial and final merging GBD particle checkpoints for its scene |
| VPM 02 vortex ring | Diagnostic CSVs and run metadata for four variants; final LES transposed particle checkpoint for scenes |
| VPM 03 vortex interactions | Four core-section PVD/VTS series, diagnostic CSVs and run metadata; the literature CSV is already tracked |
| VPM 04 flat plate | Force and loading CSVs and run metadata for 20 runs; selected impulse histories; final moving 12-degree particle checkpoint and VLM surface |
| VPM 05 delta wing | Force/integral CSVs, three wake-plane series, metadata and any accepted lineage segments; coupled checkpoints for final animation |
| VPM 06 rotor | Force/loading/integral/profile CSVs, two wake-plane series, metadata and coupled checkpoints for impulse/animation |
| VPM 07 quadcopter | Surface forces, integrals, two plane series and run metadata |
| FVM airfoil | Force history, surface pressure CSV and velocity VTU |
| FVM boundary layer | Profiles and skin-friction CSVs |
| FVM cube | Force history and velocity/vorticity VTU |
| FVM cylinder IBM | IBM forces and FVM snapshots |
| FVM step profile | Field and reattachment-history CSVs |
| FVM Taylor Green | Decay-history CSV |
| Coupled cylinder | Completed production campaign reports |
| Coupled NACA | IBM force history |
| Coupled cube and reference | Matched fields/profiles, mesh/PVD/metadata/diagnostics for both runs |

The coupled-cube comparison manifest includes absolute paths as cache
fingerprints. Its preparation stage rebuilds that cache from raw solution
inputs; copying the manifest alone is insufficient. A portable plotting
archive must include the consumed inputs or replace the cache with exported
comparison data, with a relocation test. That remains outstanding.

The thesis plotters require LaTeX, dvipng and New PX fonts. ParaView scene
renderers additionally require `pvpython`. These are external rendering
dependencies; numerical solver installation does not require them. No
machine-specific imports or fallback fonts were added to bypass them.

## Code changes and verification

- CPython 3.11 is the supported minor version, with patch updates allowed;
  package metadata, installation checks, Conda environments and CI agree.
- The delta-wing README animation was recovered unchanged into its case assets
  and included in the source distribution, wheel and tutorial materializer.
- The benchmark no longer edits `sys.path`. Copied tutorial setups execute via
  the installed package, including edited local assets, outside the repository.
- Lamb–Oseen's setup exposes case and diffusion scheme. Ensemble orchestration
  lives in its existing dedicated asset script; launchers call that script.
- Five FVM plot helpers, Taylor Green and coupled NACA use the thesis theme and
  validated canvas geometry. Export preserves the validated layout, including
  colorbars. Missing required inputs now fail clearly instead of silently
  reporting a successful plot run.
- Flat-plate ParaView screenshots explicitly assign the view to a layout and
  check screenshot creation. The actual renderer produced a readable PNG.
- The Lamb–Oseen merger scene reads its saved step-zero particle cloud rather
  than comparing a reconstructed cloud against restart metadata. The actual
  renderer produced both initial and final scenes (3,618 and 118,435 particles).
  Four regression tests cover the resumed state and rejected invalid backups.

Synthetic fixture renders checked all five FVM helpers, Taylor Green, NACA and
an equal-aspect colorbar layout. These tests validate rendering only; no
synthetic data were substituted for missing tutorial results. Actual FVM and
coupled plotting attempts did not establish a complete production suite:
eight cases lack inputs, and coupled cube reported zero matched comparison
states before two bounded runs timed out (90 and 120 seconds).

Actual VPM rendering results:

| Case | Result |
| --- | --- |
| Lamb–Oseen | Main comparisons, surfaces and energy rendered; corrected final merger command subsequently rendered both scenes |
| Vortex ring | Analytical plots rendered; sandbox denied MPI sockets for ParaView, then the scene command succeeded with normal host permissions |
| Vortex interactions | Entire `allplot.sh` exited 0 and regenerated four figures |
| Flat plate | Seven analytical figures rendered; corrected ParaView scene command succeeded with normal host permissions |
| Delta wing | `allplot.sh` exited 0 with five partial figures; final animation remains unavailable until the run completes |
| Rotor | `allplot.sh` failed: saved run time 6.48 s disagrees with loading CSV tails at 6.492 s; data were neither trimmed nor relabelled |
| Quadcopter | `allplot.sh` exited 0 and rendered three figures from the available history; solver health stop remains unresolved |

A successful plot of an available partial history is not evidence that its
simulation is complete or numerically converged. The FVM production plots,
coupled-cube comparison and rotor inconsistency remain outstanding, as does
the data archive required for fresh-clone plotting.

### Installation and regression checks

The selected Git index was exported into a clean temporary source tree, without
local results or unrelated untracked campaign scripts. Its source distribution
and wheel both built successfully. The wheel was installed into a separate
CPython 3.11.15 virtual environment without system site-packages.

`python -I -m openonda.verify_install --require-site-packages` ran successfully
from `/tmp`: imports resolved to the installed package, tutorial resources and
direct commands were available, a plot rendered, and meshing, native FVM/VPM
steps and checkpoint restart passed. `pip check` found no broken requirements.
The restored animation's SHA-256 matches the original in both distribution
formats. Linux CPU execution was tested; this audit did not run macOS or CUDA.

The clean-tree regression run passed **79 tests** covering interpreter support,
installed tutorials, copied setup/launcher execution, tutorial style, RWM
statistics, snapshot discovery and merger initial-state loading. A further
**38 continuation/cleanup tests** passed. Ruff, Python formatting, modified
shell-script syntax and Git whitespace checks passed. An obsolete launcher
expectation was updated to assert the exact dedicated ensemble command,
including its realization count and convergence flag.

## Remaining decisions and work

- [x] Measure sample sizes, compression and current output cadence.
- [x] Verify installation outside the checkout and recover the README asset.
- [x] Run existing plot launchers and fix the reproduced rendering defects.
- [ ] Agree retention and transport for current and superseded sample data.
- [ ] Qualify any reduced sampling cadence against spectra and plot convergence.
- [ ] Package the selected plot inputs and metadata with portable references.
- [ ] Resolve the rotor data-clock mismatch and obtain missing FVM/coupled inputs.
- [ ] Run every complete plotting suite from a fresh clone of the chosen archive.
