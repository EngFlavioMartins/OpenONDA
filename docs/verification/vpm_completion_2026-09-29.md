# VPM completion audit — 29 September 2026

This is a snapshot of local results, not a claim of numerical convergence.
The current tutorial setups and `allcontinue.sh` launchers were compared with
the native metadata and latest backups. All 98 current backups passed the
solver's HDF5 validation (schema, configuration hash, particle arrays and finite
values); stored VLM and VPM clocks agree. Archived restart branches were excluded.
The completion table was refreshed later that day from all 98 native checkpoint
clocks: 90 completed states, six health stops and two active simulations.
At the 20:45 UTC refresh, delta wing had stopped after a keyboard interrupt;
rotor was still running. Only those two changing histories were reread for this
refresh; the earlier 98-backup validation remains the coverage of the full audit.

| Tutorial / variant | Latest backup step / target | Physical time / target [s] | Result and next action |
| --- | ---: | ---: | --- |
| 01 Lamb–Oseen: vortex, dipole, merging; CS/DVH/GBD | 927 / 927 each | 29.973 / 29.973 | Complete; no continuation needed. |
| 01 Lamb–Oseen: RWM | 927 / 927 each | 29.973 / 29.973 | All 58 members complete (16 vortex, 28 dipole, 14 merging). |
| 02 vortex ring: DNS direct | 108 / 3000 | 2.16 / 60 | Stopped at the 25° misalignment limit. |
| 02 vortex ring: DNS mixed | 321 / 3000 | 6.42 / 60 | Stopped at the 25° misalignment limit. |
| 02 vortex ring: DNS transposed, LES transposed | 3000 / 3000 each | 60 / 60 | Complete. |
| 03 vortex interactions: baseline | 1573 / 2400 | 5.89875 / 9 | Stopped at the 0.12 divergence limit. |
| 03 vortex interactions: particle splitting | 1654 / 2400 | 6.2025 / 9 | Stopped at the 0.12 divergence limit. |
| 03 vortex interactions: Pedrizzetti relaxation | 1671 / 2400 | 6.26625 / 9 | Stopped at the 0.12 divergence limit. |
| 03 vortex interactions: selective eddy viscosity | 2400 / 2400 | 9 / 9 | Complete. |
| 04 flat plate: 10 moving angles | 197 / 197 each | 2.4625 / 2.4625 | Complete. |
| 04 flat plate: 10 static angles | 192 / 192 each | 2.4 / 2.4 | Complete. |
| 05 delta wing | 2780 / 8000 | 6.95 / 20 | Interrupted after accepted step 2781; the last saved state is step 2780. Continue-ready when no replacement process is active. |
| 06 rotor flow | 1188 / 1667 | 7.128 / 10.002 | Active; continue only after its current process exits. |
| 07 quadcopter | 296 / 2304 | 0.04625 / 0.36 | Stopped on the strain health limit; stretching diagnosed, refinement candidate requires a new clean qualification. |

The rotor setup requests 10 s, rounded to 1667 steps of 0.006 s (10.002 s).
The Lamb–Oseen ensemble tables also meet the 7.5% MCSE criterion: maximum
relative errors are 6.7353% (vortex), 6.8199% (dipole), and 7.0706% (merging).

## Continuation commands

These are the continuation commands for delta-wing and rotor runs. Delta wing
was interrupted at the latest inspection; rotor remains active. Check for a
replacement process before starting delta wing, and do not duplicate rotor.
Run interrupted cases from the repository root, one at a time:

```bash
(cd tutorials/vpm/05_delta_wing && ./allcontinue.sh)
(cd tutorials/vpm/06_rotor_flow && ./allcontinue.sh)
```

Both current setups match their saved numerical configurations, and both
latest checkpoints passed validation. The refreshed native clocks above have
corresponding live processes. These commands resume the last saved
state. They do not guarantee that a later physical state will remain resolved.
Do not use `allrun.sh` to resume; it cleans the existing output first.

The incomplete variants in 02, 03 and 07 are health-limit stops, not ordinary
interruptions. Their native logs record the failures listed above. Repeating
`allcontinue.sh` with unchanged numerical inputs is not a remedy for those
limits. They need numerical investigation before being described as complete;
no health threshold was relaxed by this audit. The completed variants remain
usable, and early stops in comparison cases can themselves be comparison results.

## README animation

The original 107-frame, 1280×576 delta-wing GIF was recovered unchanged from
commit `24cc275a`, blob `684e5725bfe471f4518d094261b37b6a1e444ba1`, into
`tutorials/vpm/05_delta_wing/assets/delta_wing_30fps.gif`. All frames decoded
successfully. SHA-256:
`3cd7b47adb8bd1ca1e49de323122414340b3d8d559e92a7fb14d2c0ffe9a04c5`.
It illustrates an archived run and does not imply that the current partial
local solution has reached 20 s. Current-run rendering still writes to
`figures/`, with its own source-data sidecar.

## Installation verification

CPython 3.11 is now the sole supported minor version, with patch updates
allowed (`Requires-Python: ==3.11.*`). Metadata, installer admission, runtime
version constant, Conda recipes, CI and installation documentation agree.
Python 3.11 has wheels for both [Taichi 1.7.4 on Linux and Apple Silicon](https://pypi.org/project/taichi/1.7.4/#files)
and [Taichi 1.7.1 on Intel macOS](https://pypi.org/project/taichi/1.7.1/#files).

Verification on this Linux machine used CPython 3.11.15:

- Built the source distribution and wheel with `python -m build --no-isolation`.
  Both include the recovered GIF with its original SHA-256.
- Installed the wheel and all dependencies from PyPI into a fresh temporary
  virtual environment without access to system site-packages.
- Ran `python -I -m openonda.verify_install --require-site-packages` from `/tmp`:
  exit 0. Verified 19 tutorial resources, 10 direct tutorial commands, a rendered
  plot, a Numba-balanced Cartesian mesh (1392 cells), native FVM iterative solves,
  CPU VPM stepping and checkpoint restart. Taichi 1.7.4 and Numba 0.67.0 were used.
- `pip check`: no broken requirements.
- Copied the delta-wing tutorial using the installed materializer and verified
  the recovered GIF hash and 107-frame count.
- Pip's interpreter-target check accepted 3.11 and rejected 3.12 and 3.13 for
  the actual wheel. Installer tests also reject other minor versions and PyPy.
- All 17 interpreter/installer tests and 23 installed-tutorial tests passed;
  Ruff, formatting, shell syntax and whitespace checks passed.

The macOS CI jobs remain configured for Apple Silicon and Intel using Python
3.11; macOS execution was not performed on this Linux host. These installation
checks do not constitute full-duration tutorial simulation or grid-convergence
verification.

## Quadcopter follow-up

Direct induction of the saved step-296 cloud reproduced the strain violation.
One nearby particle had amplified its strength by about 167 times while its core
radius grew by only 1.71 times. The setup now uses conservative filament
refinement at each step. This changes its numerical configuration: do not bypass
the restart mismatch to apply it to the old cloud, whose disabled refinement
never recorded lineage references. An isolated clean run is required before the
new setup can be called stable. See [the diagnosis](../../studies/quadcopter_stability_audit.md).

The isolated clean CUDA qualification has since passed the original step-296
failure with the corrected lineage initialization. Its bounded 384-step run is
still active, and the full 2304-step particle capacity is not qualified. See
[the growth audit](../../studies/quadcopter_refinement_growth_audit_2026-09-29.json).
The protected rotor history has not yet passed 7.5 s; its current refinement
and runtime evidence are recorded in
[the rotor audit](../../studies/rotor_refinement_status_2026-09-29.md).
