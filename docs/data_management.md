# Data Management (Git + DVC + Nextcloud)

OpenONDA keeps the **code lightweight in git** and the **large simulation data in
[DVC](https://dvc.org/)**, with [Nextcloud](https://nextcloud.com/) as the DVC
remote so results sync across all your machines. DVC stores only small metadata
(`*.dvc` files, hashes) in git; the heavy files live in the remote cache.

This single document replaces the older `dvc_workflow.md`,
`DVC_QUICK_REFERENCE.md` and `multi_computer_workflow.md`.

---

## 1. What goes where — the folder policy

The rule of thumb: **anything needed to _run_ a case is git; anything _produced_
by a run is DVC; regenerable scratch is ignored.**

| Category | Examples | Tracked by |
|---|---|---|
| **Run inputs** (small, canonical) | setup `*.py`, `allrun.sh`/`allplot.sh`, `system/`, `constant/transportProperties`, `constant/turbulenceProperties`, **`constant/polyMesh.orig/`** (canonical pre-built mesh), **`0.orig/`** (canonical initial conditions), `assets/`, `Readme.md` | **git** |
| **Visualizations** (small) | `figures/*.png`, `*.pdf`, `*.svg` | **git** |
| **Results** (large, produced) | `solution/` (VPM + coupler diagnostics & backups), `referenceFlow/` (gate reference data), OpenFOAM **reconstructed time directories** (`0.08`, `0.16`, …, `15`) | **DVC** |
| **Scratch** (regenerable) | `processor*/` (decomposed — rebuild with `decomposePar`), `constant/polyMesh/` (runtime copy of `polyMesh.orig`), the runtime `0/` copy, `*.foam`, `log.*`, `VTK/`, `postProcessing/` | **ignored** (neither) |

Why these choices:

- **`0.orig/` in git, `0/` ignored.** `0.orig` is the canonical initial state;
  `0/` is copied from it (and, for coupler cases, patched) at run time — fully
  regenerable, so it is not committed.
- **`polyMesh.orig/` in git, `polyMesh/` ignored.** OFW cases ship the canonical
  mesh as `polyMesh.orig` (these tutorials' meshes are small, ≤ a few MB); the
  runtime `polyMesh/` is just a copy (or a fresh `cartesianMesh` build for the
  FVM-VPM cases).
- **Reconstructed time dirs in DVC, `processor*/` ignored.** `allrun.sh` always
  runs `reconstructPar`, so the reconstructed time directories are the canonical
  output worth backing up. The decomposed `processor*/` data is intermediate and
  regenerable from `decomposePar`, so storing it too would only double the
  remote footprint.

All of this is enforced by the repository-root [`.gitignore`](../.gitignore);
the DVC side is applied by [`scripts/dvc_add_solutions.sh`](../scripts/dvc_add_solutions.sh).

---

## 2. One-time setup on a machine

```bash
# DVC ships in the OpenONDA conda env; otherwise: pip install dvc
dvc remote list
# storage  /home/flavio/Nextcloud/Research/Simulation_Backup  (default)
```

The committed remote points at this workstation's Nextcloud path. **On any other
machine** where Nextcloud lives elsewhere, override it locally (this is written
to `.dvc/config.local`, which is git-ignored, so it never clobbers other
machines):

```bash
dvc remote modify --local storage url /your/path/to/Nextcloud/Research/Simulation_Backup
```

Then pull whatever data you need:

```bash
git pull
dvc pull                 # everything, or:
dvc pull tutorials/VPM/rotorFlow/solution.dvc   # one case
```

---

## 3. Daily workflow

### Start of a session
```bash
git pull && dvc pull
```

### After running a simulation
Use the helper — it finds every untracked result directory under `tutorials/`
(`solution/`, `referenceFlow/`, reconstructed time dirs), `dvc add`s them, and
stages the `*.dvc` / `.gitignore` files. It does **not** push or commit:

```bash
scripts/dvc_add_solutions.sh             # or: scripts/dvc_add_solutions.sh tutorials/VPM
git status                               # review what was staged
dvc push                                 # upload data to Nextcloud
git commit -m "Add <case> results"
git push
```

To track a single directory by hand instead:

```bash
dvc add tutorials/VPM/myCase/solution
dvc push
git add tutorials/VPM/myCase/solution.dvc tutorials/VPM/myCase/.gitignore
git commit -m "Add myCase solution" && git push
```

### The golden order
`dvc add` → `dvc push` → `git add *.dvc` → `git commit` → `git push`.
Skip `dvc push` and other machines get a `.dvc` file pointing at data they
cannot fetch.

---

## 4. Multi-computer workflow (two machines, one Nextcloud)

Both machines share the same Nextcloud folder, so the remote is the single
source of truth. Each machine sets its own remote path **once** (§2, via
`--local`); everything else is the routine below.

### Machine that PRODUCED data — back it up (so nothing is lost)

Run this whenever a simulation finished and you want the results saved:

```bash
scripts/dvc_add_solutions.sh     # dvc add every new solution/, referenceFlow/, time dir
dvc push                         # 1) upload the data to Nextcloud  ← the actual backup
git add -A                       # 2) stage the new *.dvc + .gitignore (and any code/inputs)
git commit -m "Add <case> results"
git push                         # 3) share the metadata
```

**What to stage:** the `*.dvc` and `.gitignore` files the script created — never
the result directories themselves (they are git-ignored by design). `git add -A`
does the right thing because the data dirs are ignored.

### Machine that CONSUMES data — fetch it

```bash
git pull        # gets the .dvc metadata (and any new cases/inputs)
dvc pull        # downloads the actual results from Nextcloud into the workspace
```

### Nothing-lost checklist

- Data is safe **only after `dvc push` succeeds** — `git push` alone shares
  pointers to data that may not be uploaded yet. Always `dvc push` first.
- Before wiping or re-cloning a machine, confirm `dvc status -c` reports nothing
  to push (everything is already in the remote).
- Do **not** run the same case on both machines at once: re-running overwrites
  `solution/` and produces conflicting `.dvc` hashes. Coordinate, then on the
  second machine `dvc pull` to sync.
- `dvc push`/`pull` is idempotent and content-addressed: re-pushing already-saved
  data is a cheap no-op, so push liberally.

---

## 5. Common commands

```bash
dvc status                       # local vs tracked
dvc status -c                    # local vs remote (cloud)
dvc pull <target.dvc>            # fetch one case
dvc pull --force                 # re-download, overwriting local
dvc remote list                  # show remotes
du -sh ~/Nextcloud/Research/Simulation_Backup/   # remote size
```

### Free local disk without losing data
```bash
rm -rf tutorials/VPM/oldCase/solution/     # delete local copy
dvc pull tutorials/VPM/oldCase/solution.dvc  # restore later from remote
```

---

## 6. Troubleshooting

**`dvc status` shows "deleted" outputs** — you removed a tracked result dir
locally; restore with `dvc pull` (or re-`dvc add` if you meant to update it).

**"file already tracked by git"** when adding to DVC — remove it from git first,
then add to DVC:
```bash
git rm -r --cached path/to/dir && git commit -m "Untrack; move to DVC"
dvc add path/to/dir
```

**Different Nextcloud path per machine** — use the `--local` override in §2;
never commit a machine-specific path to `.dvc/config`.

**Cache not found on pull** — the data was never pushed. On the producing
machine run `dvc push`, and confirm files appear under
`~/Nextcloud/Research/Simulation_Backup/files/`.

---

## See also

- [`scripts/dvc_add_solutions.sh`](../scripts/dvc_add_solutions.sh) — batch result tracker
- [`scripts/install/build_solvers.sh`](../scripts/install/build_solvers.sh) — (re)build the OFW native extension
- Root [`.gitignore`](../.gitignore) — the git side of the folder policy
