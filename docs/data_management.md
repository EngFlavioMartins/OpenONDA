# Data management with Git and DVC

OpenONDA keeps source code and small canonical inputs in Git. Large numerical
results are tracked by DVC and stored in the configured remote.

## Folder policy

| Data | Examples | Owner |
|---|---|---|
| Run inputs | setup scripts, meshes/STL, configuration JSON, plotting tools | Git |
| Small reference figures | PNG, PDF, SVG | Git |
| Results | `solution/`, `samples/`, checkpoints, reference datasets | DVC |
| Scratch | caches, logs, VTK staging, temporary partitions | ignored |

Native FVM cases can generate their meshes at startup, so generated mesh data
is treated as output unless a tutorial deliberately ships a small canonical
mesh fixture.

## Configure a machine

```bash
dvc remote list
dvc remote modify --local storage url /path/to/your/dvc/storage
dvc pull
```

The local override is written to `.dvc/config.local` and is never committed.

## Save new results

```bash
scripts/dvc_add_solutions.sh
dvc status
dvc push
git status
git commit -m "Update <case> results"
git push
```

The safe order is `dvc add` → `dvc push` → Git commit → Git push. A committed
`.dvc` pointer is not a backup until `dvc push` succeeds.

To retrieve results on another machine:

```bash
git pull
dvc pull
```

Do not run the same case concurrently on two machines when both write the same
result directory.

## Useful commands

```bash
dvc status
dvc status -c
dvc pull path/to/result.dvc
dvc push path/to/result.dvc
dvc remote list
```

If DVC reports that an output is already tracked by Git, remove that output
from Git's index, commit the ownership change, and then run `dvc add`.
