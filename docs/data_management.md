# Result data policy

Tutorial and solver output belongs below the selected case directory:

```text
<case_dir>/
├── solution/   local solver state, logs, checkpoints, and field archives
└── samples/    versioned forces, probes, lines, surfaces, and diagnostics
```

Every tutorial `samples/` tree is committed so post-processing data remains
available after cloning on another device. This includes CSV/JSON diagnostics,
PVD collections, and sampled VTK fields from ordinary cases and grid studies.
Review regenerated samples before committing so accidental or incomplete runs
do not replace a qualified dataset.

Generated `solution/`, `runs/`, build, cache, and environment directories remain
local and must not be committed. Compact reference tables, summary JSON, and
figures may also be committed when a test or durable validation record uses
them and their generation command is recorded.

Large restart states and raw solver-state series should live in a release asset,
data repository, or archival service with a checksum and provenance note. Keep
them out of `samples/`. Before committing, inspect:

```bash
git status --short
git diff --check
find . -type d -name __pycache__ -prune
```

The tutorial cleanup scripts remove local solver output and may regenerate
versioned samples. Inspect `git diff` afterward; do not point cleanup commands
at a parent directory or a directory containing irreplaceable data.
