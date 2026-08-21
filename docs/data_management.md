# Result data policy

Tutorial and solver output belongs below the selected case directory:

```text
<case_dir>/
├── solution/   solver state, logs, checkpoints, and field archives
└── samples/    forces, probes, lines, surfaces, and diagnostics
```

Generated `solution/`, `samples/`, `runs/`, `grid_study/`, build, cache, and
environment directories are not source and must not be committed. Keep compact
reference tables or summary JSON only when a test or durable validation record
uses them. Figures may be committed when documentation links to them and their
generation command is recorded.

Large restart states and raw simulation series should live in a release asset,
data repository, or archival service with a checksum and provenance note. Do
not use Git as an output store. Before committing, inspect:

```bash
git status --short
git diff --check
find . -type d -name __pycache__ -prune
```

The tutorial cleanup scripts remove only generated case output. Do not point
them at a parent directory or a directory containing irreplaceable data.
