# Tutorial authoring

Tutorial cases and user-facing scripts are physics-focused. Keep them minimal,
clear, well-spaced, and briefly commented. Show the physical inputs, geometry,
models, numerical choices, observations, and direct solver calls.

Do not put try/except/finally blocks, decorators, error-handling or validation
checks, caching, import/path bootstraps, compatibility code, or test workarounds
in tutorials or user-facing scripts. Generally required runtime, I/O, resource
management, validation, and restart behavior belong in the native framework.
Keep tutorial-specific physical inputs and defaults in their cases.

Keep verification tooling under tests, outside tutorial assets. Remove retired
code and tests instead of keeping compatibility branches. Validate changes in
temporary cases, clean test intermediates, and commit completed work when
requested. Preserve running cases and unrelated user edits.

Use CFD terminology and plain, self-descriptive technical names throughout the
solver, tutorials, tests, messages, and documentation. Name the physical or
numerical quantity directly: blending weight, restart validation, mesh geometry,
particle state, error bound, or solver settings. Avoid business and abstract
software jargon. Keep established mathematical terms such as tensor contraction.
