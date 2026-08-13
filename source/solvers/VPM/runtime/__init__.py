"""Runtime lifecycle for the VPM solver.

This package owns the taichi runtime lifecycle: backend selection,
initialization, memory policy and reset.  ``config`` must stay a leaf, so the
runtime initialiser lives here and ``config.backend`` only re-exports it for
backwards compatibility (see ARCHITECTURE.md).
"""
