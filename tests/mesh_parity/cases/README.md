# Differential parity cases

Each JSON file is a complete, explicit input to:

```text
python -m tools.mesh_parity.parity_report CASE.json --output ARTIFACT_DIR
```

The runner copies the original surface triangles into an isolated cfMesh input
and records the original STL SHA-256 in the resulting report. It refuses a
surface that crosses the outer domain: that input must first be clipped/capped
and frozen, rather than being silently normalized by one mesher only.

Current baseline cases are intentionally small enough to use as the first
oracle ladder rung:

- `cube_aligned.json` — root size, outer patch assignment, and Cartesian
  extraction;
- `cube_oblique.json` — translated, 23-degree arbitrarily-axis-rotated cube
  with a frozen 12-triangle authority STL;
- `cylinder_coarse.json` — a closed curved surface at a bounded resolution;
- `cylinder_box_refinement.json` — cfMesh `objectRefinements` / OpenONDA box
  refinement input;
- `cylinder_patch_refinement.json` — cfMesh `localRefinement`. OpenONDA
  currently reports this as unsupported, which is an intentional visible gate.

The normalized 508-triangle reference-cylinder input remains to be added only
after its exact source STL is frozen. The repository's checked-in long cylinder
is usable for the coarse cases below, but it is not the historical
508-triangle oracle described in the audit.
