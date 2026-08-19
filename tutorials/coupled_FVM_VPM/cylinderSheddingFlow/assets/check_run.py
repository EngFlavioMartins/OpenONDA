"""Numerical-integrity gate for the cylinder FVM–VPM validation workflow.

Runs after the coupled and reference simulations.  It checks:

* non-finite FVM fields, unconverged linear solves, peak CFL and continuity
  in both the coupled run and the reference;
* conservative transfer, VPM-BC flux balance, and population-cap loss in the
  coupler diagnostics;
* open vortex lines: the Karman wake is spanwise-coherent, so the vortex lines
  (omega_z) run along z and pierce the slab's z-faces by construction — the VPM
  carries them on beyond the slab, so that crossing is not a leak.  What must
  NOT happen is lateral or upstream leakage through the y- or x-faces, and the
  slip z-faces must stay closed (mean normal velocity ~ 0 relative to the
  downstream throughput);
* that both force histories are finite and reach the configured end time;
* that the hybrid and reference fine Cartesian lattices coincide (the couple
  and reference domains are offset by an integer number of finest cells, so
  local resolutions are exactly comparable).

Exit code 1 on any failure; 0 on pass.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = CASE_DIR / "referenceFlow"

Z_FACE_LEAK_LIMIT = 0.25
Z_SLIP_LIMIT = 0.05
UNCONVERGED_ALLOWED = 0
CFL_LIMIT = 5.0
CONTINUITY_LIMIT = 1e-3
CAP_LOSS_LIMIT = 0.02
DRAG_BIAS_LIMIT = 0.20


def _json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"FAIL: expected diagnostics file was not written: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"FAIL: diagnostics file is empty: {path}")
    return records


def _check_solver_history(diagnostics: list[dict], label: str) -> tuple[float, float]:
    max_cfl = max(float(record["cfl_max"]) for record in diagnostics)
    max_continuity = max(float(record["continuity_max"]) for record in diagnostics)
    if any(int(record["nonfinite_count"]) != 0 for record in diagnostics):
        raise SystemExit(f"FAIL ({label}): non-finite FVM fields were detected")
    failed = [
        (record["step"], solve.get("equation", "unknown"))
        for record in diagnostics
        for solve in record.get("linear_solves", ())
        if not solve.get("converged", False)
    ]
    if len(failed) > UNCONVERGED_ALLOWED:
        raise SystemExit(
            f"FAIL ({label}): {len(failed)} unconverged FVM linear solves (first: {failed[:5]})"
        )
    if max_cfl > CFL_LIMIT:
        raise SystemExit(f"FAIL ({label}): peak FVM CFL is excessive ({max_cfl:.3g})")
    if max_continuity > CONTINUITY_LIMIT:
        raise SystemExit(
            f"FAIL ({label}): peak continuity residual is excessive ({max_continuity:.3g})"
        )
    return max_cfl, max_continuity


def _check_coupling_history(coupling: list[dict]) -> None:
    circulation_error = max(
        abs(float(record["conservation"]["corrected_mismatch"]["circulation"]))
        for record in coupling
    )
    vpm_bc_error = max(
        abs(float(record["vpm_bc_flux"]["corrected_mismatch"])) for record in coupling
    )
    if circulation_error > 1e-8:
        raise SystemExit(
            f"FAIL: corrected transfer circulation mismatch is {circulation_error:.3g}"
        )
    if vpm_bc_error > 1e-8:
        raise SystemExit(f"FAIL: corrected VPM-BC-flux mismatch is {vpm_bc_error:.3g}")
    transfer_cfl = max(float(record.get("transfer", {}).get("cfl", 0.0)) for record in coupling)
    if transfer_cfl > 1.0:
        raise SystemExit(f"FAIL: peak transfer CFL is excessive ({transfer_cfl:.3g})")
    cap_loss = max(
        float(record.get("transfer", {}).get("population_pruned_circulation_fraction", 0.0))
        for record in coupling
    )
    if cap_loss > CAP_LOSS_LIMIT:
        raise SystemExit(f"FAIL: population-cap pruning discarded {cap_loss:.2%} of circulation")

    # Spanwise-coherent Karman wake: the vortex lines run along z, so they
    # cross the slab's z-faces by construction (the VPM continues them beyond
    # the slab).  Leakage must instead be absent laterally and upstream, and
    # the slip z-faces must stay closed: mean |u.n| there must be a small
    # fraction of the downstream throughput (|u.n| on the xmax face).
    worst: dict[str, float] = {}
    for name in ("xmin", "ymin", "ymax", "xmax"):
        peak = max(
            float(record.get("vortex_line_closure", {}).get(name, 0.0)) for record in coupling
        )
        worst[name] = peak
    for name in ("xmin", "ymin", "ymax"):
        if worst[name] > Z_FACE_LEAK_LIMIT:
            raise SystemExit(
                f"FAIL: open vortex lines leave through the {name} hand-off face "
                f"(mean |omega.n|/|omega| = {worst[name]:.3f} > {Z_FACE_LEAK_LIMIT}). "
                "The Karman wake is spanwise-coherent and must not leak laterally "
                "or upstream."
            )
    xmax_throughput = max(
        abs(float(record.get("interface_normal_velocity", {}).get("xmax", 0.0)))
        for record in coupling
    )
    slip: dict[str, float] = {}
    for name in ("zmin", "zmax"):
        peak = max(
            abs(float(record.get("interface_normal_velocity", {}).get(name, 0.0)))
            for record in coupling
        )
        slip[name] = peak
        if xmax_throughput > 0.0 and peak / xmax_throughput > Z_SLIP_LIMIT:
            raise SystemExit(
                f"FAIL: the slip {name} face is open (mean |u.n|/U_inf = "
                f"{peak / xmax_throughput:.3f} > {Z_SLIP_LIMIT})."
            )
    print(
        "  vortex-line closure: "
        + " ".join(f"{name}={worst[name]:.3f}" for name in ("xmin", "ymin", "ymax"))
        + f"  (xmax={worst['xmax']:.3f})"
    )
    print(
        "  z-face slip: "
        + " ".join(f"{name}={slip[name] / xmax_throughput:.3f} U_inf" for name in ("zmin", "zmax"))
        + f"  (xmax throughput={xmax_throughput:.3f})"
    )


def _load_forces(path: Path, expected_end: float) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"FAIL: force history was not written: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"FAIL: force history is empty: {path}")
    forces = np.asarray([[float(row[key]) for key in ("time", "Cd", "Cl")] for row in rows])
    if not np.all(np.isfinite(forces)):
        raise SystemExit(f"FAIL: force history contains non-finite values: {path}")
    if forces[-1, 0] + 1e-10 < expected_end:
        raise SystemExit(
            f"FAIL: force history ends at t={forces[-1, 0]:g}, expected {expected_end:g}: {path}"
        )
    return forces


def _load_setup_constants(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("case_setup", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(path.parents[1]))
    spec.loader.exec_module(module)
    return module


def _check_lattice_coincidence() -> None:
    """Hybrid and reference fine lattices must coincide exactly.

    Both cases refine to the same finest cell size around the body.  Because
    the two domains have different origins, the refined cells coincide only if
    the origin offsets are integer multiples of that finest size.  The offsets
    are verified against the case configuration and against the coupled run's
    recorded FVM domain.
    """
    hybrid_setup = _load_setup_constants(CASE_DIR / "cylinderSheddingFlow_setup.py")
    reference_setup = _load_setup_constants(REFERENCE_DIR / "referenceFlow_setup.py")
    h = float(getattr(hybrid_setup, "FVM_BODY_CELL_SIZE"))
    ref_h = float(getattr(reference_setup, "FVM_BODY_CELL_SIZE"))
    if abs(h - ref_h) > 1e-12:
        raise SystemExit(f"FAIL: hybrid and reference body cell sizes differ ({h:g} vs {ref_h:g})")
    hybrid_box = tuple(float(v) for v in getattr(hybrid_setup, "FVM_BOX"))
    reference_box = tuple(float(v) for v in getattr(reference_setup, "FVM_DOMAIN"))
    offsets = tuple((a - b) / h for a, b in zip(hybrid_box[::2], reference_box[::2]))
    integer_offsets = tuple(abs(offset - round(offset)) < 1e-6 for offset in offsets)
    meta = json.loads((CASE_DIR / "solution" / "run_metadata.json").read_text())
    recorded = meta.get("fvm_solver", {}).get("fvm_domain", {})
    recorded_box = tuple(
        float(recorded[k]) for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    )
    if recorded_box and tuple(recorded_box[i] for i in (0, 2, 4)) != tuple(
        hybrid_box[i] for i in (0, 2, 4)
    ):
        raise SystemExit(f"FAIL: hybrid FVM domain does not match configuration: {recorded_box}")
    if not all(integer_offsets):
        raise SystemExit(
            "FAIL: hybrid/reference fine lattice does not coincide — origin offsets "
            f"{offsets} are not integer multiples of the finest cell size {h:g}"
        )
    print(
        "  lattice coincidence: fine cell size "
        f"{h:g}; origin offsets (x, y, z) = "
        + " ".join(f"{o:g} h" for o in offsets)
        + " (integer multiples)"
    )


def main() -> None:
    metadata_path = CASE_DIR / "solution" / "run_metadata.json"
    if not metadata_path.is_file():
        raise SystemExit("FAIL: coupled run metadata was not written")
    expected_end = float(json.loads(metadata_path.read_text())["physics"]["t_end"])

    print("== solver integrity (coupled) ==")
    hybrid_diag = _json_lines(CASE_DIR / "solution" / "diagnostics.jsonl")
    max_cfl, max_continuity = _check_solver_history(hybrid_diag, "coupled")

    print("== solver integrity (reference) ==")
    ref_diag = _json_lines(REFERENCE_DIR / "solution" / "diagnostics.jsonl")
    ref_cfl, ref_continuity = _check_solver_history(ref_diag, "reference")

    print("== coupling integrity ==")
    coupling = _json_lines(CASE_DIR / "solution" / "coupler_diagnostics.jsonl")
    _check_coupling_history(coupling)

    print("== force histories ==")
    hybrid_forces = _load_forces(CASE_DIR / "samples" / "fvm_ibm_forces_history.csv", expected_end)
    ref_forces = _load_forces(REFERENCE_DIR / "samples" / "ibm_forces_history.csv", expected_end)

    overlap = (ref_forces[:, 0] >= expected_end - 20.0) & (ref_forces[:, 0] <= expected_end)
    hybrid_sat = np.mean(hybrid_forces[overlap, 1]) if np.count_nonzero(overlap) else np.nan
    ref_sat = np.mean(ref_forces[overlap, 1]) if np.count_nonzero(overlap) else np.nan
    if expected_end < 20.0:
        # Smoke runs stop before any saturation window exists; the last-20-unit
        # mean would be the impulsive-start transient, not the shedding mean.
        print(f"  (drag-bias gate skipped: run is {expected_end:g} long, no saturation window yet)")
    elif np.isfinite(hybrid_sat) and np.isfinite(ref_sat):
        bias = abs(hybrid_sat - ref_sat) / abs(ref_sat)
        print(
            f"  saturated mean Cd: hybrid={hybrid_sat:.4f}, reference={ref_sat:.4f}, bias={bias:.2%}"
        )
        if bias > DRAG_BIAS_LIMIT:
            raise SystemExit(f"FAIL: hybrid mean drag bias versus reference is {bias:.2%}")
    else:
        raise SystemExit("FAIL: no saturated drag window is available for the bias check")

    print("== lattice coincidence ==")
    _check_lattice_coincidence()

    print(
        "PASS: cylinder runs completed with converged FVM solves "
        f"(CFL {max_cfl:.3g}/{ref_cfl:.3g}, continuity "
        f"{max_continuity:.3g}/{ref_continuity:.3g}), conservative transfer, "
        f"and no vortex-line leakage through the lateral/upstream faces "
        f"through t={hybrid_forces[-1, 0]:g}."
    )


if __name__ == "__main__":
    main()
