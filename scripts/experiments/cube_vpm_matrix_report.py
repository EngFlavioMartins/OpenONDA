"""Score the VPM matrix on the quantity that is actually wrong.

The complaint is that the coupled run's interior velocity profile is flat
compared with the fully meshed reference: the wake's peaks and valleys are
missing.  The scalar that captures this is *amplitude retention* -- the
peak-to-peak span of the centreline Ux inside the FVM box, divided by the
reference's span at the same instant.

    retention = 1.00   profile amplitude fully preserved

Both the centreline and the off-axis line at y=0.75 are scored.  The off-axis
line is the sharper test and is reported first: at t=2.4 the reference reaches a
minimum of about -0.07 there, so a run that fails to go negative at all has lost
the structure outright, which a peak-to-peak span alone can hide.  The minimum's
value and its location are therefore tracked next to the span.

Correlation is reported because a run can keep its amplitude while putting the
structure in the wrong place; both must hold.

No production number is quoted here on purpose.  The archived hybrid baselines
were run under a different configuration (different FVM mesh, and in at least
one case a different particle spacing and pressure BC), so they do not define
the current control.  That number comes from a current-configuration run.

Runs are compared against each other at fixed h.  A coarse particle lattice
cannot resolve the wake whatever the modules do, so absolute retention at
h=0.15 is not meaningful on its own -- the differences between variants are.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CUBE = ROOT / "tutorials/coupled_FVM_VPM/cubeFlow"
REFERENCE_DIR = CUBE / "referenceFlow/samples"
# (file stem, x window) -- off-axis first: it is the sharper discriminator.
# The off-axis window must span the shoulder acceleration and the separation
# dip beside the body: at t=2.4 the reference peaks at +1.36 near x=-0.3 and
# reaches -0.0711 at x=0.32.  A wake-only window starting at x=0.55 misses the
# minimum entirely and reports it at the window edge.
LINES = (("offaxis_y075", (-0.5, 2.0)), ("centerline", (0.55, 3.15)))

# Wake only, and clear of both the body and the outflow face.
X_LO, X_HI = 0.55, 3.15
TIMES = (1.2, 1.8, 2.4)


def load_centerline(path: Path) -> dict[float, np.ndarray]:
    by_time: dict[float, list[tuple[float, float]]] = defaultdict(list)
    with open(path) as handle:
        for row in csv.DictReader(handle):
            by_time[round(float(row["flow_time"]), 4)].append((float(row["x"]), float(row["Ux"])))
    return {t: np.array(sorted(v)) for t, v in by_time.items()}


def score(
    ref: dict[float, np.ndarray], run: dict[float, np.ndarray], window: tuple[float, float]
) -> dict[float, dict]:
    grid = np.arange(window[0], window[1], 0.02)
    out = {}
    for t in TIMES:
        if t not in ref or t not in run:
            continue
        a = np.interp(grid, ref[t][:, 0], ref[t][:, 1])
        b = np.interp(grid, run[t][:, 0], run[t][:, 1])
        span_a = float(a.max() - a.min())
        span_b = float(b.max() - b.min())
        out[t] = {
            "retention": span_b / span_a if span_a else float("nan"),
            "corr": float(np.corrcoef(a, b)[0, 1]),
            "rmse": float(np.sqrt(((a - b) ** 2).mean())),
            "min_ref": float(a.min()),
            "min_run": float(b.min()),
            "xmin_ref": float(grid[int(a.argmin())]),
            "xmin_run": float(grid[int(b.argmin())]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, default=CUBE / "matrix")
    ap.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    ap.add_argument(
        "--control",
        type=Path,
        default=CUBE / "samples",
        help="current-configuration production run, scored alongside the matrix",
    )
    args = ap.parse_args()

    runs: list[tuple[str, Path]] = []
    if (args.control / "centerline.csv").exists():
        runs.append(("CONTROL(prod)", args.control))
    runs += [
        (p.parent.name, p)
        for p in sorted(args.matrix.glob("*/samples"))
        if (p / "centerline.csv").exists()
    ]
    if not runs:
        raise SystemExit(f"no scored runs under {args.matrix} or {args.control}")

    all_rows: dict[str, dict[str, dict]] = {}
    for stem, window in LINES:
        ref_path = args.reference_dir / f"{stem}.csv"
        if not ref_path.exists():
            print(f"[skip] no reference for {stem}")
            continue
        ref = load_centerline(ref_path)
        print(f"\n=== {stem}   x in [{window[0]}, {window[1]}]   (1.00 = reference)\n")
        header = "run".ljust(16)
        for t in TIMES:
            header += f"  t={t:<4.1f} ret  corr "
        print(header + "   min(ref->run) @ t=2.4")
        print("-" * (len(header) + 24))
        for name, sample_dir in runs:
            path = sample_dir / f"{stem}.csv"
            if not path.exists():
                continue
            s_ = score(ref, load_centerline(path), window)
            all_rows.setdefault(stem, {})[name] = s_
            line = name.ljust(16)
            for t in TIMES:
                if t in s_:
                    line += f"   {s_[t]['retention']:5.3f} {s_[t]['corr']:6.3f} "
                else:
                    line += "       --     -- "
            if 2.4 in s_:
                d = s_[2.4]
                line += f"   {d['min_ref']:+.4f} -> {d['min_run']:+.4f}"
                line += f"  (x {d['xmin_ref']:.2f} -> {d['xmin_run']:.2f})"
            print(line)

    # Each B variant is one change away from A0, so the delta is that component's
    # cost.  B5 moves precision and device together and is a sanity check only.
    scored = all_rows.get("offaxis_y075") or all_rows.get("centerline") or {}
    deltas = [
        (s_[2.4]["retention"] - scored["A0_bare"][2.4]["retention"], name)
        for name, s_ in scored.items()
        if name.startswith("B") and "A0_bare" in scored and 2.4 in s_ and 2.4 in scored["A0_bare"]
    ]
    if deltas:
        print("\ncomponent cost in off-axis retention at t=2.4 (negative = flattens)\n")
        for d, name in sorted(deltas):
            print(f"  {d:+6.3f}  {name}")


if __name__ == "__main__":
    main()
