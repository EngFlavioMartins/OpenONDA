"""Compare saved Gaussian velocity fields at identical independent targets."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .audit_fields import target_velocity
from .render_study import states
from .. import setup


def compare(reference, candidates, output, centre_on_reference=False):
    def load(spec):
        run, step = spec.rsplit(":", 1)
        return next(state for state in states(run) if int(state["step"]) == int(step))

    base = load(reference)
    bounds = np.array([[-0.75, -2, -2], [0.75, 2, 2]], dtype=float)
    if centre_on_reference:
        centre = np.average(
            base["position"][:, 0], weights=np.linalg.norm(base["vortex_strength"], axis=1)
        )
        bounds[:, 0] = centre + np.array([-1.5, 1.5])
    targets = np.random.default_rng(1729).uniform(bounds[0], bounds[1], (1024, 3))
    velocity = target_velocity(targets, base)
    rows = []
    for candidate in candidates:
        state = load(candidate)
        if not np.isclose(float(state["time"]), float(base["time"]), rtol=0, atol=1e-8):
            raise ValueError("Field comparison requires identical physical times")
        trial = target_velocity(targets, state)
        error = float(np.linalg.norm(trial - velocity) / np.linalg.norm(velocity))
        rows.append(dict(candidate=candidate, relative_velocity_l2=error))
        print(candidate, error, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                reference=reference,
                time=float(base["time"]),
                targets=1024,
                seed=1729,
                bounds=bounds.tolist(),
                method="direct unbounded Gaussian Biot–Savart; identical physical target points",
                comparisons=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--centre-on-reference", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=setup.TUTORIAL_DIR / "figures" / "study" / "velocity_comparison.json",
    )
    args = parser.parse_args()
    compare(args.reference, args.candidates, args.output, args.centre_on_reference)
