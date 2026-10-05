"""Bounded actual periodic cylinder test, invoked in a private temporary case."""

import json
from pathlib import Path
import sys

import numpy as np

from tests.coupler.test_cylinder_planar_native import _advance


def main(directory, mesh_path=None):
    first = _advance(directory, limit=1, start_from="initial", cores=4, mesh_path=mesh_path)
    assert first["accepted"] == 1
    # The execution limit is outside numerical restart identity.
    resumed = _advance(
        directory,
        limit=1,
        start_from="latest",
        cores=4,
        expected_restart=first["restart_state"],
    )
    assert resumed["accepted"] == 2
    times = np.array([float(row["time"]) for row in resumed["forces"]])
    np.testing.assert_allclose(times, [0.04, 0.08], atol=1e-12, rtol=0)
    report = {key: resumed[key] for key in ("rank", "accepted", "cells", "parallel_mode")}
    report["force_rows"] = len(resumed["forces"])
    report["forces"] = resumed["forces"]
    report["exchanges"] = resumed["exchanges"]
    (directory / f"rank-{resumed['rank']}.json").write_text(json.dumps(report) + "\n")
    np.savez(
        directory / f"rank-{resumed['rank']}-state.npz",
        **{name: resumed[name] for name in ("velocity", "pressure", "positions", "strengths")},
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
