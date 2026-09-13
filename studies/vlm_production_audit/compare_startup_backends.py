"""Reproduce CPU/Metal startup agreement without assuming atomic insertion order."""

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


def main():
    """Require matched clocks, groups and a bijective sub-micrometre position match."""
    base = Path(__file__).parent
    files = [base / run / "solution/vpm_000008.h5"
             for run in ("rotor_smoke_compiled", "rotor_metal_unrestricted")]
    with h5py.File(files[0]) as cpu, h5py.File(files[1]) as gpu:
        assert cpu["solver"].attrs["time"] == gpu["solver"].attrs["time"] == .048
        x, y = [archive["particles/position"][:] for archive in (cpu, gpu)]
        a, b = [archive["particles/group_id"][:] for archive in (cpu, gpu)]
        assert len(x) == len(y)
        assert set(a) == set(b)
        match = np.empty(len(x), dtype=int)
        distances = np.empty(len(x))
        for group in np.unique(a):
            source, target = np.flatnonzero(a == group), np.flatnonzero(b == group)
            distance, neighbour = cKDTree(y[target]).query(x[source])
            assert len(np.unique(neighbour)) == len(source) == len(target)
            match[source], distances[source] = target[neighbour], distance
        assert distances.max() < 1e-6
        fields = {}
        for name in ("particles/position", "particles/vortex_strength", "particles/velocity",
                     "particles/core_radius", "solver/vlm/circulation", "solver/vlm/panel_force",
                     "solver/vlm/panel_corner_position"):
            left, right = cpu[name][:].astype(float), gpu[name][:].astype(float)
            if name.startswith("particles/"):
                right = right[match]
            delta = right - left
            fields[name] = {"max_absolute_difference": float(np.abs(delta).max()),
                            "relative_l2_difference": float(np.linalg.norm(delta) / np.linalg.norm(left))}
    result = {"scope": "identical startup at t=0.048; CPU vs Metal; not mature-flow convergence",
              "particle_matching": "bijective nearest positions within provenance group; GPU atomic append order is not a persistent particle identity",
              "fields": fields, "matching_max_distance": float(distances.max())}
    (base / "cpu_metal_startup_comparison_reproduced.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
