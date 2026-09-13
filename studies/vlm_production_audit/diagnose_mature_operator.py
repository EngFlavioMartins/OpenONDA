"""Compare the retained rotor's worst strain with a bounded f64 direct sum.

This diagnoses one fixed source state; it neither advances nor qualifies the
rotor. The independent host operator is evaluated at all 32 largest-strain
particles plus 32 deterministic random particles, including finite self terms.
"""

import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from source.solvers.vpm.kernels.base import make_vortex_kernel


def main():
    """Record approximation errors, locations and instantaneous growth rates."""
    base = Path(__file__).parent
    checkpoint = ROOT / "tutorials/vpm/06_rotor_flow_PENDING/solution/vpm_001152.h5"
    with h5py.File(checkpoint) as archive:
        x, gamma, sigma, volume, viscosity = [
            archive[f"particles/{name}"][:].astype(float)
            for name in ("position", "vortex_strength", "core_radius", "particle_volume", "effective_viscosity")
        ]
    tree = np.load(base / "mature_induction_sort0_block128.npz")
    jacobian = tree["gradient"].astype(float)
    strain = .5 * (jacobian + jacobian.transpose(0, 2, 1))
    strain_norm = np.linalg.norm(strain, axis=(1, 2))
    worst = np.argsort(strain_norm)[-32:][::-1]
    selected = np.unique(np.r_[worst, np.random.default_rng(911).choice(len(x), 32, replace=False)])
    kernel = make_vortex_kernel("GAUSSIAN")
    velocity = np.zeros((len(selected), 3))
    gradient = np.zeros((len(selected), 3, 3))
    for i, index in enumerate(selected):
        for start in range(0, len(x), 4096):
            part = slice(start, start + 4096)
            displacement = x[index] - x[part]
            velocity[i] += kernel.velocity_pair(displacement, gamma[part], sigma[index], sigma[part]).sum(axis=0)
            gradient[i] += kernel.gradient_pair(displacement, gamma[part], sigma[index], sigma[part]).sum(axis=0)
    rate = np.einsum("nji,nj->ni", gradient, gamma[selected])
    errors = {}
    for name, direct in (("velocity", velocity), ("gradient", gradient), ("rate", rate)):
        difference = tree[name][selected] - direct
        errors[name] = {"relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(direct)),
                        "maximum_absolute": float(np.max(np.abs(difference)))}
    rows = []
    for index in worst[:8]:
        i = np.flatnonzero(selected == index)[0]
        strength = gamma[index]
        growth = float(strength @ rate[i] / (strength @ strength))
        curl = np.array([gradient[i, 2, 1] - gradient[i, 1, 2],
                         gradient[i, 0, 2] - gradient[i, 2, 0],
                         gradient[i, 1, 0] - gradient[i, 0, 1]])
        rows.append({"index": int(index), "position_m": x[index].tolist(),
                     "radius_from_axis_m": float(np.linalg.norm(x[index, 1:])),
                     "core_radius_m": float(sigma[index]), "particle_volume_m3": float(volume[index]),
                     "effective_viscosity_m2_per_s": float(viscosity[index]),
                     "free_wake_strain_norm_per_s": float(strain_norm[index]),
                     "strain_times_dt006": float(strain_norm[index] * .006),
                     "relative_strength_growth_per_s": growth,
                     "diffusive_relative_core_growth_per_s": float(2 * viscosity[index] / sigma[index]**2),
                     "strength_curl_cosine": float(strength @ curl / (np.linalg.norm(strength) * np.linalg.norm(curl)))})
    result = {"source_checkpoint": str(checkpoint), "scope": "fixed source operator; no time convergence claim",
              "targets": len(selected), "selection": "32 largest free-wake strain norms plus 32 seeded random particles",
              "tree_vs_direct": errors, "largest_strain": rows}
    (base / "mature_operator_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(base / "mature_operator_direct.npz", selected=selected, velocity=velocity, gradient=gradient, rate=rate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
