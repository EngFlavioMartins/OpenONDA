"""Observe the first native cube renewal without altering its numerical result.

Capture its actual source arrays and identify each intermediate contribution to
the Gaussian field, its curl mismatch, and velocity. The pre-existing frozen
donor driver reconstructs the native FVM state and verifies it against VTK.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cube_wake_frozen_renewal as frozen
from cube_wake_operator_audit import evaluate, reflection_asymmetry
from cube_wake_particle_probe import load_case, rms
from numba import set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits


def run(args):
    load_case(args.source_tree)
    from source.coupler import stable_renewal as renewal

    captured = {}
    originals = {}

    def capture(name):
        original = getattr(renewal, name)
        originals[name] = original

        def wrapped(*positional, **keywords):
            result = original(*positional, **keywords)
            if name not in captured:
                copied = tuple(
                    value.copy() if isinstance(value, np.ndarray) else value for value in positional
                )
                copied_result = result.copy() if isinstance(result, np.ndarray) else result
                captured[name] = (copied, keywords.copy(), copied_result)
            return result

        setattr(renewal, name, wrapped)

    for name in (
        "scatter_m4_prime_to_lattice",
        "blend_represented_state",
        "recover_vortex_invariants",
    ):
        capture(name)
    try:
        frozen.run(SimpleNamespace(**vars(args), iterations=1))
    finally:
        for name, original in originals.items():
            setattr(renewal, name, original)

    scatter_args, _, scattered = captured["scatter_m4_prime_to_lattice"]
    lattice = scatter_args[2]
    blend_args, blend_keywords, blended = captured["blend_represented_state"]
    vpm, target, authority, shape, spacing = blend_args
    represented = renewal.gaussian_represented_vortex_strength
    physical_target = represented(vpm, shape, spacing, core_radius=0.066)
    physical_target += authority[:, None] * (target - physical_target)
    raw_blend = vpm + authority[:, None] * (target - vpm)
    residual = physical_target - represented(raw_blend, shape, spacing, core_radius=0.066)
    gain = min(blend_keywords["amplification_cap"] - 1, 1)
    corrected = raw_blend + gain * residual
    np.testing.assert_allclose(
        corrected * blend_keywords["output_weight"][:, None],
        blended.vortex_strength,
        rtol=1e-13,
        atol=1e-15,
    )
    with np.load(args.operator_audit / "particles_accepted_t8.npz") as data:
        initial = {key: data[key].copy() for key in data.files}
    with np.load(args.operator_audit / "fields_t8.npz") as data:
        points = data["points"].copy()
    outside = ~np.all(
        (initial["position"] >= lattice.renewal_bounds[::2])
        & (initial["position"] <= lattice.renewal_bounds[1::2]),
        axis=1,
    )
    outer_p, outer_g = initial["position"][outside], initial["vortex_strength"][outside]
    seam = (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62)
    states = {"accepted": initial}
    for name, strength in {
        "after_m4_scatter": scattered,
        "after_fluid_mask": vpm,
        "after_coefficient_blend": raw_blend,
        "after_representation_correction": corrected,
        "after_output_fluid_mask": blended.vortex_strength,
    }.items():
        active = np.linalg.norm(strength, axis=1) > 0
        states[name] = {
            "position": np.vstack((lattice.positions[active], outer_p)),
            "vortex_strength": np.vstack((strength[active], outer_g)),
            "core_radius": np.r_[
                np.full(int(active.sum()), 0.066), initial["core_radius"][outside]
            ],
        }
    with np.load(args.output / "particles_native_final.npz") as data:
        states["after_pruning_and_invariants"] = {key: data[key].copy() for key in data.files}
    records = []
    for name, state in states.items():
        fields = evaluate(points, state)
        row = {
            "stage": name,
            "seam_vorticity_minus_curl_rms": rms((fields["omega"] - fields["curl"])[seam]),
            "seam_vorticity_rms": rms(fields["omega"][seam]),
            "seam_divergence_rms": float(np.sqrt(np.mean(fields["divergence"][seam] ** 2))),
            "seam_reflection_rms": rms(reflection_asymmetry(points, fields["velocity"])[seam]),
        }
        records.append(row)
        np.savez_compressed(args.output / f"stage_{name}.npz", points=points, **fields)
        print(json.dumps(row), flush=True)
    np.savez_compressed(
        args.output / "transfer_inputs.npz",
        position=lattice.positions,
        shape=shape,
        spacing=spacing,
        authority=authority,
        vpm_vortex_strength=vpm,
        fvm_vortex_strength=target,
        output_weight=blend_keywords["output_weight"],
        physical_target=physical_target,
    )
    (args.output / "stages.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--operator-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)
