"""Export the sample-based numerical evidence used in the thesis subsection."""

import json
import numpy as np
import pandas as pd

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from ..assets import ring_metrics as rm


def summarize():
    result = {}
    for name in rm.CURRENT_VARIANTS:
        root = rm.SAMPLES_DIR / name
        metadata = rm.load_metadata(name)
        if not metadata or not (root / "flow_integrals.csv").is_file():
            continue
        d = rm.load_sampled_ring_data(root / "ring_diagnostics.csv")
        if d is None:
            continue
        f = (
            pd.read_csv(root / "flow_integrals.csv")
            .sort_values("step")
            .drop_duplicates("step", keep="last")
        )
        numerics = metadata["configuration"]["numerics"]
        vector = d[
            ["net_vortex_strength_x", "net_vortex_strength_y", "net_vortex_strength_z"]
        ].to_numpy()
        scale = d.vortex_strength_magnitude_sum.iloc[0]
        drift = np.linalg.norm(vector - vector[0], axis=1) / scale
        tube = d.tube_circulation.to_numpy() / d.tube_circulation.iloc[0] - 1
        time, speed = rm.load_sampled_ring_speed(root / "ring_diagnostics.csv")
        reference = rm.saffman_speed(time * rm.REFERENCE_TIME) / rm.REFERENCE_VELOCITY
        valid = time * rm.REFERENCE_TIME <= rm.saffman_valid_time_limit()
        out = {
            "status": metadata["lifecycle"]["status"],
            "normalized_end_time": float(d.time.iloc[-1] / rm.REFERENCE_TIME),
            "final_vector_drift": float(drift[-1]),
            "initial_vector_norm": float(np.linalg.norm(vector[0]) / scale),
            "final_tube_change_percent": float(100 * tube[-1]),
            "maximum_absolute_tube_change_percent": float(100 * np.max(abs(tube))),
            "speed_reference_relative_rms_percent": float(
                100 * np.sqrt(np.mean(((speed[valid] - reference[valid]) / reference[valid]) ** 2))
            ),
            "final_speed_reference_difference_percent": float(100 * (speed[-1] / reference[-1] - 1))
            if valid[-1]
            else None,
            "energy_change_percent": float(
                100 * (f.total_kinetic_energy.iloc[-1] / f.total_kinetic_energy.iloc[0] - 1)
            ),
            "positive_energy_rates": int((f.kinetic_energy_rate > 0).sum()),
            "max_misalignment_degrees": float(f.vortex_strength_misalignment_degrees.max()),
            "maximum_strength_ratio": float(
                f.max_vortex_strength_magnitude.max() / f.max_vortex_strength_magnitude.iloc[0]
            ),
            "final_mean_particle_radius_over_R0": float(
                f.mean_particle_core_radius.iloc[-1] / rm.RING_RADIUS
            ),
            "missing_sample_steps": sorted(
                set(range(0, int(d.step.iloc[-1]) + 1, 5)) - set(d.step)
            ),
            "common_step_100": None,
        }
        common = d[d.step == 100]
        if not common.empty:
            i = common.index[0]
            fs = f[(f.step > 0) & (f.step <= 100)]
            ts = time <= 100 * numerics["time_step_size"] / rm.REFERENCE_TIME
            out["common_step_100"] = {
                "normalized_time": float(d.time.iloc[i] / rm.REFERENCE_TIME),
                "tube_change_percent": float(tube[i] * 100),
                "vector_drift": float(drift[i]),
                "relative_speed_rms_percent": float(
                    100 * np.sqrt(np.mean(((speed[ts] - reference[ts]) / reference[ts]) ** 2))
                ),
                "mean_absolute_energy_rate_residual": float(
                    np.mean(abs(fs.kinetic_energy_rate - fs.viscous_kinetic_energy_rate)) / rm.P_REF
                ),
            }
        result[name] = out
    return result


def main():
    output = rm.FIGURES_DIR / "vortex_ring_metrics.json"
    data = summarize()
    output.write_text(json.dumps(data, indent=2) + "\n")
    for name, values in data.items():
        print(
            f"{name}: tube change {values['final_tube_change_percent']:.2g}%, vector drift {values['final_vector_drift']:.2g}"
        )
        if values["missing_sample_steps"]:
            print(f"  Missing sample steps (not interpolated): {values['missing_sample_steps']}")
    print(f"Saved sample-based evidence: {output}")


if __name__ == "__main__":
    main()
