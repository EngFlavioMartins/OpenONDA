"""Compare transverse velocity at coincident native saved clocks."""

import pandas as pd

from openonda.results import read_json, read_velocity_profile_frames
from openonda.saved_times import match_saved_times

from . import postprocess as data


def profile_geometry():
    """Use the saved force scales and physical domains."""
    solution = data.CASE_DIR / "solution"
    metadata = read_json(solution / "run_metadata.json")
    configuration = read_json(solution / "fvm_metadata.json")["configuration"]
    force = next(
        sample
        for sample in configuration["samplers"]
        if sample["type"] == "ForceSampler" and "cylinder" in sample["patch_names"]
    )
    return {
        "diameter": force["reference_length"],
        "speed": force["reference_velocity"],
        "fvm_box": metadata["fvm_solver"]["fvm_domain"],
        "transfer_box": metadata["coupler"]["transfer_region_bounds"],
    }


def coincident_velocity_profiles(profiles, geometry):
    """Compare near-body FVM observations and the outer VPM wake."""
    paths = tuple(path for _, reference, vpm, _ in profiles for path in (reference, vpm))
    states = list(data.coincident_profiles(paths))
    box = geometry["fvm_box"]
    frames, updated = {}, list(profiles)
    for index, (x, reference, vpm, fvm) in enumerate(profiles):
        if box["xmin"] <= x <= box["xmax"]:
            frames[fvm] = list(
                read_velocity_profile_frames(
                    fvm,
                    query_path=reference,
                    collection=data.CASE_DIR / "solution/fvm.pvd",
                    mesh=data.CASE_DIR / "solution/fvm/mesh.npz",
                    k=12,
                    bounds=box,
                )
            )
        else:
            updated[index] = (x, reference, vpm, None)
    common = match_saved_times(
        [time for time, _ in states],
        *([time for time, _, _ in records] for records in frames.values()),
    )
    for indices in zip(*common.indices, strict=True):
        time, saved = states[indices[0]]
        samples, sources = dict(saved), {}
        for (path, records), frame_index in zip(frames.items(), indices[1:], strict=True):
            _, values, source = records[frame_index]
            frame = pd.DataFrame(values)
            samples[path] = frame
            if source is not None:
                x = float(frame.position_x.iloc[0])
                sources[f"x{x:g}"] = {
                    **source,
                    "native_field": str(source["native_field"].relative_to(data.CASE_DIR)),
                    "sampled_y_interval": [
                        float(frame.position_y.min()),
                        float(frame.position_y.max()),
                    ],
                }
        yield time, updated, samples, sources
