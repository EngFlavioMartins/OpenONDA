"""Bounded full-state smoke harness, explicitly separate from the solver."""

import argparse
import json
from pathlib import Path
import runpy

from paraview.simple import GetTimeKeeper, RenderAllViews, SaveScreenshot

parser = argparse.ArgumentParser()
parser.add_argument("state", type=Path)
parser.add_argument("output", type=Path)
arguments = parser.parse_args()
if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
    raise FileExistsError(arguments.output)
state = runpy.run_path(str(arguments.state.resolve()))
expected = tuple(state["matched_times"].times)
actual = tuple(GetTimeKeeper().TimestepValues)
assert (
    len(actual) == len(expected) and state["match_saved_times"](actual, expected).times == actual
), (actual, expected)
RenderAllViews()
written = SaveScreenshot(str(arguments.output.resolve()), state["layout1"])
if not written or not arguments.output.is_file():
    raise RuntimeError("Full ParaView checkout-state screenshot was not written")
record = {
    "state": str(arguments.state.resolve()),
    "image": str(arguments.output.resolve()),
    "common_saved_times": list(state["matched_times"].times),
    "scene_time": float(state["animationScene1"].AnimationTime),
    "timekeeper_times": list(GetTimeKeeper().TimestepValues),
    "playback_clock_validation": "all advertised snap-to-timestep clocks match actual common saved states",
    "scope": "full scene render at common saved time; no simulation advancement",
}
arguments.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
