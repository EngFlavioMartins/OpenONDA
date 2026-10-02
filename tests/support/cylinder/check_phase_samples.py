#!/usr/bin/env python3
"""Fail closed on missing phase diagnostics; periodicity is not grid convergence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import find_peaks


def _legacy_case():
    # Legacy helper callers retain their old API. The normal CLI below never
    # imports a solver simply to inspect recorded observation metadata.
    if __package__:
        from . import phase_benchmark
    else:
        import phase_benchmark
    return phase_benchmark


def inspect_csv(path, count, interval, end, fields):
    case = _legacy_case()
    times = {}
    previous_key = None
    rows = 0
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as stream:
        reader = csv.DictReader(stream)
        assert set(fields) <= set(reader.fieldnames), (path, reader.fieldnames)
        for raw in reader:
            row = {k: float(raw[k]) for k in fields}
            assert all(np.isfinite(v) for v in row.values()), path
            t = round(row["time"], 8)
            assert abs(row["time"]-t) < 1e-8 and abs(t/interval-round(t/interval)) < 1e-6
            position = tuple(row.get("position_"+c, 0.) for c in "xyz")
            key = (t, position)
            assert previous_key is None or t >= previous_key[0], (path, "time reversal")
            previous_key = key
            group = times.setdefault(t, {})
            assert position not in group, (path, "duplicate time/point", key)
            group[position] = row
            rows += 1
    expected = np.round(np.arange(1, case.steps(end, interval)+1)*interval, 8)
    assert set(expected) <= set(times), (path, "missing accepted sample time")
    assert max(times) == round(end, 8), (path, "incomplete horizon")
    geometry = None
    for t, group in times.items():
        assert len(group) == count, (path, t, len(group), count)
        if geometry is None:
            geometry = set(group)
        assert set(group) == geometry, (path, "moving sample geometry")
    return dict(path=str(path), sha256=digest, rows=rows, events=len(times),
                start=min(times), end=max(times), points=count, cadence=interval)


def force_signal(path, start=40., end=100.):
    case = _legacy_case()
    data = np.genfromtxt(path, delimiter=",", names=True)
    t, drag, lift = (data[k] for k in ("time", "drag_coefficient", "lift_coefficient"))
    assert np.allclose(drag, 2*data["total_force_x"]/case.SPAN, atol=1e-9, rtol=0)
    assert np.allclose(lift, 2*data["total_force_y"]/case.SPAN, atol=1e-9, rtol=0)
    for c in "xyz":
        assert np.allclose(data["total_force_"+c], data["pressure_force_"+c]+data["viscous_force_"+c], atol=1e-9, rtol=0)
    selected = (t >= start-1e-8) & (t <= end+1e-8)
    t, drag, lift = t[selected], drag[selected], lift[selected]
    assert len(t) > 20 and np.all(np.diff(t) > 0)
    centered = lift-lift.mean()
    peaks, _ = find_peaks(centered, prominence=max(.01, .5*centered.std()),
                          distance=case.steps(2., case.FORCES))
    # Quadratic peak interpolation removes the .04s sampling quantization.
    peak_times = []
    for i in peaks:
        denominator = centered[i-1]-2*centered[i]+centered[i+1]
        offset = .5*(centered[i-1]-centered[i+1])/denominator
        peak_times.append(float(t[i]+offset*case.FORCES))
    periods = np.diff(peak_times)
    usable = len(periods) >= 8 and centered.std() > .01 and np.std(periods)/np.mean(periods) < .05
    return dict(window=[start, end], mean_drag=float(drag.mean()), lift_rms=float(centered.std()),
        peak_times=peak_times, complete_cycles=len(periods),
        mean_period=float(np.mean(periods)) if len(periods) else None,
        strouhal=float(1/np.mean(periods)) if len(periods) else None,
        period_cv=float(np.std(periods)/np.mean(periods)) if len(periods) else None,
        usable_periodic_phase_signal=bool(usable), grid_independent=False)


def inspect(root, kind, end):
    case = _legacy_case()
    directory = root/kind/"samples"
    specs = [("forces_history", 1, case.FORCES)]
    specs += [(n, k, case.FORCES) for n, _, _, k in case.phase_lines()]
    specs += [(n, k, case.PROFILES) for n, _, _, k in case.field_lines(kind == "reference")]
    if kind == "reference":
        specs += [(n, k, case.FORCES) for n, _, _, k in case.phase_lines(True)]
    else:
        specs += [("vpm_"+n, k, case.FORCES) for n, _, _, k in case.phase_lines()+case.phase_lines(True)]
        specs += [(f"vpm_transverse_x{x}", 51, case.PROFILES) for x in (2, 4)]
    results = []
    for name, count, interval in specs:
        if name == "forces_history":
            fields = ["time", "drag_coefficient", "lift_coefficient", "side_force_coefficient"]
        else:
            fields = ["time"] + [f"{q}_{c}" for q in ("position", "velocity", "vorticity") for c in "xyz"]
            if not name.startswith("vpm_"):
                fields += ["kinematic_pressure"]
        results.append(inspect_csv(directory/(name+".csv"), count, interval, end, fields))
    collections = ["midspan"] + (["vpm_midspan"] if kind == "coupled" else [])
    for name in collections:
        path = directory/(name+".pvd")
        tree = ET.parse(path)
        frames = tree.findall(".//DataSet")
        times = {round(float(frame.attrib["timestep"]), 8) for frame in frames}
        assert set(np.round(np.arange(1,case.steps(end,case.SLICES)+1)*case.SLICES,8)) <= times
        assert all((path.parent/frame.attrib["file"]).is_file() for frame in frames)
        results.append(dict(path=str(path), frames=len(frames), end=max(times)))
    output = dict(kind=kind, checked_end=end, samples=results)
    if end == case.END:
        output["force_signal"] = force_signal(directory/"forces_history.csv")
    target = root/kind/("sample-audit.json" if end == case.END else "pilot-sample-audit.json")
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(output, indent=2)+"\n")
    if end == case.END:
        assert output["force_signal"]["usable_periodic_phase_signal"], "Insufficient settled shedding; review before comparison"
    return output


if __name__ == "__main__":
    if __package__:
        from .audit_saved_samples import audit_normal
    else:
        from audit_saved_samples import audit_normal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("reference", "coupled"))
    parser.add_argument("--case-dir", type=Path, default=(Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"),
                        help="ordinary coupled case directory; reference uses its reference_flow child")
    parser.add_argument("--end", type=float, default=None,
                        help="observed horizon (default: last force sample); slower samplers need only scheduled events")
    parser.add_argument("--output", type=Path, help="optional new JSON report; existing files are never overwritten")
    options = parser.parse_args()
    if options.output is not None and options.output.exists():
        parser.error(f"Report already exists: {options.output}")
    result = audit_normal(options.case_dir, options.kind, options.end)
    if options.output is not None:
        with options.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "samples"}, allow_nan=False))
