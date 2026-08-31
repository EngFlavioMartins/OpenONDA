"""Run one compact cube-reference grid trial in an isolated directory.

The production case writes dense field and surface archives. A grid study
only needs force histories and two fixed physical probe lines, so this runner
suppresses VTK output and leaves the caller free to discard its scratch case.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time as wall_clock


REFERENCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFERENCE_DIR))

import reference_flow_setup as case  # noqa: E402


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _package_temperature_c() -> float | None:
    """Return the hottest available Linux CPU-package temperature."""
    temperatures: dict[str, list[float]] = {"TCPU": [], "x86_pkg_temp": []}
    for type_path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/type")):
        try:
            zone_type = type_path.read_text(encoding="utf-8").strip()
            if zone_type not in temperatures:
                continue
            temperatures[zone_type].append(
                float(type_path.with_name("temp").read_text(encoding="utf-8").strip()) / 1000.0
            )
        except (OSError, ValueError):
            continue
    preferred = temperatures["TCPU"] or temperatures["x86_pkg_temp"]
    return max(preferred) if preferred else None


def _wait_for_temperature(maximum_c: float, resume_c: float, *, is_root: bool) -> None:
    temperature = _package_temperature_c()
    if temperature is None or temperature <= maximum_c:
        return
    if is_root:
        print(
            f"Thermal guard: CPU package is {temperature:.1f} C; "
            f"pausing until it reaches {resume_c:.1f} C",
            flush=True,
        )
    while temperature > resume_c:
        wall_clock.sleep(5.0)
        temperature = _package_temperature_c()
        if temperature is None:
            return
    if is_root:
        print(f"Thermal guard: resuming at {temperature:.1f} C", flush=True)


def _samplers(force_interval: float, line_interval: float):
    force_schedule = case.fvm.SamplingSchedule(every_time=force_interval)
    line_schedule = case.fvm.SamplingSchedule(every_time=line_interval)
    return (
        case.fvm.ForceSampler(
            patch_names=["cube"],
            reference_velocity=float(case.np.linalg.norm(case.FREESTREAM_VELOCITY)),
            reference_area=case.CUBE_SIDE**2,
            reference_length=case.CUBE_SIDE,
            moment_centre=[0.0, 0.0, 0.0],
            file_name="forces_history",
            schedule=force_schedule,
        ),
        case.fvm.LineSampler(
            start=[case.FVM_DOMAIN[0], 0.0, 0.0],
            end=[case.FVM_DOMAIN[1], 0.0, 0.0],
            spacing=case.SAMPLE_SPACING,
            file_name="centreline",
            schedule=line_schedule,
        ),
        case.fvm.LineSampler(
            start=[case.FVM_DOMAIN[0], case.OFFAXIS_Y, 0.0],
            end=[case.FVM_DOMAIN[1], case.OFFAXIS_Y, 0.0],
            spacing=case.SAMPLE_SPACING,
            file_name="offaxis_y075",
            schedule=line_schedule,
        ),
    )


def _mesh(surface_cell_size: float, background_cell_size: float):
    return case.fvm.AdaptiveCartesianMesher(
        domain=case.FVM_DOMAIN,
        max_cell_size=background_cell_size,
        surface_file=case.CUBE_STL,
        wall_patch_name="cube",
        surface_cell_size=surface_cell_size,
        refinements=(
            case.fvm.BoxRefinement(case.WAKE_BOX, surface_cell_size * 2.0, "wakeBox"),
            case.fvm.BoxRefinement(
                case.DOWNSTREAM_WAKE_BOX,
                surface_cell_size * 4.0,
                "downstreamWakeBox",
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-name", required=True)
    parser.add_argument("--surface-cell-size", type=float, required=True)
    parser.add_argument("--background-cell-size", type=float, default=0.5)
    parser.add_argument("--end-time", type=float, default=case.END_TIME)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--force-interval", type=float, default=0.05)
    parser.add_argument("--line-interval", type=float, default=0.25)
    parser.add_argument("--max-cpu-temperature", type=float, default=80.0)
    parser.add_argument("--resume-cpu-temperature", type=float, default=65.0)
    parser.add_argument("--thermal-check-steps", type=int, default=1)
    parser.add_argument("--minimum-step-wall-time", type=float, default=1.0)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    surface_h = _positive(arguments.surface_cell_size, "surface-cell-size")
    background_h = _positive(arguments.background_cell_size, "background-cell-size")
    end_time = _positive(arguments.end_time, "end-time")
    force_interval = _positive(arguments.force_interval, "force-interval")
    line_interval = _positive(arguments.line_interval, "line-interval")
    maximum_temperature = _positive(arguments.max_cpu_temperature, "max-cpu-temperature")
    resume_temperature = _positive(arguments.resume_cpu_temperature, "resume-cpu-temperature")
    if resume_temperature >= maximum_temperature:
        raise ValueError("resume-cpu-temperature must be below max-cpu-temperature")
    if arguments.cores < 1:
        raise ValueError("cores must be at least one")
    if arguments.thermal_check_steps < 1:
        raise ValueError("thermal-check-steps must be at least one")
    if arguments.minimum_step_wall_time < 0.0:
        raise ValueError("minimum-step-wall-time must be non-negative")

    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    markers = (
        output_directory / "samples" / "forces_history.csv",
        output_directory / "samples" / "grid_metadata.json",
    )
    if any(path.exists() for path in markers):
        raise SystemExit(f"Refusing to append to an existing trial in {output_directory}")

    mesh = _mesh(surface_h, background_h)
    samplers = _samplers(force_interval, line_interval)
    total_steps = int(math.ceil(end_time / case.FVM_TIME_STEP_SIZE))
    time_config = replace(
        case.FVM_SETUP.time,
        start_time=0.0,
        end_time=end_time,
        output_interval_steps=total_steps + 1,
        output_interval_time=None,
    )
    setup = replace(
        case.FVM_SETUP,
        case_name=f"reference_flow_grid_{arguments.grid_name}",
        cores=arguments.cores,
        logging=replace(case.FVM_SETUP.logging, interval_steps=25),
        time=time_config,
        samplers=samplers,
    )

    started = wall_clock.monotonic()
    solver = case.fvm.create_fvm_solver(setup, case_dir=output_directory, mesh=mesh)
    is_root = bool(solver.parallel.is_root)
    local_cells = (
        solver.parallel.n_owned if solver.parallel.is_partitioned else solver.mesh_data["n_cells"]
    )
    global_cells = int(solver.parallel.global_sum(int(local_cells)))
    initial_temperature = _package_temperature_c()
    if is_root:
        print(
            f"Grid {arguments.grid_name}: h/D={surface_h:g}, "
            f"cells={global_cells:,}, ranks={arguments.cores}, end_time={end_time:g}",
            flush=True,
        )

    completed = False
    try:
        tolerance = 0.5 * setup.time.time_step_size
        while solver.time < setup.time.end_time - tolerance:
            if solver.step % arguments.thermal_check_steps == 0:
                _wait_for_temperature(maximum_temperature, resume_temperature, is_root=is_root)
            step_started = wall_clock.monotonic()
            solver.advance()
            remaining_rest = arguments.minimum_step_wall_time - (
                wall_clock.monotonic() - step_started
            )
            if remaining_rest > 0.0:
                wall_clock.sleep(remaining_rest)
        completed = True
    finally:
        solver.close()

    if not completed:
        return

    elapsed = wall_clock.monotonic() - started
    final_temperature = _package_temperature_c()
    if is_root:
        samples_directory = output_directory / "samples"
        samples_directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": "openonda-cube-grid-trial/1",
            "status": "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "grid": arguments.grid_name,
            "physics": {
                "reynolds_number": case.REYNOLDS,
                "kinematic_viscosity": case.KINEMATIC_VISCOSITY,
                "freestream_velocity": list(case.FREESTREAM_VELOCITY),
                "cube_side": case.CUBE_SIDE,
            },
            "mesh": {
                "requested_surface_cell_size": surface_h,
                "surface_cell_size": mesh.effective_cell_size(surface_h),
                "near_wake_cell_size": mesh.effective_cell_size(surface_h * 2.0),
                "downstream_wake_cell_size": mesh.effective_cell_size(surface_h * 4.0),
                "background_cell_size": mesh.max_cell_size,
                "requested_domain": list(mesh.requested_domain),
                "effective_domain": list(mesh.effective_domain),
                "cell_count": global_cells,
            },
            "time": {
                "time_step_size": setup.time.time_step_size,
                "end_time": solver.time,
                "steps": solver.step,
                "force_interval": force_interval,
                "line_interval": line_interval,
            },
            "execution": {
                "mpi_ranks": arguments.cores,
                "wall_clock_seconds": elapsed,
                "initial_cpu_temperature_c": initial_temperature,
                "final_cpu_temperature_c": final_temperature,
                "max_cpu_temperature_c": maximum_temperature,
                "resume_cpu_temperature_c": resume_temperature,
                "minimum_step_wall_time_seconds": arguments.minimum_step_wall_time,
                "vtk_output": False,
            },
        }
        (samples_directory / "grid_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Completed grid {arguments.grid_name} in {elapsed / 60.0:.1f} min; "
            f"compact samples are in {samples_directory}",
            flush=True,
        )


if __name__ == "__main__":
    main()
