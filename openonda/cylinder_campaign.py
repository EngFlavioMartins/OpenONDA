"""Bounded process execution and measurements for the cylinder study."""

from __future__ import annotations

from collections import deque
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np
import psutil
from scipy.integrate import trapezoid


def cylinder_initial_velocity(
    positions: np.ndarray, span: float, amplitude: float = 1e-3
) -> np.ndarray:
    """Return a reproducible divergence-free 3D perturbation of unit inflow.

    Coordinates and span are in metres for the D=1, U=1 cylinder case. The
    perturbation is curl(0, A_y, A_z), with compact XY support around x=.65,
    sinusoidal A_y vanishing at the slip planes and span-independent A_z.
    The latter breaks transverse reflection symmetry to seed shedding without
    relying on mesh asymmetry or roundoff. Thus w=0 and the normal derivative
    of tangential velocity vanishes on both planes. Amplitude is
    the dimensionless vector-potential coefficient; this is an initial test
    disturbance, not a sustained forcing or a prescribed turbulent state.
    """
    if not np.isfinite(span) or span <= 0 or not np.isfinite(amplitude):
        raise ValueError("span must be positive and perturbation amplitude finite")
    position = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    x, y, z = position.T
    radius_squared = 0.5**2
    support = np.maximum(1 - ((x - 0.65) ** 2 + y**2) / radius_squared, 0)
    wave_number = np.pi / span
    phase = wave_number * (z + 0.5 * span)
    velocity = np.zeros_like(position)
    velocity[:, 0] = (
        1
        - amplitude * wave_number * support**4 * np.cos(phase)
        - 8 * amplitude * y / radius_squared * support**3
    )
    velocity[:, 1] = 8 * amplitude * (x - 0.65) / radius_squared * support**3
    velocity[:, 2] = -8 * amplitude * (x - 0.65) / radius_squared * support**3 * np.sin(phase)
    return velocity


def initialize_cylinder_perturbation(solver, span: float) -> None:
    """Install the same small 3D initial disturbance on each local FVM mesh."""
    count = solver.mesh_data["n_cells"]
    solver.set_initial_velocity(
        cylinder_initial_velocity(solver.geo_data["cell_centre"][:count], span)
    )


def profile_statistics(path: Path, start: float = 40, end: float = 100) -> dict:
    """Time-weight a transverse velocity profile over an exact covered window.

    Returns y coordinates, three-component means and fluctuation RMS values
    in SI units. The physical span probe uses fixed x=1 transverse lines.
    Missing/duplicate timestamps and changing spatial support are rejected.
    """
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    required = ("time", "position_y", "velocity_x", "velocity_y", "velocity_z")
    if not all(name in (data.dtype.names or ()) for name in required):
        raise ValueError(f"missing velocity profile columns: {path}")
    if any(not np.all(np.isfinite(data[name])) for name in required):
        raise ValueError(f"nonfinite profile values: {path}")
    coordinates = np.unique(data["position_y"])
    means, fluctuations = [], []
    common_times = None
    for coordinate in coordinates:
        rows = data[data["position_y"] == coordinate]
        time_values = rows["time"]
        if len(rows) < 2 or np.any(np.diff(time_values) <= 0):
            raise ValueError(f"profile times are not strictly increasing: {path}")
        if common_times is None:
            common_times = time_values
        elif not np.array_equal(common_times, time_values):
            raise ValueError(f"profile spatial support changes with time: {path}")
        if time_values[0] > start or time_values[-1] < end:
            raise ValueError(f"profile does not cover [{start}, {end}]: {path}")
        times = np.r_[start, time_values[(time_values > start) & (time_values < end)], end]
        values = np.column_stack(
            [np.interp(times, time_values, rows[f"velocity_{axis}"]) for axis in "xyz"]
        )
        mean = trapezoid(values, times, axis=0) / (end - start)
        centered = values - mean
        variance = np.sum(
            np.diff(times)[:, None]
            * (centered[:-1] ** 2 + centered[:-1] * centered[1:] + centered[1:] ** 2)
            / 3,
            axis=0,
        ) / (end - start)
        rms = np.sqrt(np.maximum(variance, 0))
        means.append(mean.tolist())
        fluctuations.append(rms.tolist())
    return {"y": coordinates.tolist(), "mean_velocity": means, "rms_velocity": fluctuations}


def compare_profiles(candidate: dict, reference: dict) -> dict:
    """Return spatial L2 errors of time-mean and RMS velocity, normalized by U=1."""
    left, right = np.asarray(candidate["y"]), np.asarray(reference["y"])
    if min(len(left), len(right)) < 2 or not np.allclose(left[[0, -1]], right[[0, -1]], atol=1e-7):
        raise ValueError("profile endpoints must describe the same physical observation line")
    coordinate = np.unique(np.r_[left, right])
    result = {}
    for field in ("mean_velocity", "rms_velocity"):
        a, b = np.asarray(candidate[field]), np.asarray(reference[field])
        difference = np.column_stack(
            [
                np.interp(coordinate, left, a[:, axis]) - np.interp(coordinate, right, b[:, axis])
                for axis in range(3)
            ]
        )
        result[field + "_l2"] = float(
            np.sqrt(
                trapezoid(np.sum(difference**2, axis=1), coordinate)
                / (coordinate[-1] - coordinate[0])
            )
        )
    return result


def _normalized_trial_command(command: list[str]) -> list[str]:
    """Normalize resume invocations so repeated attempts share one identity."""
    return [argument for argument in command if argument != "--resume"]


def _prior_trial_attempts(
    path: Path,
    normalized_command: list[str],
    *,
    accept_command_changes: bool = False,
) -> list[dict]:
    """Read and validate prior attempts for a resumed trial."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resume malformed trial record: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"cannot resume malformed trial record: {path}")

    def validate(attempts: object) -> list[dict]:
        if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
            raise ValueError(f"cannot resume malformed trial attempts: {path}")
        for item in attempts:
            command = item.get("command")
            wall_seconds = item.get("wall_seconds")
            if (
                not isinstance(command, list)
                or not all(isinstance(argument, str) for argument in command)
                or isinstance(wall_seconds, bool)
                or not isinstance(wall_seconds, (int, float))
                or not np.isfinite(wall_seconds)
                or wall_seconds < 0.0
            ):
                raise ValueError(f"cannot resume malformed trial attempt: {path}")
        return attempts

    if payload.get("normalized_command") == normalized_command and "attempts" in payload:
        return validate(payload["attempts"])
    if accept_command_changes and "attempts" in payload:
        return validate(payload["attempts"])
    legacy_command = payload.get("command")
    legacy_wall = payload.get("wall_seconds")
    if (
        isinstance(legacy_command, list)
        and all(isinstance(argument, str) for argument in legacy_command)
        and (
            accept_command_changes
            or _normalized_trial_command(legacy_command) == normalized_command
        )
        and not isinstance(legacy_wall, bool)
        and isinstance(legacy_wall, (int, float))
        and np.isfinite(legacy_wall)
        and legacy_wall >= 0.0
    ):
        return [
            {
                "command": legacy_command,
                "returncode": payload.get("returncode"),
                "timed_out": bool(payload.get("timed_out", False)),
                "wall_seconds": float(legacy_wall),
            }
        ]
    if accept_command_changes:
        raise ValueError(f"cannot resume trial record with unsupported schema: {path}")
    return []


class _ProcessTreeRSSSampler:
    """Sample a worker and all descendants without coupling to solver code."""

    def __init__(self, pid: int, period: float = 0.25):
        try:
            self._process = psutil.Process(pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            self._process = None
        self.period = period
        self.peak_bytes = 0
        self.sample_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        if self._process is None:
            return
        try:
            children = self._process.children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            return
        rss = 0
        for process in (self._process, *children):
            try:
                rss += process.memory_info().rss
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if rss <= 0:
            return
        self.peak_bytes = max(self.peak_bytes, rss)
        self.sample_count += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.period)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, 2.0 * self.period))


def run_trial(
    command: list[str],
    directory: Path,
    *,
    cwd: Path,
    wall_limit: float,
    resume: bool = False,
) -> dict:
    """Run one case, retain its log, and stop its MPI process group on timeout.

    Parameters
    ----------
    command : list of str
        Explicit Python command and arguments; no shell interpretation.
    directory : Path
        Trial record directory, separate from the worker's output directory.
    cwd : Path
        Working directory for the child process.
    wall_limit : float
        Positive total wall-clock budget in seconds for this case, including
        meshing and startup. Compatible resumed attempts consume the same
        budget.
    resume : bool, default False
        Treat an existing compatible trial record as a resumed attempt. The
        command's literal ``--resume`` flag also enables this mode so existing
        pipeline callers retain cumulative execution accounting.

    Returns
    -------
    dict
        Command, status, elapsed time and log path. The same record is saved
        as ``trial.json`` even when a process fails or exceeds its allowance.
    """
    if not np.isfinite(wall_limit) or wall_limit <= 0:
        raise ValueError("wall_limit must be positive and finite")
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "console.log"
    trial_path = directory / "trial.json"
    normalized_command = _normalized_trial_command(command)
    resumed = resume or "--resume" in command
    prior_attempts = (
        _prior_trial_attempts(trial_path, normalized_command, accept_command_changes=True)
        if resumed
        else []
    )
    prior_wall_seconds = sum(float(item.get("wall_seconds", 0.0)) for item in prior_attempts)
    remaining_wall_limit = wall_limit - prior_wall_seconds
    if remaining_wall_limit <= 0.0:
        cumulative_peak_rss = max(
            (int(item.get("peak_process_tree_rss_bytes") or 0) for item in prior_attempts),
            default=0,
        )
        cumulative_rss_samples = sum(
            int(item.get("rss_sample_count", 0)) for item in prior_attempts
        )
        sample_period = float(
            prior_attempts[-1].get("rss_sample_period_seconds", 0.25) if prior_attempts else 0.25
        )
        log.touch(exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(
                "wall budget exhausted before launching resumed attempt; "
                f"prior={prior_wall_seconds:.6f}s limit={wall_limit:.6f}s\n"
            )
        record = {
            "command": command,
            "returncode": 124,
            "timed_out": True,
            "budget_exhausted": True,
            "normalized_command": normalized_command,
            "resume": resumed,
            "attempt_wall_seconds": 0.0,
            "cumulative_wall_seconds": prior_wall_seconds,
            "peak_process_tree_rss_bytes": cumulative_peak_rss or None,
            "rss_sample_count": cumulative_rss_samples,
            "rss_sample_period_seconds": sample_period,
            "wall_limit_seconds": wall_limit,
            "remaining_wall_seconds": 0.0,
            "wall_seconds": prior_wall_seconds,
            "attempts": prior_attempts,
            "console_log": str(log),
        }
        trial_path.write_text(json.dumps(record, indent=2) + "\n")
        return record
    started = time.monotonic()
    timed_out = False
    with log.open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        sampler = _ProcessTreeRSSSampler(child.pid)
        sampler.start()
        try:
            try:
                returncode = child.wait(timeout=remaining_wall_limit)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                timed_out = isinstance(error, subprocess.TimeoutExpired)
                if os.name == "posix":
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        with suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait()
                returncode = 124 if timed_out else 130
        finally:
            sampler.stop()
    attempt = {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - started,
        "peak_process_tree_rss_bytes": sampler.peak_bytes,
        "rss_sample_count": sampler.sample_count,
        "rss_sample_period_seconds": sampler.period,
        "wall_limit_seconds": wall_limit,
        "remaining_wall_seconds": remaining_wall_limit,
    }
    attempts = [*prior_attempts, attempt]
    cumulative_wall_seconds = sum(float(item.get("wall_seconds", 0.0)) for item in attempts)
    cumulative_peak_rss = max(
        (int(item.get("peak_process_tree_rss_bytes") or 0) for item in attempts), default=0
    )
    cumulative_rss_samples = sum(int(item.get("rss_sample_count", 0)) for item in attempts)
    record = {
        **attempt,
        "normalized_command": normalized_command,
        "resume": resumed,
        "attempt_wall_seconds": attempt["wall_seconds"],
        "cumulative_wall_seconds": cumulative_wall_seconds,
        "peak_process_tree_rss_bytes": cumulative_peak_rss or None,
        "rss_sample_count": cumulative_rss_samples,
        "rss_sample_period_seconds": sampler.period,
        "wall_seconds": cumulative_wall_seconds,
        "attempts": attempts,
        "console_log": str(log),
    }
    trial_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def collect_cost(root: Path) -> dict:
    """Summarize saved solver measurements without loading whole histories.

    Late step samples are bounded to 256 records per journal. RSS is the
    reported aggregate over ranks; process startup and I/O are represented by
    the enclosing trial's wall time, not reconstructed from phase sums.
    """
    result = {
        "peak_rank_aggregate_rss_bytes": None,
        "peak_particles": 0,
        "journals": [],
        "accepted_coupling_intervals": 0,
        "unconverged_stationary_intervals": 0,
    }
    for path in sorted(root.rglob("performance.jsonl")):
        late = deque(maxlen=256)
        count = 0
        first_time = last_time = None
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                late.append(float(row["step_seconds"]["max"]))
                first_time = row["time"] if first_time is None else first_time
                last_time = row["time"]
                rss = row.get("memory", {}).get("aggregate_peak_rss_end_bytes")
                if rss is not None and float(rss) > 0.0:
                    result["peak_rank_aggregate_rss_bytes"] = max(
                        result["peak_rank_aggregate_rss_bytes"] or 0.0, float(rss)
                    )
        result["journals"].append(
            {
                "path": str(path),
                "steps": count,
                "first_time": first_time,
                "last_time": last_time,
                "late_step_median_seconds": float(np.median(late)) if late else None,
                "late_step_p90_seconds": float(np.quantile(late, 0.9)) if late else None,
            }
        )
    for path in root.rglob("coupler_diagnostics.jsonl"):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                # Retain the raw final diagnostic so every conservation and
                # interface gate remains inspectable alongside runtime.
                result["last_coupler_diagnostic"] = row
                result["accepted_coupling_intervals"] += 1
                if (
                    float(row.get("time", 0)) >= 40
                    and row.get("interface_iteration", {}).get("converged") is False
                ):
                    result["unconverged_stationary_intervals"] += 1
                result["peak_particles"] = max(
                    result["peak_particles"], int(row.get("n_transfer_particles", 0))
                )
    return result


__all__ = ["collect_cost", "run_trial"]
