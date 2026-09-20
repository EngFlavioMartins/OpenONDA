"""One-shot local completion stage for the already launched 160 s cylinder pair.

Exit 0: postprocessed and mature gate passed; 2: operational failure;
3: postprocessed but scientific gate did not pass. Never starts a simulation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "studies/panel_removal/runs/cylinder_no_panel_converged"
DEFAULT_CAMPAIGN = ROOT / (
    "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/campaigns/geometric_xy_v1"
)


def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # Some solver metadata writers replace content in place. Retry next poll.
        return {}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity(pid):
    """Pin kernel start time and full command, including MPI exec replacement."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat=", "-o", "command="],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        check=False,
    )
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return None
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(f"Cannot inspect PID {pid}: {result.stderr.strip()}")
    fields = result.stdout.strip().split(maxsplit=6)
    if len(fields) != 7:
        raise RuntimeError(f"Unrecognized ps identity for PID {pid}")
    return {
        "pid": int(pid),
        "started": " ".join(fields[:5]),
        "command": fields[6],
        "alive": not any(c in fields[5] for c in "ZX"),
    }


def validate_command(identity, role, run, campaign):
    args = shlex.split(identity["command"])
    if role == "experiment":
        script = str(ROOT / "studies/panel_removal/run_cylinder.py")
        module = ["-m", "studies.panel_removal.run_cylinder"]
        has_script = script in args or any(args[i : i + 2] == module for i in range(len(args)))
        has_output = any(args[i : i + 2] == ["--output", str(run)] for i in range(len(args)))
        valid = has_script and has_output
    else:
        valid = str(campaign.parents[1] / "allrun.sh") in args
    if not valid:
        raise RuntimeError(f"PID command does not identify the expected {role}: {identity}")


def check_identity(expected):
    current = process_identity(expected["pid"])
    if current is None:
        return False
    for key in ("started", "command"):
        if current[key] != expected[key]:
            raise RuntimeError(f"PID {expected['pid']} was reused or its command changed")
    return current["alive"]


@contextmanager
def exclusive_lock(run):
    # Stable inode: do not unlink on exit. Kernel releases the lock even after SIGKILL.
    with (run / ".cylinder_comparison.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another cylinder completion stage holds this run's lock") from error
        yield


def at_end(value, end):
    try:
        return math.isfinite(float(value)) and math.isclose(float(value), end, abs_tol=1e-7)
    except (TypeError, ValueError):
        return False


def tail(path, size=131072):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode("utf-8", errors="replace")


def last_sample_time(path):
    for line in reversed(tail(path).splitlines()):
        try:
            return float(line.split(",", 1)[0])
        except ValueError:
            continue
    return None


def failure_evidence(directory):
    # A fresh experiment/campaign launcher log belongs to this launch; no restart
    # through a failed lineage is silently accepted by this one-shot stage.
    text = tail(directory / "launcher.log")
    return any(
        marker in text
        for marker in ("Traceback (most recent call last)", "RUN FAILED", "MPI_ABORT")
    )


def completion_evidence(run, campaign, end=160.0):
    reference = campaign / "spatial"
    solution = reference / "solution/xy_fine"
    samples = reference / "samples/xy_fine"
    experiment_fvm = read_json(run / "solution/fvm_metadata.json")
    experiment_vpm = read_json(run / "solution/vpm_metadata.json")
    reference_fvm = read_json(solution / "fvm_metadata.json")
    for label, metadata in (
        ("experiment FVM", experiment_fvm),
        ("experiment VPM", experiment_vpm),
        ("reference FVM", reference_fvm),
    ):
        if metadata.get("lifecycle", {}).get("status") in ("failed", "not_converged"):
            raise RuntimeError(f"{label} reports failed lifecycle")
    for label, directory in (("experiment", run), ("reference campaign", campaign)):
        if failure_evidence(directory):
            raise RuntimeError(f"{label} launcher reports a failure")
    backup = read_json(run / "solution/backups/manifest.json")
    diagnostic = {}
    lines = tail(run / "solution/coupler_diagnostics.jsonl").splitlines()
    if lines:
        with suppress(json.JSONDecodeError):
            diagnostic = json.loads(lines[-1])
    manifest = read_json(run / "experiment.json")
    dt = manifest.get("coupling_dt", 0)
    step = round(end / dt) if isinstance(dt, int | float) and dt > 0 else None
    grid = read_json(samples / "grid_run.json")
    artifacts = backup.get("artifacts", {})
    backup_files = all(
        (run / "solution/backups" / str(artifacts.get(k, "missing"))).exists()
        for k in ("fvm", "vpm", "vpm_boundary_condition")
    )
    trial_checks = {
        "configured_end": at_end(manifest.get("end_time"), end),
        "fvm_closed": experiment_fvm.get("lifecycle", {}).get("status") == "complete",
        "fvm_end": at_end(experiment_fvm.get("state", {}).get("time"), end),
        "vpm_closed": experiment_vpm.get("lifecycle", {}).get("status") == "completed",
        "vpm_end": at_end(experiment_vpm.get("state", {}).get("time"), end),
        "committed_backup": backup.get("kind") == "openonda.coupled_backup" and backup_files,
        "backup_end": at_end(backup.get("time"), end)
        and step is not None
        and backup.get("coupling_step") == step,
        "diagnostic_end": at_end(diagnostic.get("time"), end) and diagnostic.get("step") == step,
        "force_end": at_end(last_sample_time(run / "samples/forces_history.csv"), end),
    }
    ref_checks = {
        "registered_grid": grid.get("schema") == "openonda-fvm-grid-run/1"
        and grid.get("case") == "xy_fine",
        "registered_end": at_end(grid.get("end_time"), end),
        "fvm_closed": reference_fvm.get("lifecycle", {}).get("status") == "complete",
        "fvm_end": at_end(reference_fvm.get("state", {}).get("time"), end),
        "force_end": at_end(last_sample_time(samples / "forces_history.csv"), end),
    }
    return {
        "experiment": {"complete": all(trial_checks.values()), "checks": trial_checks},
        "reference": {"complete": all(ref_checks.values()), "checks": ref_checks},
    }


def ready(evidence, identities):
    alive = {
        role: check_identity(identity) if identity else False
        for role, identity in identities.items()
    }
    for role in ("experiment", "reference"):
        if not alive[role] and not evidence[role]["complete"]:
            raise RuntimeError(f"{role} process is dead before validated 160 s completion")
    # The reference campaign may continue finer/control cases after xy_fine closes.
    return all(item["complete"] for item in evidence.values()) and not alive["experiment"], alive


def comparator_command(run, campaign, destination):
    return [
        sys.executable,
        str(ROOT / "studies/panel_removal/compare_cylinder_run.py"),
        str(run),
        "--reference",
        str(campaign / "spatial/samples/xy_fine"),
        "--start",
        "80",
        "--end",
        "160",
        "--output",
        str(destination / "comparison.json"),
        "--plot",
        str(destination / "fvm.png"),
        "--plot-vpm",
        str(destination / "vpm.png"),
        "--snapshot-time",
        "160",
    ]


def execute(run, campaign, interval):
    destination = run / "mature_comparison"
    destination.mkdir(exist_ok=True)
    state_path = destination / "status.json"
    state = read_json(state_path)
    if state.get("status") == "postprocessed":
        print(f"Already postprocessed: {state_path}", flush=True)
        return 0 if state.get("mature_gate_passed") else 3
    if state and (state.get("run") != str(run) or state.get("campaign") != str(campaign)):
        raise RuntimeError("Existing completion stage belongs to different input paths")
    identities = state.get("identities", {})

    def publish(status, **details):
        state.update(
            status=status,
            updated_utc=datetime.now(UTC).isoformat(),
            run=str(run),
            campaign=str(campaign),
            identities=identities,
            **details,
        )
        write_json(state_path, state)
        print(f"{state['updated_utc']} {status}: {details}", flush=True)

    try:
        evidence = completion_evidence(run, campaign)
        for role, directory in (("experiment", run), ("reference", campaign)):
            if role in identities:
                continue
            record = read_json(directory / "process.json")
            if not isinstance(record.get("pid"), int) or record["pid"] <= 1:
                raise RuntimeError(f"Missing valid {role} process.json PID")
            identity = process_identity(record["pid"])
            if identity:
                validate_command(identity, role, run, campaign)
            if identity is None and not evidence[role]["complete"]:
                raise RuntimeError(
                    f"Cannot attach: {role} PID is already dead and outputs are incomplete"
                )
            identities[role] = identity
        publish("attached", evidence=evidence)
        while True:
            evidence = completion_evidence(run, campaign)
            finished, alive = ready(evidence, identities)
            publish("ready" if finished else "waiting", evidence=evidence, alive=alive)
            if finished:
                break
            time.sleep(interval)
        command = comparator_command(run, campaign, destination)
        publish("postprocessing", command=command)
        with (destination / "comparator.log").open("w") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "MPLCONFIGDIR": str(destination / "mpl_cache")},
            )
        if result.returncode:
            raise RuntimeError(
                f"Comparator exited {result.returncode}; see {destination / 'comparator.log'}"
            )
        comparison = read_json(destination / "comparison.json")
        if not isinstance(comparison.get("mature_gate_passed"), bool):
            raise RuntimeError("Comparator did not produce its required scientific gate")
        for name in ("fvm.png", "vpm.png"):
            if not (destination / name).is_file():
                raise RuntimeError(f"Comparator did not produce {name}")
        passed = comparison["mature_gate_passed"]
        publish(
            "postprocessed",
            mature_gate_passed=passed,
            comparison=str(destination / "comparison.json"),
        )
        return 0 if passed else 3
    except Exception as error:
        publish("failed", error=str(error))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 1 and 60")
    try:
        with exclusive_lock(args.run.resolve()):
            return execute(args.run.resolve(), args.campaign.resolve(), args.poll_seconds)
    except Exception as error:
        print(f"Completion stage refused: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
