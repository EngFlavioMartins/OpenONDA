#!/usr/bin/env python3
"""Freeze inputs, then run the cylinder reference before a matched coupled run.

Never clean outputs or stop unrelated jobs. Only owned children can be stopped
at their declared wall limit. Dependency/resource waits are not simulations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[4]
RELATIVE = Path("tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow")
CUBE = Path("/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/02_cube_flow/study_results/cause-20260923.1V1kOE")
PYTHON = "/home/flavio-martins/anaconda3/envs/OpenONDA/bin/python"


def files(root):
    return sorted(p for part in ("source", "openonda", str(RELATIVE))
                  for p in (root/part).rglob("*") if p.suffix in (".py", ".stl"))


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files(root)}


def jobs(frozen):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry/"cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        scripts = [a for a in args if a.endswith(".py") and
                   (a.startswith(str(CUBE)+"/") or a.startswith(str(frozen)+"/"))]
        # Also refuse a reference/coupled MPI run owned by a different task.
        mpi = args and Path(args[0]).name in ("prterun", "mpiexec", "mpirun", "orterun")
        if scripts or mpi:
            found.append(dict(pid=int(entry.name), scripts=scripts, mpi=bool(mpi)))
    return found


def resources(gpu):
    mem = {r.split(":")[0]: int(r.split()[1]) for r in Path("/proc/meminfo").read_text().splitlines()}
    available = mem["MemAvailable"]//1024
    facts = dict(available_host_mib=available, load_average=os.getloadavg()[0])
    ready = available >= (6000 if gpu else 5000)
    ready &= facts["load_average"] < max(4, .75*(os.cpu_count() or 1))
    if gpu:
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"], text=True)
        for line in raw.splitlines():
            if not line.strip():
                continue
            pid, memory = map(int, line.split(","))
            try:
                executable = str((Path("/proc")/str(pid)/"exe").resolve(strict=True))
            except FileNotFoundError:
                continue
            if executable != "/usr/share/rustdesk/rustdesk" or memory > 256:
                ready = False
                facts["other_gpu_process"] = pid
        free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"], text=True).strip())
        facts["free_gpu_mib"] = free
        ready &= free >= 4500
    return bool(ready), facts


def append(root, event, **facts):
    row = dict(epoch=time.time(), event=event, **facts)
    with (root/"events.jsonl").open("a") as stream:
        stream.write(json.dumps(row)+"\n")
    print(json.dumps(row), flush=True)


def benchmark_command(assets, kind, root, *flags):
    # Keep the queue's resource gate and the actual supported backend aligned.
    return [PYTHON, str(assets/"run_phase_benchmark.py"), kind, "--root", str(root),
            "--device", "CPU", *flags]


def worker(root):
    from openonda.cylinder_campaign import run_trial, collect_cost
    from check_phase_samples import inspect
    frozen = root/"frozen"
    launch = json.loads((root/"launch.json").read_text())
    assets = frozen/RELATIVE/"assets"
    deadline = time.time()+24*3600

    def wait_idle(gpu):
        reported = False
        while True:
            active = jobs(frozen)
            ready, facts = resources(gpu)
            if not active and ready:
                assert hashes(frozen) == launch["source_sha256"], "Frozen numerical inputs changed"
                return facts
            if not reported:
                append(root, "waiting_for_idle_resources", jobs=active, gpu_required=gpu, **facts)
                reported = True
            if time.time() > deadline:
                raise TimeoutError("Dependency/resources unavailable for24h; no competing run launched")
            time.sleep(30)

    def child(kind, label, limit, *flags):
        wait_idle(False)  # Both queued stages explicitly execute on CPU.
        command = benchmark_command(assets, kind, root, *flags)
        append(root, "child_starting", kind=kind, stage=label, wall_limit=limit)
        result = run_trial(command, root/"logs"/label, cwd=frozen, wall_limit=limit)
        append(root, "child_exited", stage=label, returncode=result["returncode"],
               timed_out=result["timed_out"], wall_seconds=result["wall_seconds"])
        if result["returncode"] or result["timed_out"]:
            raise RuntimeError(f"{label} did not complete; preserve checkpoint and inspect log")
        return result

    try:
        # Existing owned cube continuation/traces remain undisturbed. The idle
        # cube refinement queue was explicitly deferred before this was queued.
        while not (CUBE/"host-recovery-batch-20260929/result.json").exists() or jobs(frozen):
            if time.time() > deadline:
                raise TimeoutError("Existing simulation has not exited in24h")
            time.sleep(30)
        predecessor = json.loads((CUBE/"host-recovery-batch-20260929/result.json").read_text())
        append(root, "predecessor_exited", result=predecessor,
               note="Cylinder reference is an independent experiment, not a retry of the cube failure.")
        child("reference", "reference", 12*3600)
        assert json.loads((root/"reference/run-result.json").read_text())["status"] == "completed"
        reference = inspect(root, "reference", 100.)
        append(root, "reference_samples_verified", force_signal=reference["force_signal"])
        pilot = child("coupled", "coupled-pilot", 3600, "--pilot")
        assert json.loads((root/"coupled/pilot-result.json").read_text())["status"] == "pilot-completed"
        inspect(root, "coupled", .8)
        cost = collect_cost(root/"coupled")
        (root/"coupled/pilot-cost.json").write_text(json.dumps(cost, indent=2)+"\n")
        append(root, "coupled_pilot_samples_verified", accepted_intervals=20,
               remaining_wall_budget=12*3600-pilot["wall_seconds"])
        child("coupled", "coupled-continuation", 12*3600-pilot["wall_seconds"], "--resume")
        assert json.loads((root/"coupled/continuation-result.json").read_text())["status"] == "completed"
        coupled = inspect(root, "coupled", 100.)
        result = dict(status="completed", reference_signal=reference["force_signal"],
            coupled_signal=coupled["force_signal"],
            scope="Matched provisional benchmark data ready; phase cause/fix and grid convergence not yet proved.")
    except BaseException as error:
        result = dict(status="needs_review", error=repr(error))
        raise
    finally:
        (root/"result.json").write_text(json.dumps(result, indent=2)+"\n")


def launch(root):
    import phase_benchmark
    decision_path = CUBE/"cylinder-recovery-decision-20260929.json"
    if not decision_path.is_file():
        raise RuntimeError("Await the user's legacy-output recovery decision before new write-heavy runs")
    decision = json.loads(decision_path.read_text())
    assert decision.get("choice") in ("proceed_fresh", "restored_then_proceed")
    validation = phase_benchmark.contract()
    assert root.is_absolute() and not root.exists()
    assert (CUBE/"phase-temporal-batch-20260929/deferred-for-cylinder.json").is_file()
    frozen = root/"frozen"
    root.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "solution", "samples", "figures", "study_results")
    for folder in ("source", "openonda", str(RELATIVE)):
        shutil.copytree(REPOSITORY/folder, frozen/folder, ignore=ignore)
    source_hashes = hashes(frozen)
    assert all(hashlib.sha256((REPOSITORY/p).read_bytes()).hexdigest() == h
               for p, h in source_hashes.items())
    launch = dict(epoch=time.time(), config=validation, source_sha256=source_hashes,
        recovery_decision=decision,
        repository=str(REPOSITORY), prior_cube_batch=str(CUBE/"host-recovery-batch-20260929"),
        limits=__doc__, reference_wall_limit=43200, coupled_total_wall_limit=43200,
        grid_choice="h=.04D (25 cells/diameter,24 span layers), a provisional practical choice; legacy force study is not stationary/grid-qualified.")
    (root/"launch.json").write_text(json.dumps(launch, indent=2)+"\n")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(frozen), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLCONFIGDIR=str(root/"matplotlib-cache"))
    with (root/"worker.log").open("xb") as log:
        process = subprocess.Popen([PYTHON, str(frozen/RELATIVE/"assets/queue_phase_benchmark.py"),
            "--root", str(root), "--worker"], cwd=frozen, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root/"worker.json").write_text(json.dumps(dict(pid=process.pid))+"\n")
    print(json.dumps(dict(status="queued", pid=process.pid, root=str(root))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    options = parser.parse_args()
    (worker if options.worker else launch)(options.root.resolve())
