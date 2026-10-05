"""Continue the production 3D cylinder from a hash-verified native checkpoint.

Run into a new directory; the input checkpoint and tutorial samples remain
untouched. --forces-only changes output run_stages settings, which native restart
validation excludes from the numerical configuration, while retaining all equations,
the exact saved mesh, 24 span layers and the SlipSlabInduction configuration.
No startup perturbation is applied to a developed checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.vpm as vpm

TUTORIAL = (
    Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
)


def _positive_integer(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    restart = parser.add_mutually_exclusive_group(required=True)
    restart.add_argument("--restart-from", type=Path)
    restart.add_argument("--resume", action="store_true")
    parser.add_argument("--mesh", type=Path, default=TUTORIAL / "solution/fvm/mesh.npz")
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--max-coupling-steps", type=_positive_integer)
    parser.add_argument("--forces-only", action="store_true")
    options = parser.parse_args(argv)
    options.output_dir = options.output_dir.resolve()
    mesh = options.mesh.resolve(strict=True)
    source_description = None
    if not options.resume:
        options.restart_from = options.restart_from.resolve(strict=True)
        metadata_path = options.restart_from / "checkpoint_info.json"
        metadata_bytes = metadata_path.read_bytes()
        checkpoint_info = json.loads(metadata_bytes)
        source_description = {
            "restart_from": str(options.restart_from),
            "source_checkpoint_info_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "source_configuration_sha256": checkpoint_info["config_sha256"],
            "source_time": checkpoint_info["time"],
            "source_coupling_step": checkpoint_info["coupling_step"],
        }
    spec = importlib.util.spec_from_file_location(
        "cylinder_3d_drag_recovery_case", TUTORIAL / "setup.py"
    )
    tutorial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tutorial)
    setup, particles, exchange, _ = tutorial.build_case(end_time=options.end_time)
    if options.forces_only:
        setup = replace(
            setup,
            samplers=tuple(
                sample for sample in setup.samplers if isinstance(sample, fvm.ForceSampler)
            ),
            time=replace(setup.time, output_schedule=fvm.RunSchedule(final_only=True)),
        )
        particles = replace(particles, samplers=vpm.Samplers())
    source_hashes = {
        name: hashlib.sha256((TUTORIAL.parents[2] / name).read_bytes()).hexdigest()
        for name in (
            "source/coupler/stable_renewal.py",
            "source/solvers/fvm/coupling/coupler_interface.py",
        )
    }
    with coupling.create_coupler(
        setup,
        particles,
        exchange,
        mesh=mesh,
        case_dir=options.output_dir,
        require_empty_output=not options.resume,
    ) as solver:
        if solver._is_master:
            record = options.output_dir / "drag_recovery_continuation.json"
            if not options.resume:
                record.write_text(
                    json.dumps(
                        {
                            "scope": "production three-dimensional slip-slab continuation",
                            **source_description,
                            "mesh": str(mesh),
                            "end_time": setup.time.end_time,
                            "forces_only": options.forces_only,
                            "numerical_restart_allowed_paths": [],
                            "source_sha256": source_hashes,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            with (options.output_dir / "drag_recovery_attempts.jsonl").open("a") as attempts:
                attempts.write(
                    json.dumps(
                        {
                            "started_utc": datetime.now(UTC).isoformat(),
                            "resume": options.resume,
                            "max_coupling_steps": options.max_coupling_steps,
                            "source_sha256": source_hashes,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        solver.run(
            restart_from=None if options.resume else options.restart_from.resolve(strict=True),
            start_from="latest" if options.resume else None,
            max_coupling_steps=options.max_coupling_steps,
            backup_at_stop=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
