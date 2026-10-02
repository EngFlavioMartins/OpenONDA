"""Isolated full-operator screen of scope-owned target geometry reuse.

The existing read-only native checkpoint profiler owns input admission,
complete u/J/rate evaluation and unchanged NPZ publication. This driver only
installs the unwired geometry-bank context, then appends qualification hashes
and per-repeat geometry timings to that profiler's unique JSON artifact.

Use ``--bank-bytes 0`` for a contemporary unchanged-preparation control. Both
variants time the same preparation wrapper and complete native particle set.
Cold compilation remains the first measurement, not a warm speedup claim.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--bank-bytes", type=int, default=64 * 1024 * 1024)
    options, remaining = parser.parse_known_args()
    if options.bank_bytes < 0:
        raise ValueError("geometry bank byte cap must be nonnegative")
    excluded = {"--self-only", "--image-shell", "--exact-reuse", "--legacy-targets"}
    if excluded.intersection(remaining):
        raise ValueError("geometry-bank timing requires the complete uncached image operator")
    sys.argv = [sys.argv[0], *remaining]
    root = Path(__file__).resolve().parents[2]
    requested_root = Path(remaining[remaining.index("--source-root") + 1]).resolve()
    if requested_root != root:
        raise ValueError("this qualification must use its matching live production source root")
    output = Path(remaining[remaining.index("--output") + 1]).resolve()
    report_path = output.with_suffix(".json")
    if report_path.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(f"Refusing to overwrite qualification evidence: {output}")
    assets = root / "tests/support/cylinder"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(assets))
    from tests.vpm._fmm_target_geometry_bank_prototype import target_geometry_bank_factory

    prototype = Path(__file__).with_name("_fmm_target_geometry_bank_prototype.py")
    profiler = assets / "profile_induction_checkpoint.py"
    qualification_hashes = {str(path): _digest(path) for path in (Path(__file__), prototype, profiler)}
    specification = importlib.util.spec_from_file_location("native_geometry_bank_screen", profiler)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    production_hashes, records, failure = {}, [], None
    try:
        with target_geometry_bank_factory(max_bytes=options.bank_bytes) as records:
            # Factory imports all numerical backend dependencies before the
            # measurement, so capture their actual resolved implementation.
            for name, loaded in tuple(sys.modules.items()):
                if name.startswith(("source.", "openonda.")) and getattr(loaded, "__file__", None):
                    path = Path(loaded.__file__).resolve()
                    if not path.is_relative_to(root):
                        raise RuntimeError(f"Numerical source escaped the selected root: {path}")
                    production_hashes[str(path)] = _digest(path)
            module.main()
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        if report_path.exists():
            report = json.loads(report_path.read_text())
            changed = [
                path for path, expected in (production_hashes | qualification_hashes).items()
                if not Path(path).is_file() or _digest(Path(path)) != expected
            ]
            report["target_geometry_bank_qualification"] = {
                "memory_cap_bytes": options.bank_bytes,
                "mode": "unchanged-preparation-control" if options.bank_bytes == 0 else "geometry-bank",
                "scope": "one explicitly immutable complete Slab._images invocation",
                "shell_order_and_tail_tests_unchanged": True,
                "field_npz_written_by_original_profiler": True,
                "payload_bytes_per_capacity_slot": 96,
                "allocation_policy": "one bounded owner per backend; invalidated logical tiles per scope",
                "production_source_hashes": production_hashes,
                "qualification_source_hashes": qualification_hashes,
                "sources_changed": changed,
                "scopes": records,
            }
            if failure is not None or changed:
                report.update(status="failed", error=failure or "Qualification source changed during execution")
            elif report.get("status") != "complete":
                report.update(status="failed", error="Native profiler did not complete")
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "complete":
        raise RuntimeError(report.get("error", "Incomplete geometry qualification"))


if __name__ == "__main__":
    main()
