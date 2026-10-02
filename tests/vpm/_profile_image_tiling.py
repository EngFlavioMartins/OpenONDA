"""Isolated fair-prefix screen using the unchanged native operator profiler."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    index = sys.argv.index("--tile-capacity")
    capacity = int(sys.argv[index + 1])
    del sys.argv[index:index + 2]
    if "--image-shell" not in sys.argv or "--target-limit" not in sys.argv:
        raise ValueError("finite-shell and explicit identical target-prefix selection required")
    if int(sys.argv[sys.argv.index("--target-limit") + 1]) != 32768:
        raise ValueError("both variants must evaluate the full 32768-target prefix")
    root = Path(__file__).resolve().parents[2]
    assets = root / "tests/support/cylinder"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(assets))
    from tests.vpm._fmm_image_tiling_prototype import image_tiling_factory

    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    prototype = Path(__file__).with_name("_fmm_image_tiling_prototype.py")
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (prototype, Path(__file__))}
    spec = importlib.util.spec_from_file_location("native_tiling_screen", assets / "profile_induction_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    try:
        with image_tiling_factory(capacity) as records:
            module.main()
    finally:
        if output.with_suffix(".json").exists():
            report = json.loads(output.with_suffix(".json").read_text())
            report["tiling_qualification"] = {
                "tile_capacity": capacity, "full_target_prefix": 32768,
                "hard_pair_capacity_unchanged": 4194304, "calls": records,
                "all_tile_work_is_recorded_here": True,
                "original_outer_block_diagnostics_describe_final_tile_only": True,
                "qualification_source_hashes": hashes,
                "qualification_sources_changed": [
                    path for path, digest in hashes.items()
                    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
                ],
            }
            if report["tiling_qualification"]["qualification_sources_changed"]:
                report.update(status="failed", error="Qualification source changed while running")
            output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
