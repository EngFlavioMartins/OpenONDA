"""Keep the checked-in differential-case specifications executable."""

from __future__ import annotations

from pathlib import Path

from tools.mesh_parity.cfmesh_oracle import load_parity_spec

CASE_DIRECTORY = Path(__file__).with_name("cases")


def test_checked_in_parity_cases_are_complete_and_resolve_their_authority_stls():
    cases = [load_parity_spec(path) for path in sorted(CASE_DIRECTORY.glob("*.json"))]

    assert [case.name for case in cases] == [
        "cube_aligned",
        "cube_oblique",
        "cylinder_box_refinement",
        "cylinder_coarse",
        "cylinder_patch_refinement",
    ]
    assert all(surface.path.is_file() for case in cases for surface in case.surfaces)
