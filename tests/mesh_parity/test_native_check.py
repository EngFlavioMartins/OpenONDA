"""Regression tests for diagnostic, rather than exit-code, checkMesh gates."""

from tests.mesh_parity.native_check import parse_checkmesh_output


def test_checkmesh_failed_summary_overrides_zero_exit_status():
    result = parse_checkmesh_output("Mesh OK.\nFailed 7 mesh checks.\n", exit_code=0)

    assert result["status"] == "fail"
    assert result["failed_checks"] == 7
    assert result["passed"] is False


def test_checkmesh_mesh_ok_summary_passes():
    result = parse_checkmesh_output("Mesh OK.\n", exit_code=0)

    assert result == {
        "status": "pass",
        "passed": True,
        "failed_checks": None,
        "mesh_ok_summary": True,
        "exit_code": 0,
    }


def test_checkmesh_missing_summary_is_not_a_pass():
    result = parse_checkmesh_output("checkMesh terminated before the summary", exit_code=0)

    assert result["status"] == "indeterminate"
    assert result["passed"] is False
