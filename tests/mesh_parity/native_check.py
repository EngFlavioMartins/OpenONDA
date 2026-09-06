#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parse OpenFOAM ``checkMesh`` diagnostics for acceptance evidence.

OpenFOAM checkMesh can print failed checks and still return status zero.  The
acceptance harness therefore consumes the diagnostic text, not only the
process exit code.  This module is intentionally independent of an installed
OpenFOAM runtime so its parser can be unit-tested everywhere.
"""

from __future__ import annotations

import re

_FAILED_CHECKS = re.compile(r"\bFailed\s+(\d+)\s+mesh checks?\.?", re.IGNORECASE)
_PASSED_CHECKS = re.compile(r"\bMesh\s+OK\.?", re.IGNORECASE)


def parse_checkmesh_output(output: str, *, exit_code: int | None = None) -> dict[str, object]:
    """Return a pass/fail record from complete ``checkMesh`` output.

    A zero exit status is not sufficient: the explicit ``Failed N mesh
    checks`` summary takes precedence.  Output without either a success or a
    failure summary is treated as indeterminate and therefore fails the
    acceptance gate.
    """
    failed_match = _FAILED_CHECKS.search(output)
    passed = bool(_PASSED_CHECKS.search(output))
    failed_checks = int(failed_match.group(1)) if failed_match else None
    if failed_checks is not None:
        status = "pass" if failed_checks == 0 else "fail"
    elif passed:
        status = "pass"
    else:
        status = "indeterminate"
    if exit_code not in (None, 0):
        status = "fail"
    return {
        "status": status,
        "passed": status == "pass",
        "failed_checks": failed_checks,
        "mesh_ok_summary": passed,
        "exit_code": exit_code,
    }


__all__ = ["parse_checkmesh_output"]
