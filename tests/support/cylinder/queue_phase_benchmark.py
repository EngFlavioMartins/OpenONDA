#!/usr/bin/env python3
"""Retired sequential phase queue; use the ordinary case launchers.

The original orchestration and its inputs remain preserved in native restart
evidence. This module deliberately has no resource polling, child execution,
cleanup or output-directory creation.
"""

import sys


def main() -> int:
    print(
        "This phase queue is retired and cannot be relaunched. "
        "Run ./allrun.sh or ./allcontinue.sh from the cylinder case directory; "
        "use --max-coupling-steps N for a bounded native continuation.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
