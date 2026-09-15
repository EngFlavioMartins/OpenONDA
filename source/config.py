"""Read-only build information for the command-line interface."""

import platform

from source import log_style
from source.version import __version__


def show():
    """Print build identity using the same layout as solver initialization."""
    print(
        log_style.block_report(
            "OpenONDA build",
            [
                (
                    "run",
                    [
                        ("Python", platform.python_version()),
                        ("OpenONDA", __version__),
                    ],
                )
            ],
        )
    )
