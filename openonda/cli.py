"""Command-line interface for an installed OpenONDA distribution."""

from __future__ import annotations

import argparse
from pathlib import Path
import pydoc
import sys

import openonda


def _show_api(name: str) -> int:
    target = "openonda" if not name else f"openonda.{name.removeprefix('openonda.')}"
    value = pydoc.locate(target)
    if value is None:
        raise ValueError(f"Could not find public API object {target!r}")
    print(pydoc.render_doc(value, title="OpenONDA help on %s", renderer=pydoc.plaintext))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the public OpenONDA command-line parser."""
    parser = argparse.ArgumentParser(
        prog="openonda",
        description="Inspect the installed OpenONDA package and its public API.",
    )
    parser.add_argument("--version", action="version", version=f"OpenONDA {openonda.__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="show the installed version and package location")
    info.set_defaults(handler=lambda _arguments: _print_info())

    api = commands.add_parser("api", help="show Python docstrings for a public API object")
    api.add_argument("name", nargs="?", default="", help="for example: vpm.DirectInduction")
    api.set_defaults(handler=lambda arguments: _show_api(arguments.name))

    return parser


def _print_info() -> int:
    print(f"OpenONDA {openonda.__version__}")
    print(Path(openonda.__file__).resolve())
    return 0


def main(arguments: list[str] | None = None) -> int:
    """Run the OpenONDA command line and return its process status."""
    parser = build_parser()
    try:
        parsed = parser.parse_args(arguments)
        return int(parsed.handler(parsed))
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
