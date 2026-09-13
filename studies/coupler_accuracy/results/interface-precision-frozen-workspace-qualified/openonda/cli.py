"""Command-line interface for an installed OpenONDA distribution."""

from __future__ import annotations

import argparse
from pathlib import Path
import pydoc
import sys

import openonda
from openonda.tutorials import (
    TUTORIALS,
    default_workspace,
    execute_tutorial,
    get_tutorial,
    materialize_tutorial,
    tutorial_case_path,
)


def _workspace(value: str | None, tutorial_name: str) -> Path:
    return default_workspace(tutorial_name) if value is None else Path(value).expanduser().resolve()


def _print_tutorials() -> int:
    width = max(len(item.name) for item in TUTORIALS)
    for item in TUTORIALS:
        print(f"{item.name:<{width}}  {item.description}")
    return 0


def _show_api(name: str) -> int:
    target = "openonda" if not name else f"openonda.{name.removeprefix('openonda.')}"
    value = pydoc.locate(target)
    if value is None:
        raise ValueError(f"Could not find public API object {target!r}")
    print(pydoc.render_doc(value, title="OpenONDA help on %s", renderer=pydoc.plaintext))
    return 0


def _create(arguments: argparse.Namespace) -> int:
    tutorial = get_tutorial(arguments.name)
    workspace = _workspace(arguments.workspace, tutorial.name)
    case_path = materialize_tutorial(tutorial.name, workspace)
    print(f"Created {tutorial.name} at {case_path}")
    print(f"Run it with: openonda tutorial run {tutorial.name} --workspace {workspace}")
    return 0


def _execute(arguments: argparse.Namespace) -> int:
    tutorial = get_tutorial(arguments.name)
    workspace = _workspace(arguments.workspace, tutorial.name)
    case_path = tutorial_case_path(workspace, tutorial)
    if arguments.action == "run" and not case_path.exists():
        print(f"Creating tutorial workspace at {workspace}", flush=True)
    return execute_tutorial(tutorial.name, workspace, arguments.action)


def build_parser() -> argparse.ArgumentParser:
    """Build the public OpenONDA command-line parser."""
    parser = argparse.ArgumentParser(
        prog="openonda",
        description="Inspect OpenONDA and run installed tutorial cases from any directory.",
    )
    parser.add_argument("--version", action="version", version=f"OpenONDA {openonda.__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="show the installed version and package location")
    info.set_defaults(handler=lambda _arguments: _print_info())

    api = commands.add_parser("api", help="show Python docstrings for a public API object")
    api.add_argument("name", nargs="?", default="", help="for example: vpm.DirectInduction")
    api.set_defaults(handler=lambda arguments: _show_api(arguments.name))

    tutorial = commands.add_parser("tutorial", help="list, create, run, plot, or clean tutorials")
    tutorial_commands = tutorial.add_subparsers(dest="tutorial_command", required=True)
    listing = tutorial_commands.add_parser("list", help="list installed tutorial templates")
    listing.set_defaults(handler=lambda _arguments: _print_tutorials())

    create = tutorial_commands.add_parser("create", help="copy a tutorial to a user workspace")
    create.add_argument("name", help="tutorial identifier from `openonda tutorial list`")
    create.add_argument(
        "workspace",
        nargs="?",
        help="workspace root (default: ./openonda-<tutorial-name>)",
    )
    create.set_defaults(handler=_create)

    for action in ("run", "plot", "clean"):
        action_parser = tutorial_commands.add_parser(action, help=f"{action} a tutorial workspace")
        action_parser.add_argument("name", help="tutorial identifier")
        action_parser.add_argument(
            "--workspace",
            help="workspace root (default: ./openonda-<tutorial-name>)",
        )
        action_parser.set_defaults(handler=_execute, action=action)
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
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
