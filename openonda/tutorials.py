"""Discover, copy, and execute OpenONDA's installed tutorial templates.

Tutorials are immutable resources inside the installed distribution.  Before
execution they are copied to a normal user-owned workspace, so solver output
never modifies ``site-packages`` and the resulting case remains inspectable,
editable, and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Final, Literal

TutorialAction = Literal["run", "plot", "clean"]


@dataclass(frozen=True, slots=True)
class Tutorial:
    """Metadata for one installed runnable tutorial."""

    name: str
    description: str

    @property
    def relative_path(self) -> Path:
        """Return the case path below the packaged ``tutorials`` directory."""
        return Path(*self.name.split("/"))

    @property
    def slug(self) -> str:
        """Return a filesystem-friendly case name."""
        return self.name.replace("/", "-").replace("_", "-")


TUTORIALS: Final[tuple[Tutorial, ...]] = (
    Tutorial("fvm/airfoil_flow", "Laminar flow around an airfoil"),
    Tutorial("fvm/boundary_layer", "Laminar flat-plate boundary layer"),
    Tutorial("fvm/cube_flow", "Square-cylinder wake"),
    Tutorial("fvm/cylinder_ibm", "Immersed-boundary cylinder flow"),
    Tutorial("fvm/step_profile", "Backward-facing-step profile"),
    Tutorial("fvm/taylor_green", "Taylor-Green vortex decay"),
    Tutorial("vpm/delta_wing", "Delta-wing VLM-VPM flow"),
    Tutorial("vpm/flat_plate", "Finite flat-plate VLM-VPM loading"),
    Tutorial("vpm/lamb_oseen_vortex", "Lamb-Oseen vortex diffusion and interaction"),
    Tutorial("vpm/quadcopter", "Four-rotor VLM-VPM flow"),
    Tutorial("vpm/rotor_flow", "Single-rotor VLM-VPM flow"),
    Tutorial("vpm/vortex_interactions", "Vortex-ring interaction stabilization"),
    Tutorial("vpm/vortex_ring", "Viscous vortex-ring propagation"),
    Tutorial("coupled_fvm_vpm/cube_flow", "Coupled FVM-VPM cube flow"),
    Tutorial(
        "coupled_fvm_vpm/cube_flow/reference_flow",
        "Fully resolved reference for the coupled cube flow",
    ),
    Tutorial(
        "coupled_fvm_vpm/cylinder_shedding_flow",
        "Coupled FVM-VPM cylinder shedding",
    ),
    Tutorial(
        "coupled_fvm_vpm/cylinder_shedding_flow/reference_flow",
        "FVM grid study for cylinder shedding",
    ),
    Tutorial("coupled_fvm_vpm/naca4412_flow", "Coupled NACA 4412 flow"),
)

_BY_NAME: Final = {tutorial.name: tutorial for tutorial in TUTORIALS}
_EXCLUDED_PARTS: Final = {
    "solution",
    "samples",
    "figures",
    "__pycache__",
    ".matplotlib",
    "animation",
}
_EXCLUDED_NAMES: Final = {
    ".DS_Store",
    "paraview_state.py",
    "paraview_tracer.py",
    "run_manifest.json",
}
_ALLOWED_SUFFIXES: Final = {".py", ".sh", ".md", ".json", ".csv", ".stl", ".vsp3"}


def tutorial_names() -> tuple[str, ...]:
    """Return every installed tutorial identifier in display order."""
    return tuple(tutorial.name for tutorial in TUTORIALS)


def get_tutorial(name: str) -> Tutorial:
    """Resolve a tutorial name, accepting hyphens in place of underscores.

    Raises:
        ValueError: If no installed tutorial matches ``name``.
    """
    normalized = name.strip().strip("/")
    if normalized in _BY_NAME:
        return _BY_NAME[normalized]
    matches = [item for item in TUTORIALS if item.name.replace("_", "-") == normalized]
    if len(matches) == 1:
        return matches[0]
    choices = ", ".join(tutorial_names())
    raise ValueError(f"Unknown tutorial {name!r}. Available tutorials: {choices}")


def default_workspace(tutorial: Tutorial | str, directory: Path | None = None) -> Path:
    """Return the default workspace path for a tutorial."""
    item = get_tutorial(tutorial) if isinstance(tutorial, str) else tutorial
    parent = Path.cwd() if directory is None else Path(directory)
    return (parent / f"openonda-{item.slug}").resolve()


def tutorial_case_path(workspace: Path, tutorial: Tutorial | str) -> Path:
    """Return a materialized case directory below ``workspace``."""
    item = get_tutorial(tutorial) if isinstance(tutorial, str) else tutorial
    return Path(workspace).expanduser().resolve() / "tutorials" / item.relative_path


def _include_resource(relative: PurePosixPath) -> bool:
    if any(part in _EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.name in _EXCLUDED_NAMES or relative.name.startswith("."):
        return False
    return relative.suffix.lower() in _ALLOWED_SUFFIXES


def _copy_tree(source, destination: Path, relative: PurePosixPath = PurePosixPath()) -> int:
    copied = 0
    for child in source.iterdir():
        child_relative = relative / child.name
        if any(part in _EXCLUDED_PARTS for part in child_relative.parts):
            continue
        if child.is_dir():
            copied += _copy_tree(child, destination, child_relative)
        elif _include_resource(child_relative):
            output = destination.joinpath(*child_relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with child.open("rb") as input_stream, output.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
            if output.suffix == ".sh":
                output.chmod(output.stat().st_mode | 0o100)
            copied += 1
    return copied


def _copy_file_if_missing(source, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)


def _installed_root() -> Path:
    """Return the directory containing the unpacked distribution packages."""
    package = resources.files("openonda")
    if not isinstance(package, Path):
        raise RuntimeError("OpenONDA tutorials require a normal unpacked pip or Conda installation")
    return package.resolve().parent


def materialize_tutorial(name: str, workspace: Path) -> Path:
    """Copy one installed tutorial into a user-owned workspace.

    The destination may already contain other materialized tutorials, but an
    existing case is never overwritten.  The returned path is the directory
    containing ``setup.py`` and the ``all*.sh`` launchers.

    Raises:
        FileExistsError: If this tutorial already exists in the workspace.
        RuntimeError: If the installed distribution lacks a required resource.
    """
    tutorial = get_tutorial(name)
    workspace = Path(workspace).expanduser().resolve()
    case_path = tutorial_case_path(workspace, tutorial)
    if case_path.exists():
        raise FileExistsError(
            f"Tutorial already exists at {case_path}. Choose another workspace or use it in place."
        )

    installed_root = _installed_root()
    tutorial_root = installed_root / "tutorials"
    source = tutorial_root / tutorial.relative_path
    if not source.is_dir():
        raise RuntimeError(f"Installed tutorial resources are incomplete: {tutorial.name}")

    try:
        copied = _copy_tree(source, case_path)
        if copied == 0 or not (case_path / "setup.py").is_file():
            raise RuntimeError(f"Installed tutorial has no runnable source: {tutorial.name}")

        # Preserve the package topology used by module-based launchers.
        _copy_file_if_missing(tutorial_root / "__init__.py", workspace / "tutorials/__init__.py")
        category = tutorial.relative_path.parts[0]
        category_init = tutorial_root / category / "__init__.py"
        if category_init.is_file():
            _copy_file_if_missing(
                category_init,
                workspace / "tutorials" / category / "__init__.py",
            )

        # Plot scripts resolve this stable workspace-relative location.  Both
        # resources are part of the wheel, not paths into a source checkout.
        theme_root = installed_root / "docs/themes"
        for file_name in ("matplotlib_setup.py", "DejaVuSerif.ttf"):
            theme_file = theme_root / file_name
            if not theme_file.is_file():
                raise RuntimeError(f"Installed plotting resource is missing: {file_name}")
            _copy_file_if_missing(theme_file, workspace / "docs/themes" / file_name)

        readme = workspace / "README.md"
        if not readme.exists():
            readme.write_text(
                "# OpenONDA tutorial workspace\n\n"
                "This workspace was created by the installed `openonda` command.\n"
                "Generated solver data stays here; the installed package remains immutable.\n",
                encoding="utf-8",
            )
    except Exception:
        if case_path.exists():
            shutil.rmtree(case_path)
        raise
    return case_path


def ensure_tutorial(name: str, workspace: Path) -> tuple[Path, bool]:
    """Return a case path, materializing it when it is not present."""
    case_path = tutorial_case_path(workspace, name)
    if case_path.is_dir():
        return case_path, False
    return materialize_tutorial(name, workspace), True


def execute_tutorial(name: str, workspace: Path, action: TutorialAction = "run") -> int:
    """Execute a tutorial launcher with the current Python environment.

    ``run`` creates the workspace on first use.  ``plot`` and ``clean`` require
    an existing workspace so accidental plotting of an empty case is avoided.
    The process return code is returned unchanged.
    """
    tutorial = get_tutorial(name)
    workspace = Path(workspace).expanduser().resolve()
    case_path = tutorial_case_path(workspace, tutorial)
    if action == "run":
        case_path, _ = ensure_tutorial(tutorial.name, workspace)
    elif not case_path.is_dir():
        raise FileNotFoundError(
            f"No materialized tutorial at {case_path}. Run `openonda tutorial create "
            f"{tutorial.name} {workspace}` first."
        )

    launcher = {"run": "allrun.sh", "plot": "allplot.sh", "clean": "allclean.sh"}[action]
    script = case_path / launcher
    if not script.is_file():
        raise RuntimeError(f"Tutorial launcher is missing: {script}")

    environment = os.environ.copy()
    environment["OPENONDA_PYTHON"] = sys.executable
    environment["MPLCONFIGDIR"] = str(workspace / ".matplotlib")
    environment.setdefault("XDG_CACHE_HOME", str(workspace / ".cache"))
    environment.setdefault(
        "TI_OFFLINE_CACHE_FILE_PATH",
        str(workspace / ".cache/taichi"),
    )
    # Do not resolve a virtual-environment Python symlink: its directory is the
    # environment whose console command launched us and must remain first.
    executable_directory = str(Path(sys.executable).parent)
    environment["PATH"] = executable_directory + os.pathsep + environment.get("PATH", "")
    return subprocess.run(
        ["bash", str(script)],
        cwd=case_path,
        env=environment,
        check=False,
    ).returncode


__all__ = [
    "TUTORIALS",
    "Tutorial",
    "TutorialAction",
    "default_workspace",
    "ensure_tutorial",
    "execute_tutorial",
    "get_tutorial",
    "materialize_tutorial",
    "tutorial_case_path",
    "tutorial_names",
]
