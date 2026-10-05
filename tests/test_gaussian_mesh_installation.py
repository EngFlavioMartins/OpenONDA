"""Public optional settings and installer commands; no optional GPU execution."""

import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

import install

ROOT = Path(__file__).resolve().parents[1]


def test_optional_extra_is_separate_and_fenv_is_installation_time():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert not any("cupy" in dependency for dependency in project["dependencies"])
    extra = project["optional-dependencies"]["gaussian-mesh-cuda12"]
    assert extra == ["cupy-cuda12x[ctk]>=14.2,<15; platform_system=='Linux'"]
    setup = (ROOT / "setup.py").read_text()
    assert '"source.solvers.vpm.numerics._fenv"' in setup and "optional=True" in setup


def test_installer_optional_extra_and_explicit_gpu_verification(monkeypatch):
    calls = []
    monkeypatch.setattr(
        install.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    assert install.main(["--gaussian-mesh-cuda12"]) == 0
    expected = f"{ROOT}[dev,gaussian-mesh-cuda12]"
    assert calls[0][4:] == ["-e", expected]
    assert calls[-1][-1] == "--with-gaussian-mesh"
    assert "--require-site-packages" not in calls[-1]


def test_public_settings_imports_leave_cupy_and_compiled_guard_optional(tmp_path):
    script = """
import sys
class RejectOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'cupy_backends'} or fullname.endswith('._fenv'):
            raise AssertionError('optional runtime imported: '+fullname)
sys.meta_path.insert(0, RejectOptional())
from openonda import vpm
from source.solvers.vpm.physics.induction import GaussianSlabSettings, GaussianMeshParameters
assert vpm.GaussianSlabSettings is GaussianSlabSettings
assert vpm.GaussianMeshParameters is GaussianMeshParameters
backend = vpm.SlipSlabInduction(vpm.FMMInduction(), z_min=-.5, z_max=.5,
                              gaussian_mesh_settings=vpm.GaussianSlabSettings())
assert backend.gaussian_mesh_settings.mesh == vpm.GaussianMeshParameters()
assert vpm.FMMInduction().kernel.name == 'GAUSSIAN'
"""
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("enabled", [False, True])
def test_verifier_gpu_execution_is_explicit_opt_in(monkeypatch, capsys, enabled):
    import openonda.verify_install as verifier

    monkeypatch.setattr(sys, "argv", ["verify"] + (["--with-gaussian-mesh"] if enabled else []))
    monkeypatch.setattr(verifier.numba, "get_num_threads", lambda: 1)
    monkeypatch.setattr(verifier.numba.config, "reload_config", lambda: None)
    monkeypatch.setattr(verifier, "_verify_package_location", lambda flag: "/installed/openonda")
    for name in (
        "_verify_distribution_resources",
        "_verify_tutorial_commands",
        "_verify_cartesian_mesher",
        "_verify_native_fvm",
        "_verify_native_vpm",
        "_verify_native_coupled",
    ):
        monkeypatch.setattr(verifier, name, lambda: {})
    monkeypatch.setattr(verifier, "_verify_taichi", lambda: ("qualified", "CPU"))
    calls = []
    monkeypatch.setattr(
        verifier, "_verify_gaussian_mesh", lambda: calls.append("gpu") or {"smoke": True}
    )
    assert verifier.main() == 0
    assert calls == (["gpu"] if enabled else [])
    assert ('"gaussian_mesh"' in capsys.readouterr().out) is enabled
