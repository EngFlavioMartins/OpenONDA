"""docs/rename-manifest.md: no compatibility aliases; obsolete public names
must fail rather than silently resolve to the renamed target."""

import pytest


@pytest.mark.parametrize(
    "module,name",
    [
        ("openonda.fvm", "Solver"),
        ("openonda.vpm", "Solver"),
        ("openonda.fvm", "setup_fvm_solver"),
        ("openonda.vpm", "setup_vpm_solver"),
        ("openonda.coupler", "setup_coupler"),
    ],
)
def test_pre_rename_public_name_is_gone(module, name):
    mod = __import__(module, fromlist=[name])
    assert not hasattr(mod, name)
