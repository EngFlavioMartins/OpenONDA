"""Installed-package safety checks for reproducibility manifests."""

from source.solvers.FVM.io.manifest import _git_identity


def test_git_identity_is_optional_outside_a_checkout(tmp_path):
    assert _git_identity(tmp_path) == (None, None)
