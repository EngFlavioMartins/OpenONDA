"""A rejected VPM setting names the value it got and the values it accepts."""

import pytest

from openonda import vpm
from openonda.core.restart import select_backup
from openonda.vpm.config.case import Numerics


def test_unknown_viscous_scheme_names_the_schemes():
    with pytest.raises(ValueError, match=r"'cs', 'rwm', 'none', 'dvh' or 'gbd' \(got 'GBD'\)"):
        vpm.ViscousConfig(scheme="GBD")


def test_unknown_remeshing_kernel_names_the_kernels():
    with pytest.raises(ValueError, match=r"'m4_prime' or 'lagrange6' \(got 'M4_PRIME'\)"):
        vpm.ViscousConfig.gbd(particle_spacing=0.1, remeshing_kernel="M4_PRIME")


def test_cutoff_radius_factor_other_than_the_kernel_value_is_refused():
    with pytest.raises(ValueError, match=r"fixed at 100 by the kernels \(got 50\)"):
        Numerics(cutoff_radius_factor=50)


def test_mistyped_start_from_is_not_taken_for_a_backup_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="'lastest' is not 'latest', 'initial'"):
        select_backup("lastest", directory=tmp_path, kind="vpm")


def test_slab_resource_limits_name_both_settings():
    with pytest.raises(ValueError, match=r"max_plan_bytes \(4096\) must be smaller"):
        vpm.GaussianSlabSettings(max_plan_bytes=4096, max_scratch_bytes=4096)
