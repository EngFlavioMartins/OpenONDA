from types import SimpleNamespace

import numpy as np

from source.solvers.VPM.io.logging import Logging


def test_flow_diagnostics_reports_particle_count(capsys):
    system = SimpleNamespace(
        time_step=90,
        flow_time=3.6,
        particles=SimpleNamespace(number_of_particles=1234),
        total_strength_magnitude=1.0,
        total_strength=np.array([1.0, 2.0, 3.0]),
        total_linear_impulse=np.zeros(3),
        total_angular_impulse=np.zeros(3),
        total_enstrophy=4.0,
        total_helicity=5.0,
        total_kinetic_energy=6.0,
        vorticity_dissipation_rate=7.0,
        kinetic_energy_dissipation_rate=-8.0,
        centroid_of_circulation=None,
        centroids_of_circulation={},
        _diagnostics_history={},
        vlm_solver=None,
    )

    Logging.flow_diagnostics(system)

    out = capsys.readouterr().out
    assert "FLOW DIAGNOSTICS" in out
    assert "Number of Particles" in out
    assert "1234" in out
