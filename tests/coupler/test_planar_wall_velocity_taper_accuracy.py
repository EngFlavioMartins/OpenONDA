"""Independent no-slip identities and consistency order for wall-velocity taper."""

import numpy as np

from tests.support.cylinder.audit_saved_wall_circulation import gaussian_velocity_and_gradient
from tests.support.cylinder.audit_wall_velocity_taper import audit, no_slip_velocity


def test_manufactured_circular_flow_is_no_slip_and_divergence_free():
    angle = np.arange(24) * 2 * np.pi / 24
    wall = np.column_stack((0.5 * np.cos(angle), 0.5 * np.sin(angle), np.zeros(24)))
    np.testing.assert_allclose(no_slip_velocity(wall), 0, atol=2e-14)
    points = wall * 1.3
    step = 1e-5
    divergence = np.zeros(len(points))
    for axis in (0, 1):
        offset = np.zeros(3)
        offset[axis] = step
        divergence += (
            no_slip_velocity(points + offset)[:, axis] - no_slip_velocity(points - offset)[:, axis]
        ) / (2 * step)
    np.testing.assert_allclose(divergence, 0, atol=1e-7)


def test_wall_taper_changes_outer_velocity_but_converges_with_spacing():
    result = audit()
    rows = [row["complete_cell_circulation"] for row in result["rows"]]
    assert all(abs(row["circulation_change"]) < 1e-13 for row in rows)
    error = [row["maximum_outer_velocity_change"] for row in rows]
    assert all(left > right > 0 for left, right in zip(error[:-1], error[1:], strict=True))
    # A no-slip velocity is O(distance) at the wall; a taper over O(h)
    # modifies its annular integral by O(h^2). This test measures that error
    # at fixed physical targets rather than comparing duplicate algorithms.
    assert 1.6 < rows[-1]["observed_order_from_previous_spacing"] < 2.4


def test_omitting_solid_centred_cells_loses_local_contour_circulation():
    report = audit(spacings=(0.04,))
    contour = report["rows"][0]["upper_rectangle_circulation"]
    for row in contour.values():
        np.testing.assert_allclose(
            row["complete_cell_circulation"], row["independent_contour_circulation"], atol=2e-14
        )
        assert abs(row["relative_lost_circulation_from_omitted_cut_cells"]) > 1e-3
    # The exterior taper partly shifts circulation out of discarded cut cells;
    # simply removing that taper while retaining the exclusion is worse here.
    assert abs(contour["untapered"]["relative_lost_circulation_from_omitted_cut_cells"]) > abs(
        contour["tapered"]["relative_lost_circulation_from_omitted_cut_cells"]
    )


def test_independent_gaussian_trace_has_exact_curl_and_finite_difference_gradient():
    sources = np.array([[0.03, 0.11, 0.0], [-0.1, 0.2, 0.0]])
    strength = np.array([0.6, -0.9])
    points = np.array([[0.03, 0.11, 0.0], [0.7, -0.6, 0.0], [-0.1, 0.2, 0.0]])
    spacing = 0.04
    _, gradient = gaussian_velocity_and_gradient(points, sources, strength, spacing)
    squared = np.sum((points[:, None, :2] - sources[None, :, :2]) ** 2, axis=2)
    omega = np.sum(strength * np.exp(-squared / spacing**2) / (np.pi * spacing**2), axis=1)
    np.testing.assert_allclose(gradient[:, 1, 0] - gradient[:, 0, 1], omega, atol=2e-13)
    np.testing.assert_allclose(np.trace(gradient, axis1=1, axis2=2), 0, atol=1e-14)
    for axis in (0, 1):
        offset = np.zeros(3)
        offset[axis] = 1e-6
        plus, _ = gaussian_velocity_and_gradient(points + offset, sources, strength, spacing)
        minus, _ = gaussian_velocity_and_gradient(points - offset, sources, strength, spacing)
        np.testing.assert_allclose((plus - minus) / 2e-6, gradient[:, :, axis], atol=1e-7)
