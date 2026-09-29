"""Run the real cube tutorial at bounded resolution, including failed startup."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys

from defusedxml.ElementTree import parse
import h5py
import numpy as np


def main(directory, cores):
    root = Path(__file__).resolve().parents[2]
    case = runpy.run_path(str(root / "tutorials/coupled_fvm_vpm/02_cube_flow/setup.py"))
    flow = replace(
        case["FVM_SETUP"],
        cores=cores,
        time=replace(case["FVM_SETUP"].time, end_time=0.1),
    )
    numerics = case["VPM_CASE"].numerics
    particles = replace(
        case["VPM_CASE"],
        numerics=replace(
            numerics,
            compute_device="CPU",
            max_n_particles=100_000,
            max_evaluation_points=100_000,
            viscous=replace(
                numerics.viscous,
                particle_spacing=0.25,
                gbd_grid_spacing=0.25,
                gbd_max_nodes=100_000,
                gbd_threshold=case["GBD_VORTICITY_FLOOR"] * 0.25**3,
            ),
        ),
    )
    mesh = case["msh"].CachedMesh(
        case["msh"].CartesianMesher(
            domain=case["FVM_MESH"].requested_domain,
            surfaces=case["FVM_MESH"].surfaces,
            max_cell_size=0.25,
            cell_size_anchor=0.25,
        ),
        directory / "constant/mesh.npz",
    )

    def build(name):
        return case["coupling"].create_coupler(
            flow,
            particles,
            case["COUPLER_SETUP"],
            mesh=mesh,
            case_dir=directory / name,
        )

    # Interrupt exactly where the reported exception left an orphan FVM save.
    with build("continued") as failed:

        def fail_backup(_path):
            raise RuntimeError("interrupted initial VPM backup")

        def interrupt(vpm):
            vpm._save_backup_to = fail_backup

        failed.apply_vpm(interrupt)
        try:
            failed.run(start_from="latest", max_coupling_steps=1)
        except RuntimeError as error:
            assert "interrupted initial VPM backup" in str(error)
        else:
            raise AssertionError("Initial backup should have failed")
    manifest = directory / "continued/solution/backups/manifest.json"
    assert not manifest.exists()

    with build("continued") as first:
        assert first.run(start_from="latest", max_coupling_steps=1) == 1
        if first._is_master:
            assert first.vorticity_transfer._body_bounds is not None
            assert not first.vorticity_transfer._solid_bodies
            assert first.vpm_solver.particles.n_particles_total > 0
    assert json.loads(manifest.read_text())["coupling_step"] == 1
    log = (directory / "continued/solution/coupler.log").read_bytes()

    with build("continued") as resumed:
        assert resumed.run(start_from="latest") == 2
        actual_velocity = resumed.fvm_solver.get_velocity_field().copy()
        actual_pressure = resumed.fvm_solver.get_pressure_field().copy()
    assert json.loads(manifest.read_text())["coupling_step"] == 2
    assert (directory / "continued/solution/coupler.log").read_bytes().startswith(log)
    history = directory / "continued/solution/coupler_diagnostics.jsonl"
    assert [json.loads(row)["step"] for row in history.read_text().splitlines()] == [1, 2]
    before = history.read_bytes()
    with build("continued") as completed:
        assert completed.run(start_from="latest") == 2
    assert history.read_bytes() == before

    with build("reference") as reference:
        assert reference.run(start_from="latest") == 2
        np.testing.assert_allclose(
            actual_velocity,
            reference.fvm_solver.get_velocity_field(),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            actual_pressure,
            reference.fvm_solver.get_pressure_field(),
            rtol=1e-6,
            atol=flow.linear.pressure_tolerance,
        )
    with (
        h5py.File(directory / "continued/solution/vpm/vpm_000002.h5") as actual,
        h5py.File(directory / "reference/solution/vpm/vpm_000002.h5") as expected,
    ):
        for field in ("position", "vortex_strength", "core_radius", "particle_volume"):
            np.testing.assert_allclose(
                actual[f"particles/{field}"][:],
                expected[f"particles/{field}"][:],
                rtol=1e-6,
                atol=1e-8,
            )
    samples = directory / "reference/samples"
    for sampler in (*case["FVM_SAMPLERS"], *case["VPM_SAMPLERS"]):
        assert any(
            (samples / f"{sampler.file_name}.{suffix}").is_file() for suffix in ("csv", "pvd")
        )
    for path in samples.glob("*.csv"):
        with (
            path.open() as expected,
            (directory / "continued/samples" / path.name).open() as actual,
        ):
            assert [row["time"] for row in csv.DictReader(actual)] == [
                row["time"] for row in csv.DictReader(expected)
            ], path.name
    for path in samples.glob("*.pvd"):
        actual = parse(directory / "continued/samples" / path.name).findall(".//DataSet")
        expected = parse(path).findall(".//DataSet")
        assert [row.attrib["timestep"] for row in actual] == [
            row.attrib["timestep"] for row in expected
        ], path.name

    # Start a new branch over a completed coupled case. The old native VPM
    # frame must be retired so later native discovery cannot select it.
    stale_frame = directory / "continued/solution/vpm/vpm_999999.h5"
    stale_frame.parent.mkdir(parents=True, exist_ok=True)
    stale_frame.write_bytes(b"stale VPM backup")
    with build("continued") as restarted:
        assert restarted.run(start_from="initial", max_coupling_steps=1) == 1
        assert restarted.fvm_solver.step == restarted.n_fvm_substeps
    assert not stale_frame.exists()
    assert json.loads(manifest.read_text())["coupling_step"] == 1
    assert [json.loads(row)["step"] for row in history.read_text().splitlines()] == [1]
    with build("continued") as renewed:
        assert renewed.run(start_from="latest") == 2
        np.testing.assert_allclose(
            renewed.fvm_solver.get_velocity_field(), actual_velocity, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            renewed.fvm_solver.get_pressure_field(), actual_pressure,
            rtol=1e-6, atol=flow.linear.pressure_tolerance,
        )
    assert [json.loads(row)["step"] for row in history.read_text().splitlines()] == [1, 2]


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
