"""Offline post-processing for archived FVM states.

:class:`PostProcess` replays archived solver snapshots (``solution/<case>.pvd``)
through the *same* sampler objects and the *same* executor used by a live run.
It never instantiates the transient solver, never evolves the case, and never
overwrites the archive — each archived snapshot becomes a read-only
:class:`SnapshotContext` that the :class:`~.executor.FVMSamplerExecutor` drives
exactly like a live solver.

The archive stores cell-centred fields only (boundary ghost rows are stripped
by the VTK exporter).  ``PostProcess`` therefore rebuilds the boundary/ghost
values from the original mesh and boundary conditions, recomputes the velocity
gradient with the same FVM gradient implementation, and uses the archived
``nut`` when the force calculation needs it — then calls the same
``ForceSampler.sample()`` used online.  If a required quantity is genuinely
absent, ``PostProcess`` fails with a useful error instead of manufacturing
values.

Examples
--------
>>> post = PostProcess(
...     case_dir=case_dir,
...     config=setup,
...     samplers=(ForceSampler(patch_names=["cube"]),),
...     mesh=FVM_MESH,
... )
>>> post.run()   # writes case_dir/samples/forces_history.csv
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .executor import FVMSamplerExecutor


class _NullLogger:
    """Logger stand-in so offline contexts satisfy the executor interface."""

    def yplus_info(self, yplus_stats):  # noqa: D401
        pass

    def ibm_force_info(self, _coefficients, slip):
        pass

    def force_info(self, forces):
        pass


class SnapshotContext:
    """Read-only sampling context exposing one archived FVM state.

    Presents the same sampling interface a live solver does (``mesh_data``,
    ``geo_data``, ``boundaries``, ``config``, ``U``, ``p``, ``nut``,
    ``parallel``, ``flow_time``, ``time_step``, ``_current_dt``, plus the
    derived ``_velocity_gradient()``/``_vorticity_field()``), so the samplers
    and executor cannot tell online and offline apart.
    """

    def __init__(
        self,
        config,
        case_dir: str,
        mesh_data: dict,
        geo_data: dict,
        boundaries: list,
        U: np.ndarray,
        p: np.ndarray,
        nut: np.ndarray | None,
        flow_time: float,
        time_step: int,
        dt: float,
    ):
        from ..core.parallel import ParallelContext

        self.config = config
        self.case_dir = case_dir
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.boundaries = boundaries
        self.U = U
        self.p = p
        self.nut = nut
        self.flow_time = flow_time
        self.time_step = time_step
        self._current_dt = dt
        self.parallel = ParallelContext()
        self.logger = _NullLogger()
        self.last_forces = None
        self.last_yplus = None
        self.ibm = None
        self._derived_fields: dict[object, np.ndarray] = {}

    def _velocity_gradient(self) -> np.ndarray:
        from ..fields import gradients

        gradient = self._derived_fields.get("velocity_gradient")
        if gradient is None:
            gradient = gradients._resolve_gradient_fn(self.geo_data)(
                self.U, self.mesh_data, self.geo_data
            )
            self._derived_fields["velocity_gradient"] = gradient
        return gradient

    def _vorticity_field(self) -> np.ndarray:
        from ..fields import diagnostics

        vorticity = self._derived_fields.get("vorticity")
        if vorticity is None:
            vorticity = diagnostics.compute_vorticity(
                self.U,
                self.mesh_data,
                self.geo_data,
                gradient=self._velocity_gradient(),
            )
            self._derived_fields["vorticity"] = vorticity
        return vorticity


def _materialize_mesh(mesh) -> dict:
    """Turn a mesher callable / dict / path into FVM ``mesh_data``."""
    if mesh is None:
        raise TypeError("PostProcess requires the mesh that produced the archive")
    if isinstance(mesh, str | Path):
        from ..factory import _load_mesh_file

        return _load_mesh_file(str(mesh))
    if callable(mesh):
        generated = mesh()
        if isinstance(generated, tuple):
            generated = generated[0]
        if not isinstance(generated, dict):
            raise TypeError("mesh callable must return a mesh_data dict")
        return generated
    if not isinstance(mesh, dict):
        raise TypeError("mesh must be a path, a mesh dict, or a callable returning one")
    return mesh


class PostProcess:
    """Replay archived FVM snapshots through the configured samplers."""

    def __init__(
        self,
        case_dir,
        config,
        samplers=None,
        mesh=None,
        overwrite: bool = True,
    ):
        from dataclasses import replace

        self.case_dir = str(Path(case_dir).resolve())
        samplers = tuple(samplers) if samplers is not None else tuple(config.samplers or ())
        self.config = replace(config, samplers=samplers)
        self.mesh_data = _materialize_mesh(mesh)
        self.boundaries = self._setup_boundaries(self.mesh_data)
        self.geo_data = self._build_geometry(self.mesh_data)
        self.overwrite = bool(overwrite)

    def _build_geometry(self, mesh_data: dict) -> dict:
        from ..fields.gradients import compute_lsq_geometry
        from ..mesh import geometry
        from ..mesh.coupled import configure_cyclic_boundaries

        gs = self.config.schemes.gradient_scheme
        geo = geometry.compute_mesh_geometry(
            mesh_data, gradient_scheme=gs, compute_lsq=False, logger=None
        )
        configure_cyclic_boundaries(mesh_data, geo)
        if gs == "lsq":
            geo.update(compute_lsq_geometry(mesh_data, geo))  # type: ignore
        return geo

    def _setup_boundaries(self, mesh_data: dict) -> list:
        boundaries = mesh_data["boundary"]
        for b_cfg in self.config.boundaries:
            for b_mesh in boundaries:
                if b_mesh["name"] == b_cfg.name:
                    velocity = np.asarray(b_cfg.value_velocity, dtype=np.float64)
                    b_mesh.update(
                        {
                            "bc_type_velocity": b_cfg.type_velocity,
                            "bc_type_p": b_cfg.type_p,
                            "value_p": b_cfg.value_p,
                            "bc_type_nut": b_cfg.type_nut,
                            "value_nut": b_cfg.value_nut,
                        }
                    )
                    if b_cfg.mesh_type is not None:
                        b_mesh["type"] = b_cfg.mesh_type
                    else:
                        b_mesh.setdefault("type", "patch")
                    if b_cfg.neighbour_patch is not None:
                        b_mesh["neighbourPatch"] = b_cfg.neighbour_patch
                    if velocity.shape == (3,):
                        b_mesh["value_velocity"] = velocity
                        b_mesh.pop("value_velocity_field", None)
                    else:
                        b_mesh["value_velocity_field"] = velocity
                    break
        return boundaries

    def _pvd_frames(self) -> list[tuple[float, int, Path]]:
        solution_dir = Path(self.case_dir) / "solution"
        pvd_path = solution_dir / f"{self.config.case_name}.pvd"
        if not pvd_path.exists():
            candidates = sorted(solution_dir.glob("*.pvd"))
            if not candidates:
                raise FileNotFoundError(
                    f"No PVD index in {solution_dir}; PostProcess needs archived snapshots"
                )
            pvd_path = candidates[0]
        text = pvd_path.read_text(encoding="utf-8")
        matches = re.finditer(r'timestep="([^"]+)"[^>]*file="([^"]+)"', text)
        frames = []
        for match in matches:
            path = pvd_path.parent / match.group(2)
            step_match = re.search(r"_(\d+)\.(?:pvtu|vtu)$", path.name)
            if step_match is None:
                raise ValueError(f"Cannot recover time step from snapshot name {path.name!r}")
            frames.append((float(match.group(1)), int(step_match.group(1)), path))
        if not frames:
            raise ValueError(f"No snapshots listed in {pvd_path}")
        return sorted(frames)

    def _read_snapshot(self, path: Path, n_cells: int) -> dict[str, np.ndarray | None]:
        """Return interior cell fields for one archived snapshot.

        Partitioned ``.pvtu`` pieces are scattered back to global cell order
        using ``GlobalCellIds`` (owned cells only); serial ``.vtu`` files are
        used directly.
        """
        import pyvista as pv

        grid = pv.read(str(path))
        cell_data = grid.cell_data
        if "U" not in cell_data or "p" not in cell_data:
            raise ValueError(f"Archived snapshot {path} lacks U/p cell data")

        def read_array(name: str):
            if name not in cell_data:
                return None
            return np.asarray(cell_data[name], dtype=np.float64)

        ghost = read_array("vtkGhostType")
        global_ids = read_array("GlobalCellIds")
        local_count = int(np.asarray(cell_data["U"]).shape[0])
        keep = np.ones(local_count, dtype=bool) if ghost is None else np.asarray(ghost) == 0
        order = None
        if global_ids is not None:
            order = np.asarray(global_ids, dtype=np.int64)[keep]
        kept = np.count_nonzero(keep)

        def scatter(values: np.ndarray | None):
            if values is None:
                return None
            values = np.asarray(values, dtype=np.float64)[keep]
            if order is None:
                return values
            result = np.full((n_cells, *values.shape[1:]), np.nan, dtype=np.float64)
            result[order] = values
            if np.any(~np.isfinite(result)):
                raise ValueError(
                    f"Archived snapshot {path} does not cover every global cell; "
                    "the post-processed mesh must match the archived one"
                )
            return result

        fields = {
            "U": scatter(np.asarray(cell_data["U"], dtype=np.float64)),
            "p": scatter(np.asarray(cell_data["p"], dtype=np.float64)),
            "nut": scatter(read_array("nut")),
        }
        if kept != n_cells:
            raise ValueError(
                f"Archived snapshot {path} has {kept} owned cells but the "
                f"post-processed mesh has {n_cells}; the mesh must match the archive"
            )
        return fields

    def _reconstruct_state(self, fields: dict) -> tuple[np.ndarray, np.ndarray]:
        """Assemble full U/p arrays (interior + reconstructed boundary ghosts)."""
        from ..assemble import convection
        from ..solve import simple_solver

        mesh_data = self.mesh_data
        n_elements = mesh_data["n_elements"]
        n_total = mesh_data["n_faces"] - mesh_data["n_interior_faces"] + n_elements

        U = np.zeros((n_total, 3), dtype=np.float64)
        p = np.zeros(n_total, dtype=np.float64)
        U[:n_elements] = fields["U"]
        p[:n_elements] = fields["p"]

        phi = convection.compute_volumetric_face_flux(U, mesh_data, self.geo_data)
        simple_solver._update_velocity_bcs(
            U,
            phi,
            self.boundaries,
            mesh_data["owners"],
            self.geo_data,
            n_elements,
            mesh_data["n_interior_faces"],
            mesh_data=mesh_data,
        )
        simple_solver.update_scalar_boundaries(p, mesh_data, self.boundaries, "p", face_flux=phi)
        return U, p

    def _archived_timesteps(self) -> dict[int, float]:
        """Return accepted ``dt`` values keyed by archived solver step."""
        diagnostics = Path(self.case_dir) / "solution" / "diagnostics.jsonl"
        if not diagnostics.exists():
            return {}
        values: dict[int, float] = {}
        with diagnostics.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                    step = int(record["step"])
                    dt = float(record["dt"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid diagnostics record at {diagnostics}:{line_number}"
                    ) from exc
                if dt <= 0.0:
                    raise ValueError(f"Invalid non-positive dt at {diagnostics}:{line_number}")
                values[step] = dt
        return values

    def run(self) -> list[tuple[float, int]]:
        """Replay every archived snapshot through the configured samplers.

        Output is *fresh*: the previous run's sampler products under
        ``samples/`` are cleared first, so re-running ``PostProcess`` never
        appends duplicate rows into an existing CSV or PVD.  The snapshot
        ``dt`` passed to the samplers is the *archived* inter-frame advance
        (not ``config.time.delta_t``), so adaptive-dt cases resample offline
        with the same cadence they selected online.
        """
        if self.overwrite:
            self._clear_previous_output()
        frames = self._pvd_frames()
        archived_dt = self._archived_timesteps()
        sampled: list[tuple[float, int]] = []
        n_elements = self.mesh_data["n_elements"]
        default_dt = float(self.config.time.delta_t)
        for index, (flow_time, step, path) in enumerate(frames):
            # Recover the real archived advance: the time between this archived
            # snapshot and the previously archived one (falls back to the
            # nominal dt for the first frame).
            if step in archived_dt:
                dt = archived_dt[step]
            elif index == 0:
                dt = default_dt
            else:
                dt = float(frames[index][0] - frames[index - 1][0])
            fields = self._read_snapshot(path, n_elements)
            U, p = self._reconstruct_state(fields)
            context = SnapshotContext(
                config=self.config,
                case_dir=self.case_dir,
                mesh_data=self.mesh_data,
                geo_data=self.geo_data,
                boundaries=self.boundaries,
                U=U,
                p=p,
                nut=fields["nut"],
                flow_time=flow_time,
                time_step=step,
                dt=dt,
            )
            FVMSamplerExecutor.execute(context, strict=True)
            sampled.append((flow_time, step))
        return sampled

    def _clear_previous_output(self) -> None:
        """Remove prior sampler output so replay is idempotent."""
        from .base import samples_dir

        samples = Path(samples_dir(self.case_dir))
        if samples.exists():
            for child in samples.iterdir():
                if child.is_file():
                    child.unlink()
