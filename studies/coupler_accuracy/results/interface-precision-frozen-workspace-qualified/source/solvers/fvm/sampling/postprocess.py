"""Offline post-processing for archived FVM states.

:class:`PostProcess` replays archived solver snapshots (``solutions/<case>.pvd``)
through the *same* sampler objects and the *same* executor used by a live run.
It never instantiates the transient solver, never evolves the case, and never
overwrites the archive — each archived snapshot becomes a read-only
:class:`SnapshotContext` that the :class:`~.executor.FVMSamplerExecutor` drives
exactly like a live solver.

The archive stores cell-centred fields only (boundary ghost rows are stripped
by the VTK exporter).  ``PostProcess`` therefore rebuilds the boundary/ghost
values from the original mesh and boundary conditions, recomputes the velocity
gradient with the same FVM gradient implementation, and uses the archived
``eddy_viscosity`` when the force calculation needs it — then calls the same
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

    def warning(self, _text, **_kwargs):
        pass


class SnapshotContext:
    """Read-only sampling context exposing one archived FVM state.

    Presents the same sampling interface a live solver does (``mesh_data``,
    ``geo_data``, ``boundaries``, ``setup``, ``velocity``,
    ``kinematic_pressure``, ``eddy_viscosity``,
    ``parallel``, ``time``, ``step``, ``_current_dt``, plus the
    derived ``_velocity_gradient()``/``_vorticity_field()``), so the samplers
    and executor cannot tell online and offline apart.
    """

    def __init__(
        self,
        setup,
        case_dir: str,
        mesh_data: dict,
        geo_data: dict,
        boundaries: list,
        velocity: np.ndarray,
        kinematic_pressure: np.ndarray,
        eddy_viscosity: np.ndarray | None,
        time: float,
        step: int,
        time_step_size: float,
        solution_dir: str | None = None,
        samples_dir: str | None = None,
    ):
        """Create a read-only-compatible context for one archived frame.

        Parameters
        ----------
        setup : FVMSetup
            Configuration used to reconstruct boundaries and schemes.
        case_dir : str
            Resolved case root.
        mesh_data, geo_data : dict
            Native topology and derived geometry matching the archive.
        boundaries : list[dict]
            Boundary records with configured field values.
        velocity, kinematic_pressure : numpy.ndarray
            Full cell/ghost arrays with shapes ``(n_total, 3)`` and
            ``(n_total,)`` in m/s and m²/s².
        eddy_viscosity : numpy.ndarray or None
            Archived cell-centred eddy viscosity in m²/s, when available.
        time : float
            Archived physical time in seconds.
        step : int
            Archived accepted-step index.
        time_step_size : float
            Archived inter-frame step size in seconds.
        solution_dir, samples_dir : str or None, optional
            Paths used by replay diagnostics and output.

        Notes
        -----
        The context does not own or mutate the archive. It lazily caches only
        derived gradient/vorticity arrays for the current frame.
        """
        from ..core.parallel import ParallelContext

        self.setup = setup
        self.case_dir = case_dir
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.boundaries = boundaries
        self.velocity = velocity
        self.kinematic_pressure = kinematic_pressure
        self.eddy_viscosity = eddy_viscosity
        self.time = time
        self.step = step
        self._accepted_time_step_size = time_step_size
        self._current_dt = time_step_size
        self.solution_dir = solution_dir
        self.samples_dir = samples_dir
        self.parallel = ParallelContext()
        self.logger = _NullLogger()
        self.last_forces = None
        self.last_y_plus = None
        self.ibm = None
        self._derived_fields: dict[object, np.ndarray] = {}
        # Runtime output indexes belong to this replay context, never to a
        # reusable sampler specification.
        self._sample_pvd_entries: dict[str, list[tuple[float, str]]] = {}

    def _velocity_gradient(self) -> np.ndarray:
        from ..fields import gradients

        gradient = self._derived_fields.get("velocity_gradient")
        if gradient is None:
            gradient = gradients._resolve_gradient_fn(self.geo_data)(
                self.velocity, self.mesh_data, self.geo_data
            )
            self._derived_fields["velocity_gradient"] = gradient
        return gradient

    def _vorticity_field(self) -> np.ndarray:
        from ..fields import diagnostics

        vorticity = self._derived_fields.get("vorticity")
        if vorticity is None:
            vorticity = diagnostics.compute_vorticity(
                self.velocity,
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
    """Replay archived FVM snapshots through the configured samplers.

    Offline replay rebuilds ghost fields and gradients with the same mesh and
    boundary contracts used online, then invokes :class:`FVMSamplerExecutor`.
    It never advances the numerical solver or overwrites archives.
    """

    def __init__(
        self,
        case_dir,
        config,
        samplers=None,
        mesh=None,
        overwrite: bool = False,
        solution_dir=None,
        samples_dir=None,
    ):
        """Configure deterministic replay of an archived FVM case.

        Parameters
        ----------
        case_dir : str or pathlib.Path
            Case root containing archived solutions and diagnostics.
        config : FVMSetup
            Setup with boundary, scheme, and sampler configuration.
        samplers : sequence[Sampler] or None, default=None
            Optional sampler override; ``config.samplers`` is used otherwise.
        mesh : dict, path, or callable
            Mesh that produced the archive. Required for reconstruction.
        overwrite : bool, default=False
            Replace only selected products in the destination when true.
        solution_dir, samples_dir : path-like or None, optional
            Explicit archive/output directories. Replay defaults to an isolated
            ``samples/replay*`` directory.

        Raises
        ------
        TypeError
            If the mesh is missing or has an unsupported representation.
        """
        from dataclasses import replace

        self.case_dir = str(Path(case_dir).resolve())
        samplers = tuple(samplers) if samplers is not None else tuple(config.samplers or ())
        self.setup = replace(config, samplers=samplers)
        self.mesh_data = _materialize_mesh(mesh)
        self.boundaries = self._setup_boundaries(self.mesh_data)
        self.geo_data = self._build_geometry(self.mesh_data)
        self.overwrite = bool(overwrite)
        self.solution_dir = self._resolve_directory(
            solution_dir,
            default_names=("solutions", "solution"),
        )
        if samples_dir is not None:
            self.samples_dir = self._resolve_directory(samples_dir)
        elif self.overwrite:
            self.samples_dir = (Path(self.case_dir) / "samples").resolve()
        else:
            # Replay is isolated by default.  This keeps a live run's samples
            # and a previous replay intact, while still giving each replay a
            # deterministic, discoverable location.
            samples_root = (Path(self.case_dir) / "samples").resolve()
            candidate = samples_root / "replay"
            suffix = 1
            while candidate.exists():
                candidate = samples_root / f"replay-{suffix}"
                suffix += 1
            self.samples_dir = candidate

    def _resolve_directory(self, value, *, default_names=()) -> Path:
        if value is None:
            for name in default_names:
                candidate = Path(self.case_dir) / name
                if candidate.exists():
                    return candidate.resolve()
            if default_names:
                return (Path(self.case_dir) / default_names[0]).resolve()
            raise TypeError("a directory path is required")
        path = Path(value)
        return (path if path.is_absolute() else Path(self.case_dir) / path).resolve()

    def _build_geometry(self, mesh_data: dict) -> dict:
        from ..fields.gradients import compute_lsq_geometry
        from ..mesh import geometry
        from ..mesh.coupled import configure_cyclic_boundaries

        gs = self.setup.schemes.gradient_scheme
        geo = geometry.compute_mesh_geometry(
            mesh_data, gradient_scheme=gs, compute_lsq=False, logger=None
        )
        configure_cyclic_boundaries(mesh_data, geo)
        if gs == "lsq":
            geo.update(compute_lsq_geometry(mesh_data, geo))  # type: ignore
        return geo

    def _setup_boundaries(self, mesh_data: dict) -> list:
        boundaries = mesh_data["boundary"]
        for b_cfg in self.setup.boundaries:
            for b_mesh in boundaries:
                if b_mesh["name"] == b_cfg.name:
                    velocity = np.asarray(b_cfg.velocity_value, dtype=np.float64)
                    b_mesh.update(
                        {
                            "velocity_type": b_cfg.velocity_type,
                            "pressure_type": b_cfg.pressure_type,
                            "kinematic_pressure_value": b_cfg.kinematic_pressure_value,
                            "eddy_viscosity_type": b_cfg.eddy_viscosity_type,
                            "eddy_viscosity_value": b_cfg.eddy_viscosity_value,
                        }
                    )
                    if b_cfg.mesh_type is not None:
                        b_mesh["type"] = b_cfg.mesh_type
                    else:
                        b_mesh.setdefault("type", "patch")
                    if b_cfg.neighbour_patch is not None:
                        b_mesh["neighbour_patch"] = b_cfg.neighbour_patch
                    if velocity.shape == (3,):
                        b_mesh["velocity_value"] = velocity
                        b_mesh.pop("velocity_value_field", None)
                    else:
                        b_mesh["velocity_value_field"] = velocity
                    break
        return boundaries

    def _pvd_frames(self) -> list[tuple[float, int, Path]]:
        solution_dir = self.solution_dir
        pvd_path = solution_dir / f"{self.setup.case_name}.pvd"
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
        frames = sorted(frames)
        missing = [str(path) for _time, _step, path in frames if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Archived PVD references missing snapshot file(s): " + ", ".join(missing)
            )
        return frames

    def _read_snapshot(self, path: Path, n_cells: int) -> dict[str, np.ndarray | None]:
        """Return interior cell fields for one archived snapshot.

        Partitioned ``.pvtu`` pieces are scattered back to global cell order
        using ``global_cell_id`` (owned cells only); serial ``.vtu`` files are
        used directly.
        """
        import pyvista as pv

        grid = pv.read(str(path))
        cell_data = grid.cell_data

        def field_array(canonical_name: str, *, required: bool = True):
            if canonical_name in cell_data:
                return np.asarray(cell_data[canonical_name], dtype=np.float64)
            if required:
                raise ValueError(
                    f"Archived snapshot {path} lacks canonical field {canonical_name!r}"
                )
            return None

        velocity_values = field_array("velocity")
        pressure_values = field_array("kinematic_pressure")
        eddy_values = field_array("eddy_viscosity", required=False)
        if velocity_values is None or pressure_values is None:
            raise RuntimeError("required canonical snapshot fields were not loaded")

        def read_array(name: str):
            if name not in cell_data:
                return None
            return np.asarray(cell_data[name], dtype=np.float64)

        ghost = read_array("vtkGhostType")
        global_ids = read_array("global_cell_id")
        local_count = int(velocity_values.shape[0])
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
            "velocity": scatter(velocity_values),
            "kinematic_pressure": scatter(pressure_values),
            "eddy_viscosity": scatter(eddy_values),
        }
        if kept != n_cells:
            raise ValueError(
                f"Archived snapshot {path} has {kept} owned cells but the "
                f"post-processed mesh has {n_cells}; the mesh must match the archive"
            )
        return fields

    def _reconstruct_state(self, fields: dict) -> tuple[np.ndarray, np.ndarray]:
        """Assemble full velocity/pressure arrays with boundary ghost values."""
        from ..assemble import convection
        from ..solve import simple_solver

        mesh_data = self.mesh_data
        n_cells = mesh_data["n_cells"]
        n_total = mesh_data["n_faces"] - mesh_data["n_interior_faces"] + n_cells

        velocity = np.zeros((n_total, 3), dtype=np.float64)
        kinematic_pressure = np.zeros(n_total, dtype=np.float64)
        velocity[:n_cells] = fields["velocity"]
        kinematic_pressure[:n_cells] = fields["kinematic_pressure"]

        volumetric_face_flux = convection.compute_volumetric_face_flux(
            velocity, mesh_data, self.geo_data
        )
        simple_solver._update_velocity_bcs(
            velocity,
            volumetric_face_flux,
            self.boundaries,
            mesh_data["owners"],
            self.geo_data,
            n_cells,
            mesh_data["n_interior_faces"],
            mesh_data=mesh_data,
        )
        simple_solver.update_scalar_boundaries(
            kinematic_pressure,
            mesh_data,
            self.boundaries,
            "kinematic_pressure",
            volumetric_face_flux=volumetric_face_flux,
        )
        return velocity, kinematic_pressure

    def _archived_time_steps(self) -> dict[int, float]:
        """Return accepted ``time_step_size`` values keyed by archived solver step."""
        diagnostics = self.solution_dir / "diagnostics.jsonl"
        if not diagnostics.exists():
            return {}
        values: dict[int, float] = {}
        with diagnostics.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                    step = int(record["step"])
                    time_step_size = float(record["time_step_size"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid diagnostics record at {diagnostics}:{line_number}"
                    ) from exc
                if time_step_size <= 0.0:
                    raise ValueError(
                        f"Invalid non-positive time step at {diagnostics}:{line_number}"
                    )
                values[step] = time_step_size
        return values

    def run(self) -> list[tuple[float, int]]:
        """Replay every archived snapshot through the configured samplers.

        By default output is written below an isolated ``samples/replay*``
        directory.  With ``overwrite=True`` only products owned by the
        selected samplers are replaced; unrelated files are preserved.  The snapshot
        ``time_step_size`` passed to the samplers is the *archived* inter-frame advance
        (not ``config.time.time_step_size``), so adaptive-step cases resample offline
        with the same cadence they selected online.

        Returns
        -------
        list[tuple[float, int]]
            ``(time, step)`` for every replayed archived frame.

        Raises
        ------
        FileNotFoundError
            If the PVD index or a referenced snapshot is missing.
        ValueError
            If the archive is corrupt, does not match the mesh, or sampler
            output would be non-monotonic.
        """
        frames = self._pvd_frames()
        archived_time_step_size = self._archived_time_steps()
        self._validate_replay_prerequisites(frames)
        if self.overwrite:
            self._clear_previous_output()
        sampled: list[tuple[float, int]] = []
        last_context = None
        n_cells = self.mesh_data["n_cells"]
        default_time_step_size = float(self.setup.time.time_step_size)
        for index, (time, step, path) in enumerate(frames):
            # Recover the real archived advance: the time between this archived
            # snapshot and the previously archived one (falls back to the
            # nominal dt for the first frame).
            if step in archived_time_step_size:
                time_step_size = archived_time_step_size[step]
            elif index == 0:
                time_step_size = default_time_step_size
            else:
                time_step_size = float(frames[index][0] - frames[index - 1][0])
            fields = self._read_snapshot(path, n_cells)
            velocity, kinematic_pressure = self._reconstruct_state(fields)
            context = SnapshotContext(
                setup=self.setup,
                case_dir=self.case_dir,
                mesh_data=self.mesh_data,
                geo_data=self.geo_data,
                boundaries=self.boundaries,
                velocity=velocity,
                kinematic_pressure=kinematic_pressure,
                eddy_viscosity=fields["eddy_viscosity"],
                time=time,
                step=step,
                time_step_size=time_step_size,
                solution_dir=str(self.solution_dir),
                samples_dir=str(self.samples_dir),
            )
            FVMSamplerExecutor.execute(context, strict=True)
            last_context = context
            sampled.append((time, step))
        if last_context is not None:
            FVMSamplerExecutor.execute(last_context, strict=True, event="final")
        return sampled

    def _clear_previous_output(self) -> None:
        """Remove only selected sampler products from an explicit destination."""
        samples = self.samples_dir
        if samples.exists():
            for sampler in self.setup.samplers:
                stem = str(getattr(sampler, "file_name", None) or sampler.name)
                for pattern in (f"{stem}.csv", f"{stem}.pvd", f"{stem}_*.vts"):
                    for child in samples.glob(pattern):
                        if child.is_file():
                            child.unlink()

    def _validate_replay_prerequisites(self, frames) -> None:
        """Validate archive and sampler inputs before destructive overwrite."""
        for sampler in self.setup.samplers:
            if not callable(getattr(sampler, "sample", None)):
                raise TypeError(
                    f"Configured sampler {type(sampler).__name__} does not implement sample()"
                )
        # Read every archived field set before clearing an explicit output
        # directory.  Replay archives are normally modest; this preflight is
        # the data-integrity guard that prevents a corrupt late frame from
        # deleting an otherwise valid previous analysis.
        n_cells = self.mesh_data["n_cells"]
        for _time, _step, path in frames:
            self._read_snapshot(path, n_cells)
