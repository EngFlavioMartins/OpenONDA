"""Configuration for the supported FVM–VPM coupling path."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CouplerSetup:
    """Shared interface and OpenFOAM case settings."""

    u_inf: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    nu: float | None = None
    rho: float | None = None
    dt: float | None = None
    t_end: float | None = None
    backup_period: int = 1
    log_period: int = 1

    backend: str = "fvm"
    fvm_box: tuple[float, float, float, float, float, float] | None = None
    patch_name: str = "numericalBoundary"
    wall_patch_name: str | None = "cube"
    grid_spacing: float | None = None
    initial_U: list[float] | None = None
    case_dir: str = "."
    surface: dict = field(default_factory=dict)

    h: float = 0.05
    buffer_thickness: float = 0.30
    dead_zone_h: float = 3.0
    prune_vorticity_min: float = 0.005
    handoff_max_particles: int | None = None
    overlap_radius_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.backend not in ("ofw", "fvm"):
            raise ValueError("backend must be 'ofw' or 'fvm'")

        u_inf = np.asarray(self.u_inf, dtype=np.float64)
        if u_inf.shape != (3,) or not np.all(np.isfinite(u_inf)):
            raise ValueError("u_inf must be a finite three-component vector")

        if self.initial_U is not None:
            initial_u = np.asarray(self.initial_U, dtype=np.float64)
            if initial_u.shape != (3,) or not np.all(np.isfinite(initial_u)):
                raise ValueError("initial_U must be a finite three-component vector")

        if self.fvm_box is not None:
            box = np.asarray(self.fvm_box, dtype=np.float64)
            if box.shape != (6,) or not np.all(np.isfinite(box)):
                raise ValueError("fvm_box must contain six finite bounds")
            if np.any(box[1::2] <= box[::2]):
                raise ValueError("Each fvm_box upper bound must exceed its lower bound")

        positive = {
            "nu": self.nu,
            "rho": self.rho,
            "dt": self.dt,
            "t_end": self.t_end,
            "grid_spacing": self.grid_spacing,
            "h": self.h,
            "buffer_thickness": self.buffer_thickness,
            "overlap_radius_ratio": self.overlap_radius_ratio,
        }
        invalid = [
            name
            for name, value in positive.items()
            if value is not None and (not np.isfinite(value) or value <= 0.0)
        ]
        if invalid:
            raise ValueError(f"Coupling values must be positive: {', '.join(invalid)}")
        if self.backup_period < 0 or self.log_period < 1:
            raise ValueError("backup_period must be non-negative and log_period positive")
        if self.dead_zone_h < 0.0 or self.prune_vorticity_min < 0.0:
            raise ValueError("dead_zone_h and prune_vorticity_min must be non-negative")
        if self.buffer_thickness <= self.dead_zone_h * self.h:
            raise ValueError("buffer_thickness must exceed dead_zone_h * h")
        if self.overlap_radius_ratio < 1.0:
            raise ValueError("overlap_radius_ratio must be at least one")
        if self.handoff_max_particles is not None and self.handoff_max_particles < 1:
            raise ValueError("handoff_max_particles must be positive")

    def require_case_fields(
        self, names: tuple[str, ...] = ("nu", "dt", "t_end", "fvm_box", "grid_spacing")
    ) -> None:
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ValueError(f"backend='ofw' requires: {', '.join(missing)}")

    @property
    def U_inf(self) -> np.ndarray:
        return np.asarray(self.u_inf, dtype=np.float64)

    def to_dict(self) -> dict:
        domain = None
        if self.fvm_box is not None:
            domain = dict(
                zip(
                    ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
                    self.fvm_box,
                    strict=True,
                )
            )
        return {
            "physics": {
                "u_inf": self.u_inf,
                "nu": self.nu,
                "rho": self.rho,
                "dt": self.dt,
                "t_end": self.t_end,
                "backup_period": self.backup_period,
                "log_period": self.log_period,
            },
            "fvm_solver": {
                "backend": self.backend,
                "patch_name": self.patch_name,
                "wall_patch_name": self.wall_patch_name,
                "grid_spacing": self.grid_spacing,
                "initial_U": self.initial_U,
                "fvm_domain": domain,
                "surface": self.surface,
            },
            "vpm_solver": {
                "particle_spacing": self.h,
                "buffer_thickness": self.buffer_thickness,
                "dead_zone_h": self.dead_zone_h,
                "h": self.h,
            },
            "coupler": {
                "prune_vorticity_min": self.prune_vorticity_min,
                "handoff_max_particles": self.handoff_max_particles,
                "overlap_radius_ratio": self.overlap_radius_ratio,
            },
        }
