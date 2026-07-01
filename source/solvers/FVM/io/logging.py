"""
Logging module for FVM solver.
==============================

Author: OpenONDA Team
Date: January 2026
"""

from datetime import datetime
import getpass
import os
import platform
import socket
import sys

from source.version import __version__


def print_openonda_header(precision="f64"):
    """
    Print the OpenONDA solver header banner in OpenFOAM style.

    Displays version, build architecture, precision, execution info,
    hostname, user, and PID in a formatted box.

    Args:
        precision (str): Floating-point precision label (e.g., ``"f64"``).
            Defaults to ``"f64"``.

    Example:
        >>> print_openonda_header("f64")
    """
    now = datetime.now()
    date_str = now.strftime("%b %d %Y")
    time_str = now.strftime("%H:%M:%S")

    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    # Get Python and platform info
    python_version = platform.python_version()
    system_info = f"{platform.system()}; python={python_version}; arch={platform.machine()}"

    pid = os.getpid()

    width = 91

    s = "\n/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /\n"
    s += "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n"
    s += "   ░██████                                      ░██████   ░███    ░██ ░███████      ░███    \n"
    s += "  ░██   ░██                                    ░██   ░██  ░████   ░██ ░██   ░██    ░██░██   \n"
    s += " ░██     ░██ ░████████   ░███████  ░████████  ░██     ░██ ░██░██  ░██ ░██    ░██  ░██  ░██  \n"
    s += " ░██     ░██ ░██    ░██ ░██    ░██ ░██    ░██ ░██     ░██ ░██ ░██ ░██ ░██    ░██ ░█████████ \n"
    s += " ░██     ░██ ░██    ░██ ░█████████ ░██    ░██ ░██     ░██ ░██  ░██░██ ░██    ░██ ░██    ░██ \n"
    s += "  ░██   ░██  ░███   ░██ ░██        ░██    ░██  ░██   ░██  ░██   ░████ ░██   ░██  ░██    ░██ \n"
    s += "   ░██████   ░██░█████   ░███████  ░██    ░██   ░██████   ░██    ░███ ░███████   ░██    ░██ \n"
    s += "             ░██                                                                            \n"
    s += "             ░██                                                                            \n"
    s += "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n"
    s += "| O pen       | " + "".ljust(width - 16) + "|\n"
    s += (
        "| O perator   | "
        + "OpenONDA: Operator for Numerical Design & Aerodynamics.".ljust(width - 16)
        + "|\n"
    )
    s += "| N umer.     | " + f"Version: {__version__}".ljust(width - 16) + "|\n"
    s += (
        "| D esign     | "
        + "Website: https://github.com/EngFlavioMartins".ljust(width - 16)
        + "|\n"
    )
    s += "| A erodyn.   | " + "FVM Solver: Finite Volume Method".ljust(width - 16) + "|\n"
    s += "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n"
    s += f"Build  : OpenONDA={__version__}\n"
    s += f"Arch   : {system_info}\n"
    if precision is not None:
        s += f"Precision: {precision}\n"
    s += "Exec   : FVM Solver\n"
    s += f"Date   : {date_str}\n"
    s += f"Time   : {time_str}\n"
    s += f"Host   : {hostname}\n"
    s += f"User   : {username}\n"
    s += f"PID    : {pid}\n"
    s += "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n"
    s += "/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /\n"

    print(s)
    sys.stdout.flush()


class Logging:
    """
    Centralised logging class for FVM solver output.

    Provides static methods to print solver initialisation summaries,
    per-step information, convergence residuals, wall y+ statistics,
    and turbulence diagnostics.

    All methods print directly to stdout and flush immediately to
    ensure progress is visible in real-time (e.g. in CI or remote runs).
    """

    @staticmethod
    def solver_summary(solver):
        """
        Print a detailed FVM solver initialisation summary.

        Reports case configuration (algorithm, linear solver), mesh
        statistics (cells, faces, boundaries), numerical parameters
        (time step, relaxation factors), transport properties, and
        turbulence-model settings.

        Args:
            solver: The FVM solver instance. Expected to have:
                ``solver.config`` with ``case_name``, ``solver``,
                ``time``, ``transport``, and optionally ``turbulence``;
                ``solver.mesh_data`` with ``n_elements``, ``n_faces``,
                ``n_interior_faces``, and ``boundary``.

        Example:
            >>> Logging.solver_summary(solver)
        """
        print("-" * 60)
        print("FVM SOLVER INITIALIZATION SUMMARY")
        print("-" * 60)

        config = solver.config
        mesh = solver.mesh_data

        print("CASE CONFIGURATION:")
        print(f"  Case Name                : {config.case_name}")
        print(f"  Algorithm                : {config.solver.algorithm}")
        print(f"  Linear Solver            : {config.solver.linear_solver}")

        print("\nMESH STATISTICS:")
        print(f"  Elements (Cells)         : {mesh['n_elements']}")
        print(f"  Faces                    : {mesh['n_faces']}")
        print(f"  Interior Faces           : {mesh['n_interior_faces']}")
        print(f"  Boundary Faces           : {mesh['n_faces'] - mesh['n_interior_faces']}")
        print(f"  Boundaries               : {len(mesh['boundary'])}")
        for b in mesh["boundary"]:
            print(f"    - {b['name']:<15} : {b['nFaces']} faces ({b['type']})")

        print("\nNUMERICAL PARAMETERS:")
        print(f"  Time Step (dt)           : {config.time.delta_t} s")
        print(f"  End Time                 : {config.time.end_time} s")
        print(f"  Under-relaxation (U)     : {config.solver.alpha_u}")
        print(f"  Under-relaxation (p)     : {config.solver.alpha_p}")
        if config.solver.algorithm.upper() == "PIMPLE":
            print(f"  Correctors               : {config.solver.n_correctors}")
            print(f"  Outer Correctors         : {config.solver.n_outer_correctors}")

        print("\nTRANSPORT PROPERTIES:")
        print(f"  Density (density)        : {config.transport.density}")
        print(f"  Viscosity (nu)           : {config.transport.nu}")

        # Turbulence model
        if hasattr(config, "turbulence") and config.turbulence is not None:
            turb = config.turbulence
            print("")
            print("TURBULENCE MODEL:")
            print(f"  Model                    : {turb.model}")
            print(f"  Smagorinsky C_s         : {turb.Cs}")
            print(f"  Dynamic                  : {turb.dynamic}")
        else:
            print("")
            print("TURBULENCE MODEL:")
            print("  Status: Not initialized")

        print("-" * 60)
        print()
        sys.stdout.flush()

    @staticmethod
    def step_info(time, step, dt):
        """
        Print the current time-step header.

        Args:
            time (float): Current flow time in seconds.
            step (int): Current time-step index (1-based).
            dt (float): Current time-step size in seconds.

        Example:
            >>> Logging.step_info(0.015, 150, 0.0001)
            >>> [Time-step: 150] Flow time: 0.015 s (dt = 0.0001 s)
        """
        print(f"\n>>> [Time-step: {step:d}] Flow time: {time:.5g} s (dt = {dt:.5g} s)")

    @staticmethod
    def convergence_info(residuals):
        """
        Print formatted convergence residuals for the current iteration.

        Each field and its associated residual value is printed using
        scientific notation.

        Args:
            residuals (dict): Dictionary mapping field names (str) to
                residual values (float), e.g. ``{"U_x": 1.2e-4, "p": 3.5e-6}``.

        Returns:
            None

        Example:
            >>> Logging.convergence_info({"U_x": 1.2e-4, "p": 3.5e-6})
              Convergence residuals:
                U_x        : 1.200e-04
                p          : 3.500e-06
        """
        if not residuals:
            return

        print("  Convergence residuals:")
        for field, res in residuals.items():
            print(f"    {field:<10} : {res:.3e}")
        sys.stdout.flush()

    @staticmethod
    def yplus_info(yplus_stats):
        """
        Print y+ statistics for wall boundaries.

        For each wall patch, the minimum, maximum, and average y+ values
        are displayed. This is useful for verifying near-wall mesh
        resolution (target values depend on the turbulence model).

        Args:
            yplus_stats (dict): Dictionary mapping patch names (str) to
                dicts with keys ``"min"``, ``"max"``, ``"avg"`` (float),
                e.g. ``{"lower_wall": {"min": 0.5, "max": 2.1, "avg": 1.3}}``.

        Returns:
            None

        Example:
            >>> Logging.yplus_info({"wall": {"min": 0.3, "max": 1.8, "avg": 1.1}})
              y+ statistics:
                wall           : min=0.30, max=1.80, avg=1.10
        """
        if not yplus_stats:
            return

        print("  y+ statistics:")
        for name, stats in yplus_stats.items():
            print(
                f"    {name:<15} : min={stats['min']:.2f}, max={stats['max']:.2f}, avg={stats['avg']:.2f}"
            )

    @staticmethod
    def turbulence_info(nut, nu_molecular):
        """
        Print turbulence diagnostics for the turbulent viscosity field.

        Computes and prints the minimum, maximum, and mean of the
        turbulent viscosity (nut), as well as the ratio nut/nu_molecular
        to assess the relative importance of turbulent vs. molecular
        diffusion.

        Args:
            nut (np.ndarray | None): Turbulent viscosity field
                (1-D array of shape ``(n_cells,)``). If ``None`` or
                empty the method returns immediately.
            nu_molecular (float): Molecular kinematic viscosity (m^2/s).
                Used to compute the nut/nu ratio.

        Returns:
            None

        Example:
            >>> Logging.turbulence_info(solver.nut, 1.5e-5)
              Turbulence diagnostics:
                nut [m2/s] - Min: 1.000e-06, Max: 5.000e-04, Mean: 2.500e-05
                nut/nu ratio: [6.667e-02, 3.333e+01]
        """
        if nut is None:
            return
        import numpy as _np

        nut_arr = _np.asarray(nut)
        if nut_arr.size == 0:
            return

        nut_min = float(_np.min(nut_arr))
        nut_max = float(_np.max(nut_arr))
        nut_mean = float(_np.mean(nut_arr))
        ratio_min = nut_min / nu_molecular if nu_molecular > 0 else float("inf")
        ratio_max = nut_max / nu_molecular if nu_molecular > 0 else float("inf")

        print("  Turbulence diagnostics:")
        print(f"    nut [m2/s] - Min: {nut_min:.3e}, Max: {nut_max:.3e}, Mean: {nut_mean:.3e}")
        print(f"    nut/nu ratio: [{ratio_min:.3e}, {ratio_max:.3e}]")
        sys.stdout.flush()


class Timer:
    """
    Lightweight wall-clock timer utility for profiling FVM solver
    components.

    Uses ``time.time()`` for high-resolution wall-clock measurements.
    Timers are stored in a class-level dictionary keyed by name, which
    allows multiple named timers to run concurrently.

    Example:
        >>> Timer.start("assembly")
        >>> ...  # build matrices
        >>> elapsed = Timer.stop("assembly")
        >>> Timer.log("assembly")
    """

    _timers = {}

    @staticmethod
    def start(name):
        """
        Start (or restart) a named timer.

        Records ``time.time()`` in the class-level ``_timers`` dict.
        If the timer already exists its reference is overwritten.

        Args:
            name (str): Unique identifier for the timer.

        Example:
            >>> Timer.start("assemble_matrix")
        """
        import time

        Timer._timers[name] = time.time()

    @staticmethod
    def stop(name):
        """
        Stop a named timer and return the elapsed wall-clock time.

        Args:
            name (str): Timer identifier previously passed to
                :meth:`start`.

        Returns:
            float: Elapsed time in seconds. Returns ``0.0`` if the
            timer has not been started.

        Example:
            >>> elapsed = Timer.stop("assemble_matrix")
            >>> print(f"Assembly took {elapsed:.3f} s")
        """
        import time

        if name in Timer._timers:
            elapsed = time.time() - Timer._timers[name]
            return elapsed
        return 0.0

    @staticmethod
    def log(name):
        """
        Stop a named timer, print the elapsed time, and return it.

        Convenience wrapper around :meth:`stop` that also prints the
        result to stdout.

        Args:
            name (str): Timer identifier to stop and log.

        Returns:
            float: Elapsed time in seconds (same as :meth:`stop`).

        Example:
            >>> Timer.log("solve_system")
                - solve_system        : 0.423 s
        """
        elapsed = Timer.stop(name)
        if elapsed > 0:
            print(f"    - {name:<20}: {elapsed:.3f} s")
        return elapsed
