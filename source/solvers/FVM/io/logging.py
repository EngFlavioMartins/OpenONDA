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
    Print OpenONDA solver header in OpenFOAM style.
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
    Centralized logging class for FVM solver output.
    """

    @staticmethod
    def solver_summary(solver):
        """
        Print detailed FVM solver initialization summary.
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
        """Print current time step information."""
        print(f"\n>>> [Time-step: {step:d}] Flow time: {time:.5g} s (dt = {dt:.5g} s)")

    @staticmethod
    def convergence_info(residuals):
        """
        Print formatted convergence residuals.

        Args:
            residuals (dict): Dictionary mapping field names to residual values.
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
        """Print statistics for turbulent viscosity (nut) and compare with molecular nu."""
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
    Simple timer utility for profiling FVM processes.
    """

    _timers = {}

    @staticmethod
    def start(name):
        import time

        Timer._timers[name] = time.time()

    @staticmethod
    def stop(name):
        import time

        if name in Timer._timers:
            elapsed = time.time() - Timer._timers[name]
            return elapsed
        return 0.0

    @staticmethod
    def log(name):
        elapsed = Timer.stop(name)
        if elapsed > 0:
            print(f"    - {name:<20}: {elapsed:.3f} s")
        return elapsed
