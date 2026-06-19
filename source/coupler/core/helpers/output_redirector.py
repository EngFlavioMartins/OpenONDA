"""
Output Redirector for FVM-VPM Coupling.
=======================================
Context manager to capture C-level stdout/stderr (e.g. OpenFOAM output)
and redirect it to log files using os.dup2.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026
"""

import logging
import os
import sys


class OutputRedirector:
    """
    Context manager to redirect stdout and stderr to a log file.
    Critically, it uses os.dup2 to capture C-level output (like OpenFOAM's)
    and redirect it to the file descriptor of the log file.

    It preserves Python's sys.stdout/stderr to continue logging to console if needed,
    but since os.dup2 redirects the underlying file descriptors (1 and 2),
    anything written to them (including Python's print) will go to the file unless
    restored or handled carefully.

    However, typically we want OpenFOAM output (C++) to go to file, but we might want
    Python logs to stay on console. This is tricky because os.dup2 affects the process.
    """

    def __init__(self, logfile=None, append=True):
        self.logfile = logfile
        self.append = append
        self._devnull = None
        self._log_fd = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None

    def __enter__(self):
        if not self.logfile:
            return

        # Open log file
        mode = "a" if self.append else "w"
        try:
            self.log_file = open(self.logfile, mode)
            self._log_fd = self.log_file.fileno()
        except Exception as e:
            logging.getLogger("coupler").warning(
                "[OutputRedirector] Failed to open log file %s: %s", self.logfile, e
            )
            return

        # Save original file descriptors
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        # Flush Python buffers before redirecting
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect stdout (1) and stderr (2) to the log file
        os.dup2(self._log_fd, 1)
        os.dup2(self._log_fd, 2)

    def __exit__(self, _exc_type, _exc_value, traceback):
        if not self.logfile:
            return

        # Flush before restoring
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original file descriptors
        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
            os.close(self._saved_stdout_fd)

        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)

        # Close log file
        if self.log_file:
            self.log_file.close()
