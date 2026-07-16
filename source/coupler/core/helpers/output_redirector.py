"""Redirect process-level stdout and stderr for coupled solver calls."""

import os
import sys


class OutputRedirector:
    """Capture Python and native output by temporarily replacing file descriptors."""

    def __init__(self, logfile=None, append=True):
        self.logfile = logfile
        self.append = append
        self._devnull = None
        self._log_fd = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None
        self.log_file = None

    def __enter__(self):
        if not self.logfile:
            return self

        mode = "a" if self.append else "w"
        self.log_file = open(self.logfile, mode)
        self._log_fd = self.log_file.fileno()

        # Save original file descriptors
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        # Flush Python buffers before redirecting
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect stdout (1) and stderr (2) to the log file
        os.dup2(self._log_fd, 1)
        os.dup2(self._log_fd, 2)
        return self

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
