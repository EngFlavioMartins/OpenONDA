"""Qualification-only thread-scoped IEEE environment with exact restoration.

Taichi initialization enables host flush-to-zero, even for the CUDA backend.
An interval computation must not waive its gradual-underflow admission. This
helper uses a separately built small C bridge to save the calling thread's
public fenv_t, enter FE_DFL_ENV, and restore the saved environment on all exits.
No Taichi or GPU call should be made inside this arithmetic-only scope. No
platform fenv struct layout or FE_DFL_ENV integer encoding is hard-coded.
"""

from contextlib import contextmanager
import ctypes
from pathlib import Path
import threading

from tests.vpm._gaussian_tail_arithmetic import _platform


class IntervalEnvironment:
    def __init__(self, library):
        path = Path(library).resolve(strict=True)
        self.library_path = path
        self.library = ctypes.CDLL(str(path))
        self.library.openonda_qualification_fp_enter.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.library.openonda_qualification_fp_enter.restype = ctypes.c_int
        self.library.openonda_qualification_fp_restore.argtypes = [ctypes.c_void_p]
        self.library.openonda_qualification_fp_restore.restype = ctypes.c_int

    @contextmanager
    def ieee(self):
        owner = threading.get_ident()
        handle = ctypes.c_void_p()
        status = self.library.openonda_qualification_fp_enter(ctypes.byref(handle))
        if status or not handle.value:
            raise RuntimeError(f"Cannot enter IEEE interval environment: status={status}")
        try:
            _platform()
            yield
        finally:
            if threading.get_ident() != owner:
                raise RuntimeError("Interval scope must restore on its original thread")
            if self.library.openonda_qualification_fp_restore(handle):
                raise RuntimeError("Failed restoring the host floating-point environment")
