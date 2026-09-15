"""Output ownership, lazy rendering and bounded log work across solver modules."""

import ast
from io import StringIO
import logging
from pathlib import Path
from types import SimpleNamespace

from source import log_style
from source.coupler import reporting
from source.solvers.fvm.io.logging import Logging as FVMLogging
from source.solvers.vpm.io.logging import Logging as VPMLogging


class Unrenderable:
    def __str__(self):
        raise AssertionError("Skipped reports must not format values")

    def __array__(self, *args, **kwargs):
        raise AssertionError("Logging must not download or reduce a skipped field")


def test_skipped_vpm_step_does_not_format_or_accumulate_history(monkeypatch):
    monkeypatch.setattr(VPMLogging, "_last_progress_wall", 1.0)
    monkeypatch.setattr(VPMLogging, "_reported_step", 1)
    monkeypatch.setattr(VPMLogging, "_active_step", None)
    monkeypatch.setattr(VPMLogging, "_pending_sections", {})
    monkeypatch.setattr(VPMLogging, "_routine_messages_enabled", True)
    for step in range(2, 1002):
        VPMLogging.begin_step(step)
        VPMLogging.record("regularization", ("measurement", Unrenderable()))
        VPMLogging.time_step(step, step * 0.1, 2.0)
        assert len(VPMLogging._pending_sections) == 1
    VPMLogging.begin_step(1002)
    assert not VPMLogging._pending_sections


def test_disabled_fvm_and_skipped_turbulence_do_not_touch_values(tmp_path, monkeypatch):
    logger = FVMLogging(tmp_path, enabled=False)
    logger.record("mesh", ("cells", Unrenderable()))
    logger.turbulence_info(Unrenderable(), 1.0)
    assert not (tmp_path / "solution").exists()
    logger.enabled = True
    monkeypatch.setattr(logger, "schedule", SimpleNamespace(is_due=lambda *args: False))
    logger.step_begin(2, 0.2, 0.1)
    logger.turbulence_info(Unrenderable(), 1.0)
    logger.record("LES", ("field", Unrenderable()))
    assert logger._step.turbulence is None
    assert not logger._step.events
    logger.step_end(0.1)


def test_coupler_defers_transfer_rows_and_keeps_warnings_immediate(tmp_path, monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(reporting.time, "perf_counter", lambda: clock[0])
    console = StringIO()
    logger = logging.Logger("coupler-test")
    handler = logging.StreamHandler(console)
    handler.setFormatter(log_style.Formatter())
    logger.addHandler(handler)
    file_handler = reporting.configure_logging(tmp_path, logger)
    calls = []

    def rows():
        calls.append(1)
        return (("closure", 1e-12),)

    try:
        reporting.begin_coupling_step(logger, 1, 10, 0.1)
        logger.info(log_style.Event("interface transfer", rows))
        assert not calls
        reporting.finish_coupling_step(logger)
        assert len(calls) == 1
        for step in range(2, 10):
            reporting.begin_coupling_step(logger, step, 10, step * 0.1)
            logger.info(log_style.Event("interface transfer", rows))
            reporting.finish_coupling_step(logger)
        assert len(calls) == 1
        reporting.begin_coupling_step(logger, 10, 10, 1.0)
        logger.warning("Interface mismatch exceeds the configured limit")
        assert "WARNINGS" in console.getvalue()
        assert "Interface mismatch" in console.getvalue()
        assert "COUPLER TIME STEP 10" not in console.getvalue()
        logger.info(log_style.Event("interface transfer", rows))
        reporting.finish_coupling_step(logger)
        assert len(calls) == 2
        assert "COUPLER TIME STEP 10 / 10" in console.getvalue()
    finally:
        file_handler.close()


def test_fvm_warning_is_visible_before_step_end(tmp_path, monkeypatch):
    from source.solvers.fvm.io import logging as module

    stream = StringIO()
    monkeypatch.setattr(module, "_CONSOLE_STDOUT", stream)
    logger = FVMLogging(tmp_path)
    try:
        logger.step_begin(5, 0.5, 0.1)
        logger.warning("Linear solve failed to reach its residual target")
        assert "WARNINGS" in stream.getvalue()
        assert "Linear solve failed" in stream.getvalue()
        assert "FVM TIME STEP" not in stream.getvalue()
    finally:
        logger.close()


def test_shared_report_wraps_paths_vectors_and_module_headings():
    text = log_style.block_report(
        "diagnostics",
        [
            (
                "module " * 20,
                [
                    ("long label " * 10, "/long/path/" * 30),
                    ("impulse", (1e5, -1e-7, 2.2), "m^4/s"),
                    ("relative error", 2.5e-12),
                ],
            )
        ],
    )
    assert max(map(len, text.splitlines())) <= log_style.WIDTH
    assert "m^4/s" in text
    assert "2.5000e-12" in text


def test_solver_prints_are_owned_by_reviewed_sinks():
    """Reject future direct-print bypasses in any solver, mesher or operator."""
    root = Path(__file__).resolve().parents[1]
    allowed = {
        Path("source/config.py"),  # Explicit build-information CLI.
        Path("source/solvers/fvm/io/logging.py"),
        Path("source/solvers/vpm/io/logging.py"),
    }
    violations = []
    for path in (root / "source").rglob("*.py"):
        relative = path.relative_to(root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if (
                name in {"print", "sys.stdout.write", "sys.stderr.write", "warnings.warn"}
                and relative not in allowed
            ):
                violations.append(f"{relative}:{node.lineno}: {name}")
    assert not violations, "Unreviewed output bypasses:\n" + "\n".join(violations)


def test_vpm_integral_report_never_evaluates_particle_properties(monkeypatch, capsys):
    class Sample:
        step, time, elapsed_wall_time = 4, 0.4, 30.0
        _flow_integrals = {
            "kinetic_energy_rate_source": "direct_energy_backward_difference",
            "total_kinetic_energy": 1.0,
            "kinetic_energy_rate": -0.1,
            "viscous_kinetic_energy_rate": -0.1,
            "vortex_strength_magnitude_sum": 2.0,
            "net_vortex_strength": (0.0, 0.0, 1.0),
            "linear_impulse": (0.0, 1.0, 0.0),
            "angular_impulse": (1.0, 0.0, 0.0),
            "total_enstrophy": 3.0,
            "total_helicity": 0.0,
        }
        _diagnostics_history = {"time": [0.4], "vortex_centroid": [(1.0, 2.0, 3.0)]}

        @property
        def total_linear_impulse(self):
            raise AssertionError("Logger invoked a particle reduction")

        @property
        def vortex_centroid(self):
            raise AssertionError("Logger invoked a centroid kernel")

        @property
        def vortex_centroids_by_group(self):
            raise AssertionError("Logger downloaded group fields")

    monkeypatch.setattr(VPMLogging, "_routine_messages_enabled", True)
    monkeypatch.setattr(VPMLogging, "_pending_sections", {})
    monkeypatch.setattr(VPMLogging, "_reported_step", None)
    VPMLogging.flow_diagnostics(Sample())
    output = capsys.readouterr().out
    assert "Linear impulse / density" in output
    assert "Centroid" in output


def test_rejected_fvm_step_is_never_published_as_accepted(tmp_path, monkeypatch):
    from source.solvers.fvm.io import logging as module

    stream = StringIO()
    monkeypatch.setattr(module, "_CONSOLE_STDOUT", stream)
    logger = FVMLogging(tmp_path)
    logger.step_begin(1, 0.1, 0.1)
    logger.convergence_info({"velocity": 1e-3})
    logger.step_end(0.2, accepted=False)
    logger.close(status="failed", failure=RuntimeError("injected failure"))
    assert "FVM TIME STEP" not in stream.getvalue()
    assert "injected failure" in stream.getvalue()
