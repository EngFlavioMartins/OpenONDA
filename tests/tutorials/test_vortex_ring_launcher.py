"""Campaign logging must preserve vortex-ring command ordering and failures."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("fail_les", [False, True])
@pytest.mark.parametrize("clean", [False, True])
def test_vortex_ring_campaign_uses_current_variants_and_preserves_exit_status(
    tmp_path, monkeypatch, fail_les, clean
):
    root = tmp_path / "checkout"
    tutorial = root / "tutorials/vpm/vortex_ring"
    tutorial.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "tutorials/vpm/vortex_ring/allrun.sh"
    shutil.copy2(source, tutorial / "allrun.sh")
    for name in ("allclean.sh", "allplot.sh"):
        script = tutorial / name
        script.write_text(f'#!/usr/bin/env bash\nprintf "[stub] {name} %s\\n" "$*"\n')
        script.chmod(0o755)
    interpreter = tmp_path / "python-stub"
    interpreter.write_text(
        '#!/usr/bin/env bash\nprintf "[stub] %s\\n" "$*"\n'
        'if [[ "$*" == *"--variant les_transposed"* && "${FAIL_LES}" == 1 ]]; then exit 23; fi\n'
        "exit 0\n"
    )
    interpreter.chmod(0o755)
    monkeypatch.setenv("OPENONDA_PYTHON", str(interpreter))
    monkeypatch.setenv("TI_OFFLINE_CACHE_FILE_PATH", str(tmp_path / "cache"))
    monkeypatch.setenv("FAIL_LES", "1" if fail_les else "0")

    result = subprocess.run(
        ["bash", str(tutorial / "allrun.sh"), *(["--clean"] if clean else []), "--steps", "12"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == (23 if fail_les else 0)
    assert "[campaign] START | DNS Direct" in result.stdout
    assert "--variant dns_direct --resume --steps 12" in result.stdout
    assert "--variant dns_transposed --resume --steps 12" in result.stdout
    assert "--variant dns_mixed --resume --steps 12" in result.stdout
    assert result.stdout.count("[stub] allclean.sh") == int(clean)
    assert not list((tmp_path / "cache").iterdir())
    if fail_les:
        assert "FAILED | LES Transposed | exit 23" in result.stderr
        assert "DONE  | LES Transposed" not in result.stdout
        assert "[stub] allplot.sh" not in result.stdout
        assert "[campaign] COMPLETE" not in result.stdout
    else:
        assert "[campaign] START | LES Transposed" in result.stdout
        assert "[stub] allplot.sh --strict" in result.stdout
        assert "[campaign] COMPLETE" in result.stdout


@pytest.mark.parametrize("fail_pdf", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_vortex_ring_plot_campaign_runs_all_modules_and_stops_on_failure(
    tmp_path, monkeypatch, fail_pdf, strict
):
    root = tmp_path / "checkout"
    tutorial = root / "tutorials/vpm/vortex_ring"
    tutorial.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "tutorials/vpm/vortex_ring/allplot.sh"
    shutil.copy2(source, tutorial / "allplot.sh")
    samples = tutorial / "samples/dns_treecode"
    samples.mkdir(parents=True)
    (samples / "ring_diagnostics.csv").write_text("snapshot input\n", encoding="utf-8")

    interpreter = tmp_path / "python-stub"
    interpreter.write_text(
        '#!/usr/bin/env bash\nprintf "[stub] %s\\n" "$*"\n'
        'if [[ "$*" == *"plot_vortex_ring_energy --format pdf"* && "${FAIL_PDF}" == 1 ]]; then exit 29; fi\n'
        "exit 0\n"
    )
    interpreter.chmod(0o755)
    monkeypatch.setenv("OPENONDA_PYTHON", str(interpreter))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("FAIL_PDF", "1" if fail_pdf else "0")

    result = subprocess.run(
        ["bash", str(tutorial / "allplot.sh"), *(["--strict"] if strict else [])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == (29 if fail_pdf else 0)
    validation_argument = "" if strict else " --available"
    assert f"assets.postprocess{validation_argument} --pre-plot" in result.stdout
    assert "assets.plot_vortex_ring_motion --format png" in result.stdout
    assert "assets.plot_vortex_ring_energy --format pdf" in result.stdout
    if fail_pdf:
        assert "assets.plot_vortex_ring_stability --format pdf" not in result.stdout
        assert "assets.postprocess --manifest" not in result.stdout
    else:
        assert "assets.plot_vortex_ring_stability --format pdf" in result.stdout
        assert "assets.postprocess --manifest" in result.stdout
        assert result.stdout.rstrip().endswith(f"assets.postprocess{validation_argument}")
