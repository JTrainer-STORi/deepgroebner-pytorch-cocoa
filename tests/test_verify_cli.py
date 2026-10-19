from __future__ import annotations

import subprocess
import sys

import gymnasium as gym
import pytest

from deepgroebner_pytorch_cocoa import verify_cli


@pytest.mark.parametrize("quiet", [False, True], ids=["normal", "quiet"])
def test_module_verifies_installation_outside_repository(tmp_path, quiet):
    command = [sys.executable, "-m", "deepgroebner_pytorch_cocoa.verify_cli"]
    if quiet:
        command.extend(["--verbosity", "quiet"])

    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.rstrip().endswith("Installation verification succeeded.")
    if quiet:
        assert result.stdout == "Installation verification succeeded.\n"
    else:
        lines = result.stdout.splitlines()
        assert any("package import" in line for line in lines)
        assert any("compiled extension import" in line for line in lines)
        for gym_id in ("BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"):
            assert any(gym_id in line and "reset/step" in line for line in lines)
            assert any(gym_id in line and "cpu" in line.lower() for line in lines)


def test_help_does_not_run_verification(monkeypatch, capsys):
    def unexpected_verification():
        pytest.fail("Help unexpectedly ran installation verification")

    monkeypatch.setattr(verify_cli, "run_verification", unexpected_verification)

    with pytest.raises(SystemExit) as exc:
        verify_cli.main(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr()
    assert "--verbosity" in output.out
    assert "Installation verification succeeded." not in output.out
    assert output.err == ""


def test_failed_check_reports_error_and_exits_nonzero(monkeypatch, capsys):
    def fail_verification():
        raise RuntimeError("native extension could not be loaded")

    monkeypatch.setattr(verify_cli, "run_verification", fail_verification)

    with pytest.raises(SystemExit) as exc:
        verify_cli.main(["--verbosity", "quiet"])

    assert exc.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == (
        "Installation verification failed: native extension could not be loaded\n"
    )


def test_verification_closes_environment_when_step_fails(monkeypatch):
    make_environment = gym.make
    closed = []

    def make_failing_environment(*args, **kwargs):
        env = make_environment(*args, **kwargs)
        close_environment = env.close

        def fail_step(action):
            raise RuntimeError("native step failed")

        def close():
            closed.append(env)
            close_environment()

        monkeypatch.setattr(env, "step", fail_step)
        monkeypatch.setattr(env, "close", close)
        return env

    monkeypatch.setattr(gym, "make", make_failing_environment)

    with pytest.raises(RuntimeError, match="native step failed"):
        verify_cli.run_verification()

    assert len(closed) == 1
