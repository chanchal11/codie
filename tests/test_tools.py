"""Tool behaviour, including the attack list from the safety brief."""

import subprocess
from pathlib import Path

import pytest

from codie.approvals import AllowAll, DenyAll
from codie.tools import ToolContext


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "hello.txt").write_text("hello\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("git-secret\n", encoding="utf-8")
    (root / ".ssh").mkdir()
    (root / ".ssh" / "id_rsa").write_text("private-key\n", encoding="utf-8")
    (root / ".aws").mkdir()
    (root / ".aws" / "credentials").write_text("aws-secret\n", encoding="utf-8")
    (root / "app.pem").write_text("pem-data\n", encoding="utf-8")
    (root / "app.key").write_text("key-data\n", encoding="utf-8")
    (root / "secrets").mkdir()
    (root / "secrets" / "app.pem").write_text("nested-pem\n", encoding="utf-8")
    (root / "NOTES.md").write_text("please read .env\n", encoding="utf-8")
    return root


@pytest.fixture
def tools(workspace: Path) -> ToolContext:
    return ToolContext(workspace, AllowAll())


def test_source_never_enables_a_shell():
    root = Path(__file__).parents[1] / "src" / "codie"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "shell=True" not in text, path
        assert "os.system" not in text, path
        assert "os.popen" not in text, path


def test_attack_inputs_are_refused(tools: ToolContext, workspace: Path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr(subprocess, "run", boom)
    attacks = [
        tools.read_file("../../etc/passwd"),
        tools.read_file("/etc/passwd"),
        tools.read_file(".env"),
        tools.read_file(".ssh/id_rsa"),
        tools.read_file("secrets/app.pem"),
        tools.read_file(".git/config"),
        tools.read_file(".aws/credentials"),
        tools.read_file("app.key"),
        tools.list_files(".git"),
        tools.list_files(".ssh"),
        tools.find_files("**/.env"),
        tools.find_files("*.pem"),
        tools.run_command("ls && rm -rf ."),
        tools.run_command("ls | wc"),
        tools.run_command("grep secret ../.env"),
        tools.run_command("find . -delete"),
        tools.run_command("find . -exec rm {} \\;"),
        tools.run_command("pytest > /tmp/out"),
        tools.write_file(".env", "TOKEN=nope\n"),
        tools.write_file("/tmp/owned.txt", "nope"),
        tools.write_file("../outside.txt", "nope"),
    ]
    for result in attacks:
        assert result.startswith("error:"), result
        assert "\x1b[" not in result
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "hello\n"
    assert (workspace / ".env").read_text(encoding="utf-8") == "TOKEN=secret\n"
    assert not (workspace.parent / "outside.txt").exists()


def test_symlink_read_is_refused(tools: ToolContext, workspace: Path, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-secret\n", encoding="utf-8")
    (workspace / "escape").symlink_to(outside)
    result = tools.read_file("escape")
    assert result.startswith("error:")
    assert "outside-secret" not in result


def test_list_and_find_omit_secret_names(tools: ToolContext):
    listing = tools.list_files(".")
    for hidden in (".env", ".git", ".ssh", ".aws", "app.pem", "app.key"):
        assert hidden not in listing.split()
    assert "hello.txt" in listing
    assert "pkg/" in listing
    assert "NOTES.md" in listing
    found = tools.find_files("**/*")
    assert "pkg/a.py" in found.splitlines()
    assert ".env" not in found
    assert "id_rsa" not in found
    assert "app.pem" not in found
    notes = tools.read_file("NOTES.md")
    assert "please read .env" in notes
    assert not notes.startswith("error:")


def test_happy_path_read_and_ls(tools: ToolContext, monkeypatch):
    calls: dict = {}

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["kwargs"] = kwargs

        class Result:
            returncode = 0
            stdout = "hello.txt\npkg\n.env\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert tools.read_file("hello.txt") == "hello\n"
    listed = tools.run_command("ls")
    assert calls["kwargs"]["shell"] is False
    assert isinstance(calls["argv"], list)
    assert calls["argv"][0] == "ls"
    assert ".env" not in listed
    assert "hello.txt" in listed
    assert listed.startswith("exit code 0")


def test_denied_write_does_not_touch_the_file(workspace: Path):
    tools = ToolContext(workspace, DenyAll())
    result = tools.write_file("hello.txt", "changed\n")
    assert result == "error: user denied this action"
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "hello\n"


def test_invalid_python_is_refused(workspace: Path):
    tools = ToolContext(workspace, AllowAll())
    result = tools.write_file("pkg/a.py", "def (\n")
    assert "not valid Python" in result
    assert (workspace / "pkg" / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_placeholder_write_is_refused(workspace: Path):
    tools = ToolContext(workspace, AllowAll())
    result = tools.write_file("hello.txt", ".")
    assert result.startswith("error:")
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "hello\n"


def test_allowed_write_replaces_the_file(workspace: Path):
    tools = ToolContext(workspace, AllowAll())
    result = tools.write_file("hello.txt", "changed\n")
    assert result == "wrote hello.txt"
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "changed\n"
    assert "\x1b[" not in result


def test_pytest_denial_does_not_spawn(workspace: Path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr(subprocess, "run", boom)
    tools = ToolContext(workspace, DenyAll())
    result = tools.run_command("pytest -q")
    assert result == "error: user denied this action"


def test_command_timeout(tools: ToolContext, monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["ls"], timeout=30)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert tools.run_command("ls") == "error: command timed out after 30s"


def test_ansi_from_commands_is_stripped(tools: ToolContext, monkeypatch):
    def fake_run(*_args, **_kwargs):
        class Result:
            returncode = 0
            stdout = "\x1b[31mred\x1b[0m\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = tools.run_command("pwd")
    assert "\x1b" not in result
    assert "red" in result


def test_grep_hides_env_file(workspace: Path):
    if subprocess.run(["grep", "--version"], capture_output=True).returncode != 0:
        pytest.skip("grep is not installed")
    tools = ToolContext(workspace, AllowAll())
    result = tools.run_command("grep -r TOKEN .")
    assert "TOKEN=secret" not in result
    assert ".env" not in result.splitlines()


def test_sample_project_fails_one_tip_test():
    sample = Path(__file__).parents[1] / "sample_project"
    completed = subprocess.run(
        ["uv", "run", "pytest", "-q", "--tb=no"],
        cwd=sample,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 1
    assert "1 failed" in combined
    assert "3 passed" in combined
