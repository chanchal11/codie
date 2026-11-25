"""Guard unit tests: paths, secrets, and the command allow-list."""

from pathlib import Path

from codie.guards import cap_output, check_path, plan_command, strip_ansi


def test_cap_output_strips_ansi_and_truncates():
    text = "\x1b[31mhello\x1b[0m"
    assert strip_ansi(text) == "hello"
    assert "\x1b" not in cap_output(text)
    assert cap_output("abcdef", limit=3).startswith("abc")
    assert cap_output("abcdef", limit=3).endswith("[truncated]")


def test_path_attacks(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cases = {
        "../../etc/passwd": "escapes the workspace",
        "/etc/passwd": "absolute paths",
        ".env": "looks like a secret",
        ".ssh/id_rsa": "looks like a secret",
        "secrets/app.pem": "looks like a secret",
        "id_rsa": "looks like a secret",
        ".git/config": "looks like a secret",
        "certs/server.key": "looks like a secret",
    }
    for raw, phrase in cases.items():
        result = check_path(workspace, raw)
        assert result is not None, raw
        assert result.startswith("error:"), result
        assert phrase in result, result


def test_symlink_escape(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    link = workspace / "link"
    link.symlink_to(outside)
    result = check_path(workspace, "link")
    assert result is not None
    assert "escapes the workspace" in result


def test_relative_path_inside_workspace_is_allowed(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert check_path(workspace, "notes.md") is None
    assert check_path(workspace, ".") is None
    assert check_path(workspace, "pkg/a.py") is None


def test_command_attacks(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    refused = [
        "ls && rm -rf .",
        "ls | wc",
        "ls ; rm -rf .",
        "grep secret ../.env",
        "find . -delete",
        "find . -exec rm {} \\;",
        "pytest > /tmp/out",
        "cat .env",
        "head .ssh/id_rsa",
        "cat /etc/passwd",
        "wc ../../etc/passwd",
        "/bin/ls",
        "./script",
    ]
    for command in refused:
        result = plan_command(workspace, command)
        assert isinstance(result, str), command
        assert result.startswith("error:"), result


def test_find_exec_and_delete_are_blocked(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    deleted = plan_command(workspace, "find . -delete")
    assert isinstance(deleted, str)
    assert "-delete" in deleted or "not allowed" in deleted
    executed = plan_command(workspace, "find . -name '*.py'")
    assert not isinstance(executed, str)
    assert executed.argv[0] == "find"


def test_unknown_bare_command_needs_approval(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for command, program in (
        ("python -c 'print(1)'", "python"),
        ("rm -rf .", "rm"),
        ("git status", "git"),
    ):
        result = plan_command(workspace, command)
        assert not isinstance(result, str), command
        assert result.needs_approval is True
        assert result.argv[0] == program


def test_pytest_needs_approval_and_ls_does_not(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pytest_plan = plan_command(workspace, "pytest -q")
    assert not isinstance(pytest_plan, str)
    assert pytest_plan.needs_approval is True
    assert pytest_plan.argv == ["pytest", "-q"]
    ls_plan = plan_command(workspace, "ls")
    assert not isinstance(ls_plan, str)
    assert ls_plan.needs_approval is False
    assert ls_plan.argv == ["ls"]


def test_quoted_query_string_is_one_argument(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    command = (
        "curl -X GET 'https://api.open-meteo.com/v1/forecast"
        "?current_weather=true&latitude=22.65&longitude=88.35'"
    )
    plan = plan_command(workspace, command)
    assert not isinstance(plan, str), plan
    assert plan.argv[:3] == ["curl", "-X", "GET"]
    assert plan.argv[3].startswith("https://api.open-meteo.com/")
    assert "&latitude=" in plan.argv[3]
    assert plan.needs_approval is True
    double = plan_command(workspace, 'grep -n "a|b" notes.md')
    assert not isinstance(double, str), double
    assert "a|b" in double.argv


def test_unclosed_quote_is_a_parse_error(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    result = plan_command(workspace, "curl 'https://example.com?a=1&b=2")
    assert isinstance(result, str)
    assert result.startswith("error: could not parse command")


def test_grep_injects_secret_excludes(tmp_path: Path):
    workspace = tmp_path / "ws"
    (workspace / "notes.md").parent.mkdir()
    (workspace / "notes.md").write_text("hello", encoding="utf-8")
    plan = plan_command(workspace, "grep -n hello notes.md")
    assert not isinstance(plan, str)
    assert "--exclude=.env" in plan.argv
    assert "--exclude-dir=.git" in plan.argv
    assert plan.scrub == "grep"
