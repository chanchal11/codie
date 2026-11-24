"""One-shot CLI behaviour. The TUI is covered in test_tui.py."""

from pathlib import Path

from langchain_core.messages import AIMessage

from codie.approvals import ApprovalRequest, CliApprover, colorize_diff
from codie.cli import main, run_oneshot
from tests.support import ScriptedModel


def test_colorize_diff_uses_ansi_only_for_display():
    plain = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
    painted = colorize_diff(plain)
    assert "\x1b[31m-old" in painted
    assert "\x1b[32m+new" in painted
    assert "\x1b" not in plain


def test_cli_approver_defaults_to_no(capsys):
    approver = CliApprover(input_fn=lambda: "", output_fn=print)
    request = ApprovalRequest(
        call_id="1",
        kind="write",
        summary='write_file("a.py")',
        diff="--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n",
        path="a.py",
    )
    assert approver.ask(request) is False
    output = capsys.readouterr().out
    assert "Allow? [y/N]" in output
    assert "+new" in output
    assert "\x1b[" not in output


def test_cli_approver_accepts_yes():
    approver = CliApprover(input_fn=lambda: "y", output_fn=lambda *_args, **_kwargs: None)
    request = ApprovalRequest(call_id="1", kind="pytest", summary="run pytest", command="pytest -q")
    assert approver.ask(request) is True


def test_oneshot_prints_steps_and_answer(capsys, tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "list_files", "args": {"path": "."}, "id": "1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="The folder has hello.txt."),
        ]
    )
    assert run_oneshot(model, tmp_path, "What is here?", 4) == 0
    output = capsys.readouterr().out
    assert '-> list_files(".")' in output
    assert "<-" in output
    assert "The folder has hello.txt." in output
    assert "\x1b[" not in output


def test_main_rejects_a_missing_workspace(tmp_path: Path, capsys):
    missing = tmp_path / "nope"
    assert main(["--workspace", str(missing), "hello"]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_main_exit_does_not_start_a_model(tmp_path: Path):
    assert main(["--workspace", str(tmp_path), "exit"]) == 0
    assert main(["--workspace", str(tmp_path), "/new"]) == 0
