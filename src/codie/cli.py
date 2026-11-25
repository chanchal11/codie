"""Command line entry. A question runs once; no question opens the TUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from codie.agent import CodingAgent
from codie.approvals import CliApprover, CliSink
from codie.guards import MAX_ROUNDS_DEFAULT
from codie.providers import build_chat_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="codie",
        description="Coding agent for one folder. Ollama by default, Bedrock optional.",
    )
    parser.add_argument("--workspace", default=".", help="Folder the agent may read and edit")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "bedrock"])
    parser.add_argument("--model", default=None, help="Model name or Bedrock model id")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS_DEFAULT)
    parser.add_argument("prompt", nargs="*", help="One-off question. Omit this to open the TUI.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"error: workspace does not exist: {workspace}", file=sys.stderr)
        return 1
    if args.max_rounds < 1:
        print("error: --max-rounds must be at least 1", file=sys.stderr)
        return 1
    question = " ".join(args.prompt).strip()
    if question in {"exit", "/exit"}:
        return 0
    if question in {"new", "/new"}:
        print("Started a new conversation.")
        return 0
    try:
        model = build_chat_model(args.provider, args.model)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if question:
        return run_oneshot(model, workspace, question, args.max_rounds)
    from codie.tui.app import CodieApp

    model_name = args.model or _model_label(args.provider, model)
    CodieApp(
        workspace=workspace,
        provider=args.provider,
        model_name=model_name,
        max_rounds=args.max_rounds,
        model=model,
    ).run()
    return 0


def run_oneshot(model, workspace: Path, question: str, max_rounds: int) -> int:
    """Print steps and the answer without opening the TUI."""
    sink = CliSink()
    agent = CodingAgent(model, workspace, CliApprover(workspace=workspace), sink, max_rounds)
    try:
        answer = agent.run_turn(question)
    except Exception as exc:
        sink.finish()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sink.finish()
    if answer.strip() and answer.strip() != sink.streamed.strip():
        print(answer)
    return 0


def _model_label(provider: str, model) -> str:
    label = getattr(model, "model", None) or getattr(model, "model_id", None)
    if label:
        return str(label)
    return "bedrock" if provider == "bedrock" else "qwen2.5-coder"
