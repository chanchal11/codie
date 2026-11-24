# Design note

## How the loop works

A turn is a LangGraph graph, not a single chat completion.

1. The agent node sends the system prompt plus the message list to the chat model. The prompt is sent on every call. Ollama and Bedrock do not keep it for us.
2. If the model returns plain text, the turn ends.
3. If it returns tool calls, every call is answered with a `ToolMessage` that has the same id. Several calls in one reply are all run before the model is asked again.
4. Writes and `pytest` pause the graph with `interrupt()` before they run. The CLI or the Textual modal resumes with yes or no. The tools node then runs once, so a command is not executed twice.
5. The graph stops when the model stops calling tools, or when the round cap is hit. On the cap, each unanswered tool call gets `error: step limit reached before this tool ran`, then a short assistant message. The next question is not stuck behind an open tool call.

Conversation memory is the message list on the agent. `/new` clears it. Switching models builds a new chat model and keeps that list, so a follow-up still sees the earlier tool results.

## What each guard stops

Guards live in `src/codie/guards.py` and return `error: ...` text. They do not raise, and they do not include ANSI codes.

- **Workspace jail.** The path is resolved, then checked with `Path.relative_to` against the resolved workspace. `../../etc/passwd` fails that check. An absolute path is refused before the join. A symlink is followed by `resolve()`, so a link to a file outside the folder fails the same check.
- **Secrets.** Any path part that is `.env`, `.ssh`, `.aws`, `.git`, `id_rsa`, `*.pem`, or `*.key` is refused for read, write, list, and find. `list_files` omits those names. `grep` is launched with `--exclude` and `--exclude-dir`, and listing output is scrubbed, so a recursive grep does not print secret files.
- **No shell.** The command string is rejected if it contains a pipe, semicolon, ampersand, redirect, backtick, dollar sign, or newline. What remains is split with `shlex` and passed to `subprocess.run(..., shell=False)`. `ls && rm -rf .` never runs, because `&` is rejected and because the program is not handed to a shell that would interpret it.
- **Allow-list.** The first argv token must be a bare name: `ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `pwd`, or `pytest`. `find` with `-exec`, `-ok`, or `-delete` is refused even though `find` is listed. `pytest` is marked for approval. `rm`, `python`, and a path like `/bin/ls` are refused.
- **Limits.** Commands die after 30 seconds. Tool text sent back to the model is capped and stripped of colour codes.

## Why approval sits in code

The model can be talked into anything, including by a file it just read. A sentence in the prompt that says "ask first" is a wish. `interrupt()` runs in our process, before `write_text` or `subprocess.run`, and the only way past it is the yes/no returned by the CLI or the TUI. A denial is a tool result, so the model can change its plan, and it cannot skip the check by claiming it already has permission.
