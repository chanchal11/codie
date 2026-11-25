# Design note

## How the loop works

A turn is a LangGraph graph, not a single chat completion.

1. The agent node sends the system prompt plus the message list to the chat model. The prompt is sent on every call. Ollama and Bedrock do not keep it for us.
2. If the model returns plain text, the turn ends.
3. If it returns tool calls, every call is answered with a `ToolMessage` that has the same id. Several calls in one reply are all run before the model is asked again.
4. Writes, edits, `pytest`, MCP calls, saved guidelines, and a switch from plan to agent pause the graph with `interrupt()` before they run. The CLI or the Textual modal resumes with yes, always, or no. An always-allow rule stored for this workspace skips the pause. The tools node then runs once, so a command is not executed twice.
5. The graph stops when the model stops calling tools, or when the round cap is hit. On the cap, each unanswered tool call gets `error: step limit reached before this tool ran`, then a short assistant message. The next question is not stuck behind an open tool call.

Conversation memory is the message list on the agent. The TUI writes that list to `~/.codie` after a successful turn. `/new` archives it and starts an empty one. Switching models builds a new chat model and keeps that list, so a follow-up still sees the earlier tool results.

## What each guard stops

Guards live in `src/codie/guards.py` and return `error: ...` text. They do not raise, and they do not include ANSI codes.

- **Workspace jail.** The path is resolved, then checked with `Path.relative_to` against the resolved workspace. `../../etc/passwd` fails that check. An absolute path is refused before the join. A symlink is followed by `resolve()`, so a link to a file outside the folder fails the same check.
- **Secrets.** Any path part that is `.env`, `.ssh`, `.aws`, `.git`, `id_rsa`, `*.pem`, or `*.key` is refused for read, write, list, and find. `list_files` omits those names. `grep` is launched with `--exclude` and `--exclude-dir`, and listing output is scrubbed, so a recursive grep does not print secret files.
- **No shell.** The command string is rejected if it contains a pipe, semicolon, ampersand, redirect, backtick, dollar sign, or newline. What remains is split with `shlex` and passed to `subprocess.run(..., shell=False)`. `ls && rm -rf .` never runs, because `&` is rejected and because the program is not handed to a shell that would interpret it.
- **Commands.** The first argv token must be a bare name. `ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, and `pwd` run immediately. `pytest` asks for approval and can be always-allowed. Any other bare program asks Allow, Deny, or Block. Block remembers the program name for this workspace and later calls are refused without a prompt. `find` with `-exec`, `-ok`, or `-delete` is refused. A path like `/bin/ls` is refused. Shell syntax, secrets, and paths outside the workspace are still refused with no prompt.
- **Limits.** Commands die after 30 seconds. Tool text sent back to the model is capped and stripped of colour codes.

## Where durable state lives

`~/.codie` (or `CODIE_HOME` in tests) is the only place Codie writes account data. The workspace jail does not apply to it, and the model has no tool that edits it.

- `config.json` stores the active search provider, the SearXNG URL, the last mode, and the selected TUI theme. API keys stay in the environment.
- `mcp.json` is the MCP server list. The TUI writes it. A failed server is reported and the others stay available.
- `conversations/<workspace key>/` is one JSON file per conversation: title, mode, model, and the message list.
- `workspaces/<workspace key>/guidelines.md` holds preferences the user asked the agent to remember. Each line is approved before it is appended, then included in the system prompt.
- `workspaces/<workspace key>/permissions.json` holds always-allow rules and blocked program names. A rule skips the prompt only. Guards still run, and a rule cannot name a secret path or a command other than `pytest`. A blocked name refuses that program with no prompt.
- Skills are not stored here. Each workspace keeps its own in `<workspace>/.codie/skills/<name>/SKILL.md`. The prompt lists the description. `read_skill` loads the body. A skill from another folder is not visible.

The workspace key is a hash of the resolved folder path. `AGENTS.md` and `.codie/GUIDELINES.md` inside the workspace are read-only project instructions. `save_guideline` does not edit them.

## Modes

`agent` can edit files and run commands. Free commands run immediately. Other bare programs wait for Allow, Deny, or Block. `plan` can list, read, search, grep, and load skills. A write, an edit, a command, or an MCP call in plan mode returns `error: plan mode cannot change the workspace` and does not ask for approval.

The human can switch with `/mode` at any time. The model can call `switch_mode`. Agent to plan applies immediately. Plan to agent pauses on the approval modal. A denial stays in plan mode. The tool list is rebound before the next model call.

## Web search and MCP

`web_search` calls the provider named in `config.json`. SearXNG, Tavily, and the OpenAI Responses `web_search` tool are implemented. Exa, Firecrawl, and Brave are named slots the TUI does not offer until a backend is registered. With no provider selected, DuckDuckGo is the fallback.

MCP servers are stdio only. Their tools are registered as `mcp_<server>_<tool>`, hidden in plan mode, and approved like a write unless an always-allow rule names that server and tool.

## Why approval sits in code

The model can be talked into anything, including by a file it just read. A sentence in the prompt that says "ask first" is a wish. `interrupt()` runs in our process, before `write_text` or `subprocess.run`. The only way past it is the yes, always, or no returned by the CLI or the TUI, or an always-allow rule that same person already stored for this workspace. Guards still run first. A denial is a tool result, so the model can change its plan, and it cannot skip the check by claiming it already has permission.
