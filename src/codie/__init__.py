"""codie: a workspace-jailed coding agent for the terminal."""


def main() -> None:
    from codie.cli import main as cli_main

    raise SystemExit(cli_main())
