"""Command-line interface for thinkpack."""

import argparse
import sys
from pathlib import Path

from thinkpack.skill import Tool, generate, write


def _skill(args: argparse.Namespace) -> None:
    """Handle the 'skill' subcommand."""
    tool = Tool(args.tool) if args.tool else None
    directory = Path(args.dir) if args.dir else None

    if tool is None:
        # no tool specified — print the raw llms.txt content to stdout
        content, _ = generate(tool=None)
        print(content)
        return

    path = write(tool=tool, directory=directory)
    print(f"Skill file written to {path}")


def main() -> None:
    """Entry point for the thinkpack CLI."""
    parser = argparse.ArgumentParser(
        prog="thinkpack",
        description="ThinkPack command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # skill subcommand — generates an agent skill file embedding llms.txt
    skill_parser = subparsers.add_parser(
        "skill",
        help="Generate an agent skill file embedding thinkpack's llms.txt context.",
    )
    skill_parser.add_argument(
        "--tool",
        choices=[t.value for t in Tool],
        default=None,
        help=(
            "Agent tool to target (claude, cursor, windsurf)."
            " If omitted, prints the raw llms.txt content to stdout."
        ),
    )
    skill_parser.add_argument(
        "--dir",
        default=None,
        metavar="PATH",
        help=(
            "Base directory to write the skill file into."
            " Defaults to the current working directory."
            " Ignored when --tool is not specified."
        ),
    )

    args = parser.parse_args()

    if args.command == "skill":
        _skill(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
