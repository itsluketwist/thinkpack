"""Generate agent skill files that embed thinkpack's llms.txt context."""

import importlib.resources
from enum import StrEnum
from pathlib import Path


class Tool(StrEnum):
    """Supported agent tools for skill file generation."""

    CLAUDE = "claude"
    CURSOR = "cursor"
    WINDSURF = "windsurf"


# destination path for each tool's skill file, relative to the project root
_TOOL_PATHS: dict[Tool, str] = {
    Tool.CLAUDE: ".claude/commands/thinkpack.md",
    Tool.CURSOR: ".cursor/rules/thinkpack.mdc",
    Tool.WINDSURF: ".windsurf/rules/thinkpack.md",
}

# tool-specific frontmatter or preamble prepended before the llms.txt content
_TOOL_HEADERS: dict[Tool, str] = {
    Tool.CLAUDE: (
        "Use the following context about the thinkpack library"
        " to assist with: $ARGUMENTS\n\n---\n\n"
    ),
    Tool.CURSOR: (
        "---\n"
        "description: ThinkPack library context for reasoning model utilities\n"
        "alwaysApply: false\n"
        "---\n\n"
    ),
    Tool.WINDSURF: (
        "---\n"
        "trigger: manual\n"
        "description: ThinkPack library context for reasoning model utilities\n"
        "---\n\n"
    ),
}


def _load_llms_txt() -> str:
    """Load the bundled llms.txt content from the package data directory.

    Returns the full text content of llms.txt as a string.
    """
    ref = importlib.resources.files("thinkpack.data").joinpath("llms.txt")
    return ref.read_text(encoding="utf-8")


def generate(
    tool: Tool | None = None,
    directory: Path | None = None,
) -> tuple[str, Path | None]:
    """Generate skill file content and the path it should be written to.

    Loads the bundled llms.txt content and prepends tool-specific frontmatter
    or a preamble. If tool is None, the raw llms.txt content is returned with
    no write path — suitable for printing to stdout.

    Returns a (content, path) tuple; path is None when tool is None.
    """
    content = _load_llms_txt()

    if tool is None:
        # no tool specified — caller can print the raw content directly
        return content, None

    # prepend the tool-specific header before the shared llms.txt body
    content = _TOOL_HEADERS[tool] + content
    path = (directory or Path.cwd()) / _TOOL_PATHS[tool]
    return content, path


def write(
    tool: Tool,
    directory: Path | None = None,
) -> Path:
    """Write the skill file for the given tool to the appropriate path.

    Creates any missing parent directories as needed.

    Returns the path the file was written to.
    """
    content, path = generate(tool=tool, directory=directory)
    if path is None:
        raise RuntimeError("generate() returned no path for a non-None tool")

    # ensure the destination directory exists before writing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
