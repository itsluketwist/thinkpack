"""Tests for thinkpack._cli — basic coverage of CLI entry point and skill handler."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from thinkpack._cli import _skill, main
from thinkpack._skill import Tool


class TestSkillHandler:
    """Tests for the _skill() internal handler."""

    def test_no_tool_prints_raw_content(self, capsys) -> None:
        """Without --tool, the raw llms.txt content is printed to stdout."""
        args = MagicMock()
        args.tool = None
        args.dir = None

        with patch(
            "thinkpack._cli.generate", return_value=("raw content", None)
        ) as mock_gen:
            _skill(args)
            mock_gen.assert_called_once_with(tool=None)

        assert "raw content" in capsys.readouterr().out

    def test_with_tool_calls_write_and_prints_path(self, capsys, tmp_path) -> None:
        """With --tool, write() is called and the returned path is printed."""
        args = MagicMock()
        args.tool = "claude"
        args.dir = str(tmp_path)

        expected_path = tmp_path / ".claude/commands/thinkpack.md"
        with patch("thinkpack._cli.write", return_value=expected_path) as mock_write:
            _skill(args)
            mock_write.assert_called_once_with(
                tool=Tool.CLAUDE,
                directory=tmp_path,
            )

        assert str(expected_path) in capsys.readouterr().out

    def test_with_tool_no_dir_passes_none_to_write(self) -> None:
        """When --dir is absent, write() receives directory=None."""
        args = MagicMock()
        args.tool = "cursor"
        args.dir = None

        fake_path = Path("/fake/thinkpack.mdc")
        with patch("thinkpack._cli.write", return_value=fake_path) as mock_write:
            _skill(args)
            mock_write.assert_called_once_with(tool=Tool.CURSOR, directory=None)

    def test_all_tool_values_are_accepted(self, tmp_path) -> None:
        """Each Tool enum value can be passed as --tool without errors."""
        for tool in Tool:
            args = MagicMock()
            args.tool = tool.value
            args.dir = str(tmp_path)
            fake_path = tmp_path / "out.md"
            with patch("thinkpack._cli.write", return_value=fake_path):
                _skill(args)  # should not raise


class TestMain:
    """Tests for the main() CLI entry point and argument parsing."""

    def test_no_subcommand_exits_with_code_1(self) -> None:
        """Calling main() with no subcommand exits with code 1."""
        with patch("sys.argv", ["thinkpack"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

    def test_skill_subcommand_dispatches_to_skill_handler(self) -> None:
        """The 'skill' subcommand is routed to _skill()."""
        with patch("sys.argv", ["thinkpack", "skill"]):
            with patch("thinkpack._cli._skill") as mock_skill:
                main()
                mock_skill.assert_called_once()

    def test_skill_tool_arg_is_parsed(self) -> None:
        """--tool is parsed and forwarded to _skill() via the args namespace."""
        with patch("sys.argv", ["thinkpack", "skill", "--tool", "claude"]):
            with patch("thinkpack._cli._skill") as mock_skill:
                main()
                args = mock_skill.call_args[0][0]
                assert args.tool == "claude"

    def test_skill_dir_arg_is_parsed(self, tmp_path) -> None:
        """--dir is parsed and forwarded to _skill() via the args namespace."""
        with patch("sys.argv", ["thinkpack", "skill", "--dir", str(tmp_path)]):
            with patch("thinkpack._cli._skill") as mock_skill:
                main()
                args = mock_skill.call_args[0][0]
                assert args.dir == str(tmp_path)
