"""
    @file:              test_style.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       Guards the one formatting rule the codebase actually keeps: no source line runs past
                        MAX_LINE_LENGTH characters. The docstrings here are long and carry real explanation, so
                        an over-long line is easy to introduce and easy to miss in review.

                        This lives in the test suite rather than in a linter so that it needs no extra
                        dependency and runs everywhere the rest of the suite runs.
"""

from pathlib import Path
from typing import List, Tuple

import pytest

MAX_LINE_LENGTH = 120

ROOT = Path(__file__).resolve().parent.parent
SEARCHED_DIRECTORIES = ("jetgo", "tests", "examples", "images")


def _python_files() -> List[Path]:
    """Every Python file the project owns, skipping build artifacts and virtual environments."""
    files = []
    for directory in SEARCHED_DIRECTORIES:
        files.extend(sorted((ROOT / directory).rglob("*.py")))

    return files


def _over_long_lines(path: Path) -> List[Tuple[int, int]]:
    """The (line number, length) of every line in ``path`` that runs past the limit."""
    lines = path.read_text(encoding="utf-8").splitlines()

    return [(n, len(line)) for n, line in enumerate(lines, start=1) if len(line) > MAX_LINE_LENGTH]


class TestLineLength:

    def test_the_project_has_python_files_to_check(self):
        """Guards against the glob silently matching nothing and the check passing vacuously."""
        assert len(_python_files()) > 10

    @pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(ROOT)))
    def test_no_line_runs_past_the_limit(self, path):
        offenders = _over_long_lines(path)

        assert not offenders, "\n".join(
            f"{path.relative_to(ROOT)}:{n} is {length} characters, limit is {MAX_LINE_LENGTH}"
            for n, length in offenders
        )
