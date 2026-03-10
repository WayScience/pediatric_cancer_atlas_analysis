"""
Helper functions for virtual staining evaluation notebooks.
"""

import pathlib
import inspect

def find_git_root(start_path: pathlib.Path | None = None) -> pathlib.Path:
    """
    Recursively search upward for a Git repository root
    (directory containing .git).

    Works for both .py and .ipynb contexts.
    """
    if start_path is None:
        # Case 1: Called from a normal .py file
        if "__file__" in globals():
            start_path = pathlib.Path(__file__).resolve()
        else:
            # Case 2: Notebook / interactive
            # Try to infer from call stack
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            if module and hasattr(module, "__file__"):
                start_path = pathlib.Path(module.__file__).resolve()
            else:
                # Fallback: current working directory
                start_path = pathlib.Path.cwd()

    start_path = start_path if start_path.is_dir() else start_path.parent

    for parent in [start_path, *start_path.parents]:
        git_path = parent / ".git"
        if git_path.exists():
            return parent

    raise RuntimeError("No Git repository found.")
