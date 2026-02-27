"""PyTMatrix package."""

from pathlib import Path
from typing import Sequence


def run_tests(pytest_args: Sequence[str] | None = None) -> int:
    """Run the bundled test suite with pytest.

    Parameters
    ----------
    pytest_args : sequence of str, optional
        Extra arguments passed to ``pytest``.

    Returns
    -------
    int
        Pytest exit code.
    """
    try:
        import pytest
    except ImportError as exc:
        raise RuntimeError(
            "pytest is required to run tests. Install it with `pip install pytest`."
        ) from exc

    tests_dir = Path(__file__).resolve().parent / "tests"
    args = [str(tests_dir)]
    if pytest_args:
        args.extend(pytest_args)
    return int(pytest.main(args))
