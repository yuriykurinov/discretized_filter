"""Filesystem paths shared across the package.

Resolves the repository root relative to this file so that code works the
same way regardless of the current working directory (scripts, notebooks,
or an installed package).
"""

from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file until a directory containing ``pyproject.toml``
    is found. Fall back to three levels up if it cannot be located.
    """
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / 'pyproject.toml').is_file():
            return candidate
    return here.parents[2]


PROJECT_ROOT = _find_project_root()

RESULTS_DIR = PROJECT_ROOT / 'results'
CONFIGS_DIR = PROJECT_ROOT / 'configs'


def saved_path_dir(exp_id) -> Path:
    """Directory where the results of experiment ``exp_id`` are saved."""
    return PROJECT_ROOT / f'saved_path_{exp_id}'
