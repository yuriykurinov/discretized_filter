"""Filesystem paths shared across the package."""

from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / 'pyproject.toml').is_file():
            return candidate
    return here.parents[2]


PROJECT_ROOT = _find_project_root()

RESULTS_DIR = PROJECT_ROOT / 'results'
CONFIGS_DIR = PROJECT_ROOT / 'configs'


def saved_path_dir(exp_id) -> Path:
    return PROJECT_ROOT / f'saved_path_{exp_id}'
