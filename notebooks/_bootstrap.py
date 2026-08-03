"""Make ``discretized_filter`` importable without an editable install.

Walks up from this file to find the repository root (the directory
containing ``pyproject.toml``) and prepends ``<root>/src`` to ``sys.path``
if the package cannot already be imported. Useful on clusters where the
project is not installed into the active environment.
"""

import sys
from pathlib import Path


def _bootstrap() -> None:
    try:
        import discretized_filter  # noqa: F401
        return
    except ImportError:
        pass

    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / 'pyproject.toml').is_file():
            src_dir = str(candidate / 'src')
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            return


_bootstrap()
