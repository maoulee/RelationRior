from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from .paths import project_root, src_root


def ensure_import_paths(*extra_paths: Path) -> None:
    candidates = [project_root(), src_root(), *extra_paths]
    for path in candidates:
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


@lru_cache(maxsize=None)
def load_module(module_name: str, module_path: str | Path) -> ModuleType:
    path = Path(module_path).resolve()
    ensure_import_paths(path.parent)

    spec = importlib.util.spec_from_file_location(
        f"subgraph_kgqa_legacy_{module_name}",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


def load_repo_module(module_name: str, relative_path: str) -> ModuleType:
    return load_module(module_name, project_root() / relative_path)
