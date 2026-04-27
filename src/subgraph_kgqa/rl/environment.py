from __future__ import annotations

from subgraph_kgqa.legacy import load_repo_module

_environment = load_repo_module("v10_environment", "v10_environment.py")

__all__ = [name for name in vars(_environment) if not name.startswith("_")]

for _name in __all__:
    globals()[_name] = getattr(_environment, _name)


def __getattr__(name: str):
    return getattr(_environment, name)
