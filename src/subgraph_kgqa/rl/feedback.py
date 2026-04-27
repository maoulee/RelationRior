from __future__ import annotations

from subgraph_kgqa.legacy import load_repo_module

_feedback = load_repo_module("plug_v12_feedback", "plug_v12_feedback.py")

__all__ = [name for name in vars(_feedback) if not name.startswith("_")]

for _name in __all__:
    globals()[_name] = getattr(_feedback, _name)


def __getattr__(name: str):
    return getattr(_feedback, name)
