from __future__ import annotations

from subgraph_kgqa.legacy import load_repo_module

_environment = load_repo_module("v10_environment", "v10_environment.py")
_feedback = load_repo_module("plug_v12_feedback", "plug_v12_feedback.py")
_legacy_import_error = None
try:
    _legacy = load_repo_module("plug_v11", "plug_v11.py")
except Exception as exc:  # pragma: no cover - environment dependent
    from . import legacy_fallback as _legacy

    _legacy_import_error = exc

_modules = (_environment, _feedback, _legacy)


def _public_names(module) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is not None:
        return [name for name in names if not name.startswith("_")]
    return [name for name in vars(module) if not name.startswith("_")]


__all__: list[str] = []
for _module in _modules:
    for _name in _public_names(_module):
        globals()[_name] = getattr(_module, _name)
        if _name not in __all__:
            __all__.append(_name)


LEGACY_PLUGIN_MODULE = _legacy
FEEDBACK_MODULE = _feedback
ENVIRONMENT_MODULE = _environment
LEGACY_PLUGIN_IMPORT_ERROR = _legacy_import_error


def __getattr__(name: str):
    for module in (_legacy, _feedback, _environment):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
