"""Optional compiled accelerators."""
import importlib
import os


def load_native(name):
    """Return the compiled submodule ``name`` or None when it is unavailable or disabled."""
    if os.environ.get("GPTIMG_RESTORE_PURE_PYTHON") == "1":
        return None
    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ImportError:
        return None
