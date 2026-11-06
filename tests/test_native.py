from gptimg_restore._native import load_native


def test_native_loading_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GPTIMG_RESTORE_PURE_PYTHON", "1")

    assert load_native("_native_missing") is None


def test_missing_native_module_returns_none(monkeypatch):
    monkeypatch.delenv("GPTIMG_RESTORE_PURE_PYTHON", raising=False)

    assert load_native("_native_missing") is None
