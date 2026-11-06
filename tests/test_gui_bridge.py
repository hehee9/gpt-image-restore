"""데스크톱 브리지의 파일 및 복원 계약 검증."""

import base64
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from gptimg_restore.gui import app
from gptimg_restore.pipeline import PRESETS


@pytest.fixture
def bridge():
    instance = app.Bridge()
    yield instance
    instance.close()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (24, 24), (120, 100, 80)).save(path)
    return path


def test_add_process_export_and_remove(bridge, source, tmp_path, monkeypatch):
    result = bridge.add_paths([str(source), str(tmp_path / "bad.txt")])
    assert result["rejected"] == [{"name": "bad.txt", "reason": "unreadable"}]
    item = result["added"][0]
    assert (item["name"], item["width"], item["height"]) == ("sample.png", 24, 24)
    seen = []

    def restore(loaded, preset="balanced", overrides=None, progress=None, **options):
        assert preset == "balanced"
        assert overrides["dots"] == 0.4
        assert overrides["dots_mode"] == "strict"
        assert options["bit_depth"] == 16
        progress(0.25, "dots")
        seen.append(bridge.get_progress(item["id"])["fraction"])
        progress(1, "dither")
        seen.append(bridge.get_progress(item["id"])["fraction"])
        return loaded.rgb + 0.25, {}, {}

    monkeypatch.setattr(app, "restore_array", restore)
    strengths = dict(PRESETS["strong"], dots=0.4)
    assert bridge.process(item["id"], {"strengths": strengths, "dotsMode": "strict", "bitDepth": 16}) == {"ready": True}
    outcome = bridge.finish_process(item["id"])
    assert outcome["restoredUrl"].startswith("http://127.0.0.1:")
    assert seen == sorted(seen)
    out = bridge.export([item["id"]], {"format": "png", "bitDepth": 16, "quality": 90},
                        {"kind": "folder", "path": str(tmp_path)})
    assert out == {"count": 1, "location": str(tmp_path)}
    assert (tmp_path / "sample_restored.png").read_bytes()[24] == 16
    assert np.any(bridge._entries[item["id"]]["result"] % 1)
    bridge.export([item["id"]], {"format": "webp", "bitDepth": 8, "quality": 90},
                  {"kind": "folder", "path": str(tmp_path)})
    assert Image.open(tmp_path / "sample_restored.webp").format == "WEBP"
    bridge.remove(item["id"])
    assert item["id"] not in bridge._entries


def test_cancel_at_stage_boundary(bridge, source, monkeypatch):
    item_id = bridge.add_paths([str(source)])["added"][0]["id"]
    previous = np.full((24, 24, 3), 42, dtype=np.float32)
    bridge._entries[item_id]["result"] = previous
    started = threading.Event()
    resume = threading.Event()

    def restore(loaded, progress=None, **kwargs):
        started.set()
        resume.wait()
        progress(0.5, "dots")
        return loaded.rgb, {}, {}

    monkeypatch.setattr(app, "restore_array", restore)
    result = []
    thread = threading.Thread(target=lambda: result.append(bridge.process(
        item_id, {"strengths": PRESETS["safe"], "dotsMode": "conservative", "bitDepth": 8})))
    thread.start()
    assert started.wait(5)
    bridge.cancel(item_id)
    resume.set()
    thread.join(5)
    assert result == [{"cancelled": True}]
    assert bridge._entries[item_id]["result"] is previous


def test_cancel_before_result_commit(bridge, source, monkeypatch):
    item_id = bridge.add_paths([str(source)])["added"][0]["id"]
    previous = np.full((24, 24, 3), 42, dtype=np.float32)
    bridge._entries[item_id]["result"] = previous
    monkeypatch.setattr(app, "restore_array", lambda loaded, **kwargs: (loaded.rgb, {}, {}))
    assert bridge.process(item_id, {"strengths": PRESETS["safe"], "dotsMode": "conservative", "bitDepth": 8}) == {"ready": True}
    bridge.cancel(item_id)
    assert bridge.finish_process(item_id) == {"cancelled": True}
    assert bridge._entries[item_id]["result"] is previous


def test_cancel_after_commit_restores_previous_result(bridge, source, monkeypatch):
    item_id = bridge.add_paths([str(source)])["added"][0]["id"]
    entry = bridge._entries[item_id]
    previous = np.full((24, 24, 3), 42, dtype=np.float32)
    previous_url = bridge._previews.put(b"previous")
    entry["result"] = previous
    entry["urls"].append(previous_url)
    monkeypatch.setattr(app, "restore_array", lambda loaded, **kwargs: (loaded.rgb, {}, {}))
    settings = {"strengths": PRESETS["safe"], "dotsMode": "conservative", "bitDepth": 8}
    assert bridge.process(item_id, settings, "job-1") == {"ready": True}
    url = bridge.finish_process(item_id)["restoredUrl"]
    bridge.cancel(item_id, "job-1")
    bridge.ack_process(item_id, "job-1")
    assert entry["result"] is previous
    assert entry["urls"][-1] == previous_url
    assert previous_url.rsplit("/", 1)[1][:-4] in bridge._previews.images
    assert url.rsplit("/", 1)[1][:-4] not in bridge._previews.images


def test_dropped_files_are_removed_and_native_files_preserved(bridge, source):
    data = base64.b64encode(source.read_bytes()).decode("ascii")
    dropped = bridge.add_bytes("dropped.png", data)["added"][0]
    owned = bridge._entries[dropped["id"]]["path"]
    assert owned.exists()
    bridge.remove(dropped["id"])
    assert not owned.exists() and not owned.parent.exists()
    rejected = bridge.add_bytes("invalid.png", base64.b64encode(b"invalid").decode("ascii"))
    assert rejected["rejected"][0]["reason"] == "unreadable"
    assert list(Path(bridge._temp.name).iterdir()) == []
    native = bridge.add_paths([str(source)])["added"][0]
    bridge.remove(native["id"])
    assert source.exists()


def test_remove_waits_for_export(bridge, source, tmp_path, monkeypatch):
    item_id = bridge.add_bytes("dropped.png", base64.b64encode(source.read_bytes()).decode("ascii"))["added"][0]["id"]
    entry = bridge._entries[item_id]
    entry["result"] = entry["loaded"].rgb
    path = entry["path"]
    started, resume = threading.Event(), threading.Event()
    save = app.save_image

    def save_blocked(*args, **kwargs):
        started.set()
        assert resume.wait(5)
        save(*args, **kwargs)

    monkeypatch.setattr(app, "save_image", save_blocked)
    result = []
    exporter = threading.Thread(target=lambda: result.append(bridge.export(
        [item_id], {"format": "png", "bitDepth": 8, "quality": 90}, {"path": str(tmp_path)})))
    exporter.start()
    assert started.wait(5)
    remover = threading.Thread(target=lambda: bridge.remove(item_id))
    remover.start()
    assert path.exists()
    resume.set()
    exporter.join(5)
    remover.join(5)
    assert result == [{"count": 1, "location": str(tmp_path)}]
    assert (tmp_path / "dropped_restored.png").exists()
    assert not path.exists() and item_id not in bridge._entries


@pytest.mark.parametrize(("code", "label"), [
    ("en", "Images"),
    ("ko", "이미지"),
    ("ja", "画像"),
    ("zh-CN", "图片"),
])
def test_set_locale_changes_open_dialog_filter_label(bridge, monkeypatch, code, label):
    calls = []
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(FileDialog=SimpleNamespace(OPEN="open")))

    class Window:
        def create_file_dialog(self, dialog_type, **options):
            calls.append((dialog_type, options))
            return []

    bridge._window = Window()
    bridge.pick_files()
    bridge.set_locale(code)
    bridge.pick_files()

    assert calls == [
        ("open", {"allow_multiple": True, "file_types": ("Images (*.png;*.jpg;*.jpeg;*.webp)",)}),
        ("open", {"allow_multiple": True, "file_types": (f"{label} (*.png;*.jpg;*.jpeg;*.webp)",)}),
    ]


@pytest.mark.parametrize(("platform", "uses_xdg"), [
    ("win32", False),
    ("darwin", False),
    ("linux", True),
    ("linux", False),
])
def test_webview_storage_path_is_per_user_and_created(tmp_path, platform, uses_xdg):
    home = tmp_path / "home"
    env = {"HOME": str(home)}
    if platform == "win32":
        env["LOCALAPPDATA"] = str(tmp_path / "local-app-data")
        expected = Path(env["LOCALAPPDATA"]) / "GPT Image Restore" / "webview"
    elif platform == "darwin":
        expected = home / "Library" / "Application Support" / "GPT Image Restore" / "webview"
    elif uses_xdg:
        env["XDG_DATA_HOME"] = str(tmp_path / "xdg-data")
        expected = Path(env["XDG_DATA_HOME"]) / "gpt-image-restore" / "webview"
    else:
        expected = home / ".local" / "share" / "gpt-image-restore" / "webview"

    storage_path = app._webview_storage_path(platform, env)

    assert storage_path == expected
    assert storage_path.is_dir()


@pytest.mark.parametrize(("stage", "expected"), [
    ("document_background", "grime"),
    ("sparkle_final", "sparkle"),
])
def test_worker_runtime_maps_progress_stage(monkeypatch, stage, expected):
    worker_path = Path(app.__file__).parent / "web" / "worker-runtime.py"
    if not worker_path.exists():
        pytest.skip("web worker runtime is not included in this checkout")
    spec = importlib.util.spec_from_file_location("gptimg_restore_worker_runtime", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    real_path = Path

    class InputPath:
        def __init__(self, value):
            self.value = real_path(value)

        @property
        def suffix(self):
            return self.value.suffix

        def unlink(self):
            pass

    image = SimpleNamespace(rgb=np.zeros((1, 1, 3)), alpha=None, icc_profile=None)
    seen = []
    monkeypatch.setattr(worker, "Path", InputPath)
    monkeypatch.setattr(worker, "load_image", lambda source: image)
    monkeypatch.setattr(worker, "save_image", lambda *args, **kwargs: None)

    def restore_array(loaded, overrides, bit_depth, progress):
        progress(0.5, stage)
        return loaded.rgb, {}, {}

    monkeypatch.setattr(worker, "restore_array", restore_array)
    worker.process_image("item", "sample.png", json.dumps({
        "strengths": {}, "dotsMode": "strict", "bitDepth": 8,
    }), lambda fraction, current_stage: seen.append((fraction, current_stage)))

    assert seen == [(0.5, expected)]


@pytest.mark.parametrize("code", [None, 1, [], "fr", "EN", "zh-cn"])
def test_set_locale_rejects_unsupported_codes(bridge, code):
    with pytest.raises(ValueError, match="Unsupported language"):
        bridge.set_locale(code)
