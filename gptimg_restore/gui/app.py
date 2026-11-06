"""데스크톱 창과 이미지 복원 브리지."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError

from gptimg_restore.io_utils import load_image, save_image
from gptimg_restore.pipeline import PRESETS, restore_array


class Cancelled(Exception):
    """단계 경계에서 복원 취소."""


_FILE_FILTER_LABELS = {"en": "Images", "ko": "이미지", "ja": "画像", "zh-CN": "图片"}


def _webview_storage_path(platform, env):
    """현재 사용자의 webview 저장소 경로를 만들고 반환."""
    if platform == "win32":
        storage_path = Path(env["LOCALAPPDATA"]) / "GPT Image Restore" / "webview"
    elif platform == "darwin":
        home = Path(env.get("HOME", Path.home()))
        storage_path = home / "Library" / "Application Support" / "GPT Image Restore" / "webview"
    elif platform == "linux":
        home = Path(env.get("HOME", Path.home()))
        data_home = env.get("XDG_DATA_HOME") or home / ".local" / "share"
        storage_path = Path(data_home) / "gpt-image-restore" / "webview"
    else:
        raise ValueError(f"Unsupported platform: {platform}")
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path


def _preview(rgb, alpha=None, thumbnail=False):
    """배열에서 브라우저용 PNG 생성."""
    pixels = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
    if alpha is not None:
        pixels = np.dstack((pixels, np.rint(np.clip(alpha, 0, 255)).astype(np.uint8)))
    image = Image.fromarray(pixels)
    if thumbnail:
        scale = min(1, 160 / min(image.size))
        image.thumbnail((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _PreviewServer(ThreadingHTTPServer):
    """임시 미리보기 이미지만 로컬 브라우저에 제공."""

    def __init__(self):
        self.images = {}
        super().__init__(("127.0.0.1", 0), _PreviewHandler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def put(self, image):
        key = uuid.uuid4().hex
        self.images[key] = image
        return f"http://127.0.0.1:{self.server_port}/{key}.png"

    def remove(self, url):
        self.images.pop(url.rsplit("/", 1)[1][:-4], None)


class _PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        image = self.server.images.get(self.path.removeprefix("/").removesuffix(".png"))
        if image is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(image)))
        self.end_headers()
        self.wfile.write(image)

    def log_message(self, format, *args):
        pass


class Bridge:
    """창과 독립적으로 사용할 수 있는 복원 및 파일 관리 API."""

    def __init__(self):
        self._entries = {}
        self._entries_lock = threading.RLock()
        self._window = None
        self._last_folder = str(Path.home())
        self._locale = "en"
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._jobs = {}
        self._commits = {}
        self._temp = tempfile.TemporaryDirectory(prefix="gpt-image-restore-")
        self._previews = _PreviewServer()

    def add_paths(self, paths):
        added, rejected = [], []
        for raw in paths:
            path = Path(raw)
            try:
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    raise ValueError("Unsupported format")
                loaded = load_image(path)
            except (OSError, ValueError, UnidentifiedImageError):
                rejected.append({"name": path.name, "reason": "unreadable"})
                continue
            item_id = uuid.uuid4().hex
            original_url = self._previews.put(_preview(loaded.rgb, loaded.alpha))
            thumbnail_url = self._previews.put(_preview(loaded.rgb, loaded.alpha, thumbnail=True))
            self._entries[item_id] = {"path": path, "loaded": loaded, "result": None,
                                     "urls": [original_url, thumbnail_url]}
            added.append({"id": item_id, "name": path.name, "width": loaded.size[0], "height": loaded.size[1],
                          "originalUrl": original_url, "thumbnailUrl": thumbnail_url})
        return {"added": added, "rejected": rejected}

    def add_bytes(self, name, data):
        path = Path(self._temp.name) / uuid.uuid4().hex / Path(name).name
        path.parent.mkdir()
        path.write_bytes(base64.b64decode(data))
        result = self.add_paths([str(path)])
        if result["added"]:
            self._entries[result["added"][0]["id"]]["owned"] = True
        else:
            path.unlink()
            path.parent.rmdir()
        return result

    def pick_files(self):
        import webview
        paths = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True,
                                               file_types=(f"{_FILE_FILTER_LABELS[self._locale]} (*.png;*.jpg;*.jpeg;*.webp)",))
        return self.add_paths(paths or [])

    def set_locale(self, code):
        if not isinstance(code, str) or code not in _FILE_FILTER_LABELS:
            raise ValueError("Unsupported language")
        self._locale = code

    def process(self, item_id, settings, token=None):
        event = threading.Event()
        job = {"event": event, "lock": threading.Lock(), "token": token or uuid.uuid4().hex,
               "progress": {"fraction": 0, "stage": "load"}}
        self._jobs[item_id] = job
        future = self._executor.submit(self._restore, self._entries[item_id], settings, job)
        try:
            future.result()
            return {"ready": True}
        except Cancelled:
            self._jobs.pop(item_id, None)
            return {"cancelled": True}
        except Exception:
            self._jobs.pop(item_id, None)
            raise

    def _restore(self, entry, settings, job):
        def progress(fraction, stage):
            if job["event"].is_set():
                raise Cancelled()
            stage = {"document_background": "grime", "sparkle_final": "sparkle"}.get(stage, stage)
            job["progress"] = {"fraction": fraction * 0.95, "stage": stage}

        overrides = dict(settings["strengths"])
        overrides["dots_mode"] = settings["dotsMode"]
        rgb, _, _ = restore_array(entry["loaded"], overrides=overrides, progress=progress,
                                  bit_depth=settings["bitDepth"])
        progress(1, "encode")
        preview = _preview(rgb, entry["loaded"].alpha)
        if job["event"].is_set():
            raise Cancelled()
        job["result"] = rgb
        job["preview"] = preview
        job["progress"] = {"fraction": 1, "stage": "encode"}

    def finish_process(self, item_id):
        job = self._jobs[item_id]
        with job["lock"]:
            if job["event"].is_set() or item_id not in self._entries:
                return {"cancelled": True}
            entry = self._entries[item_id]
            url = self._previews.put(job["preview"])
            job["previous_result"] = entry["result"]
            job["previous_url"] = entry["urls"].pop() if len(entry["urls"]) == 3 else None
            job["url"] = url
            job["entry"] = entry
            self._commits[job["token"]] = job
            entry["urls"].append(url)
            entry["result"] = job["result"]
            return {"restoredUrl": url}

    def ack_process(self, item_id, token):
        job = self._commits.pop(token, None)
        if job:
            with job["lock"]:
                if job.get("previous_url"):
                    self._previews.remove(job["previous_url"])
        current = self._jobs.get(item_id)
        if current and current["token"] == token:
            self._jobs.pop(item_id, None)

    def get_progress(self, item_id):
        job = self._jobs.get(item_id)
        return job["progress"] if job else None

    def cancel(self, item_id, token=None):
        job = self._jobs.get(item_id)
        if job and (token is None or job["token"] == token):
            with job["lock"]:
                job["event"].set()
                if "url" in job:
                    entry = job["entry"]
                    self._previews.remove(job.pop("url"))
                    entry["urls"].pop()
                    if job["previous_url"]:
                        entry["urls"].append(job.pop("previous_url"))
                    entry["result"] = job["previous_result"]

    def remove(self, item_id):
        job = self._jobs.get(item_id)
        if job and "url" not in job:
            self.cancel(item_id)
        with self._entries_lock:
            entry = self._entries.pop(item_id, None)
            if entry:
                for url in entry["urls"]:
                    self._previews.remove(url)
                if entry.get("owned"):
                    entry["path"].unlink()
                    entry["path"].parent.rmdir()

    def pick_export_target(self):
        import webview
        result = self._window.create_file_dialog(webview.FileDialog.FOLDER, directory=self._last_folder)
        if not result:
            return None
        self._last_folder = result[0]
        return {"kind": "folder", "path": result[0]}

    def export(self, ids, output, target):
        folder = Path(target["path"])
        with self._entries_lock:
            saved = 0
            for item_id in ids:
                entry = self._entries.get(item_id)
                if entry is None:
                    continue
                loaded = entry["loaded"]
                fmt = output["format"]
                save_image(folder / f"{entry['path'].stem}_restored.{fmt}", entry["result"], loaded.alpha,
                           image_format=fmt, quality=output["quality"],
                           bit_depth=output["bitDepth"] if fmt == "png" else 8, icc_profile=loaded.icc_profile)
                saved += 1
            return {"count": saved, "location": str(folder)}

    def close(self):
        for job in self._jobs.values():
            job["event"].set()
        self._executor.shutdown(wait=True)
        self._previews.shutdown()
        self._previews.server_close()
        self._temp.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", nargs=2, metavar=("IN", "OUT"), help=argparse.SUPPRESS)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="balanced", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.process:
        source, destination = args.process
        loaded = load_image(source)
        rgb, _, _ = restore_array(loaded, preset=args.preset)
        fmt = Path(destination).suffix.lower().lstrip(".")
        save_image(destination, rgb, loaded.alpha, image_format=fmt, icc_profile=loaded.icc_profile)
        return
    import webview
    bridge = Bridge()
    url = str(files("gptimg_restore.gui").joinpath("web", "index.html")) + "?backend=desktop"
    bridge._window = webview.create_window("GPT Image Restore", url, js_api=bridge, width=1200, height=800)
    try:
        webview.start(gui="gtk" if sys.platform.startswith("linux") else None, private_mode=False,
                      storage_path=str(_webview_storage_path(sys.platform, os.environ)))
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
