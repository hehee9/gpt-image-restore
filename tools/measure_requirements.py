"""Measure restoration wall time and peak memory for system requirements."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

try:
    import psutil
except ImportError as error:
    raise SystemExit(
        "psutil is required to measure system requirements. Install it with `python -m pip install psutil`."
    ) from error

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESTORE_SCRIPT = REPO_ROOT / "restore_gpt_image.py"
PRESETS = ("safe", "balanced", "strong")
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _positive_megapixels(value: str) -> float:
    """Parse a positive megapixel target."""
    try:
        megapixels = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must be a number in megapixels") from error
    if not math.isfinite(megapixels) or megapixels <= 0:
        raise argparse.ArgumentTypeError("size must be a finite megapixel value greater than zero")
    return megapixels


def _positive_integer(value: str) -> int:
    """Parse a positive integer."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("count must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("count must be at least 1")
    return number


def _core_count(value: str) -> int:
    """Parse a core count, where zero selects all available logical CPUs."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("core count must be an integer. Use 0 for all cores") from error
    if number < 0:
        raise argparse.ArgumentTypeError("core count must be zero or a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Measure image restore time and peak memory to estimate system requirements."
    )
    parser.add_argument("--image", required=True, type=Path, help="source image to resize and restore")
    parser.add_argument(
        "--sizes", nargs="+", type=_positive_megapixels, default=(1.0, 4.0, 16.0),
        metavar="MP", help="image sizes to measure in megapixels (default: 1 4 16)",
    )
    parser.add_argument(
        "--presets", nargs="+", choices=PRESETS, default=PRESETS,
        help="presets to measure (default: safe balanced strong)",
    )
    parser.add_argument(
        "--cores", nargs="+", type=_core_count, default=(0,), metavar="N",
        help="logical CPU cores per run, 0 for all cores (default: 0)",
    )
    parser.add_argument("--repeat", type=_positive_integer, default=1, help="repetitions per combination (default: 1)")
    parser.add_argument(
        "--out", type=Path, default=Path("requirements_measurements.json"),
        help="path to the JSON results file (default: requirements_measurements.json)",
    )
    parser.add_argument("--keep-temp", action="store_true", help="keep resized images and run output directory")
    return parser


def _cpu_model() -> str | None:
    """Return the CPU model when the host exposes it."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            result = None
        if result is not None and result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    elif sys.platform == "win32":
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                processor = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
                if processor:
                    return processor
        except OSError:
            pass

    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            label, separator, value = line.partition(":")
            if separator and label.strip().lower() in ("model name", "hardware"):
                return value.strip() or None
    except OSError:
        pass
    return platform.processor().strip() or None


def _system_info() -> tuple[dict, int, list[int] | None]:
    """Collect host metadata and the available CPU affinity mask."""
    logical_cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    if sys.platform == "darwin":
        available_cpus = None
    else:
        available_cpus = psutil.Process().cpu_affinity()

    if available_cpus is not None:
        logical_cpu_count = max(logical_cpu_count, len(available_cpus))

    info = {
        "platform": {
            "os": platform.system(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu_model": _cpu_model(),
        "logical_cpu_count": logical_cpu_count,
        "total_ram_mib": round(psutil.virtual_memory().total / (1024 ** 2), 1),
        "python_version": platform.python_version(),
    }
    return info, logical_cpu_count, available_cpus


def _dimensions(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """Scale dimensions to a megapixel target without exceeding its pixel count."""
    target_pixels = max(1, round(megapixels * 1_000_000))
    scale = math.sqrt(target_pixels / (width * height))
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))

    if target_width * target_height > target_pixels:
        if target_width >= target_height:
            target_width = max(1, target_pixels // target_height)
        else:
            target_height = max(1, target_pixels // target_width)

    return target_width, target_height


def _prepare_image(source: Path, megapixels: float, path: Path) -> tuple[int, int]:
    """Resize the source image with an aspect-preserving OpenCV interpolation."""
    image_bytes = np.fromfile(source, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read source image: {source}")

    height, width = image.shape[:2]
    target_width, target_height = _dimensions(width, height, megapixels)
    interpolation = cv2.INTER_LANCZOS4 if target_width * target_height > width * height else cv2.INTER_AREA
    if (target_width, target_height) != (width, height):
        image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    encoded_ok, encoded = cv2.imencode(".png", image)
    if not encoded_ok:
        raise OSError(f"cannot encode resized PNG: {path}")
    encoded.tofile(path)
    return target_width, target_height


def _read_linux_peak_rss(pid: int) -> int | None:
    """Read Linux VmHWM in bytes when procfs exposes it."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return None


def _run_child(command: list[str], requested_cores: int, core_count: int,
               available_cpus: list[int] | None) -> dict:
    """Run one CLI child while measuring time and process peak memory."""
    env = os.environ.copy()
    if requested_cores == 0:
        for name in THREAD_ENV_VARS:
            env.pop(name, None)
        affinity_cpus = None
    else:
        for name in THREAD_ENV_VARS:
            env[name] = str(core_count)
        affinity_cpus = available_cpus[:core_count] if available_cpus is not None else None

    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_process = psutil.Process(child.pid)
    if affinity_cpus is not None:
        child_process.cpu_affinity(affinity_cpus)

    sampled_peak = None
    windows_peak = None
    linux_peak = None
    while child.poll() is None:
        try:
            memory = child_process.memory_info()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            memory = None
        if memory is not None:
            sampled_peak = max(sampled_peak or 0, memory.rss)
            if platform.system() == "Windows":
                windows_peak = max(windows_peak or 0, memory.peak_wset)
        if sys.platform.startswith("linux"):
            current_linux_peak = _read_linux_peak_rss(child.pid)
            if current_linux_peak is not None:
                linux_peak = current_linux_peak
        time.sleep(0.05)

    try:
        final_memory = child_process.memory_info()
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        final_memory = None
    if final_memory is not None:
        sampled_peak = max(sampled_peak or 0, final_memory.rss)
        if platform.system() == "Windows":
            windows_peak = max(windows_peak or 0, final_memory.peak_wset)
    if sys.platform.startswith("linux"):
        current_linux_peak = _read_linux_peak_rss(child.pid)
        if current_linux_peak is not None:
            linux_peak = current_linux_peak

    if platform.system() == "Windows":
        peak_bytes = windows_peak if windows_peak is not None else sampled_peak
    elif sys.platform.startswith("linux"):
        peak_bytes = linux_peak if linux_peak is not None else sampled_peak
    else:
        peak_bytes = sampled_peak

    return {
        "wall_s": round(time.perf_counter() - started, 3),
        "peak_rss_mib": round(peak_bytes / (1024 ** 2), 2) if peak_bytes is not None else None,
        "exit_code": child.returncode,
        "status": "success" if child.returncode == 0 else "failed",
        "affinity_cpu_ids": affinity_cpus,
    }


def _format_value(value: float | int | None, decimals: int) -> str:
    """Format a measurement value for the Markdown table."""
    return "-" if value is None else f"{value:.{decimals}f}"


def _print_table(records: list[dict]) -> None:
    """Print run records as a Markdown table."""
    columns = ("Size", "Width×height", "Preset", "Core count", "Runtime (s)", "Peak memory (MiB)", "Exit code", "Status")
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for record in records:
        size = record["size_label"]
        width = record["image_size"]["width"]
        height = record["image_size"]["height"]
        cells = (
            size,
            f"{width}×{height}",
            record["preset"],
            str(record["cores"]),
            _format_value(record["wall_s"], 3),
            _format_value(record["peak_rss_mib"], 2),
            "-" if record["exit_code"] is None else str(record["exit_code"]),
            {"success": "Success", "failed": "Failed", "affinity_unsupported": "Core limit unsupported"}[record["status"]],
        )
        print("| " + " | ".join(cells) + " |")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace,
                   available_cpus: list[int] | None, logical_cpu_count: int) -> None:
    """Validate paths and selected CPU counts before creating temporary files."""
    args.image = args.image.expanduser().resolve()
    if not args.image.is_file():
        parser.error(f"source image not found: {args.image}")

    args.out = args.out.expanduser().resolve()
    if args.out == args.image:
        parser.error("--out must not point to the source image")
    if available_cpus is not None:
        max_cores = len(available_cpus)
    else:
        max_cores = logical_cpu_count
    for core_count in args.cores:
        if core_count > max_cores:
            parser.error(f"--cores {core_count}: only {max_cores} logical CPU cores are available")


def main(argv: list[str] | None = None) -> int:
    """Prepare scaled inputs, run each requested combination, and save results."""
    parser = _parser()
    args = parser.parse_args(argv)
    system_info, logical_cpu_count, available_cpus = _system_info()
    _validate_args(parser, args, available_cpus, logical_cpu_count)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp_context = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="gpt-image-restore-measure-"))
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="gpt-image-restore-measure-")
        temp_root = Path(temp_context.name)

    records = []
    try:
        input_dir = temp_root / "inputs"
        input_dir.mkdir()
        resized_inputs = []
        for index, megapixels in enumerate(args.sizes, start=1):
            image_path = input_dir / f"input_{index}.png"
            width, height = _prepare_image(args.image, megapixels, image_path)
            resized_inputs.append((megapixels, image_path, width, height))

        for size_index, (megapixels, image_path, width, height) in enumerate(resized_inputs, start=1):
            size_label = f"{megapixels:g} MP"
            for preset in args.presets:
                for requested_cores in args.cores:
                    if requested_cores == 0:
                        core_count = len(available_cpus) if available_cpus is not None else logical_cpu_count
                    else:
                        core_count = requested_cores
                    for repeat_index in range(1, args.repeat + 1):
                        output_dir = temp_root / "outputs" / (
                            f"size-{size_index}-preset-{preset}-cores-{core_count}-repeat-{repeat_index}"
                        )
                        output_dir.mkdir(parents=True, exist_ok=True)
                        command = [
                            sys.executable,
                            str(RESTORE_SCRIPT),
                            str(image_path),
                            "-o",
                            str(output_dir),
                            "--preset",
                            preset,
                        ]

                        if requested_cores != 0 and available_cpus is None:
                            run = {
                                "wall_s": None,
                                "peak_rss_mib": None,
                                "exit_code": None,
                                "status": "affinity_unsupported",
                                "affinity_cpu_ids": None,
                            }
                        else:
                            run = _run_child(
                                command,
                                requested_cores,
                                core_count,
                                available_cpus,
                            )

                        records.append({
                            **system_info,
                            "size_label": size_label,
                            "image_size": {"width": width, "height": height},
                            "preset": preset,
                            "cores": core_count,
                            "wall_s": run["wall_s"],
                            "peak_rss_mib": run["peak_rss_mib"],
                            "exit_code": run["exit_code"],
                            "status": run["status"],
                            "affinity_cpu_ids": run["affinity_cpu_ids"],
                        })

        args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _print_table(records)
        if args.keep_temp:
            print(f"Temporary files kept in: {temp_root}", file=sys.stderr)
    finally:
        if temp_context is not None:
            temp_context.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
