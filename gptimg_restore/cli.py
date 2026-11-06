"""Command-line entry and stage orchestration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import dither
from .pipeline import PRESETS as PRESET_STRENGTHS, STAGE_ORDER, StageUnavailableError, _load_stage, restore_array
from .io_utils import load_image, save_image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_strengths(
    preset: str,
    dots: float | None,
    periodic: float | None,
    grime: float | None,
    surface: float | None,
    mottle: float | None = None,
    sparkle: float | None = None,
    lattice: float | None = None,
    dither: float | None = None,
    contour: float | None = None,
    defocus: float | None = None,
    cel_fill: float | None = None,
) -> dict[str, float]:
    """Resolve preset defaults and explicit per-stage strengths."""
    strengths = dict(PRESET_STRENGTHS[preset])
    for name, value in (("dots", dots), ("periodic", periodic), ("grime", grime),
                        ("mottle", mottle), ("surface", surface), ("sparkle", sparkle),
                        ("lattice", lattice), ("dither", dither), ("contour", contour),
                        ("defocus", defocus), ("cel_fill", cel_fill)):
        if value is not None:
            strengths[name] = value
    return strengths


def _parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Reduce generation noise in GPT Image output.")
    parser.add_argument("inputs", nargs="+", help="input images or image folders")
    parser.add_argument("-o", "--outdir", help="folder for all results")
    parser.add_argument("--format", choices=("png", "jpg", "webp"), default="png", help="output format (default: png)")
    parser.add_argument("--quality", type=int, default=90, help="JPG/WebP quality, 0-100 (default: 90)")
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=8, help="PNG output bit depth (default: 8)")
    parser.add_argument("--preset", choices=tuple(PRESET_STRENGTHS), default="balanced", help="restoration preset")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="device to run on (default: cpu)")
    parser.add_argument("--dots", type=float, help="strength for removing bright dots")
    parser.add_argument("--dots-mode", choices=("conservative", "strict"), default="conservative", help="bright dot detection range: conservative is narrow, strict is wide")
    parser.add_argument("--sparkle", type=float, help="strength for removing bright and dark sparkle on textured surfaces")
    parser.add_argument("--periodic", type=float, help="strength for removing periodic patterns such as repeating stripes and grids")
    parser.add_argument("--grime", type=float, help="strength for removing crosshatch on surfaces and document backgrounds")
    parser.add_argument("--mottle", type=float, help="strength for removing medium-sized blotches on flat backgrounds")
    parser.add_argument("--lattice", type=float, help="strength for removing fine grids on flat backgrounds, excluding skin")
    parser.add_argument("--surface", type=float, help="strength for cleaning up fine color grain on surfaces")
    parser.add_argument("--contour", type=float, help="strength for removing ripples along smooth object edges")
    parser.add_argument("--defocus", type=float, help="strength for softening dark dots in blurred backgrounds")
    parser.add_argument("--cel-fill", type=float, help="strength for evening out uneven fills and boundaries in line-art illustrations")
    parser.add_argument("--dither", type=float, help="dithering strength that reduces banding when restored areas are saved as 8-bit")
    parser.add_argument("--analyze", action="store_true", help="run analysis only and print a JSON report")
    parser.add_argument("--report", action="store_true", help="save a JSON report for each image")
    parser.add_argument("--save-diff", action="store_true", help="save a before/after difference image")
    return parser


def _input_jobs(inputs: list[str], outdir: str | None) -> list[tuple[Path, Path]]:
    """Expand paths into image and output-directory pairs."""
    jobs: list[tuple[Path, Path]] = []
    explicit_outdir = Path(outdir) if outdir else None
    for raw in inputs:
        source = Path(raw)
        if source.is_file():
            if source.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported input format: {source}")
            target_dir = explicit_outdir or source.parent / "restored"
            jobs.append((source, target_dir))
            continue
        if not source.is_dir():
            raise ValueError(f"Input path not found: {source}")

        target_dir = explicit_outdir or source.parent / "restored"
        source_resolved = source.resolve()
        target_resolved = target_dir.resolve()
        try:
            target_resolved.relative_to(source_resolved)
            excluded_dir = target_resolved if target_resolved != source_resolved else None
        except ValueError:
            excluded_dir = None
        files = []
        for candidate in sorted(source.rglob("*")):
            if candidate.suffix.lower() not in IMAGE_SUFFIXES or not candidate.is_file():
                continue
            if excluded_dir is not None:
                try:
                    candidate.resolve().relative_to(excluded_dir)
                    continue
                except ValueError:
                    pass
            files.append(candidate)
        if not files:
            raise ValueError(f"No supported images found in folder: {source}")
        jobs.extend((candidate, target_dir) for candidate in files)
    return jobs


def _path_key(path: Path) -> str:
    """Normalize a path for collision checks."""
    return os.path.normcase(str(path.resolve()))


def _planned_outputs(path: Path, outdir: Path, args: argparse.Namespace) -> list[Path]:
    """List all output paths for one input before processing begins."""
    outputs = []
    stem = path.stem
    if not args.analyze:
        extension = "jpg" if args.format == "jpg" else args.format
        outputs.append(outdir / f"{stem}_restored.{extension}")
        if args.save_diff:
            outputs.append(outdir / f"{stem}_diff.png")
    if args.report:
        outputs.append(outdir / f"{stem}_report.json")
    return outputs


def _preflight_outputs(jobs: list[tuple[Path, Path]], args: argparse.Namespace) -> None:
    """Reject output collisions with inputs or other planned outputs."""
    input_paths = {_path_key(path): path for path, _ in jobs}
    output_owners = {}
    conflicts = []
    for input_path, outdir in jobs:
        for output_path in _planned_outputs(input_path, outdir, args):
            key = _path_key(output_path)
            conflicting_input = input_paths.get(key)
            if conflicting_input is not None:
                conflicts.append(f"{output_path} (output of {input_path}) overlaps input file {conflicting_input}")
            previous_owner = output_owners.get(key)
            if previous_owner is not None:
                conflicts.append(f"{output_path} is used as output for both {previous_owner} and {input_path}")
            else:
                output_owners[key] = input_path
    if conflicts:
        details = "\n".join(f"- {conflict}" for conflict in conflicts)
        raise ValueError(f"Output path conflicts found; nothing was processed:\n{details}")


def _write_json(path: Path, report: dict) -> None:
    """Write an indented UTF-8 JSON report."""
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _process_one(
    path: Path,
    outdir: Path,
    args: argparse.Namespace,
    strengths: dict[str, float],
    analysis_modules: dict,
    backend=None,
) -> dict:
    """Analyze or restore one image and write requested outputs."""
    started = time.perf_counter()
    load_started = time.perf_counter()
    loaded = load_image(path)
    load_seconds = time.perf_counter() - load_started
    output_stem = path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    if args.analyze:
        analysis_reports = {}
        analysis_timings = {}
        for name in ("dots", "sparkle", "periodic", "grime", "mottle", "lattice", "contour", "defocus", "celfill"):
            stage_started = time.perf_counter()
            if name == "dots":
                analysis_reports[name] = analysis_modules[name].analyze(loaded.rgb, mode=args.dots_mode)
            elif name == "celfill":
                analysis_reports["cel_fill"] = analysis_modules[name].analyze(
                    loaded.rgb, context_rgb=loaded.rgb)
            elif name in ("periodic", "grime", "mottle", "lattice", "defocus") and backend is not None:
                analysis_reports[name] = analysis_modules[name].analyze(loaded.rgb, backend=backend)
            else:
                analysis_reports[name] = analysis_modules[name].analyze(loaded.rgb)
            analysis_timings["cel_fill" if name == "celfill" else name] = time.perf_counter() - stage_started
        report = {
            "input": str(path),
            "size": {"width": loaded.size[0], "height": loaded.size[1]},
            "mode": loaded.mode,
            "bit_depth": loaded.bit_depth,
            "stages": analysis_reports,
            "timings": {
                "load_seconds": load_seconds,
                "stages_seconds": analysis_timings,
                "total_seconds": time.perf_counter() - started,
            },
        }
        if args.report:
            _write_json(outdir / f"{output_stem}_report.json", report)
        return report

    rgb, stage_reports, stage_timings = restore_array(
        loaded, preset=args.preset, overrides={**strengths, "dots_mode": args.dots_mode},
        device=args.device, backend=backend, bit_depth=args.bit_depth,
    )

    extension = "jpg" if args.format == "jpg" else args.format
    output_path = outdir / f"{output_stem}_restored.{extension}"
    save_image(
        output_path,
        rgb,
        loaded.alpha,
        image_format=args.format,
        quality=args.quality,
        bit_depth=args.bit_depth,
        icc_profile=loaded.icc_profile,
    )

    if args.save_diff:
        diff = 128.0 + (np.clip(rgb, 0.0, 255.0) - loaded.rgb) * 16.0
        save_image(outdir / f"{output_stem}_diff.png", diff, loaded.alpha, icc_profile=loaded.icc_profile)

    report = {
        "input": str(path),
        "output": str(output_path),
        "size": {"width": loaded.size[0], "height": loaded.size[1]},
        "mode": loaded.mode,
        "bit_depth": loaded.bit_depth,
        "stages": stage_reports,
        "timings": {
            "load_seconds": load_seconds,
            "stages_seconds": stage_timings,
            "total_seconds": time.perf_counter() - started,
        },
    }
    if args.report:
        _write_json(outdir / f"{output_stem}_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not 0 <= args.quality <= 100:
        parser.error("--quality must be between 0 and 100.")
    if args.bit_depth == 16 and args.format != "png":
        parser.error("--bit-depth 16 requires --format png.")
    if args.analyze and args.save_diff:
        parser.error("--analyze does not write image files, so it cannot be used with --save-diff.")

    strengths = resolve_strengths(args.preset, args.dots, args.periodic, args.grime, args.surface,
                                  args.mottle, args.sparkle, args.lattice, args.dither, args.contour,
                                  args.defocus, args.cel_fill)
    try:
        jobs = _input_jobs(args.inputs, args.outdir)
        _preflight_outputs(jobs, args)
        backend = None
        if args.device == "gpu":
            from ._gpu import GPUBackend, GPUUnavailableError

            try:
                backend = GPUBackend()
            except GPUUnavailableError as error:
                print(f"Error: {error}", file=sys.stderr)
                return 1
        analysis_modules = {}
        if args.analyze:
            analysis_modules = {name: _load_stage(name) for name in ("dots", "sparkle", "periodic", "grime", "mottle", "lattice", "contour", "defocus", "celfill")}
        reports = []
        for path, outdir in jobs:
            reports.append(_process_one(path, outdir, args, strengths, analysis_modules, backend))
        if args.analyze:
            print(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            for report in reports:
                print(f"Restored: {report['output']}")
                if args.report:
                    print(f"Report saved: {Path(report['output']).with_name(Path(report['input']).stem + '_report.json')}")
        return 0
    except (OSError, ValueError, StageUnavailableError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
