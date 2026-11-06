"""Collect third-party license notices for a PyInstaller release bundle."""

from __future__ import annotations

import argparse
from io import BytesIO
from importlib import metadata
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile


SEPARATOR = "=" * 79
LICENSE_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE")
DPKG_CHUNK_SIZE = 200
PYPI_TIMEOUT = 60


class PyPIRequestError(Exception):
    """Represent a failed PyPI request that must stop notice generation."""


class PyPIReleaseNotFound(Exception):
    """Represent a release metadata endpoint that returned HTTP 404."""


def _parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Collect third-party license notices for a PyInstaller bundle."
    )
    parser.add_argument("bundle_dir", type=Path, help="PyInstaller output directory")
    parser.add_argument("output_file", type=Path, help="path for the notices text file")
    return parser


def _python_license() -> str:
    """Read the CPython license from the active Python installation."""
    candidates = (
        Path(sys.base_prefix) / "LICENSE.txt",
        Path(sysconfig.get_path("stdlib")) / "LICENSE.txt",
    )
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    paths = " and ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"CPython license file not found at {paths}")


def _metadata_license(distribution: metadata.Distribution) -> str:
    """Return the most useful short license value in distribution metadata."""
    license_expression = distribution.metadata.get("License-Expression")
    if license_expression:
        return f"License: {license_expression.strip()}"

    license_value = distribution.metadata.get("License", "").strip()
    if (
        license_value
        and len(license_value) < 100
        and len(license_value.splitlines()) == 1
    ):
        return f"License: {license_value}"

    classifiers = [
        value
        for value in distribution.metadata.get_all("Classifier", [])
        if value.startswith("License ::")
    ]
    if classifiers:
        return f"License: {'; '.join(classifiers)}"
    return "License: Not specified in metadata"


def _pypi_request(url: str, allow_not_found: bool = False) -> bytes:
    """Read a PyPI resource, allowing a 404 only for the release metadata URL."""
    try:
        with urllib.request.urlopen(url, timeout=PYPI_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if allow_not_found and error.code == 404:
            raise PyPIReleaseNotFound from error
        raise PyPIRequestError(
            f"PyPI request failed for {url}: HTTP {error.code} {error.reason}"
        ) from error
    except OSError as error:
        raise PyPIRequestError(f"PyPI request failed for {url}: {error}") from error


def _source_license_files(archive_data: bytes, filename: str) -> list[tuple[str, str]]:
    """Read unique license-like files from an in-memory source archive."""
    entries = []
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(BytesIO(archive_data)) as archive:
            entries = [
                (member.filename, archive.read(member))
                for member in archive.infolist()
                if not member.is_dir()
                and PurePosixPath(member.filename)
                .name.upper()
                .startswith(LICENSE_PREFIXES)
            ]
    else:
        with tarfile.open(fileobj=BytesIO(archive_data), mode="r:*") as archive:
            for member in archive.getmembers():
                if member.isfile() and PurePosixPath(
                    member.name
                ).name.upper().startswith(LICENSE_PREFIXES):
                    file = archive.extractfile(member)
                    if file is not None:
                        entries.append((member.name, file.read()))

    license_files = []
    seen_texts = set()
    for member_path, content in sorted(entries, key=lambda entry: entry[0]):
        text = content.decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if text in seen_texts:
            continue
        seen_texts.add(text)
        license_files.append((member_path, text))
    return license_files


def _pypi_license_files(
    distribution: metadata.Distribution,
) -> tuple[str | None, list[tuple[str, str]]]:
    """Find license files in the distribution's PyPI source archive."""
    name = urllib.parse.quote(distribution.metadata["Name"], safe="")
    version = urllib.parse.quote(distribution.version, safe="")
    metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        metadata_data = _pypi_request(metadata_url, allow_not_found=True)
    except PyPIReleaseNotFound:
        return None, []

    release = json.loads(metadata_data)
    source_distribution = next(
        (url for url in release["urls"] if url["packagetype"] == "sdist"),
        None,
    )
    if source_distribution is None:
        return None, []

    filename = source_distribution["filename"]
    archive_data = _pypi_request(source_distribution["url"])
    return filename, _source_license_files(archive_data, filename)


def _distribution_sections() -> list[str]:
    """Build license sections for installed Python distributions."""
    seen_names = set()
    package_sections = []
    installed = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        if normalized_name == "gpt-image-restore" or normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        installed[normalized_name] = distribution

    for normalized_name in sorted(installed):
        distribution = installed[normalized_name]
        name = distribution.metadata["Name"]
        section = [f"{name} {distribution.version}", _metadata_license(distribution)]
        seen_texts = set()
        license_files = []
        license_entries = [
            entry
            for entry in sorted(distribution.files or (), key=str)
            if Path(entry).name.upper().startswith(LICENSE_PREFIXES)
        ]
        for entry in license_entries:
            path = Path(distribution.locate_file(entry))
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if text in seen_texts:
                continue
            seen_texts.add(text)
            license_files.append((str(entry), text))

        source_filename = None
        if not license_entries:
            source_filename, license_files = _pypi_license_files(distribution)
            if not license_files:
                section.append("No license file is included in this distribution.")
                if source_filename is None:
                    section.append("No PyPI source distribution was found.")
                else:
                    section.append(
                        f"The PyPI source distribution {source_filename} has no license files either."
                    )
        elif not license_files:
            section.append("No license file is included in this distribution.")

        if license_files:
            for relative_path, text in license_files:
                prefix = (
                    f"License file (from {source_filename}):"
                    if source_filename is not None
                    else "License file:"
                )
                section.extend(("", f"{prefix} {relative_path}", text.rstrip("\n")))
        package_sections.append("\n".join(section))
    return package_sections


def _dpkg_packages(bundle_dir: Path) -> set[str]:
    """Find Debian package owners for shared libraries in the bundle."""
    dpkg = shutil.which("dpkg")
    if dpkg is None:
        return set()

    library_names = sorted(
        {
            path.name
            for path in bundle_dir.rglob("*")
            if path.is_file() and ".so" in path.name
        }
    )
    packages = set()
    for start in range(0, len(library_names), DPKG_CHUNK_SIZE):
        chunk = library_names[start : start + DPKG_CHUNK_SIZE]
        result = subprocess.run(
            [dpkg, "-S", *chunk],
            capture_output=True,
            text=True,
            check=False,
        )
        requested_names = set(chunk)
        for line in result.stdout.splitlines():
            package_list, separator, file_path = line.partition(": ")
            if not separator or Path(file_path).name not in requested_names:
                continue
            for package in package_list.split(","):
                packages.add(package.strip().split(":", 1)[0])
    return packages


def _system_library_sections(bundle_dir: Path) -> list[str]:
    """Build license sections for Debian packages owning bundled libraries."""
    sections = []
    for package in sorted(_dpkg_packages(bundle_dir)):
        section = [f"{package} (Debian/Ubuntu package)"]
        copyright_file = Path("/usr/share/doc") / package / "copyright"
        if copyright_file.is_file():
            section.append(copyright_file.read_text(encoding="utf-8", errors="replace"))
        else:
            section.append("Copyright file was not found.")
        sections.append("\n".join(section))
    return sections


def _notices(bundle_dir: Path, python_license: str) -> str:
    """Assemble all requested notice sections in their required order."""
    sections = [
        "Third-party software license notices\n"
        "This file lists the licenses of third-party software bundled with GPT Image Restore. "
        "GPT Image Restore itself is licensed under AGPL-3.0-or-later (see LICENSE).",
        "CPython runtime license\n" + python_license.rstrip("\n"),
        *_distribution_sections(),
        *_system_library_sections(bundle_dir),
    ]
    return (
        f"\n\n{SEPARATOR}\n\n".join(section.rstrip("\n") for section in sections) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Write the third-party license notices file."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.bundle_dir.is_dir():
        parser.error(f"bundle directory not found: {args.bundle_dir}")

    try:
        python_license = _python_license()
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    try:
        notices = _notices(args.bundle_dir, python_license)
    except PyPIRequestError as error:
        print(str(error), file=sys.stderr)
        return 1
    with args.output_file.open("w", encoding="utf-8", newline="\n") as output:
        output.write(notices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
