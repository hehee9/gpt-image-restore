from pathlib import Path
import warnings

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).parent


def discover_extensions():
    sources = sorted((ROOT / "gptimg_restore").glob("_native_*.pyx"))
    if not sources:
        return []

    try:
        from Cython.Build import cythonize
    except ImportError as error:
        warnings.warn(
            f"Skipping optional Cython extensions. Cython could not be imported: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    extensions = [
        Extension(
            f"gptimg_restore.{source.stem}",
            [str(source.relative_to(ROOT))],
        )
        for source in sources
    ]
    try:
        return cythonize(
            extensions,
            build_dir="build/cython",
            quiet=True,
        )
    except Exception as error:
        warnings.warn(
            f"Failed to cythonize optional extensions; installing with pure Python: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []


class OptionalBuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type in {"unix", "mingw32"}:
            for extension in self.extensions:
                extension.extra_compile_args = ["-O2", "-ffp-contract=off"]
        super().build_extensions()

    def run(self):
        extension_outputs = [Path(self.get_ext_fullpath(ext.name)) for ext in self.extensions]
        try:
            super().run()
        except Exception as error:
            for output in extension_outputs:
                if output.is_file():
                    output.unlink()
            self.extensions = []
            warnings.warn(
                f"Failed to build optional Cython extensions; installing with pure Python: {error}",
                RuntimeWarning,
                stacklevel=2,
            )


setup(
    ext_modules=discover_extensions(),
    cmdclass={"build_ext": OptionalBuildExt},
)
