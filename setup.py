from __future__ import annotations

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PACKAGE_DIR = SRC / "deepgroebner_pytorch_cocoa"


def _first_existing(paths: list[str]) -> str | None:
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            return str(path)
    return None


def _resolve_cocoa_include_dir() -> str:
    env_value = os.environ.get("COCOA_INCLUDE_DIR")
    if env_value:
        return env_value

    resolved = _first_existing(
        [
            "/usr/local/include",
            "/usr/include",
            "/opt/homebrew/include",
            "/opt/local/include",
        ]
    )
    if resolved is None:
        raise RuntimeError(
            "Could not find a CoCoALib include directory. Set COCOA_INCLUDE_DIR "
            "to the directory containing the CoCoA headers."
        )
    return resolved


def _resolve_library_dir() -> str | None:
    env_value = os.environ.get("COCOA_LIB_DIR")
    if env_value:
        return env_value

    return _first_existing(
        [
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/opt/homebrew/lib",
            "/opt/local/lib",
        ]
    )


def _resolve_extension() -> Extension:
    include_dirs = [
        str(SRC),
        str(PACKAGE_DIR),
        _resolve_cocoa_include_dir(),
    ]
    library_dirs: list[str] = []
    extra_objects: list[str] = []
    libraries: list[str] = []

    cocoa_library_file = os.environ.get("COCOA_LIBRARY_FILE")
    if cocoa_library_file:
        extra_objects.append(cocoa_library_file)
        libraries.append("gmp")
    else:
        cocoa_library = os.environ.get("COCOA_LIBRARY", "cocoa")
        libraries.extend([cocoa_library, "gmp"])
        resolved_lib_dir = _resolve_library_dir()
        if resolved_lib_dir is not None:
            library_dirs.append(resolved_lib_dir)

    return Extension(
        name="deepgroebner_pytorch_cocoa.wrapped",
        sources=[
            "src/deepgroebner_pytorch_cocoa/wrapped.pyx",
            "src/deepgroebner_pytorch_cocoa/ideals.cpp",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_objects=extra_objects,
        language="c++",
        extra_compile_args=["-std=c++17"],
    )


setup(
    ext_modules=cythonize([_resolve_extension()], language_level=3),
)
