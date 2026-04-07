"""Windows long path mitigation via subst drive mapping.

On Windows, the MAX_PATH limit (260 characters) causes build failures when
the PyTorch source tree is located in a deeply nested directory.  The CMake
build generates intermediate files (object files, stamp files, Ninja manifests)
whose full paths can exceed this limit.

This module provides a transparent workaround: when the build directory path
is long enough to be risky, it creates a virtual drive letter via ``subst``
that maps directly to the build directory.  CMake then operates under the
short drive path (e.g. ``P:\\``) instead of the full path, keeping all
intermediate file paths well under the limit.

The ``subst`` command requires no administrator privileges and the mapping
is session-scoped (does not survive reboot), making it safe for development
use.

See https://github.com/pytorch/pytorch/issues/134956
"""

from __future__ import annotations

import atexit
import functools
import os
import platform
import signal
import string
import subprocess
import sys
import types
from collections.abc import Callable
from typing import Any


eprint = functools.partial(print, file=sys.stderr, flush=True)

# If the absolute build-directory path is at least this long, we activate
# the subst workaround.  The value is chosen so that even a generous amount
# of intermediate path nesting (~180 chars for the deepest XPU/SYCL object
# files) stays under 260.
_BUILD_PATH_THRESHOLD = 70

# Filename stored inside the real build directory to remember which drive
# letter was used, so that subsequent incremental builds reuse it.
_MARKER_FILENAME = ".pytorch_subst_drive"


def _is_drive_available(letter: str) -> bool:
    """Return True if the given drive letter is not currently in use."""
    return not os.path.exists(f"{letter}:\\")


def _find_free_drive() -> str | None:
    """Find an unused drive letter, preferring letters near the end of the
    alphabet to avoid collisions with physical drives and network shares."""
    for letter in reversed(string.ascii_uppercase):
        if letter in ("A", "B", "C"):
            # A/B are floppy drives, C is the typical system drive
            continue
        if _is_drive_available(letter):
            return letter
    return None


def _read_marker(build_dir: str) -> str | None:
    """Read the previously-used drive letter from the marker file."""
    marker = os.path.join(build_dir, _MARKER_FILENAME)
    try:
        with open(marker) as f:
            drive = f.read().strip().upper()
            if len(drive) == 1 and drive in string.ascii_uppercase:
                return drive
    except OSError:
        pass
    return None


def _write_marker(build_dir: str, drive: str) -> None:
    """Persist the drive letter so incremental rebuilds reuse it."""
    marker = os.path.join(build_dir, _MARKER_FILENAME)
    try:
        with open(marker, "w") as f:
            f.write(drive)
    except OSError:
        pass


def _remove_subst(drive: str) -> None:
    """Remove a subst mapping, ignoring errors."""
    try:
        subprocess.call(
            ["subst", f"{drive}:", "/d"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def _create_subst(drive: str, target: str) -> bool:
    """Create a subst mapping.  Returns True on success."""
    try:
        subprocess.check_call(
            ["subst", f"{drive}:", target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def maybe_subst_build_dir(cmake: Any) -> Callable[[], None] | None:
    """If on Windows with a long build path, subst the build directory to a
    short drive letter.

    Mutates ``cmake.build_dir`` in-place to point at the short drive path.

    Returns a cleanup callable that removes the subst mapping, or ``None``
    if no subst was needed or possible.

    Args:
        cmake: A ``tools.setup_helpers.cmake.CMake`` instance.
    """
    if platform.system() != "Windows":
        return None

    build_dir: str = cmake.build_dir
    abs_build = os.path.abspath(build_dir)

    if len(abs_build) < _BUILD_PATH_THRESHOLD:
        return None

    # Ensure the directory exists (subst requires an existing target)
    os.makedirs(abs_build, exist_ok=True)

    prev_drive = _read_marker(abs_build)
    drive: str | None = None

    if prev_drive is not None:
        if _is_drive_available(prev_drive):
            drive = prev_drive
        else:
            # Previous drive is taken.  The CMake cache contains paths
            # referencing the old drive, so we must invalidate it.
            eprint(
                f"WARNING: Previous subst drive {prev_drive}: is in use by "
                f"another mapping.  Clearing CMake cache to avoid stale paths."
            )
            cache_file = os.path.join(abs_build, "CMakeCache.txt")
            if os.path.isfile(cache_file):
                os.remove(cache_file)
            ninja_file = os.path.join(abs_build, "build.ninja")
            if os.path.isfile(ninja_file):
                os.remove(ninja_file)

    if drive is None:
        drive = _find_free_drive()

    if drive is None:
        eprint(
            "WARNING: No free drive letter available for subst.  "
            "Build may fail if paths exceed MAX_PATH (260 chars).  "
            "Consider using a shorter source directory."
        )
        return None

    if not _create_subst(drive, abs_build):
        eprint(
            f"WARNING: Failed to create subst {drive}: -> {abs_build}.  "
            "Build may fail if paths exceed MAX_PATH (260 chars)."
        )
        return None

    _write_marker(abs_build, drive)

    short_path = f"{drive}:\\"
    eprint(
        f"-- Windows long path mitigation: mapped {drive}: -> {abs_build}\n"
        f"-- Build directory shortened from {len(abs_build)} to {len(short_path)} chars"
    )

    cmake.build_dir = short_path

    cleaned_up = False

    def cleanup() -> None:
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        _remove_subst(drive)
        eprint(f"-- Removed subst mapping {drive}:")

    atexit.register(cleanup)

    if hasattr(signal, "SIGBREAK"):
        prev_handler = signal.getsignal(signal.SIGBREAK)

        def _sigbreak_handler(signum: int, frame: types.FrameType | None) -> None:
            cleanup()
            if callable(prev_handler) and prev_handler not in (
                signal.SIG_IGN,
                signal.SIG_DFL,
            ):
                prev_handler(signum, frame)
            else:
                sys.exit(1)

        signal.signal(signal.SIGBREAK, _sigbreak_handler)

    return cleanup
