# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Durable, no-overwrite publication for VQ2 compressed artifacts."""

import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_file(path: Path) -> None:
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(staging: Path, output: Path) -> None:
    """Atomically rename without replacing any independently created output."""
    if os.name == "nt":
        os.rename(staging, output)
        return
    if sys.platform != "linux":
        raise RuntimeError("Atomic no-overwrite artifact publication requires Linux or Windows.")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("renameat2 is required for atomic no-overwrite publication.")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if rename(at_fdcwd, os.fsencode(staging), at_fdcwd, os.fsencode(output), rename_noreplace):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def _write_json(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    _sync_directory(path.parent)
