# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict TP1 reader for the same packed-zN storage contract used by TP2.

The shared shard/spec types carry storage metadata only. The TP1 entry point
fixes rank zero and full N/K: it never accepts a TP2 artifact or pads its K.
"""

from pathlib import Path

from .vq2a8_tp2_runtime import VQ2TP2Artifact, _file, _json, _no_links, _open_vq2a8_zn_artifact


def artifact_format(artifact_path: str | Path) -> str:
    """Read only the discriminator; the selected strict reader validates all else."""
    root = _no_links(Path(artifact_path))
    value = _json(_file(root, "manifest.json")).get("format")
    if not isinstance(value, str):
        raise ValueError("Artifact manifest must declare a string format.")
    return value


def open_vq2a8_tp1_zn_artifact(
    artifact_path: str | Path, model_config_path: str | Path, *, verify_tensor_hashes: bool = True
) -> VQ2TP2Artifact:
    """Open full TP1 packed-zN with strict paths, hashes, geometry and value checks.

    ``runtime_compatible=False`` preserves the exporter's unverified device
    claim; it is not an instruction to reject an otherwise valid artifact.
    """
    return _open_vq2a8_zn_artifact(
        artifact_path, model_config_path, tp_size=1, tp_rank=0, verify_tensor_hashes=verify_tensor_hashes
    )
