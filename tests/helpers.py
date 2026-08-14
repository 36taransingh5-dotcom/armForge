"""Test helpers shared across modules.

Kept out of ``conftest.py`` deliberately: pytest imports conftest as a plugin,
so importing it again by name loads the module twice under two identities.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

_STRING, _ARRAY, _UINT32 = 8, 9, 4


def _encode_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _encode_value(value: Any) -> bytes:
    """Encode a metadata value with its type tag."""
    if isinstance(value, str):
        return struct.pack("<I", _STRING) + _encode_string(value)
    if isinstance(value, int):
        return struct.pack("<I", _UINT32) + struct.pack("<I", value)
    if isinstance(value, list):
        body = b"".join(_encode_string(v) for v in value)
        return (
            struct.pack("<I", _ARRAY)
            + struct.pack("<I", _STRING)
            + struct.pack("<Q", len(value))
            + body
        )
    raise TypeError(f"test builder cannot encode {type(value)}")


def build_gguf(
    path: Path,
    *,
    metadata: dict[str, Any],
    tensors: list[tuple[str, tuple[int, ...]]],
    version: int = 3,
    magic: bytes = b"GGUF",
    padding_bytes: int = 0,
) -> Path:
    """Write a syntactically valid GGUF header with no weight data.

    Enough for the analyzer, which never reads past the tensor index.
    """
    out = bytearray()
    out += magic
    out += struct.pack("<I", version)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata))

    for key, value in metadata.items():
        out += _encode_string(key)
        out += _encode_value(value)

    for name, dims in tensors:
        out += _encode_string(name)
        out += struct.pack("<I", len(dims))
        for dim in dims:
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", 0)  # ggml type
        out += struct.pack("<Q", 0)  # offset

    out += b"\x00" * padding_bytes
    path.write_bytes(bytes(out))
    return path
