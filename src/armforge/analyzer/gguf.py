"""A minimal GGUF metadata reader.

ArmForge needs to know a model's real quantisation, architecture and parameter
count before it can decide which optimisations are worth attempting. Guessing
from the filename is unreliable -- files get renamed, and ``model-q4.gguf``
says nothing about whether it is ``Q4_0`` (which llama.cpp can repack for
SMMLA) or ``Q4_K_M`` (which it cannot). Those two take different code paths on
Arm, so the distinction decides what we benchmark.

Only the header is read: magic, the key/value metadata block, and the tensor
index. No weights are loaded, so this is fast and needs almost no memory even
for very large files.

Format reference: ggml/docs/gguf.md in the llama.cpp repository.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

#: Bytes of header we are willing to read before giving up, so a corrupt or
#: hostile file cannot make us allocate without bound.
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_STRING_BYTES = 1 * 1024 * 1024
MAX_ARRAY_ITEMS = 1_000_000


class GGUFError(ValueError):
    """Raised when a file is not valid GGUF, or is too damaged to read."""


# GGUF metadata value type tags.
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL = 4, 5, 6, 7
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12

_SIMPLE_FORMATS: dict[int, tuple[str, int]] = {
    _UINT8: ("<B", 1),
    _INT8: ("<b", 1),
    _UINT16: ("<H", 2),
    _INT16: ("<h", 2),
    _UINT32: ("<I", 4),
    _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4),
    _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8),
    _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}

#: ``general.file_type`` (llama.cpp's ``llama_ftype``) to a human name.
FILE_TYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
}

#: Quantisations that llama.cpp can repack into an int8 matrix-multiply
#: friendly layout at load time. This is the property that makes ``FEAT_I8MM``
#: worth having, and it is why ArmForge distinguishes Q4_0 from the K-quants.
REPACKABLE_FOR_I8MM: frozenset[str] = frozenset({"Q4_0", "Q8_0", "IQ4_NL"})


@dataclass
class GGUFModel:
    """Header-level facts about a GGUF file."""

    path: Path
    version: int
    tensor_count: int
    file_size_bytes: int
    architecture: str | None = None
    name: str | None = None
    quantization: str | None = None
    parameter_count: int | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def repackable_for_i8mm(self) -> bool:
        """Whether this quantisation has an int8 matrix-multiply fast path.

        Capability of the *format*. Whether the runtime actually takes that
        path depends on the build and the CPU, and is settled by measurement.
        """
        return (self.quantization or "") in REPACKABLE_FOR_I8MM

    @property
    def bits_per_weight(self) -> float | None:
        """Average bits per parameter, derived from file size."""
        if not self.parameter_count:
            return None
        return (self.file_size_bytes * 8) / self.parameter_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "architecture": self.architecture,
            "quantization": self.quantization,
            "gguf_version": self.version,
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
            "context_length": self.context_length,
            "embedding_length": self.embedding_length,
            "block_count": self.block_count,
            "file_size_bytes": self.file_size_bytes,
            "file_size_gb": round(self.file_size_bytes / (1024**3), 3),
            "bits_per_weight": (
                round(self.bits_per_weight, 2) if self.bits_per_weight else None
            ),
            "repackable_for_i8mm": self.repackable_for_i8mm,
        }


class _Reader:
    """Bounded little-endian reader over a GGUF header."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._consumed = 0

    def read(self, count: int) -> bytes:
        if count < 0 or self._consumed + count > MAX_HEADER_BYTES:
            raise GGUFError("header exceeds the maximum size ArmForge will read")
        data = self._stream.read(count)
        if len(data) != count:
            raise GGUFError("file ended in the middle of the header")
        self._consumed += count
        return data

    def scalar(self, fmt: str, size: int) -> Any:
        return struct.unpack(fmt, self.read(size))[0]

    def u32(self) -> int:
        return self.scalar("<I", 4)

    def u64(self) -> int:
        return self.scalar("<Q", 8)

    def string(self) -> str:
        length = self.u64()
        if length > MAX_STRING_BYTES:
            raise GGUFError(f"implausible string length {length}")
        return self.read(length).decode("utf-8", errors="replace")

    def value(self, value_type: int) -> Any:
        if value_type in _SIMPLE_FORMATS:
            fmt, size = _SIMPLE_FORMATS[value_type]
            return self.scalar(fmt, size)
        if value_type == _STRING:
            return self.string()
        if value_type == _ARRAY:
            element_type = self.u32()
            count = self.u64()
            if count > MAX_ARRAY_ITEMS:
                raise GGUFError(f"implausible array length {count}")
            # Tokeniser vocabularies live in these arrays and are large; we
            # read them to stay positioned correctly but keep only a summary.
            return [self.value(element_type) for _ in range(count)]
        raise GGUFError(f"unknown GGUF value type {value_type}")


def _summarise(value: Any) -> Any:
    """Collapse huge arrays so metadata stays printable and JSON-friendly."""
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    if isinstance(value, str) and len(value) > 256:
        return value[:256] + "..."
    return value


def read_gguf(path: str | Path) -> GGUFModel:
    """Read GGUF header metadata.

    Raises :class:`GGUFError` if the file is not GGUF or its header is
    unreadable. It never raises for a *missing* optional field -- those stay
    ``None``, because an unknown value must not be confused with a real one.
    """
    path = Path(path)
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise GGUFError(f"cannot stat {path}: {exc}") from exc

    with path.open("rb") as stream:
        reader = _Reader(stream)

        if reader.read(4) != GGUF_MAGIC:
            raise GGUFError(f"{path.name} is not a GGUF file (bad magic)")

        version = reader.u32()
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")

        tensor_count = reader.u64()
        kv_count = reader.u64()

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = reader.string()
            value_type = reader.u32()
            metadata[key] = reader.value(value_type)

        # Tensor index: dimensions give us an exact parameter count.
        parameter_count = 0
        for _ in range(tensor_count):
            reader.string()  # tensor name
            n_dims = reader.u32()
            if n_dims > 8:
                raise GGUFError(f"implausible tensor rank {n_dims}")
            elements = 1
            for _ in range(n_dims):
                elements *= reader.u64()
            reader.u32()  # ggml type
            reader.u64()  # offset
            parameter_count += elements

    architecture = metadata.get("general.architecture")
    file_type = metadata.get("general.file_type")
    quantization = FILE_TYPES.get(file_type) if isinstance(file_type, int) else None

    def arch_key(suffix: str) -> Any:
        return metadata.get(f"{architecture}.{suffix}") if architecture else None

    return GGUFModel(
        path=path,
        version=version,
        tensor_count=tensor_count,
        file_size_bytes=file_size,
        architecture=architecture,
        name=metadata.get("general.name") or path.stem,
        quantization=quantization,
        parameter_count=parameter_count or None,
        context_length=arch_key("context_length"),
        embedding_length=arch_key("embedding_length"),
        block_count=arch_key("block_count"),
        metadata={k: _summarise(v) for k, v in metadata.items()},
    )
