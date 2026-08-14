"""Tests for the GGUF header analyzer."""

from __future__ import annotations

import pytest

from armforge.analyzer import GGUFError, read_gguf
from tests.helpers import build_gguf


def test_reads_architecture_quantization_and_params(tmp_path):
    path = build_gguf(
        tmp_path / "qwen-q4_0.gguf",
        metadata={
            "general.architecture": "qwen2",
            "general.name": "Qwen2.5 0.5B Instruct",
            "general.file_type": 2,  # Q4_0
            "qwen2.context_length": 32768,
            "qwen2.embedding_length": 896,
            "qwen2.block_count": 24,
        },
        tensors=[("token_embd.weight", (896, 151936)), ("blk.0.attn_q.weight", (896, 896))],
    )

    model = read_gguf(path)

    assert model.architecture == "qwen2"
    assert model.name == "Qwen2.5 0.5B Instruct"
    assert model.quantization == "Q4_0"
    assert model.context_length == 32768
    assert model.block_count == 24
    assert model.tensor_count == 2
    # Parameter count is summed from real tensor dimensions.
    assert model.parameter_count == 896 * 151936 + 896 * 896


def test_q4_0_is_repackable_but_q4_k_m_is_not(tmp_path):
    """The distinction that decides whether FEAT_I8MM can be exploited."""
    q4_0 = read_gguf(
        build_gguf(
            tmp_path / "a.gguf",
            metadata={"general.architecture": "llama", "general.file_type": 2},
            tensors=[("w", (32, 32))],
        )
    )
    q4_k_m = read_gguf(
        build_gguf(
            tmp_path / "b.gguf",
            metadata={"general.architecture": "llama", "general.file_type": 15},
            tensors=[("w", (32, 32))],
        )
    )

    assert q4_0.quantization == "Q4_0"
    assert q4_0.repackable_for_i8mm is True
    assert q4_k_m.quantization == "Q4_K_M"
    assert q4_k_m.repackable_for_i8mm is False


@pytest.mark.parametrize(
    ("file_type", "expected"),
    [(0, "F32"), (1, "F16"), (7, "Q8_0"), (15, "Q4_K_M"), (18, "Q6_K"), (32, "BF16")],
)
def test_known_file_types_decode(tmp_path, file_type, expected):
    model = read_gguf(
        build_gguf(
            tmp_path / f"m{file_type}.gguf",
            metadata={"general.architecture": "llama", "general.file_type": file_type},
            tensors=[("w", (8, 8))],
        )
    )
    assert model.quantization == expected


def test_unknown_file_type_is_none_not_a_guess(tmp_path):
    """An unrecognised quantisation must stay unknown, never be invented."""
    model = read_gguf(
        build_gguf(
            tmp_path / "future.gguf",
            metadata={"general.architecture": "llama", "general.file_type": 999},
            tensors=[("w", (8, 8))],
        )
    )
    assert model.quantization is None
    assert model.repackable_for_i8mm is False


def test_missing_optional_metadata_stays_none(tmp_path):
    model = read_gguf(
        build_gguf(
            tmp_path / "sparse.gguf",
            metadata={"general.architecture": "llama"},
            tensors=[("w", (8, 8))],
        )
    )
    assert model.quantization is None
    assert model.context_length is None
    assert model.architecture == "llama"


def test_array_metadata_is_summarised_not_dumped(tmp_path):
    """Tokeniser vocabularies must not be inlined into the report."""
    model = read_gguf(
        build_gguf(
            tmp_path / "vocab.gguf",
            metadata={
                "general.architecture": "llama",
                "tokenizer.ggml.tokens": [f"tok{i}" for i in range(500)],
            },
            tensors=[("w", (8, 8))],
        )
    )
    assert model.metadata["tokenizer.ggml.tokens"] == {"type": "array", "length": 500}


def test_bits_per_weight_is_derived_from_real_size(tmp_path):
    path = build_gguf(
        tmp_path / "sized.gguf",
        metadata={"general.architecture": "llama", "general.file_type": 2},
        tensors=[("w", (1000, 1000))],
        padding_bytes=500_000,
    )
    model = read_gguf(path)
    assert model.parameter_count == 1_000_000
    # 500 KB of padding plus a small header over 1M params -> ~4 bits each.
    assert 3.9 < model.bits_per_weight < 4.3


def test_rejects_non_gguf_file(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"NOTAGGUF" + b"\x00" * 64)
    with pytest.raises(GGUFError, match="not a GGUF file"):
        read_gguf(path)


def test_rejects_unsupported_version(tmp_path):
    path = build_gguf(
        tmp_path / "v9.gguf",
        metadata={"general.architecture": "llama"},
        tensors=[],
        version=9,
    )
    with pytest.raises(GGUFError, match="unsupported GGUF version"):
        read_gguf(path)


def test_truncated_header_raises_rather_than_returning_partial_data(tmp_path):
    full = build_gguf(
        tmp_path / "full.gguf",
        metadata={"general.architecture": "llama", "general.file_type": 2},
        tensors=[("w", (8, 8))],
    )
    truncated = tmp_path / "truncated.gguf"
    truncated.write_bytes(full.read_bytes()[:20])

    with pytest.raises(GGUFError, match="ended in the middle"):
        read_gguf(truncated)


def test_missing_file_raises_gguf_error(tmp_path):
    with pytest.raises(GGUFError, match="cannot stat"):
        read_gguf(tmp_path / "absent.gguf")


def test_implausible_string_length_is_rejected(tmp_path):
    """A hostile header must not make us allocate without bound."""
    import struct

    path = tmp_path / "hostile.gguf"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", 2**40)  # absurd key length
    )
    with pytest.raises(GGUFError, match="implausible string length"):
        read_gguf(path)
