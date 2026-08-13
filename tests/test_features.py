"""Tests for the Arm feature registry and MIDR decoding."""

from __future__ import annotations

import pytest

from armforge.hardware.features import (
    CORE_PARTS,
    DARWIN_BARE_FEATURE_MAP,
    DARWIN_FEATURE_MAP,
    FEATURES,
    LINUX_FEATURE_MAP,
    Relevance,
    decode_midr,
    notable_absent,
    relevant_present,
)


def _midr(implementer: int, part: int, variant: int = 0, revision: int = 0) -> int:
    return (implementer << 24) | (variant << 20) | (part << 4) | revision


@pytest.mark.parametrize(
    ("implementer", "part", "expected_impl", "expected_core"),
    [
        (0x41, 0xD0C, "Arm", "Neoverse-N1"),  # AWS Graviton2
        (0x41, 0xD40, "Arm", "Neoverse-V1"),  # AWS Graviton3
        (0x41, 0xD4F, "Arm", "Neoverse-V2"),  # AWS Graviton4
        (0x41, 0xD0B, "Arm", "Cortex-A76"),  # Raspberry Pi 5
        (0x41, 0xD80, "Arm", "Cortex-A520"),
        (0x61, 0x000, "Apple", None),  # known vendor, unknown part
    ],
)
def test_decode_midr_identifies_known_cores(implementer, part, expected_impl, expected_core):
    assert decode_midr(_midr(implementer, part)) == (expected_impl, expected_core)


def test_decode_midr_ignores_variant_and_revision():
    """Two steppings of the same part must decode identically."""
    assert decode_midr(_midr(0x41, 0xD40, variant=0, revision=0)) == decode_midr(
        _midr(0x41, 0xD40, variant=3, revision=7)
    )


def test_decode_midr_returns_none_for_unknown_implementer():
    assert decode_midr(_midr(0x99, 0xFFF)) == (None, None)


@pytest.mark.parametrize(
    "mapping",
    [DARWIN_FEATURE_MAP, DARWIN_BARE_FEATURE_MAP, LINUX_FEATURE_MAP],
    ids=["darwin", "darwin_bare", "linux"],
)
def test_every_mapped_feature_exists_in_the_registry(mapping):
    """Guards against a typo in an OS map silently producing an unknown key."""
    unknown = {key for key in mapping.values() if key not in FEATURES}
    assert not unknown, f"mapped to keys absent from FEATURES: {sorted(unknown)}"


def test_registry_keys_are_self_consistent():
    for key, info in FEATURES.items():
        assert info.key == key


def test_core_parts_table_has_no_duplicate_names_per_implementer():
    seen: dict[tuple[int, str], int] = {}
    for (implementer, part), name in CORE_PARTS.items():
        previous = seen.get((implementer, name))
        assert previous is None, f"{name} mapped from both 0x{previous:x} and 0x{part:x}"
        seen[(implementer, name)] = part


def test_relevant_present_ranks_critical_first():
    ranked = relevant_present(frozenset({"fp16", "i8mm", "bf16", "neon"}))
    assert [f.relevance for f in ranked] == sorted(
        [f.relevance for f in ranked],
        key=lambda r: [
            Relevance.CRITICAL,
            Relevance.HIGH,
            Relevance.MODERATE,
            Relevance.LOW,
        ].index(r),
    )
    assert ranked[0].relevance is Relevance.CRITICAL


def test_relevant_present_ignores_unknown_keys():
    assert relevant_present(frozenset({"not_a_real_feature"})) == []


def test_notable_absent_reports_missing_high_value_features():
    """An Apple-style vector: SME present, SVE absent."""
    absent = notable_absent(frozenset({"neon", "dotprod", "i8mm", "sme2"}), is_arm64=True)
    keys = {info.key for info in absent}
    assert "sve" in keys
    assert "i8mm" not in keys
    assert all(info.relevance in (Relevance.CRITICAL, Relevance.HIGH) for info in absent)


def test_notable_absent_is_empty_off_arm():
    assert notable_absent(frozenset(), is_arm64=False) == []
