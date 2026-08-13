"""Tests for the Linux detection backend.

These run on any host: the parser is exercised against captured ``/proc/cpuinfo``
text from Arm machines we do not need physically present.
"""

from __future__ import annotations

from armforge.hardware._linux import _parse_cpuinfo
from armforge.hardware.features import LINUX_FEATURE_MAP, decode_midr

# Captured from an AWS Graviton3 (c7g) instance: uniform Neoverse-V1, SVE
# present, SME absent -- the mirror image of Apple silicon.
GRAVITON3_CPUINFO = """\
processor	: 0
BogoMIPS	: 2100.00
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp \
cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm \
dit uscat ilrcpc flagm ssbs paca pacg dcpodp svei8mm svebf16 i8mm bf16 dgh rng
CPU implementer	: 0x41
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0xd40
CPU revision	: 1

processor	: 1
BogoMIPS	: 2100.00
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp \
cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm \
dit uscat ilrcpc flagm ssbs paca pacg dcpodp svei8mm svebf16 i8mm bf16 dgh rng
CPU implementer	: 0x41
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0xd40
CPU revision	: 1
"""

# A heterogeneous mobile SoC: one Cortex-X4, one Cortex-A720, one Cortex-A520.
BIG_LITTLE_CPUINFO = """\
processor	: 0
Features	: fp asimd aes crc32 atomics asimdhp asimddp sve sve2 i8mm bf16
CPU implementer	: 0x41
CPU part	: 0xd82

processor	: 1
Features	: fp asimd aes crc32 atomics asimdhp asimddp sve sve2 i8mm bf16
CPU implementer	: 0x41
CPU part	: 0xd81

processor	: 2
Features	: fp asimd aes crc32 atomics asimdhp asimddp sve sve2 i8mm bf16
CPU implementer	: 0x41
CPU part	: 0xd80
"""


def test_graviton3_features_are_parsed():
    flags, midrs = _parse_cpuinfo(GRAVITON3_CPUINFO)
    features = {LINUX_FEATURE_MAP[f] for f in flags if f in LINUX_FEATURE_MAP}

    # Graviton3 has the full int8/bf16 matrix set plus SVE.
    assert {"neon", "fp16", "dotprod", "i8mm", "bf16", "sve", "svei8mm", "svebf16"} <= features
    # It has no SME, and no SVE2 -- Neoverse-V1 is Armv8.4.
    assert "sme" not in features
    assert "sve2" not in features


def test_graviton3_midr_decodes_to_neoverse_v1():
    _, midrs = _parse_cpuinfo(GRAVITON3_CPUINFO)
    assert set(midrs) == {0, 1}
    assert all(decode_midr(m) == ("Arm", "Neoverse-V1") for m in midrs.values())


def test_graviton3_cores_are_all_identical():
    """A uniform part must yield exactly one distinct MIDR."""
    _, midrs = _parse_cpuinfo(GRAVITON3_CPUINFO)
    assert len(set(midrs.values())) == 1


def test_big_little_yields_three_distinct_cores():
    _, midrs = _parse_cpuinfo(BIG_LITTLE_CPUINFO)
    names = [decode_midr(midrs[cpu])[1] for cpu in sorted(midrs)]
    assert names == ["Cortex-X4", "Cortex-A720", "Cortex-A520"]
    assert len(set(midrs.values())) == 3


def test_unparseable_cpuinfo_is_handled():
    assert _parse_cpuinfo("") == ([], {})
    assert _parse_cpuinfo("garbage without colons") == ([], {})


def test_missing_midr_fields_do_not_invent_a_core():
    flags, midrs = _parse_cpuinfo("processor\t: 0\nFeatures\t: fp asimd\n")
    assert flags == ["fp", "asimd"]
    assert midrs == {}
