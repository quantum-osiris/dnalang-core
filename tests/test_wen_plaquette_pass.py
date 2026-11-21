"""Tests for the Wen-plaquette feed-forward TKET pass and provenance capsule."""

import pytest

pytket = pytest.importorskip("pytket")

from pytket import Circuit, OpType  # noqa: E402

from dnalang.passes.wen_plaquette_pass import (  # noqa: E402
    END_REG,
    EXPECTED_CONDITIONALS,
    EXPECTED_RESETS,
    NUM_QUBITS,
    ROUND1_REGS,
    ROUND2_LUT_BE,
    ROUND2_REG,
    EndiannessPolicy,
    WenPlaquetteFeedForwardPass,
    encode_syndrome_condition,
    reconstruct_stabilizer_parity,
)


ALL_EVEN_4BIT = [f"{i:04b}" for i in range(16) if f"{i:04b}".count("1") % 2 == 0]
NONZERO_EVEN = [s for s in ALL_EVEN_4BIT if s != "0000"]


def test_round2_lut_has_seven_nonzero_even_parity_entries():
    assert len(ROUND2_LUT_BE) == 7
    assert all(s.count("1") % 2 == 0 and s != "0000" for s in ROUND2_LUT_BE)
    assert set(ROUND2_LUT_BE) == set(NONZERO_EVEN)


def test_big_endian_to_little_endian_condition_encoding():
    # "0011" big-endian is 3; TKET little-endian reverses to int("1100",2)=12
    assert encode_syndrome_condition("0011", 4, EndiannessPolicy.TKET_LITTLE_ENDIAN) == 12
    assert encode_syndrome_condition("0011", 4, EndiannessPolicy.QISKIT_BIG_ENDIAN) == 3
    for s in NONZERO_EVEN:
        assert encode_syndrome_condition(s, 4, EndiannessPolicy.TKET_LITTLE_ENDIAN) == int(s[::-1], 2)
        assert encode_syndrome_condition(s, 4, EndiannessPolicy.QISKIT_BIG_ENDIAN) == int(s, 2)


def test_odd_parity_syndromes_rejected():
    for s in [f"{i:04b}" for i in range(16) if f"{i:04b}".count("1") % 2 == 1]:
        with pytest.raises(ValueError, match="odd-parity|anyon"):
            encode_syndrome_condition(s, 4, EndiannessPolicy.TKET_LITTLE_ENDIAN)
    # parity enforcement can be disabled explicitly
    assert encode_syndrome_condition("0001", 4, EndiannessPolicy.QISKIT_BIG_ENDIAN, enforce_even_parity=False) == 1


def test_malformed_bitstrings_rejected():
    for bad in ["", "012", "01", "01010", "abcd", None, 12]:
        with pytest.raises(ValueError, match="malformed"):
            encode_syndrome_condition(bad, 4, EndiannessPolicy.TKET_LITTLE_ENDIAN)


def test_pass_structure_exact_counts():
    p = WenPlaquetteFeedForwardPass()
    c = p.build_protocol_circuit()
    assert c.n_qubits == NUM_QUBITS == 20
    counts = p.structural_audit(c)
    assert counts == {"conditional": EXPECTED_CONDITIONALS, "reset": EXPECTED_RESETS}
    reg_sizes = {name: 4 for name in ROUND1_REGS}
    reg_sizes[ROUND2_REG] = 4
    reg_sizes[END_REG] = 6
    assert p.register_map == reg_sizes
    for name, size in reg_sizes.items():
        assert c.get_c_register(name).size == size


def test_to_tket_pass_requires_20_qubits_and_audits():
    p = WenPlaquetteFeedForwardPass()
    tpass = p.to_tket_pass()
    with pytest.raises(ValueError, match="20 qubits"):
        tpass.apply(Circuit(3))
    circ = Circuit(NUM_QUBITS)
    tpass.apply(circ)  # must not raise; protocol itself is audited in build_protocol_circuit
    cmds = circ.get_commands()
    assert sum(1 for x in cmds if x.op.type == OpType.Conditional) == EXPECTED_CONDITIONALS
    assert sum(1 for x in cmds if x.op.type == OpType.Reset) == EXPECTED_RESETS


def test_capsule_sha256_deterministic():
    p = WenPlaquetteFeedForwardPass()
    pre = Circuit(NUM_QUBITS)
    c1 = p.provenance_capsule(pre)
    c2 = WenPlaquetteFeedForwardPass().provenance_capsule(Circuit(NUM_QUBITS))
    assert c1.capsule_sha256 == c2.capsule_sha256
    assert len(c1.capsule_sha256) == 64
    assert c1.dd_defaults["apply_DD"] is True and c1.dd_defaults["DD_threshold_times"] == [0.03]
    assert all(a["policy"] == "tket_little_endian" for a in c1.endianness_audit)
    assert c1.register_map[END_REG] == 6


def test_stabilizer_parity_reconstruction():
    # even total parity -> 0, odd -> 1; malformed input rejected
    assert reconstruct_stabilizer_parity("000000", {}) == 0
    assert reconstruct_stabilizer_parity("100000", {"c0_1st": "0011"}) == 1  # 1 + 2 = 3 -> odd
    with pytest.raises(ValueError):
        reconstruct_stabilizer_parity("0000000")
    with pytest.raises(ValueError):
        reconstruct_stabilizer_parity("000000", {"c0_1st": "01x1"})
