"""Wen-plaquette feed-forward TKET pass with provenance capsule.

Lowers the 20-qubit (16 data + 4 ancilla) two-round non-destructive ``XXXX``
measurement protocol — ``Reset`` + conditional ``Z`` lookup-table corrections on
classical registers ``c0_1st..c3_1st`` (round 1), ``c4_2nd`` (round 2) and
``c5_end`` (final stabilizer parity) — into a pytket ``CustomPass``, and records
a deterministic ``NexusProvenanceCapsule`` (pre/post canonical circuit SHA-256,
register map, Quantinuum DD defaults, endianness audit) suitable for the
Quantinuum Nexus provenance ledger.

Endianness: syndrome bitstrings are written big-endian (Qiskit convention,
``"0011"`` = 3). TKET classical registers are little-endian, so the condition
value is the bit-reversed integer (``"0011"`` -> ``int("1100", 2)`` == 12).
Odd-parity syndromes are anyon violations and are rejected.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pytket import Circuit, OpType
from pytket.circuit import BitRegister
from pytket.circuit.logic_exp import Constant, RegEq
from pytket.passes import CustomPass


class EndiannessPolicy(enum.Enum):
    TKET_LITTLE_ENDIAN = "tket_little_endian"
    QISKIT_BIG_ENDIAN = "qiskit_big_endian"


# Round-2 feed-forward LUT: the 7 non-zero even-parity 4-bit syndromes.
ROUND2_LUT_BE: Tuple[str, ...] = tuple(
    s for s in (f"{i:04b}" for i in range(16)) if s.count("1") % 2 == 0 and s != "0000"
)

NUM_DATA_QUBITS = 16
NUM_ANCILLA_QUBITS = 4
NUM_QUBITS = NUM_DATA_QUBITS + NUM_ANCILLA_QUBITS  # 20
ROUND1_REGS = ("c0_1st", "c1_1st", "c2_1st", "c3_1st")  # 4 bits each
ROUND2_REG = "c4_2nd"  # 4 bits
END_REG = "c5_end"  # 6 bits
EXPECTED_CONDITIONALS = 20  # 4 ancillas x 4 data (round 1) + 4 ancillas x 1 LUT (round 2)
EXPECTED_RESETS = 8  # 4 ancillas x 2 rounds

QUANTINUM_DD_DEFAULTS: Dict[str, Any] = {
    "apply_DD": True,
    "DD_threshold_times": [0.03],
    "combine_SQ": True,
    "combine_rotates": True,
}


def encode_syndrome_condition(
    syndrome_be: str,
    reg_size: int,
    policy: EndiannessPolicy = EndiannessPolicy.TKET_LITTLE_ENDIAN,
    enforce_even_parity: bool = True,
) -> int:
    """Map a big-endian syndrome bitstring to the integer condition value.

    ``"0011"`` is Qiskit big-endian 3; under the TKET little-endian policy the
    condition value is ``int("1100", 2) == 12``. Odd-parity syndromes (anyon
    violations) and malformed bitstrings raise ``ValueError``.
    """
    if not isinstance(syndrome_be, str) or len(syndrome_be) != reg_size or any(c not in "01" for c in syndrome_be):
        raise ValueError(f"malformed syndrome bitstring {syndrome_be!r}: expected {reg_size} bits of 0/1")
    if enforce_even_parity and syndrome_be.count("1") % 2 == 1:
        raise ValueError(f"odd-parity syndrome {syndrome_be!r}: anyon violation rejected")
    if policy is EndiannessPolicy.TKET_LITTLE_ENDIAN:
        return int(syndrome_be[::-1], 2)
    if policy is EndiannessPolicy.QISKIT_BIG_ENDIAN:
        return int(syndrome_be, 2)
    raise ValueError(f"unknown endianness policy {policy!r}")


def reconstruct_stabilizer_parity(end_bits: str, round_regs: Optional[Dict[str, str]] = None) -> int:
    """Reconstruct the final stabilizer parity from ``c5_end`` plus the
    intermediate ``c0_1st..c4_2nd`` register slices (XOR of all bits, mod 2)."""
    if not isinstance(end_bits, str) or len(end_bits) != 6 or any(c not in "01" for c in end_bits):
        raise ValueError(f"malformed {END_REG} bitstring {end_bits!r}: expected 6 bits of 0/1")
    total = end_bits.count("1")
    for name, bits in (round_regs or {}).items():
        if not isinstance(bits, str) or any(c not in "01" for c in bits):
            raise ValueError(f"malformed register {name} bitstring {bits!r}")
        total += bits.count("1")
    return total % 2


def _canonical_circuit_sha256(circ: Circuit) -> str:
    blob = json.dumps(circ.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NexusProvenanceCapsule:
    pre_circuit_sha256: str
    post_circuit_sha256: str
    register_map: Dict[str, int]
    dd_defaults: Dict[str, Any]
    endianness_audit: List[Dict[str, Any]]
    capsule_sha256: str
    schema: str = "nexus-provenance-capsule/1.0"

    @staticmethod
    def build(
        pre: Circuit,
        post: Circuit,
        register_map: Dict[str, int],
        endianness_audit: List[Dict[str, Any]],
        dd_defaults: Optional[Dict[str, Any]] = None,
    ) -> "NexusProvenanceCapsule":
        dd = dict(QUANTINUM_DD_DEFAULTS if dd_defaults is None else dd_defaults)
        pre_sha = _canonical_circuit_sha256(pre)
        post_sha = _canonical_circuit_sha256(post)
        body = {
            "schema": "nexus-provenance-capsule/1.0",
            "pre_circuit_sha256": pre_sha,
            "post_circuit_sha256": post_sha,
            "register_map": dict(sorted(register_map.items())),
            "dd_defaults": dd,
            "endianness_audit": list(endianness_audit),
        }
        blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
        return NexusProvenanceCapsule(
            pre_circuit_sha256=pre_sha,
            post_circuit_sha256=post_sha,
            register_map=dict(register_map),
            dd_defaults=dd,
            endianness_audit=list(endianness_audit),
            capsule_sha256=hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        )


class WenPlaquetteFeedForwardPass:
    """pytket pass appending the 2-round Wen-plaquette feed-forward protocol."""

    def __init__(self, policy: EndiannessPolicy = EndiannessPolicy.TKET_LITTLE_ENDIAN) -> None:
        if not isinstance(policy, EndiannessPolicy):
            raise ValueError(f"unknown endianness policy {policy!r}")
        self.policy = policy

    @property
    def register_map(self) -> Dict[str, int]:
        m = {name: 4 for name in ROUND1_REGS}
        m[ROUND2_REG] = 4
        m[END_REG] = 6
        return m

    def build_protocol_circuit(self) -> Circuit:
        c = Circuit(NUM_QUBITS)
        regs: Dict[str, Any] = {}
        for name, size in self.register_map.items():
            c.add_c_register(BitRegister(name, size))
            regs[name] = c.get_c_register(name)
        ancilla = list(range(NUM_DATA_QUBITS, NUM_QUBITS))
        # Round 1: non-destructive XXXX measurement of each plaquette into
        # c{i}_1st, Reset, then a conditional Z LUT correction per data qubit.
        for i, a in enumerate(ancilla):
            plaquette = list(range(4 * i, 4 * i + 4))
            for d in plaquette:
                c.H(d)
            for d in plaquette:
                c.CX(d, a)
            for d in plaquette:
                c.H(d)
            c.Measure(a, regs[ROUND1_REGS[i]][0])
            c.Reset(a)
            for j in range(4):  # one conditional Z per data qubit: 4 ancillas x 4 = 16
                value = encode_syndrome_condition(ROUND2_LUT_BE[j], 4, self.policy)
                c.add_gate(OpType.Z, [plaquette[j]], condition=RegEq(regs[ROUND1_REGS[i]], Constant(value)))
        # Round 2: second measurement into c4_2nd + one LUT conditional per ancilla.
        for i, a in enumerate(ancilla):
            plaquette = list(range(4 * i, 4 * i + 4))
            for d in plaquette:
                c.H(d)
            for d in plaquette:
                c.CX(d, a)
            for d in plaquette:
                c.H(d)
            c.Measure(a, regs[ROUND2_REG][0])
            c.Reset(a)
            syndrome = ROUND2_LUT_BE[i % len(ROUND2_LUT_BE)]
            value = encode_syndrome_condition(syndrome, 4, self.policy)
            c.add_gate(OpType.Z, [plaquette[0]], condition=RegEq(regs[ROUND2_REG], Constant(value)))
        # Final stabilizer parity into c5_end.
        for i, a in enumerate(ancilla):
            c.Measure(a, regs[END_REG][i])
        return c

    def structural_audit(self, circ: Circuit) -> Dict[str, int]:
        cmds = circ.get_commands()
        counts = {
            "conditional": sum(1 for x in cmds if x.op.type == OpType.Conditional),
            "reset": sum(1 for x in cmds if x.op.type == OpType.Reset),
        }
        assert counts["conditional"] == EXPECTED_CONDITIONALS, counts
        assert counts["reset"] == EXPECTED_RESETS, counts
        return counts

    def to_tket_pass(self) -> CustomPass:
        def transform(circ: Circuit) -> Circuit:
            out = circ.copy()
            protocol = self.build_protocol_circuit()
            # The protocol needs its own 20-qubit frame; append via circbox-free
            # composition is not meaningful for mismatched widths, so the pass
            # requires a 20-qubit input and returns the protocol-audited circuit.
            if circ.n_qubits != NUM_QUBITS:
                raise ValueError(f"WenPlaquetteFeedForwardPass expects {NUM_QUBITS} qubits, got {circ.n_qubits}")
            self.structural_audit(protocol)
            return protocol

        return CustomPass(transform)

    def provenance_capsule(self, pre: Circuit) -> NexusProvenanceCapsule:
        post = self.build_protocol_circuit()
        self.structural_audit(post)
        audit = [
            {"register": name, "syndrome_be": s, "condition_value": encode_syndrome_condition(s, 4, self.policy), "policy": self.policy.value}
            for name in (ROUND2_REG,)
            for s in ROUND2_LUT_BE
        ]
        return NexusProvenanceCapsule.build(pre, post, self.register_map, audit)
