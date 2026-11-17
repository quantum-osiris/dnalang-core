"""
dna::}{::lang — Multi-Typed Sequence Operators, Polymorphic AST Dispatcher & OpenQASM 3 Lowerer.

Enforces static frame-invariants (mod 3), epigenetic attenuation bounds (@methylate),
intron excision & wire-shifted splicing (@splice), complementarity zipping (@anneal),
chiasmic expression (}{), and scheduled IR / OpenQASM 3.0 / Qiskit lowering.
"""
from __future__ import annotations

import hashlib
import math
import os
import platform
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import ast as core_ast
from .qc_ir import Circuit, Op
from .sema import eval_expr


# ---------------------------------------------------------------------------
# Dual-Substrate Hardware Probe
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubstrateProfile:
    node_role: str
    arch: str
    timeout_horizon_s: float
    max_circuit_depth: int


def probe_substrate() -> SubstrateProfile:
    """Dynamically detects Node Alpha (ARM64 mobile edge / S26 Ultra Termux) vs Node Beta (x86_64)."""
    try:
        import osiris_hardware_mesh  # type: ignore
        fn = getattr(osiris_hardware_mesh, "get_current_profile", None) or getattr(
            osiris_hardware_mesh, "get_active_profile", None
        )
        if callable(fn):
            prof = fn()
            if isinstance(prof, SubstrateProfile):
                return prof
            arch = str(getattr(prof, "machine_arch", "") or platform.machine()).lower()
            is_edge = bool(
                getattr(prof, "is_termux", False)
                or getattr(prof, "is_android", False)
                or arch in ("aarch64", "arm64")
            )
            timeout_s = float(getattr(prof, "execution_timeout_s", 300.0 if is_edge else 1800.0))
            if is_edge:
                return SubstrateProfile("Node_Alpha_Edge", arch or "aarch64", timeout_s, 256)
            return SubstrateProfile("Node_Beta_Workstation", arch or "x86_64", timeout_s, 4096)
    except Exception:
        pass
    arch = platform.machine().lower()
    is_arm_edge = (
        arch in ("aarch64", "arm64")
        or "ANDROID_ROOT" in os.environ
        or "TERMUX_VERSION" in os.environ
    )
    if is_arm_edge:
        return SubstrateProfile("Node_Alpha_Edge", arch or "aarch64", 300.0, 256)
    return SubstrateProfile("Node_Beta_Workstation", arch or "x86_64", 1800.0, 4096)


ACTIVE_SUBSTRATE = probe_substrate()


# ---------------------------------------------------------------------------
# 1. Domain Type System & Compiler Exceptions
# ---------------------------------------------------------------------------
class DnaTypeError(TypeError):
    """Raised when an AST gate invariant or multi-typed signature fails."""


class GateMnemonic(Enum):
    IDENTITY = "id"     # Epigenetically silenced no-op
    HELIX = "helix"     # Hadamard (H)
    TWIST = "twist"     # Phase rotation RZ(theta)
    FOLD = "fold"       # Amplitude rotation RY(phi)
    BOND = "bond"       # Two-qubit CNOT(ctrl, tgt)


class OpKind(Enum):
    SPLICE = "@splice"
    ANNEAL = "@anneal"
    METHYLATE = "@methylate"
    EXPRESS = "}{"


@dataclass(frozen=True)
class DomainType:
    """Compile-time structural type descriptor attached to every AST node."""
    kind: str                       # 'Gene', 'Mask', 'Transcript', 'QuantumState'
    length: int                     # Sequence codon count or instruction count
    register_width: int = 0         # Required minimum qubit index capacity
    frame_locked: bool = True       # Enforces len % 3 == 0 on symbolic strands


# ---------------------------------------------------------------------------
# 2. Immutable AST Node Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceSpan:
    line: int
    col: int


@dataclass(frozen=True)
class ASTNode:
    span: Optional[SourceSpan] = field(default=None, compare=False, kw_only=True)

    @property
    def domain_type(self) -> DomainType:
        raise NotImplementedError


@dataclass(frozen=True)
class CodonOp:
    """Elementary genetic-quantum primitive inside a GeneNode or TranscriptSeq."""
    mnemonic: GateMnemonic
    qubits: Tuple[int, ...]
    angle: Optional[float] = None
    trait_tag: str = "neutral"

    @property
    def max_qubit(self) -> int:
        return max(self.qubits) if self.qubits else -1


_CORE_GATE_TO_MNEMONIC: Dict[str, GateMnemonic] = {
    "h": GateMnemonic.HELIX,
    "helix": GateMnemonic.HELIX,
    "rz": GateMnemonic.TWIST,
    "twist": GateMnemonic.TWIST,
    "ry": GateMnemonic.FOLD,
    "fold": GateMnemonic.FOLD,
    "cx": GateMnemonic.BOND,
    "bond": GateMnemonic.BOND,
}


@dataclass(frozen=True)
class GeneNode(ASTNode):
    """Symbolic genotype strand (`dna::[ ... ]::`)."""
    name: str
    codons: Tuple[CodonOp, ...]
    intron_spans: Tuple[Tuple[int, int], ...] = ()
    enforce_mod3: bool = True

    @property
    def domain_type(self) -> DomainType:
        width = max((c.max_qubit for c in self.codons), default=-1) + 1
        return DomainType(
            kind="Gene",
            length=len(self.codons),
            register_width=width,
            frame_locked=self.enforce_mod3,
        )

    def excise_introns(self) -> Tuple[CodonOp, ...]:
        """Strips non-coding intron indices prior to splicing or expression."""
        if not self.intron_spans:
            return self.codons
        excluded = {idx for start, end in self.intron_spans for idx in range(start, end)}
        return tuple(c for i, c in enumerate(self.codons) if i not in excluded)

    @classmethod
    def from_ast_gene(
        cls,
        gene: core_ast.Gene,
        param_values: Optional[Dict[str, float]] = None,
        qubit_map: Optional[Dict[str, int]] = None,
        enforce_mod3: bool = False,
    ) -> "GeneNode":
        """Bridges a parsed `dnalang.ast.Gene` circuit template into a typed `GeneNode`."""
        env = dict(param_values or {})
        ports = qubit_map or {p: idx for idx, p in enumerate(gene.ports)}
        codons: List[CodonOp] = []
        for stmt in gene.body:
            if isinstance(stmt, core_ast.Gate):
                mnem = _CORE_GATE_TO_MNEMONIC.get(stmt.op)
                if mnem is None:
                    raise DnaTypeError(f"Gate '{stmt.op}' cannot be lifted to a CodonOp mnemonic.")
                wires = tuple(ports[q] for q in stmt.qubits)
                ang = eval_expr(stmt.args[0], env) if stmt.args else None
                codons.append(CodonOp(mnem, wires, ang, trait_tag=gene.name))
        span = SourceSpan(gene.pos.line, gene.pos.col) if getattr(gene, "pos", None) else None
        return cls(name=gene.name, codons=tuple(codons), enforce_mod3=enforce_mod3, span=span)


@dataclass(frozen=True)
class EpigeneticMaskNode(ASTNode):
    """Attenuation weights in [0.0, 1.0] applied via `@methylate`."""
    name: str
    weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        for idx, w in enumerate(self.weights):
            if not (0.0 <= w <= 1.0):
                raise DnaTypeError(
                    f"EpigeneticMask '{self.name}' weight[{idx}]={w} outside [0.0, 1.0]."
                )

    @property
    def domain_type(self) -> DomainType:
        return DomainType(kind="Mask", length=len(self.weights), frame_locked=False)


@dataclass(frozen=True)
class TranscriptSeqNode(ASTNode):
    """Lowered unitary instruction stream ready for chiasmic `}{` expression."""
    ops: Tuple[CodonOp, ...]
    provenance: Tuple[str, ...] = ()

    @property
    def domain_type(self) -> DomainType:
        width = max((op.max_qubit for op in self.ops), default=-1) + 1
        return DomainType(
            kind="Transcript",
            length=len(self.ops),
            register_width=width,
            frame_locked=False,
        )

    def adjoint(self) -> "TranscriptSeqNode":
        """Unitary inversion (`~`): reverses op order and negates rotation angles."""
        inverted = tuple(
            replace(op, angle=-op.angle if op.angle is not None else None)
            for op in reversed(self.ops)
        )
        return TranscriptSeqNode(
            ops=inverted,
            provenance=self.provenance + ("adjoint(~)",),
            span=self.span,
        )


@dataclass(frozen=True)
class QuantumStateNode(ASTNode):
    """Target Hilbert-space register state (`QuantumState<Q>`) with circuit history."""
    register_id: str
    num_qubits: int
    applied_ir: Tuple[CodonOp, ...] = ()
    entropy_floor: float = 0.0

    @property
    def domain_type(self) -> DomainType:
        return DomainType(
            kind="QuantumState",
            length=len(self.applied_ir),
            register_width=self.num_qubits,
            frame_locked=False,
        )


@dataclass(frozen=True)
class BinarySeqOpNode(ASTNode):
    """Unresolved AST binary expression (`left <op> right`)."""
    op: OpKind
    left: ASTNode
    right: ASTNode
    insertion_offset: int = 0


# ---------------------------------------------------------------------------
# 3. Multi-Typed Operator Dispatch Engine
# ---------------------------------------------------------------------------
DispatchKey = Tuple[OpKind, str, str]
HandlerFn = Callable[[ASTNode, ASTNode, BinarySeqOpNode], ASTNode]

COMPLEMENTARY_TRAITS: Dict[str, str] = {
    "purine_A": "pyrimidine_T",
    "pyrimidine_T": "purine_A",
    "purine_G": "pyrimidine_C",
    "pyrimidine_C": "purine_G",
    "stabilizer_X": "stabilizer_Z",
    "stabilizer_Z": "stabilizer_X",
}


class MultiTypedDispatcher:
    """Type-directed multiple-dispatch resolver for dna::}{::lang AST operators."""

    def __init__(self, substrate: SubstrateProfile = ACTIVE_SUBSTRATE) -> None:
        self.substrate = substrate
        self._table: Dict[DispatchKey, HandlerFn] = {}
        self._register_core_operators()

    def register(self, op: OpKind, left_kind: str, right_kind: str) -> Callable[[HandlerFn], HandlerFn]:
        def decorator(fn: HandlerFn) -> HandlerFn:
            self._table[(op, left_kind, right_kind)] = fn
            return fn
        return decorator

    def evaluate(self, node: ASTNode) -> ASTNode:
        """Recursively resolves BinarySeqOpNodes into verified AST / IR nodes."""
        if not isinstance(node, BinarySeqOpNode):
            return node

        lhs = self.evaluate(node.left)
        rhs = self.evaluate(node.right)
        key: DispatchKey = (node.op, lhs.domain_type.kind, rhs.domain_type.kind)

        handler = self._table.get(key)
        if handler is None:
            raise DnaTypeError(
                f"No overload for operator '{node.op.value}' with operand types "
                f"({lhs.domain_type.kind}, {rhs.domain_type.kind})."
            )
        return handler(lhs, rhs, node)

    def _register_core_operators(self) -> None:

        @self.register(OpKind.METHYLATE, "Mask", "Gene")
        def _methylate_gene(lhs: ASTNode, rhs: ASTNode, expr: BinarySeqOpNode) -> GeneNode:
            assert isinstance(lhs, EpigeneticMaskNode) and isinstance(rhs, GeneNode)
            if len(lhs.weights) != len(rhs.codons):
                raise DnaTypeError(
                    f"@methylate length mismatch: Mask '{lhs.name}' ({len(lhs.weights)}) "
                    f"!= Gene '{rhs.name}' ({len(rhs.codons)})."
                )

            modulated: List[CodonOp] = []
            for weight, codon in zip(lhs.weights, rhs.codons):
                if math.isclose(weight, 0.0, abs_tol=1e-12):
                    modulated.append(CodonOp(GateMnemonic.IDENTITY, codon.qubits, None, codon.trait_tag))
                elif codon.angle is not None and codon.mnemonic in (GateMnemonic.TWIST, GateMnemonic.FOLD):
                    modulated.append(replace(codon, angle=weight * codon.angle))
                else:
                    modulated.append(codon)

            return replace(rhs, name=f"{rhs.name}[methyl:{lhs.name}]", codons=tuple(modulated), span=expr.span)

        @self.register(OpKind.SPLICE, "Gene", "Gene")
        def _splice_genes(lhs: ASTNode, rhs: ASTNode, expr: BinarySeqOpNode) -> GeneNode:
            assert isinstance(lhs, GeneNode) and isinstance(rhs, GeneNode)
            exons_a = lhs.excise_introns()
            exons_b = rhs.excise_introns()
            joined = exons_a + exons_b

            enforce_frame = lhs.enforce_mod3 or rhs.enforce_mod3
            if enforce_frame and (len(joined) % 3 != 0):
                raise DnaTypeError(
                    f"@splice frame-shift violation: '{lhs.name}' ({len(exons_a)} exons) + "
                    f"'{rhs.name}' ({len(exons_b)} exons) = {len(joined)} codons (not ≡ 0 mod 3)."
                )

            return GeneNode(
                name=f"({lhs.name}+splice+{rhs.name})",
                codons=joined,
                intron_spans=(),
                enforce_mod3=enforce_frame,
                span=expr.span,
            )

        @self.register(OpKind.SPLICE, "Gene", "QuantumState")
        def _splice_gene_into_register(lhs: ASTNode, rhs: ASTNode, expr: BinarySeqOpNode) -> QuantumStateNode:
            assert isinstance(lhs, GeneNode) and isinstance(rhs, QuantumStateNode)
            offset = expr.insertion_offset
            remapped_ops: List[CodonOp] = []

            for codon in lhs.excise_introns():
                if codon.mnemonic == GateMnemonic.IDENTITY:
                    continue
                shifted_qubits = tuple(q + offset for q in codon.qubits)
                if max(shifted_qubits, default=-1) >= rhs.num_qubits or min(shifted_qubits, default=0) < 0:
                    raise DnaTypeError(
                        f"@splice qubit overflow: shifted wires {shifted_qubits} exceed "
                        f"QuantumState '{rhs.register_id}' capacity ({rhs.num_qubits} qubits)."
                    )
                remapped_ops.append(replace(codon, qubits=shifted_qubits))

            return self._append_ir_checked(rhs, tuple(remapped_ops), expr.span)

        @self.register(OpKind.ANNEAL, "Gene", "Gene")
        def _anneal_strands(lhs: ASTNode, rhs: ASTNode, expr: BinarySeqOpNode) -> TranscriptSeqNode:
            assert isinstance(lhs, GeneNode) and isinstance(rhs, GeneNode)
            exons_a = lhs.excise_introns()
            exons_b = rhs.excise_introns()
            if len(exons_a) != len(exons_b):
                raise DnaTypeError(
                    f"@anneal requires equal exon lengths, got {len(exons_a)} and {len(exons_b)}."
                )

            duplex_ops: List[CodonOp] = []
            mismatch_penalty_rad = 0.15 * math.pi

            for ca, cb in zip(exons_a, exons_b):
                if ca.mnemonic != GateMnemonic.IDENTITY:
                    duplex_ops.append(ca)
                if cb.mnemonic != GateMnemonic.IDENTITY:
                    duplex_ops.append(cb)

                if not ca.qubits or not cb.qubits:
                    continue
                qa, qb = ca.qubits[0], cb.qubits[0]
                if qa == qb:
                    raise DnaTypeError(
                        f"@anneal wire collision on q{qa}: complementary strands must target distinct wires."
                    )

                if COMPLEMENTARY_TRAITS.get(ca.trait_tag) == cb.trait_tag:
                    duplex_ops.append(CodonOp(GateMnemonic.BOND, (qa, qb), None, "annealed_bond"))
                else:
                    duplex_ops.append(CodonOp(GateMnemonic.TWIST, (qa,), mismatch_penalty_rad, "mismatch_tension"))

            return TranscriptSeqNode(
                ops=tuple(duplex_ops),
                provenance=(f"anneal({lhs.name},{rhs.name})",),
                span=expr.span,
            )

        @self.register(OpKind.EXPRESS, "Transcript", "QuantumState")
        def _express_transcript(lhs: ASTNode, rhs: ASTNode, expr: BinarySeqOpNode) -> QuantumStateNode:
            assert isinstance(lhs, TranscriptSeqNode) and isinstance(rhs, QuantumStateNode)
            if lhs.domain_type.register_width > rhs.num_qubits:
                raise DnaTypeError(
                    f"Chiasmic '}}{{' width violation: Transcript requires {lhs.domain_type.register_width} "
                    f"qubits, but register '{rhs.register_id}' has only {rhs.num_qubits}."
                )
            active_ops = tuple(op for op in lhs.ops if op.mnemonic != GateMnemonic.IDENTITY)
            return self._append_ir_checked(rhs, active_ops, expr.span)

        @self.register(OpKind.EXPRESS, "Gene", "QuantumState")
        def _express_gene_direct(lhs: ASTNode, rhs: ASTNode, expr: BinarySeqOpNode) -> QuantumStateNode:
            assert isinstance(lhs, GeneNode) and isinstance(rhs, QuantumStateNode)
            exons = lhs.excise_introns()
            if lhs.enforce_mod3 and (len(exons) % 3 != 0):
                raise DnaTypeError(
                    f"Cannot express Gene '{lhs.name}' via '}}{{' — exon length {len(exons)} violates mod-3 frame."
                )
            transcript = TranscriptSeqNode(ops=exons, provenance=(f"transcribe({lhs.name})",), span=lhs.span)
            return _express_transcript(transcript, rhs, expr)

    def _append_ir_checked(
        self,
        state: QuantumStateNode,
        new_ops: Tuple[CodonOp, ...],
        span: Optional[SourceSpan],
    ) -> QuantumStateNode:
        combined = state.applied_ir + new_ops
        if len(combined) > self.substrate.max_circuit_depth:
            raise DnaTypeError(
                f"Circuit depth {len(combined)} exceeds {self.substrate.node_role} horizon "
                f"({self.substrate.max_circuit_depth} ops)."
            )
        return replace(state, applied_ir=combined, span=span)


# ---------------------------------------------------------------------------
# 4. Canonical QuantumCircuitIR & Scheduled OpenQASM 3.0 / Qiskit Lowering
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IRInstruction:
    """Target-agnostic lowered quantum instruction."""
    opcode: str
    qubits: Tuple[int, ...]
    angle_rad: Optional[float] = None
    duration_dt: Optional[int] = None
    clbits: Tuple[int, ...] = ()
    comment: str = ""


@dataclass(frozen=True)
class LoweringConfig:
    """Controls barrier granularity, dynamical-decoupling idle delays, and readout."""
    barrier_policy: str = "epoch"       # 'none', 'entangle', 'epoch'
    insert_idle_delays: bool = True
    single_qubit_dt: int = 160
    two_qubit_dt: int = 640
    dt_seconds: float = 0.2222222222e-9
    measure_all: bool = True
    float_precision: int = 8


@dataclass(frozen=True)
class QuantumCircuitIR:
    """Immutable lowered circuit artifact with cryptographic provenance."""
    register_id: str
    num_qubits: int
    num_clbits: int
    instructions: Tuple[IRInstruction, ...]
    substrate_role: str
    entropy_floor: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def sha256_digest(self) -> str:
        """Deterministic digest over register geometry and instruction stream."""
        hasher = hashlib.sha256()
        hasher.update(f"{self.register_id}:{self.num_qubits}:{self.num_clbits}\n".encode("utf-8"))
        for inst in self.instructions:
            angle_repr = f"{inst.angle_rad:.10f}" if inst.angle_rad is not None else ""
            dur_repr = str(inst.duration_dt) if inst.duration_dt is not None else ""
            rec = f"{inst.opcode}|{inst.qubits}|{angle_repr}|{dur_repr}|{inst.clbits}\n"
            hasher.update(rec.encode("utf-8"))
        return hasher.hexdigest()

    def to_qc_ir(self, dt_seconds: float = 0.2222222222e-9) -> Circuit:
        """Converts lowered `QuantumCircuitIR` into `dnalang.qc_ir.Circuit`."""
        circ = Circuit(n_qubits=self.num_qubits, n_cbits=self.num_clbits, name=self.register_id)
        for inst in self.instructions:
            if inst.opcode in ("h", "cx"):
                circ.ops.append(Op(inst.opcode, inst.qubits))
            elif inst.opcode in ("rz", "ry"):
                assert inst.angle_rad is not None
                circ.ops.append(Op(inst.opcode, inst.qubits, (inst.angle_rad,)))
            elif inst.opcode == "delay":
                assert inst.duration_dt is not None
                circ.ops.append(Op("delay", inst.qubits, duration_s=inst.duration_dt * dt_seconds))
            elif inst.opcode == "barrier":
                circ.ops.append(Op("barrier", inst.qubits))
            elif inst.opcode == "measure":
                circ.ops.append(Op("measure", inst.qubits, cbit=inst.clbits[0]))
        return circ


_MNEMONIC_TO_OPCODE: Dict[GateMnemonic, str] = {
    GateMnemonic.HELIX: "h",
    GateMnemonic.TWIST: "rz",
    GateMnemonic.FOLD: "ry",
    GateMnemonic.BOND: "cx",
}


class DnaLangLowerer:
    """Lowers verified QuantumStateNode AST artifacts to QuantumCircuitIR, QASM 3, and Qiskit."""

    def __init__(
        self,
        config: LoweringConfig = LoweringConfig(),
        substrate: SubstrateProfile = ACTIVE_SUBSTRATE,
    ) -> None:
        self.config = config
        self.substrate = substrate

    def lower_to_ir(self, state: QuantumStateNode) -> QuantumCircuitIR:
        if state.num_qubits <= 0:
            raise DnaTypeError(f"Register '{state.register_id}' must allocate at least 1 qubit.")

        all_wires = tuple(range(state.num_qubits))
        wire_clock_dt: List[int] = [0] * state.num_qubits
        epoch_ceiling_dt: int = 0
        ir_stream: List[IRInstruction] = []

        active_codons = [c for c in state.applied_ir if c.mnemonic != GateMnemonic.IDENTITY]

        for idx, codon in enumerate(active_codons):
            self._validate_codon(codon, state.num_qubits)
            opcode = _MNEMONIC_TO_OPCODE[codon.mnemonic]

            if self.config.barrier_policy == "entangle" and opcode == "cx":
                epoch_ceiling_dt = self._flush_epoch_barrier(
                    ir_stream, wire_clock_dt, epoch_ceiling_dt, codon.qubits, "pre-bond sync"
                )

            gate_dt = self.config.two_qubit_dt if opcode == "cx" else self.config.single_qubit_dt
            start_dt = max(wire_clock_dt[q] for q in codon.qubits)
            end_dt = start_dt + gate_dt
            for q in codon.qubits:
                wire_clock_dt[q] = end_dt

            ir_stream.append(
                IRInstruction(
                    opcode=opcode,
                    qubits=codon.qubits,
                    angle_rad=codon.angle,
                    comment=codon.trait_tag,
                )
            )

            is_frame_boundary = (idx + 1) % 3 == 0
            is_annealed_feature = codon.trait_tag in ("annealed_bond", "mismatch_tension")

            if self.config.barrier_policy == "entangle" and opcode == "cx":
                epoch_ceiling_dt = self._flush_epoch_barrier(
                    ir_stream, wire_clock_dt, epoch_ceiling_dt, all_wires, "post-bond sync"
                )
            elif self.config.barrier_policy == "epoch" and (is_frame_boundary or is_annealed_feature):
                label = "duplex-anneal sync" if is_annealed_feature else f"codon-frame {(idx + 1) // 3}"
                epoch_ceiling_dt = self._flush_epoch_barrier(
                    ir_stream, wire_clock_dt, epoch_ceiling_dt, all_wires, label
                )

        num_clbits = state.num_qubits if self.config.measure_all else 0
        if self.config.measure_all:
            if not ir_stream or ir_stream[-1].opcode != "barrier":
                epoch_ceiling_dt = self._flush_epoch_barrier(
                    ir_stream, wire_clock_dt, epoch_ceiling_dt, all_wires, "readout boundary"
                )
            for q in all_wires:
                ir_stream.append(
                    IRInstruction(opcode="measure", qubits=(q,), clbits=(q,), comment="z_readout")
                )

        return QuantumCircuitIR(
            register_id=state.register_id,
            num_qubits=state.num_qubits,
            num_clbits=num_clbits,
            instructions=tuple(ir_stream),
            substrate_role=self.substrate.node_role,
            entropy_floor=state.entropy_floor,
            metadata={
                "arch": self.substrate.arch,
                "barrier_policy": self.config.barrier_policy,
            },
        )

    def _flush_epoch_barrier(
        self,
        ir_stream: List[IRInstruction],
        wire_clock_dt: List[int],
        prev_ceiling_dt: int,
        barrier_wires: Tuple[int, ...],
        label: str,
    ) -> int:
        new_ceiling_dt = max(wire_clock_dt) if wire_clock_dt else prev_ceiling_dt

        if self.config.insert_idle_delays and new_ceiling_dt > prev_ceiling_dt:
            for q in barrier_wires:
                slack_dt = new_ceiling_dt - wire_clock_dt[q]
                if slack_dt > 0:
                    ir_stream.append(
                        IRInstruction(
                            opcode="delay",
                            qubits=(q,),
                            duration_dt=slack_dt,
                            comment="idle_slack",
                        )
                    )
                    wire_clock_dt[q] = new_ceiling_dt

        ir_stream.append(IRInstruction(opcode="barrier", qubits=barrier_wires, comment=label))
        return new_ceiling_dt

    @staticmethod
    def _validate_codon(codon: CodonOp, num_qubits: int) -> None:
        if codon.mnemonic not in _MNEMONIC_TO_OPCODE:
            raise DnaTypeError(f"Unsupported mnemonic '{codon.mnemonic}' during IR lowering.")

        for q in codon.qubits:
            if q < 0 or q >= num_qubits:
                raise DnaTypeError(f"Qubit index q[{q}] out of bounds for width {num_qubits}.")

        if codon.mnemonic == GateMnemonic.BOND:
            if len(codon.qubits) != 2 or codon.qubits[0] == codon.qubits[1]:
                raise DnaTypeError(f"BOND requires 2 distinct qubits, got {codon.qubits}.")
        else:
            if len(codon.qubits) != 1:
                raise DnaTypeError(f"{codon.mnemonic.value} requires 1 qubit, got {codon.qubits}.")

        if codon.mnemonic in (GateMnemonic.TWIST, GateMnemonic.FOLD):
            if codon.angle is None or not math.isfinite(codon.angle):
                raise DnaTypeError(f"{codon.mnemonic.value} requires a finite angle, got {codon.angle}.")

    def to_openqasm3(self, state_or_ir: QuantumStateNode | QuantumCircuitIR) -> str:
        ir = self.lower_to_ir(state_or_ir) if isinstance(state_or_ir, QuantumStateNode) else state_or_ir
        qreg = self._sanitize_ident(ir.register_id)
        creg = f"c_{qreg}"

        lines: List[str] = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            "",
            f"// dna::}}{{::lang Lowered Artifact | Substrate: {ir.substrate_role} ({ir.metadata.get('arch', 'unknown')})",
            f"// Provenance SHA-256: {ir.sha256_digest}",
            f"// NCLM Entropy Floor: {ir.entropy_floor:.6f}",
            f"qubit[{ir.num_qubits}] {qreg};",
        ]
        if ir.num_clbits > 0:
            lines.append(f"bit[{ir.num_clbits}] {creg};")
        lines.append("")

        for inst in ir.instructions:
            sfx = f"  // {inst.comment}" if inst.comment else ""
            if inst.opcode == "h":
                lines.append(f"h {qreg}[{inst.qubits[0]}];{sfx}")
            elif inst.opcode in ("rz", "ry"):
                assert inst.angle_rad is not None
                ang = self._format_qasm_angle(inst.angle_rad)
                lines.append(f"{inst.opcode}({ang}) {qreg}[{inst.qubits[0]}];{sfx}")
            elif inst.opcode == "cx":
                c, t = inst.qubits
                lines.append(f"cx {qreg}[{c}], {qreg}[{t}];{sfx}")
            elif inst.opcode == "delay":
                lines.append(f"delay[{inst.duration_dt}dt] {qreg}[{inst.qubits[0]}];{sfx}")
            elif inst.opcode == "barrier":
                wire_list = ", ".join(f"{qreg}[{q}]" for q in inst.qubits)
                lines.append(f"barrier {wire_list};{sfx}")
            elif inst.opcode == "measure":
                lines.append(f"{creg}[{inst.clbits[0]}] = measure {qreg}[{inst.qubits[0]}];")
            else:
                raise DnaTypeError(f"Unhandled IR opcode '{inst.opcode}' in OpenQASM 3 emitter.")

        return "\n".join(lines) + "\n"

    def _format_qasm_angle(self, rad: float) -> str:
        if math.isclose(rad, 0.0, abs_tol=1e-12):
            return "0.0"
        ratio = rad / math.pi
        for denom in (1, 2, 3, 4, 6, 8, 12, 16, 20):
            numer = round(ratio * denom)
            if numer != 0 and math.isclose(ratio, numer / denom, abs_tol=1e-9):
                sign = "-" if numer < 0 else ""
                abs_num = abs(numer)
                if denom == 1:
                    return f"{sign}pi" if abs_num == 1 else f"{sign}{abs_num}*pi"
                num_prefix = "" if abs_num == 1 else f"{abs_num}*"
                return f"{sign}{num_prefix}pi/{denom}"
        return f"{rad:.{self.config.float_precision}f}"

    @staticmethod
    def _sanitize_ident(name: str) -> str:
        cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
        return f"q_{cleaned}" if not cleaned or cleaned[0].isdigit() else cleaned

    def to_qiskit(self, state_or_ir: QuantumStateNode | QuantumCircuitIR) -> Any:
        ir = self.lower_to_ir(state_or_ir) if isinstance(state_or_ir, QuantumStateNode) else state_or_ir

        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        except ImportError as exc:
            raise RuntimeError(
                f"Qiskit is not installed on {self.substrate.node_role} ({self.substrate.arch}). "
                "Use `to_openqasm3()` on lightweight edge nodes or `pip install qiskit`."
            ) from exc

        qreg_name = self._sanitize_ident(ir.register_id)
        qreg = QuantumRegister(ir.num_qubits, name=qreg_name)
        if ir.num_clbits > 0:
            creg = ClassicalRegister(ir.num_clbits, name=f"c_{qreg_name}")
            qc = QuantumCircuit(qreg, creg, name=qreg_name)
        else:
            creg = None
            qc = QuantumCircuit(qreg, name=qreg_name)

        qc.metadata = {
            "dnalang_sha256": ir.sha256_digest,
            "substrate_role": ir.substrate_role,
            "entropy_floor": ir.entropy_floor,
            **ir.metadata,
        }

        for inst in ir.instructions:
            if inst.opcode == "h":
                qc.h(qreg[inst.qubits[0]])
            elif inst.opcode == "rz":
                assert inst.angle_rad is not None
                qc.rz(inst.angle_rad, qreg[inst.qubits[0]])
            elif inst.opcode == "ry":
                assert inst.angle_rad is not None
                qc.ry(inst.angle_rad, qreg[inst.qubits[0]])
            elif inst.opcode == "cx":
                qc.cx(qreg[inst.qubits[0]], qreg[inst.qubits[1]])
            elif inst.opcode == "delay":
                assert inst.duration_dt is not None
                qc.delay(inst.duration_dt, qreg[inst.qubits[0]], unit="dt")
            elif inst.opcode == "barrier":
                qc.barrier(*[qreg[q] for q in inst.qubits], label=inst.comment or None)
            elif inst.opcode == "measure":
                assert creg is not None
                qc.measure(qreg[inst.qubits[0]], creg[inst.clbits[0]])
            else:
                raise DnaTypeError(f"Unhandled IR opcode '{inst.opcode}' in Qiskit synthesizer.")

        return qc
