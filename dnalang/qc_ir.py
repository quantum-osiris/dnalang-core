"""Flat circuit IR: a list of operations on physical qubits with exact rational durations.

Enforces logical-IR structural invariants (`Circuit.validate()`) and explicit timing
conversion (`Circuit.resolve_timing(dt, policy="exact"|"quantize")`) at all export
boundaries (`to_qasm3`, `to_qiskit`, `canonical`).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, List, Optional, Tuple, Union

from .timing import (
    IRValidationError,
    QuantizedDelayRecord,
    TimingError,
    TimingResolutionReport,
    canonical_duration_str,
    coerce_exact_seconds,
    format_exact_ns_qasm,
    resolve_delay_ticks,
)

ONE_Q = {"h", "x", "y", "z", "s", "t", "sx", "rx", "ry", "rz"}
TWO_Q = {"cx", "cz", "swap"}
PARAM_GATES = {"rx", "ry", "rz"}


@dataclass
class Op:
    name: str                                    # gate name | delay | barrier | measure
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()
    duration_s: Optional[float] = None           # delay only (float view in seconds)
    cbit: Optional[int] = None                   # measure only
    duration_exact_s: Optional[Fraction] = None  # exact rational seconds (delay only)

    def __post_init__(self) -> None:
        if self.duration_exact_s is not None and self.duration_s is None:
            self.duration_s = float(self.duration_exact_s)
        elif (
            self.duration_s is not None
            and self.duration_exact_s is None
            and isinstance(self.duration_s, (int, float, Decimal, Fraction))
            and not isinstance(self.duration_s, bool)
            and math.isfinite(float(self.duration_s))
        ):
            self.duration_exact_s = coerce_exact_seconds(self.duration_s)


@dataclass
class Circuit:
    n_qubits: int
    n_cbits: int
    ops: List[Op] = field(default_factory=list)
    name: str = "circuit"

    # --- logical-IR validation boundary (S1-03) ------------------------------
    def validate(self) -> "Circuit":
        """Validate all logical-IR structural, parameter, qubit, cbit, and duration invariants."""
        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int) or self.n_qubits < 0:
            raise IRValidationError(f"Circuit.n_qubits must be a non-negative integer (got {self.n_qubits!r})")
        if isinstance(self.n_cbits, bool) or not isinstance(self.n_cbits, int) or self.n_cbits < 0:
            raise IRValidationError(f"Circuit.n_cbits must be a non-negative integer (got {self.n_cbits!r})")

        for idx, o in enumerate(self.ops):
            if not isinstance(o, Op):
                raise IRValidationError(f"Op #{idx} is not an Op instance (got {type(o).__name__})")
            if not isinstance(o.qubits, tuple) or len(o.qubits) == 0:
                raise IRValidationError(f"Op #{idx} ({o.name}) must have a non-empty qubit tuple")
            for q in o.qubits:
                if isinstance(q, bool) or not isinstance(q, int) or q < 0 or q >= self.n_qubits:
                    raise IRValidationError(
                        f"Op #{idx} ({o.name}) qubit index {q!r} out of range [0, {self.n_qubits})"
                    )
            if len(set(o.qubits)) != len(o.qubits):
                raise IRValidationError(
                    f"Op #{idx} ({o.name}) binds duplicate/aliased physical qubits {o.qubits}"
                )

            if o.name in ONE_Q or o.name in TWO_Q:
                expected_q = 1 if o.name in ONE_Q else 2
                expected_p = 1 if o.name in PARAM_GATES else 0
                if len(o.qubits) != expected_q:
                    raise IRValidationError(
                        f"Op #{idx} ({o.name}) expects {expected_q} qubit(s), got {len(o.qubits)}"
                    )
                if len(o.params) != expected_p:
                    raise IRValidationError(
                        f"Op #{idx} ({o.name}) expects {expected_p} parameter(s), got {len(o.params)}"
                    )
                for p in o.params:
                    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(float(p)):
                        raise IRValidationError(
                            f"Op #{idx} ({o.name}) has unresolved or non-finite parameter {p!r}"
                        )
                if o.duration_s is not None or o.duration_exact_s is not None:
                    raise IRValidationError(f"Gate op #{idx} ({o.name}) must not carry a delay duration")
                if o.cbit is not None:
                    raise IRValidationError(f"Gate op #{idx} ({o.name}) must not carry a cbit destination")

            elif o.name == "delay":
                if len(o.qubits) != 1:
                    raise IRValidationError(f"Delay op #{idx} expects 1 qubit, got {len(o.qubits)}")
                if o.params:
                    raise IRValidationError(f"Delay op #{idx} must not carry gate parameters")
                if o.cbit is not None:
                    raise IRValidationError(f"Delay op #{idx} must not carry a cbit destination")
                if o.duration_s is None and o.duration_exact_s is None:
                    raise IRValidationError(f"Delay op #{idx} missing duration")
                if o.duration_s is not None and (
                    isinstance(o.duration_s, bool)
                    or not isinstance(o.duration_s, (int, float))
                    or not math.isfinite(float(o.duration_s))
                ):
                    raise IRValidationError(f"Delay op #{idx} has invalid duration {o.duration_s!r}")
                exact_d = (
                    o.duration_exact_s
                    if o.duration_exact_s is not None
                    else coerce_exact_seconds(o.duration_s)  # type: ignore[arg-type]
                )
                if exact_d < 0:
                    raise TimingError(
                        f"Delay op #{idx} has negative duration {canonical_duration_str(exact_d)} s"
                    )
                o.duration_exact_s = exact_d
                o.duration_s = float(exact_d)

            elif o.name == "measure":
                if len(o.qubits) != 1:
                    raise IRValidationError(f"Measure op #{idx} expects 1 qubit, got {len(o.qubits)}")
                if o.params:
                    raise IRValidationError(f"Measure op #{idx} must not carry gate parameters")
                if o.duration_s is not None or o.duration_exact_s is not None:
                    raise IRValidationError(f"Measure op #{idx} must not carry a delay duration")
                if (
                    o.cbit is None
                    or isinstance(o.cbit, bool)
                    or not isinstance(o.cbit, int)
                    or o.cbit < 0
                    or o.cbit >= self.n_cbits
                ):
                    raise IRValidationError(
                        f"Measure op #{idx} has invalid cbit destination {o.cbit!r} "
                        f"(valid range: [0, {self.n_cbits}))"
                    )

            elif o.name == "barrier":
                if o.params or o.duration_s is not None or o.duration_exact_s is not None or o.cbit is not None:
                    raise IRValidationError(f"Barrier op #{idx} must not carry params, duration, or cbit")

            else:
                raise IRValidationError(f"Op #{idx} has unsupported operation {o.name!r}")

        return self

    # --- timing conversion boundary (S1-04) ----------------------------------
    def resolve_timing(
        self,
        dt: Optional[Union[Fraction, Decimal, int, float, str]],
        *,
        policy: str = "exact",
    ) -> TimingResolutionReport:
        """Validate logical IR and resolve all delay operations into target `dt` ticks."""
        self.validate()
        if dt is None:
            raise TimingError("Conversion to target ticks requires a non-null clock period 'dt'")
        dt_exact = coerce_exact_seconds(dt)
        if dt_exact <= 0:
            raise TimingError(f"Target clock period 'dt' must be > 0 (got {canonical_duration_str(dt_exact)} s)")
        if policy not in ("exact", "quantize"):
            raise TimingError(f"Unsupported timing_policy {policy!r}; expected 'exact' or 'quantize'")

        records: List[QuantizedDelayRecord] = []
        for idx, o in enumerate(self.ops):
            if o.name == "delay":
                assert o.duration_exact_s is not None
                rec = resolve_delay_ticks(
                    o.duration_exact_s,
                    dt=dt_exact,
                    policy=policy,
                    op_index=idx,
                    qubits=o.qubits,
                )
                records.append(rec)
        return TimingResolutionReport(dt_s=dt_exact, policy=policy, records=records)

    # --- introspection ------------------------------------------------------
    def count(self, name: str) -> int:
        return sum(1 for o in self.ops if o.name == name)

    def two_qubit_count(self) -> int:
        return sum(1 for o in self.ops if o.name in TWO_Q)

    def depth(self) -> int:
        level = [0] * self.n_qubits
        d = 0
        for o in self.ops:
            if o.name == "barrier":
                continue
            t = max(level[q] for q in o.qubits) + 1
            for q in o.qubits:
                level[q] = t
            d = max(d, t)
        return d

    def canonical(self) -> str:
        self.validate()
        rows = []
        for o in self.ops:
            dur_repr = (
                canonical_duration_str(o.duration_exact_s)
                if (o.name == "delay" and o.duration_exact_s is not None)
                else None
            )
            rows.append(
                [o.name, list(o.qubits), [round(float(p), 12) for p in o.params], dur_repr, o.cbit]
            )
        return json.dumps(rows, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    @classmethod
    def from_staggered_schedule(
        cls,
        schedule: Any,
        state_prep: str = "plus",
        measure: bool = True,
    ) -> "Circuit":
        """Construct a flat Circuit IR representation from a StaggeredDDSchedule."""
        qubits = getattr(schedule, "qubits", [])
        n_qubits = max(qubits) + 1 if qubits else 0
        n_cbits = len(qubits) if measure else 0
        c = cls(n_qubits=n_qubits, n_cbits=n_cbits, name=getattr(schedule, "schedule_id", "staggered_dd"))

        if state_prep == "plus":
            for q in qubits:
                c.ops.append(Op("h", (q,)))
        elif state_prep == "zero":
            pass
        if qubits:
            c.ops.append(Op("barrier", tuple(qubits)))

        num_pulses = getattr(schedule, "num_pulses", 4)
        pulse_names = ["x", "y", "x", "y"] if num_pulses == 4 else ["x", "y", "x", "y", "y", "x", "y", "x"]
        pulse_map = getattr(schedule, "pulse_offsets_ns", {})
        total_delay_us = getattr(schedule, "total_delay_us", 0.0)
        t_total_exact = coerce_exact_seconds(total_delay_us) * Fraction(1, 1_000_000)

        for q in qubits:
            offsets_ns = pulse_map.get(q, [])
            last_t_exact = Fraction(0, 1)
            for idx, t_ns in enumerate(offsets_ns):
                t_exact = coerce_exact_seconds(t_ns) * Fraction(1, 1_000_000_000)
                delta_s = max(Fraction(0, 1), t_exact - last_t_exact)
                if delta_s > 0:
                    c.ops.append(Op("delay", (q,), duration_exact_s=delta_s))
                gate = pulse_names[idx % len(pulse_names)]
                c.ops.append(Op(gate, (q,)))
                last_t_exact = t_exact
            rem_s = max(Fraction(0, 1), t_total_exact - last_t_exact)
            if rem_s > 0:
                c.ops.append(Op("delay", (q,), duration_exact_s=rem_s))

        if qubits:
            c.ops.append(Op("barrier", tuple(qubits)))

        if measure:
            if state_prep == "plus":
                for q in qubits:
                    c.ops.append(Op("h", (q,)))
            for idx, q in enumerate(qubits):
                c.ops.append(Op("measure", (q,), cbit=idx))
        return c

    # --- exporters ------------------------------------------------------------
    def to_qiskit(
        self,
        dt: Optional[Union[Fraction, Decimal, int, float, str]] = None,
        *,
        timing_policy: str = "exact",
        require_dt: bool = False,
    ):
        """Return a `qiskit.QuantumCircuit` after enforcing `validate()` and `timing_policy`."""
        self.validate()
        tick_by_op = {}
        if dt is not None or require_dt:
            report = self.resolve_timing(dt, policy=timing_policy)
            tick_by_op = {r.op_index: r.ticks for r in report.records}

        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits, self.n_cbits, name=self.name)
        for idx, o in enumerate(self.ops):
            if o.name == "delay":
                if idx in tick_by_op:
                    qc.delay(tick_by_op[idx], o.qubits[0], unit="dt")
                else:
                    qc.delay(o.duration_s, o.qubits[0], unit="s")
            elif o.name == "barrier":
                qc.barrier(*o.qubits)
            elif o.name == "measure":
                qc.measure(o.qubits[0], o.cbit)
            else:
                getattr(qc, o.name)(*o.params, *o.qubits)
        return qc

    def to_qasm3(
        self,
        dt: Optional[Union[Fraction, Decimal, int, float, str]] = None,
        *,
        timing_policy: str = "exact",
        require_dt: bool = False,
    ) -> str:
        """Emit OpenQASM 3.0 text after enforcing `validate()` and `timing_policy`."""
        self.validate()
        tick_by_op = {}
        if dt is not None or require_dt:
            report = self.resolve_timing(dt, policy=timing_policy)
            tick_by_op = {r.op_index: r.ticks for r in report.records}

        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{self.n_qubits}] q;",
            f"bit[{self.n_cbits}] c;",
        ]
        for idx, o in enumerate(self.ops):
            qs = ", ".join(f"q[{i}]" for i in o.qubits)
            if o.name == "delay":
                if idx in tick_by_op:
                    lines.append(f"delay[{tick_by_op[idx]}dt] {qs};")
                else:
                    assert o.duration_exact_s is not None
                    lines.append(f"delay[{format_exact_ns_qasm(o.duration_exact_s)}] {qs};")
            elif o.name == "barrier":
                lines.append(f"barrier {qs};")
            elif o.name == "measure":
                lines.append(f"c[{o.cbit}] = measure {qs};")
            elif o.params:
                ps = ", ".join(repr(float(p)) for p in o.params)
                lines.append(f"{o.name}({ps}) {qs};")
            else:
                lines.append(f"{o.name} {qs};")
        return "\n".join(lines) + "\n"
