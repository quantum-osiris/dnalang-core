"""Semantic checks for dnalang.

Enforces (`S1-02` / `S1-03` / `S1-04`):
  * Port declaration, distinctness, and arity rules on gates, waits, echoes, and barriers
  * Exact rational constant folding with dimensional analysis (`duration` vs `scalar` / `angle` / `int`)
  * Non-negative durations with source-located (`line:col`) diagnostics on both gene-body
    `wait(...)` expressions and genome instance arguments
  * Measurement (`cleave q -> c`) port validity, port-alias prohibition, and per-gene
    unique classical destination checks
  * Genome physical qubit arity, distinctness (no multi-port aliasing), and non-negativity
  * Dynamical-decoupling pulse parity warning (`dd-parity`)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional

from . import ast as A
from .timing import CompilerContractError, coerce_exact_seconds


GATE_ARITY = {  # op -> (n_params, n_qubits)
    "h": (0, 1), "x": (0, 1), "y": (0, 1), "z": (0, 1), "s": (0, 1), "t": (0, 1), "sx": (0, 1),
    "rx": (1, 1), "ry": (1, 1), "rz": (1, 1), "cx": (0, 2), "cz": (0, 2), "swap": (0, 2),
}

_FUNCS = {
    "pi": lambda: math.pi,
    "phi": lambda: (1 + 5 ** 0.5) / 2,
    "sqrt": math.sqrt,
}


class SemaError(CompilerContractError):
    pass


@dataclass
class Diagnostics:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class TypedValue:
    """Dimension-tagged value produced during semantic checking and constant folding."""

    value: float
    exact: Optional[Fraction]
    dim: str  # "scalar" | "duration" | "zero"

    @classmethod
    def from_scalar(cls, val: float, exact: Optional[Fraction] = None) -> "TypedValue":
        if exact is None and math.isfinite(val) and val == 0.0:
            exact = Fraction(0, 1)
        dim = "zero" if (exact is not None and exact == 0) else "scalar"
        return cls(float(val), exact, dim)

    @classmethod
    def from_duration(cls, exact_s: Fraction) -> "TypedValue":
        return cls(float(exact_s), exact_s, "duration")


def eval_typed_expr(e: A.Expr, env: Dict[str, TypedValue]) -> TypedValue:
    """Evaluate expression `e` preserving exact `Fraction` rationals and physical dimensions."""
    if isinstance(e, A.Const):
        exact = e.exact if e.exact is not None else (
            coerce_exact_seconds(e.value) if math.isfinite(e.value) else None
        )
        dim = getattr(e, "dim", "scalar")
        if dim == "scalar" and exact is not None and exact == 0:
            dim = "zero"
        return TypedValue(float(e.value), exact, dim)

    if isinstance(e, A.ParamRef):
        if e.name not in env:
            raise SemaError(f"{e.pos}: unknown parameter '{e.name}'")
        return env[e.name]

    if isinstance(e, A.BinOp):
        a = eval_typed_expr(e.left, env)
        b = eval_typed_expr(e.right, env)

        if e.op in ("+", "-"):
            if a.dim == "zero":
                res_dim = b.dim
            elif b.dim == "zero":
                res_dim = a.dim
            elif a.dim == b.dim:
                res_dim = a.dim
            else:
                raise SemaError(
                    f"{e.pos}: dimension mismatch in '{e.op}': cannot combine {a.dim} and {b.dim}"
                )
            fv = (a.value + b.value) if e.op == "+" else (a.value - b.value)
            if a.exact is not None and b.exact is not None:
                ex = (a.exact + b.exact) if e.op == "+" else (a.exact - b.exact)
                return TypedValue(float(ex) if res_dim == "duration" else fv, ex, res_dim)
            return TypedValue(fv, None, res_dim)

        if e.op == "*":
            if a.dim == "duration" and b.dim == "duration":
                raise SemaError(f"{e.pos}: dimension error: cannot multiply two durations")
            if a.dim == "duration" or b.dim == "duration":
                res_dim = "duration"
            elif a.dim == "zero" or b.dim == "zero":
                res_dim = "zero"
            else:
                res_dim = "scalar"
            fv = a.value * b.value
            if a.exact is not None and b.exact is not None:
                ex = a.exact * b.exact
                return TypedValue(float(ex) if res_dim == "duration" else fv, ex, res_dim)
            ex_fb = coerce_exact_seconds(fv) if (res_dim == "duration" and math.isfinite(fv)) else None
            return TypedValue(fv, ex_fb, res_dim)

        if e.op == "/":
            if (b.exact is not None and b.exact == 0) or b.value == 0:
                raise SemaError(f"{e.pos}: division by zero")
            if b.dim == "duration":
                if a.dim in ("duration", "zero"):
                    res_dim = "scalar"
                else:
                    raise SemaError(
                        f"{e.pos}: dimension error: cannot divide {a.dim} by duration"
                    )
            elif a.dim == "duration":
                res_dim = "duration"
            elif a.dim == "zero":
                res_dim = "zero"
            else:
                res_dim = "scalar"

            fv = a.value / b.value
            if a.exact is not None and b.exact is not None:
                ex = a.exact / b.exact
                return TypedValue(float(ex) if res_dim == "duration" else fv, ex, res_dim)
            ex_fb = coerce_exact_seconds(fv) if (res_dim == "duration" and math.isfinite(fv)) else None
            return TypedValue(fv, ex_fb, res_dim)

    if isinstance(e, A.Call):
        if e.name not in _FUNCS:
            raise SemaError(f"{e.pos}: unknown function '{e.name}'")
        arg_vals = [eval_typed_expr(a, env) for a in e.args]
        for av in arg_vals:
            if av.dim == "duration":
                raise SemaError(
                    f"{e.pos}: function '{e.name}' requires dimensionless argument, got duration"
                )
        try:
            fv = float(_FUNCS[e.name](*[av.value for av in arg_vals]))
        except TypeError as exc:
            raise SemaError(f"{e.pos}: wrong argument count for '{e.name}': {exc}") from exc
        except ValueError as exc:
            raise SemaError(f"{e.pos}: domain error in '{e.name}': {exc}") from exc
        return TypedValue.from_scalar(fv)

    raise SemaError(f"unhandled expression {e!r}")


def eval_expr(e: A.Expr, env: Dict[str, float]) -> float:
    """Evaluate expression `e` to float (compatible entry point wrapping `eval_typed_expr`)."""
    typed_env = {
        k: (v if isinstance(v, TypedValue) else TypedValue.from_scalar(float(v)))
        for k, v in env.items()
    }
    return eval_typed_expr(e, typed_env).value


def check_kv(org: A.Organism, d: Diagnostics) -> None:
    """Rule and regulator genes: unique ids, ternary conditions of one width, valid closed-DSL
    actions, valid triggers, known dependencies, no dependency cycles."""
    from .rules_ir import LowerError, lower_regulation, lower_rules
    ids = set()
    for i, g in enumerate(org.kv_genes):
        gid = str(g.get("id", f"G{i}"))
        if gid in ids:
            d.errors.append(f"{g.pos}: duplicate gene id '{gid}'")
        ids.add(gid)
        if g.kind == "data":
            d.warnings.append(f"{g.pos}: gene '{g.name}' has neither condition nor trigger (data only)")
    if org.rules:
        try:
            lower_rules(org)
        except LowerError as e:
            d.errors.append(str(e))
    if org.regulators:
        try:
            lower_regulation(org)
        except LowerError as e:
            d.errors.append(str(e))
    declared = org.meta.get("metrics")
    if isinstance(declared, list):
        from .regulation import Trigger
        for g in org.regulators:
            try:
                t = Trigger.parse(str(g.get("trigger")))
            except ValueError:
                continue
            if t.kind == "when" and t.ref not in declared:
                d.warnings.append(f"{g.pos}: gene '{g.name}' triggers on undeclared metric '{t.ref}'")


def _expr_pos(e: A.Expr, fallback: A.Pos) -> A.Pos:
    return getattr(e, "pos", fallback)


def check(org: A.Organism) -> Diagnostics:
    d = Diagnostics()
    check_kv(org, d)
    if not org.genes and not org.genome:
        return d  # pure rule/regulatory organism has no circuit to check

    genes: Dict[str, A.Gene] = {}
    for g in org.genes:
        if g.name in genes:
            d.errors.append(f"{g.pos}: duplicate gene '{g.name}'")
        genes[g.name] = g
        ports = set(g.ports)
        if len(ports) != len(g.ports):
            d.errors.append(f"{g.pos}: gene '{g.name}' declares a port twice")
        pnames = [p.name for p in g.params]
        if len(set(pnames)) != len(pnames):
            d.errors.append(f"{g.pos}: gene '{g.name}' declares a parameter twice")

        # Prototype environment to validate parameter references and dimensions inside gene body
        proto_env: Dict[str, TypedValue] = {}
        for p in g.params:
            if p.type == "duration":
                proto_env[p.name] = TypedValue.from_duration(Fraction(1, 1_000_000))
            else:
                proto_env[p.name] = TypedValue(1.0, Fraction(1, 1), "scalar")

        echo_x: Dict[str, int] = {p: 0 for p in g.ports}
        echo_y: Dict[str, int] = {p: 0 for p in g.ports}
        seen_gene_cbits: set[str] = set()

        for s in g.body:
            qs = [s.qubit] if isinstance(s, A.Measure) else s.qubits
            for q in qs:
                if q not in ports:
                    d.errors.append(f"{s.pos}: port '{q}' not declared in gene '{g.name}'")

            if isinstance(s, A.Gate):
                np_, nq = GATE_ARITY[s.op]
                if len(s.args) != np_:
                    d.errors.append(f"{s.pos}: {s.op} takes {np_} parameter(s), got {len(s.args)}")
                if len(s.qubits) != nq:
                    d.errors.append(f"{s.pos}: {s.op} acts on {nq} qubit(s), got {len(s.qubits)}")
                if nq >= 2 and len(set(s.qubits)) != len(s.qubits):
                    d.errors.append(f"{s.pos}: {s.op} needs {nq} distinct qubits")
                for arg_expr in s.args:
                    try:
                        tv = eval_typed_expr(arg_expr, proto_env)
                        if tv.dim == "duration":
                            d.errors.append(
                                f"{_expr_pos(arg_expr, s.pos)}: gate '{s.op}' expects angle parameter, got duration"
                            )
                        elif not math.isfinite(tv.value):
                            d.errors.append(
                                f"{_expr_pos(arg_expr, s.pos)}: gate '{s.op}' parameter is non-finite"
                            )
                    except SemaError as exc:
                        d.errors.append(str(exc))

            elif isinstance(s, A.Wait):
                if len(set(s.qubits)) != len(s.qubits):
                    d.errors.append(f"{s.pos}: wait acts on duplicate qubit ports")
                try:
                    tv = eval_typed_expr(s.duration, proto_env)
                    if tv.dim not in ("duration", "zero"):
                        d.errors.append(
                            f"{_expr_pos(s.duration, s.pos)}: wait requires a duration expression (ns/us/ms/s), got {tv.dim}"
                        )
                    elif (tv.exact is not None and tv.exact < 0) or tv.value < 0:
                        # Catch literal negative waits (e.g. wait(-5ns)) immediately with line:col
                        if not any(isinstance(n, A.ParamRef) for n in _walk_expr(s.duration)):
                            d.errors.append(
                                f"{s.pos}: wait duration must be non-negative (got {tv.value} s)"
                            )
                except SemaError as exc:
                    d.errors.append(str(exc))

            elif isinstance(s, A.Echo):
                if len(set(s.qubits)) != len(s.qubits):
                    d.errors.append(f"{s.pos}: echo acts on duplicate qubit ports")
                for q in s.qubits:
                    if q in echo_x:
                        (echo_x if s.pauli == "X" else echo_y)[q] += 1

            elif isinstance(s, A.Barrier):
                if len(set(s.qubits)) != len(s.qubits):
                    d.errors.append(f"{s.pos}: barrier acts on duplicate qubit ports")

            elif isinstance(s, A.Measure):
                if not s.cbit or not s.cbit.isidentifier():
                    d.errors.append(f"{s.pos}: invalid measurement destination '{s.cbit}'")
                elif s.cbit in ports:
                    d.errors.append(
                        f"{s.pos}: measurement destination '{s.cbit}' collides with declared qubit port"
                    )
                elif s.cbit in seen_gene_cbits:
                    d.errors.append(
                        f"{s.pos}: duplicate measurement destination '{s.cbit}' in gene '{g.name}'"
                    )
                else:
                    seen_gene_cbits.add(s.cbit)

        for p in g.ports:
            if echo_x[p] % 2 or echo_y[p] % 2:
                d.warnings.append(
                    f"{g.pos}: dd-parity: gene '{g.name}' port '{p}' has "
                    f"{echo_x[p]} X and {echo_y[p]} Y echoes; net action is a Pauli, not identity"
                )

    # Genome instantiation checks
    cbits: List[str] = []
    for inst in org.genome:
        g = genes.get(inst.gene)
        if g is None:
            d.errors.append(f"{inst.pos}: unknown gene '{inst.gene}'")
            continue
        if len(inst.args) != len(g.params):
            d.errors.append(f"{inst.pos}: gene '{g.name}' takes {len(g.params)} arg(s), got {len(inst.args)}")
        if len(inst.qubits) != len(g.ports):
            d.errors.append(f"{inst.pos}: gene '{g.name}' has {len(g.ports)} port(s), got {len(inst.qubits)} qubits")
        if len(set(inst.qubits)) != len(inst.qubits):
            d.errors.append(f"{inst.pos}: instance binds the same qubit to two ports")
        for q in inst.qubits:
            if q < 0:
                d.errors.append(f"{inst.pos}: negative qubit index {q}")

        inst_env: Dict[str, TypedValue] = {}
        for p, arg_expr in zip(g.params, inst.args):
            try:
                tv = eval_typed_expr(arg_expr, {})
            except SemaError as exc:
                d.errors.append(str(exc))
                continue
            apos = _expr_pos(arg_expr, inst.pos)
            if p.type == "duration":
                if tv.dim not in ("duration", "zero"):
                    d.errors.append(
                        f"{apos}: parameter '{p.name}' requires a duration unit (ns/us/ms/s), got {tv.dim}"
                    )
                elif (tv.exact is not None and tv.exact < 0) or tv.value < 0:
                    d.errors.append(f"{apos}: duration '{p.name}' must be non-negative")
                else:
                    ex = tv.exact if tv.exact is not None else coerce_exact_seconds(tv.value)
                    inst_env[p.name] = TypedValue.from_duration(ex)
            elif p.type == "int":
                if tv.dim == "duration":
                    d.errors.append(f"{apos}: parameter '{p.name}' must be an integer, got duration")
                elif tv.value != int(tv.value):
                    d.errors.append(f"{apos}: parameter '{p.name}' must be an integer")
                else:
                    inst_env[p.name] = TypedValue.from_scalar(float(int(tv.value)), Fraction(int(tv.value), 1))
            else:  # angle
                if tv.dim == "duration":
                    d.errors.append(f"{apos}: parameter '{p.name}' expects an angle, got duration")
                elif not math.isfinite(tv.value):
                    d.errors.append(f"{apos}: parameter '{p.name}' is non-finite")
                else:
                    inst_env[p.name] = tv

        # Validate instantiated Wait durations (catches e.g. wait(T - 20us) when T=10us)
        if len(inst_env) == len(g.params):
            for s in g.body:
                if isinstance(s, A.Wait):
                    try:
                        wt = eval_typed_expr(s.duration, inst_env)
                        if (wt.exact is not None and wt.exact < 0) or wt.value < 0:
                            d.errors.append(
                                f"{s.pos}: wait duration must be non-negative (instantiated at {inst.pos} to {wt.value} s)"
                            )
                    except SemaError as exc:
                        d.errors.append(str(exc))

        for s in g.body:
            if isinstance(s, A.Measure):
                name = f"{inst.gene}{len(cbits)}_{s.cbit}"
                cbits.append(name)

    org.cbits = cbits
    if not org.genome:
        d.errors.append(f"{org.pos}: genome is empty")
    return d


def _walk_expr(e: A.Expr):
    yield e
    if isinstance(e, A.BinOp):
        yield from _walk_expr(e.left)
        yield from _walk_expr(e.right)
    elif isinstance(e, A.Call):
        for a in e.args:
            yield from _walk_expr(a)
