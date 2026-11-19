"""Lower an Organism AST to the flat circuit IR.

Total and deterministic: each genome instance instantiates its gene template with
constant-folded arguments (preserving exact rational durations), renames ports to
physical qubits, appends the resulting ops, and validates the resulting logical IR.
"""
from __future__ import annotations

from typing import Dict

from . import ast as A
from .qc_ir import Circuit, Op
from .sema import SemaError, TypedValue, check, eval_typed_expr


def lower(org: A.Organism, *, gene_barriers: bool = True) -> Circuit:
    diag = check(org)
    if not diag.ok():
        raise SemaError("; ".join(diag.errors))
    genes: Dict[str, A.Gene] = {g.name: g for g in org.genes}
    n_q = 1 + max(q for inst in org.genome for q in inst.qubits)
    circ = Circuit(n_qubits=n_q, n_cbits=len(org.cbits), name=org.name)
    cbit_index = 0
    for inst in org.genome:
        g = genes[inst.gene]
        env: Dict[str, TypedValue] = {
            p.name: eval_typed_expr(a, {}) for p, a in zip(g.params, inst.args)
        }
        port = dict(zip(g.ports, inst.qubits))
        for s in g.body:
            if isinstance(s, A.Gate):
                circ.ops.append(
                    Op(
                        s.op,
                        tuple(port[q] for q in s.qubits),
                        tuple(eval_typed_expr(a, env).value for a in s.args),
                    )
                )
            elif isinstance(s, A.Wait):
                wt = eval_typed_expr(s.duration, env)
                for q in s.qubits:
                    circ.ops.append(
                        Op("delay", (port[q],), duration_s=wt.value, duration_exact_s=wt.exact)
                    )
            elif isinstance(s, A.Echo):
                for q in s.qubits:
                    circ.ops.append(Op(s.pauli.lower(), (port[q],)))
            elif isinstance(s, A.Measure):
                circ.ops.append(Op("measure", (port[s.qubit],), cbit=cbit_index))
                cbit_index += 1
            elif isinstance(s, A.Barrier):
                circ.ops.append(Op("barrier", tuple(port[q] for q in s.qubits)))
        if gene_barriers:
            circ.ops.append(Op("barrier", tuple(inst.qubits)))
    return circ.validate()


def strip_barriers(circ: Circuit) -> Circuit:
    return Circuit(
        circ.n_qubits,
        circ.n_cbits,
        [o for o in circ.ops if o.name != "barrier"],
        circ.name,
    )
