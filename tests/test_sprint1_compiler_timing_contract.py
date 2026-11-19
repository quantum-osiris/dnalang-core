"""Sprint 1 Compiler Correctness Baseline & Timing Contract Regression Suite (`S1-05`).

Covers all 12 required regression cases end-to-end with zero skips:
  1. Equivalent physical durations (`1000ns`, `1us`, `0.001ms`, `0.000001s`) normalize identically.
  2. Constant folding preserves duration dimensions and rejects dimension mismatches.
  3. Conversion to target ticks (`dt`) without required clock information fails.
  4. Unrepresentable duration fails under `exact` timing policy.
  5. Explicit `quantize` policy reports requested duration, effective duration, ticks, and adjustment.
  6. Negative durations fail with a source-located (`line:col`) diagnostic.
  7. Unresolved parameters fail before export.
  8. Invalid qubit references and prohibited aliases fail.
  9. Invalid measurement destinations fail.
 10. Identical inputs & compilation settings produce identical canonical output across processes.
 11. Bell-circuit fixture preserves logical and measurement mapping through export.
 12. Timing-sensitive sequence fixture preserves declared order and duration constraints through
     included compilation stages (without claiming downstream vendor hardware transpilation).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from dnalang import (
    Circuit,
    IRValidationError,
    Op,
    TimingError,
    check,
    lower,
    parse,
    strip_barriers,
)
from dnalang.sema import SemaError

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


# ---------------------------------------------------------------------------
# Case 1: Equivalent physical durations normalize identically
# ---------------------------------------------------------------------------
def test_case01_equivalent_physical_durations_normalize_identically():
    spellings = ["1000ns", "1us", "0.001ms", "0.000001s", "1e-6s"]
    circuits = []
    for lit in spellings:
        src = f"""
        organism DurNorm {{
          gene g() on q {{
            helix q;
            wait({lit}) q;
            helix q;
            cleave q -> c;
          }}
          genome {{
            g() @ [0];
          }}
        }}
        """
        circ = strip_barriers(lower(parse(src)))
        circuits.append(circ)

    ref = circuits[0]
    delay_ops = [o for o in ref.ops if o.name == "delay"]
    assert len(delay_ops) == 1
    assert delay_ops[0].duration_exact_s == Fraction(1, 1_000_000)

    for lit, circ in zip(spellings[1:], circuits[1:]):
        d_op = [o for o in circ.ops if o.name == "delay"][0]
        assert d_op.duration_exact_s == Fraction(1, 1_000_000), f"Mismatch for {lit}"
        assert circ.canonical() == ref.canonical(), f"Canonical mismatch for {lit}"
        assert circ.sha256() == ref.sha256(), f"SHA-256 mismatch for {lit}"
        assert circ.to_qasm3() == ref.to_qasm3(), f"QASM3 mismatch for {lit}"
    assert "delay[1000ns] q[0];" in ref.to_qasm3()


# ---------------------------------------------------------------------------
# Case 2: Constant folding preserves duration dimensions & rejects mismatches
# ---------------------------------------------------------------------------
def test_case02_constant_folding_preserves_and_enforces_duration_dimensions():
    valid_src = """
    organism FoldValid {
      gene g(T: duration) on q {
        wait(500ns + 0.5us) q;
        wait(2 * 500ns) q;
        wait(3us / 3) q;
        wait((0.1 + 0.2) * 10us) q;
        rz((1us / 500ns) * pi() / 4) q;
        wait(T / 2 + 500ns) q;
        cleave q -> c;
      }
      genome {
        g(1us) @ [0];
      }
    }
    """
    circ = strip_barriers(lower(parse(valid_src)))
    delays = [o.duration_exact_s for o in circ.ops if o.name == "delay"]
    assert delays == [
        Fraction(1, 1_000_000),
        Fraction(1, 1_000_000),
        Fraction(1, 1_000_000),
        Fraction(3, 1_000_000),
        Fraction(1, 1_000_000),
    ]

    invalid_programs = [
        # duration + scalar
        "organism E1 { gene g() on q { wait(10ns + 1.5) q; } genome { g() @ [0]; } }",
        # duration * duration
        "organism E2 { gene g() on q { wait(10ns * 20ns) q; } genome { g() @ [0]; } }",
        # scalar / duration
        "organism E3 { gene g() on q { rz(1.5 / 10ns) q; } genome { g() @ [0]; } }",
        # unitless non-zero scalar passed to wait
        "organism E4 { gene g() on q { wait(1.5) q; } genome { g() @ [0]; } }",
        # duration passed to angle gate
        "organism E5 { gene g() on q { rz(10ns) q; } genome { g() @ [0]; } }",
        # duration passed to math function
        "organism E6 { gene g() on q { rz(sqrt(10ns)) q; } genome { g() @ [0]; } }",
    ]
    for bad in invalid_programs:
        diag = check(parse(bad))
        assert not diag.ok(), f"Expected dimensional error for: {bad}"
        with pytest.raises(SemaError):
            lower(parse(bad))


# ---------------------------------------------------------------------------
# Case 3: Conversion to target ticks without required clock information fails
# ---------------------------------------------------------------------------
def test_case03_tick_conversion_without_clock_info_fails():
    src = """
    organism ClockReq {
      gene g() on q { wait(100ns) q; cleave q -> c; }
      genome { g() @ [0]; }
    }
    """
    circ = lower(parse(src))

    with pytest.raises(TimingError, match="non-null clock period 'dt'"):
        circ.resolve_timing(dt=None)

    with pytest.raises(TimingError, match="non-null clock period 'dt'"):
        circ.to_qasm3(dt=None, require_dt=True)

    with pytest.raises(TimingError, match="non-null clock period 'dt'"):
        circ.to_qiskit(dt=None, require_dt=True)

    for non_pos_dt in (0, -1e-9, "0ns"):
        with pytest.raises(TimingError, match="must be > 0"):
            circ.resolve_timing(dt=non_pos_dt)


# ---------------------------------------------------------------------------
# Case 4: Unrepresentable duration fails under `exact` timing policy
# ---------------------------------------------------------------------------
def test_case04_unrepresentable_duration_fails_under_exact_policy():
    src = """
    organism ExactFail {
      gene g() on q { wait(15ns) q; cleave q -> c; }
      genome { g() @ [0]; }
    }
    """
    circ = strip_barriers(lower(parse(src)))

    with pytest.raises(TimingError, match="not an exact integer multiple of dt"):
        circ.resolve_timing(dt="4ns", policy="exact")

    with pytest.raises(TimingError, match="not an exact integer multiple of dt"):
        circ.to_qasm3(dt="4ns", timing_policy="exact")

    with pytest.raises(TimingError, match="not an exact integer multiple of dt"):
        circ.to_qiskit(dt="4ns", timing_policy="exact")


# ---------------------------------------------------------------------------
# Case 5: Explicit `quantize` policy reports requested, effective, ticks & delta
# ---------------------------------------------------------------------------
def test_case05_quantize_policy_reports_requested_effective_ticks_and_adjustment():
    src = """
    organism QuantizeAudit {
      gene g() on q {
        helix q;
        wait(15ns) q;
        wait(8ns) q;
        helix q;
        cleave q -> c;
      }
      genome { g() @ [0]; }
    }
    """
    circ = strip_barriers(lower(parse(src)))
    report = circ.resolve_timing(dt="4ns", policy="quantize")

    assert report.policy == "quantize"
    assert report.dt_s == Fraction(4, 1_000_000_000)
    assert len(report.records) == 2
    assert report.adjusted_count == 1
    assert report.max_abs_adjustment_s == Fraction(1, 1_000_000_000)

    r0, r1 = report.records
    assert r0.requested_duration_s == Fraction(15, 1_000_000_000)
    assert r0.ticks == 4
    assert r0.effective_duration_s == Fraction(16, 1_000_000_000)
    assert r0.adjustment_s == Fraction(1, 1_000_000_000)

    assert r1.requested_duration_s == Fraction(8, 1_000_000_000)
    assert r1.ticks == 2
    assert r1.effective_duration_s == Fraction(8, 1_000_000_000)
    assert r1.adjustment_s == Fraction(0, 1)

    qasm = circ.to_qasm3(dt="4ns", timing_policy="quantize")
    assert "delay[4dt] q[0];" in qasm
    assert "delay[2dt] q[0];" in qasm

    qc = circ.to_qiskit(dt="4ns", timing_policy="quantize")
    q_delays = [(i.operation.duration, i.operation.unit) for i in qc.data if i.operation.name == "delay"]
    assert q_delays == [(4, "dt"), (2, "dt")]


# ---------------------------------------------------------------------------
# Case 6: Negative durations fail with a source-located (`line:col`) diagnostic
# ---------------------------------------------------------------------------
def test_case06_negative_durations_fail_with_source_location():
    src_literal_neg = (
        "organism NegBody {\n"
        "  gene g() on q {\n"
        "    helix q;\n"
        "    wait(-5ns) q;\n"
        "  }\n"
        "  genome { g() @ [0]; }\n"
        "}\n"
    )
    diag1 = check(parse(src_literal_neg))
    assert not diag1.ok()
    assert any(re.search(r"^4:5:.*non-negative", err) for err in diag1.errors), diag1.errors

    src_arg_neg = (
        "organism NegArg {\n"
        "  gene g(T: duration) on q { wait(T) q; }\n"
        "  genome {\n"
        "    g(-10us) @ [0];\n"
        "  }\n"
        "}\n"
    )
    diag2 = check(parse(src_arg_neg))
    assert not diag2.ok()
    assert any(re.search(r"^4:\d+:.*non-negative", err) for err in diag2.errors), diag2.errors

    src_inst_expr_neg = (
        "organism NegInstExpr {\n"
        "  gene g(T: duration) on q {\n"
        "    wait(T - 20us) q;\n"
        "  }\n"
        "  genome { g(10us) @ [0]; }\n"
        "}\n"
    )
    diag3 = check(parse(src_inst_expr_neg))
    assert not diag3.ok()
    assert any(re.search(r"^3:5:.*non-negative", err) for err in diag3.errors), diag3.errors

    bad_ir = Circuit(1, 0, [Op("delay", (0,), duration_s=-5e-9)])
    with pytest.raises(TimingError, match="negative duration"):
        bad_ir.validate()


# ---------------------------------------------------------------------------
# Case 7: Unresolved parameters fail before export
# ---------------------------------------------------------------------------
def test_case07_unresolved_parameters_fail_before_export():
    # Source-level undeclared parameter in wait and gate
    for bad_src in (
        "organism U1 { gene g() on q { wait(tau) q; } genome { g() @ [0]; } }",
        "organism U2 { gene g() on q { rz(theta) q; } genome { g() @ [0]; } }",
    ):
        diag = check(parse(bad_src))
        assert not diag.ok()
        assert any("unknown parameter" in e for e in diag.errors)
        with pytest.raises(SemaError, match="unknown parameter"):
            lower(parse(bad_src))

    # Logical-IR level unresolved / non-finite / symbolic parameter before export
    malformed_irs = [
        Circuit(1, 0, [Op("rz", (0,), params=())]),
        Circuit(1, 0, [Op("rz", (0,), params=(float("nan"),))]),
        Circuit(1, 0, [Op("rz", (0,), params=("theta",))]),  # type: ignore[arg-type]
    ]
    for bad_circ in malformed_irs:
        for exporter in (bad_circ.validate, bad_circ.canonical, bad_circ.to_qasm3, bad_circ.to_qiskit):
            with pytest.raises(IRValidationError):
                exporter()


# ---------------------------------------------------------------------------
# Case 8: Invalid qubit references and prohibited aliases fail
# ---------------------------------------------------------------------------
def test_case08_invalid_qubit_references_and_prohibited_aliases_fail():
    source_failures = [
        ("organism Q1 { gene g() on q { helix r; } genome { g() @ [0]; } }", "port 'r'"),
        ("organism Q2 { gene g() on q, q { helix q; } genome { g() @ [0, 1]; } }", "declares a port twice"),
        ("organism Q3 { gene g() on a { bond a, a; } genome { g() @ [0]; } }", "distinct qubits"),
        ("organism Q4 { gene g() on a { wait(10ns) a, a; } genome { g() @ [0]; } }", "duplicate qubit ports"),
        ("organism Q5 { gene g() on a, b { bond a, b; } genome { g() @ [1, 1]; } }", "same qubit"),
        ("organism Q6 { gene g() on a { helix a; } genome { g() @ [-1]; } }", "negative qubit index"),
    ]
    for src, expected_msg in source_failures:
        diag = check(parse(src))
        assert not diag.ok(), f"Expected failure for {src}"
        assert any(expected_msg in e for e in diag.errors), (expected_msg, diag.errors)

    ir_failures = [
        Circuit(2, 0, [Op("h", (-1,))]),
        Circuit(2, 0, [Op("h", (2,))]),
        Circuit(2, 0, [Op("cx", (0, 0))]),
    ]
    for bad_circ in ir_failures:
        with pytest.raises(IRValidationError):
            bad_circ.to_qasm3()
        with pytest.raises(IRValidationError):
            bad_circ.to_qiskit()


# ---------------------------------------------------------------------------
# Case 9: Invalid measurement destinations fail
# ---------------------------------------------------------------------------
def test_case09_invalid_measurement_destinations_fail():
    # Classical destination collides with declared quantum port
    src_port_collision = "organism M1 { gene g() on q { helix q; cleave q -> q; } genome { g() @ [0]; } }"
    d1 = check(parse(src_port_collision))
    assert not d1.ok() and any("collides with declared qubit port" in e for e in d1.errors)

    # Duplicate measurement destination within the same gene
    src_dup_cbit = (
        "organism M2 { gene g() on a, b { cleave a -> c; cleave b -> c; } genome { g() @ [0, 1]; } }"
    )
    d2 = check(parse(src_dup_cbit))
    assert not d2.ok() and any("duplicate measurement destination" in e for e in d2.errors)

    # Out-of-bounds or missing cbit at logical-IR boundary
    for bad_cbit in (None, -1, 1, True):
        bad_circ = Circuit(1, 1, [Op("measure", (0,), cbit=bad_cbit)])  # type: ignore[arg-type]
        with pytest.raises(IRValidationError, match="invalid cbit destination"):
            bad_circ.to_qasm3()
        with pytest.raises(IRValidationError, match="invalid cbit destination"):
            bad_circ.to_qiskit()


# ---------------------------------------------------------------------------
# Case 10: Cross-process canonical determinism under distinct PYTHONHASHSEED
# ---------------------------------------------------------------------------
def test_case10_cross_process_determinism_across_hash_seeds():
    worker = "\n".join(
        [
            "import json",
            "from pathlib import Path",
            "from dnalang import lower, parse",
            f"ex = Path({str(EXAMPLES)!r})",
            "out = {}",
            "for name in ('bell.dna', 'ghz.dna', 'dd_staggered_xy4.dna'):",
            "    c = lower(parse((ex / name).read_text()))",
            "    out[name] = {'canonical': c.canonical(), 'sha256': c.sha256(), 'qasm3': c.to_qasm3(dt='50ns')}",
            "print(json.dumps(out, sort_keys=True))",
        ]
    )
    outputs = []
    for seed in ("1", "424242"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = str(REPO_ROOT)
        proc = subprocess.run(
            [sys.executable, "-c", worker],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(proc.stdout))

    assert outputs[0] == outputs[1]
    assert len(outputs[0]["dd_staggered_xy4.dna"]["sha256"]) == 64


# ---------------------------------------------------------------------------
# Case 11: Bell-circuit fixture preserves logical & measurement mapping
# ---------------------------------------------------------------------------
def test_case11_bell_fixture_preserves_logical_and_measurement_mapping():
    from qiskit import qasm3

    bell_src = (EXAMPLES / "bell.dna").read_text()
    circ = strip_barriers(lower(parse(bell_src)))
    assert [(o.name, o.qubits, o.cbit) for o in circ.ops] == [
        ("h", (0,), None),
        ("cx", (0, 1), None),
        ("measure", (0,), 0),
        ("measure", (1,), 1),
    ]

    qasm_text = circ.to_qasm3()
    qasm_qc = qasm3.loads(qasm_text)
    direct_qc = circ.to_qiskit()

    for label, qc in (("qasm3", qasm_qc), ("to_qiskit", direct_qc)):
        ops = [
            (
                inst.operation.name,
                tuple(qc.find_bit(q).index for q in inst.qubits),
                tuple(qc.find_bit(c).index for c in inst.clbits),
            )
            for inst in qc.data
        ]
        assert ops == [
            ("h", (0,), ()),
            ("cx", (0, 1), ()),
            ("measure", (0,), (0,)),
            ("measure", (1,), (1,)),
        ], f"Mapping mismatch in {label}: {ops}"


# ---------------------------------------------------------------------------
# Case 12: Timing-sensitive sequence preserves order & duration constraints
# ---------------------------------------------------------------------------
def test_case12_timing_sensitive_sequence_preserves_order_and_durations():
    from qiskit import qasm3

    # Asymmetric sequence so any reordering or duration swap is immediately caught
    asym_src = """
    organism AsymEcho {
      gene seq(t1: duration, t2: duration, t3: duration) on q0, q1 {
        helix q0;
        bond q0, q1;
        wait(t1) q0;
        echo(X) q0;
        wait(t2) q0;
        echo(Y) q0;
        wait(t3) q1;
        barrier q0, q1;
        cleave q0 -> c0;
        cleave q1 -> c1;
      }
      genome {
        seq(100ns, 0.25us, 0.0004ms) @ [0, 1];
      }
    }
    """
    circ = lower(parse(asym_src), gene_barriers=False)
    report = circ.resolve_timing(dt="50ns", policy="exact")
    assert [r.ticks for r in report.records] == [2, 5, 8]
    assert report.adjusted_count == 0

    expected_ir_trace = [
        ("h", (0,), None),
        ("cx", (0, 1), None),
        ("delay", (0,), Fraction(100, 1_000_000_000)),
        ("x", (0,), None),
        ("delay", (0,), Fraction(250, 1_000_000_000)),
        ("y", (0,), None),
        ("delay", (1,), Fraction(400, 1_000_000_000)),
        ("barrier", (0, 1), None),
        ("measure", (0,), None),
        ("measure", (1,), None),
    ]
    assert [(o.name, o.qubits, o.duration_exact_s) for o in circ.ops] == expected_ir_trace

    # Verify untranspiled Qiskit export preserves instruction order and exact dt ticks
    qc_dt = circ.to_qiskit(dt="50ns", timing_policy="exact")
    qiskit_trace = [
        (
            i.operation.name,
            tuple(qc_dt.find_bit(q).index for q in i.qubits),
            getattr(i.operation, "duration", None),
            getattr(i.operation, "unit", None),
        )
        for i in qc_dt.data
    ]
    assert qiskit_trace == [
        ("h", (0,), None, None),
        ("cx", (0, 1), None, None),
        ("delay", (0,), 2, "dt"),
        ("x", (0,), None, None),
        ("delay", (0,), 5, "dt"),
        ("y", (0,), None, None),
        ("delay", (1,), 8, "dt"),
        ("barrier", (0, 1), None, None),
        ("measure", (0,), None, None),
        ("measure", (1,), None, None),
    ]

    # Verify physical-unit OpenQASM 3 export and StaggeredXY4 fixture
    qasm_phys = circ.to_qasm3()
    parsed_phys = qasm3.loads(qasm_phys)
    phys_delays = [
        (parsed_phys.find_bit(i.qubits[0]).index, i.operation.duration, i.operation.unit)
        for i in parsed_phys.data
        if i.operation.name == "delay"
    ]
    assert phys_delays == [(0, 100.0, "ns"), (0, 250.0, "ns"), (1, 400.0, "ns")]

    dd_circ = strip_barriers(lower(parse((EXAMPLES / "dd_staggered_xy4.dna").read_text())))
    dd_report = dd_circ.resolve_timing(dt="250ns", policy="exact")
    assert dd_report.adjusted_count == 0
    # Even sites have 9 waits (1us + 7*2us + 1us = 16us = 64 ticks @ 250ns);
    # Odd sites have 8 waits (8*2us = 16us = 64 ticks @ 250ns).
    ticks_by_q = {0: 0, 1: 0, 2: 0, 3: 0}
    for rec in dd_report.records:
        ticks_by_q[rec.qubits[0]] += rec.ticks
    assert ticks_by_q == {0: 64, 1: 64, 2: 64, 3: 64}


def test_cli_qasm_timing_policy_and_require_dt_flags(tmp_path, capsys):
    from dnalang.cli import main as cli_main

    dna_file = tmp_path / "unaligned.dna"
    dna_file.write_text(
        "organism Unaligned {\n"
        "  gene g() on q { wait(15ns) q; cleave q -> c; }\n"
        "  genome { g() @ [0]; }\n"
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_req:
        cli_main(["qasm", str(dna_file), "--require-dt"])
    assert "non-null clock period 'dt'" in str(exc_req.value)

    with pytest.raises(SystemExit) as exc_exact:
        cli_main(["qasm", str(dna_file), "--dt", "4ns", "--timing-policy", "exact"])
    assert "not an exact integer multiple of dt" in str(exc_exact.value)

    rc = cli_main(["qasm", str(dna_file), "--dt", "4ns", "--timing-policy", "quantize"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "delay[4dt] q[0];" in out

