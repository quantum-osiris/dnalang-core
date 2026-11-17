"""Tests for multi-typed sequence operators (@splice, @anneal, @methylate, }{) and OpenQASM 3 lowering."""
import math
import pytest

from dnalang.parser import parse
from dnalang.seq_ops import (
    BinarySeqOpNode,
    CodonOp,
    DnaLangLowerer,
    DnaTypeError,
    EpigeneticMaskNode,
    GateMnemonic,
    GeneNode,
    LoweringConfig,
    MultiTypedDispatcher,
    OpKind,
    QuantumStateNode,
    SubstrateProfile,
)


def _sample_strands():
    promoter_a = GeneNode(
        name="promoter_a",
        codons=(
            CodonOp(GateMnemonic.HELIX, (0,), trait_tag="stabilizer_X"),
            CodonOp(GateMnemonic.TWIST, (0,), angle=0.5 * math.pi, trait_tag="purine_A"),
            CodonOp(GateMnemonic.FOLD, (0,), angle=0.25 * math.pi, trait_tag="purine_G"),
        ),
    )
    enhancer_b = GeneNode(
        name="enhancer_b",
        codons=(
            CodonOp(GateMnemonic.HELIX, (1,), trait_tag="stabilizer_Z"),
            CodonOp(GateMnemonic.TWIST, (1,), angle=0.5 * math.pi, trait_tag="pyrimidine_T"),
            CodonOp(GateMnemonic.FOLD, (1,), angle=0.25 * math.pi, trait_tag="purine_A"),  # mismatch
        ),
    )
    return promoter_a, enhancer_b


def test_methylate_anneal_express_pipeline_and_qasm3():
    dispatcher = MultiTypedDispatcher()
    promoter_a, enhancer_b = _sample_strands()
    silencer = EpigeneticMaskNode(name="m_mask", weights=(1.0, 0.5, 0.0))
    q_reg = QuantumStateNode(register_id="q_twin", num_qubits=3, entropy_floor=0.85)

    pipeline = BinarySeqOpNode(
        op=OpKind.EXPRESS,
        left=BinarySeqOpNode(
            op=OpKind.ANNEAL,
            left=BinarySeqOpNode(OpKind.METHYLATE, silencer, promoter_a),
            right=enhancer_b,
        ),
        right=q_reg,
    )

    resolved = dispatcher.evaluate(pipeline)
    assert isinstance(resolved, QuantumStateNode)
    # Silenced 3rd codon in promoter_a becomes IDENTITY and is omitted, while mismatch tension is emitted
    assert [op.mnemonic for op in resolved.applied_ir] == [
        GateMnemonic.HELIX,
        GateMnemonic.HELIX,
        GateMnemonic.BOND,
        GateMnemonic.TWIST,
        GateMnemonic.TWIST,
        GateMnemonic.BOND,
        GateMnemonic.FOLD,
        GateMnemonic.TWIST,
    ]
    # 0.5 * (0.5 * pi) == pi / 4
    assert math.isclose(resolved.applied_ir[3].angle, 0.25 * math.pi)
    # Mismatch penalty == 3*pi/20
    assert math.isclose(resolved.applied_ir[-1].angle, 0.15 * math.pi)

    lowerer = DnaLangLowerer(LoweringConfig(barrier_policy="epoch", insert_idle_delays=True))
    qasm3 = lowerer.to_openqasm3(resolved)
    assert "OPENQASM 3.0;" in qasm3
    assert "rz(pi/4) q_twin[0];" in qasm3
    assert "rz(3*pi/20) q_twin[0];  // mismatch_tension" in qasm3
    assert "delay[800dt] q_twin[2];  // idle_slack" in qasm3
    assert "c_q_twin[2] = measure q_twin[2];" in qasm3

    ir = lowerer.lower_to_ir(resolved)
    assert len(ir.sha256_digest) == 64
    qc_ir = ir.to_qc_ir()
    assert qc_ir.n_qubits == 3
    assert qc_ir.n_cbits == 3
    assert qc_ir.two_qubit_count() == 2


def test_homogeneous_and_heterogeneous_splice_and_frame_invariant():
    dispatcher = MultiTypedDispatcher()
    promoter_a, enhancer_b = _sample_strands()

    spliced = dispatcher.evaluate(BinarySeqOpNode(OpKind.SPLICE, promoter_a, enhancer_b))
    assert isinstance(spliced, GeneNode)
    assert len(spliced.codons) == 6

    # Intron excision violating mod-3 must fail closed
    broken_intron = GeneNode(
        name="bad_intron",
        codons=promoter_a.codons,
        intron_spans=((0, 1),),  # leaves 2 exons -> 2 + 3 = 5 (not mod 3)
    )
    with pytest.raises(DnaTypeError, match="frame-shift violation"):
        dispatcher.evaluate(BinarySeqOpNode(OpKind.SPLICE, broken_intron, enhancer_b))

    # Heterogeneous @splice shifts wires and checks register capacity
    q4 = QuantumStateNode(register_id="reg4", num_qubits=4)
    shifted = dispatcher.evaluate(
        BinarySeqOpNode(OpKind.SPLICE, promoter_a, q4, insertion_offset=2)
    )
    assert isinstance(shifted, QuantumStateNode)
    assert all(op.qubits == (2,) for op in shifted.applied_ir)

    with pytest.raises(DnaTypeError, match="qubit overflow"):
        dispatcher.evaluate(BinarySeqOpNode(OpKind.SPLICE, promoter_a, q4, insertion_offset=4))


def test_adjoint_and_substrate_depth_guard_and_parser_bridge():
    promoter_a, enhancer_b = _sample_strands()
    dispatcher = MultiTypedDispatcher(
        substrate=SubstrateProfile("TinyEdge", "aarch64", 30.0, max_circuit_depth=4)
    )
    transcript = dispatcher.evaluate(BinarySeqOpNode(OpKind.ANNEAL, promoter_a, enhancer_b))
    inv = transcript.adjoint()
    assert inv.ops[0].mnemonic == transcript.ops[-1].mnemonic
    assert math.isclose(inv.ops[0].angle, -transcript.ops[-1].angle)

    with pytest.raises(DnaTypeError, match="exceeds TinyEdge horizon"):
        dispatcher.evaluate(
            BinarySeqOpNode(OpKind.EXPRESS, transcript, QuantumStateNode("q", 2))
        )

    # Bridge from existing dnalang.parser.parse Gene AST
    org = parse(
        """
        organism demo {
            gene bell_unit(theta: angle) on a, b {
                helix a;
                twist(theta) a;
                bond a, b;
            }
            genome {
                bell_unit(pi / 2) @ [0, 1];
            }
            fitness w2_robustness;
        }
        """
    )
    lifted = GeneNode.from_ast_gene(org.genes[0], {"theta": math.pi / 2}, enforce_mod3=True)
    full_disp = MultiTypedDispatcher()
    out_state = full_disp.evaluate(
        BinarySeqOpNode(OpKind.EXPRESS, lifted, QuantumStateNode("bell_reg", 2))
    )
    assert len(out_state.applied_ir) == 3


def test_probe_substrate_integrates_with_osiris_hardware_mesh_and_s26_edge(monkeypatch):
    import sys
    import types
    from dnalang.seq_ops import probe_substrate

    prof = probe_substrate()
    assert isinstance(prof, SubstrateProfile)
    assert prof.max_circuit_depth in (256, 4096)

    fake_mesh = types.ModuleType("osiris_hardware_mesh")
    fake_mesh.get_current_profile = lambda: types.SimpleNamespace(
        machine_arch="aarch64", is_termux=True, is_android=True, execution_timeout_s=300
    )
    monkeypatch.setitem(sys.modules, "osiris_hardware_mesh", fake_mesh)
    s26_prof = probe_substrate()
    assert s26_prof == SubstrateProfile("Node_Alpha_Edge", "aarch64", 300.0, 256)

    explicit = probe_substrate(
        profile=types.SimpleNamespace(
            machine_arch="x86_64", is_termux=False, is_android=False, execution_timeout_s=1800
        )
    )
    assert explicit == SubstrateProfile("Node_Beta_Workstation", "x86_64", 1800.0, 4096)


def test_import_dnalang_and_qkernel_are_isolated_from_cross_repo_and_eager_numpy(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Plant a broken osiris_hardware_mesh.py on PYTHONPATH: dnalang must never import it.
    (tmp_path / "osiris_hardware_mesh.py").write_text("raise RuntimeError('must not be imported')\n")
    repo_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tmp_path}:{repo_root}"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, dnalang, dnalang.qkernel; "
                "assert 'osiris_hardware_mesh' not in sys.modules; "
                "assert 'numpy' not in sys.modules; "
                "assert 'PrecomputedSVC' in dir(dnalang.qkernel)"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
