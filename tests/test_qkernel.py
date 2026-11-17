import numpy as np
import pytest
from scipy.optimize import minimize

from dnalang.qkernel import feature_map, fidelity_circuit, fidelity_kernel, psd_repair, sample_kernel, simulate
from dnalang.qkernel.experiment import Config, run
from dnalang.qkernel.genotypes import Preprocessor, stratified_folds, synthetic_populations
from dnalang.qkernel.svm import PrecomputedSVC, dual_objective, smo


@pytest.mark.parametrize("family", ["product", "nn", "zz"])
def test_simulator_matches_qiskit(family):
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, size=(3, 4))
    t = feature_map(family, 4, reps=2)
    S = simulate(t, X)
    for i in range(3):
        ref = Statevector(t.instantiate(X[i]).to_qiskit()).data
        assert np.allclose(S[i], ref, atol=1e-10)


@pytest.mark.parametrize("family", ["product", "nn", "zz"])
def test_compute_uncompute_probability_is_kernel(family):
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(2)
    X = rng.uniform(0, np.pi, size=(2, 3))
    t = feature_map(family, 3, reps=2)
    qc = fidelity_circuit(t, X[0], X[1]).to_qiskit(); qc.remove_final_measurements()
    p0 = abs(Statevector(qc).data[0]) ** 2
    S = simulate(t, X)
    assert p0 == pytest.approx(fidelity_kernel(S[:1], S[1:])[0, 0], abs=1e-10)
    assert np.allclose(np.diag(fidelity_kernel(S, S)), 1.0)


def test_fidelity_circuit_exports_qasm3():
    t = feature_map("zz", 3)
    q = fidelity_circuit(t, np.ones(3), np.zeros(3)).to_qasm3()
    assert q.startswith("OPENQASM 3.0;") and q.count("measure") == 3


def test_finite_shot_kernel_is_unbiased_and_repair_is_psd():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, np.pi, size=(30, 4))
    S = simulate(feature_map("zz", 4), X)
    F = fidelity_kernel(S, S)
    K = sample_kernel(F, 64, rng, symmetric=True)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(psd_repair(K)).min() > -1e-10
    mean = np.mean([sample_kernel(F, 256, rng, True) for _ in range(200)], axis=0)
    assert np.abs(mean - F).max() < 0.03


def test_smo_matches_scipy_dual():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(-1, 1, (15, 2)), rng.normal(1, 1, (15, 2))])
    y = np.r_[-np.ones(15), np.ones(15)]
    K = np.exp(-0.5 * ((X[:, None] - X[None]) ** 2).sum(-1))
    C = 1.0
    a, _ = smo(K, y, C, tol=1e-6)
    r = minimize(lambda v: dual_objective(K, y, v), np.zeros(30), bounds=[(0, C)] * 30,
                 constraints=[{"type": "eq", "fun": lambda v: v @ y}], method="SLSQP",
                 options={"ftol": 1e-12, "maxiter": 1000})
    assert dual_objective(K, y, a) == pytest.approx(r.fun, abs=1e-4)
    assert abs(a @ y) < 1e-8 and a.min() >= 0 and a.max() <= C


def test_multiclass_svc_separates_blobs():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(c, 0.3, (20, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    y = np.repeat(np.array(["a", "b", "c"]), 20)
    K = X @ X.T + 1
    assert (PrecomputedSVC(10).fit(K, y).predict(K) == y).mean() > 0.95


def test_preprocessor_uses_training_rows_only():
    G, y, _ = synthetic_populations(20, 2, 300, 0.05, seed=1)
    tr, te = stratified_folds(y, 4, np.random.default_rng(0))[0]
    a = Preprocessor(3).fit(G[tr])
    G2 = G.copy(); G2[te] = 2 - G2[te]           # change every test genotype
    b = Preprocessor(3).fit(G2[tr])
    assert np.allclose(a.classical(G[te]), b.classical(G[te]))
    assert np.allclose(a.angles(G[tr]).min(0), 0) and np.allclose(a.angles(G[tr]).max(0), np.pi)


def test_end_to_end_small_run(tmp_path):
    G, y, s = synthetic_populations(12, 2, 200, 0.05, seed=2)
    cfg = Config(n_components=3, outer_folds=2, inner_folds=2, shots=[None, 256], families=["product", "zz"],
                 bandwidths=[1.0], bootstrap=50, save_kernel_shots=[None])
    r = run(G, y, s, tmp_path, cfg, "test", log=lambda *_: None)
    assert r["ledger_ok"]
    assert {"rbf_svm", "q_zz_exact", "q_zz_s256", "q_product_exact"} <= set(r["models"])
    assert (tmp_path / "report.md").read_text().startswith("# Quantum-kernel genomics benchmark")
    assert (tmp_path / "circuits" / "fidelity_zz.qasm").exists()
    kinds = [line.split('"kind":"')[1].split('"')[0] for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert kinds[0] == "qkernel.preregister" and kinds[-1] == "qkernel.result"
