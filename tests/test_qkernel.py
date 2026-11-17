import gzip

import numpy as np
import pytest
from scipy.optimize import minimize

from dnalang.qkernel import (
    GradientBoosting,
    PrecomputedSVC,
    Preprocessor,
    RandomForest,
    feature_map,
    fidelity_circuit,
    fidelity_kernel,
    genomic_relationship_matrix,
    ld_prune_r2,
    psd_repair,
    relatedness_aware_folds,
    sample_kernel,
    simulate,
)
from dnalang.qkernel.experiment import Config, multiclass_brier_score, multiclass_log_loss, run
from dnalang.qkernel.genotypes import read_vcf, stratified_folds, synthetic_populations
from dnalang.qkernel.svm import dual_objective, smo


@pytest.mark.parametrize("family", ["product", "nn", "zz"])
def test_simulator_matches_qiskit(family):
    pytest.importorskip("qiskit")
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
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(2)
    X = rng.uniform(0, np.pi, size=(2, 3))
    t = feature_map(family, 3, reps=2)
    qc = fidelity_circuit(t, X[0], X[1]).to_qiskit()
    qc.remove_final_measurements()
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


def test_multiclass_svc_separates_blobs_and_calibrates_probabilities():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(c, 0.3, (20, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    y = np.repeat(np.array(["a", "b", "c"]), 20)
    K = X @ X.T + 1
    clf = PrecomputedSVC(10, calibrate=True, seed=5).fit(K, y)
    assert (clf.predict(K) == y).mean() > 0.95
    proba = clf.predict_proba(K)
    assert proba.shape == (60, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert multiclass_log_loss(y, proba, clf.classes_) < 0.35


def test_preprocessor_uses_training_rows_only():
    G, y, _ = synthetic_populations(20, 2, 300, 0.05, seed=1)
    tr, te = stratified_folds(y, 4, np.random.default_rng(0))[0]
    a = Preprocessor(3).fit(G[tr])
    G2 = G.copy()
    G2[te] = 2 - G2[te]           # change every test genotype
    b = Preprocessor(3).fit(G2[tr])
    assert np.allclose(a.classical(G[te]), b.classical(G[te]))
    assert np.allclose(a.angles(G[tr]).min(0), 0) and np.allclose(a.angles(G[tr]).max(0), np.pi)


def test_ld_pruning_removes_collinear_snps_and_uses_training_rows_only():
    G_base, y, _ = synthetic_populations(25, 2, 40, 0.03, seed=11)
    # Plant exact duplicate and inverted SNPs inside LD windows
    G = np.hstack([G_base[:, :10], G_base[:, :10], 2.0 - G_base[:, :10], G_base[:, 10:]])
    kept = ld_prune_r2(G, window=35, step=5, max_r2=0.2)
    assert len(kept) < G.shape[1] - 15
    # Verify every pair within the same 35-SNP window of kept SNPs satisfies r^2 <= 0.2
    Gk = G[:, kept]
    Z = (Gk - Gk.mean(0)) / np.linalg.norm(Gk - Gk.mean(0), axis=0)
    R2 = (Z.T @ Z) ** 2
    np.fill_diagonal(R2, 0.0)
    in_win = np.abs(kept[:, None] - kept[None, :]) < 35
    assert R2[in_win].max() <= 0.2 + 1e-9

    # Full-chromosome window guarantees global r^2 <= 0.2 across all retained SNPs
    kept_all = ld_prune_r2(G, window=G.shape[1], step=5, max_r2=0.2)
    Za = (G[:, kept_all] - G[:, kept_all].mean(0)) / np.linalg.norm(G[:, kept_all] - G[:, kept_all].mean(0), axis=0)
    R2_all = (Za.T @ Za) ** 2
    np.fill_diagonal(R2_all, 0.0)
    assert R2_all.max() <= 0.2 + 1e-9

    # Train-only leakage invariance with LD pruning active
    tr, te = stratified_folds(y, 4, np.random.default_rng(9))[0]
    p1 = Preprocessor(3, ld_max_r2=0.2, ld_window=25, ld_step=5).fit(G[tr])
    G_mut = G.copy()
    G_mut[te] = 2.0 - G_mut[te]
    p2 = Preprocessor(3, ld_max_r2=0.2, ld_window=25, ld_step=5).fit(G_mut[tr])
    assert np.array_equal(p1.keep_, p2.keep_)
    assert np.allclose(p1.classical(G[te]), p2.classical(G[te]))


def test_relatedness_aware_folds_keep_relatives_and_pedigrees_together():
    G, y, _ = synthetic_populations(15, 2, 400, 0.04, seed=12)
    # Plant duplicate/twin pairs (0,1), (15,16) and explicit pedigree family_ids
    G[1] = G[0].copy()
    G[16] = G[15].copy()
    fams = np.array([f"FAM_{i}" for i in range(len(y))])
    fams[3] = fams[2]  # pedigree-linked pair even without identical genotypes

    folds = relatedness_aware_folds(
        y, 3, np.random.default_rng(7), G=G, max_kinship=0.177, family_ids=fams
    )
    _, kin = genomic_relationship_matrix(G)
    assert kin[0, 1] > 0.35 and kin[15, 16] > 0.35
    for tr, te in folds:
        assert len(set(tr) & set(te)) == 0
        # Zero cross-fold kinship above threshold and zero split families
        assert float(np.max(kin[np.ix_(tr, te)])) <= 0.177
        assert set(fams[tr]).isdisjoint(set(fams[te]))


def test_random_forest_and_gradient_boosting_baselines():
    rng = np.random.default_rng(21)
    X = np.vstack([rng.normal(loc, 0.45, (25, 4)) for loc in ([-2, 0, 0, 0], [2, 0, 0, 0], [0, 2.5, 0, 0])])
    y = np.repeat(np.array(["POP0", "POP1", "POP2"]), 25)
    classes = np.unique(y)

    for clf in (RandomForest(n_estimators=35, max_depth=4, seed=3), GradientBoosting(n_estimators=25, learning_rate=0.15)):
        clf.fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        assert (preds == y).mean() > 0.92
        assert proba.shape == (75, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert multiclass_log_loss(y, proba, classes) < 0.40
        assert multiclass_brier_score(y, proba, classes) < 0.20


def test_read_vcf_gzip_offline_fixture(tmp_path):
    panel = tmp_path / "samples.panel"
    samples = [f"HG{i:04d}" for i in range(24)]
    lines = ["sample\tpop\tsuper_pop"] + [
        f"{s}\t{'GBR' if i < 12 else 'YRI'}\t{'EUR' if i < 12 else 'AFR'}" for i, s in enumerate(samples)
    ]
    panel.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rng = np.random.default_rng(33)
    vcf_path = tmp_path / "mini_chr22.vcf.gz"
    with gzip.open(vcf_path, "wt", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        for pos in range(1, 41):
            gts = []
            for i in range(24):
                p = 0.2 if i < 12 else 0.8
                a1, a2 = rng.binomial(1, p, size=2)
                gts.append("./." if (pos == 5 and i == 0) else f"{a1}|{a2}")
            fh.write(f"22\t{pos * 100}\trs{pos}\tA\tG\t100\tPASS\t.\tGT\t" + "\t".join(gts) + "\n")

    G, y, S = read_vcf(vcf_path, panel, label_column="super_pop", min_samples_per_label=10)
    assert G.shape == (24, 40)
    assert np.isnan(G[0, 4])
    assert set(y) == {"EUR", "AFR"} and len(S) == 24


def test_end_to_end_small_run(tmp_path):
    G, y, s = synthetic_populations(12, 2, 200, 0.05, seed=2)
    cfg = Config(n_components=3, outer_folds=2, inner_folds=2, shots=[None, 256], families=["product", "zz"],
                 bandwidths=[1.0], bootstrap=50, save_kernel_shots=[None])
    r = run(G, y, s, tmp_path, cfg, "test", log=lambda *_: None)
    assert r["ledger_ok"]
    assert {
        "rbf_svm", "random_forest", "gradient_boosting",
        "q_zz_exact", "q_zz_s256", "q_product_exact",
    } <= set(r["models"])
    for m_res in r["models"].values():
        assert np.isfinite(m_res["log_loss"]) and m_res["log_loss"] >= 0.0
        assert 0.0 <= m_res["brier_score"] <= 2.0
        assert 0.0 <= m_res["ece"] <= 1.0
    assert r["mean_variants_kept_after_ld"] > 0
    assert (tmp_path / "report.md").read_text().startswith("# Quantum-kernel genomics benchmark")
    assert (tmp_path / "circuits" / "fidelity_zz.qasm").exists()
    kinds = [line.split('"kind":"')[1].split('"')[0] for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert kinds[0] == "qkernel.preregister" and kinds[-1] == "qkernel.result"


def test_import_qkernel_is_lazy_and_avoids_eager_numpy():
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=repo_root)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, dnalang, dnalang.qkernel; "
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
