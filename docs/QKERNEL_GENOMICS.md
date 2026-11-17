# Quantum-kernel genomics benchmark (`dnalang.qkernel`)

Tests whether quantum fidelity-kernel SVMs add predictive value on genotype data, against
tuned classical baselines, with every step written to a hash-chained `dnalang.Ledger`.

```
python -m dnalang.qkernel --out runs/qk1                     # synthetic populations (default F_ST 0.004)
python -m dnalang.qkernel --fst 0.02 --out runs/qk2          # easier synthetic task
python -m dnalang.qkernel --vcf chr22.vcf.gz --labels integrated_call_samples.panel \
       --label-column super_pop --stride 50 --ld-max-r2 0.2 --out runs/1kg
dnalang verify-ledger runs/qk1/ledger.jsonl
```

Pipeline: individuals split first using relatedness-aware group folds (VanRaden/Yang genomic
kinship cutoff `--max-kinship 0.177` and/or pedigree IDs) → MAF filter, sliding-window pairwise
$r^2$ LD pruning (`--ld-max-r2 0.2`), standardisation, PCA and angle scaling fitted on training
rows only (refitted inside every inner fold) → kernels → C-SVM (SMO, one-vs-one, plus out-of-fold
temperature/bias probability calibration) and classical baselines → pooled outer-fold predictions
→ paired bootstrap + proper scoring rules (multiclass log-loss, Brier score, ECE).

| piece | what it is |
|---|---|
| `feature_maps.py` | `product` (no entanglement, negative control), `nn` (CZ chain), `zz` (Havlíček ZZ). Templates instantiate `qc_ir.Circuit`; `fidelity_circuit` builds U(y)†U(x) + measure for OpenQASM 3 export |
| `statevector.py` | batched exact simulator, checked against `qiskit.quantum_info.Statevector` in the tests |
| `genotypes.py` | synthetic Balding-Nichols & VCF/gzip loaders, PLINK-style sliding-window $r^2$ LD pruning (`ld_prune_r2`), VanRaden/Yang GRM & connected-component `relatedness_aware_folds`, train-only `Preprocessor` |
| `kernels.py` | K = \|⟨φ(x)\|φ(y)⟩\|², binomial finite-shot sampling (diagonal measured too), optional global-depolarizing *model*, PSD repair by eigenvalue clipping, diagnostics (off-diagonal spread = concentration, effective rank, kernel-target alignment) |
| `svm.py` | precomputed-kernel C-SVM (dual optimum checked against scipy SLSQP) with cross-validated out-of-fold multiclass probability calibration (`predict_proba`) |
| `baselines.py` | multinomial logistic regression, kNN, vectorised Random Forest (`RandomForest`), and multiclass Gradient Boosting (`GradientBoosting`); `experiment.py` adds linear/RBF/poly SVMs and a random-Fourier-feature control of matching dimension 2^n |
| `experiment.py` | pre-registration entry (config, dataset hash, kinship cluster count, code hash, H0, decision rule) before fitting; per-fold entries with chosen hyperparameters, calibrated `test_log_loss`, and kernel hashes; raw + repaired kernels saved for exact and 2048-shot models |

H0: best quantum balanced accuracy − best classical ≤ margin (0.02). Rejected only if the
95% bootstrap CI lower bound exceeds the margin.

Not included: physical QPU hardware execution (the circuits in `circuits/` are what a hardware
run would submit) and live 1000 Genomes download/execution without operator authorization.
