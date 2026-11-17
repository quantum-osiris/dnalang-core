# Quantum-kernel genomics benchmark (`dnalang.qkernel`)

Tests whether quantum fidelity-kernel SVMs add predictive value on genotype data, against
tuned classical baselines, with every step written to a hash-chained `dnalang.Ledger`.

```
python -m dnalang.qkernel --out runs/qk1                     # synthetic populations (default F_ST 0.004)
python -m dnalang.qkernel --fst 0.02 --out runs/qk2          # easier synthetic task
python -m dnalang.qkernel --vcf chr22.vcf.gz --labels integrated_call_samples.panel \
       --label-column super_pop --stride 50 --out runs/1kg   # real VCF + sample table
dnalang verify-ledger runs/qk1/ledger.jsonl
```

Pipeline: individuals split first → MAF filter, standardisation, PCA and angle scaling fitted on
training rows only (refitted inside every inner fold) → kernels → C-SVM (SMO, one-vs-one) →
pooled outer-fold predictions → paired bootstrap.

| piece | what it is |
|---|---|
| `feature_maps.py` | `product` (no entanglement, negative control), `nn` (CZ chain), `zz` (Havlíček ZZ). Templates instantiate `qc_ir.Circuit`; `fidelity_circuit` builds U(y)†U(x) + measure for OpenQASM 3 export |
| `statevector.py` | batched exact simulator, checked against `qiskit.quantum_info.Statevector` in the tests |
| `kernels.py` | K = \|⟨φ(x)\|φ(y)⟩\|², binomial finite-shot sampling (diagonal measured too), optional global-depolarizing *model*, PSD repair by eigenvalue clipping, diagnostics (off-diagonal spread = concentration, effective rank, kernel-target alignment) |
| `svm.py` | precomputed-kernel C-SVM; dual optimum checked against scipy SLSQP |
| `baselines.py` | logistic regression, kNN; `experiment.py` adds linear/RBF/poly SVMs and a random-Fourier-feature control of matching dimension 2^n |
| `experiment.py` | pre-registration entry (config, dataset hash, code hash, H0, decision rule) before fitting; per-fold entries with chosen hyperparameters and kernel hashes; raw + repaired kernels saved for exact and 2048-shot models |

H0: best quantum balanced accuracy − best classical ≤ margin (0.02). Rejected only if the
95% bootstrap CI lower bound exceeds the margin.

Not included: hardware execution (the circuits in `circuits/` are what a run would submit),
LD pruning beyond `--stride` thinning, relatedness-aware splits, random forest / boosting
baselines, probability calibration and log loss for the SVMs.
