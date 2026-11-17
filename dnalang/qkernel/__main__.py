"""python -m dnalang.qkernel [--vcf FILE --labels TSV] [--out DIR] ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiment import Config, run
from .genotypes import read_vcf, synthetic_populations


def _shots(s: str):
    return [None if x in ("exact", "inf") else int(x) for x in s.split(",")]


def _opt_float(s: str):
    return None if s.lower() in ("none", "off", "disable", "disabled") else float(s)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m dnalang.qkernel",
        description="Quantum fidelity-kernel SVM vs classical baselines on genotypes.",
    )
    src = ap.add_argument_group("data (default: synthetic Balding-Nichols populations)")
    src.add_argument("--vcf", help="plain or .gz VCF with GT fields")
    src.add_argument("--labels", help="TSV with a 'sample' column and a label column (1000G panel file works)")
    src.add_argument("--label-column", default="pop")
    src.add_argument("--max-variants", type=int, default=20000)
    src.add_argument("--stride", type=int, default=1, help="keep every k-th biallelic SNP before LD pruning")
    src.add_argument("--ld-max-r2", type=_opt_float, default=0.2,
                     help="sliding-window pairwise r^2 LD pruning threshold on training folds ('none' disables)")
    src.add_argument("--ld-window", type=int, default=50)
    src.add_argument("--ld-step", type=int, default=10)
    src.add_argument("--max-kinship", type=_opt_float, default=0.177,
                     help="VanRaden/Yang kinship cutoff for relatedness-aware folds ('none' disables)")
    src.add_argument("--pops", type=int, default=4)
    src.add_argument("--per-pop", type=int, default=50)
    src.add_argument("--snps", type=int, default=2000)
    src.add_argument("--fst", type=float, default=0.004)
    src.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--out", default="runs/qkernel")
    ap.add_argument("--components", type=int, default=6, help="PCs = qubits")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--shots", type=_shots, default=_shots("exact,128,512,2048,8192"))
    ap.add_argument("--families", default="product,nn,zz")
    ap.add_argument("--depolarizing", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out / "ledger.jsonl").exists():
        sys.exit(f"error: {out} already holds a run; choose a new --out (ledgers are append-only)")
    if a.vcf:
        if not a.labels:
            sys.exit("error: --vcf needs --labels")
        G, y, s = read_vcf(a.vcf, a.labels, a.label_column, a.max_variants, a.stride)
        name = f"VCF {Path(a.vcf).name} ({G.shape[1]} SNPs, label {a.label_column})"
    else:
        G, y, s = synthetic_populations(a.per_pop, a.pops, a.snps, a.fst, a.data_seed)
        name = (f"SYNTHETIC Balding-Nichols, {a.pops} populations x {a.per_pop}, "
                f"{a.snps} SNPs, F_ST={a.fst}, seed {a.data_seed}")
    cfg = Config(
        n_components=a.components,
        reps=a.reps,
        outer_folds=a.outer_folds,
        inner_folds=a.inner_folds,
        shots=a.shots,
        families=a.families.split(","),
        depolarizing=a.depolarizing,
        margin=a.margin,
        seed=a.seed,
        ld_max_r2=a.ld_max_r2,
        ld_window=a.ld_window,
        ld_step=a.ld_step,
        max_kinship=a.max_kinship,
    )
    print(f"{name}: {len(y)} individuals, {G.shape[1]} variants -> {out}")
    r = run(G, y, s, out, cfg, name)
    print(r["verdict"])
    print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
