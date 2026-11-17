"""Genotype inputs and leakage-safe preprocessing.

Two sources:

* `synthetic_populations`: Balding-Nichols model (ancestral allele frequencies, population
  frequencies drawn from Beta with differentiation F_ST, diploid genotypes). Synthetic and
  labelled as such everywhere it is reported.
* `read_vcf`: biallelic SNP genotypes (GT field) from a plain or gzipped VCF, e.g. a
  1000 Genomes chromosome subset, with a sample -> label table (the 1000G panel file works).

`Preprocessor` is fitted on training individuals only: allele-frequency filter, per-SNP
standardisation, PCA and the angle scaling for the quantum map all use training statistics.
"""
from __future__ import annotations

import gzip
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def synthetic_populations(n_per_pop=50, n_pops=4, n_snps=2000, fst=0.004, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n_snps)
    a, b = p * (1 - fst) / fst, (1 - p) * (1 - fst) / fst
    P = rng.beta(a, b, size=(n_pops, n_snps))
    G = np.vstack([rng.binomial(2, P[k], size=(n_per_pop, n_snps)) for k in range(n_pops)]).astype(float)
    y = np.repeat(np.array([f"POP{k}" for k in range(n_pops)]), n_per_pop)
    samples = np.array([f"S{i:04d}" for i in range(len(y))])
    return G, y, samples


def read_vcf(path, labels_path, label_column="pop", max_variants=20000, stride=1,
             min_samples_per_label=10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = {}
    with open(labels_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        si, li = header.index("sample"), header.index(label_column)
        for line in f:
            r = line.rstrip("\n").split("\t")
            if len(r) > max(si, li):
                lab[r[si]] = r[li]
    opener = gzip.open if str(path).endswith(".gz") else open
    rows, samples, keep = [], None, None
    n_seen = 0
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                keep = [i for i, s in enumerate(samples) if s in lab]
                continue
            r = line.rstrip("\n").split("\t")
            ref, alt = r[3], r[4]
            if len(ref) != 1 or len(alt) != 1 or alt == "." :
                continue
            n_seen += 1
            if (n_seen - 1) % stride:
                continue
            fmt = r[8].split(":")
            gi = fmt.index("GT")
            g = np.empty(len(keep))
            for k, i in enumerate(keep):
                gt = r[9 + i].split(":")[gi].replace("|", "/").split("/")
                g[k] = np.nan if "." in gt else sum(int(x != "0") for x in gt)
            rows.append(g)
            if len(rows) >= max_variants:
                break
    if samples is None or not rows:
        raise ValueError(f"{path}: no #CHROM header or no biallelic SNPs")
    S = np.array([samples[i] for i in keep])
    y = np.array([lab[s] for s in S])
    G = np.array(rows).T
    labels, counts = np.unique(y, return_counts=True)
    ok = np.isin(y, labels[counts >= min_samples_per_label])
    return G[ok], y[ok], S[ok]


def dataset_sha256(G, y, samples) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.nan_to_num(G, nan=-1), dtype=np.float32).tobytes())
    h.update("\n".join(map(str, y)).encode()); h.update("\n".join(map(str, samples)).encode())
    return h.hexdigest()


@dataclass
class Preprocessor:
    n_components: int = 6
    min_maf: float = 0.05

    def fit(self, G: np.ndarray) -> "Preprocessor":
        mu = np.nanmean(G, axis=0)
        maf = np.minimum(mu / 2, 1 - mu / 2)
        self.keep_ = np.where(np.nan_to_num(maf) >= self.min_maf)[0]
        mu = mu[self.keep_]
        X = np.where(np.isnan(G[:, self.keep_]), mu, G[:, self.keep_])
        self.mu_ = mu
        self.sd_ = np.sqrt(np.clip(mu / 2 * (1 - mu / 2) * 2, 1e-8, None))   # binomial sd, as in EIGENSTRAT
        Z = (X - mu) / self.sd_
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        self.V_ = Vt[: self.n_components].T
        P = Z @ self.V_
        self.pc_sd_ = P.std(0) + 1e-12
        S = P / self.pc_sd_
        self.lo_, self.hi_ = S.min(0), S.max(0)
        return self

    def classical(self, G: np.ndarray) -> np.ndarray:
        X = np.where(np.isnan(G[:, self.keep_]), self.mu_, G[:, self.keep_])
        return ((X - self.mu_) / self.sd_) @ self.V_ / self.pc_sd_

    def angles(self, G: np.ndarray) -> np.ndarray:
        """Training range -> [0, pi]; test points outside the range are not clipped."""
        S = self.classical(G)
        return math.pi * (S - self.lo_) / np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)


def stratified_folds(y: np.ndarray, k: int, rng: np.random.Generator):
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]
