"""Genotype inputs, LD pruning, kinship clustering, and leakage-safe preprocessing.

Two sources:

* `synthetic_populations`: Balding-Nichols model (ancestral allele frequencies, population
  frequencies drawn from Beta with differentiation F_ST, diploid genotypes). Synthetic and
  labelled as such everywhere it is reported.
* `read_vcf`: biallelic SNP genotypes (GT field) from a plain or gzipped VCF, e.g. a
  1000 Genomes chromosome subset, with a sample -> label table (the 1000G panel file works).

`Preprocessor` is fitted on training individuals only: allele-frequency filter, optional
sliding-window pairwise r^2 linkage-disequilibrium (LD) pruning, per-SNP standardisation,
PCA and the angle scaling for the quantum map all use training statistics exclusively.

`relatedness_aware_folds` groups related individuals (by explicit pedigree/family IDs and/or
empirical VanRaden/Yang genomic kinship exceeding `max_kinship`, default 0.177 = 2^-2.5) into
connected components so close relatives never straddle train and test partitions.
"""
from __future__ import annotations

import gzip
import hashlib
import math
from dataclasses import dataclass
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
            if len(ref) != 1 or len(alt) != 1 or alt == ".":
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
    h.update("\n".join(map(str, y)).encode())
    h.update("\n".join(map(str, samples)).encode())
    return h.hexdigest()


def ld_prune_r2(G: np.ndarray, window: int = 50, step: int = 10, max_r2: float = 0.2) -> np.ndarray:
    """PLINK-style sliding-window pairwise r^2 LD pruning on training genotypes `G`.

    Within each window of `window` SNPs (advancing by `step` SNPs), repeatedly drops the
    lower-MAF SNP (ties broken toward the higher index) from any pair exceeding `max_r2`.
    """
    if G.shape[1] == 0:
        return np.empty(0, dtype=int)
    window = max(int(window), 2)
    step = max(int(step), 1)
    mu = np.nanmean(G, axis=0)
    maf = np.nan_to_num(np.minimum(mu / 2.0, 1.0 - mu / 2.0), nan=0.0)
    X = np.where(np.isnan(G), mu, G)
    centered = X - mu
    norms = np.linalg.norm(centered, axis=0)
    alive = norms > 1e-12
    U = np.zeros_like(centered, dtype=np.float64)
    U[:, alive] = centered[:, alive] / norms[alive]

    m = G.shape[1]
    start = 0
    while start < m:
        end = min(start + window, m)
        active = np.where(alive[start:end])[0] + start
        while len(active) >= 2:
            R2 = (U[:, active].T @ U[:, active]) ** 2
            np.fill_diagonal(R2, 0.0)
            flat_idx = int(np.argmax(R2))
            a_loc, b_loc = divmod(flat_idx, len(active))
            if R2[a_loc, b_loc] <= max_r2:
                break
            ia, ib = int(active[a_loc]), int(active[b_loc])
            # Prune lower MAF; break ties deterministically by dropping higher column index
            if maf[ia] < maf[ib] - 1e-12:
                drop_loc = a_loc
            elif maf[ib] < maf[ia] - 1e-12:
                drop_loc = b_loc
            else:
                drop_loc = b_loc if ib > ia else a_loc
            alive[active[drop_loc]] = False
            active = np.delete(active, drop_loc)
        if end == m:
            break
        start += step
    return np.where(alive)[0]


def genomic_relationship_matrix(G: np.ndarray, min_maf: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Compute VanRaden/Yang genomic relationship matrix (GRM) and kinship matrix (0.5 * GRM)."""
    mu = np.nanmean(G, axis=0)
    p = np.clip(np.nan_to_num(mu / 2.0, nan=0.0), 0.0, 1.0)
    maf = np.minimum(p, 1.0 - p)
    keep = np.where(maf >= min_maf)[0]
    if len(keep) == 0:
        eye = np.eye(G.shape[0], dtype=np.float64)
        return eye, 0.5 * eye
    pk = p[keep]
    denom = np.sqrt(np.clip(2.0 * pk * (1.0 - pk), 1e-8, None))
    W = np.where(np.isnan(G[:, keep]), 0.0, (G[:, keep] - 2.0 * pk) / denom)
    grm = (W @ W.T) / float(len(keep))
    return grm, 0.5 * grm


def kinship_clusters(
    G: Optional[np.ndarray] = None,
    *,
    n: Optional[int] = None,
    max_kinship: Optional[float] = 0.177,
    family_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Group related individuals into connected components by family_ids and/or empirical kinship."""
    n_ind = int(G.shape[0]) if G is not None else int(n or (len(family_ids) if family_ids is not None else 0))
    parent = list(range(n_ind))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    if family_ids is not None:
        by_fam = {}
        for idx, fid in enumerate(family_ids):
            key = str(fid)
            if key in by_fam:
                union(by_fam[key], idx)
            else:
                by_fam[key] = idx

    if G is not None and max_kinship is not None:
        _, kin = genomic_relationship_matrix(G)
        pairs = np.argwhere(np.triu(kin, k=1) > float(max_kinship))
        for a, b in pairs:
            union(int(a), int(b))

    roots = [find(i) for i in range(n_ind)]
    remap = {r: cid for cid, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots], dtype=int)


@dataclass
class Preprocessor:
    n_components: int = 6
    min_maf: float = 0.05
    ld_max_r2: Optional[float] = None
    ld_window: int = 50
    ld_step: int = 10

    def fit(self, G: np.ndarray) -> "Preprocessor":
        mu_all = np.nanmean(G, axis=0)
        maf = np.minimum(mu_all / 2, 1 - mu_all / 2)
        keep = np.where(np.nan_to_num(maf) >= self.min_maf)[0]
        if self.ld_max_r2 is not None and len(keep) > 1:
            pruned_rel = ld_prune_r2(G[:, keep], window=self.ld_window, step=self.ld_step, max_r2=self.ld_max_r2)
            if len(pruned_rel) >= 1:
                keep = keep[pruned_rel]
        self.keep_ = keep
        self.n_variants_kept_ = int(len(keep))
        mu = mu_all[self.keep_]
        X = np.where(np.isnan(G[:, self.keep_]), mu, G[:, self.keep_])
        self.mu_ = mu
        self.sd_ = np.sqrt(np.clip(mu / 2 * (1 - mu / 2) * 2, 1e-8, None))   # binomial sd, as in EIGENSTRAT
        Z = (X - mu) / self.sd_
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        V = Vt[: self.n_components].T
        if V.shape[1] < self.n_components:
            pad = np.zeros((V.shape[0], self.n_components - V.shape[1]), dtype=V.dtype)
            V = np.hstack([V, pad])
        self.V_ = V
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


def relatedness_aware_folds(
    y: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    G: Optional[np.ndarray] = None,
    max_kinship: Optional[float] = 0.177,
    family_ids: Optional[np.ndarray] = None,
):
    """Stratified group k-fold keeping kinship/pedigree clusters strictly inside a single fold."""
    if G is None and family_ids is None:
        return stratified_folds(y, k, rng)
    if max_kinship is None and family_ids is None:
        return stratified_folds(y, k, rng)

    clusters = kinship_clusters(G=G, n=len(y), max_kinship=max_kinship, family_ids=family_ids)
    uniq_clusters = np.unique(clusters)
    if len(uniq_clusters) == len(y):
        return stratified_folds(y, k, rng)

    classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    perm = rng.permutation(uniq_clusters)
    members = [np.where(clusters == cid)[0] for cid in perm]
    # Place largest multi-member families first, then singletons, balancing per-class & total counts
    members.sort(key=lambda idx: -len(idx))

    fold = np.empty(len(y), dtype=int)
    fold_class_counts = np.zeros((k, len(classes)), dtype=int)
    fold_totals = np.zeros(k, dtype=int)

    for grp in members:
        grp_classes = [class_to_idx[val] for val in y[grp]]
        maj_c = max(set(grp_classes), key=grp_classes.count)
        best_f = min(range(k), key=lambda f: (fold_class_counts[f, maj_c], fold_totals[f]))
        fold[grp] = best_f
        for ci in grp_classes:
            fold_class_counts[best_f, ci] += 1
        fold_totals[best_f] += len(grp)

    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]
