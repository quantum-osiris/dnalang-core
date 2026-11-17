"""Pre-registered, leakage-safe benchmark: quantum fidelity-kernel SVMs vs classical baselines
on genotype-derived features, with every step recorded in a hash-chained dnalang Ledger.

Pipeline per outer fold (individuals are split first; nothing is fitted on test individuals):
    genotypes -> MAF filter, standardise, PCA (train only) -> classical view / angle view
    -> inner CV (preprocessing refitted on each inner-train split) selects hyperparameters
    -> refit on outer train -> predict outer test.

Null hypothesis H0 (written to the ledger before any model is fitted): the best quantum
kernel's balanced accuracy does not exceed the best classical model's by more than `margin`.
H0 is rejected only if the lower end of the paired-bootstrap 95% CI of the difference
exceeds `margin`. Comparing against the *best* classical model is deliberately conservative
toward any advantage claim.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..ledger import Ledger
from . import kernels as kn
from .baselines import KNN, LogisticRegression
from .feature_maps import FAMILIES, feature_map, fidelity_circuit
from .genotypes import Preprocessor, dataset_sha256, stratified_folds
from .statevector import simulate
from .svm import PrecomputedSVC

C_GRID = [0.1, 1.0, 10.0, 100.0]


@dataclass
class Config:
    n_components: int = 6
    reps: int = 2
    outer_folds: int = 5
    inner_folds: int = 3
    shots: List[Optional[int]] = field(default_factory=lambda: [None, 128, 512, 2048, 8192])
    families: List[str] = field(default_factory=lambda: list(FAMILIES))
    bandwidths: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    depolarizing: float = 0.0
    margin: float = 0.02
    bootstrap: int = 2000
    seed: int = 7
    save_kernel_shots: List[Optional[int]] = field(default_factory=lambda: [None, 2048])


# --- metrics --------------------------------------------------------------------
def balanced_accuracy(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def macro_f1(y, p):
    f = []
    for c in np.unique(y):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def confusion(y, p, classes):
    return [[int(np.sum((y == a) & (p == b))) for b in classes] for a in classes]


# --- models ---------------------------------------------------------------------
def _quantum(family, shots):
    def kernel(tr, te, ytr, prm, rng, cfg, keep=None):
        t = feature_map(family, tr.shape[1], cfg.reps)
        Str, Ste = simulate(t, prm["bandwidth"] * tr), simulate(t, prm["bandwidth"] * te)
        Ftr, Fte = kn.fidelity_kernel(Str, Str), kn.fidelity_kernel(Ste, Str)
        if shots is None:
            Ktr, Kte, raw = Ftr, Fte, Ftr
        else:
            raw = kn.sample_kernel(Ftr, shots, rng, True, cfg.depolarizing, tr.shape[1])
            Kte = kn.sample_kernel(Fte, shots, rng, False, cfg.depolarizing, tr.shape[1])
            Ktr = kn.psd_repair(raw)
        if keep is not None:
            keep.update(exact_train=Ftr, raw_train=raw, repaired_train=Ktr, test=Kte)
        return Ktr, Kte
    name = f"q_{family}" + ("_exact" if shots is None else f"_s{shots}")
    return dict(name=name, kind="kernel", view="angles", quantum=True, family=family, shots=shots,
                grid=[], kernel=kernel)


def build_models(cfg: Config):
    d = cfg.n_components
    m = [
        dict(name="linear_svm", kind="kernel", view="classical", grid=[{}],
             kernel=lambda tr, te, y, p, r, c, keep=None: (tr @ tr.T, te @ tr.T)),
        dict(name="rbf_svm", kind="kernel", view="classical",
             grid=[{"gamma": g / d} for g in (0.03, 0.1, 0.3, 1.0, 3.0)],
             kernel=lambda tr, te, y, p, r, c, keep=None: (kn.rbf_kernel(tr, tr, p["gamma"]), kn.rbf_kernel(tr, te, p["gamma"]))),
        dict(name="poly3_svm", kind="kernel", view="classical",
             grid=[{"gamma": g / d} for g in (0.1, 0.3, 1.0)],
             kernel=lambda tr, te, y, p, r, c, keep=None: (kn.poly_kernel(tr, tr, p["gamma"]), kn.poly_kernel(tr, te, p["gamma"]))),
        dict(name="rff_svm_control", kind="kernel", view="angles",
             grid=[{"scale": s} for s in (0.1, 0.3, 1.0, 3.0)],
             kernel=lambda tr, te, y, p, r, c, keep=None: (
                 kn.random_fourier_kernel(tr, tr, 2 ** tr.shape[1], p["scale"], c.seed),
                 kn.random_fourier_kernel(tr, te, 2 ** tr.shape[1], p["scale"], c.seed))),
        dict(name="logistic", kind="direct", view="classical", grid=[{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
             make=lambda p: LogisticRegression(p["C"])),
        dict(name="knn", kind="direct", view="classical", grid=[{"k": k} for k in (1, 5, 15)],
             make=lambda p: KNN(p["k"])),
    ]
    for fam in cfg.families:
        for s in cfg.shots:
            q = _quantum(fam, s)
            q["grid"] = [{"bandwidth": b} for b in cfg.bandwidths]
            m.append(q)
    for x in m:
        x.setdefault("quantum", False)
        if x["kind"] == "kernel":
            x["grid"] = [dict(g, C=c) for g in x["grid"] for c in C_GRID]
    return m


def _views(pre: Preprocessor, G, tr, te):
    return {"classical": (pre.classical(G[tr]), pre.classical(G[te])),
            "angles": (pre.angles(G[tr]), pre.angles(G[te]))}


def _fit_predict(model, params, Xtr, ytr, Xte, rng, cfg, keep=None):
    if model["kind"] == "direct":
        return model["make"](params).fit(Xtr, ytr).predict(Xte)
    kp = {k: v for k, v in params.items() if k != "C"}
    Ktr, Kte = model["kernel"](Xtr, Xte, ytr, kp, rng, cfg, keep)
    return PrecomputedSVC(params["C"]).fit(Ktr, ytr).predict(Kte)


# --- run ------------------------------------------------------------------------
def _code_sha256() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def run(G, y, samples, out_dir: Path, cfg: Config, dataset_name: str, log=print) -> Dict:
    out_dir = Path(out_dir); (out_dir / "kernels").mkdir(parents=True, exist_ok=True)
    (out_dir / "circuits").mkdir(exist_ok=True)
    ledger = Ledger(out_dir / "ledger.jsonl")
    classes = np.unique(y)
    models = build_models(cfg)
    data_sha = dataset_sha256(G, y, samples)
    ledger.append("qkernel.preregister", {
        "dataset": dataset_name, "dataset_sha256": data_sha, "n_individuals": int(len(y)),
        "n_variants": int(G.shape[1]), "classes": classes.tolist(), "config": asdict(cfg),
        "models": [m["name"] for m in models], "code_sha256": _code_sha256(),
        "primary_metric": "balanced_accuracy",
        "H0": f"best quantum balanced accuracy - best classical balanced accuracy <= {cfg.margin}",
        "decision_rule": "reject H0 only if the paired-bootstrap 95% CI lower bound exceeds the margin",
        "python": platform.python_version(), "numpy": np.__version__})
    rng = np.random.default_rng(cfg.seed)
    outer = stratified_folds(y, cfg.outer_folds, rng)
    preds = {m["name"]: np.empty(len(y), dtype=classes.dtype) for m in models}
    chosen = {m["name"]: [] for m in models}
    diag = {m["name"]: [] for m in models if m["quantum"]}
    t0 = time.time()
    for f, (tr, te) in enumerate(outer):
        inner = stratified_folds(y[tr], cfg.inner_folds, np.random.default_rng(cfg.seed + 100 + f))
        inner_views = []
        for itr, ite in inner:
            pre = Preprocessor(cfg.n_components).fit(G[tr][itr])
            inner_views.append((_views(pre, G[tr], itr, ite), y[tr][itr], y[tr][ite]))
        pre = Preprocessor(cfg.n_components).fit(G[tr])
        views = _views(pre, G, tr, te)
        if f == 0:
            _export_circuits(views["angles"][0], cfg, out_dir, ledger)
        for m in models:
            scores = []
            for gi, prm in enumerate(m["grid"]):
                s = []
                for k, (v, yi, yo) in enumerate(inner_views):
                    r = np.random.default_rng([cfg.seed, f, k, gi])
                    p = _fit_predict(m, prm, v[m["view"]][0], yi, v[m["view"]][1], r, cfg)
                    s.append(balanced_accuracy(yo, p))
                scores.append(float(np.mean(s)))
            best = m["grid"][int(np.argmax(scores))]
            keep = {} if m["quantum"] else None
            Xtr, Xte = views[m["view"]]
            p = _fit_predict(m, best, Xtr, y[tr], Xte, np.random.default_rng([cfg.seed, f, 999]), cfg, keep)
            preds[m["name"]][te] = p
            chosen[m["name"]].append(best)
            rec = {"fold": f, "model": m["name"], "params": best, "inner_score": max(scores),
                   "test_balanced_accuracy": balanced_accuracy(y[te], p),
                   "train_ids_sha256": hashlib.sha256(",".join(samples[tr]).encode()).hexdigest()}
            if keep:
                d = kn.diagnostics(keep["raw_train"], y[tr], keep["exact_train"] if m["shots"] else None)
                diag[m["name"]].append(d)
                rec["kernel_diagnostics"] = d
                rec["kernel_sha256"] = {k: kn.matrix_sha256(v) for k, v in keep.items()}
                if m["shots"] in cfg.save_kernel_shots:
                    path = out_dir / "kernels" / f"{m['name']}_fold{f}.npz"
                    np.savez_compressed(path, train_ids=samples[tr], test_ids=samples[te], **keep)
                    rec["kernel_file"] = str(path.relative_to(out_dir))
            ledger.append("qkernel.fold", rec)
        log(f"outer fold {f + 1}/{cfg.outer_folds} done ({time.time() - t0:.0f}s)")

    results = _summarise(y, classes, models, preds, chosen, diag, outer, cfg)
    results.update(dataset=dataset_name, dataset_sha256=data_sha, n=int(len(y)), config=asdict(cfg))
    ledger.append("qkernel.result", {"verdict": results["verdict"], "comparison": results["comparison"],
                                     "summary": {k: v["balanced_accuracy"] for k, v in results["models"].items()}})
    results["ledger_ok"] = ledger.verify() is None
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "report.md").write_text(render_report(results))
    return results


def _export_circuits(A, cfg, out_dir, ledger):
    """Fidelity circuits for one training pair per family, as dnalang IR -> OpenQASM 3, with a
    Qiskit cross-check that P(0^n) equals the simulator's kernel entry."""
    from qiskit.quantum_info import Statevector
    for fam in cfg.families:
        t = feature_map(fam, A.shape[1], cfg.reps)
        c = fidelity_circuit(t, A[0], A[1])
        (out_dir / "circuits" / f"fidelity_{fam}.qasm").write_text(c.to_qasm3())
        qc = c.to_qiskit(); qc.remove_final_measurements()
        p0 = float(abs(Statevector(qc).data[0]) ** 2)
        S = simulate(t, A[:2])
        k = float(kn.fidelity_kernel(S[:1], S[1:2])[0, 0])
        ledger.append("qkernel.circuit", {"family": fam, "circuit_sha256": c.sha256(), "depth": c.depth(),
                                          "two_qubit_gates": c.two_qubit_count(), "n_qubits": c.n_qubits,
                                          "qiskit_p0": p0, "simulator_kernel": k,
                                          "agree": abs(p0 - k) < 1e-9})


def _boot_diff(y, pa, pb, B, rng):
    n = len(y); d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = balanced_accuracy(y[i], pa[i]) - balanced_accuracy(y[i], pb[i])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _summarise(y, classes, models, preds, chosen, diag, outer, cfg):
    rng = np.random.default_rng(cfg.seed + 1)
    out = {}
    for m in models:
        p = preds[m["name"]]
        fold_scores = [balanced_accuracy(y[te], p[te]) for _, te in outer]
        r = {"balanced_accuracy": balanced_accuracy(y, p), "macro_f1": macro_f1(y, p),
             "accuracy": float(np.mean(p == y)), "fold_balanced_accuracy": fold_scores,
             "quantum": m["quantum"], "chosen_params": chosen[m["name"]],
             "confusion": confusion(y, p, classes)}
        if m["quantum"]:
            r["family"], r["shots"] = m["family"], m["shots"]
            r["diagnostics_mean"] = {k: float(np.mean([d[k] for d in diag[m["name"]]]))
                                     for k in diag[m["name"]][0]}
            ntr = int(np.mean([len(tr) for tr, _ in outer])); nte = len(y) - ntr
            circ = ntr * (ntr + 1) // 2 + nte * ntr
            r["circuits_per_outer_refit"] = circ
            r["shots_per_outer_refit"] = None if m["shots"] is None else circ * m["shots"]
        out[m["name"]] = r
    q = [k for k, v in out.items() if v["quantum"]]
    c = [k for k, v in out.items() if not v["quantum"]]
    bq = max(q, key=lambda k: out[k]["balanced_accuracy"]) if q else None
    bc = max(c, key=lambda k: out[k]["balanced_accuracy"])
    comp = {"best_classical": bc, "best_classical_ba": out[bc]["balanced_accuracy"]}
    verdict = "no quantum models run"
    if bq:
        lo, hi = _boot_diff(y, preds[bq], preds[bc], cfg.bootstrap, rng)
        comp.update(best_quantum=bq, best_quantum_ba=out[bq]["balanced_accuracy"],
                    diff=out[bq]["balanced_accuracy"] - out[bc]["balanced_accuracy"], ci95=[lo, hi],
                    margin=cfg.margin)
        comp["vs_rbf"] = {k: dict(zip(("lo", "hi"), _boot_diff(y, preds[k], preds["rbf_svm"], cfg.bootstrap, rng)))
                          for k in q if out[k]["shots"] is None}
        verdict = ("H0 rejected: quantum kernel beats the best classical model by more than the margin"
                   if lo > cfg.margin else
                   "H0 not rejected: no evidence that the quantum kernel beats the best classical model")
    return {"models": out, "comparison": comp, "verdict": verdict}


def render_report(r: Dict) -> str:
    c = r["comparison"]
    L = [f"# Quantum-kernel genomics benchmark: {r['dataset']}", "",
         f"- individuals: {r['n']}, dataset sha256 `{r['dataset_sha256'][:16]}…`",
         f"- features: {r['config']['n_components']} PCs (fitted on training individuals only) → "
         f"{r['config']['n_components']} qubits, {r['config']['reps']} feature-map repetitions",
         f"- outer/inner CV: {r['config']['outer_folds']}/{r['config']['inner_folds']} folds; "
         f"depolarizing model p = {r['config']['depolarizing']}",
         f"- ledger chain intact: {r['ledger_ok']}", "",
         f"## Verdict", "", f"**{r['verdict']}**", ""]
    if "best_quantum" in c:
        L += [f"Best quantum: `{c['best_quantum']}` {c['best_quantum_ba']:.3f}; best classical: "
              f"`{c['best_classical']}` {c['best_classical_ba']:.3f}; difference {c['diff']:+.3f}, "
              f"paired bootstrap 95% CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], margin {c['margin']}.", "",
              "The best quantum model is picked after seeing test results, which favours it; the best "
              "classical model is picked the same way, so the comparison is symmetric in that respect.", ""]
    L += ["## All models (pooled outer-fold predictions)", "",
          "| model | balanced acc | macro-F1 | fold range | shots | circuits / refit | off-diag mean ± sd | eff. rank | KTA |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(r["models"].items(), key=lambda kv: -kv[1]["balanced_accuracy"]):
        fs = v["fold_balanced_accuracy"]
        if v["quantum"]:
            d = v["diagnostics_mean"]
            extra = (f"{'exact' if v['shots'] is None else v['shots']} | {v['circuits_per_outer_refit']:,} | "
                     f"{d['offdiag_mean']:.3f} ± {d['offdiag_std']:.3f} | {d['effective_rank']:.1f} | "
                     f"{d['kernel_target_alignment']:.3f}")
        else:
            extra = "– | – | – | – | –"
        L.append(f"| `{k}` | {v['balanced_accuracy']:.3f} | {v['macro_f1']:.3f} | {min(fs):.2f}–{max(fs):.2f} | {extra} |")
    if c.get("vs_rbf"):
        L += ["", "## Exact quantum kernels vs RBF SVM (paired bootstrap 95% CI of the balanced-accuracy difference)", ""]
        for k, v in c["vs_rbf"].items():
            L.append(f"- `{k}`: [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    L += ["", "## How to read this", "",
          "- `q_product_*` has no entangling gates. If it matches `q_nn_*`/`q_zz_*`, entanglement added nothing here.",
          "- `rff_svm_control` is a classical random nonlinear map of the same dimension (2^n) on the same angle inputs.",
          "- Finite-shot kernels: each entry is a binomial estimate from that many shots; training matrices are "
          "PSD-repaired by eigenvalue clipping, and the raw matrices and their diagnostics are kept.",
          "- A small off-diagonal sd means kernel concentration: entries become hard to tell apart with finite shots.",
          "- Hardware was not used. `circuits/` holds the OpenQASM 3 fidelity circuits that a hardware run would execute.",
          "- Accuracy on this task says nothing about clinical use or about the biology behind the labels.", ""]
    return "\n".join(L)
