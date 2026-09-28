"""Fit the LDA branch on the shared count matrix.

Backends (config.LDA_BACKEND):
  tomotopy  collapsed Gibbs sampling, LDA_GIBBS_ITERS sweeps, all cores.
            The sampler state is checkpointed every 100 sweeps so an
            interrupted fit resumes. This is the backend used in the paper.
  sklearn   online variational Bayes; slower, kept as a fallback.

Outputs in results/ (VARIANT defaults to "lda"):
  theta_daily_{VARIANT}.parquet   daily topic attention, rows sum to one
  phi_{VARIANT}.npy               topic-term distributions over the vocabulary
  labels_{VARIANT}.json           three highest-weighted terms per topic
  meta_{VARIANT}.json             settings and log likelihood per word

Usage: python scripts/fit_lda.py [--variant NAME] [--drop-checkpoint]
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

import config
from src.interfaces import AttentionModel
from src.stage1_lda import auto_label
from src.textprep import aggregate_daily

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def fit_sklearn(counts, L):
    from sklearn.decomposition import LatentDirichletAllocation
    lda = LatentDirichletAllocation(
        n_components=L, random_state=config.SEED, max_iter=12,
        learning_method="online", batch_size=8192, n_jobs=config.LDA_N_JOBS,
        evaluate_every=-1, verbose=1)
    doc_theta = lda.fit_transform(counts)
    doc_theta = doc_theta / doc_theta.sum(axis=1, keepdims=True)
    phi = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    return doc_theta, phi, None


def fit_tomotopy(counts, vocab, L, variant, drop_checkpoint, t0):
    import tomotopy as tp
    n_docs = counts.shape[0]
    empty = [i for i in range(n_docs) if counts.indptr[i] == counts.indptr[i + 1]]
    suffix = "" if variant == "lda" else f"_{variant}"
    ckpt = DATA / f"lda_gibbs{suffix}.ckpt"
    ckpt_meta = DATA / f"lda_gibbs{suffix}_meta.json"
    fingerprint = {"n_docs": int(n_docs), "L": int(L), "seed": int(config.SEED),
                   "V": len(vocab)}

    model, iters_done = None, 0
    if ckpt.exists() and ckpt_meta.exists():
        saved = json.loads(ckpt_meta.read_text())
        if all(saved.get(k) == v for k, v in fingerprint.items()):
            model = tp.LDAModel.load(str(ckpt))
            iters_done = int(saved.get("iters_done", 0))
            print(f"  resumed checkpoint at sweep {iters_done} [{time.time() - t0:.0f}s]")
        else:
            print("  checkpoint belongs to a different corpus or setting; refitting")
    if model is None:
        model = tp.LDAModel(k=L, seed=config.SEED)
        for i in range(n_docs):
            row = counts.getrow(i)
            tokens = [vocab[j] for j, c in zip(row.indices, row.data)
                      for _ in range(int(c))]
            if tokens:
                model.add_doc(tokens)
        print(f"  documents added ({len(empty)} empty) [{time.time() - t0:.0f}s]")
        model.train(0)

    step = 100
    for it in range(iters_done, config.LDA_GIBBS_ITERS, step):
        n_it = min(step, config.LDA_GIBBS_ITERS - it)
        model.train(n_it, workers=0)
        model.save(str(ckpt) + ".tmp", True)
        os.replace(str(ckpt) + ".tmp", ckpt)
        ckpt_meta.write_text(json.dumps({**fingerprint, "iters_done": it + n_it}))
        print(f"  sweep {it + n_it}/{config.LDA_GIBBS_ITERS} "
              f"ll/word={model.ll_per_word:.3f} [{time.time() - t0:.0f}s]", flush=True)

    # Map tomotopy's internal vocabulary order back onto the shared vocabulary.
    position = {w: j for j, w in enumerate(vocab)}
    used = list(model.used_vocabs)
    phi = np.full((L, len(vocab)), 1e-12)
    for l in range(L):
        for w, p in zip(used, model.get_topic_word_dist(l)):
            phi[l, position[w]] = p
    phi = phi / phi.sum(axis=1, keepdims=True)

    doc_theta = np.full((n_docs, L), 1.0 / L)
    empty_set = set(empty)
    nonempty = (i for i in range(n_docs) if i not in empty_set)
    filled = 0
    for i, d in zip(nonempty, model.docs):   # docs iterate in add_doc order
        doc_theta[i] = d.get_topic_dist()
        filled += 1
    assert filled == n_docs - len(empty), "document read-back count mismatch"
    doc_theta = doc_theta / doc_theta.sum(axis=1, keepdims=True)

    if drop_checkpoint:
        for path in (ckpt, ckpt_meta):
            if path.exists():
                path.unlink()
    return doc_theta, phi, float(model.ll_per_word)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="lda", help="output file suffix")
    parser.add_argument("--drop-checkpoint", action="store_true",
                        help="delete the Gibbs sampler state after saving")
    args = parser.parse_args()

    t0 = time.time()
    counts = sparse.load_npz(DATA / "counts.npz").tocsr()
    vocab = json.load(open(DATA / "vocab.json"))
    meta = pq.read_table(DATA / "doc_meta.parquet").to_pandas()
    L = config.L
    print(f"counts {counts.shape}, L={L}, backend={config.LDA_BACKEND}")

    if config.LDA_BACKEND == "tomotopy":
        doc_theta, phi, ll = fit_tomotopy(counts, vocab, L, args.variant,
                                          args.drop_checkpoint, t0)
    else:
        doc_theta, phi, ll = fit_sklearn(counts, L)
    print(f"fit done [{time.time() - t0:.0f}s]")

    theta_daily = aggregate_daily(doc_theta, meta["date"], meta["n_tokens"].to_numpy())
    model = AttentionModel(
        theta_daily=theta_daily, phi=phi, vocab=vocab,
        labels=auto_label(phi, vocab),
        meta={"variant": args.variant, "L": L, "seed": config.SEED,
              "backend": config.LDA_BACKEND, "gibbs_iters": config.LDA_GIBBS_ITERS,
              "ll_per_word": ll})
    model.validate()

    RESULTS.mkdir(exist_ok=True)
    theta = model.theta_daily.copy()
    theta.columns = [str(c) for c in theta.columns]
    theta.reset_index(names="date").to_parquet(RESULTS / f"theta_daily_{args.variant}.parquet")
    np.save(RESULTS / f"phi_{args.variant}.npy", model.phi)
    json.dump(model.labels, open(RESULTS / f"labels_{args.variant}.json", "w"), indent=1)
    json.dump(model.meta, open(RESULTS / f"meta_{args.variant}.json", "w"), indent=1)
    print(f"saved results/*_{args.variant}.* [{time.time() - t0:.0f}s]")
    for l in range(0, L, 8):
        print(f"  {l:3d}: {model.labels[l]}")


if __name__ == "__main__":
    main()
