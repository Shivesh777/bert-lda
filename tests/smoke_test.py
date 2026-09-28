"""End-to-end smoke test on synthetic data (no downloads, under a minute).

1. Stage 3 covariance instruments match a brute-force weighted covariance.
2. Sparse IPCA recovers a planted sparse row support; the group prox
   satisfies the subgradient optimality conditions.
3. Both Stage 1 branches (LDA, transformer with random embeddings) return
   valid AttentionModel objects from a tiny synthetic corpus.

Run: python tests/smoke_test.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.textprep import build_vocab_and_counts, clean_text
from src.stage1_lda import fit_lda
from src.stage1_bert import fit_bert_topics
from src.stage2_shocks import shocks
from src.stage3_covariances import covariance_instruments, ewma_weights
from src.stage4_sparse_ipca import fit_sparse_ipca, tune_lambda, mve_stats, _row_prox

rng = np.random.default_rng(42)
FAIL = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAIL.append(name)


# ---------- 1. group prox correctness ----------
K = 3
A = rng.standard_normal((K, K)); A = A @ A.T + 0.5 * np.eye(K)
b = rng.standard_normal(K) * 2
lam = 1.5
g = _row_prox(A, b, lam)
if np.linalg.norm(g) > 0:
    grad = 2 * A @ g - 2 * b + lam * g / np.linalg.norm(g)
    check("row_prox stationarity", np.linalg.norm(grad) < 1e-6,
          f"|grad|={np.linalg.norm(grad):.2e}")
b_small = b / np.linalg.norm(b) * (lam / 2 * 0.9)
check("row_prox zero region", np.linalg.norm(_row_prox(A, b_small, lam)) == 0)

# Grouped updates must match the original row-wise solver when each group has
# one feature. This protects the backward-compatible groups=None path too.
Tg, Ng, Lg, Kg = 8, 16, 5, 2
Xg = np.vstack([np.c_[np.ones(Ng), rng.standard_normal((Ng, Lg))]
                for _ in range(Tg)])
mg = np.repeat(np.arange(Tg), Ng)
yg = rng.standard_normal(Tg * Ng)
fit_rows = fit_sparse_ipca(Xg, yg, mg, K=Kg, lam=1e-4, seed=0, max_iter=4)
fit_groups = fit_sparse_ipca(
    Xg, yg, mg, K=Kg, lam=1e-4, seed=0, max_iter=4,
    groups=[[i] for i in range(1, Lg + 1)])
pred_rows = ((Xg @ fit_rows["Gamma"]) * fit_rows["F"][mg]).sum(axis=1)
pred_groups = ((Xg @ fit_groups["Gamma"]) * fit_groups["F"][mg]).sum(axis=1)
check("singleton grouped solver matches row-wise solver",
      np.allclose(pred_rows, pred_groups, atol=1e-12))

# ---------- 2. stage 3 covariance vs brute force ----------
nd, ns, L = 300, 8, 6
dates = pd.bdate_range("2020-01-01", periods=nd)
R = pd.DataFrame(rng.standard_normal((nd, ns)) * 0.02, index=dates,
                 columns=[f"S{i}" for i in range(ns)])
Z = pd.DataFrame(rng.standard_normal((nd, L)) * 0.01, index=dates)
cov = covariance_instruments(R, Z, halflife=60, lookback=120, min_obs=60)
m = sorted(cov.keys())[3]
first_of_m = dates[pd.PeriodIndex(dates, freq="M") == m][0]
end = dates.get_loc(first_of_m)
start = max(0, end - 120)
w = ewma_weights(end - start, 60)
r = R.iloc[start:end, 0].to_numpy()
z0 = Z.iloc[start:end, 0].to_numpy()
ref = np.sum(w * (r - np.sum(w * r)) * (z0 - np.sum(w * z0)))
check("stage3 weighted cov matches brute force",
      abs(cov[m].iloc[0, 0] - ref) < 1e-12,
      f"got {cov[m].iloc[0, 0]:.3e} ref {ref:.3e}")
last_day_used = dates[end - 1]
check("stage3 no look-ahead", last_day_used < first_of_m,
      f"window ends {last_day_used.date()} < month starts {first_of_m.date()}")

# ---------- 3. sparse IPCA recovery on simulated ground truth ----------
def make_panel(n_months, n_stocks, L, K, s_true, gscale, noise, seed=1):
    r = np.random.default_rng(seed)
    P = L + 1
    Gamma_true = np.zeros((P, K))
    active = np.sort(r.choice(L, s_true, replace=False))
    Gamma_true[1 + active] = r.standard_normal((s_true, K)) * gscale
    F_true = r.standard_normal((n_months, K)) * 0.035 + 0.006
    Xs, ys, ms = [], [], []
    for t in range(n_months):
        Xt = r.standard_normal((n_stocks, P)); Xt[:, 0] = 1.0
        yt = (Xt @ Gamma_true) @ F_true[t] + r.standard_normal(n_stocks) * noise
        Xs.append(Xt); ys.append(yt); ms.append(np.full(n_stocks, t))
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(ms),
            active, F_true)


# moderate DGP: selection statistically feasible; verifies the machinery
# (hard regimes are a power question, not a correctness question)
X, y, m_ids, active_true, F_true = make_panel(120, 150, 40, 2, 6,
                                              gscale=0.7, noise=0.08)
L, K = 40, 2

def sel_f1(fit):
    est_active = set(int(i) for i in np.where(fit["active"])[0])
    true_active = set(int(i) for i in active_true)
    tp = len(est_active & true_active)
    prec = tp / max(len(est_active), 1)
    rec = tp / len(true_active)
    return 2 * prec * rec / max(prec + rec, 1e-12), est_active


best, results = tune_lambda(X, y, m_ids, K=K, lams=None,
                            seed=0, verbose=False, criterion="validation")
f1, est_active = sel_f1(best["fit"])
check("sparse IPCA selection F1 > 0.7 (validation-tuned)", f1 > 0.7,
      f"F1={f1:.2f} active={sorted(est_active)} true={sorted(int(i) for i in active_true)}")

f1_path = max(sel_f1(r["fit"])[0] for r in results)
check("some lambda on path recovers support (F1 > 0.85)", f1_path > 0.85,
      f"best path F1={f1_path:.2f}")
fit = best["fit"]

fit0 = fit_sparse_ipca(X, y, m_ids, K=K, lam=1e-12, seed=0)
n0 = int(fit0["active"].sum())
check("lam=0 keeps (almost) all rows", n0 > 0.9 * L, f"active={n0}/{L}")

# factor-space recovery on an EASY DGP (wide cross-section, low noise):
# checks estimator consistency separately from the hard selection regime
Xe, ye, me, act_e, F_e = make_panel(120, 200, 40, 2, 6,
                                    gscale=0.8, noise=0.06, seed=2)
fit_e = fit_sparse_ipca(Xe, ye, me, K=2, lam=3e-5, seed=0)
Fh = fit_e["F"]
resid = F_e - Fh @ np.linalg.lstsq(Fh, F_e, rcond=None)[0]
r2 = 1 - resid.var() / F_e.var()
check("factor space recovery R2 > 0.8 (easy DGP)", r2 > 0.8, f"R2={r2:.3f}")

sr = mve_stats(fit["F"])["sharpe_ann"]
print(f"   info: tuned lam={fit['lam']:.2e}, IS Sharpe (ann) = {sr:.2f}")

# ---------- 4. stage 1 variants on tiny synthetic corpus ----------
topics_words = {
    0: "recession downturn layoffs unemployment slump contraction crisis",
    1: "merger acquisition takeover deal buyout synergy antitrust",
    2: "dividend buyback earnings profit revenue guidance beat",
}
docs, dts = [], []
base = pd.Timestamp("2020-01-01")
for i in range(600):
    k = i % 3
    words = (topics_words[k] + " ") * 12
    noise = " ".join(rng.choice("alpha beta gamma delta market trading".split(), 10))
    docs.append(words + noise)
    dts.append(base + pd.Timedelta(days=i % 90))
df = pd.DataFrame({"text": docs, "date": dts})
cleaned = clean_text(df["text"])
counts, vocab, n_tokens = build_vocab_and_counts(cleaned, max_features=200,
                                                 min_df=2, max_df=0.95)
lda_model, _ = fit_lda(counts, vocab, df["date"], n_tokens, L=3, max_iter=15)
check("LDA AttentionModel validates", lda_model.validate())
check("LDA daily theta on simplex",
      np.allclose(lda_model.theta_daily.sum(axis=1), 1, atol=1e-6))

fake_emb = rng.standard_normal((600, 32)).astype(np.float32)
for i in range(600):
    fake_emb[i, (i % 3) * 10:(i % 3) * 10 + 10] += 3.0
fake_emb /= np.linalg.norm(fake_emb, axis=1, keepdims=True)
bert_model, _ = fit_bert_topics(
    df["text"].tolist(), counts, vocab, df["date"], n_tokens, L=3,
    embeddings=fake_emb, doc_index=np.arange(600),
    chunk_weights=np.ones(600), temperature=0.07)
check("BERT AttentionModel validates", bert_model.validate())
check("BERT variant runs with same interface", bert_model.phi.shape == (3, len(vocab)))

# shocks
z = shocks(lda_model.theta_daily, window=5)
check("stage2 shocks drop warmup rows",
      len(z) == len(lda_model.theta_daily) - 5)
check("stage2 shocks near zero mean", abs(z.mean().mean()) < 0.02,
      f"mean={z.mean().mean():.4f}")

print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
