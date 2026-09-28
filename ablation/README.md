# Ablation: number of topics (paper Appendix B)

Both text layers were refit at L = 50, 60, 70 and 80 on the corpus,
vocabulary and passage embeddings produced by steps 1-5 of the pipeline as
shipped, and the L = 80 fits were carried through the pricing stages.
Everything downstream of the text layer is unchanged: same encoder, seed,
Gibbs schedule, vocabulary rule, return panel, K = 3, penalty grids and
evaluation window.

## Running

```
python run_main.py                       # steps 1-5 must have run (corpus, counts, embeddings)
python ablation/sweep_topics.py          # L from config.SWEEP_L (50,60,70,80)
bash ablation/price_L80.sh data/crsp_daily_2014_2024.parquet
```

`sweep_topics.py` fits `lda_L{L}`, `st_spherical_soft_L{L}` and
`st_spherical_L{L}` (sparse-soft head on the same centroids) for every L,
scores NPMI with `scripts/coherence.py` (appended to
`results/coherence.json`), and writes
`ablation/results/sweep_summary.{json,csv,md}`. Weights, labels and
metadata are written to `results/` with the `_L{L}` suffix. On a T4 machine
each LDA fit takes about an hour and each transformer fit about two minutes.

`price_L80.sh` runs the two L = 80 text layers through the single-horizon
model (total and excess returns) and the grouped multi-horizon model,
collects the outputs in `ablation/results/pricing_L80/` and writes
`pricing_L80_summary.csv` with `collect_pricing_L80.py`. Runs whose summary
already exists are skipped. To check the stored fits and pricing outputs
without running anything, use `python tests/check_results.py` from the
package root.

## Included files

All twelve sweep fits (attention, topic-term weights, centroids, labels
and metadata, with the `_L{L}` suffix) are in `results/` next to the main
fits, their per-topic NPMI is in the shared coherence file, and the six
L = 80 pricing runs (selection, OOS returns, summaries, multi-horizon
diagnostics) are in `ablation/results/pricing_L80/`, with
`pricing_L80_summary.csv` collecting them into the table behind Table B.2.
The spherical clustering at L = 80 stopped at the 25-iteration cap; the
fits at L = 50, 60 and 70 converged in 23 iterations.

The sweep was run on a rebuilt corpus, so its vocabulary differs slightly
from that of the main runs: the `_L{L}` topic-term files index into
`data/vocab_ablation.json` and the main-run files into `data/vocab.json`.

## Results

Coherence (`ablation/results/sweep_summary.md`; per-topic NPMI in
`results/coherence.json`):

| arm | L | mean NPMI | median NPMI | diversity |
|---|---:|---:|---:|---:|
| LDA | 50 | 0.2867 | 0.2847 | 0.736 |
| LDA | 60 | 0.2683 | 0.2436 | 0.683 |
| LDA | 70 | 0.2910 | 0.2883 | 0.639 |
| LDA | 80 | 0.2824 | 0.2572 | 0.606 |
| Spherical ST, soft head | 50 | 0.3440 | 0.3181 | 0.448 |
| Spherical ST, soft head | 60 | 0.3890 | 0.3484 | 0.468 |
| Spherical ST, soft head | 70 | 0.3791 | 0.3397 | 0.449 |
| Spherical ST, soft head | 80 | 0.4009 | 0.3751 | 0.470 |
| Spherical ST, sparse-soft head | 50 | 0.4414 | 0.4591 | 0.700 |
| Spherical ST, sparse-soft head | 60 | 0.4579 | 0.5239 | 0.678 |
| Spherical ST, sparse-soft head | 70 | 0.4622 | 0.4545 | 0.669 |
| Spherical ST, sparse-soft head | 80 | 0.4943 | 0.4781 | 0.666 |

The L = 60 rows above are the sweep's own refits. Table B.1 in the paper
shows the main-text L = 60 fits in that row (0.2875, 0.3702 and 0.4691),
which were estimated separately; the other rows are identical.

Pricing at L = 80 (`ablation/results/pricing_L80/`; OOS September 2020 to
January 2024, 41 months, K = 3):

| specification | returns | OOS Sharpe | active topics (final refit) |
|---|---|---:|---:|
| Spherical ST, sparse-soft, multi-horizon | excess | 0.7156 | 80 |
| LDA, multi-horizon | excess | 0.6125 | 80 |
| Spherical ST, sparse-soft, single horizon | total | 0.7219 | 80 |
| LDA, single horizon | total | 0.7675 | 78 |
| Spherical ST, sparse-soft, single horizon | excess | 0.6412 | 80 |
| LDA, single horizon | excess | 0.6470 | 78 |
| Equal-weight market, same universe | total | 0.8166 | |
| Equal-weight market, same universe | excess | 0.7164 | |

The per-refit penalties, active counts and in-sample Sharpe ratios of the
multi-horizon runs are in `pricing_L80/multihorizon_*/diagnostics.json`.
