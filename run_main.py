"""Run the main pipeline at L = 60.

    python run_main.py                                   # text layer only
    python run_main.py --crsp data/crsp_daily_2014_2024.parquet   # plus pricing

Steps (each skipped when its outputs exist, so an interrupted run resumes):

  1. download FNSPID                     scripts/download_fnspid.py
  2. build the corpus                    scripts/prepare_corpus.py
  3. vocabulary, counts, passages        scripts/build_vocab_and_chunks.py
  4. fit LDA                             scripts/fit_lda.py
  5. embed passages                      scripts/embed_chunks.py
  6. fit the three transformer variants  scripts/fit_transformer_topics.py
  7. coherence and head comparison       scripts/coherence.py, compare_topic_models.py
  8. pricing (needs --crsp)              scripts/price_single_horizon.py,
                                         scripts/price_multi_horizon.py
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SCRIPTS, DATA, RESULTS = ROOT / "scripts", ROOT / "data", ROOT / "results"
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
FF_ZIP = DATA / "external" / "F-F_Research_Data_Factors_daily_CSV.zip"


def run(args, name):
    print(f"\n[run ] {name}", flush=True)
    t0 = time.time()
    code = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT).returncode
    print(f"[done] {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if code != 0:
        sys.exit(f"{name} failed (exit {code}); rerun to resume")


def fitted(variant, **expected):
    meta = RESULTS / f"meta_{variant}.json"
    if not meta.exists() or not (RESULTS / f"theta_daily_{variant}.parquet").exists():
        return False
    values = json.loads(meta.read_text())
    return all(values.get(k) == v for k, v in expected.items())


def step(name, done, args):
    if done():
        print(f"[skip] {name}")
    else:
        run(args, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crsp", type=Path, help="CRSP daily parquet from scripts/wrds_pull.py")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda for clustering")
    args = parser.parse_args()
    stored = (DATA / "vocab.json").exists() or any(RESULTS.glob("theta_daily_*.parquet"))
    if stored and not (DATA / "counts.npz").exists():
        sys.exit("results/ or data/vocab.json holds stored fits but data/counts.npz is "
                 "missing: step 3 would rebuild the vocabulary and later steps would mix "
                 "stored and new files. For a fresh run move results/ and data/vocab.json "
                 "aside first (README, 'Running the main model').")
    for d in (DATA, RESULTS, DATA / "external"):
        d.mkdir(parents=True, exist_ok=True)

    step("download FNSPID",
         lambda: (status := list((DATA / "fnspid_status").glob("worker*.json"))) and all(
             json.loads(f.read_text()).get("done") for f in status),
         [SCRIPTS / "download_fnspid.py"])
    step("build corpus", lambda: (DATA / "corpus.parquet").exists(),
         [SCRIPTS / "prepare_corpus.py"])
    step("vocabulary, counts and passages",
         lambda: all((DATA / f).exists() for f in ("counts.npz", "vocab.json", "doc_meta.parquet", "chunks.parquet")),
         [SCRIPTS / "build_vocab_and_chunks.py"])
    step("LDA", lambda: fitted("lda"), [SCRIPTS / "fit_lda.py", "--variant", "lda"])
    step("passage embeddings", lambda: (DATA / "chunk_embeddings.npz").exists(),
         [SCRIPTS / "embed_chunks.py"])
    fit = SCRIPTS / "fit_transformer_topics.py"
    step("frozen ST (mini-batch k-means, hard c-TF-IDF)",
         lambda: fitted("st_frozen", clusterer="minibatch_kmeans", term_weighting="hard"),
         [fit, "--variant", "st_frozen", "--clusterer", "minibatch_kmeans",
          "--term-weighting", "hard", "--device", args.device])
    step("spherical ST, soft head",
         lambda: fitted("st_spherical_soft", clusterer="spherical", term_weighting="soft"),
         [fit, "--variant", "st_spherical_soft", "--clusterer", "spherical",
          "--term-weighting", "soft", "--device", args.device])
    step("spherical ST, sparse-soft head",
         lambda: fitted("st_spherical", clusterer="spherical", term_weighting="sparse_soft"),
         [fit, "--variant", "st_spherical", "--clusterer", "spherical",
          "--term-weighting", "sparse_soft", "--device", args.device,
          "--centroids-path", RESULTS / "centroids_st_spherical_soft.npy"])
    run([SCRIPTS / "coherence.py"], "coherence")
    run([SCRIPTS / "compare_topic_models.py", "st_spherical_soft", "st_spherical",
         "--require-same-attention"], "term-head comparison")

    if args.crsp is None:
        print("\ntext layer complete; pass --crsp to run the pricing stages")
        return
    if not FF_ZIP.exists():
        print(f"downloading {FF_URL}")
        urllib.request.urlretrieve(FF_URL, FF_ZIP)
    price1, price3 = SCRIPTS / "price_single_horizon.py", SCRIPTS / "price_multi_horizon.py"
    step("LDA, single horizon, total returns",
         lambda: (RESULTS / "summary_lda_total.json").exists(),
         [price1, "lda", args.crsp, "3"])
    step("frozen ST, single horizon, total returns",
         lambda: (RESULTS / "summary_st_frozen_total.json").exists(),
         [price1, "st_frozen", args.crsp, "3"])
    step("frozen ST, multi-horizon, excess returns",
         lambda: (RESULTS / "multihorizon_st_frozen" / "summary.json").exists(),
         [price3, "st_frozen", args.crsp, FF_ZIP, RESULTS / "multihorizon_st_frozen"])
    step("spherical ST, multi-horizon, excess returns",
         lambda: (RESULTS / "multihorizon_st_spherical" / "summary.json").exists(),
         [price3, "st_spherical", args.crsp, FF_ZIP, RESULTS / "multihorizon_st_spherical"])
    print("\nall steps complete; see results/")


if __name__ == "__main__":
    main()
