"""Pipeline settings. The defaults are the values used for the paper.

Environment variables override the settings that the ablation changes
(TOPIC_L, LDA_GIBBS_ITERS, SWEEP_L); everything else is edited here.
"""
import os

# Corpus (scripts/download_fnspid.py, scripts/prepare_corpus.py)
DATE_MIN = "2015-01-01"
DATE_MAX = "2023-12-31"
N_WORKERS = 6                 # parallel HTTP range downloads
MAX_PER_DAY = 120             # articles kept per calendar day
MIN_CHARS = 400               # drop shorter article bodies
VOCAB_SIZE = 15000            # shared unigram + bigram vocabulary
MAX_CHUNKS_PER_DOC = 3        # leading passages embedded per article

# Topic models (both branches)
L = int(os.getenv("TOPIC_L", 60))       # number of topics
SEED = 0

# LDA branch (scripts/fit_lda.py)
LDA_BACKEND = "tomotopy"      # collapsed Gibbs; "sklearn" = online variational Bayes
LDA_N_JOBS = -1               # sklearn backend only
LDA_GIBBS_ITERS = int(os.getenv("LDA_GIBBS_ITERS", 800))

# Transformer branch (scripts/embed_chunks.py, scripts/fit_transformer_topics.py)
ENCODER = "sentence-transformers/paraphrase-MiniLM-L3-v2"
TOPIC_TEMPERATURE = 0.07      # softmax temperature for passage-to-topic weights
TOPIC_CLUSTERER = "spherical"           # or "minibatch_kmeans" (frozen ST baseline)
TOPIC_TERM_WEIGHTING = "sparse_soft"    # "soft", "hard", or "sparse_soft"
TOPIC_CLUSTER_BATCH_SIZE = 16384        # rows per streamed block (memory only)
TOPIC_CLUSTER_MAX_ITER = 25
TOPIC_CLUSTER_N_INIT = 10
TOPIC_CLUSTER_INIT_SIZE = 100000        # rows used for k-means++ initialisation
TOPIC_CLUSTER_TOL = 1e-5
TOPIC_CLUSTER_DEVICE = os.getenv("TOPIC_CLUSTER_DEVICE", "auto")  # auto, cpu, cuda
TOPIC_ASSIGNMENT_BATCH_SIZE = 65536

# Sparse-soft c-TF-IDF head (paper Eq. 6). Affects topic term lists only,
# not the daily attention series.
TOPIC_CTFIDF_TOP_K = 2
TOPIC_CTFIDF_POWER = 2.0
TOPIC_CTFIDF_CONFIDENCE_POWER = 1.0
TOPIC_CTFIDF_CONFIDENCE_QUANTILE = 0.5
TOPIC_CTFIDF_SUBLINEAR_TF = True

# Variants scored by scripts/coherence.py
COHERENCE_VARIANTS = ("lda", "st_frozen", "st_spherical_soft", "st_spherical")

# Topic-count ablation (ablation/sweep_topics.py)
SWEEP_L = tuple(int(x) for x in os.getenv("SWEEP_L", "50,60,70,80").split(","))
