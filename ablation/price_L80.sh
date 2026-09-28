#!/bin/bash
# Pricing stages for the L = 80 topic fits (paper Table B.2).
# Run after ablation/sweep_topics.py --L 80 (the script changes to the
# repository root itself; give the CRSP path relative to that root or as an
# absolute path). Logs go to logs/.
#
#   bash ablation/price_L80.sh data/crsp_daily_2014_2024.parquet
#
# Six runs: both text layers through the single-horizon model on total and
# on excess returns, and through the grouped multi-horizon model. Each
# single-horizon run takes about 45 minutes on four cores and each
# multi-horizon run about two hours. A run whose summary already exists in
# ablation/results/pricing_L80/ is skipped. The per-run summaries are
# collected into pricing_L80_summary.csv at the end.
set -e
cd "$(dirname "$0")/.."
CRSP=${1:?path to crsp_daily_2014_2024.parquet}
FF=data/external/F-F_Research_Data_Factors_daily_CSV.zip
OUT=ablation/results/pricing_L80
mkdir -p "$OUT" logs data/external
if [ ! -f "$FF" ]; then
  python -c "import urllib.request; urllib.request.urlretrieve(
    'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip', '$FF')"
fi

for v in st_spherical_L80 lda_L80; do
  for kind in total excess; do
    if [ -f "$OUT/summary_${v}_${kind}.json" ]; then
      echo "skip $v $kind"; continue
    fi
    if [ "$kind" = total ]; then rf=""; else rf="$FF"; fi
    python -u scripts/price_single_horizon.py $v "$CRSP" 3 $rf \
        > logs/price_${v}_${kind}.log 2>&1
    mv results/summary_${v}_${kind}.json results/oos_${v}_${kind}.parquet \
       results/selection_${v}_${kind}.csv "$OUT"/
  done
done

for v in st_spherical_L80 lda_L80; do
  if [ -f "$OUT/multihorizon_$v/summary.json" ]; then
    echo "skip $v multi-horizon"; continue
  fi
  python -u scripts/price_multi_horizon.py $v "$CRSP" "$FF" "$OUT/multihorizon_$v" \
      > logs/multihorizon_${v}.log 2>&1
done

python ablation/collect_pricing_L80.py
echo "done: $OUT"
