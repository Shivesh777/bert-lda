"""Collect the six L = 80 pricing summaries into one table
(ablation/results/pricing_L80/pricing_L80_summary.csv, the source of Table B.2).

    python ablation/collect_pricing_L80.py
"""
from pathlib import Path
import csv
import json

OUT = Path(__file__).resolve().parent / "results" / "pricing_L80"
VARIANTS = ("st_spherical_L80", "lda_L80")
FIELDS = ["variant", "pricing_stage", "returns", "K", "oos_sharpe_mve",
          "oos_sharpe_ew_market", "is_sharpe", "n_active_final", "oos_months"]


def main():
    rows = []
    for kind in ("total", "excess"):
        for variant in VARIANTS:
            s = json.load(open(OUT / f"summary_{variant}_{kind}.json"))
            rows.append({"variant": variant, "pricing_stage": "single-horizon sparse IPCA",
                         "returns": kind, "K": s["K"],
                         "oos_sharpe_mve": round(s["oos_sharpe_nf"], 4),
                         "oos_sharpe_ew_market": round(s["oos_sharpe_ewmkt"], 4),
                         "is_sharpe": round(s["is_sharpe"], 4),
                         "n_active_final": s["n_active"], "oos_months": s["oos_months"]})
    for variant in VARIANTS:
        s = json.load(open(OUT / f"multihorizon_{variant}" / "summary.json"))
        rows.append({"variant": variant,
                     "pricing_stage": "grouped multi-horizon IPCA (20/60/120)",
                     "returns": s["return_type"], "K": s["K"],
                     "oos_sharpe_mve": round(s["oos_sharpe_nf"], 4),
                     "oos_sharpe_ew_market": "", "is_sharpe": "", "n_active_final": "",
                     "oos_months": s["oos_months"]})
    with open(OUT / "pricing_L80_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['variant']:18s} {r['pricing_stage']:40s} {r['returns']:7s} "
              f"OOS Sharpe {r['oos_sharpe_mve']:.4f}")


if __name__ == "__main__":
    main()
