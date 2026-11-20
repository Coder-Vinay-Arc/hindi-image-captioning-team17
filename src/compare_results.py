"""
compare_results.py -- print a Markdown comparison table from the score files written by evaluate.py.

Reads results/<name>_<split>_scores.json for each name and prints one table row per approach, so numbers are
copied from the files and never typed by hand.

Run (from the project root):
    python src/compare_results.py                                   # baseline vs transformer, test split
    python src/compare_results.py --names baseline transformer translation
"""

import argparse
import json
import os

LABELS = {"baseline": "CNN + LSTM + Attention (baseline)", "transformer": "ResNet-50 + Transformer decoder",
          "translation": "English captioning + translation"}
METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["baseline", "transformer"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    print("| Approach | " + " | ".join(METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    rows = {}
    for name in args.names:
        path = os.path.join(args.results_dir, f"{name}_{args.split}_scores.json")
        if not os.path.exists(path):
            print(f"| {LABELS.get(name, name)} | " + " | ".join("n/a" for _ in METRICS) + " |")
            continue
        with open(path, encoding="utf-8") as f:
            rows[name] = json.load(f)["scores"]
        print(f"| {LABELS.get(name, name)} | " + " | ".join(f"{rows[name][m]:.2f}" for m in METRICS) + " |")

    if len(rows) == 2:
        a, b = list(rows)
        print(f"\nChange {b} vs {a}: " + ", ".join(f"{m} {rows[b][m] - rows[a][m]:+.2f}" for m in METRICS))


if __name__ == "__main__":
    main()
