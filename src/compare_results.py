"""
compare_results.py -- print a Markdown comparison table from the score files written by evaluate.py.

Reads results/<name>_<split>_scores.json for each name and prints one table row per approach, so numbers are
copied from the files and never typed by hand.

Run (from the project root):
    python src/compare_results.py                                   # baseline vs transformer, test split
    python src/compare_results.py --names baseline transformer translation
"""

# Standard library: argparse reads the command-line options, json reads the score files, os builds file paths.
import argparse
import json
import os

# Nice names shown in the table. The key is the approach name, which is also the prefix of its score file.
LABELS = {"baseline": "CNN + LSTM + Attention (baseline)", "transformer": "ResNet-50 + Transformer decoder",
          "translation": "English captioning + translation"}
# The five scores we report, in the same order as the table columns.
METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr"]


# Idea: every approach (baseline, transformer, translation) saved its scores in a JSON file. This function
# reads those files and prints one Markdown table, so we never type scores by hand.
def main():
    # Command-line options: which approaches to compare, which split, and the folder of the score files.
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["baseline", "transformer"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    # Print the table header and the separator line (Markdown table syntax).
    print("| Approach | " + " | ".join(METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    # rows keeps the scores of every approach that was found, so two of them can be compared at the end.
    rows = {}
    # Go through each approach. A missing score file is shown as n/a instead of crashing.
    for name in args.names:
        # Score file that evaluate.py wrote for this approach.
        path = os.path.join(args.results_dir, f"{name}_{args.split}_scores.json")
        if not os.path.exists(path):
            print(f"| {LABELS.get(name, name)} | " + " | ".join("n/a" for _ in METRICS) + " |")
            continue
        with open(path, encoding="utf-8") as f:
            rows[name] = json.load(f)["scores"]
        print(f"| {LABELS.get(name, name)} | " + " | ".join(f"{rows[name][m]:.2f}" for m in METRICS) + " |")

    # If exactly two approaches were found, also print how much the second one changed compared to the first
    # (+ = better, - = worse).
    if len(rows) == 2:
        a, b = list(rows)
        print(f"\nChange {b} vs {a}: " + ", ".join(f"{m} {rows[b][m] - rows[a][m]:+.2f}" for m in METRICS))


# Run main() only when the file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()
