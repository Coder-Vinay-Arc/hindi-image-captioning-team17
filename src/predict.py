"""
predict.py -- Generate Hindi captions for a whole split with a trained checkpoint.

Writes results/<name>_<split>_predictions.csv with columns
    file_name, reference, prediction
which evaluate.py can score with BLEU / CIDEr (same format for every approach).

Usage (run from the repo root):
    python src/predict.py --ckpt checkpoints/baseline_best.pth --split test
    python src/predict.py --ckpt checkpoints/baseline_best.pth --split test --beam_size 1   # greedy, faster
"""

import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import HVGDataset, make_collate  # noqa: E402
from models import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Caption a split with a trained model")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam_size", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=None, help="output CSV (default: results/<name>_<split>_predictions.csv)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, _ = load_checkpoint(args.ckpt, device)
    ds = HVGDataset(args.data_root, args.split, vocab)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate(vocab.pad_id))

    name = os.path.basename(args.ckpt).replace("_best.pth", "").replace("_last.pth", "").replace(".pth", "")
    out = args.out or os.path.join("results", f"{name}_{args.split}_predictions.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    rows, idx = [], 0
    for images, _, _ in tqdm(loader, desc=f"captioning {args.split}"):
        ids, _ = model.caption(images.to(device), vocab.start_id, vocab.end_id,
                               max_len=args.max_len, beam_size=args.beam_size)
        for pred in ids:
            row = ds.df.iloc[idx]
            rows.append({"file_name": row["file_name"], "reference": row["hindi_caption"],
                         "prediction": vocab.decode(pred)})
            idx += 1

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "reference", "prediction"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions -> {out}")
    for r in rows[:5]:
        print(f"  ref : {r['reference']}\n  pred: {r['prediction']}")


if __name__ == "__main__":
    main()
