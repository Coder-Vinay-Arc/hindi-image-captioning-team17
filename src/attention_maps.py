"""
attention_maps.py -- save attention-map figures for a few test images (one PNG per image).

For every chosen image the figure shows the original crop followed by one panel per generated word:
the (num_words, 49) attention returned by model.caption(...) is reshaped row by row to 7x7, upsampled and
overlaid on the image. Each word's map is min-max scaled on its own so the peak is always visible.
For the Transformer these are the cross-attention weights of the LAST decoder layer, averaged over heads.

Images are picked at random with a fixed --seed (not hand-picked), so the figures are not cherry-picked.

Run (from the project root):
    python src/attention_maps.py --model transformer --ckpt checkpoints/transformer_best.pth --n 6 --beam 3
    -> results/transformer_attention_1.png ... results/transformer_attention_6.png

Hindi text needs a Devanagari font. On Colab run once:  !apt-get -qq install fonts-noto-core && rm -rf ~/.cache/matplotlib
(Windows already has "Nirmala UI".) If no font is found the script still runs but Hindi shows as empty boxes.
"""

import argparse
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed (Colab / servers)
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import default_transform  # noqa: E402
from train import load_checkpoint  # noqa: E402

FONT_CANDIDATES = ["Noto Sans Devanagari", "Noto Serif Devanagari", "Lohit Devanagari", "Nirmala UI", "Mangal",
                   "Devanagari Sangam MN", "DejaVu Sans"]


def pick_hindi_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for name in FONT_CANDIDATES[:-1]:
        if name in available:
            return name
    print("[warn] no Devanagari font found: Hindi will show as boxes. On Colab: !apt-get -qq install fonts-noto-core "
          "&& rm -rf ~/.cache/matplotlib")
    return FONT_CANDIDATES[-1]


def draw_one(path, image, words, att, reference, font, out_file):
    """att: (>= len(words), 49) tensor. Saves a figure: original | one overlay per word."""
    cols = 1 + len(words)
    fig, axes = plt.subplots(1, cols, figsize=(2.1 * cols, 2.9), squeeze=False)
    axes = axes[0]
    W, H = image.size
    axes[0].imshow(image)
    axes[0].set_title("original", fontsize=10)
    for k, (word, ax) in enumerate(zip(words, axes[1:])):
        a = att[k].reshape(7, 7).float()
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        ax.imshow(image)
        ax.imshow(a.numpy(), extent=(0, W, H, 0), cmap="jet", alpha=0.55, interpolation="bilinear")
        ax.set_title(word, fontsize=13, fontname=font)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.text(0.5, 0.02, f"prediction: {' '.join(words)}     reference: {reference}", ha="center", va="bottom",
             fontsize=11, fontname=font)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Save attention-map figures for random test images")
    ap.add_argument("--model", default="transformer", help="only used to name the output files")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--n", type=int, default=6, help="number of images")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--name", default=None, help="output prefix (default: <model>_attention)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_checkpoint(args.ckpt, device)
    font = pick_hindi_font()
    prefix = args.name or f"{args.model}_attention"
    os.makedirs(args.out_dir, exist_ok=True)

    split_dir = os.path.join(args.data_root, args.split)
    df = pd.read_csv(os.path.join(split_dir, "captions.csv"))
    rows = random.Random(args.seed).sample(range(len(df)), args.n)
    transform = default_transform()

    for k, r in enumerate(rows, 1):
        row = df.iloc[r]
        image = Image.open(os.path.join(split_dir, "images", row["file_name"])).convert("RGB")
        ids, maps = model.caption(transform(image).unsqueeze(0).to(device), vocab.start_id, vocab.end_id,
                                  beam_size=args.beam)
        ids, att = ids[0], maps[0]
        words = [vocab.itos[i] for i in ids]
        if not words:
            print(f"[skip] {row['file_name']}: empty caption")
            continue
        assert att.size(0) >= len(words) and att.size(1) == 49, f"unexpected attention shape {tuple(att.shape)}"
        out_file = os.path.join(args.out_dir, f"{prefix}_{k}.png")
        draw_one(row["file_name"], image, words, att, row["hindi_caption"], font, out_file)
        print(f"saved {out_file}  ({row['file_name']})  pred: {' '.join(words)}  | ref: {row['hindi_caption']}")


if __name__ == "__main__":
    main()
