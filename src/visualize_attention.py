"""
visualize_attention.py -- Draw "where the model looks" for each generated Hindi word.

For each chosen image it runs the trained model, then saves one figure:
the original crop + one panel per word with the 7x7 attention map upsampled over the image.
    results/attention/<name>_<split>_<file_name>.png

Usage (run from the repo root):
    python src/visualize_attention.py --ckpt checkpoints/baseline_best.pth --split test --n 8

Hindi font: matplotlib needs a Devanagari font to print the words. It tries
Nirmala UI (Windows), Noto Sans Devanagari, Lohit Devanagari, Mangal. On Colab run
    !apt-get -qq install fonts-noto-core
first. (matplotlib does not shape Devanagari conjuncts perfectly, so the reference and
predicted captions are also printed to the console for copying into the report.)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import HVGDataset  # noqa: E402
from models import load_checkpoint  # noqa: E402

HINDI_FONTS = ["Nirmala UI", "Noto Sans Devanagari", "Lohit Devanagari", "Mangal"]


def pick_hindi_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in HINDI_FONTS:
        if name in available:
            return name
    print("warning: no Devanagari font found -- Hindi words may show as boxes")
    return None


def plot(image_path, words, alphas, out_path, font, grid=7):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    cols = min(5, len(words) + 1)
    rows = int(np.ceil((len(words) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("input", fontsize=11)
    for t, (word, alpha) in enumerate(zip(words, alphas), start=1):
        ax = axes.flat[t]
        heat = alpha.view(1, 1, grid, grid)
        heat = F.interpolate(heat, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(img)
        ax.imshow(heat, cmap="jet", alpha=0.5)
        ax.set_title(word, fontsize=14, fontname=font) if font else ax.set_title(word, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.3)  # room for Devanagari matras
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Save attention-map figures")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=8, help="number of images")
    ap.add_argument("--start", type=int, default=0, help="index of the first image")
    ap.add_argument("--beam_size", type=int, default=3)
    ap.add_argument("--out_dir", default="results/attention")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, _ = load_checkpoint(args.ckpt, device)
    ds = HVGDataset(args.data_root, args.split, vocab)
    font = pick_hindi_font()
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.basename(args.ckpt).replace(".pth", "")

    for i in range(args.start, min(args.start + args.n, len(ds))):
        image, _ = ds[i]
        ids, alphas = model.caption(image.unsqueeze(0).to(device), vocab.start_id, vocab.end_id,
                                    beam_size=args.beam_size)
        words = [vocab.itos[j] for j in ids[0]]
        file_name = ds.df.iloc[i]["file_name"]
        out = os.path.join(args.out_dir, f"{name}_{args.split}_{os.path.splitext(file_name)[0]}.png")
        plot(os.path.join(ds.dir, "images", file_name), words, alphas[0].float().cpu(), out, font)
        print(f"{file_name}\n  ref : {ds.reference(i)}\n  pred: {' '.join(words)}\n  -> {out}")


if __name__ == "__main__":
    main()
