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

Hindi text: matplotlib cannot shape Devanagari (the short-i matra "ि" lands in the wrong place), so Hindi is
drawn with Pillow + the Raqm layout engine and pasted into the figure. This needs a Devanagari font and Raqm.
On Colab run once:
    !apt-get -qq install fonts-noto-core libraqm0 libfribidi0 && rm -rf ~/.cache/matplotlib
(Windows already has "Nirmala UI" but its Pillow has no Raqm, so Hindi will be mis-shaped there.)
The script prints a [warn] line when Raqm or a font is missing. Do not use such figures in the report.
"""

# Standard library: argparse (command-line options), os (paths), random (pick random images), sys (imports,
# UTF-8 output).
import argparse
import os
import random
import sys

# matplotlib draws the figures. 'Agg' is a backend that saves files without opening a window (needed on
# Colab).
import matplotlib

matplotlib.use("Agg")  # no display needed (Colab / servers)
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, features  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Our own code: the same image preprocessing as in training, and the function that reloads a trained model.
from dataset import default_transform  # noqa: E402
from train import load_checkpoint  # noqa: E402

# Devanagari fonts we try, best first. The first one installed on the computer is used.
FONT_CANDIDATES = ["Noto Sans Devanagari", "Noto Serif Devanagari", "Lohit Devanagari", "Nirmala UI", "Mangal",
                   "Devanagari Sangam MN"]


# Finds a font file that can draw Hindi letters. Returns None (and warns) if there is none.
def pick_hindi_font():
    """Path of a Devanagari font file, or None."""
    # Names of all fonts that matplotlib can see.
    available = {f.name for f in fm.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in available:
            return fm.findfont(name, fallback_to_default=False)
    print("[warn] no Devanagari font found: Hindi will show as boxes. On Colab: "
          "!apt-get -qq install fonts-noto-core && rm -rf ~/.cache/matplotlib")
    return None


# Only needed when Raqm is missing: tells matplotlib to use the Hindi font first and DejaVu for English
# letters.
def use_fallback_fonts(deva_path):
    """Only matters for the no-Raqm fallback, where matplotlib draws the text itself."""
    if deva_path:
        plt.rcParams["font.family"] = [fm.FontProperties(fname=deva_path).get_name(), "DejaVu Sans"]


# Hindi letters are joined and re-ordered (for example the short-i matra is written BEFORE its consonant).
# Raqm is the text engine that does this correctly. Without it Hindi looks wrong.
def pick_layout():
    """Raqm shapes Devanagari correctly. Returns the Raqm engine, or None when it is not installed."""
    if features.check("raqm"):
        return ImageFont.Layout.RAQM
    print("[warn] Pillow has no Raqm, so Hindi letters (e.g. the short-i matra) will be drawn in the wrong place. "
          "On Colab: !apt-get -qq install libraqm0 libfribidi0   (then run this script again)")
    return None


# Draws text with Pillow and returns it as an image, so matplotlib does not have to shape Hindi itself. 'runs'
# is a list of (text, font file) pieces written one after another.
def text_image(runs, size=44, layout=None):
    """runs: [(text, font_file), ...] drawn left to right on one baseline -> RGBA image with black text."""
    # Load each font at the chosen size.
    fonts = [(t, ImageFont.truetype(p, size, layout_engine=layout)) for t, p in runs]
    # Width of each piece, so the next piece starts exactly where the previous one ends.
    widths = [f.getlength(t) for t, f in fonts]
    # Ascent / descent = how far letters go above / below the baseline. We need them to size the image.
    asc = max(f.getmetrics()[0] for _, f in fonts)
    desc = max(f.getmetrics()[1] for _, f in fonts)
    # Small empty border around the text.
    pad = size // 4
    # Transparent canvas that is big enough for all pieces.
    img = Image.new("RGBA", (int(sum(widths)) + 2 * pad, asc + desc + 2 * pad), (255, 255, 255, 0))
    draw, x = ImageDraw.Draw(img), pad
    # Write every piece on the same baseline (anchor 'ls' = left, baseline), then move to the right.
    for (t, f), w in zip(fonts, widths):
        draw.text((x, pad + asc), t, font=f, fill=(0, 0, 0, 255), anchor="ls")
        x += w
    return img


# Writes text into an empty axes: as a correctly shaped image if Raqm works, otherwise as normal matplotlib
# text.
def put_text(ax, runs, size, layout):
    """Write runs into an (axis-less) axes: as a shaped image when Raqm is on, else as plain matplotlib text."""
    ax.axis("off")
    if layout is not None and all(p for _, p in runs):
        ax.imshow(text_image(runs, size, layout))
    else:
        ax.text(0.5, 0.5, "".join(t for t, _ in runs), ha="center", va="center", fontsize=size / 3.5)


# An attention map answers one question: when the model wrote this word, which part of the picture did it look
# at? This function draws one figure: the original image, then one heat map per predicted word, then the
# prediction and the reference text.
def draw_one(image, words, att, reference, deva_font, latin_font, layout, out_file):
    """att: (>= len(words), 49) tensor. Saves a figure: original | one overlay per word, caption lines below."""
    # One column for the original image plus one column for each word.
    cols = 1 + len(words)
    # Figure size grows with the number of words. The grid has 4 rows: word label, image, prediction,
    # reference.
    fig = plt.figure(figsize=(2.1 * cols, 3.7))
    gs = fig.add_gridspec(4, cols, height_ratios=[0.2, 1, 0.14, 0.14], hspace=0.06, wspace=0.05)
    W, H = image.size

    # First column: the original image.
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(image)
    ax.axis("off")
    put_text(fig.add_subplot(gs[0, 0]), [("original", latin_font)], 40, layout)
    # One panel for every predicted word.
    for k, word in enumerate(words):
        # The model gives 49 attention numbers for this word, one for each cell of the 7x7 image grid.
        # Reshaping them to 7x7 gives a small map of 'where the model looked'.
        a = att[k].reshape(7, 7).float()
        # Scale the map to 0..1 so its brightest cell is always visible, even when all values are small.
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        ax = fig.add_subplot(gs[1, k + 1])
        ax.imshow(image)
        # Stretch the 7x7 map over the whole image (smooth interpolation) and lay it on top of the photo. Red
        # = high attention, blue = low. alpha=0.55 keeps the photo visible.
        ax.imshow(a.numpy(), extent=(0, W, H, 0), cmap="jet", alpha=0.55, interpolation="bilinear")
        ax.axis("off")
        put_text(fig.add_subplot(gs[0, k + 1]), [(word, deva_font)], 52, layout)
    # Bottom lines: what the model predicted and what the real caption is.
    put_text(fig.add_subplot(gs[2, :]), [("prediction: ", latin_font), (" ".join(words), deva_font)], 44, layout)
    put_text(fig.add_subplot(gs[3, :]), [("reference:   ", latin_font), (reference, deva_font)], 44, layout)
    # Save the picture and free the memory.
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Loads the trained model, picks random test images, generates a caption for each and saves one attention
# figure per image.
def main():
    # Command-line options (checkpoint, split, number of images, beam size, seed, output folder).
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

    # Use the GPU if there is one, then reload the trained model and its vocabulary from the checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_checkpoint(args.ckpt, device)
    # Find a Hindi font and check if the Raqm text engine is available.
    deva_font, layout = pick_hindi_font(), pick_layout()
    use_fallback_fonts(deva_font)
    latin_font = fm.findfont("DejaVu Sans")
    prefix = args.name or f"{args.model}_attention"
    os.makedirs(args.out_dir, exist_ok=True)

    # Read the caption table of the split (test by default).
    split_dir = os.path.join(args.data_root, args.split)
    df = pd.read_csv(os.path.join(split_dir, "captions.csv"))
    # Pick n random rows with a fixed seed: the same images in every run, and not hand-picked, so there is no
    # cherry-picking.
    rows = random.Random(args.seed).sample(range(len(df)), args.n)
    transform = default_transform()

    # For every chosen image: load it, generate a caption with beam search, and plot the attention.
    for k, r in enumerate(rows, 1):
        row = df.iloc[r]
        image = Image.open(os.path.join(split_dir, "images", row["file_name"])).convert("RGB")
        # ids = the predicted word ids. maps = one attention row (49 numbers) for each predicted word.
        ids, maps = model.caption(transform(image).unsqueeze(0).to(device), vocab.start_id, vocab.end_id,
                                  beam_size=args.beam)
        # We sent only one image, so we take element 0 of the results.
        ids, att = ids[0], maps[0]
        # Turn the ids back into Hindi words (one attention row belongs to each word).
        words = [vocab.itos[i] for i in ids]
        if not words:
            print(f"[skip] {row['file_name']}: empty caption")
            continue
        # Safety check: we need one attention row per word and exactly 49 columns (the 7x7 grid).
        assert att.size(0) >= len(words) and att.size(1) == 49, f"unexpected attention shape {tuple(att.shape)}"
        out_file = os.path.join(args.out_dir, f"{prefix}_{k}.png")
        draw_one(image, words, att, row["hindi_caption"], deva_font, latin_font, layout, out_file)
        print(f"saved {out_file}  ({row['file_name']})  pred: {' '.join(words)}  | ref: {row['hindi_caption']}")


if __name__ == "__main__":
    main()
