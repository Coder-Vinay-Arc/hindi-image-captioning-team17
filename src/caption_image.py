"""
caption_image.py -- generate a Hindi caption for your OWN image with a trained model.

The models were trained on 224x224 REGION crops (a bounding box of a Visual Genome image), so they
describe "a thing / a small scene" (e.g. "a white plate with a sandwich") rather than a whole busy photo.
For the best results give a photo (or a crop) that shows ONE main object or a small scene; for a big photo
use --box to cut out the part you want captioned.

Run (from the project root; the checkpoint must be downloaded from the shared Drive into checkpoints/):
    # one image, one model
    python src/caption_image.py --image my_photo.jpg --ckpt checkpoints/baseline_best.pth

    # compare the baseline and the Transformer on the same image
    python src/caption_image.py --image my_photo.jpg --ckpt checkpoints/baseline_best.pth checkpoints/transformer_best.pth

    # several images / a whole folder
    python src/caption_image.py --image img1.jpg img2.png --ckpt checkpoints/baseline_best.pth
    python src/caption_image.py --image my_photos/ --ckpt checkpoints/baseline_best.pth

    # caption only a region of a big photo: --box X Y WIDTH HEIGHT (pixels, like the HVG annotations)
    python src/caption_image.py --image street.jpg --box 120 80 200 160 --ckpt checkpoints/baseline_best.pth

    # also save a copy of the image with the caption (and the cropped region) written on it
    python src/caption_image.py --image my_photo.jpg --ckpt checkpoints/baseline_best.pth --save_dir results/demo

Only the Hindi caption (and the model name) is printed; the English translation pipeline is a separate script
(src/translation_pipeline.py).
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `train` and `dataset` import from src/
from dataset import default_transform  # noqa: E402  (same resize + ImageNet normalisation as in training)
from train import load_checkpoint  # noqa: E402  (rebuilds the right model class + vocab from a .pth file)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_images(paths):
    """Turn the --image arguments (files and/or folders) into a flat list of image file paths."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            # a folder: take every image file in it, sorted so the output order is stable
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(IMAGE_EXTENSIONS))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[warn] not found, skipped: {p}")
    return files


def load_image(path, box=None):
    """Open an image as RGB and optionally cut out the region box = (x, y, width, height)."""
    img = Image.open(path).convert("RGB")  # convert: PNGs may have an alpha channel / greyscale
    if box is not None:
        x, y, w, h = box
        W, H = img.size
        # clamp the box to the image so a slightly-too-big box does not crash (same idea as preprocess_hvg.py)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"--box {box} is outside the image ({W}x{H})")
        img = img.crop((x0, y0, x1, y1))
    return img


@torch.no_grad()
def caption_one(model, vocab, img, transform, device, beam):
    """Return the Hindi caption (str) the model generates for one PIL image."""
    # the transform resizes to 224x224 and normalises -> (3, 224, 224); unsqueeze adds the batch dim -> (1, 3, 224, 224)
    x = transform(img).unsqueeze(0).to(device)
    # model.caption returns (list of id lists, list of attention maps); we have a batch of 1, so take element 0
    ids, _ = model.caption(x, vocab.start_id, vocab.end_id, beam_size=beam)
    return vocab.decode(ids[0])  # ids -> Hindi words (stops at <end>, drops <start>/<pad>)


def save_with_caption(img, caption, out_path):
    """Save the (possibly cropped) image next to a strip of text with the Hindi caption.

    Pillow draws Devanagari without proper conjunct shaping unless it was built with Raqm, so for a nicely
    shaped caption in a figure use src/attention_maps.py; here we use matplotlib's default text for a quick demo.
    """
    import matplotlib
    matplotlib.use("Agg")  # no display needed (works on servers / Colab)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # try to find a font that has Devanagari glyphs; otherwise matplotlib would draw empty boxes
    for name in ("Nirmala UI", "Noto Sans Devanagari", "Lohit Devanagari", "Mangal"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break

    fig, ax = plt.subplots(figsize=(4, 4.6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=13, wrap=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Caption your own image(s) in Hindi")
    ap.add_argument("--image", nargs="+", required=True, help="image file(s) and/or folder(s)")
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints (e.g. baseline and transformer)")
    ap.add_argument("--box", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                    help="caption only this region (pixels); applied to every image given")
    ap.add_argument("--beam", type=int, default=3, help="beam size (1 = greedy, 3 = what the reported scores use)")
    ap.add_argument("--save_dir", default=None, help="if set, save <image>_<model>.png with the caption on it")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # so Hindi prints correctly in a Windows console
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = default_transform()  # identical preprocessing to training -> same inputs the model has seen

    images = collect_images(args.image)
    if not images:
        sys.exit("no images found")

    # load every checkpoint once (loading is the slow part), then caption all images with each
    models = []
    for ckpt in args.ckpt:
        model, vocab = load_checkpoint(ckpt, device)
        # label = file name without extension, e.g. "baseline_best"; used in the printout and output file names
        models.append((os.path.splitext(os.path.basename(ckpt))[0], model, vocab))

    for path in images:
        img = load_image(path, args.box)
        print(f"\n{path}" + (f"  (region {tuple(args.box)})" if args.box else ""))
        for label, model, vocab in models:
            caption = caption_one(model, vocab, img, transform, device, args.beam)
            print(f"  {label:22s} {caption}")
            if args.save_dir:
                stem = os.path.splitext(os.path.basename(path))[0]
                save_with_caption(img, caption, os.path.join(args.save_dir, f"{stem}_{label}.png"))


if __name__ == "__main__":
    main()
