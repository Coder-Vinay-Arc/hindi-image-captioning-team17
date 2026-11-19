"""
preprocess_hvg.py -- Hindi Visual Genome (HVG) region-crop preprocessing.

Reads a raw HVG annotation file (tab-separated, NO header):
    image_id  X  Y  Width  Height  English  Hindi
crops each bounding-box region out of its image, resizes it, and writes
    <out_dir>/images/<image_id>_<row>.jpg
    <out_dir>/captions.csv     (columns: file_name, hindi_caption, english_caption)

Each region is treated as an independent (image, caption) pair.

Usage:
    python src/preprocess_hvg.py \
        --annotations data/raw/hindi-visual-genome-train.txt \
        --images_dir data/raw/hindi-visual-genome-train.images \
        --out_dir data/processed/train

    # quick subset for teammates (first 500 rows):
    python src/preprocess_hvg.py \
        --annotations data/raw/hindi-visual-genome-train.txt \
        --images_dir data/raw/hindi-visual-genome-train.images \
        --out_dir data/processed/sample --max_rows 500
"""

import argparse
import csv
import os

from PIL import Image

COLUMNS = ["image_id", "x", "y", "w", "h", "english", "hindi"]


def parse_args():
    p = argparse.ArgumentParser(description="Crop HVG regions and build captions.csv")
    p.add_argument("--annotations", required=True, help="raw HVG .txt file (tab-separated)")
    p.add_argument("--images_dir", required=True, help="folder with <image_id>.jpg files")
    p.add_argument("--out_dir", required=True, help="output folder")
    p.add_argument("--size", type=int, default=224, help="output square size (default 224)")
    p.add_argument("--max_rows", type=int, default=None, help="only process the first N rows")
    p.add_argument("--min_side", type=int, default=8, help="skip regions smaller than this (px)")
    return p.parse_args()


def read_rows(path, max_rows=None):
    """Yield dicts for each valid annotation line."""
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if max_rows is not None and n >= max_rows:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != len(COLUMNS):
                print(f"[warn] line {n + 1}: expected {len(COLUMNS)} fields, got {len(parts)} -- skipped")
                continue
            row = dict(zip(COLUMNS, parts))
            try:
                for k in ("x", "y", "w", "h"):
                    row[k] = int(float(row[k]))
            except ValueError:
                print(f"[warn] line {n + 1}: bad bbox -- skipped")
                continue
            row["row"] = n
            yield row


def crop_region(img, x, y, w, h, size):
    """Crop the bbox (clamped to the image) and resize to size x size."""
    W, H = img.size
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return img.crop((x0, y0, x1, y1)).resize((size, size), Image.BICUBIC)


def main():
    args = parse_args()
    img_out = os.path.join(args.out_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    kept = skipped = 0
    cache_id, cache_img = None, None  # consecutive rows may share an image

    with open(os.path.join(args.out_dir, "captions.csv"), "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["file_name", "hindi_caption", "english_caption"])

        for row in read_rows(args.annotations, args.max_rows):
            if row["w"] < args.min_side or row["h"] < args.min_side or not row["hindi"].strip():
                skipped += 1
                continue

            if row["image_id"] != cache_id:
                path = os.path.join(args.images_dir, row["image_id"] + ".jpg")
                if not os.path.exists(path):
                    skipped += 1
                    continue
                cache_img = Image.open(path).convert("RGB")
                cache_id = row["image_id"]

            crop = crop_region(cache_img, row["x"], row["y"], row["w"], row["h"], args.size)
            if crop is None:
                skipped += 1
                continue

            name = f"{row['image_id']}_{row['row']}.jpg"
            crop.save(os.path.join(img_out, name), quality=95)
            writer.writerow([name, row["hindi"].strip(), row["english"].strip()])
            kept += 1
            if kept % 2000 == 0:
                print(f"  {kept} regions done...")

    print(f"Done. kept={kept} skipped={skipped} -> {args.out_dir}")


if __name__ == "__main__":
    main()
