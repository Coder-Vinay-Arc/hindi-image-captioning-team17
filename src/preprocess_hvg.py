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

# argparse = command-line options, csv = write captions.csv, os = file paths, PIL = open, crop and resize
# images.
import argparse
import csv
import os

from PIL import Image

# Column order of the raw HVG annotation file (the file has no header line).
COLUMNS = ["image_id", "x", "y", "w", "h", "english", "hindi"]


# Reads the command-line options: input files, output folder, crop size and so on.
def parse_args():
    p = argparse.ArgumentParser(description="Crop HVG regions and build captions.csv")
    p.add_argument("--annotations", required=True, help="raw HVG .txt file (tab-separated)")
    p.add_argument("--images_dir", required=True, help="folder with <image_id>.jpg files")
    p.add_argument("--out_dir", required=True, help="output folder")
    p.add_argument("--size", type=int, default=224, help="output square size (default 224)")
    p.add_argument("--max_rows", type=int, default=None, help="only process the first N rows")
    p.add_argument("--min_side", type=int, default=8, help="skip regions smaller than this (px)")
    return p.parse_args()


# Reads the annotation file line by line and gives back one clean dictionary for every valid line.
def read_rows(path, max_rows=None):
    """Yield dicts for each valid annotation line."""
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if max_rows is not None and n >= max_rows:
                break
            # Every line must have exactly 7 tab-separated fields, otherwise it is skipped.
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != len(COLUMNS):
                print(f"[warn] line {n + 1}: expected {len(COLUMNS)} fields, got {len(parts)} -- skipped")
                continue
            # Make a dictionary from the fields (image id, box, English and Hindi caption).
            row = dict(zip(COLUMNS, parts))
            try:
                # Convert the box numbers to integers. Some are written like '12.0', so we go through float
                # first.
                for k in ("x", "y", "w", "h"):
                    row[k] = int(float(row[k]))
            # A broken box value means we cannot crop this region, so the line is skipped.
            except ValueError:
                print(f"[warn] line {n + 1}: bad bbox -- skipped")
                continue
            # Remember the line number. It makes every output file name unique.
            row["row"] = n
            yield row


# Idea of the project: a Hindi caption describes a REGION (box) of an image. So we cut that box out and treat
# the crop as its own small image.
def crop_region(img, x, y, w, h, size):
    """Crop the bbox (clamped to the image) and resize to size x size."""
    W, H = img.size
    # Clamp: some boxes stick out of the image, so we limit the coordinates to the image border.
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    # Nothing is left after clamping, so there is no region to crop.
    if x1 <= x0 or y1 <= y0:
        return None
    # Crop the box and resize it to a fixed size (smooth bicubic resize). All crops get the same size, so they
    # can be put into batches.
    return img.crop((x0, y0, x1, y1)).resize((size, size), Image.BICUBIC)


# Goes through all annotation rows, crops each region, and writes the images and captions.csv.
def main():
    args = parse_args()
    img_out = os.path.join(args.out_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    # Counters for the final summary.
    kept = skipped = 0
    cache_id, cache_img = None, None  # consecutive rows may share an image

    # captions.csv is written row by row while we crop, so we never keep all rows in memory.
    with open(os.path.join(args.out_dir, "captions.csv"), "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["file_name", "hindi_caption", "english_caption"])

        for row in read_rows(args.annotations, args.max_rows):
            # Skip tiny boxes (too small to show anything) and rows that have no Hindi caption.
            if row["w"] < args.min_side or row["h"] < args.min_side or not row["hindi"].strip():
                skipped += 1
                continue

            # Open the full image only when it changes. Consecutive rows often share one image, so this saves
            # a lot of time.
            if row["image_id"] != cache_id:
                path = os.path.join(args.images_dir, row["image_id"] + ".jpg")
                if not os.path.exists(path):
                    skipped += 1
                    continue
                cache_img = Image.open(path).convert("RGB")
                cache_id = row["image_id"]

            # Crop the region and resize it to size x size (224x224 by default).
            crop = crop_region(cache_img, row["x"], row["y"], row["w"], row["h"], args.size)
            if crop is None:
                skipped += 1
                continue

            # Unique file name = image id + line number.
            name = f"{row['image_id']}_{row['row']}.jpg"
            # Save the crop as a JPEG (quality 95) and write its caption to captions.csv.
            crop.save(os.path.join(img_out, name), quality=95)
            writer.writerow([name, row["hindi"].strip(), row["english"].strip()])
            kept += 1
            if kept % 2000 == 0:
                print(f"  {kept} regions done...")

    # Final summary.
    print(f"Done. kept={kept} skipped={skipped} -> {args.out_dir}")


if __name__ == "__main__":
    main()
