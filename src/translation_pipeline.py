"""
translation_pipeline.py -- Approach 3: English captioning + Hindi translation.

Pipeline: image -> BLIP English caption -> Hindi (Google Translate) -> results CSV.
The CSV has columns file_name, prediction and is scored by evaluate.py:

    python src/translation_pipeline.py --split test
    python src/evaluate.py --model translation --split test --pred_csv results/translation_test_preds.csv

Options:
    --limit 20       quick smoke test
    --gold_english   translate the dataset's gold English captions instead of BLIP (MT upper bound)
    --out PATH       output CSV (default results/translation_<split>_preds.csv)

Rerunning the same command resumes: rows already present in the output CSV are skipped.
"""

import argparse
import os
import sys
import time

import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser(description="English captioning (BLIP) + Hindi translation pipeline")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (smoke test)")
    ap.add_argument("--caption_model", default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--gold_english", action="store_true", help="translate gold English captions instead of BLIP")
    ap.add_argument("--sleep", type=float, default=0.2, help="delay between translation requests")
    ap.add_argument("--save_every", type=int, default=20)
    return ap.parse_args()


def translate(text, translator, retries=5):
    for attempt in range(retries):
        try:
            out = translator.translate(text)
            if out:
                return out.strip()
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return ""


def main():
    args = parse_args()
    split_dir = os.path.join(args.data_root, args.split)
    df = pd.read_csv(os.path.join(split_dir, "captions.csv"), keep_default_na=False)
    if args.limit:
        df = df.head(args.limit).copy()
    out_path = args.out or os.path.join("results", f"translation_{args.split}_preds.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    done = {}
    if os.path.exists(out_path):
        old = pd.read_csv(out_path, encoding="utf-8", keep_default_na=False)
        done = {str(f): str(p) for f, p in zip(old["file_name"], old["prediction"]) if str(p).strip()}
        print(f"resuming: {len(done)} rows already done")

    names = df["file_name"].astype(str).tolist()
    pending = [i for i, n in enumerate(names) if n not in done]

    if args.gold_english:
        english = {i: str(df.iloc[i]["english_caption"]) for i in pending}
    else:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading {args.caption_model} on {device}")
        processor = BlipProcessor.from_pretrained(args.caption_model)
        model = BlipForConditionalGeneration.from_pretrained(args.caption_model).to(device).eval()
        english = {}
        for i in tqdm(pending, desc="BLIP English captions"):
            path = os.path.join(split_dir, "images", names[i])
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            english[i] = processor.decode(ids[0], skip_special_tokens=True).strip()

    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="en", target="hi")

    for i in tqdm(pending, desc="EN -> HI translation"):
        done[names[i]] = translate(english[i], translator)
        time.sleep(args.sleep)
        if args.save_every and len(done) % args.save_every == 0:
            pd.DataFrame({"file_name": list(done), "prediction": list(done.values())}).to_csv(
                out_path, index=False, encoding="utf-8")

    result = pd.DataFrame({"file_name": names, "prediction": [done.get(n, "") for n in names]})
    result.to_csv(out_path, index=False, encoding="utf-8")
    empty = int((result["prediction"].str.strip() == "").sum())
    print(f"saved {len(result)} rows -> {out_path} (empty: {empty})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
